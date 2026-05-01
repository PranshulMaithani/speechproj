"""
Builder for ssl_layer_fusion.ipynb

Produces a single self-contained notebook that:
  1. Extracts (resume-safe) for audios2/audios3/audios4/audios5:
       - Whisper transcripts
       - 50 text/disfluency/pause/prosodic/voice-quality/perplexity features
       - Multi-layer WavLM-base-plus (layers 6, 9, 12) mean-pool
       - Multi-layer Whisper-medium encoder (layers 12, 18, 24) mean-pool
       - Speaking-time durations
  2. Builds 17 base models (text XGBs + per-layer audio RF/XGB).
  3. Baseline pipeline (a2 + a4 train -> a5 test):
       - 5-fold candidate-isolated GroupKFold OOF for each base
       - Isotonic calibration on OOF per base
       - Caruana (2004) greedy ensemble selection on calibrated OOFs
       - Refit on full pool, score a5, evaluate at the OOF-derived F1-max threshold
  4. SSL round on audios3:
       - Score a3 with each base, calibrate
       - T-similarity (1 - 4*Var across bases) as agreement signal
       - Select high-agreement extreme-score samples as pseudo-labels (capped per class)
       - Retrain pipeline on (a2 + a4 + pseudo-labelled a3), re-evaluate on a5
  5. Comparison CSV: baseline vs SSL.

Usage:
    python _build_ssl_layer_fusion.py
    -> writes ssl_layer_fusion.ipynb next to this file
"""
import json
from pathlib import Path

CELLS = []

def md(text):
    CELLS.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.splitlines(keepends=True),
    })

def code(text):
    CELLS.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': text.splitlines(keepends=True),
    })

# ============================================================
md("""# SSL + Layer-Fusion Pipeline

**Goal:** improve the IND-side model. Implements:
1. **Multi-layer encoder features** — WavLM layers {6, 9, 12} and Whisper-medium-encoder layers {12, 18, 24} (last layer of each is the `whole-pool` we already had).
2. **Per-base isotonic calibration** before fusion (puts XGB and RF on the same probability scale).
3. **Caruana (2004) greedy ensemble selection** instead of grid search over weighted averages — searches the full 17-base space without overfitting OOF.
4. **Semi-supervised round** — pseudo-label `audios3` (unlabelled IND) using T-similarity (variance across bases as a confidence signal), retrain on the expanded pool, re-evaluate.

**Train / test:**
- Labelled train pool: `audios2 + audios4`
- Test (one-shot): `audios5`
- Unlabelled SSL source: `audios3`

**Why this design:**
- Layer sweep targets paralinguistic signal (prosody, hesitation), which lives in middle layers, not the last layer.
- Calibration before fusion is mandatory once you mix XGB and RF — they output very different probability distributions.
- Caruana with replacement handles any number of bases without weight overfitting (the weights are integer counts).
- T-similarity uses your 9+ models as an implicit ensemble; only pseudo-label where every base agrees.

Drop your `audios{2,3,4,5}` folders + `audios{2,4,5}GT.csv` next to this notebook (audios3 has no GT — it's the unlabelled set), then run all cells. Extraction is resume-safe and skipped if cached.""")

# ============================================================
code("""# === 0. CONFIG ===
from pathlib import Path

# Batch roles
BATCH_LABELED   = ['audios2', 'audios4']     # labelled train pool
BATCH_UNLABELED = 'audios3'                  # SSL source (unlabelled)
BATCH_TEST      = 'audios5'                  # one-shot test
ALL_BATCHES     = BATCH_LABELED + [BATCH_UNLABELED, BATCH_TEST]

# Encoder layers to extract (last layer of each = whole-pool baseline)
WAVLM_LAYERS   = [6, 9, 12]   # WavLM-base-plus has 12 transformer layers
WHISPER_LAYERS = [12, 18, 24] # Whisper-medium encoder has 24 layers

# CV / fitting
N_FOLDS         = 5
RANDOM_SEED     = 42
DEPLOY_POS_RATE = 0.17
SPW_DEPLOY      = (1.0 - DEPLOY_POS_RATE) / DEPLOY_POS_RATE

# Filter
MIN_SPEAKING_S  = 30

# Caruana ensemble selection
CARUANA_MAX_ITERS = 50

# SSL pseudo-labelling thresholds
T_SIM_MIN          = 0.85   # min agreement (1 - 4*var) to trust
POS_THRESH         = 0.75   # fused-calibrated proba >= this -> consider positive pseudo
NEG_THRESH         = 0.25   # fused-calibrated proba <= this -> consider negative pseudo
MAX_PSEUDO_PER_CLASS = 80   # cap per class per round
SSL_PSEUDO_WEIGHT  = 1.0    # weight pseudo-labels in train (1.0 = same as real labels)

# Paths
NB_DIR        = Path('.').resolve()
SAVE_DIR      = NB_DIR / 'checkpoints_ssl_layer'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DURATIONS_DIR = NB_DIR / 'checkpoints_honest_eval'
DURATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Extraction toggles (flip OFF if you want to skip a step entirely)
EXTRACT_TRANSCRIPTS  = True
EXTRACT_TEXT_FEATS   = True
EXTRACT_WAVLM        = True
EXTRACT_WHISPER_ENC  = True
EXTRACT_DURATIONS    = True

# Encoder model IDs / runtime
WHISPER_TRANSCRIBE_MODEL = 'small'
WAVLM_MODEL              = 'microsoft/wavlm-base-plus'
WHISPER_ENC_MODEL        = 'openai/whisper-medium'
WHISPER_CHUNK_SEC        = 30
MAX_DURATION_SEC         = 120
SR                       = 16000
AUDIO_EXTS               = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.webm', '.mp4'}

print(f'NB_DIR        = {NB_DIR}')
print(f'SAVE_DIR      = {SAVE_DIR}')
print(f'BATCH_LABELED = {BATCH_LABELED}')
print(f'BATCH_UNLABELED = {BATCH_UNLABELED}')
print(f'BATCH_TEST    = {BATCH_TEST}')
print(f'WAVLM_LAYERS  = {WAVLM_LAYERS}   WHISPER_LAYERS = {WHISPER_LAYERS}')
print(f'N_FOLDS={N_FOLDS}  SPW_DEPLOY={SPW_DEPLOY:.2f}  MIN_SPEAKING_S={MIN_SPEAKING_S}')
print(f'SSL: T_SIM_MIN={T_SIM_MIN}  POS_THRESH={POS_THRESH}  NEG_THRESH={NEG_THRESH}  '
      f'cap/class={MAX_PSEUDO_PER_CLASS}')""")

# ============================================================
code("""# === 1. Imports + helpers ===
import re, json, warnings, time, itertools, gc
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              confusion_matrix, roc_auc_score)

warnings.filterwarnings('ignore')

LABEL_MAP = {
    'read':1,'cheating':1,'reading':1,'scripted':1,'yes':1,'1':1,1:1,
    'spontaneous':0,'not cheating':0,'not_cheating':0,'no':0,'0':0,0:0,'genuine':0,
}

_RE_CAND = re.compile(r'^(.+)_(\\d{1,3})\\.[a-zA-Z0-9]+$')
def attach_candidate_id(df):
    df = df.copy()
    df['candidate_id'] = df['filename'].astype(str).map(
        lambda f: (_RE_CAND.match(f).group(1) if _RE_CAND.match(f) else f))
    return df

def _metrics_at(p, y, thr):
    if thr is None or len(p) == 0:
        return dict(prec=np.nan, rec=np.nan, f1=np.nan, tp=0, fp=0, fn=0, tn=0, n=int(len(y)))
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0,1])
    return dict(
        prec = float(precision_score(y, pred, zero_division=0)),
        rec  = float(recall_score(y, pred, zero_division=0)),
        f1   = float(f1_score(y, pred, zero_division=0)),
        tp   = int(cm[1,1]), fp = int(cm[0,1]),
        fn   = int(cm[1,0]), tn = int(cm[0,0]),
        n    = int(len(y)),
    )

F1_THR_GRID = np.arange(0.05, 0.96, 0.01)
def best_f1_thr(p, y):
    if len(p) == 0:
        return 0.5, 0.0
    y_b = y.astype(bool)
    preds = (p[None, :] >= F1_THR_GRID[:, None])
    tp = (preds &  y_b[None, :]).sum(axis=1)
    fp = (preds & ~y_b[None, :]).sum(axis=1)
    fn = (~preds &  y_b[None, :]).sum(axis=1)
    denom = (2*tp + fp + fn)
    f1 = np.where(denom > 0, 2*tp / np.maximum(denom, 1), 0.0)
    k = int(np.argmax(f1))
    return float(F1_THR_GRID[k]), float(f1[k])

def caruana_select(oof_dict, y, max_iters=CARUANA_MAX_ITERS):
    \"\"\"Caruana 2004 greedy forward ensemble selection.

    At each step, pick the base that, when added (with replacement), maximises
    F1-max on the running average. Returns weights as integer counts / total picks.
    \"\"\"
    names    = list(oof_dict.keys())
    cur_sum  = np.zeros(len(y))
    selected = []
    history  = []
    best_f1  = 0.0
    for it in range(max_iters):
        best_pick     = None
        best_pick_f1  = best_f1
        best_pick_thr = 0.5
        for n in names:
            new_sum = cur_sum + oof_dict[n]
            new_avg = new_sum / (len(selected) + 1)
            thr, f1 = best_f1_thr(new_avg, y)
            if f1 > best_pick_f1 + 1e-9:
                best_pick     = n
                best_pick_f1  = f1
                best_pick_thr = thr
        if best_pick is None:
            break
        selected.append(best_pick)
        cur_sum += oof_dict[best_pick]
        best_f1  = best_pick_f1
        history.append({'iter': it+1, 'pick': best_pick, 'f1': best_pick_f1,
                         'thr': best_pick_thr, 'n_selected': len(selected)})
    if not selected:
        # Fallback: equal weights
        weights = {n: 1.0/len(names) for n in names}
        return weights, [], 0.0, 0.5, history
    counts  = {n: selected.count(n) for n in set(selected)}
    weights = {n: c/len(selected) for n, c in counts.items()}
    final_avg = cur_sum / len(selected)
    final_thr, final_f1 = best_f1_thr(final_avg, y)
    return weights, selected, final_f1, final_thr, history

def fused_proba(weights, proba_dict):
    \"\"\"Weighted average of per-base probas. Bases not in weights get weight 0.\"\"\"
    out = None
    total_w = 0.0
    for n, w in weights.items():
        if n not in proba_dict or w <= 0: continue
        if out is None: out = np.zeros_like(proba_dict[n], dtype=float)
        out += float(w) * proba_dict[n]
        total_w += float(w)
    if out is None or total_w == 0:
        # All zero -> return constant 0.5
        any_p = next(iter(proba_dict.values()))
        return np.full_like(any_p, 0.5, dtype=float)
    return out / total_w

print('Imports + helpers ready.')""")

# ============================================================
md("""## 1. Extraction (resume-safe, all four batches)

Each step caches per-batch and skips files that are already extracted. First run on a fresh batch is the heavy one — subsequent reruns are seconds.""")

# ============================================================
code("""# === 1.0 Audio scan + paths ===
def batch_paths(name):
    return {
        'audio_root': NB_DIR / name,
        'gt':         NB_DIR / f'{name}GT.csv',
        'transcripts': NB_DIR / f'{name}_transcripts.json',
        'features':    NB_DIR / f'{name}_features.csv',
        'durations':   DURATIONS_DIR / f'{name}_durations.csv',
        'wavlm_layers':   {L: NB_DIR / f'{name}_wavlm_L{L}.csv'   for L in WAVLM_LAYERS},
        'whisper_layers': {L: NB_DIR / f'{name}_whisper_L{L}.csv' for L in WHISPER_LAYERS},
    }

batch_audio_files = {}
for name in ALL_BATCHES:
    P = batch_paths(name)
    if not P['audio_root'].exists():
        print(f'WARN: missing folder {P[\"audio_root\"]} — extraction will skip {name}')
        batch_audio_files[name] = []
        continue
    files = sorted(f for f in P['audio_root'].rglob('*')
                   if f.suffix.lower() in AUDIO_EXTS and f.is_file())
    batch_audio_files[name] = files
    has_gt = 'unlabelled (no GT expected)' if name == BATCH_UNLABELED else (
        'GT present' if P['gt'].exists() else f'MISSING GT: {P[\"gt\"].name}')
    print(f'  {name:8s}  audios={len(files):4d}   {has_gt}')""")

# ============================================================
code("""# === 1.1 Whisper transcription (all batches, resume-safe) ===
import soundfile as sf
import librosa
from tqdm import tqdm

FILLER_PROMPT = ('Umm, let me think like, hmm... Okay here\\'s what I\\'m thinking. '
                 'So uh, basically, you know, I mean, like, right.')

_whisper_trans_model = None
def _get_whisper_transcriber():
    global _whisper_trans_model
    if _whisper_trans_model is None:
        from faster_whisper import WhisperModel
        import torch as _torch
        device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        compute_type = 'float16' if device == 'cuda' else 'int8'
        print(f'  Loading Whisper {WHISPER_TRANSCRIBE_MODEL} ({device}, {compute_type})')
        _whisper_trans_model = WhisperModel(WHISPER_TRANSCRIBE_MODEL, device=device,
                                             compute_type=compute_type)
    return _whisper_trans_model

def transcribe_batch(name):
    if not EXTRACT_TRANSCRIPTS:
        return
    P = batch_paths(name)
    files = batch_audio_files.get(name, [])
    if not files: return
    existing = json.load(open(P['transcripts'], encoding='utf-8')) if P['transcripts'].exists() else {}
    todo = [f for f in files if f.name not in existing]
    if not todo:
        print(f'  {name}: {len(existing)} transcripts cached, nothing to do.')
        return
    whisper = _get_whisper_transcriber()
    print(f'  {name}: transcribing {len(todo)} new files')
    for fp in tqdm(todo, desc=name):
        try:
            segs, info = whisper.transcribe(
                str(fp), language='en', word_timestamps=True,
                initial_prompt=FILLER_PROMPT,
                vad_filter=True, vad_parameters={'min_silence_duration_ms': 100},
            )
            words, parts, segments = [], [], []
            for seg in segs:
                parts.append(seg.text)
                segments.append({'start': float(seg.start), 'end': float(seg.end)})
                if seg.words:
                    for w in seg.words:
                        words.append({'word': w.word.strip(),
                                      'start': round(w.start, 3),
                                      'end':   round(w.end,   3)})
            existing[fp.name] = {
                'text': ' '.join(parts).strip(),
                'words': words,
                'segments': segments,
                'duration_sec': round(info.duration, 2),
            }
        except Exception as e:
            print(f'  FAIL {fp.name}: {e}')
            existing[fp.name] = {'text': '', 'words': [], 'segments': [], 'duration_sec': 0}
        with open(P['transcripts'], 'w', encoding='utf-8') as fh:
            json.dump(existing, fh, ensure_ascii=False, indent=1)
    print(f'  {name}: total {len(existing)} transcripts cached.')

for b in ALL_BATCHES:
    transcribe_batch(b)
del _whisper_trans_model
_whisper_trans_model = None
gc.collect()""")

# ============================================================
code("""# === 1.2 Text feature extraction (all batches, resume-safe) ===
# Mirrors text_cheating_detection.ipynb. 50 features, 7 groups.
from collections import Counter

USE_PROSODIC      = True
USE_VOICE_QUALITY = True
USE_PERPLEXITY    = True

try:
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False
    print('  spaCy not available — POS features will be 0.')

if USE_VOICE_QUALITY:
    try:
        import parselmouth
        from parselmouth.praat import call as praat_call
        HAS_PARSELMOUTH = True
    except ImportError:
        HAS_PARSELMOUTH = False
        print('  parselmouth not installed — voice-quality features will be 0.')
else:
    HAS_PARSELMOUTH = False

if USE_PERPLEXITY:
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _gpt2_tok = None; _gpt2_mdl = None
        def _gpt2():
            global _gpt2_tok, _gpt2_mdl
            if _gpt2_mdl is None:
                _gpt2_tok = GPT2TokenizerFast.from_pretrained('gpt2')
                _gpt2_mdl = GPT2LMHeadModel.from_pretrained('gpt2').eval()
            return _gpt2_mdl, _gpt2_tok
        HAS_GPT2 = True
    except ImportError:
        HAS_GPT2 = False
        print('  transformers not available — perplexity features will be 0.')
else:
    HAS_GPT2 = False

FILLERS           = {'um','uh','uh-huh','uhm','umm','hmm','hm','er','ah','ehm','mhm'}
DISCOURSE_MARKERS = {'you know','i mean','like','basically','actually','so','well','right','okay','oh','anyway','honestly'}
HEDGES            = {'i think','i guess','maybe','perhaps','probably','kind of','sort of','i believe','it seems','i suppose','might be'}
SELF_REF          = {'i','me','my','myself','mine',\"i'm\",\"i've\",\"i'd\",\"i'll\"}
REPAIRS           = ['i mean','no wait','sorry i','actually no','wait no','no no']
FORMAL_TRANS      = ['furthermore','moreover','however','therefore','additionally','consequently',
                     'nevertheless','hence','thus','in conclusion','firstly','secondly','thirdly',
                     'in summary','to summarize','in essence','overall','ultimately']
AI_PHRASES        = ['it is important to note','it is worth noting','it should be noted',
                     'in conclusion','to summarize','in summary','fundamentally',
                     'plays a crucial role','plays a vital role','a wide range of',
                     'on the other hand','in other words','delve into','it is crucial']
CONTENT_POS       = {'NOUN','VERB','ADJ','ADV','PROPN'}
FUNCTION_POS      = {'DET','ADP','CONJ','CCONJ','SCONJ','PRON','AUX','PART'}

G_DISFLUENCY  = ['filler_rate','filler_count','repetition_rate','repair_rate',
                 'discourse_marker_rate','hedge_rate']
G_STYLOMETRIC = ['ttr','mattr','mtld','complex_word_rate','avg_word_length',
                 'n_words','n_unique_words','avg_sentence_length','std_sentence_length',
                 'fragment_rate','n_sentences','self_ref_rate','noun_rate','verb_rate','adj_rate']
G_PAUSE       = ['pause_mean','pause_std','pause_median','pause_skew','long_pause_rate',
                 'pause_ratio','n_pauses','pause_regularity',
                 'pause_before_content_ratio','pause_before_function_ratio',
                 'mid_phrase_pause_rate','words_per_sec','articulation_rate',
                 'initial_pause','longest_pause']
G_SUSPICIOUS  = ['suspicious_gap_count','suspicious_gap_ratio']
G_FORMAL_AI   = ['formal_transition_count','formal_transition_rate',
                 'ai_phrase_count','ai_phrase_rate']
G_PROSODIC    = ['f0_mean','f0_std','f0_range','f0_skew','f0_slope',
                 'energy_mean','energy_std','speaking_rate_std']
G_VOICE_Q     = ['jitter_local','shimmer_local','hnr_mean']
G_PERPLEXITY  = ['mean_perplexity','burstiness']
ALL_FEATURES  = (G_DISFLUENCY + G_STYLOMETRIC + G_PAUSE + G_SUSPICIOUS +
                 G_FORMAL_AI + G_PROSODIC + G_VOICE_Q + G_PERPLEXITY)

WORD_RE = re.compile(r\"[a-zA-Z']+\")

def _syllable_count(word):
    word = word.lower().strip()
    if len(word) <= 3: return 1
    count, prev_vowel = 0, False
    for ch in word:
        iv = ch in 'aeiouy'
        if iv and not prev_vowel: count += 1
        prev_vowel = iv
    if word.endswith('e') and count > 1: count -= 1
    return max(count, 1)

def _mattr(words, window=50):
    if len(words) < window: return len(set(words)) / max(len(words), 1)
    return float(np.mean([len(set(words[i:i+window])) / window for i in range(len(words)-window+1)]))

def _mtld(words, threshold=0.72):
    if len(words) < 10: return 0.0
    def _run(ws):
        factors, current = 0, []
        for w in ws:
            current.append(w)
            ttr = len(set(current)) / len(current)
            if ttr <= threshold:
                factors += 1; current = []
        if current:
            ttr = len(set(current)) / len(current)
            if ttr < 1.0:
                factors += (1.0 - ttr) / (1.0 - threshold)
        return len(ws) / factors if factors > 0 else len(ws)
    return round((_run(words) + _run(words[::-1])) / 2, 2)

def compute_text_and_stylo(text):
    out = {k: 0 for k in G_DISFLUENCY + G_STYLOMETRIC}
    if not text or len(text.strip()) < 10: return out
    text_lower = text.lower().strip()
    if HAS_SPACY:
        doc = nlp(text_lower)
        words = [t.text for t in doc if t.is_alpha]
        all_toks = [t.text for t in doc]
        sentences = list(doc.sents)
        pos_c = Counter(t.pos_ for t in doc)
    else:
        words = WORD_RE.findall(text_lower)
        all_toks = words
        sentences = [s for s in re.split(r'[.!?]+', text_lower) if s.strip()]
        pos_c = Counter()
    n_words = len(words)
    if n_words < 5: return out
    filler_count = sum(1 for w in all_toks if w in FILLERS)
    bigrams = [f'{words[i]} {words[i+1]}' for i in range(len(words)-1)]
    bc = Counter(bigrams)
    rep_rate = sum(c-1 for c in bc.values() if c>1) / max(len(bigrams),1)
    repair_c = sum(text_lower.count(r) for r in REPAIRS)
    n_sents = max(len(sentences), 1)
    sl = [len([t for t in s if getattr(t,'is_alpha',True)]) for s in sentences] if HAS_SPACY \\
         else [len(WORD_RE.findall(s)) for s in sentences]
    sl = [x for x in sl if x > 0]
    tp = sum(pos_c.values()) or 1
    out.update({
        'filler_rate':       filler_count/n_words,
        'filler_count':      filler_count,
        'repetition_rate':   rep_rate,
        'repair_rate':       repair_c/n_sents,
        'discourse_marker_rate': sum(text_lower.count(d) for d in DISCOURSE_MARKERS)/n_sents,
        'hedge_rate':        sum(text_lower.count(h) for h in HEDGES)/n_sents,
        'ttr':               len(set(words))/n_words,
        'mattr':             _mattr(words),
        'mtld':              _mtld(words),
        'complex_word_rate': sum(1 for w in words if _syllable_count(w)>=3)/n_words,
        'avg_word_length':   float(np.mean([len(w) for w in words])),
        'n_words':           n_words,
        'n_unique_words':    len(set(words)),
        'avg_sentence_length': float(np.mean(sl)) if sl else 0,
        'std_sentence_length': float(np.std(sl))  if len(sl)>1 else 0,
        'fragment_rate':     sum(1 for x in sl if x<4)/n_sents,
        'n_sentences':       n_sents,
        'self_ref_rate':     sum(1 for w in all_toks if w in SELF_REF)/n_words,
        'noun_rate':         pos_c.get('NOUN',0)/tp,
        'verb_rate':         pos_c.get('VERB',0)/tp,
        'adj_rate':          pos_c.get('ADJ',0)/tp,
    })
    return out

def compute_pause_and_suspicious(words):
    out = {k: 0 for k in G_PAUSE + G_SUSPICIOUS}
    if not words or len(words) < 5: return out
    pauses = []
    for i in range(1, len(words)):
        gap = words[i]['start'] - words[i-1]['end']
        if gap > 0.05:
            pauses.append({'dur': gap, 'after_word': words[i-1].get('word',''),
                           'before_word': words[i].get('word',''), 'pos': i})
    initial_pause = words[0]['start']
    all_gaps = [words[i]['start']-words[i-1]['end']
                for i in range(1,len(words))
                if words[i]['start']-words[i-1]['end']>0.05]
    longest_pause = max(all_gaps) if all_gaps else 0.0
    out['initial_pause'] = initial_pause
    out['longest_pause'] = longest_pause
    if not pauses: return out
    durs = [p['dur'] for p in pauses]
    total_dur = max(words[-1]['end'] - words[0]['start'], 0.1)
    speaking_dur = max(total_dur - sum(durs), 0.1)
    if HAS_SPACY:
        doc = nlp(' '.join(w.get('word','') for w in words))
        tok_pos = {t.text.lower(): t.pos_ for t in doc}
    else:
        tok_pos = {}
    nbc = nbf = nmp = 0
    for p in pauses:
        pos = tok_pos.get(p['before_word'].lower().strip('.,!?'), 'X')
        if pos in CONTENT_POS:    nbc += 1
        elif pos in FUNCTION_POS: nbf += 1
        if not p['after_word'].endswith(('.',',','!','?')): nmp += 1
    n_p = len(pauses)
    positions = [p['pos'] for p in pauses]
    intervals = [positions[i]-positions[i-1] for i in range(1, len(positions))]
    suspicious = sum(1 for p in pauses
                     if 0.3 <= p['dur'] <= 0.8
                     and not p['after_word'].rstrip().endswith(('.','!','?')))
    out.update({
        'pause_mean':      float(np.mean(durs)),
        'pause_std':       float(np.std(durs)),
        'pause_median':    float(np.median(durs)),
        'pause_skew':      float(pd.Series(durs).skew()) if len(durs)>2 else 0,
        'long_pause_rate': sum(1 for d in durs if d>0.5)/n_p,
        'pause_ratio':     sum(durs)/total_dur,
        'n_pauses':        n_p,
        'pause_regularity': float(np.std(intervals)) if intervals else 0,
        'pause_before_content_ratio':  nbc/n_p,
        'pause_before_function_ratio': nbf/n_p,
        'mid_phrase_pause_rate':       nmp/n_p,
        'words_per_sec':   len(words)/total_dur,
        'articulation_rate': len(words)/speaking_dur,
        'initial_pause':   initial_pause,
        'longest_pause':   longest_pause,
        'suspicious_gap_count': suspicious,
        'suspicious_gap_ratio': suspicious/max(len(words),1),
    })
    return out

def compute_formal_ai(text):
    out = {k: 0 for k in G_FORMAL_AI}
    if not text: return out
    tl = text.lower()
    n_words = max(len(WORD_RE.findall(tl)), 1)
    formal_c = sum(tl.count(p) for p in FORMAL_TRANS)
    ai_c     = sum(tl.count(p) for p in AI_PHRASES)
    out['formal_transition_count'] = formal_c
    out['formal_transition_rate']  = 100.0 * formal_c / n_words
    out['ai_phrase_count']         = ai_c
    out['ai_phrase_rate']          = 100.0 * ai_c / n_words
    return out

def compute_prosodic(audio_path):
    out = {k: 0 for k in G_PROSODIC}
    if not USE_PROSODIC: return out
    try:
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True, duration=120)
    except Exception:
        return out
    if len(audio) < 16000: return out
    f0, _, _ = librosa.pyin(audio, fmin=75, fmax=500, sr=16000, frame_length=2048)
    fv = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if len(fv) >= 10:
        slope = float(np.polyfit(np.arange(len(fv)), fv, 1)[0])
        out['f0_mean']  = float(np.mean(fv))
        out['f0_std']   = float(np.std(fv))
        out['f0_range'] = float(fv.max()-fv.min())
        out['f0_skew']  = float(pd.Series(fv).skew())
        out['f0_slope'] = slope
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
    out['energy_mean'] = float(np.mean(rms))
    out['energy_std']  = float(np.std(rms))
    win = 2*16000
    rates = [float((librosa.feature.rms(y=audio[s:s+win], frame_length=512, hop_length=256)[0]
                    > np.percentile(rms,20)).mean())
             for s in range(0, len(audio)-win, 16000)]
    out['speaking_rate_std'] = float(np.std(rates)) if rates else 0
    return out

def compute_voice_quality(audio_path):
    out = {k: 0 for k in G_VOICE_Q}
    if not HAS_PARSELMOUTH: return out
    try:
        snd  = parselmouth.Sound(str(audio_path))
        pp   = praat_call(snd, 'To PointProcess (periodic, cc)', 75, 500)
        jit  = praat_call(pp,  'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)
        shim = praat_call([snd, pp], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = praat_call(snd, 'To Harmonicity (cc)', 0.01, 75, 0.1, 1.0)
        hnr  = praat_call(harm, 'Get mean', 0, 0)
        out.update({'jitter_local': float(jit), 'shimmer_local': float(shim), 'hnr_mean': float(hnr)})
    except Exception:
        pass
    return out

def compute_perplexity(text):
    out = {k: 0 for k in G_PERPLEXITY}
    if not HAS_GPT2 or not text or len(text.strip()) < 20: return out
    try:
        mdl, tok = _gpt2()
        sents = [s for s in re.split(r'(?<=[.!?])\\s+', text.strip()) if len(s.split())>3]
        if not sents: return out
        ppls = []
        for s in sents[:20]:
            enc = tok(s, return_tensors='pt', truncation=True, max_length=256)
            with torch.no_grad():
                loss = mdl(**enc, labels=enc['input_ids']).loss
            ppls.append(float(torch.exp(loss)))
        out['mean_perplexity'] = float(np.mean(ppls))
        out['burstiness']      = float(np.var(ppls))
    except Exception:
        pass
    return out

def compute_all_features(fp, text, words):
    r = {}
    r.update(compute_text_and_stylo(text))
    r.update(compute_pause_and_suspicious(words))
    r.update(compute_formal_ai(text))
    r.update(compute_prosodic(fp))
    r.update(compute_voice_quality(fp))
    r.update(compute_perplexity(text))
    return r

def extract_text_features_batch(name):
    if not EXTRACT_TEXT_FEATS: return
    P = batch_paths(name); files = batch_audio_files.get(name, [])
    if not files: return
    if not P['transcripts'].exists():
        print(f'  {name}: missing transcripts — run cell 1.1 first.'); return
    t = json.load(open(P['transcripts'], encoding='utf-8'))
    have_files = set()
    if P['features'].exists():
        df_cached = pd.read_csv(P['features'])
        have_files = set(df_cached['filename'].tolist())
    todo = [fp for fp in files if fp.name not in have_files]
    if not todo:
        print(f'  {name}: text features all extracted ({len(have_files)} files).'); return
    print(f'  {name}: extracting text features for {len(todo)} files')
    rows = []
    for fp in tqdm(todo, desc=f'{name} feat'):
        tr = t.get(fp.name, {'text':'','words':[],'duration_sec':0})
        r = compute_all_features(fp, tr.get('text',''), tr.get('words',[]))
        r['filename']     = fp.name
        r['duration_sec'] = tr.get('duration_sec', 0)
        rows.append(r)
        if len(rows) >= 10:
            pd.DataFrame(rows).to_csv(P['features'], mode='a',
                                       header=not P['features'].exists(), index=False)
            rows = []
    if rows:
        pd.DataFrame(rows).to_csv(P['features'], mode='a',
                                   header=not P['features'].exists(), index=False)
    print(f'  {name}: text features cached -> {P[\"features\"].name}')

for b in ALL_BATCHES:
    extract_text_features_batch(b)""")

# ============================================================
code("""# === 1.3 Multi-layer WavLM extraction (one forward pass, multiple layers) ===
def extract_wavlm_layers_batch(name):
    if not EXTRACT_WAVLM: return
    P = batch_paths(name); files = batch_audio_files.get(name, [])
    if not files: return
    layer_paths = P['wavlm_layers']
    layers_to_do = [L for L in WAVLM_LAYERS if not layer_paths[L].exists()]
    if not layers_to_do:
        print(f'  {name}: wavlm layers {WAVLM_LAYERS} all cached.'); return
    print(f'  {name}: extracting WavLM layers {layers_to_do}')
    import torch as _torch
    from transformers import AutoFeatureExtractor, WavLMModel
    device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    print(f'    Loading {WAVLM_MODEL} ({device})')
    fe  = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    mdl = WavLMModel.from_pretrained(WAVLM_MODEL).eval().to(device)

    # Per-layer accumulators (rows of dicts)
    rows_by_layer = {L: [] for L in layers_to_do}
    flush_every = 25

    def _flush():
        for L, rows in rows_by_layer.items():
            if not rows: continue
            df_new = pd.DataFrame(rows)
            if layer_paths[L].exists():
                df_new.to_csv(layer_paths[L], mode='a', header=False, index=False)
            else:
                df_new.to_csv(layer_paths[L], mode='w', header=True, index=False)
            rows_by_layer[L] = []

    # Files already in any cache for this batch -> need to figure out per-layer
    cached_per_layer = {L: set() for L in layers_to_do}
    for L in layers_to_do:
        if layer_paths[L].exists():
            cached_per_layer[L] = set(pd.read_csv(layer_paths[L], usecols=['filename'])['filename'].tolist())

    with _torch.no_grad():
        for fp in tqdm(files, desc=f'{name} wavlm'):
            # Skip if all desired layers already have this file
            if all(fp.name in cached_per_layer[L] for L in layers_to_do):
                continue
            try:
                y, sr = sf.read(str(fp), always_2d=False)
                if y.ndim > 1: y = y.mean(axis=1)
                if sr != SR:  y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
                y = y.astype(np.float32)
                if MAX_DURATION_SEC and len(y) > MAX_DURATION_SEC * SR:
                    y = y[:int(MAX_DURATION_SEC * SR)]
                inp = fe(y, sampling_rate=SR, return_tensors='pt', padding=False)
                out = mdl(inp.input_values.to(device), output_hidden_states=True)
                # out.hidden_states is a tuple: index 0 = embeddings, 1..N = layer outputs
                for L in layers_to_do:
                    if fp.name in cached_per_layer[L]: continue
                    h = out.hidden_states[L]
                    emb = h.mean(dim=1).squeeze(0).cpu().numpy()
                    row = {'filename': fp.name}
                    for i, v in enumerate(emb):
                        row[f'wavlm_L{L}_{i}'] = round(float(v), 6)
                    rows_by_layer[L].append(row)
            except Exception as e:
                print(f'  WARN wavlm {fp.name}: {e}')
            if sum(len(r) for r in rows_by_layer.values()) >= flush_every:
                _flush()
    _flush()
    print(f'  {name}: wavlm layers cached -> ' +
          ', '.join(layer_paths[L].name for L in layers_to_do))
    del mdl, fe
    if device == 'cuda': _torch.cuda.empty_cache()
    gc.collect()

for b in ALL_BATCHES:
    extract_wavlm_layers_batch(b)""")

# ============================================================
code("""# === 1.4 Multi-layer Whisper encoder extraction (mean over chunks, per layer) ===
def extract_whisper_layers_batch(name):
    if not EXTRACT_WHISPER_ENC: return
    P = batch_paths(name); files = batch_audio_files.get(name, [])
    if not files: return
    layer_paths = P['whisper_layers']
    layers_to_do = [L for L in WHISPER_LAYERS if not layer_paths[L].exists()]
    if not layers_to_do:
        print(f'  {name}: whisper layers {WHISPER_LAYERS} all cached.'); return
    print(f'  {name}: extracting Whisper-encoder layers {layers_to_do}')
    import torch as _torch
    from transformers import WhisperProcessor, WhisperModel
    device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    print(f'    Loading {WHISPER_ENC_MODEL} ({device})')
    proc = WhisperProcessor.from_pretrained(WHISPER_ENC_MODEL)
    mdl  = WhisperModel.from_pretrained(WHISPER_ENC_MODEL).eval().to(device)
    chunk_samples = WHISPER_CHUNK_SEC * SR

    rows_by_layer = {L: [] for L in layers_to_do}
    flush_every = 25
    cached_per_layer = {L: set() for L in layers_to_do}
    for L in layers_to_do:
        if layer_paths[L].exists():
            cached_per_layer[L] = set(pd.read_csv(layer_paths[L], usecols=['filename'])['filename'].tolist())

    def _flush():
        for L, rows in rows_by_layer.items():
            if not rows: continue
            df_new = pd.DataFrame(rows)
            if layer_paths[L].exists():
                df_new.to_csv(layer_paths[L], mode='a', header=False, index=False)
            else:
                df_new.to_csv(layer_paths[L], mode='w', header=True, index=False)
            rows_by_layer[L] = []

    with _torch.no_grad():
        for fp in tqdm(files, desc=f'{name} whisper-enc'):
            if all(fp.name in cached_per_layer[L] for L in layers_to_do):
                continue
            try:
                y, sr = sf.read(str(fp), always_2d=False)
                if y.ndim > 1: y = y.mean(axis=1)
                if sr != SR:  y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
                y = y.astype(np.float32)
                if MAX_DURATION_SEC and len(y) > MAX_DURATION_SEC * SR:
                    y = y[:int(MAX_DURATION_SEC * SR)]
                if len(y) <= chunk_samples:
                    chunks = [y]
                else:
                    chunks = [y[i:i+chunk_samples] for i in range(0, len(y), chunk_samples)]
                # Per-layer per-chunk means
                layer_chunk_means = {L: [] for L in layers_to_do}
                for c in chunks:
                    if len(c) < int(SR * 0.5): continue
                    feat = proc(c, sampling_rate=SR, return_tensors='pt').input_features.to(device)
                    out  = mdl.encoder(feat, output_hidden_states=True)
                    # out.hidden_states is a tuple of length n_layers + 1; last = final encoder layer
                    for L in layers_to_do:
                        h = out.hidden_states[L]
                        layer_chunk_means[L].append(h.mean(dim=1).squeeze(0).cpu().numpy())
                if not any(layer_chunk_means.values()): continue
                for L in layers_to_do:
                    if fp.name in cached_per_layer[L]: continue
                    if not layer_chunk_means[L]: continue
                    emb = np.mean(np.stack(layer_chunk_means[L]), axis=0)
                    row = {'filename': fp.name}
                    for i, v in enumerate(emb):
                        row[f'whisper_L{L}_{i}'] = round(float(v), 6)
                    rows_by_layer[L].append(row)
            except Exception as e:
                print(f'  WARN whisper-enc {fp.name}: {e}')
            if sum(len(r) for r in rows_by_layer.values()) >= flush_every:
                _flush()
    _flush()
    print(f'  {name}: whisper-enc layers cached -> ' +
          ', '.join(layer_paths[L].name for L in layers_to_do))
    del mdl, proc
    if device == 'cuda': _torch.cuda.empty_cache()
    gc.collect()

for b in ALL_BATCHES:
    extract_whisper_layers_batch(b)""")

# ============================================================
code("""# === 1.5 Durations (cached) ===
def extract_durations_batch(name):
    if not EXTRACT_DURATIONS: return
    P = batch_paths(name); files = batch_audio_files.get(name, [])
    if not files or P['durations'].exists():
        if P['durations'].exists():
            print(f'  {name}: durations cached -> {P[\"durations\"].name}')
        return
    if not P['transcripts'].exists():
        print(f'  {name}: missing transcripts — cannot compute speaking_time'); return
    t = json.load(open(P['transcripts'], encoding='utf-8'))
    rows = []
    for fp in files:
        tr = t.get(fp.name, {})
        words = tr.get('words', [])
        total = tr.get('duration_sec', 0)
        if words:
            speaking = sum((w['end'] - w['start']) for w in words if w.get('end',0) > w.get('start',0))
        else:
            speaking = total
        rows.append({'filename': fp.name,
                     'total_duration_s': float(total),
                     'speaking_time_s': float(speaking),
                     'source': 'whisper_words' if words else 'whisper_total'})
    pd.DataFrame(rows).to_csv(P['durations'], index=False)
    print(f'  {name}: durations cached -> {P[\"durations\"].name}')

for b in ALL_BATCHES:
    extract_durations_batch(b)""")

# ============================================================
md("""## 2. Load all batches + filter""")

# ============================================================
code("""# === 2.1 Load + filter ===
def load_gt(name):
    P = batch_paths(name)
    if not P['gt'].exists():
        # Unlabelled: synthesise placeholder labels = -1
        files = [f.name for f in batch_audio_files.get(name, [])]
        return pd.DataFrame({'filename': files, 'label_int': [-1]*len(files)})
    gt = pd.read_csv(P['gt'])
    fn_col  = next(c for c in gt.columns if c.lower() in ('filename','file','name'))
    lbl_col = next((c for c in gt.columns
                    if c.lower() in ('label','class','cheating','gt','label_int','ground_truth')), None)
    gt = gt.rename(columns={fn_col:'filename'})
    if lbl_col is None:
        gt['label_int'] = -1
    else:
        gt = gt.rename(columns={lbl_col:'label_raw'})
        gt['label_int'] = gt['label_raw'].map(
            lambda x: LABEL_MAP.get(x, LABEL_MAP.get(str(x).lower().strip(), -1)))
    return gt[['filename','label_int']]

def load_durations(name):
    P = batch_paths(name)
    if not P['durations'].exists(): return None
    return pd.read_csv(P['durations'])[['filename','speaking_time_s']]

def load_folder(name):
    P    = batch_paths(name)
    gt   = load_gt(name)
    text = pd.read_csv(P['features'])
    df   = gt.merge(text, on='filename', how='inner')
    for L in WAVLM_LAYERS:
        df = df.merge(pd.read_csv(P['wavlm_layers'][L]), on='filename', how='inner')
    for L in WHISPER_LAYERS:
        df = df.merge(pd.read_csv(P['whisper_layers'][L]), on='filename', how='inner')
    df['batch'] = name
    dur = load_durations(name)
    if dur is not None:
        df = df.merge(dur, on='filename', how='left')
    return df

def filter_by_duration(df, min_s):
    if min_s <= 0 or 'speaking_time_s' not in df.columns: return df
    keep = (df['speaking_time_s'] >= min_s) | df['speaking_time_s'].isna()
    return df[keep].reset_index(drop=True)

batches_full = {b: attach_candidate_id(load_folder(b))            for b in ALL_BATCHES}
batches      = {b: filter_by_duration(batches_full[b], MIN_SPEAKING_S) for b in ALL_BATCHES}

print(f'=== Per-batch counts (raw -> filtered at >= {MIN_SPEAKING_S}s) ===')
for b in ALL_BATCHES:
    f0, f1 = batches_full[b], batches[b]
    y0, y1 = f0['label_int'].values, f1['label_int'].values
    pos0, pos1 = int((y0==1).sum()), int((y1==1).sum())
    neg0, neg1 = int((y0==0).sum()), int((y1==0).sum())
    unl0, unl1 = int((y0==-1).sum()), int((y1==-1).sum())
    print(f'  {b:8s}:  rows {len(f0):4d} -> {len(f1):4d}   '
          f'cheat {pos0:3d}->{pos1:3d}   honest {neg0:3d}->{neg1:3d}   unlabelled {unl0:3d}->{unl1:3d}   '
          f'cands {f0[\"candidate_id\"].nunique()}->{f1[\"candidate_id\"].nunique()}')""")

# ============================================================
code("""# === 2.2 Feature column sets + text TopN ranking on audios2 ===
ALL_TEXT_FEATURES = [
    'filler_rate','filler_count','repetition_rate','repair_rate','discourse_marker_rate','hedge_rate',
    'ttr','mattr','mtld','complex_word_rate','avg_word_length','n_words','n_unique_words',
    'avg_sentence_length','std_sentence_length','fragment_rate','n_sentences','self_ref_rate',
    'noun_rate','verb_rate','adj_rate',
    'pause_mean','pause_std','pause_median','pause_skew','long_pause_rate','pause_ratio','n_pauses',
    'pause_regularity','pause_before_content_ratio','pause_before_function_ratio','mid_phrase_pause_rate',
    'words_per_sec','articulation_rate','initial_pause','longest_pause',
    'suspicious_gap_count','suspicious_gap_ratio',
    'formal_transition_count','formal_transition_rate','ai_phrase_count','ai_phrase_rate',
    'f0_mean','f0_std','f0_range','f0_skew','f0_slope','energy_mean','energy_std','speaking_rate_std',
    'jitter_local','shimmer_local','hnr_mean',
    'mean_perplexity','burstiness',
]
STYLO_FEATS = ['ttr','mattr','mtld','complex_word_rate','avg_word_length','n_words','n_unique_words',
               'avg_sentence_length','std_sentence_length','fragment_rate','n_sentences','self_ref_rate',
               'noun_rate','verb_rate','adj_rate']
TEXT_RANK_COLS = [f for f in ALL_TEXT_FEATURES
                  if f not in ('f0_mean','f0_std','f0_range','f0_skew','f0_slope',
                               'energy_mean','energy_std','speaking_rate_std',
                               'jitter_local','shimmer_local','hnr_mean',
                               'mean_perplexity','burstiness')]

first = batches['audios2']
TEXT_ALL   = [c for c in ALL_TEXT_FEATURES if c in first.columns]
TEXT_STYLO = [c for c in STYLO_FEATS       if c in first.columns]
TEXT_RANK  = [c for c in TEXT_RANK_COLS    if c in first.columns]
WAVLM_LAYER_COLS   = {L: [c for c in first.columns if c.startswith(f'wavlm_L{L}_')]
                      for L in WAVLM_LAYERS}
WHISPER_LAYER_COLS = {L: [c for c in first.columns if c.startswith(f'whisper_L{L}_')]
                      for L in WHISPER_LAYERS}

# Text TopN — XGB importance ranking on audios2
_X = first[TEXT_RANK].fillna(0).values
_y = first['label_int'].values
_sc = StandardScaler().fit(_X)
_rkr = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    scale_pos_weight=float(SPW_DEPLOY), eval_metric='logloss',
    random_state=RANDOM_SEED)
_rkr.fit(_sc.transform(_X), _y)
_imp = pd.Series(_rkr.feature_importances_, index=TEXT_RANK).sort_values(ascending=False)
TEXT_TOP10 = _imp.head(10).index.tolist()
TEXT_TOP15 = _imp.head(15).index.tolist()
TEXT_TOP20 = _imp.head(20).index.tolist()

print(f'text_all d={len(TEXT_ALL)}  text_stylo d={len(TEXT_STYLO)}  text_rank d={len(TEXT_RANK)}')
for L in WAVLM_LAYERS:
    print(f'  wavlm_L{L}  d={len(WAVLM_LAYER_COLS[L])}')
for L in WHISPER_LAYERS:
    print(f'  whisper_L{L} d={len(WHISPER_LAYER_COLS[L])}')""")

# ============================================================
md("""## 3. Base registry

17 base models:
- 5 text XGBs: top10 / top15 / top20 / stylo / all
- WavLM layers {6, 9, 12} × {RF, XGB} = 6
- Whisper-enc layers {12, 18, 24} × {RF, XGB} = 6""")

# ============================================================
code("""# === 3. Base registry ===
def make_xgb(n_feats, seed=RANDOM_SEED):
    cs = 0.3 if n_feats > 500 else 0.8
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=cs, min_child_weight=3,
        scale_pos_weight=float(SPW_DEPLOY), eval_metric='logloss',
        random_state=seed)

def make_rf(n_feats, seed=RANDOM_SEED):
    return RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        class_weight={0:1.0, 1:float(SPW_DEPLOY)},
        n_jobs=-1, random_state=seed)

def mk_X(cols): return lambda d: d[cols].fillna(0).values

BASE_REGISTRY = {
    # Text
    'text_top10_xgb':  (mk_X(TEXT_TOP10), lambda s=RANDOM_SEED: make_xgb(len(TEXT_TOP10), s)),
    'text_top15_xgb':  (mk_X(TEXT_TOP15), lambda s=RANDOM_SEED: make_xgb(len(TEXT_TOP15), s)),
    'text_top20_xgb':  (mk_X(TEXT_TOP20), lambda s=RANDOM_SEED: make_xgb(len(TEXT_TOP20), s)),
    'text_stylo_xgb':  (mk_X(TEXT_STYLO), lambda s=RANDOM_SEED: make_xgb(len(TEXT_STYLO), s)),
    'text_all_xgb':    (mk_X(TEXT_ALL),   lambda s=RANDOM_SEED: make_xgb(len(TEXT_ALL),   s)),
}
# Per-layer audio bases (RF + XGB)
for L in WAVLM_LAYERS:
    cols = WAVLM_LAYER_COLS[L]
    BASE_REGISTRY[f'wavlm_L{L}_rf']  = (mk_X(cols), lambda s=RANDOM_SEED, n=len(cols): make_rf (n, s))
    BASE_REGISTRY[f'wavlm_L{L}_xgb'] = (mk_X(cols), lambda s=RANDOM_SEED, n=len(cols): make_xgb(n, s))
for L in WHISPER_LAYERS:
    cols = WHISPER_LAYER_COLS[L]
    BASE_REGISTRY[f'whisper_L{L}_rf']  = (mk_X(cols), lambda s=RANDOM_SEED, n=len(cols): make_rf (n, s))
    BASE_REGISTRY[f'whisper_L{L}_xgb'] = (mk_X(cols), lambda s=RANDOM_SEED, n=len(cols): make_xgb(n, s))

ALIAS = {}
for L in WAVLM_LAYERS:
    ALIAS[f'wavlm_L{L}_rf']  = f'wv{L}.rf'
    ALIAS[f'wavlm_L{L}_xgb'] = f'wv{L}.xg'
for L in WHISPER_LAYERS:
    ALIAS[f'whisper_L{L}_rf']  = f'wh{L}.rf'
    ALIAS[f'whisper_L{L}_xgb'] = f'wh{L}.xg'
ALIAS.update({
    'text_top10_xgb':'t10','text_top15_xgb':'t15','text_top20_xgb':'t20',
    'text_stylo_xgb':'tst','text_all_xgb':'tal',
})
def short(name): return ALIAS.get(name, name)

print(f'BASE_REGISTRY: {len(BASE_REGISTRY)} bases')
for k in BASE_REGISTRY: print(f'  {k:25s} -> {short(k)}')""")

# ============================================================
md("""## 4. Baseline pipeline (a2 + a4 train -> a5 test)

For each base:
1. **OOF** via 5-fold candidate-isolated GroupKFold on the labelled pool.
2. **Isotonic calibration** on the OOF probabilities.
3. **Caruana** picks weights on the *calibrated* OOFs, F1-max.
4. **Refit on full pool**, score audios5, apply per-base isotonic, fuse with Caruana weights, evaluate at the OOF-derived threshold.""")

# ============================================================
code("""# === 4.1 OOF generation on labelled pool ===
def make_pool(*batch_names, extra_df=None):
    parts = [batches[b] for b in batch_names]
    if extra_df is not None and len(extra_df):
        parts.append(extra_df)
    df = pd.concat(parts, ignore_index=True)
    df = df[df['label_int'].isin([0,1])].reset_index(drop=True)
    return df

def oof_for_pool(df_pool, n_folds=N_FOLDS, seed=RANDOM_SEED, sample_weight_col=None):
    \"\"\"Per-base OOF on df_pool. Returns dict[name] = oof_probas.\"\"\"
    df = df_pool.reset_index(drop=True)
    y  = df['label_int'].values
    cands = df['candidate_id'].values
    sw = df[sample_weight_col].values if sample_weight_col and sample_weight_col in df.columns else None
    oof = {n: np.full(len(df), np.nan) for n in BASE_REGISTRY}
    gkf = GroupKFold(n_splits=n_folds)
    fold_log = []
    for fi, (tr_idx, va_idx) in enumerate(gkf.split(df, y, groups=cands)):
        df_tr = df.iloc[tr_idx]; df_va = df.iloc[va_idx]
        sw_tr = sw[tr_idx] if sw is not None else None
        for n, (X_fn, factory) in BASE_REGISTRY.items():
            sc  = StandardScaler().fit(X_fn(df_tr))
            clf = factory(seed + fi)
            try:
                if sw_tr is not None:
                    clf.fit(sc.transform(X_fn(df_tr)), df_tr['label_int'].values, sample_weight=sw_tr)
                else:
                    clf.fit(sc.transform(X_fn(df_tr)), df_tr['label_int'].values)
            except TypeError:
                clf.fit(sc.transform(X_fn(df_tr)), df_tr['label_int'].values)
            oof[n][va_idx] = clf.predict_proba(sc.transform(X_fn(df_va)))[:, 1]
        fold_log.append(fi)
        print(f'    fold {fi+1}/{n_folds} done  (train={len(df_tr)}, val={len(df_va)})')
    return oof, y

baseline_pool = make_pool(*BATCH_LABELED)
print(f'Baseline pool: {BATCH_LABELED}  n={len(baseline_pool)}  '
      f'cheat={int((baseline_pool[\"label_int\"]==1).sum())}  '
      f'cands={baseline_pool[\"candidate_id\"].nunique()}\\n')

t0 = time.time()
print('OOF generation on baseline pool')
oof_base, y_base = oof_for_pool(baseline_pool)
print(f'OOF done in {time.time()-t0:.1f}s')

# Per-base raw OOF F1
print('\\nPer-base raw OOF F1:')
raw_oof_f1 = {}
for n in BASE_REGISTRY:
    thr, f1 = best_f1_thr(oof_base[n], y_base)
    raw_oof_f1[n] = (thr, f1)
    print(f'  {short(n):10s}  thr={thr:.2f}  f1={f1:.4f}')""")

# ============================================================
code("""# === 4.2 Isotonic calibration (per base) on OOF ===
def fit_isotonics(oof_dict, y):
    iso = {}
    for n, p in oof_dict.items():
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(p, y)
        iso[n] = ir
    return iso

def apply_isotonics(iso, proba_dict):
    return {n: iso[n].predict(proba_dict[n]) for n in proba_dict if n in iso}

iso_base = fit_isotonics(oof_base, y_base)
oof_cal_base = apply_isotonics(iso_base, oof_base)

print('Per-base CALIBRATED OOF F1:')
cal_oof_f1 = {}
for n in BASE_REGISTRY:
    thr, f1 = best_f1_thr(oof_cal_base[n], y_base)
    cal_oof_f1[n] = (thr, f1)
    raw_thr, raw_f1 = raw_oof_f1[n]
    print(f'  {short(n):10s}  raw f1={raw_f1:.4f}  cal f1={f1:.4f}  delta={f1-raw_f1:+.4f}')""")

# ============================================================
code("""# === 4.3 Caruana ensemble selection on CALIBRATED OOFs ===
weights_base, picks_base, ens_oof_f1, ens_oof_thr, hist_base = caruana_select(
    oof_cal_base, y_base, max_iters=CARUANA_MAX_ITERS)

print(f'Caruana selected {len(picks_base)} picks (with replacement)  '
      f'OOF F1={ens_oof_f1:.4f}  thr={ens_oof_thr:.2f}')
print('\\nFinal weights (count / total picks):')
for n, w in sorted(weights_base.items(), key=lambda kv: -kv[1]):
    print(f'  {short(n):10s}  w={w:.3f}  ({n})')

print('\\nSelection trajectory:')
for h in hist_base:
    print(f'  iter {h[\"iter\"]:2d}  +{short(h[\"pick\"]):10s}  '
          f'f1={h[\"f1\"]:.4f}  thr={h[\"thr\"]:.2f}  n_selected={h[\"n_selected\"]}')""")

# ============================================================
code("""# === 4.4 Refit on full pool, score a5, evaluate ===
def refit_score(df_train, df_score, sample_weight_col=None):
    \"\"\"Fit each base on df_train, score df_score. Returns dict[name] = probas.\"\"\"
    out = {}
    sw = df_train[sample_weight_col].values if sample_weight_col and sample_weight_col in df_train.columns else None
    for n, (X_fn, factory) in BASE_REGISTRY.items():
        sc  = StandardScaler().fit(X_fn(df_train))
        clf = factory(RANDOM_SEED)
        try:
            if sw is not None:
                clf.fit(sc.transform(X_fn(df_train)), df_train['label_int'].values, sample_weight=sw)
            else:
                clf.fit(sc.transform(X_fn(df_train)), df_train['label_int'].values)
        except TypeError:
            clf.fit(sc.transform(X_fn(df_train)), df_train['label_int'].values)
        out[n] = clf.predict_proba(sc.transform(X_fn(df_score)))[:, 1]
    return out

df_test = batches[BATCH_TEST]
y_test  = df_test['label_int'].values
print(f'Test = {BATCH_TEST}: n={len(df_test)}  cheat={int((y_test==1).sum())}\\n')

raw_test_base = refit_score(baseline_pool, df_test)
cal_test_base = apply_isotonics(iso_base, raw_test_base)
fused_test_base = fused_proba(weights_base, cal_test_base)

m_test_base_at_oof = _metrics_at(fused_test_base, y_test, ens_oof_thr)
opt_thr, opt_f1    = best_f1_thr(fused_test_base, y_test)
m_test_base_opt    = _metrics_at(fused_test_base, y_test, opt_thr)

print('=== BASELINE result on audios5 ===')
print(f'  OOF-derived thr={ens_oof_thr:.2f}  ->  '
      f'F1={m_test_base_at_oof[\"f1\"]:.4f}  P={m_test_base_at_oof[\"prec\"]:.4f}  '
      f'R={m_test_base_at_oof[\"rec\"]:.4f}  TP/FP/FN={m_test_base_at_oof[\"tp\"]}/'
      f'{m_test_base_at_oof[\"fp\"]}/{m_test_base_at_oof[\"fn\"]}')
print(f'  audios5-optimal thr={opt_thr:.2f}  -> F1={opt_f1:.4f}  '
      f'(use as model-fault vs threshold-fault diagnostic only)')

# Save baseline artifacts
import pickle
with open(SAVE_DIR / 'baseline_artifacts.pkl', 'wb') as fh:
    pickle.dump({
        'weights': weights_base, 'iso': iso_base, 'thr': ens_oof_thr,
        'oof_f1': ens_oof_f1, 'cal_oof_f1': cal_oof_f1, 'raw_oof_f1': raw_oof_f1,
        'picks': picks_base, 'history': hist_base,
    }, fh)
print(f'\\nSaved: {SAVE_DIR/\"baseline_artifacts.pkl\"}')""")

# ============================================================
md("""## 5. SSL round on audios3

Score `audios3` with each base (already trained on a2+a4), calibrate via the baseline isotonic models, compute T-similarity, select pseudo-labels with high agreement and extreme calibrated-fused score.

Then:
1. Add the pseudo-labelled rows to the train pool (with `pseudo_weight`).
2. Re-do OOF + calibration + Caruana on `a2 + a4 + a3-pseudo`.
3. Refit + score audios5.""")

# ============================================================
code("""# === 5.1 Score audios3 + T-similarity + pseudo selection ===
df_unl = batches[BATCH_UNLABELED]
print(f'Unlabelled pool {BATCH_UNLABELED}: n={len(df_unl)}  '
      f'cands={df_unl[\"candidate_id\"].nunique()}')

# Score with baseline-trained models
raw_unl_base = refit_score(baseline_pool, df_unl)
cal_unl_base = apply_isotonics(iso_base, raw_unl_base)
fused_unl    = fused_proba(weights_base, cal_unl_base)

# T-similarity: variance of CALIBRATED probas across bases (only those with non-zero weight)
active_bases = [n for n, w in weights_base.items() if w > 0]
M = np.stack([cal_unl_base[n] for n in active_bases], axis=0)  # [B, N]
var_per_row = np.var(M, axis=0)
t_sim = np.clip(1.0 - 4.0 * var_per_row, 0.0, 1.0)

print(f'\\nT-sim distribution: mean={t_sim.mean():.3f}  median={np.median(t_sim):.3f}  '
      f'q25={np.percentile(t_sim,25):.3f}  q75={np.percentile(t_sim,75):.3f}')
print(f'Fused score distribution: mean={fused_unl.mean():.3f}  '
      f'median={np.median(fused_unl):.3f}  q25={np.percentile(fused_unl,25):.3f}  '
      f'q75={np.percentile(fused_unl,75):.3f}')

# Pseudo selection
pos_mask = (fused_unl >= POS_THRESH) & (t_sim >= T_SIM_MIN)
neg_mask = (fused_unl <= NEG_THRESH) & (t_sim >= T_SIM_MIN)

# Cap per class — keep the most confident
def cap_mask(mask, score, max_n):
    n = int(mask.sum())
    if n <= max_n: return mask
    idx_in_mask = np.flatnonzero(mask)
    top = idx_in_mask[np.argsort(-score[mask])[:max_n]]
    out = np.zeros_like(mask)
    out[top] = True
    return out

pos_mask = cap_mask(pos_mask, fused_unl * t_sim,         MAX_PSEUDO_PER_CLASS)
neg_mask = cap_mask(neg_mask, (1 - fused_unl) * t_sim,   MAX_PSEUDO_PER_CLASS)

print(f'\\nPseudo selected:  positives={int(pos_mask.sum())}  negatives={int(neg_mask.sum())}')
print(f'  positive  proba range: '
      f'{fused_unl[pos_mask].min() if pos_mask.any() else 0:.3f}'
      f' .. {fused_unl[pos_mask].max() if pos_mask.any() else 0:.3f}')
print(f'  negative  proba range: '
      f'{fused_unl[neg_mask].min() if neg_mask.any() else 0:.3f}'
      f' .. {fused_unl[neg_mask].max() if neg_mask.any() else 0:.3f}')

# Build pseudo-labelled dataframe
df_unl_aug = df_unl.copy()
df_unl_aug['fused_score'] = fused_unl
df_unl_aug['t_sim']       = t_sim
df_unl_aug['pseudo_label'] = -1
df_unl_aug.loc[pos_mask, 'pseudo_label'] = 1
df_unl_aug.loc[neg_mask, 'pseudo_label'] = 0
df_unl_aug.to_csv(SAVE_DIR / 'audios3_pseudo_scores.csv', index=False)

df_pseudo = df_unl_aug[df_unl_aug['pseudo_label'] != -1].copy()
df_pseudo['label_int']     = df_pseudo['pseudo_label'].astype(int)
df_pseudo['sample_weight'] = float(SSL_PSEUDO_WEIGHT)
print(f'\\nPseudo-labelled pool: n={len(df_pseudo)}  '
      f'(pos={int((df_pseudo[\"label_int\"]==1).sum())}, '
      f'neg={int((df_pseudo[\"label_int\"]==0).sum())})')""")

# ============================================================
code("""# === 5.2 SSL retrain on a2 + a4 + pseudo-labelled a3 ===
if len(df_pseudo) == 0:
    print('No pseudo-labels selected — SSL round skipped. Loosen T_SIM_MIN / POS_THRESH / NEG_THRESH if you want pseudo-labels.')
    ssl_done = False
else:
    ssl_done = True
    base_with_w = baseline_pool.copy()
    base_with_w['sample_weight'] = 1.0
    ssl_pool = pd.concat([base_with_w, df_pseudo[base_with_w.columns]], ignore_index=True)
    print(f'SSL pool: n={len(ssl_pool)}  (real={len(base_with_w)}, pseudo={len(df_pseudo)})')

    t0 = time.time()
    print('\\nOOF on SSL pool')
    oof_ssl, y_ssl = oof_for_pool(ssl_pool, sample_weight_col='sample_weight')
    print(f'OOF done in {time.time()-t0:.1f}s')

    iso_ssl     = fit_isotonics(oof_ssl, y_ssl)
    oof_cal_ssl = apply_isotonics(iso_ssl, oof_ssl)

    print('\\nPer-base SSL CALIBRATED OOF F1:')
    cal_oof_f1_ssl = {}
    for n in BASE_REGISTRY:
        thr, f1 = best_f1_thr(oof_cal_ssl[n], y_ssl)
        cal_oof_f1_ssl[n] = (thr, f1)
        baseline_thr, baseline_f1 = cal_oof_f1[n]
        print(f'  {short(n):10s}  baseline cal f1={baseline_f1:.4f}  '
              f'ssl cal f1={f1:.4f}  delta={f1-baseline_f1:+.4f}')

    weights_ssl, picks_ssl, ens_oof_f1_ssl, ens_oof_thr_ssl, hist_ssl = caruana_select(
        oof_cal_ssl, y_ssl, max_iters=CARUANA_MAX_ITERS)
    print(f'\\nCaruana on SSL pool: {len(picks_ssl)} picks  '
          f'OOF F1={ens_oof_f1_ssl:.4f}  thr={ens_oof_thr_ssl:.2f}')
    print('SSL final weights:')
    for n, w in sorted(weights_ssl.items(), key=lambda kv: -kv[1]):
        print(f'  {short(n):10s}  w={w:.3f}  ({n})')""")

# ============================================================
code("""# === 5.3 SSL refit + a5 eval ===
if ssl_done:
    raw_test_ssl   = refit_score(ssl_pool, df_test, sample_weight_col='sample_weight')
    cal_test_ssl   = apply_isotonics(iso_ssl, raw_test_ssl)
    fused_test_ssl = fused_proba(weights_ssl, cal_test_ssl)

    m_test_ssl_at_oof = _metrics_at(fused_test_ssl, y_test, ens_oof_thr_ssl)
    opt_thr_ssl, opt_f1_ssl = best_f1_thr(fused_test_ssl, y_test)
    m_test_ssl_opt    = _metrics_at(fused_test_ssl, y_test, opt_thr_ssl)

    print('=== SSL result on audios5 ===')
    print(f'  OOF-derived thr={ens_oof_thr_ssl:.2f}  ->  '
          f'F1={m_test_ssl_at_oof[\"f1\"]:.4f}  P={m_test_ssl_at_oof[\"prec\"]:.4f}  '
          f'R={m_test_ssl_at_oof[\"rec\"]:.4f}  TP/FP/FN={m_test_ssl_at_oof[\"tp\"]}/'
          f'{m_test_ssl_at_oof[\"fp\"]}/{m_test_ssl_at_oof[\"fn\"]}')
    print(f'  audios5-optimal thr={opt_thr_ssl:.2f}  -> F1={opt_f1_ssl:.4f}')

    with open(SAVE_DIR / 'ssl_artifacts.pkl', 'wb') as fh:
        pickle.dump({
            'weights': weights_ssl, 'iso': iso_ssl, 'thr': ens_oof_thr_ssl,
            'oof_f1': ens_oof_f1_ssl, 'cal_oof_f1': cal_oof_f1_ssl,
            'picks': picks_ssl, 'history': hist_ssl,
            'pseudo_n': len(df_pseudo),
            'pseudo_pos_n': int((df_pseudo['label_int']==1).sum()),
            'pseudo_neg_n': int((df_pseudo['label_int']==0).sum()),
        }, fh)
    print(f'\\nSaved: {SAVE_DIR/\"ssl_artifacts.pkl\"}')
else:
    print('SSL round did not run; skipping SSL eval.')""")

# ============================================================
md("""## 6. Comparison""")

# ============================================================
code("""# === 6. Side-by-side comparison + per-base CV deltas ===
rows = []
rows.append({
    'config':       'baseline (a2+a4)',
    'pool_n':       len(baseline_pool),
    'pseudo_pos':   0, 'pseudo_neg': 0,
    'oof_f1':       round(ens_oof_f1, 4),
    'oof_thr':      round(ens_oof_thr, 2),
    'a5_f1_at_oof': round(m_test_base_at_oof['f1'], 4),
    'a5_prec':      round(m_test_base_at_oof['prec'], 4),
    'a5_rec':       round(m_test_base_at_oof['rec'], 4),
    'a5_tp':        m_test_base_at_oof['tp'], 'a5_fp': m_test_base_at_oof['fp'],
    'a5_fn':        m_test_base_at_oof['fn'], 'a5_tn': m_test_base_at_oof['tn'],
    'a5_optimal_thr': round(opt_thr, 2),
    'a5_optimal_f1':  round(opt_f1, 4),
    'n_picks':      len(picks_base),
})
if ssl_done:
    rows.append({
        'config':       f'ssl (a2+a4+a3p)',
        'pool_n':       len(ssl_pool),
        'pseudo_pos':   int((df_pseudo['label_int']==1).sum()),
        'pseudo_neg':   int((df_pseudo['label_int']==0).sum()),
        'oof_f1':       round(ens_oof_f1_ssl, 4),
        'oof_thr':      round(ens_oof_thr_ssl, 2),
        'a5_f1_at_oof': round(m_test_ssl_at_oof['f1'], 4),
        'a5_prec':      round(m_test_ssl_at_oof['prec'], 4),
        'a5_rec':       round(m_test_ssl_at_oof['rec'], 4),
        'a5_tp':        m_test_ssl_at_oof['tp'], 'a5_fp': m_test_ssl_at_oof['fp'],
        'a5_fn':        m_test_ssl_at_oof['fn'], 'a5_tn': m_test_ssl_at_oof['tn'],
        'a5_optimal_thr': round(opt_thr_ssl, 2),
        'a5_optimal_f1':  round(opt_f1_ssl, 4),
        'n_picks':      len(picks_ssl),
    })
compare_df = pd.DataFrame(rows)
compare_df.to_csv(SAVE_DIR / 'comparison.csv', index=False)
with pd.option_context('display.max_columns', None, 'display.width', 220, 'display.max_colwidth', 30):
    print(compare_df.to_string(index=False))

# Per-base before/after CV F1
if ssl_done:
    print('\\nPer-base CALIBRATED OOF F1: baseline vs SSL')
    pb_rows = []
    for n in BASE_REGISTRY:
        b_thr, b_f1 = cal_oof_f1[n]
        s_thr, s_f1 = cal_oof_f1_ssl[n]
        pb_rows.append({
            'base':           n,
            'short':          short(n),
            'baseline_f1':    round(b_f1, 4),
            'ssl_f1':         round(s_f1, 4),
            'delta':          round(s_f1 - b_f1, 4),
            'baseline_in_caruana': round(weights_base.get(n, 0), 3),
            'ssl_in_caruana':      round(weights_ssl.get(n, 0), 3),
        })
    pb_df = pd.DataFrame(pb_rows).sort_values('delta', ascending=False)
    pb_df.to_csv(SAVE_DIR / 'per_base_compare.csv', index=False)
    with pd.option_context('display.max_columns', None, 'display.width', 220):
        print(pb_df.to_string(index=False))
print(f'\\nSaved: {SAVE_DIR/\"comparison.csv\"}, {SAVE_DIR/\"per_base_compare.csv\"}')""")

# ============================================================
md("""## 7. How to read these results

**`comparison.csv` rows:**
- `oof_f1` = Caruana F1 on the calibrated OOF (i.e. how the model expects to do at deploy threshold).
- `a5_f1_at_oof` = audios5 F1 at the OOF-derived threshold. The legitimate, no-leak number.
- `a5_optimal_f1` = audios5 F1 if you cherry-picked the threshold on audios5. Diagnostic only — gap from `a5_f1_at_oof` tells you threshold drift.

**Patterns:**
- `ssl_a5_f1 > baseline_a5_f1` *and* `ssl_oof_f1 > baseline_oof_f1` → SSL helped. Real win.
- `ssl_a5_f1 ≈ baseline_a5_f1` but `ssl_oof_f1 > baseline_oof_f1` → SSL boosted CV but didn't transfer; pseudo-labels reinforced existing decision boundary without expanding it. Common with very confident pseudo-labels.
- `ssl_a5_f1 < baseline_a5_f1` → pseudo-labels are biasing the model. Tighten thresholds (raise `T_SIM_MIN`, narrow `[NEG_THRESH, POS_THRESH]`) or reduce `SSL_PSEUDO_WEIGHT`.

**`per_base_compare.csv`:**
- Look at which bases gained or lost CV F1 with the pseudo-labels. Bases that *lost* F1 are the ones that disagreed with the pseudo-labels — that's a signal those bases were picking up something the others weren't.
- The `*_in_caruana` columns show whether the base was actually used in the fusion. Bases with weight 0 are ignored.

**Layer-sweep readout:**
- Compare `cal_oof_f1` of `wavlm_L6_xgb` / `wavlm_L9_xgb` / `wavlm_L12_xgb`. If a middle layer beats `L12` (the whole-pool baseline), the layer-sweep paid off. Same for Whisper.
- The `weights` columns from Caruana tell you which layers Caruana actually picked. If layer 9 dominates layer 12 in the weights, that's strong evidence.

**Knobs to retune:**
- `T_SIM_MIN`, `POS_THRESH`, `NEG_THRESH`, `MAX_PSEUDO_PER_CLASS` for SSL aggressiveness.
- `SSL_PSEUDO_WEIGHT` to down-weight pseudo-labels if they hurt.
- `WAVLM_LAYERS` / `WHISPER_LAYERS` to widen the layer search (extraction is incremental).""")

# ============================================================
nb = {
    'cells': CELLS,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {'name': 'python', 'version': '3.10'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

out = Path(__file__).resolve().parent / 'ssl_layer_fusion.ipynb'
out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Wrote: {out}  ({len(CELLS)} cells)')
