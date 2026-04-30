"""Build audios6_eval.ipynb. Run once: python _build_audios6_eval.py"""
import json
from pathlib import Path

CELLS = []

def md(text):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) if text else []
    })

def code(text):
    CELLS.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True) if text else []
    })

# ============================================================
md("""# audios6 evaluation — rotation A top-3 + threshold diagnostics

Goal: take the top-3 fusion picks from `fusions_cv.ipynb` rotation A (ranked by averaged `te_f1`) and re-evaluate them on the new audios6 batch, in two training-pool modes:

- **Mode A (no a5 in train)** — refit base models on `audios2 + audios4`. This preserves the rotation-A training data, so the frozen threshold from `summary_avg.csv` is directly applicable.
- **Mode B (with a5 in train)** — refit on `audios2 + audios4 + audios5`. More training data, but the model has changed, so the frozen threshold is only a reference and a fresh CV-derived threshold is reported.

For each (pick × mode), three thresholds are reported on audios6:
1. `frozen_thr` — from rotation-A `summary_avg.csv`.
2. `cv_thr` — F1-max threshold on a fresh 5-fold candidate-isolated OOF over the actual training pool (so it reflects what we'd ship without peeking at audios6).
3. `optimal_thr_a6` — F1-max threshold sweep on audios6 GT itself. **Used only as a diagnostic** — gap between this and the other two tells you whether the model is healthy and only the threshold drifted, vs the model itself degraded.

Section 2 (optional, off by default) re-runs the fusion search with audios5 in train to see if the rankings change.

Outputs land in `checkpoints_audios6_eval/`.

Per memory rule: predictions stay per-audio. No aggregation across Q25/Q26/Q27.""")

# ============================================================
code("""# === 0. CONFIG ===
from pathlib import Path

BATCH_NEW       = 'audios6'
TRAIN_BATCHES_A = ['audios2', 'audios4']            # Mode A — preserves rotation A
TRAIN_BATCHES_B = ['audios2', 'audios4', 'audios5'] # Mode B — adds a5

PICK_TOP_N      = 3
PICK_RANK_BY    = 'te_f1'   # column in summary_avg.csv to rank by
PICK_ROTATION   = 'A'
PICK_STRATEGY   = 'F1'      # only F1 strategy is in scope here

MIN_SPEAKING_S  = 30
N_FOLDS         = 5
RANDOM_SEED     = 42

DEPLOY_POS_RATE = 0.17
SPW_DEPLOY      = (1.0 - DEPLOY_POS_RATE) / DEPLOY_POS_RATE

NB_DIR        = Path('.').resolve()
SAVE_DIR      = NB_DIR / 'checkpoints_audios6_eval'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DURATIONS_DIR = NB_DIR / 'checkpoints_honest_eval'
DURATIONS_DIR.mkdir(parents=True, exist_ok=True)
FUSIONS_DIR   = NB_DIR / 'checkpoints_fusions'

# Section 2 (experimental): re-search top fusions with a5 in train.
# Off by default — flip to True only when you want to spend the extra runtime.
RUN_EXPERIMENTAL_RESEARCH = False

# Extraction toggles — flip to False if a step's outputs are already in place.
EXTRACT_TRANSCRIPTS  = True
EXTRACT_TEXT_FEATS   = True
EXTRACT_WAVLM        = True
EXTRACT_WHISPER_ENC  = True
EXTRACT_DURATIONS    = True

WHISPER_TRANSCRIBE_MODEL = 'small'
WAVLM_MODEL              = 'microsoft/wavlm-base-plus'
WHISPER_ENC_MODEL        = 'openai/whisper-medium'
WHISPER_CHUNK_SEC        = 30
MAX_DURATION_SEC         = 120
SR                       = 16000
AUDIO_EXTS               = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.webm', '.mp4'}

print(f'NB_DIR        = {NB_DIR}')
print(f'SAVE_DIR      = {SAVE_DIR}')
print(f'FUSIONS_DIR   = {FUSIONS_DIR}  (must contain summary_avg.csv from the company-laptop run)')
print(f'BATCH_NEW     = {BATCH_NEW}')
print(f'Mode A train  = {TRAIN_BATCHES_A}')
print(f'Mode B train  = {TRAIN_BATCHES_B}')
print(f'PICK_TOP_N={PICK_TOP_N}  rank_by={PICK_RANK_BY}  rotation={PICK_ROTATION}  strategy={PICK_STRATEGY}')
print(f'MIN_SPEAKING_S={MIN_SPEAKING_S}  N_FOLDS={N_FOLDS}  SPW_DEPLOY={SPW_DEPLOY:.2f}')
print(f'RUN_EXPERIMENTAL_RESEARCH={RUN_EXPERIMENTAL_RESEARCH}')""")

# ============================================================
code("""# === 1. Imports + label map + small helpers ===
import re, json, warnings, time, itertools, gc
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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

print('Imports + helpers ready.')""")

# ============================================================
md("""## 2. Extraction for audios6

Extract the same three artifacts the rest of the pipeline expects:
- `{batch}_transcripts.json` + `{batch}_features.csv` — text/disfluency/pause/prosodic features
- `{batch}_wavlm_whole.csv` — 768-dim mean-pooled WavLM-base-plus
- `{batch}_whisper_whole.csv` — 1024-dim mean-pooled Whisper-medium encoder
- `checkpoints_honest_eval/{batch}_durations.csv` — speaking-time, used for the ≥30s filter

Each step is **resume-safe** and **skipped if the cache already exists** (or if the corresponding `EXTRACT_*` flag is False). You still need `audios6GT.csv` next to the notebook with `filename,label` columns.""")

# ============================================================
code("""# === 2.0 Audio scan for audios6 ===
audio_root = NB_DIR / BATCH_NEW
gt_path    = NB_DIR / f'{BATCH_NEW}GT.csv'
trans_path = NB_DIR / f'{BATCH_NEW}_transcripts.json'
feat_path  = NB_DIR / f'{BATCH_NEW}_features.csv'
wavlm_path = NB_DIR / f'{BATCH_NEW}_wavlm_whole.csv'
wh_path    = NB_DIR / f'{BATCH_NEW}_whisper_whole.csv'
dur_path   = DURATIONS_DIR / f'{BATCH_NEW}_durations.csv'

assert audio_root.exists(), f'Missing audio folder: {audio_root}'
assert gt_path.exists(),    f'Missing GT file: {gt_path}'

audio_files = sorted(f for f in audio_root.rglob('*')
                     if f.suffix.lower() in AUDIO_EXTS and f.is_file())
print(f'audios6 audio files: {len(audio_files)}  (root={audio_root})')
print(f'  GT          : {gt_path.exists()}  -> {gt_path.name}')
print(f'  transcripts : {trans_path.exists()}  -> {trans_path.name}')
print(f'  text feats  : {feat_path.exists()}  -> {feat_path.name}')
print(f'  wavlm whole : {wavlm_path.exists()}  -> {wavlm_path.name}')
print(f'  whisper whl : {wh_path.exists()}  -> {wh_path.name}')
print(f'  durations   : {dur_path.exists()}  -> {dur_path.name}')""")

# ============================================================
code("""# === 2.1 Whisper transcription (cached, resume-safe) ===
import soundfile as sf
import librosa
from tqdm import tqdm

FILLER_PROMPT = ('Umm, let me think like, hmm... Okay here\\'s what I\\'m thinking. '
                 'So uh, basically, you know, I mean, like, right.')

def transcribe_audios6():
    if not EXTRACT_TRANSCRIPTS:
        print('  EXTRACT_TRANSCRIPTS=False — skipping.'); return
    existing = {}
    if trans_path.exists():
        existing = json.load(open(trans_path, encoding='utf-8'))
    todo = [f for f in audio_files if f.name not in existing]
    if not todo:
        print(f'  audios6: {len(existing)} transcripts cached, nothing to do.'); return
    from faster_whisper import WhisperModel
    import torch as _torch
    device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    compute_type = 'float16' if device == 'cuda' else 'int8'
    print(f'  Loading Whisper {WHISPER_TRANSCRIBE_MODEL} ({device}, {compute_type})')
    whisper = WhisperModel(WHISPER_TRANSCRIBE_MODEL, device=device, compute_type=compute_type)
    print(f'  audios6: transcribing {len(todo)} new files')
    for fp in tqdm(todo, desc='audios6'):
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
        with open(trans_path, 'w', encoding='utf-8') as fh:
            json.dump(existing, fh, ensure_ascii=False, indent=1)
    print(f'  audios6: total {len(existing)} transcripts cached.')

transcribe_audios6()""")

# ============================================================
code("""# === 2.2 Text feature extraction (cached, resume-safe) ===
# Mirrors text_cheating_detection.ipynb. Same ALL_FEATURES schema.
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

def extract_text_features_audios6():
    if not EXTRACT_TEXT_FEATS:
        print('  EXTRACT_TEXT_FEATS=False — skipping.'); return
    if not trans_path.exists():
        print(f'  Missing {trans_path.name} — run the transcription cell first.'); return
    t = json.load(open(trans_path, encoding='utf-8'))
    expected = set(ALL_FEATURES)
    have_files = set()
    df_cached = None
    if feat_path.exists():
        df_cached = pd.read_csv(feat_path)
        have_files = set(df_cached['filename'].tolist())
        missing_cols = expected - set(df_cached.columns)
        if missing_cols:
            print(f'  Patching {len(missing_cols)} missing feature columns in {feat_path.name}')
            patches = []
            for _, r in tqdm(df_cached.iterrows(), total=len(df_cached), desc='audios6 patch'):
                fp = next((f for f in audio_files if f.name == r['filename']), None)
                tr = t.get(r['filename'], {'text':'','words':[],'duration_sec':0})
                full = compute_all_features(fp, tr.get('text',''), tr.get('words',[])) if fp \\
                       else {c: 0 for c in missing_cols}
                patches.append({c: full.get(c, 0) for c in missing_cols})
            df_cached = pd.concat([df_cached.reset_index(drop=True),
                                    pd.DataFrame(patches)], axis=1)
            df_cached.to_csv(feat_path, index=False)
    todo = [fp for fp in audio_files if fp.name not in have_files]
    if not todo:
        print(f'  audios6: text features all extracted ({len(have_files)} files).'); return
    print(f'  audios6: extracting text features for {len(todo)} files')
    rows = []
    for fp in tqdm(todo, desc='audios6 feat'):
        tr = t.get(fp.name, {'text':'','words':[],'duration_sec':0})
        r = compute_all_features(fp, tr.get('text',''), tr.get('words',[]))
        r['filename']     = fp.name
        r['duration_sec'] = tr.get('duration_sec', 0)
        rows.append(r)
        if len(rows) >= 10:
            pd.DataFrame(rows).to_csv(feat_path, mode='a',
                                       header=not feat_path.exists(), index=False)
            rows = []
    if rows:
        pd.DataFrame(rows).to_csv(feat_path, mode='a',
                                   header=not feat_path.exists(), index=False)
    print(f'  audios6: text features cached -> {feat_path}')

extract_text_features_audios6()""")

# ============================================================
code("""# === 2.3 WavLM whole-pool extraction (cached) ===
def extract_wavlm_audios6():
    if not EXTRACT_WAVLM:
        print('  EXTRACT_WAVLM=False — skipping.'); return
    if wavlm_path.exists():
        print(f'  audios6: {wavlm_path.name} cached.'); return
    import torch as _torch
    from transformers import AutoFeatureExtractor, WavLMModel
    device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    print(f'  Loading {WAVLM_MODEL} ({device})')
    fe  = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    mdl = WavLMModel.from_pretrained(WAVLM_MODEL).eval().to(device)
    rows = []
    with _torch.no_grad():
        for fp in tqdm(audio_files, desc='audios6 wavlm'):
            try:
                y, sr = sf.read(str(fp), always_2d=False)
                if y.ndim > 1: y = y.mean(axis=1)
                if sr != SR:  y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
                y = y.astype(np.float32)
                if MAX_DURATION_SEC and len(y) > MAX_DURATION_SEC * SR:
                    y = y[:int(MAX_DURATION_SEC * SR)]
                inp = fe(y, sampling_rate=SR, return_tensors='pt', padding=False)
                out = mdl(inp.input_values.to(device))
                emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                row = {'filename': fp.name}
                for i, v in enumerate(emb): row[f'wavlm_{i}'] = round(float(v), 6)
                rows.append(row)
            except Exception as e:
                print(f'  WARN wavlm {fp.name}: {e}')
    pd.DataFrame(rows).to_csv(wavlm_path, index=False)
    print(f'  audios6: wavlm cached -> {wavlm_path} ({len(rows)} rows)')
    del mdl, fe
    if device == 'cuda': _torch.cuda.empty_cache()
    gc.collect()

extract_wavlm_audios6()""")

# ============================================================
code("""# === 2.4 Whisper-medium encoder whole-pool extraction (cached) ===
def extract_whisper_enc_audios6():
    if not EXTRACT_WHISPER_ENC:
        print('  EXTRACT_WHISPER_ENC=False — skipping.'); return
    if wh_path.exists():
        print(f'  audios6: {wh_path.name} cached.'); return
    import torch as _torch
    from transformers import WhisperProcessor, WhisperModel
    device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    print(f'  Loading {WHISPER_ENC_MODEL} ({device})')
    proc = WhisperProcessor.from_pretrained(WHISPER_ENC_MODEL)
    mdl  = WhisperModel.from_pretrained(WHISPER_ENC_MODEL).eval().to(device)
    chunk_samples = WHISPER_CHUNK_SEC * SR
    rows = []
    with _torch.no_grad():
        for fp in tqdm(audio_files, desc='audios6 whisper-enc'):
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
                embs = []
                for c in chunks:
                    if len(c) < int(SR * 0.5): continue
                    feat = proc(c, sampling_rate=SR, return_tensors='pt').input_features.to(device)
                    out  = mdl.encoder(feat)
                    embs.append(out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy())
                if not embs: continue
                emb = np.mean(np.stack(embs), axis=0)
                row = {'filename': fp.name}
                for i, v in enumerate(emb): row[f'whisper_{i}'] = round(float(v), 6)
                rows.append(row)
            except Exception as e:
                print(f'  WARN whisper-enc {fp.name}: {e}')
    pd.DataFrame(rows).to_csv(wh_path, index=False)
    print(f'  audios6: whisper-enc cached -> {wh_path} ({len(rows)} rows)')
    del mdl, proc
    if device == 'cuda': _torch.cuda.empty_cache()
    gc.collect()

extract_whisper_enc_audios6()""")

# ============================================================
code("""# === 2.5 Durations (cached) ===
def extract_durations_audios6():
    if not EXTRACT_DURATIONS:
        print('  EXTRACT_DURATIONS=False — skipping.'); return
    if dur_path.exists():
        print(f'  audios6: {dur_path.name} cached.'); return
    if not feat_path.exists():
        print(f'  Need {feat_path.name} first (text feats provide n_words/words_per_sec fallback).'); return
    transcripts = json.load(open(trans_path, encoding='utf-8')) if trans_path.exists() else {}
    feats = pd.read_csv(feat_path).set_index('filename')
    rows = []
    for fp in audio_files:
        try:
            info = sf.info(str(fp))
            total = info.frames / info.samplerate
        except Exception:
            total = float('nan')
        sp, src = None, 'none'
        e = transcripts.get(fp.name)
        if e is not None:
            segs = e.get('segments') if isinstance(e, dict) else None
            if segs:
                try:
                    sp = float(sum((s['end'] - s['start']) for s in segs))
                    src = 'transcript'
                except (KeyError, TypeError):
                    pass
        if sp is None and fp.name in feats.index:
            n_w = feats.at[fp.name, 'n_words']      if 'n_words'       in feats.columns else None
            wps = feats.at[fp.name, 'words_per_sec'] if 'words_per_sec' in feats.columns else None
            if pd.notna(n_w) and pd.notna(wps) and wps > 0:
                sp, src = float(n_w) / float(wps), 'derived_words_per_sec'
        rows.append({'filename': fp.name,
                     'total_duration_s': total,
                     'speaking_time_s':  sp,
                     'speech_ratio':     (sp/total) if (sp is not None and total and total>0) else None,
                     'source':           src})
    pd.DataFrame(rows).to_csv(dur_path, index=False)
    print(f'  audios6: durations cached -> {dur_path} ({len(rows)} rows)')

extract_durations_audios6()""")

# ============================================================
md("""## 3. Load all batches + filter by speaking time

Loads `audios2`, `audios4`, `audios5`, `audios6` with the same merge logic as `fusions_cv.ipynb` and applies the ≥30s speaking-time filter so numbers are directly comparable.""")

# ============================================================
code("""# === 3.1 Load + filter ===
WAVLM_WHOLE_CANDIDATES = lambda n: [f'{n}_wavlm_whole.csv', f'{n}_whole_pretrained.csv']

def _first_existing(cands):
    for c in cands:
        p = NB_DIR / c
        if p.exists(): return p
    raise FileNotFoundError(f'None of {cands} exist under {NB_DIR}')

def load_gt(name):
    gt = pd.read_csv(NB_DIR / f'{name}GT.csv')
    fn_col  = next(c for c in gt.columns if c.lower() in ('filename','file','name'))
    lbl_col = next(c for c in gt.columns if c.lower() in ('label','class','cheating','gt','label_int','ground_truth'))
    gt = gt.rename(columns={fn_col:'filename', lbl_col:'label_raw'})
    gt['label_int'] = gt['label_raw'].map(
        lambda x: LABEL_MAP.get(x, LABEL_MAP.get(str(x).lower().strip(), -1)))
    return gt[gt['label_int'].isin([0,1])][['filename','label_int']]

def load_durations(name):
    p = DURATIONS_DIR / f'{name}_durations.csv'
    if not p.exists():
        print(f'  WARN: {p} missing — duration filter is a no-op for {name}')
        return None
    return pd.read_csv(p)[['filename','speaking_time_s']]

def load_folder(name):
    gt   = load_gt(name)
    text = pd.read_csv(NB_DIR / f'{name}_features.csv')
    df   = gt.merge(text, on='filename', how='inner')
    wp   = pd.read_csv(_first_existing(WAVLM_WHOLE_CANDIDATES(name)))
    df   = df.merge(wp, on='filename', how='inner')
    wh   = pd.read_csv(NB_DIR / f'{name}_whisper_whole.csv')
    df   = df.merge(wh, on='filename', how='inner')
    df['batch'] = name
    dur = load_durations(name)
    if dur is not None:
        df = df.merge(dur, on='filename', how='left')
    return df

def filter_by_duration(df, min_s):
    if min_s <= 0 or 'speaking_time_s' not in df.columns: return df
    keep = (df['speaking_time_s'] >= min_s) | df['speaking_time_s'].isna()
    return df[keep].reset_index(drop=True)

ALL_BATCHES = ['audios2','audios4','audios5', BATCH_NEW]
batches_full = {b: attach_candidate_id(load_folder(b))            for b in ALL_BATCHES}
batches      = {b: filter_by_duration(batches_full[b], MIN_SPEAKING_S) for b in ALL_BATCHES}

print(f'=== Per-batch counts (raw -> filtered at >= {MIN_SPEAKING_S}s) ===')
for b in ALL_BATCHES:
    f0, f1 = batches_full[b], batches[b]
    y0, y1 = f0['label_int'].values, f1['label_int'].values
    print(f'  {b:8s}:  rows {len(f0):4d} -> {len(f1):4d}   '
          f'cheat {int((y0==1).sum()):3d}->{int((y1==1).sum()):3d}   '
          f'honest {int((y0==0).sum()):3d}->{int((y1==0).sum()):3d}   '
          f'cands {f0[\"candidate_id\"].nunique()}->{f1[\"candidate_id\"].nunique()}')""")

# ============================================================
code("""# === 3.2 Feature column sets + text feature ranking on audios2 ===
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
WH_COLS    = [c for c in first.columns if c.startswith('whisper_')]
WP_COLS    = [c for c in first.columns if c.startswith('wavlm_')
              and not c.startswith('wavlm_mean_') and not c.startswith('wavlm_std_')]
TEXT_ALL   = [c for c in ALL_TEXT_FEATURES if c in first.columns]
TEXT_STYLO = [c for c in STYLO_FEATS       if c in first.columns]
TEXT_RANK  = [c for c in TEXT_RANK_COLS    if c in first.columns]

# Same text-feature ranking method as fusions_cv: XGB importances on audios2 only.
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

print(f'whisper_wp d={len(WH_COLS)}  wavlm_wp d={len(WP_COLS)}  '
      f'text_all d={len(TEXT_ALL)}  text_stylo d={len(TEXT_STYLO)}  text_rank d={len(TEXT_RANK)}')""")

# ============================================================
code("""# === 3.3 Base-model registry (same 9 as fusions_cv) ===
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
    'text_top10_xgb':  (mk_X(TEXT_TOP10), lambda s=RANDOM_SEED: make_xgb(len(TEXT_TOP10), s)),
    'text_top15_xgb':  (mk_X(TEXT_TOP15), lambda s=RANDOM_SEED: make_xgb(len(TEXT_TOP15), s)),
    'text_top20_xgb':  (mk_X(TEXT_TOP20), lambda s=RANDOM_SEED: make_xgb(len(TEXT_TOP20), s)),
    'text_stylo_xgb':  (mk_X(TEXT_STYLO), lambda s=RANDOM_SEED: make_xgb(len(TEXT_STYLO), s)),
    'text_all_xgb':    (mk_X(TEXT_ALL),   lambda s=RANDOM_SEED: make_xgb(len(TEXT_ALL),   s)),
    'whisper_wp_rf':   (mk_X(WH_COLS),    lambda s=RANDOM_SEED: make_rf (len(WH_COLS),    s)),
    'whisper_wp_xgb':  (mk_X(WH_COLS),    lambda s=RANDOM_SEED: make_xgb(len(WH_COLS),    s)),
    'wavlm_wp_rf':     (mk_X(WP_COLS),    lambda s=RANDOM_SEED: make_rf (len(WP_COLS),    s)),
    'wavlm_wp_xgb':    (mk_X(WP_COLS),    lambda s=RANDOM_SEED: make_xgb(len(WP_COLS),    s)),
}
ALIAS = {
    'text_top10_xgb':'t10','text_top15_xgb':'t15','text_top20_xgb':'t20',
    'text_stylo_xgb':'tst','text_all_xgb':'tal',
    'whisper_wp_rf':'wh.rf','whisper_wp_xgb':'wh.xg',
    'wavlm_wp_rf':'wv.rf', 'wavlm_wp_xgb':'wv.xg',
}
def short_members(s): return '+'.join(ALIAS.get(m, m) for m in s.split(','))
print('Base registry ready (9 models).')""")

# ============================================================
md("""## 4. Pick top-3 from rotation A `summary_avg.csv`

Filters to `rotation == 'A'`, strategy `F1`, sorts by `te_f1` desc, drops infeasible rows, takes the top 3.""")

# ============================================================
code("""# === 4. Pick top-3 from summary_avg.csv (rotation A, strategy F1) ===
summary_path = FUSIONS_DIR / 'summary_avg.csv'
assert summary_path.exists(), (
    f'Missing {summary_path}. Copy it from the company laptop\\'s checkpoints_fusions/ folder.')
summary = pd.read_csv(summary_path)
print(f'Loaded summary_avg: {len(summary)} rows  cols={list(summary.columns)[:12]}...')

sub = summary[(summary['rotation'] == PICK_ROTATION) &
              (summary['strategy'] == PICK_STRATEGY)].copy()
sub = sub.dropna(subset=[PICK_RANK_BY, 'thr'])
sub = sub.sort_values(PICK_RANK_BY, ascending=False).reset_index(drop=True)

print(f'\\nTop 10 (rotation={PICK_ROTATION}, strategy={PICK_STRATEGY}, ranked by {PICK_RANK_BY}):')
disp_cols = ['fusion_type','members','avg_weights','thr',
             'cv_f1','te_f1','te_prec','te_rec','gap_primary']
disp_cols = [c for c in disp_cols if c in sub.columns]
sub['short'] = sub['members'].map(short_members)
with pd.option_context('display.max_columns', None, 'display.width', 220, 'display.max_colwidth', 60):
    print(sub.head(10)[['short'] + disp_cols].to_string(index=False))

picks = sub.head(PICK_TOP_N).reset_index(drop=True)
print(f'\\n=== TOP-{PICK_TOP_N} PICKS ===')

PICK_RECIPES = []
for i, row in picks.iterrows():
    members = row['members'].split(',')
    aw = row.get('avg_weights', '')
    if isinstance(aw, str) and aw.strip():
        weights = [float(x) for x in aw.split(',')]
    else:
        weights = [1.0] * len(members)
    weights = list(np.array(weights) / sum(weights))
    rec = {
        'idx':         i + 1,
        'tag':         f'pick{i+1}_' + '+'.join(ALIAS.get(m, m) for m in members),
        'members':     members,
        'weights':     weights,
        'frozen_thr':  float(row['thr']),
        'src_te_f1':   float(row[PICK_RANK_BY]),
        'src_cv_f1':   float(row['cv_f1']) if 'cv_f1' in row else None,
        'src_gap':     float(row['gap_primary']) if 'gap_primary' in row else None,
        'fusion_type': row['fusion_type'],
    }
    PICK_RECIPES.append(rec)
    print(f'  {rec[\"tag\"]:40s}  weights={[round(w,2) for w in weights]}  '
          f'frozen_thr={rec[\"frozen_thr\"]:.2f}  src_te_f1={rec[\"src_te_f1\"]:.4f}  '
          f'src_gap={rec[\"src_gap\"]:+.4f}')""")

# ============================================================
md("""## 5. Refit + score helpers

Two key functions:

- `fused_proba_for_pool(members, weights, df_train, df_score)` — fit each base model on `df_train`, score `df_score`, blend probas with `weights`. Used both for predicting on audios6 and for OOF inside CV.
- `cv_thr_for_pick(members, weights, df_pool)` — 5-fold candidate-isolated GroupKFold on `df_pool`, build OOF proba, return F1-max threshold. This is the "honest deployment threshold" derived without peeking at audios6.""")

# ============================================================
code("""# === 5. Refit + score helpers ===
def fused_proba_for_pool(members, weights, df_train, df_score):
    \"\"\"Fit each base model on df_train, score df_score, return weighted-average probas.\"\"\"
    fused = np.zeros(len(df_score))
    for m, w in zip(members, weights):
        if w == 0: continue
        X_fn, factory = BASE_REGISTRY[m]
        X_tr = X_fn(df_train); y_tr = df_train['label_int'].values
        sc   = StandardScaler().fit(X_tr)
        clf  = factory(RANDOM_SEED)
        clf.fit(sc.transform(X_tr), y_tr)
        p_sc = clf.predict_proba(sc.transform(X_fn(df_score)))[:, 1]
        fused += float(w) * p_sc
    return fused

def cv_thr_for_pick(members, weights, df_pool, n_folds=N_FOLDS, seed=RANDOM_SEED):
    \"\"\"Candidate-isolated GroupKFold OOF on df_pool, return F1-max thr on concatenated OOF.\"\"\"
    df = df_pool.reset_index(drop=True)
    oof = np.full(len(df), np.nan)
    y   = df['label_int'].values
    cands = df['candidate_id'].values
    gkf = GroupKFold(n_splits=n_folds)
    for fi, (tr_idx, va_idx) in enumerate(gkf.split(df, y, groups=cands)):
        df_tr = df.iloc[tr_idx]
        df_va = df.iloc[va_idx]
        oof[va_idx] = fused_proba_for_pool(members, weights, df_tr, df_va)
    valid = ~np.isnan(oof)
    thr, f1 = best_f1_thr(oof[valid], y[valid])
    return float(thr), float(f1), oof, y

print('Refit + score helpers ready.')""")

# ============================================================
md("""## 6. Section 1 — baseline evaluation: top-3 picks × {Mode A, Mode B}

For each pick × mode:
1. Refit base models on the mode's training pool.
2. Score audios6 with frozen weights.
3. Derive a fresh CV threshold on that training pool (`cv_thr`).
4. Compute audios6 metrics at three thresholds: `frozen_thr`, `cv_thr`, `optimal_thr_a6` (F1-max sweep on audios6).

Saves:
- `checkpoints_audios6_eval/baseline_results.csv` — one row per (pick, mode, threshold_kind)
- `checkpoints_audios6_eval/baseline_predictions.csv` — per-audio probas + binary preds at each threshold
- `checkpoints_audios6_eval/baseline_oof_{tag}_{mode}.csv` — OOF proba on the training pool used to derive `cv_thr`""")

# ============================================================
code("""# === 6. Run baseline evaluation ===
df_a6  = batches[BATCH_NEW]
y_a6   = df_a6['label_int'].values
print(f'audios6 (filtered): n={len(df_a6)}  cheat={int((y_a6==1).sum())}/{len(y_a6)}\\n')

MODES = [
    ('A', 'no_a5_in_train',  TRAIN_BATCHES_A),
    ('B', 'with_a5_in_train', TRAIN_BATCHES_B),
]

baseline_rows = []
pred_cols = {'filename': df_a6['filename'].values,
             'candidate_id': df_a6['candidate_id'].values,
             'label': y_a6}

for rec in PICK_RECIPES:
    print('=' * 100)
    print(f' PICK {rec[\"idx\"]}: {rec[\"tag\"]}')
    print(f'   members={rec[\"members\"]}  weights={[round(w,2) for w in rec[\"weights\"]]}')
    print(f'   src_cv_f1={rec[\"src_cv_f1\"]:.4f}  src_te_f1={rec[\"src_te_f1\"]:.4f}  '
          f'src_gap={rec[\"src_gap\"]:+.4f}  frozen_thr={rec[\"frozen_thr\"]:.2f}')
    print('=' * 100)

    for mode_name, mode_label, mode_batches in MODES:
        df_pool = pd.concat([batches[b] for b in mode_batches], ignore_index=True)
        y_pool  = df_pool['label_int'].values
        print(f'  --- Mode {mode_name} ({mode_label}): train pool={mode_batches}  '
              f'n={len(df_pool)} (+{int((y_pool==1).sum())}) ---')

        # Step 1: derive CV threshold on this training pool
        t0 = time.time()
        cv_thr, cv_f1, oof, y_oof = cv_thr_for_pick(
            rec['members'], rec['weights'], df_pool)
        print(f'     CV ({N_FOLDS}-fold candidate-isolated) on pool: '
              f'cv_thr={cv_thr:.2f}  cv_f1={cv_f1:.4f}  ({time.time()-t0:.1f}s)')

        # Save OOF for inspection
        oof_df = df_pool[['filename','candidate_id','label_int','batch']].copy()
        oof_df['oof_proba'] = oof
        oof_df.to_csv(SAVE_DIR / f'baseline_oof_{rec[\"tag\"]}_mode{mode_name}.csv', index=False)

        # Step 2: refit on full pool, score audios6
        t0 = time.time()
        proba_a6 = fused_proba_for_pool(rec['members'], rec['weights'], df_pool, df_a6)
        print(f'     refit+score audios6: {time.time()-t0:.1f}s')

        # Cache predictions per pick × mode
        pred_cols[f'proba_{rec[\"tag\"]}_mode{mode_name}'] = proba_a6

        # Step 3: optimal threshold sweep on audios6 (diagnostic only)
        opt_thr, opt_f1 = best_f1_thr(proba_a6, y_a6)

        # Step 4: metrics at all three thresholds
        for kind, thr in [('frozen', rec['frozen_thr']),
                           ('cv',     cv_thr),
                           ('optimal_a6', opt_thr)]:
            m = _metrics_at(proba_a6, y_a6, thr)
            row = {
                'pick':       rec['tag'],
                'fusion_type': rec['fusion_type'],
                'members':    ','.join(rec['members']),
                'weights':    ','.join(f'{w:.2f}' for w in rec['weights']),
                'mode':       mode_name,
                'mode_label': mode_label,
                'train_pool': '+'.join(mode_batches),
                'thr_kind':   kind,
                'thr':        round(float(thr), 3),
                'cv_thr_pool':round(cv_thr, 3),
                'cv_f1_pool': round(cv_f1, 4),
                'a6_f1':      round(m['f1'], 4),
                'a6_prec':    round(m['prec'], 4),
                'a6_rec':     round(m['rec'], 4),
                'a6_tp':      m['tp'], 'a6_fp': m['fp'],
                'a6_fn':      m['fn'], 'a6_tn': m['tn'],
                'a6_n':       m['n'],
                'src_te_f1':  rec['src_te_f1'],
                'src_cv_f1':  rec['src_cv_f1'],
            }
            baseline_rows.append(row)
            print(f'     thr_kind={kind:11s}  thr={thr:.2f}  '
                  f'F1={m[\"f1\"]:.4f}  P={m[\"prec\"]:.4f}  R={m[\"rec\"]:.4f}  '
                  f'TP/FP/FN={m[\"tp\"]}/{m[\"fp\"]}/{m[\"fn\"]}')
        print(f'     [opt vs cv  thr drift = {opt_thr - cv_thr:+.2f}]  '
              f'[opt vs frozen thr drift = {opt_thr - rec[\"frozen_thr\"]:+.2f}]')
        # Add binary preds at all thresholds
        pred_cols[f'pred_{rec[\"tag\"]}_mode{mode_name}_frozen']     = (proba_a6 >= rec['frozen_thr']).astype(int)
        pred_cols[f'pred_{rec[\"tag\"]}_mode{mode_name}_cv']         = (proba_a6 >= cv_thr).astype(int)
        pred_cols[f'pred_{rec[\"tag\"]}_mode{mode_name}_optimal_a6'] = (proba_a6 >= opt_thr).astype(int)
    print()

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(SAVE_DIR / 'baseline_results.csv', index=False)

pred_df = pd.DataFrame(pred_cols)
pred_df.to_csv(SAVE_DIR / 'baseline_predictions.csv', index=False)

print(f'\\nSaved: {SAVE_DIR/\"baseline_results.csv\"}  ({len(baseline_df)} rows)')
print(f'       {SAVE_DIR/\"baseline_predictions.csv\"}  ({len(pred_df)} rows × {len(pred_df.columns)} cols)')""")

# ============================================================
code("""# === 6.1 Pretty per-pick comparison table ===
display_cols = ['pick','mode','thr_kind','thr','a6_f1','a6_prec','a6_rec',
                'a6_tp','a6_fp','a6_fn','cv_thr_pool','cv_f1_pool','src_te_f1']
display_cols = [c for c in display_cols if c in baseline_df.columns]

for tag in baseline_df['pick'].unique():
    print('\\n' + '#'*120)
    print(f'#  {tag}')
    print('#'*120)
    sub = baseline_df[baseline_df['pick'] == tag].copy()
    sub = sub.sort_values(['mode','thr_kind'])
    with pd.option_context('display.max_columns', None, 'display.width', 200, 'display.max_colwidth', 60):
        print(sub[display_cols].to_string(index=False))""")

# ============================================================
code("""# === 6.2 Diagnostic: model-fault vs threshold-fault summary ===
# For each (pick, mode):
#   - frozen_f1 (using rotation A's avg thr)
#   - cv_f1     (using CV-derived thr on the pool)
#   - optimal_f1 (using audios6 F1-max thr)
# Big gap optimal vs cv => threshold transfer failed (model still healthy)
# Small gap optimal vs cv but large drop vs src_te_f1 => model itself degraded

diag_rows = []
for tag in baseline_df['pick'].unique():
    src_te_f1 = baseline_df.loc[baseline_df['pick']==tag, 'src_te_f1'].iloc[0]
    for mode in ['A','B']:
        sub = baseline_df[(baseline_df['pick']==tag) & (baseline_df['mode']==mode)]
        if not len(sub): continue
        f_frz = sub.loc[sub['thr_kind']=='frozen',     'a6_f1'].values
        f_cv  = sub.loc[sub['thr_kind']=='cv',         'a6_f1'].values
        f_opt = sub.loc[sub['thr_kind']=='optimal_a6', 'a6_f1'].values
        t_frz = sub.loc[sub['thr_kind']=='frozen',     'thr'].values
        t_cv  = sub.loc[sub['thr_kind']=='cv',         'thr'].values
        t_opt = sub.loc[sub['thr_kind']=='optimal_a6', 'thr'].values
        diag_rows.append({
            'pick':        tag,
            'mode':        mode,
            'src_te_f1':   round(src_te_f1, 4),
            'a6_frozen_f1':  round(float(f_frz[0]), 4) if len(f_frz) else None,
            'a6_cv_f1':      round(float(f_cv [0]), 4) if len(f_cv ) else None,
            'a6_optimal_f1': round(float(f_opt[0]), 4) if len(f_opt) else None,
            'frozen_thr':  round(float(t_frz[0]), 2) if len(t_frz) else None,
            'cv_thr':      round(float(t_cv [0]), 2) if len(t_cv ) else None,
            'opt_thr':     round(float(t_opt[0]), 2) if len(t_opt) else None,
            'gap_opt_minus_cv':      round(float(f_opt[0] - f_cv [0]), 4) if (len(f_opt) and len(f_cv)) else None,
            'gap_opt_minus_frozen':  round(float(f_opt[0] - f_frz[0]), 4) if (len(f_opt) and len(f_frz)) else None,
            'gap_src_minus_opt':     round(float(src_te_f1 - f_opt[0]), 4) if len(f_opt) else None,
        })
diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(SAVE_DIR / 'baseline_diagnostic.csv', index=False)

print('=' * 120)
print('  DIAGNOSTIC — interpret:')
print('   gap_opt_minus_cv     ≈ 0  -> CV-derived threshold transfers; model is healthy')
print('                        > 0  -> threshold drifted on audios6; tune the threshold, not the model')
print('   gap_src_minus_opt    ≈ 0  -> model performs as well on audios6 as on rotation-A audios5')
print('                        > 0  -> model itself degraded on audios6 (distribution shift)')
print('=' * 120)
with pd.option_context('display.max_columns', None, 'display.width', 200, 'display.max_colwidth', 50):
    print(diag_df.to_string(index=False))
print(f'\\nSaved: {SAVE_DIR/\"baseline_diagnostic.csv\"}')""")

# ============================================================
md("""## 7. Section 2 — experimental: re-search with audios5 in train

Off by default (`RUN_EXPERIMENTAL_RESEARCH = False` in cell 0). Flip the flag to enable.

Protocol when enabled:
- Pool = `audios2 + audios4 + audios5` (no held-out test inside the search; audios6 is the test).
- 5-fold candidate-isolated GroupKFold on the pool. Per fold: fit each of the 9 base models, build OOF proba.
- For every 2-way pair and 3-way triple of base models, search weights on the concatenated OOF for **F1-max** (matching the F1 strategy in fusions_cv).
- Refit on full pool with the chosen weights, score audios6, report `te_f1` at the chosen `cv_thr`.
- Rank top candidates by **`a6_f1`** (since user asked for top-by-test-F1) and compare to the rotation-A picks.

Heads-up: this is heavier than Section 1 (does a full base-model OOF on a larger pool, then 36 pairs + 84 triples × weight search). Expect 5–15 min depending on machine.""")

# ============================================================
code("""# === 7. Optional re-search with a5 in train ===
if not RUN_EXPERIMENTAL_RESEARCH:
    print('RUN_EXPERIMENTAL_RESEARCH=False — skipping. Flip the flag in cell 0 to run.')
else:
    df_pool = pd.concat([batches[b] for b in TRAIN_BATCHES_B], ignore_index=True)
    y_pool  = df_pool['label_int'].values
    cands   = df_pool['candidate_id'].values
    print(f'Pool for re-search: {TRAIN_BATCHES_B}  n={len(df_pool)} (+{int((y_pool==1).sum())})')

    # Step 1: per-base OOF on pool
    print('\\nStep 1 — per-base OOF on pool ({}-fold candidate-isolated)'.format(N_FOLDS))
    base_oof = {m: np.full(len(df_pool), np.nan) for m in BASE_REGISTRY}
    base_a6  = {m: np.zeros(len(df_a6))           for m in BASE_REGISTRY}
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fi, (tr_idx, va_idx) in enumerate(gkf.split(df_pool, y_pool, groups=cands)):
        df_tr = df_pool.iloc[tr_idx]
        df_va = df_pool.iloc[va_idx]
        for m, (X_fn, factory) in BASE_REGISTRY.items():
            sc  = StandardScaler().fit(X_fn(df_tr))
            clf = factory(RANDOM_SEED + fi)
            clf.fit(sc.transform(X_fn(df_tr)), df_tr['label_int'].values)
            base_oof[m][va_idx] = clf.predict_proba(sc.transform(X_fn(df_va)))[:, 1]
        print(f'  fold {fi} done')

    # Refit each base on full pool + score audios6
    print('\\nStep 2 — refit base models on full pool, score audios6')
    for m, (X_fn, factory) in BASE_REGISTRY.items():
        sc  = StandardScaler().fit(X_fn(df_pool))
        clf = factory(RANDOM_SEED)
        clf.fit(sc.transform(X_fn(df_pool)), y_pool)
        base_a6[m] = clf.predict_proba(sc.transform(X_fn(df_a6)))[:, 1]

    # Step 3: search 2-way + 3-way fusions on OOF
    print('\\nStep 3 — search 2-way + 3-way fusions on pool OOF')
    ALPHAS_2WAY = np.arange(0.0, 1.0 + 1e-9, 0.05)
    W3_GRID = []
    for w1 in np.arange(0.0, 1.0 + 1e-9, 0.10):
        for w2 in np.arange(0.0, 1.0 - w1 + 1e-9, 0.10):
            w3 = max(0.0, 1.0 - w1 - w2)
            W3_GRID.append((round(float(w1),2), round(float(w2),2), round(float(w3),2)))

    rows = []
    MODELS = list(BASE_REGISTRY.keys())
    for m in MODELS:
        thr_b, f1_b = best_f1_thr(base_oof[m], y_pool)
        m_a6 = _metrics_at(base_a6[m], y_a6, thr_b)
        rows.append({'fusion_type':'base','members':m,'weights':'',
                      'cv_thr':round(thr_b,3),'cv_f1':round(f1_b,4),
                      'a6_f1':round(m_a6['f1'],4),'a6_prec':round(m_a6['prec'],4),
                      'a6_rec':round(m_a6['rec'],4)})

    for a, b in itertools.combinations(MODELS, 2):
        best = None
        for alpha in ALPHAS_2WAY:
            fused_oof = alpha * base_oof[a] + (1-alpha) * base_oof[b]
            thr, f1 = best_f1_thr(fused_oof, y_pool)
            if best is None or f1 > best[2]:
                best = (alpha, thr, f1)
        alpha, thr, f1 = best
        fused_a6 = alpha * base_a6[a] + (1-alpha) * base_a6[b]
        m_a6 = _metrics_at(fused_a6, y_a6, thr)
        rows.append({'fusion_type':'2way','members':f'{a},{b}',
                      'weights':f'{alpha:.2f},{1-alpha:.2f}',
                      'cv_thr':round(thr,3),'cv_f1':round(f1,4),
                      'a6_f1':round(m_a6['f1'],4),'a6_prec':round(m_a6['prec'],4),
                      'a6_rec':round(m_a6['rec'],4)})

    for a, b, c in itertools.combinations(MODELS, 3):
        best = None
        for w1, w2, w3 in W3_GRID:
            fused_oof = w1*base_oof[a] + w2*base_oof[b] + w3*base_oof[c]
            thr, f1 = best_f1_thr(fused_oof, y_pool)
            if best is None or f1 > best[2]:
                best = ((w1,w2,w3), thr, f1)
        (w1,w2,w3), thr, f1 = best
        fused_a6 = w1*base_a6[a] + w2*base_a6[b] + w3*base_a6[c]
        m_a6 = _metrics_at(fused_a6, y_a6, thr)
        rows.append({'fusion_type':'3way','members':f'{a},{b},{c}',
                      'weights':f'{w1:.2f},{w2:.2f},{w3:.2f}',
                      'cv_thr':round(thr,3),'cv_f1':round(f1,4),
                      'a6_f1':round(m_a6['f1'],4),'a6_prec':round(m_a6['prec'],4),
                      'a6_rec':round(m_a6['rec'],4)})

    research_df = pd.DataFrame(rows)
    research_df['short'] = research_df['members'].map(short_members)
    research_df = research_df.sort_values('a6_f1', ascending=False).reset_index(drop=True)
    research_df.to_csv(SAVE_DIR / 'experimental_research.csv', index=False)

    print('\\n=== TOP 15 by a6_f1 (re-searched on a2+a4+a5 pool) ===')
    cols = ['fusion_type','short','weights','cv_thr','cv_f1','a6_f1','a6_prec','a6_rec']
    with pd.option_context('display.max_columns', None, 'display.width', 200, 'display.max_colwidth', 60):
        print(research_df.head(15)[cols].to_string(index=False))

    # Compare against rotation A picks
    print('\\n=== Rotation-A picks present in re-search ===')
    for rec in PICK_RECIPES:
        key = ','.join(rec['members'])
        match = research_df[research_df['members'] == key]
        if len(match):
            r = match.iloc[0]
            rank = research_df.index[research_df['members'] == key][0] + 1
            print(f'  rank #{rank:3d} (of {len(research_df)}): {short_members(key):40s}  '
                  f'a6_f1={r[\"a6_f1\"]:.4f}  cv_thr={r[\"cv_thr\"]:.2f}  cv_f1={r[\"cv_f1\"]:.4f}')
    print(f'\\nSaved: {SAVE_DIR/\"experimental_research.csv\"}')""")

# ============================================================
md("""## 8. How to read these results

**Per-pick × per-mode interpretation (from `baseline_diagnostic.csv`):**

| Pattern | What it means |
|---|---|
| `gap_opt_minus_cv ≈ 0` AND `a6_optimal_f1` close to `src_te_f1` | Healthy. CV-derived threshold transfers, model generalises to audios6. Ship as-is. |
| `gap_opt_minus_cv ≈ 0` BUT `a6_optimal_f1 << src_te_f1` | Threshold is fine, but the model itself underperforms on audios6 — distribution shift. |
| `gap_opt_minus_cv > 0` (large) | Model is fine, threshold drifted. Re-tune the threshold on a labelled audios6 subset before shipping. |
| `a6_frozen_f1 << a6_cv_f1` | Rotation-A frozen threshold is too aggressive/conservative for the new training pool. Trust `cv_thr`, not `frozen_thr`. |

**Mode A vs Mode B comparison:**
- If Mode B beats Mode A on `a6_optimal_f1`: adding audios5 to training helped — keep a5 in.
- If Mode B = Mode A on `a6_optimal_f1`: a5 didn't add new signal. Mode A is simpler.
- If Mode B < Mode A on `a6_optimal_f1`: a5 introduced noise / contradiction with the new distribution.

**Files in `checkpoints_audios6_eval/`:**
- `baseline_results.csv` — full results, one row per (pick, mode, threshold_kind).
- `baseline_diagnostic.csv` — compact comparison table.
- `baseline_predictions.csv` — per-audio probas + binary preds for every pick × mode × threshold.
- `baseline_oof_{tag}_modeA.csv`, `..._modeB.csv` — OOF probas on each training pool (for re-deriving thresholds or sanity checks).
- `experimental_research.csv` — only if `RUN_EXPERIMENTAL_RESEARCH=True` was set. Full re-search ranking.""")

# ============================================================
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = Path(__file__).resolve().parent / 'audios6_eval.ipynb'
out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Wrote: {out}  ({len(CELLS)} cells)')
