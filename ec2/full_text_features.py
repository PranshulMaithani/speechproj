"""Full handcrafted-feature extractor — same schema as audios6_eval.ipynb.

Produces 55 numeric features per (audio, transcript) pair:

    G_DISFLUENCY  (6) : filler_rate, filler_count, repetition_rate, repair_rate,
                        discourse_marker_rate, hedge_rate
    G_STYLOMETRIC (15): ttr, mattr, mtld, complex_word_rate, avg_word_length,
                        n_words, n_unique_words, avg_sentence_length,
                        std_sentence_length, fragment_rate, n_sentences,
                        self_ref_rate, noun_rate, verb_rate, adj_rate
    G_PAUSE       (15): pause_mean, pause_std, pause_median, pause_skew,
                        long_pause_rate, pause_ratio, n_pauses, pause_regularity,
                        pause_before_content_ratio, pause_before_function_ratio,
                        mid_phrase_pause_rate, words_per_sec, articulation_rate,
                        initial_pause, longest_pause
    G_SUSPICIOUS  (2) : suspicious_gap_count, suspicious_gap_ratio
    G_FORMAL_AI   (4) : formal_transition_count, formal_transition_rate,
                        ai_phrase_count, ai_phrase_rate
    G_PROSODIC    (8) : f0_mean, f0_std, f0_range, f0_skew, f0_slope,
                        energy_mean, energy_std, speaking_rate_std
    G_VOICE_Q     (3) : jitter_local, shimmer_local, hnr_mean
    G_PERPLEXITY  (2) : mean_perplexity, burstiness

Optional deps gracefully degrade to zeros when missing:
    spacy + en_core_web_sm  -> POS-based features
    parselmouth             -> voice quality (jitter / shimmer / HNR)
    transformers + gpt2     -> perplexity / burstiness

Required: numpy, pandas, librosa.

Transcript input format (matches what faster-whisper returns):
    {"text": str, "words": [{"word": str, "start": float, "end": float}, ...]}
"""

from __future__ import annotations

import re
from collections import Counter

import librosa
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Optional dependencies
# ----------------------------------------------------------------------------

try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False
    nlp = None

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAS_PARSELMOUTH = True
except Exception:
    HAS_PARSELMOUTH = False

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    _gpt2_tok = None
    _gpt2_mdl = None

    def _gpt2():
        global _gpt2_tok, _gpt2_mdl
        if _gpt2_mdl is None:
            _gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
            _gpt2_mdl = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        return _gpt2_mdl, _gpt2_tok
    HAS_GPT2 = True
except Exception:
    HAS_GPT2 = False
    _gpt2 = None  # type: ignore

# ----------------------------------------------------------------------------
# Lexicons
# ----------------------------------------------------------------------------

FILLERS = {"um", "uh", "uh-huh", "uhm", "umm", "hmm", "hm", "er", "ah", "ehm", "mhm"}
DISCOURSE_MARKERS = {"you know", "i mean", "like", "basically", "actually",
                     "so", "well", "right", "okay", "oh", "anyway", "honestly"}
HEDGES = {"i think", "i guess", "maybe", "perhaps", "probably", "kind of",
          "sort of", "i believe", "it seems", "i suppose", "might be"}
SELF_REF = {"i", "me", "my", "myself", "mine", "i'm", "i've", "i'd", "i'll"}
REPAIRS = ["i mean", "no wait", "sorry i", "actually no", "wait no", "no no"]
FORMAL_TRANS = ["furthermore", "moreover", "however", "therefore", "additionally",
                "consequently", "nevertheless", "hence", "thus", "in conclusion",
                "firstly", "secondly", "thirdly", "in summary", "to summarize",
                "in essence", "overall", "ultimately"]
AI_PHRASES = ["it is important to note", "it is worth noting", "it should be noted",
              "in conclusion", "to summarize", "in summary", "fundamentally",
              "plays a crucial role", "plays a vital role", "a wide range of",
              "on the other hand", "in other words", "delve into", "it is crucial"]
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
FUNCTION_POS = {"DET", "ADP", "CONJ", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}

G_DISFLUENCY = ["filler_rate", "filler_count", "repetition_rate", "repair_rate",
                "discourse_marker_rate", "hedge_rate"]
G_STYLOMETRIC = ["ttr", "mattr", "mtld", "complex_word_rate", "avg_word_length",
                 "n_words", "n_unique_words", "avg_sentence_length",
                 "std_sentence_length", "fragment_rate", "n_sentences",
                 "self_ref_rate", "noun_rate", "verb_rate", "adj_rate"]
G_PAUSE = ["pause_mean", "pause_std", "pause_median", "pause_skew", "long_pause_rate",
           "pause_ratio", "n_pauses", "pause_regularity",
           "pause_before_content_ratio", "pause_before_function_ratio",
           "mid_phrase_pause_rate", "words_per_sec", "articulation_rate",
           "initial_pause", "longest_pause"]
G_SUSPICIOUS = ["suspicious_gap_count", "suspicious_gap_ratio"]
G_FORMAL_AI = ["formal_transition_count", "formal_transition_rate",
               "ai_phrase_count", "ai_phrase_rate"]
G_PROSODIC = ["f0_mean", "f0_std", "f0_range", "f0_skew", "f0_slope",
              "energy_mean", "energy_std", "speaking_rate_std"]
G_VOICE_Q = ["jitter_local", "shimmer_local", "hnr_mean"]
G_PERPLEXITY = ["mean_perplexity", "burstiness"]
ALL_FEATURES = (G_DISFLUENCY + G_STYLOMETRIC + G_PAUSE + G_SUSPICIOUS
                + G_FORMAL_AI + G_PROSODIC + G_VOICE_Q + G_PERPLEXITY)

WORD_RE = re.compile(r"[a-zA-Z']+")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _syllable_count(word: str) -> int:
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    count = 0
    prev_vowel = False
    for ch in word:
        iv = ch in "aeiouy"
        if iv and not prev_vowel:
            count += 1
        prev_vowel = iv
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _mattr(words: list[str], window: int = 50) -> float:
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    return float(np.mean([len(set(words[i:i + window])) / window
                          for i in range(len(words) - window + 1)]))


def _mtld(words: list[str], threshold: float = 0.72) -> float:
    if len(words) < 10:
        return 0.0

    def _run(ws):
        factors, current = 0, []
        for w in ws:
            current.append(w)
            ttr = len(set(current)) / len(current)
            if ttr <= threshold:
                factors += 1
                current = []
        if current:
            ttr = len(set(current)) / len(current)
            if ttr < 1.0:
                factors += (1.0 - ttr) / (1.0 - threshold)
        return len(ws) / factors if factors > 0 else len(ws)
    return round((_run(words) + _run(words[::-1])) / 2, 2)


# ----------------------------------------------------------------------------
# Feature blocks
# ----------------------------------------------------------------------------

def compute_text_and_stylo(text: str) -> dict:
    out = {k: 0 for k in G_DISFLUENCY + G_STYLOMETRIC}
    if not text or len(text.strip()) < 10:
        return out
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
        sentences = [s for s in re.split(r"[.!?]+", text_lower) if s.strip()]
        pos_c = Counter()
    n_words = len(words)
    if n_words < 5:
        return out
    filler_count = sum(1 for w in all_toks if w in FILLERS)
    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    bc = Counter(bigrams)
    rep_rate = sum(c - 1 for c in bc.values() if c > 1) / max(len(bigrams), 1)
    repair_c = sum(text_lower.count(r) for r in REPAIRS)
    n_sents = max(len(sentences), 1)
    if HAS_SPACY:
        sl = [len([t for t in s if t.is_alpha]) for s in sentences]
    else:
        sl = [len(WORD_RE.findall(s)) for s in sentences]
    sl = [x for x in sl if x > 0]
    tp = sum(pos_c.values()) or 1
    out.update({
        "filler_rate":             filler_count / n_words,
        "filler_count":            filler_count,
        "repetition_rate":         rep_rate,
        "repair_rate":             repair_c / n_sents,
        "discourse_marker_rate":   sum(text_lower.count(d) for d in DISCOURSE_MARKERS) / n_sents,
        "hedge_rate":              sum(text_lower.count(h) for h in HEDGES) / n_sents,
        "ttr":                     len(set(words)) / n_words,
        "mattr":                   _mattr(words),
        "mtld":                    _mtld(words),
        "complex_word_rate":       sum(1 for w in words if _syllable_count(w) >= 3) / n_words,
        "avg_word_length":         float(np.mean([len(w) for w in words])),
        "n_words":                 n_words,
        "n_unique_words":          len(set(words)),
        "avg_sentence_length":     float(np.mean(sl)) if sl else 0,
        "std_sentence_length":     float(np.std(sl)) if len(sl) > 1 else 0,
        "fragment_rate":           sum(1 for x in sl if x < 4) / n_sents,
        "n_sentences":             n_sents,
        "self_ref_rate":           sum(1 for w in all_toks if w in SELF_REF) / n_words,
        "noun_rate":               pos_c.get("NOUN", 0) / tp,
        "verb_rate":               pos_c.get("VERB", 0) / tp,
        "adj_rate":                pos_c.get("ADJ", 0) / tp,
    })
    return out


def compute_pause_and_suspicious(words: list[dict]) -> dict:
    out = {k: 0 for k in G_PAUSE + G_SUSPICIOUS}
    if not words or len(words) < 5:
        return out
    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > 0.05:
            pauses.append({"dur": gap,
                           "after_word": words[i - 1].get("word", ""),
                           "before_word": words[i].get("word", ""),
                           "pos": i})
    initial_pause = words[0]["start"]
    all_gaps = [words[i]["start"] - words[i - 1]["end"]
                for i in range(1, len(words))
                if words[i]["start"] - words[i - 1]["end"] > 0.05]
    longest_pause = max(all_gaps) if all_gaps else 0.0
    out["initial_pause"] = initial_pause
    out["longest_pause"] = longest_pause
    if not pauses:
        return out
    durs = [p["dur"] for p in pauses]
    total_dur = max(words[-1]["end"] - words[0]["start"], 0.1)
    speaking_dur = max(total_dur - sum(durs), 0.1)
    if HAS_SPACY:
        doc = nlp(" ".join(w.get("word", "") for w in words))
        tok_pos = {t.text.lower(): t.pos_ for t in doc}
    else:
        tok_pos = {}
    nbc = nbf = nmp = 0
    for p in pauses:
        pos = tok_pos.get(p["before_word"].lower().strip(".,!?"), "X")
        if pos in CONTENT_POS:
            nbc += 1
        elif pos in FUNCTION_POS:
            nbf += 1
        if not p["after_word"].endswith((".", ",", "!", "?")):
            nmp += 1
    n_p = len(pauses)
    positions = [p["pos"] for p in pauses]
    intervals = [positions[i] - positions[i - 1] for i in range(1, len(positions))]
    suspicious = sum(1 for p in pauses
                     if 0.3 <= p["dur"] <= 0.8
                     and not p["after_word"].rstrip().endswith((".", "!", "?")))
    out.update({
        "pause_mean":                  float(np.mean(durs)),
        "pause_std":                   float(np.std(durs)),
        "pause_median":                float(np.median(durs)),
        "pause_skew":                  float(pd.Series(durs).skew()) if len(durs) > 2 else 0,
        "long_pause_rate":             sum(1 for d in durs if d > 0.5) / n_p,
        "pause_ratio":                 sum(durs) / total_dur,
        "n_pauses":                    n_p,
        "pause_regularity":            float(np.std(intervals)) if intervals else 0,
        "pause_before_content_ratio":  nbc / n_p,
        "pause_before_function_ratio": nbf / n_p,
        "mid_phrase_pause_rate":       nmp / n_p,
        "words_per_sec":               len(words) / total_dur,
        "articulation_rate":           len(words) / speaking_dur,
        "initial_pause":               initial_pause,
        "longest_pause":               longest_pause,
        "suspicious_gap_count":        suspicious,
        "suspicious_gap_ratio":        suspicious / max(len(words), 1),
    })
    return out


def compute_formal_ai(text: str) -> dict:
    out = {k: 0 for k in G_FORMAL_AI}
    if not text:
        return out
    tl = text.lower()
    n_words = max(len(WORD_RE.findall(tl)), 1)
    formal_c = sum(tl.count(p) for p in FORMAL_TRANS)
    ai_c = sum(tl.count(p) for p in AI_PHRASES)
    out["formal_transition_count"] = formal_c
    out["formal_transition_rate"] = 100.0 * formal_c / n_words
    out["ai_phrase_count"] = ai_c
    out["ai_phrase_rate"] = 100.0 * ai_c / n_words
    return out


def compute_prosodic(audio_path) -> dict:
    out = {k: 0 for k in G_PROSODIC}
    try:
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True, duration=120)
    except Exception:
        return out
    if len(audio) < 16000:
        return out
    f0, _, _ = librosa.pyin(audio, fmin=75, fmax=500, sr=16000, frame_length=2048)
    fv = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if len(fv) >= 10:
        slope = float(np.polyfit(np.arange(len(fv)), fv, 1)[0])
        out["f0_mean"] = float(np.mean(fv))
        out["f0_std"] = float(np.std(fv))
        out["f0_range"] = float(fv.max() - fv.min())
        out["f0_skew"] = float(pd.Series(fv).skew())
        out["f0_slope"] = slope
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
    out["energy_mean"] = float(np.mean(rms))
    out["energy_std"] = float(np.std(rms))
    win = 2 * 16000
    rates = [float((librosa.feature.rms(y=audio[s:s + win], frame_length=512, hop_length=256)[0]
                    > np.percentile(rms, 20)).mean())
             for s in range(0, len(audio) - win, 16000)]
    out["speaking_rate_std"] = float(np.std(rates)) if rates else 0
    return out


def compute_voice_quality(audio_path) -> dict:
    out = {k: 0 for k in G_VOICE_Q}
    if not HAS_PARSELMOUTH:
        return out
    try:
        snd = parselmouth.Sound(str(audio_path))
        pp = praat_call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jit = praat_call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shim = praat_call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = praat_call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = praat_call(harm, "Get mean", 0, 0)
        out.update({"jitter_local": float(jit),
                    "shimmer_local": float(shim),
                    "hnr_mean": float(hnr)})
    except Exception:
        pass
    return out


def compute_perplexity(text: str) -> dict:
    out = {k: 0 for k in G_PERPLEXITY}
    if not HAS_GPT2 or not text or len(text.strip()) < 20:
        return out
    try:
        mdl, tok = _gpt2()
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) > 3]
        if not sents:
            return out
        ppls = []
        for s in sents[:20]:
            enc = tok(s, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                loss = mdl(**enc, labels=enc["input_ids"]).loss
            ppls.append(float(torch.exp(loss)))
        out["mean_perplexity"] = float(np.mean(ppls))
        out["burstiness"] = float(np.var(ppls))
    except Exception:
        pass
    return out


def compute_all_features(audio_path, text: str, words: list[dict]) -> dict[str, float]:
    """Return all 55 features prefixed with 'feat_' (matching gt.csv convention)."""
    r: dict = {}
    r.update(compute_text_and_stylo(text))
    r.update(compute_pause_and_suspicious(words))
    r.update(compute_formal_ai(text))
    r.update(compute_prosodic(audio_path))
    r.update(compute_voice_quality(audio_path))
    r.update(compute_perplexity(text))
    return {f"feat_{k}": float(v) for k, v in r.items()}
