"""
Feature extraction for company data.
=====================================
Extracts text + pause features from WhisperX transcripts,
prosodic features from audio, and wav2vec2 scores from ONNX model.

Usage:
    # After transcription (transcripts JSON exists):
    python extract_features_company.py --transcripts transcripts.json --manifest manifest.csv --output features.csv

    # With wav2vec2 ONNX scores:
    python extract_features_company.py --transcripts transcripts.json --manifest manifest.csv --onnx-model biased_wav2vec2_quant.onnx --output features.csv
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
except OSError:
    print("Run: python -m spacy download en_core_web_sm")
    sys.exit(1)

# Lazy-loaded sentence transformer
_sbert_model = None

def _get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sbert_model


# ============================================================
# Constants
# ============================================================

FILLERS = {"um", "uh", "uh-huh", "uhm", "umm", "hmm", "hm", "er", "ah", "ehm", "mhm"}
DISCOURSE_MARKERS = {"you know", "i mean", "like", "basically", "actually", "so",
                     "well", "right", "okay", "oh", "anyway", "honestly"}
HEDGES = {"i think", "i guess", "maybe", "perhaps", "probably", "kind of",
          "sort of", "i believe", "it seems", "i suppose", "might be"}
SELF_REF = {"i", "me", "my", "myself", "mine", "i'm", "i've", "i'd", "i'll"}
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
FUNCTION_POS = {"DET", "ADP", "CONJ", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}


# ============================================================
# Text Features (20 features)
# ============================================================

def compute_text_features(text):
    if not text or len(text.strip()) < 10:
        return _empty_text_features()

    text_lower = text.lower().strip()
    doc = nlp(text_lower)
    words = [t.text for t in doc if t.is_alpha]
    all_tokens = [t.text for t in doc]
    n_words = len(words)
    if n_words < 5:
        return _empty_text_features()

    filler_count = sum(1 for w in all_tokens if w in FILLERS)
    filler_rate = filler_count / n_words

    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
    repetition_rate = repeated_bigrams / max(len(bigrams), 1)

    repair_markers = ["i mean", "no wait", "sorry i", "actually no", "wait no", "no no"]
    repair_count = sum(text_lower.count(m) for m in repair_markers)
    sentences = list(doc.sents)
    n_sents = max(len(sentences), 1)
    repair_rate = repair_count / n_sents

    unique_words = set(words)
    ttr = len(unique_words) / n_words
    mattr = _compute_mattr(words, 50)
    complex_count = sum(1 for w in words if _syllable_count(w) >= 3)
    avg_word_len = np.mean([len(w) for w in words])

    sent_lengths = [len([t for t in s if t.is_alpha]) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    std_sent_len = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    fragment_count = sum(1 for sl in sent_lengths if sl < 4)

    self_ref_count = sum(1 for w in all_tokens if w in SELF_REF)
    dm_count = sum(text_lower.count(dm) for dm in DISCOURSE_MARKERS)
    hedge_count = sum(text_lower.count(h) for h in HEDGES)

    pos_counts = Counter(t.pos_ for t in doc)
    total_pos = sum(pos_counts.values()) or 1

    return {
        "filler_rate": round(filler_rate, 4),
        "filler_count": filler_count,
        "repetition_rate": round(repetition_rate, 4),
        "repair_rate": round(repair_count / n_sents, 4),
        "ttr": round(ttr, 4),
        "mattr": round(mattr, 4),
        "complex_word_rate": round(complex_count / n_words, 4),
        "avg_word_length": round(avg_word_len, 2),
        "n_words": n_words,
        "n_unique_words": len(unique_words),
        "avg_sentence_length": round(avg_sent_len, 2),
        "std_sentence_length": round(std_sent_len, 2),
        "fragment_rate": round(fragment_count / n_sents, 4),
        "n_sentences": n_sents,
        "self_ref_rate": round(self_ref_count / n_words, 4),
        "discourse_marker_rate": round(dm_count / n_sents, 4),
        "hedge_rate": round(hedge_count / n_sents, 4),
        "noun_rate": round(pos_counts.get("NOUN", 0) / total_pos, 4),
        "verb_rate": round(pos_counts.get("VERB", 0) / total_pos, 4),
        "adj_rate": round(pos_counts.get("ADJ", 0) / total_pos, 4),
    }


def _compute_mattr(words, window=50):
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    ttrs = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i+window]
        ttrs.append(len(set(chunk)) / window)
    return np.mean(ttrs)


def _syllable_count(word):
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in "aeiouy"
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _empty_text_features():
    return {
        "filler_rate": 0, "filler_count": 0, "repetition_rate": 0, "repair_rate": 0,
        "ttr": 0, "mattr": 0, "complex_word_rate": 0, "avg_word_length": 0,
        "n_words": 0, "n_unique_words": 0,
        "avg_sentence_length": 0, "std_sentence_length": 0, "fragment_rate": 0, "n_sentences": 0,
        "self_ref_rate": 0, "discourse_marker_rate": 0, "hedge_rate": 0,
        "noun_rate": 0, "verb_rate": 0, "adj_rate": 0,
    }


# ============================================================
# Sentence Transformer Embeddings (384-dim)
# ============================================================

SBERT_DIM = 384

def compute_text_embeddings(text):
    """Encode transcript text into a 384-dim sentence embedding vector."""
    if not text or len(text.strip()) < 10:
        return {f"sbert_{i}": 0.0 for i in range(SBERT_DIM)}

    model = _get_sbert_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return {f"sbert_{i}": float(embedding[i]) for i in range(SBERT_DIM)}


# ============================================================
# Pause Features (13 features) — requires word timestamps
# ============================================================

def compute_pause_features(words):
    if not words or len(words) < 5:
        return _empty_pause_features()

    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i-1]["end"]
        if gap > 0.05:
            pauses.append({
                "duration": gap,
                "before_word": words[i]["word"],
                "after_word": words[i-1]["word"],
                "position": i,
            })

    if not pauses:
        return _empty_pause_features()

    durations = [p["duration"] for p in pauses]
    pause_mean = np.mean(durations)
    pause_std = np.std(durations)
    pause_median = np.median(durations)
    pause_skew = float(pd.Series(durations).skew()) if len(durations) > 2 else 0
    long_pause_rate = sum(1 for d in durations if d > 0.5) / len(durations)
    total_duration = words[-1]["end"] - words[0]["start"]
    pause_ratio = sum(durations) / max(total_duration, 0.1)

    if len(durations) > 1:
        positions = [p["position"] for p in pauses]
        intervals = [positions[i] - positions[i-1] for i in range(1, len(positions))]
        pause_regularity = np.std(intervals) if intervals else 0
    else:
        pause_regularity = 0

    all_text = " ".join(w["word"] for w in words)
    doc = nlp(all_text)
    token_pos = {t.text.lower(): t.pos_ for t in doc}

    n_before_content = 0
    n_before_function = 0
    n_mid_phrase = 0
    for p in pauses:
        word_after = p["before_word"].lower().strip(".,!?;:")
        pos = token_pos.get(word_after, "X")
        if pos in CONTENT_POS:
            n_before_content += 1
        elif pos in FUNCTION_POS:
            n_before_function += 1
        if not any(p["after_word"].endswith(c) for c in [".", "!", "?", ","]):
            n_mid_phrase += 1

    n_pauses = len(pauses)
    n_words_total = len(words)
    speaking_dur = total_duration - sum(durations)

    return {
        "pause_mean": round(pause_mean, 4),
        "pause_std": round(pause_std, 4),
        "pause_median": round(pause_median, 4),
        "pause_skew": round(pause_skew, 4),
        "long_pause_rate": round(long_pause_rate, 4),
        "pause_ratio": round(pause_ratio, 4),
        "n_pauses": n_pauses,
        "pause_regularity": round(pause_regularity, 4),
        "pause_before_content_ratio": round(n_before_content / n_pauses, 4),
        "pause_before_function_ratio": round(n_before_function / n_pauses, 4),
        "mid_phrase_pause_rate": round(n_mid_phrase / n_pauses, 4),
        "words_per_sec": round(n_words_total / max(total_duration, 0.1), 2),
        "articulation_rate": round(n_words_total / max(speaking_dur, 0.1), 2),
    }


def _empty_pause_features():
    return {
        "pause_mean": 0, "pause_std": 0, "pause_median": 0, "pause_skew": 0,
        "long_pause_rate": 0, "pause_ratio": 0, "n_pauses": 0, "pause_regularity": 0,
        "pause_before_content_ratio": 0, "pause_before_function_ratio": 0,
        "mid_phrase_pause_rate": 0, "words_per_sec": 0, "articulation_rate": 0,
    }


# ============================================================
# Prosodic Features (8 features) — from audio directly
# ============================================================

def compute_prosodic_features(audio_path, sr=16000):
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=120)
    except Exception:
        return _empty_prosodic_features()

    if len(audio) < sr:
        return _empty_prosodic_features()

    f0, _, _ = librosa.pyin(audio, fmin=75, fmax=500, sr=sr, frame_length=2048)
    f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if len(f0_voiced) < 10:
        return _empty_prosodic_features()

    x = np.arange(len(f0_voiced))
    f0_slope = float(np.polyfit(x, f0_voiced, 1)[0]) if len(x) > 1 else 0.0

    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]

    window = 2 * sr
    rates = []
    for start in range(0, len(audio) - window, sr):
        chunk = audio[start:start+window]
        chunk_rms = librosa.feature.rms(y=chunk, frame_length=512, hop_length=256)[0]
        rates.append(float((chunk_rms > np.percentile(rms, 20)).mean()))

    return {
        "f0_mean": round(float(np.mean(f0_voiced)), 2),
        "f0_std": round(float(np.std(f0_voiced)), 2),
        "f0_range": round(float(np.max(f0_voiced) - np.min(f0_voiced)), 2),
        "f0_skew": round(float(pd.Series(f0_voiced).skew()), 4),
        "f0_slope": round(f0_slope, 6),
        "energy_mean": round(float(np.mean(rms)), 6),
        "energy_std": round(float(np.std(rms)), 6),
        "speaking_rate_std": round(float(np.std(rates)) if rates else 0, 4),
    }


def _empty_prosodic_features():
    return {
        "f0_mean": 0, "f0_std": 0, "f0_range": 0, "f0_skew": 0, "f0_slope": 0,
        "energy_mean": 0, "energy_std": 0, "speaking_rate_std": 0,
    }


# ============================================================
# wav2vec2 ONNX scores
# ============================================================

def compute_wav2vec2_scores(audio_path, ort_session, sr=16000, window_sec=5.0, threshold=0.65):
    """Run biased wav2vec2 ONNX model on audio, return read_ratio and mean P(read)."""
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception:
        return {"wav2vec2_read_ratio": 0, "wav2vec2_mean_p_read": 0, "wav2vec2_max_p_read": 0}

    window = int(window_sec * sr)
    if len(audio) < window:
        return {"wav2vec2_read_ratio": 0, "wav2vec2_mean_p_read": 0, "wav2vec2_max_p_read": 0}

    p_reads = []
    for start in range(0, len(audio) - window + 1, window):
        chunk = audio[start:start+window].astype(np.float32)
        logits = ort_session.run(None, {"input_values": chunk.reshape(1, -1)})[0]
        p_read = 1.0 / (1.0 + np.exp(-logits.flatten()[0]))
        p_reads.append(p_read)

    if not p_reads:
        return {"wav2vec2_read_ratio": 0, "wav2vec2_mean_p_read": 0, "wav2vec2_max_p_read": 0}

    read_count = sum(1 for p in p_reads if p >= threshold)
    return {
        "wav2vec2_read_ratio": round(read_count / len(p_reads), 4),
        "wav2vec2_mean_p_read": round(float(np.mean(p_reads)), 4),
        "wav2vec2_max_p_read": round(float(np.max(p_reads)), 4),
    }


# ============================================================
# Main Pipeline
# ============================================================

def extract_all_features(transcripts_path, manifest_path, output_path,
                         onnx_model_path=None, skip_prosodic=False,
                         skip_sbert=False):
    """Extract all features for company data."""

    # Load transcripts
    with open(transcripts_path, encoding="utf-8") as f:
        transcripts = json.load(f)
    print(f"Loaded {len(transcripts)} transcripts")

    # Load manifest for labels
    manifest = pd.read_csv(manifest_path)
    label_map = dict(zip(manifest["filename"], manifest["label_int"]))
    label_str_map = dict(zip(manifest["filename"], manifest.get("label", manifest.get("label_raw", ""))))

    # Load ONNX model if provided
    ort_session = None
    if onnx_model_path and os.path.exists(onnx_model_path):
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_model_path,
                                            providers=["CPUExecutionProvider"])
        print(f"Loaded ONNX model: {onnx_model_path}")

    rows = []
    for filepath, t in tqdm(transcripts.items(), desc="Extracting features"):
        text = t.get("text", "")
        words = t.get("words", [])
        filename = t.get("filename", Path(filepath).name)

        # Text features
        text_feats = compute_text_features(text)

        # Pause features
        pause_feats = compute_pause_features(words) if words else _empty_pause_features()

        # Prosodic features
        if skip_prosodic:
            prosodic_feats = _empty_prosodic_features()
        else:
            prosodic_feats = compute_prosodic_features(filepath) if os.path.exists(filepath) else _empty_prosodic_features()

        # Sentence transformer embeddings
        if skip_sbert:
            sbert_feats = {f"sbert_{i}": 0.0 for i in range(SBERT_DIM)}
        else:
            sbert_feats = compute_text_embeddings(text)

        # wav2vec2 scores
        wav2vec2_feats = {"wav2vec2_read_ratio": 0, "wav2vec2_mean_p_read": 0, "wav2vec2_max_p_read": 0}
        if ort_session and os.path.exists(filepath):
            wav2vec2_feats = compute_wav2vec2_scores(filepath, ort_session)

        row = {
            "filepath": filepath,
            "filename": filename,
            "label": label_str_map.get(filename, ""),
            "label_int": label_map.get(filename, -1),
            "duration_sec": t.get("duration_sec", 0),
            "n_transcript_words": len(words),
            "text": text[:200],
        }
        row.update(text_feats)
        row.update(pause_feats)
        row.update(prosodic_feats)
        row.update(sbert_feats)
        row.update(wav2vec2_feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nExtracted {len(df)} rows -> {output_path}")

    # Print summary
    if "label_int" in df.columns:
        for label_val, label_name in [(1, "cheating"), (0, "not cheating")]:
            sub = df[df["label_int"] == label_val]
            if len(sub) == 0:
                continue
            print(f"\n  [{label_name}] n={len(sub)}")
            for col in ["filler_rate", "ttr", "hedge_rate", "pause_before_content_ratio",
                         "wav2vec2_read_ratio", "wav2vec2_mean_p_read"]:
                if col in sub.columns and sub[col].sum() > 0:
                    print(f"    {col}: mean={sub[col].mean():.4f}, std={sub[col].std():.4f}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Extract features for company data")
    parser.add_argument("--transcripts", required=True, help="WhisperX transcripts JSON")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with labels")
    parser.add_argument("--onnx-model", type=str, default=None,
                        help="Path to biased_wav2vec2_quant.onnx for audio scores")
    parser.add_argument("--output", type=str, default="features.csv")
    parser.add_argument("--skip-prosodic", action="store_true",
                        help="Skip prosodic features (faster)")
    parser.add_argument("--skip-sbert", action="store_true",
                        help="Skip sentence transformer embeddings")
    args = parser.parse_args()

    extract_all_features(args.transcripts, args.manifest, args.output,
                         args.onnx_model, args.skip_prosodic, args.skip_sbert)


if __name__ == "__main__":
    main()
