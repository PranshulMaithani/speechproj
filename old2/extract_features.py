"""
Multi-Signal Feature Extraction for Cheating Detection
========================================================
Extracts text features, pause features, and prosodic features from
transcripts (CrisperWhisper JSON) or ground-truth transcripts (LibriSpeech/AMI).

Features are organized by signal type:
  1. Text features (disfluency, vocabulary complexity, spontaneity markers)
  2. Pause features (placement relative to POS, distribution stats)
  3. Prosodic features (F0, speaking rate, energy)

Usage:
    # From CrisperWhisper transcripts
    python extract_features.py --transcripts datasets/transcripts_allsstar.json --output datasets/features_allsstar.csv

    # From dataset with ground-truth text (no word timestamps — text features only)
    python extract_features.py --manifest datasets/librispeech/manifest.csv --output datasets/features_libri.csv --text-only
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm


# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
except OSError:
    print("Run: python -m spacy download en_core_web_sm")
    sys.exit(1)


# ============================================================
# Constants
# ============================================================

FILLERS = {"um", "uh", "uh-huh", "uhm", "umm", "hmm", "hm", "er", "ah", "ehm", "mhm"}
DISCOURSE_MARKERS = {"you know", "i mean", "like", "basically", "actually", "so",
                     "well", "right", "okay", "oh", "anyway", "honestly"}
HEDGES = {"i think", "i guess", "maybe", "perhaps", "probably", "kind of",
          "sort of", "i believe", "it seems", "i suppose", "might be"}
SELF_REF = {"i", "me", "my", "myself", "mine", "i'm", "i've", "i'd", "i'll"}

# Content POS tags (spaCy)
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
FUNCTION_POS = {"DET", "ADP", "CONJ", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}

# High-frequency words (top 2000 — simplified)
# Words NOT in this set are considered "rare"
# Using a rough approximation
COMMON_WORDS_FILE = None  # Will use word frequency heuristic


# ============================================================
# Text Features
# ============================================================

def compute_text_features(text: str) -> dict:
    """Extract text-level features from a transcript string."""
    if not text or len(text.strip()) < 10:
        return _empty_text_features()

    text_lower = text.lower().strip()
    doc = nlp(text_lower)

    words = [t.text for t in doc if t.is_alpha]
    all_tokens = [t.text for t in doc]
    n_words = len(words)

    if n_words < 5:
        return _empty_text_features()

    # --- Disfluency features ---
    filler_count = sum(1 for w in all_tokens if w in FILLERS)
    filler_rate = filler_count / n_words

    # Repetition: count repeated bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
    repetition_rate = repeated_bigrams / max(len(bigrams), 1)

    # Self-corrections: "I mean", "no wait", "sorry", "actually" as repair markers
    repair_markers = ["i mean", "no wait", "sorry i", "actually no", "wait no", "no no"]
    repair_count = sum(text_lower.count(m) for m in repair_markers)
    sentences = list(doc.sents)
    n_sents = max(len(sentences), 1)
    repair_rate = repair_count / n_sents

    # --- Vocabulary complexity ---
    unique_words = set(words)
    ttr = len(unique_words) / n_words  # Type-Token Ratio

    # MATTR (Moving Average TTR) with 50-word window
    mattr = _compute_mattr(words, window=50)

    # Complex words (3+ syllables)
    complex_count = sum(1 for w in words if _syllable_count(w) >= 3)
    complex_rate = complex_count / n_words

    # Average word length
    avg_word_len = np.mean([len(w) for w in words])

    # --- Sentence structure ---
    sent_lengths = [len([t for t in s if t.is_alpha]) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    std_sent_len = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    # Fragment rate (sentences with < 4 words)
    fragment_count = sum(1 for sl in sent_lengths if sl < 4)
    fragment_rate = fragment_count / n_sents

    # --- Spontaneity markers ---
    # Self-reference rate
    self_ref_count = sum(1 for w in all_tokens if w in SELF_REF)
    self_ref_rate = self_ref_count / n_words

    # Discourse markers
    dm_count = sum(text_lower.count(dm) for dm in DISCOURSE_MARKERS)
    dm_rate = dm_count / n_sents

    # Hedging
    hedge_count = sum(text_lower.count(h) for h in HEDGES)
    hedge_rate = hedge_count / n_sents

    # POS distribution
    pos_counts = Counter(t.pos_ for t in doc)
    total_pos = sum(pos_counts.values()) or 1
    noun_rate = pos_counts.get("NOUN", 0) / total_pos
    verb_rate = pos_counts.get("VERB", 0) / total_pos
    adj_rate = pos_counts.get("ADJ", 0) / total_pos

    return {
        # Disfluency
        "filler_rate": round(filler_rate, 4),
        "filler_count": filler_count,
        "repetition_rate": round(repetition_rate, 4),
        "repair_rate": round(repair_rate, 4),

        # Vocabulary complexity
        "ttr": round(ttr, 4),
        "mattr": round(mattr, 4),
        "complex_word_rate": round(complex_rate, 4),
        "avg_word_length": round(avg_word_len, 2),
        "n_words": n_words,
        "n_unique_words": len(unique_words),

        # Sentence structure
        "avg_sentence_length": round(avg_sent_len, 2),
        "std_sentence_length": round(std_sent_len, 2),
        "fragment_rate": round(fragment_rate, 4),
        "n_sentences": n_sents,

        # Spontaneity markers
        "self_ref_rate": round(self_ref_rate, 4),
        "discourse_marker_rate": round(dm_rate, 4),
        "hedge_rate": round(hedge_rate, 4),

        # POS distribution
        "noun_rate": round(noun_rate, 4),
        "verb_rate": round(verb_rate, 4),
        "adj_rate": round(adj_rate, 4),
    }


def _compute_mattr(words, window=50):
    """Moving Average Type-Token Ratio."""
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    ttrs = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i+window]
        ttrs.append(len(set(chunk)) / window)
    return np.mean(ttrs)


def _syllable_count(word):
    """Rough syllable count heuristic."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
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
# Pause Features (requires word timestamps)
# ============================================================

def compute_pause_features(words: list) -> dict:
    """
    Extract pause features from word-level timestamps.
    words: list of {"word": str, "start": float, "end": float}
    """
    if not words or len(words) < 5:
        return _empty_pause_features()

    # Compute inter-word gaps (pauses)
    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i-1]["end"]
        if gap > 0.05:  # Only count gaps > 50ms as pauses
            pauses.append({
                "duration": gap,
                "before_word": words[i]["word"],
                "after_word": words[i-1]["word"],
                "position": i,
            })

    if not pauses:
        return _empty_pause_features()

    durations = [p["duration"] for p in pauses]

    # --- Pause duration stats ---
    pause_mean = np.mean(durations)
    pause_std = np.std(durations)
    pause_median = np.median(durations)
    pause_skew = float(pd.Series(durations).skew()) if len(durations) > 2 else 0
    long_pause_rate = sum(1 for d in durations if d > 0.5) / len(durations)
    total_duration = words[-1]["end"] - words[0]["start"]
    pause_ratio = sum(durations) / max(total_duration, 0.1)

    # --- Pause regularity ---
    if len(durations) > 1:
        inter_pause_intervals = []
        pause_positions = [p["position"] for p in pauses]
        for i in range(1, len(pause_positions)):
            interval = pause_positions[i] - pause_positions[i-1]
            inter_pause_intervals.append(interval)
        pause_regularity = np.std(inter_pause_intervals) if inter_pause_intervals else 0
    else:
        pause_regularity = 0

    # --- Pause placement relative to POS ---
    # Get POS tags for words following pauses
    all_text = " ".join(w["word"] for w in words)
    doc = nlp(all_text)
    token_pos = {}
    for token in doc:
        token_pos[token.text.lower()] = token.pos_

    n_before_content = 0
    n_before_function = 0
    n_mid_phrase = 0

    for pause in pauses:
        word_after = pause["before_word"].lower().strip(".,!?;:")
        pos = token_pos.get(word_after, "X")

        if pos in CONTENT_POS:
            n_before_content += 1
        elif pos in FUNCTION_POS:
            n_before_function += 1

        # Mid-phrase = not at sentence boundary
        if not any(pause["after_word"].endswith(p) for p in [".", "!", "?", ","]):
            n_mid_phrase += 1

    n_pauses = len(pauses)
    pause_before_content_ratio = n_before_content / n_pauses
    pause_before_function_ratio = n_before_function / n_pauses
    mid_phrase_pause_rate = n_mid_phrase / n_pauses

    # Speaking rate features
    n_words_total = len(words)
    speaking_dur = total_duration - sum(durations)
    words_per_sec = n_words_total / max(total_duration, 0.1)
    articulation_rate = n_words_total / max(speaking_dur, 0.1)

    return {
        # Duration stats
        "pause_mean": round(pause_mean, 4),
        "pause_std": round(pause_std, 4),
        "pause_median": round(pause_median, 4),
        "pause_skew": round(pause_skew, 4),
        "long_pause_rate": round(long_pause_rate, 4),
        "pause_ratio": round(pause_ratio, 4),
        "n_pauses": n_pauses,
        "pause_regularity": round(pause_regularity, 4),

        # Placement
        "pause_before_content_ratio": round(pause_before_content_ratio, 4),
        "pause_before_function_ratio": round(pause_before_function_ratio, 4),
        "mid_phrase_pause_rate": round(mid_phrase_pause_rate, 4),

        # Speaking rate
        "words_per_sec": round(words_per_sec, 2),
        "articulation_rate": round(articulation_rate, 2),
    }


def _empty_pause_features():
    return {
        "pause_mean": 0, "pause_std": 0, "pause_median": 0, "pause_skew": 0,
        "long_pause_rate": 0, "pause_ratio": 0, "n_pauses": 0, "pause_regularity": 0,
        "pause_before_content_ratio": 0, "pause_before_function_ratio": 0,
        "mid_phrase_pause_rate": 0, "words_per_sec": 0, "articulation_rate": 0,
    }


# ============================================================
# Prosodic Features (from audio directly)
# ============================================================

def compute_prosodic_features(audio_path: str, sr: int = 16000) -> dict:
    """Extract F0 and energy features from audio."""
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=120)
    except Exception:
        return _empty_prosodic_features()

    if len(audio) < sr:
        return _empty_prosodic_features()

    # F0 extraction
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=75, fmax=500, sr=sr, frame_length=2048
    )

    # Filter to voiced frames
    f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if len(f0_voiced) < 10:
        return _empty_prosodic_features()

    f0_mean = float(np.mean(f0_voiced))
    f0_std = float(np.std(f0_voiced))
    f0_range = float(np.max(f0_voiced) - np.min(f0_voiced))
    f0_skew = float(pd.Series(f0_voiced).skew())

    # F0 declination (slope over time)
    x = np.arange(len(f0_voiced))
    if len(x) > 1:
        slope = np.polyfit(x, f0_voiced, 1)[0]
        f0_slope = float(slope)
    else:
        f0_slope = 0.0

    # Energy features
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # Speaking rate variability (using energy-based segmentation)
    # Compute in 2-second windows
    window = 2 * sr
    hop = sr
    rates = []
    for start in range(0, len(audio) - window, hop):
        chunk = audio[start:start+window]
        chunk_rms = librosa.feature.rms(y=chunk, frame_length=512, hop_length=256)[0]
        voiced_ratio = float((chunk_rms > np.percentile(rms, 20)).mean())
        rates.append(voiced_ratio)
    speaking_rate_std = float(np.std(rates)) if rates else 0

    return {
        "f0_mean": round(f0_mean, 2),
        "f0_std": round(f0_std, 2),
        "f0_range": round(f0_range, 2),
        "f0_skew": round(f0_skew, 4),
        "f0_slope": round(f0_slope, 6),
        "energy_mean": round(energy_mean, 6),
        "energy_std": round(energy_std, 6),
        "speaking_rate_std": round(speaking_rate_std, 4),
    }


def _empty_prosodic_features():
    return {
        "f0_mean": 0, "f0_std": 0, "f0_range": 0, "f0_skew": 0, "f0_slope": 0,
        "energy_mean": 0, "energy_std": 0, "speaking_rate_std": 0,
    }


# ============================================================
# Main Pipeline
# ============================================================

def extract_from_transcripts(transcript_path, output_path, include_prosody=True):
    """Extract features from CrisperWhisper transcript JSON."""
    with open(transcript_path, encoding="utf-8") as f:
        transcripts = json.load(f)

    rows = []
    for filepath, t in tqdm(transcripts.items(), desc="Extracting features"):
        text = t.get("text", "")
        words = t.get("words", [])

        # Text features
        text_feats = compute_text_features(text)

        # Pause features (only if we have word timestamps)
        pause_feats = compute_pause_features(words) if words else _empty_pause_features()

        # Prosodic features
        prosodic_feats = _empty_prosodic_features()
        if include_prosody and os.path.exists(filepath):
            prosodic_feats = compute_prosodic_features(filepath)

        row = {
            "filepath": filepath,
            "filename": t.get("filename", Path(filepath).name),
            "label": t.get("label", ""),
            "label_int": t.get("label_int", -1),
            "source": t.get("source", ""),
            "duration_sec": t.get("duration_sec", 0),
            "text": text[:200],  # Truncate for CSV
        }
        row.update(text_feats)
        row.update(pause_feats)
        row.update(prosodic_feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nExtracted {len(df)} rows -> {output_path}")
    _print_feature_summary(df)
    return df


def extract_from_manifest(manifest_path, output_path, text_only=False, max_files=None):
    """Extract features from a manifest CSV that has ground-truth text."""
    df = pd.read_csv(manifest_path)

    if "text" not in df.columns:
        print("ERROR: Manifest has no 'text' column")
        return None

    # Subsample if requested (stratified by label)
    if max_files and max_files < len(df) and "label_int" in df.columns:
        df = df.groupby("label_int", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_files // df["label_int"].nunique()),
                               random_state=42)
        ).reset_index(drop=True)
        print(f"  Subsampled to {len(df)} files")

    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        text = str(row.get("text", ""))

        # Text features
        text_feats = compute_text_features(text)

        # No word timestamps from ground truth — skip pause features
        pause_feats = _empty_pause_features()

        # Prosodic features (if audio exists and not text-only mode)
        prosodic_feats = _empty_prosodic_features()
        filepath = str(row.get("filepath", ""))
        if not text_only and filepath and os.path.exists(filepath):
            prosodic_feats = compute_prosodic_features(filepath)

        out = {
            "filepath": filepath,
            "filename": row.get("filename", ""),
            "label": row.get("label", ""),
            "label_int": int(row.get("label_int", -1)),
            "source": row.get("source", ""),
            "duration_sec": row.get("duration_sec", 0),
            "text": text[:200],
        }
        out.update(text_feats)
        out.update(pause_feats)
        out.update(prosodic_feats)
        rows.append(out)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"\nExtracted {len(result_df)} rows -> {output_path}")
    _print_feature_summary(result_df)
    return result_df


def _print_feature_summary(df):
    """Print quick stats by label."""
    if "label" not in df.columns:
        return
    for label in df["label"].unique():
        sub = df[df["label"] == label]
        print(f"\n  [{label}] n={len(sub)}")
        for col in ["filler_rate", "ttr", "mattr", "complex_word_rate",
                     "self_ref_rate", "hedge_rate", "pause_before_content_ratio"]:
            if col in sub.columns and sub[col].sum() > 0:
                print(f"    {col}: mean={sub[col].mean():.4f}, std={sub[col].std():.4f}")


import os

def main():
    parser = argparse.ArgumentParser(description="Extract multi-signal features")
    parser.add_argument("--transcripts", type=str, help="CrisperWhisper JSON")
    parser.add_argument("--manifest", type=str, help="Manifest CSV with text column")
    parser.add_argument("--output", type=str, default="datasets/features.csv")
    parser.add_argument("--text-only", action="store_true", help="Skip prosodic features")
    parser.add_argument("--no-prosody", action="store_true", help="Skip prosodic features")
    parser.add_argument("--max-files", type=int, default=None, help="Max files to process (stratified subsample)")
    args = parser.parse_args()

    if args.transcripts:
        extract_from_transcripts(args.transcripts, args.output,
                                  include_prosody=not args.no_prosody)
    elif args.manifest:
        extract_from_manifest(args.manifest, args.output, text_only=args.text_only,
                              max_files=args.max_files)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
