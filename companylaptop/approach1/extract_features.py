"""
Step 2: Extract text + acoustic features from transcripts and audio.
Reads Whisper transcripts (from step 1) and raw audio, combines into a single CSV.

Usage:
    python -m approach1.extract_features
    python -m approach1.extract_features --skip-acoustic  # text features only (fast)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from approach1.config import (
    AUDIO_DIRS, GT_EXCEL, GT_SHEETS, TRANSCRIPTS_DIR, SPEECHTRY_ROOT,
    FEATURES_CSV, OUTPUT_DIR, LABEL_MAP, SAMPLE_RATE, MAX_DURATION_SEC,
)
from approach1.text_features import extract_text_features
from approach1.acoustic_features import extract_all_acoustic_features


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth labels from Excel file."""
    all_rows = []

    for source, cfg in GT_SHEETS.items():
        print(f"  Loading GT from sheet '{cfg['sheet']}' for {source}...")
        try:
            df = pd.read_excel(GT_EXCEL, sheet_name=cfg["sheet"])
        except Exception as e:
            print(f"    ERROR reading sheet '{cfg['sheet']}': {e}")
            continue

        # Normalize column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        fname_col = cfg["filename_col"]
        path_col = cfg["path_col"]
        label_col = cfg["label_col"]

        for col in [fname_col, label_col]:
            if col not in df.columns:
                print(f"    WARNING: column '{col}' not found. Available: {list(df.columns)}")
                continue

        for _, row in df.iterrows():
            filename = str(row.get(fname_col, "")).strip()
            filepath = str(row.get(path_col, "")).strip() if path_col in df.columns else ""
            raw_label = row.get(label_col)

            if pd.isna(raw_label) or str(raw_label).strip() == "":
                continue

            raw_label_clean = str(raw_label).strip()
            label = LABEL_MAP.get(raw_label_clean)
            if label is None:
                # Try numeric
                label = LABEL_MAP.get(raw_label)

            if label is None:
                print(f"    WARNING: unmapped label '{raw_label_clean}' for {filename}")
                continue

            all_rows.append({
                "filename": filename,
                "filepath": filepath,
                "label": label,
                "source": source,
            })

    gt = pd.DataFrame(all_rows)
    print(f"  Loaded {len(gt)} labeled files ({gt['label'].sum()} cheating, {(gt['label']==0).sum()} genuine)")
    return gt


def find_transcript(filename: str) -> Path | None:
    """Find transcript JSON file for a given audio filename."""
    # Transcript names: audios2__candidateid__candidateid_25.json
    # We search for any transcript containing this filename
    stem = Path(filename).stem
    for p in TRANSCRIPTS_DIR.glob(f"*__{stem}.json"):
        return p
    # Fallback: search all transcripts
    for p in TRANSCRIPTS_DIR.glob("*.json"):
        if p.name.startswith("_"):
            continue
        if stem in p.stem:
            return p
    return None


def find_audio_file(filename: str, filepath: str = "") -> Path | None:
    """Find the actual audio file on disk."""
    # Try filepath first (relative to speechtry root)
    if filepath:
        candidate = SPEECHTRY_ROOT / filepath
        if candidate.exists():
            return candidate

    # Search in audio dirs
    for root_name, root_dir in AUDIO_DIRS.items():
        matches = list(root_dir.rglob(filename))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract text + acoustic features")
    parser.add_argument("--skip-acoustic", action="store_true", help="Skip acoustic features (text only)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth labels...")
    gt = load_ground_truth()
    if gt.empty:
        print("ERROR: No labeled data found. Check your Excel file and config.py")
        return

    print(f"\nExtracting features for {len(gt)} files...")
    if args.skip_acoustic:
        print("  (Acoustic features SKIPPED)")

    rows = []
    errors = []
    t0 = time.time()

    for i, (_, row) in enumerate(gt.iterrows()):
        filename = row["filename"]
        filepath = row.get("filepath", "")
        label = row["label"]
        source = row["source"]

        # ── Load transcript ───────────────────────────────────────────
        transcript_path = find_transcript(filename)
        text_feats = {}
        transcript_text = ""
        duration_sec = None

        if transcript_path and transcript_path.exists():
            try:
                tdata = json.loads(transcript_path.read_text(encoding="utf-8"))
                transcript_text = tdata.get("text", "")
                # Estimate duration from last segment end time
                if tdata.get("segments"):
                    duration_sec = tdata["segments"][-1].get("end", None)
                text_feats = extract_text_features(transcript_text, duration_sec)
            except Exception as e:
                errors.append((filename, f"transcript error: {e}"))
        else:
            errors.append((filename, "transcript not found"))

        # ── Load audio & extract acoustic features ────────────────────
        acoustic_feats = {}
        if not args.skip_acoustic:
            audio_path = find_audio_file(filename, filepath)
            if audio_path and audio_path.exists():
                try:
                    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION_SEC)
                    # Peak normalize
                    peak = np.max(np.abs(audio))
                    if peak > 1e-6:
                        audio = audio / peak * 0.95
                    if duration_sec is None:
                        duration_sec = len(audio) / sr
                    acoustic_feats = extract_all_acoustic_features(audio, sr)
                except Exception as e:
                    errors.append((filename, f"audio error: {e}"))
            else:
                errors.append((filename, "audio file not found"))

        # ── Combine ───────────────────────────────────────────────────
        combined = {
            "filename": filename,
            "source": source,
            "label": label,
            "transcript": transcript_text[:500],  # truncated for CSV readability
            "duration_sec": duration_sec or 0.0,
        }
        combined.update({f"text__{k}": v for k, v in text_feats.items()})
        combined.update({f"acoustic__{k}": v for k, v in acoustic_feats.items()})
        rows.append(combined)

        if (i + 1) % 20 == 0 or i == len(gt) - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(gt)}] {elapsed:.0f}s elapsed")

    # ── Save ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {FEATURES_CSV}")
    print(f"  Text features: {sum(1 for c in df.columns if c.startswith('text__'))}")
    print(f"  Acoustic features: {sum(1 for c in df.columns if c.startswith('acoustic__'))}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for fname, err in errors[:10]:
            print(f"  {fname}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # ── Quick stats ───────────────────────────────────────────────────
    print(f"\nLabel distribution:")
    print(f"  Cheating (1): {(df['label']==1).sum()}")
    print(f"  Genuine  (0): {(df['label']==0).sum()}")


if __name__ == "__main__":
    main()
