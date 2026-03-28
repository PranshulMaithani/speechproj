"""
Build manifest CSV from company ground truth labels.
=====================================================
The gtlabels CSV already contains filepath, filename, folder, and label.
This script just normalizes labels and adds the audio_batch column for splitting.

Expected CSV columns (flexible naming):
    filepath (or file_path, path)  — full path to the WAV file
    filename (or file, name)       — e.g. candidateid_25.wav
    folder (or audio_batch, batch) — e.g. audios2, audios4
    label (or labels, class, gt)   — cheating / not cheating

Usage:
    python build_manifest.py --labels gtlabels.csv --output manifest.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def load_labels(labels_path):
    """Load and normalize the ground truth labels CSV."""
    df = pd.read_csv(labels_path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map flexible column names to standard names
    col_map = {}
    for c in df.columns:
        if c in ("filepath", "file_path", "path"):
            col_map[c] = "filepath"
        elif c in ("filename", "file", "name", "audio"):
            col_map[c] = "filename"
        elif c in ("folder", "audio_batch", "batch", "source_folder"):
            col_map[c] = "audio_batch"
        elif c in ("label", "labels", "class", "ground_truth", "gt"):
            col_map[c] = "label_raw"

    df = df.rename(columns=col_map)

    # Derive filename from filepath if missing
    if "filename" not in df.columns and "filepath" in df.columns:
        df["filename"] = df["filepath"].apply(lambda x: Path(x).name)

    # Derive audio_batch from filepath if missing (e.g. .../audios2/candidate/file.wav → audios2)
    if "audio_batch" not in df.columns and "filepath" in df.columns:
        def get_batch(fp):
            parts = Path(fp).parts
            for p in parts:
                if p.startswith("audios"):
                    return p
            return "unknown"
        df["audio_batch"] = df["filepath"].apply(get_batch)

    # Normalize labels
    if "label_raw" in df.columns:
        df["label"] = df["label_raw"].str.strip().str.lower().map(
            lambda x: "cheating" if "cheat" in str(x) and "not" not in str(x) else "not cheating"
        )
        df["label_int"] = df["label"].map({"cheating": 1, "not cheating": 0})

    print(f"Loaded {len(df)} entries from {labels_path}")
    print(f"  Columns: {list(df.columns)}")
    if "audio_batch" in df.columns:
        print(f"  Batches: {df['audio_batch'].value_counts().to_dict()}")
    if "label_int" in df.columns:
        print(f"  Cheating:     {(df['label_int']==1).sum()}")
        print(f"  Not cheating: {(df['label_int']==0).sum()}")

    return df


def build_manifest(labels_path, output_path):
    """Build manifest from labels CSV."""
    manifest = load_labels(labels_path)
    manifest.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build manifest from company labels")
    parser.add_argument("--labels", type=str, required=True,
                        help="Ground truth labels CSV (with filepath, filename, folder, label)")
    parser.add_argument("--output", type=str, default="manifest.csv")
    args = parser.parse_args()

    build_manifest(args.labels, args.output)


if __name__ == "__main__":
    main()
