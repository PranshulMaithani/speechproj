"""
Build manifest CSV from company audio folder structure.
========================================================
Scans audio folders (audios2/, audios4/, etc.) and joins with ground truth labels.

Folder structure expected:
    root/
        audios2/
            candidateid/
                candidateid_25.wav
                candidateid_26.wav
        audios4/
            candidateid/
                candidateid_25.wav

Ground truth CSV (gtlabels):
    filename,label
    candidateid_25.wav,cheating
    candidateid_26.wav,not cheating

Usage:
    python build_manifest.py --audio-root . --labels gtlabels.csv --output manifest.csv
"""

import argparse
import os
from pathlib import Path
import pandas as pd


def scan_audio_folders(root_dir):
    """Scan audios*/ folders for WAV files."""
    root = Path(root_dir)
    audio_dirs = sorted(root.glob("audios*"))

    if not audio_dirs:
        print(f"WARNING: No audios*/ folders found in {root}")
        # Try scanning root directly
        audio_dirs = [root]

    files = []
    for audio_dir in audio_dirs:
        for wav_path in sorted(audio_dir.rglob("*.wav")):
            files.append({
                "filepath": str(wav_path.resolve()),
                "filename": wav_path.name,
                "candidate_folder": wav_path.parent.name,
                "audio_batch": audio_dir.name,
            })

    print(f"Found {len(files)} audio files across {len(audio_dirs)} folders")
    for d in audio_dirs:
        count = sum(1 for f in files if f["audio_batch"] == d.name)
        print(f"  {d.name}: {count} files")

    return pd.DataFrame(files)


def load_labels(labels_path):
    """Load ground truth labels CSV."""
    df = pd.read_csv(labels_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find the filename and label columns
    fname_col = None
    label_col = None
    for c in df.columns:
        if c in ("filename", "file", "name", "audio"):
            fname_col = c
        if c in ("label", "labels", "class", "ground_truth", "gt"):
            label_col = c

    if fname_col is None or label_col is None:
        print(f"Columns found: {list(df.columns)}")
        raise ValueError("Could not find filename and label columns. "
                         "Expected 'filename' and 'label' columns.")

    labels = df[[fname_col, label_col]].copy()
    labels.columns = ["filename", "label_raw"]

    # Normalize labels: cheating -> 1, not cheating -> 0
    labels["label"] = labels["label_raw"].str.strip().str.lower().map(
        lambda x: "cheating" if "cheat" in x and "not" not in x else "not cheating"
    )
    labels["label_int"] = labels["label"].map({"cheating": 1, "not cheating": 0})

    print(f"\nLabels: {len(labels)} entries")
    print(f"  Cheating:     {(labels['label_int']==1).sum()}")
    print(f"  Not cheating: {(labels['label_int']==0).sum()}")

    return labels


def build_manifest(audio_root, labels_path, output_path):
    """Build manifest by joining audio files with labels."""
    files_df = scan_audio_folders(audio_root)
    labels_df = load_labels(labels_path)

    # Join on filename — only keep files that exist in the labels CSV
    manifest = files_df.merge(labels_df, on="filename", how="inner")

    skipped = len(files_df) - len(manifest)
    print(f"\nManifest: {len(manifest)} labeled files (skipped {skipped} unlabeled)")

    manifest.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build manifest from company audio")
    parser.add_argument("--audio-root", type=str, default=".",
                        help="Root directory containing audios*/ folders")
    parser.add_argument("--labels", type=str, required=True,
                        help="Ground truth labels CSV")
    parser.add_argument("--output", type=str, default="manifest.csv")
    args = parser.parse_args()

    build_manifest(args.audio_root, args.labels, args.output)


if __name__ == "__main__":
    main()
