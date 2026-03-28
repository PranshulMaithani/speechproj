"""
Download LibriSpeech (read speech) + AMI (spontaneous speech) datasets.
Also processes ALLSSTAR data already on disk.

Saves audio files to datasets/ directory with a unified manifest.

Usage:
    python download_datasets.py
"""

import os
import json
from pathlib import Path
from datasets import load_dataset, Audio
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("datasets")
MANIFEST_PATH = DATA_DIR / "manifest.csv"
SR = 16000


def download_librispeech():
    """Download LibriSpeech clean-100 (read speech, ~6.3GB)."""
    out_dir = DATA_DIR / "librispeech"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    marker = out_dir / ".done"
    if marker.exists():
        print("LibriSpeech already downloaded, loading manifest...")
        return pd.read_csv(out_dir / "manifest.csv")

    print("Downloading LibriSpeech clean-100 (~6.3GB)...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100",
                       trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))

    print(f"Processing {len(ds)} LibriSpeech utterances...")
    for i, item in enumerate(tqdm(ds, desc="LibriSpeech")):
        audio = item["audio"]["array"]
        duration = len(audio) / SR

        # Skip very short or very long
        if duration < 3.0 or duration > 30.0:
            continue

        fname = f"libri_{i:06d}.wav"
        fpath = out_dir / fname
        if not fpath.exists():
            sf.write(str(fpath), audio.astype(np.float32), SR)

        manifest_rows.append({
            "filepath": str(fpath.resolve()),
            "filename": fname,
            "source": "librispeech",
            "label": "read",
            "label_int": 1,
            "duration_sec": round(duration, 2),
            "text": item.get("text", ""),
            "speaker_id": item.get("speaker_id", ""),
        })

    df = pd.DataFrame(manifest_rows)
    df.to_csv(out_dir / "manifest.csv", index=False)
    marker.touch()
    print(f"LibriSpeech: {len(df)} utterances saved")
    return df


def download_ami():
    """Download AMI Meeting Corpus headset-single (spontaneous, ~7GB)."""
    out_dir = DATA_DIR / "ami"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    marker = out_dir / ".done"
    if marker.exists():
        print("AMI already downloaded, loading manifest...")
        return pd.read_csv(out_dir / "manifest.csv")

    print("Downloading AMI Meeting Corpus (~7GB)...")
    ds = load_dataset("edinburghcstr/ami", "headset-single", split="train",
                       trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))

    print(f"Processing {len(ds)} AMI utterances...")
    for i, item in enumerate(tqdm(ds, desc="AMI")):
        audio = item["audio"]["array"]
        duration = len(audio) / SR

        # Skip very short or very long
        if duration < 3.0 or duration > 30.0:
            continue

        fname = f"ami_{i:06d}.wav"
        fpath = out_dir / fname
        if not fpath.exists():
            sf.write(str(fpath), audio.astype(np.float32), SR)

        manifest_rows.append({
            "filepath": str(fpath.resolve()),
            "filename": fname,
            "source": "ami",
            "label": "spontaneous",
            "label_int": 0,
            "duration_sec": round(duration, 2),
            "text": item.get("text", ""),
            "speaker_id": item.get("speaker_id", ""),
        })

    df = pd.DataFrame(manifest_rows)
    df.to_csv(out_dir / "manifest.csv", index=False)
    marker.touch()
    print(f"AMI: {len(df)} utterances saved")
    return df


def load_allsstar():
    """Load existing ALLSSTAR manifest."""
    manifest_path = Path("old/outputs/manifest_expanded.csv")
    if not manifest_path.exists():
        print("ALLSSTAR manifest not found!")
        return pd.DataFrame()

    df = pd.read_csv(manifest_path)
    # Filter to files that exist
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    # Normalize columns
    df["source"] = "allsstar"
    df["text"] = ""  # Will be filled by CrisperWhisper
    print(f"ALLSSTAR: {len(df)} files loaded")
    return df[["filepath", "filename", "source", "label", "label_int",
               "duration_sec", "text", "speaker_id", "split"]].copy()


def build_unified_manifest(dfs):
    """Combine all datasets into one manifest with train/val/test splits."""
    from sklearn.model_selection import train_test_split

    all_dfs = []
    for df in dfs:
        if "split" not in df.columns:
            df["split"] = ""
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # For rows without splits (LibriSpeech, AMI), assign 70/15/15
    needs_split = combined["split"] == ""
    if needs_split.any():
        idx = combined[needs_split].index
        train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        combined.loc[train_idx, "split"] = "train"
        combined.loc[val_idx, "split"] = "val"
        combined.loc[test_idx, "split"] = "test"

    combined.to_csv(MANIFEST_PATH, index=False)
    print(f"\nUnified manifest: {len(combined)} total")
    print(f"  Train: {(combined['split']=='train').sum()}")
    print(f"  Val:   {(combined['split']=='val').sum()}")
    print(f"  Test:  {(combined['split']=='test').sum()}")
    print(f"  Read:  {(combined['label_int']==1).sum()}")
    print(f"  Spont: {(combined['label_int']==0).sum()}")
    print(f"Saved to: {MANIFEST_PATH}")
    return combined


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load ALLSSTAR (already on disk)
    allsstar_df = load_allsstar()

    # Download external datasets
    libri_df = download_librispeech()
    ami_df = download_ami()

    # Build unified manifest
    build_unified_manifest([allsstar_df, libri_df, ami_df])
