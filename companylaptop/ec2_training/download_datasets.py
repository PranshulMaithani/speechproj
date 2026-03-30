"""
Download LibriSpeech + AMI for wav2vec2 training + build balanced manifest.
===========================================================================
For Casual Conversations, use download_casual_conversations.py separately.

Balancing strategy:
  - Casual Conversations has BOTH scripted (read) and unscripted (spontaneous)
  - Use equal number of scripted and unscripted from Casual Conversations
  - Cap LibriSpeech so total read ≈ total spontaneous

Usage:
    python download_datasets.py
    python download_datasets.py --skip-libri
    python download_datasets.py --skip-ami
    python download_datasets.py --manifest-only   # just rebuild unified manifest
"""

import argparse
import csv
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


SR = 16000  # Target sample rate


# ── LibriSpeech ──────────────────────────────────────────────
LIBRI_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRI_DIR = Path("datasets/librispeech")


class _DownloadProgress:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="LibriSpeech")
        self.pbar.update(block_size)


def download_librispeech():
    """Download LibriSpeech train-clean-100 (~6.3 GB)."""
    LIBRI_DIR.mkdir(parents=True, exist_ok=True)
    audio_root = LIBRI_DIR / "LibriSpeech" / "train-clean-100"

    manifest_path = LIBRI_DIR / "manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        existing = df["filepath"].apply(os.path.exists).sum()
        if existing > 100:
            print(f"LibriSpeech already downloaded ({existing} files). Skipping.")
            return

    tar_path = LIBRI_DIR / "train-clean-100.tar.gz"

    if not tar_path.exists() and not audio_root.exists():
        print(f"Downloading LibriSpeech train-clean-100 (~6.3 GB)...")
        urllib.request.urlretrieve(LIBRI_URL, str(tar_path), _DownloadProgress())
        print()

    if not audio_root.exists() and tar_path.exists():
        print(f"Extracting {tar_path}...")
        with tarfile.open(str(tar_path), "r:gz") as tar:
            tar.extractall(str(LIBRI_DIR))
        tar_path.unlink()
        print("Extracted and removed tar.gz")

    rows = []
    for trans_file in sorted(audio_root.rglob("*.trans.txt")):
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                utt_id, text = parts
                flac_path = trans_file.parent / f"{utt_id}.flac"
                if flac_path.exists():
                    rows.append({
                        "filepath": str(flac_path.resolve()),
                        "filename": flac_path.name,
                        "source": "librispeech",
                        "label": "read",
                        "label_int": 1,
                        "text": text,
                        "speaker_id": utt_id.split("-")[0],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    print(f"LibriSpeech: {len(df)} files -> {manifest_path}")
    return df


# ── AMI ──────────────────────────────────────────────────────
AMI_DIR = Path("datasets/ami")


def download_ami():
    """Download AMI Meeting Corpus via HuggingFace (~7 GB)."""
    from datasets import load_dataset, Audio

    AMI_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = AMI_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    manifest_path = AMI_DIR / "manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        existing = df["filepath"].apply(os.path.exists).sum()
        if existing > 100:
            print(f"AMI already downloaded ({existing} files). Skipping.")
            return

    print("Downloading AMI corpus (ihm)...")
    ds = load_dataset("edinburghcstr/ami", "ihm", split="train", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))

    rows = []
    for i in tqdm(range(len(ds)), desc="Processing AMI"):
        try:
            item = ds[i]
            audio = item["audio"]["array"]
            duration = len(audio) / SR

            if duration < 3.0 or duration > 60.0:
                continue

            fname = f"ami_{i:06d}.wav"
            fpath = audio_dir / fname
            if not fpath.exists():
                sf.write(str(fpath), audio.astype(np.float32), SR)

            rows.append({
                "filepath": str(fpath.resolve()),
                "filename": fname,
                "source": "ami",
                "label": "spontaneous",
                "label_int": 0,
                "duration_sec": round(duration, 2),
                "text": str(item.get("text", "")),
                "speaker_id": str(item.get("speaker_id", "")),
            })
        except Exception as e:
            if i < 5:
                print(f"  Skipping {i}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    print(f"AMI: {len(df)} files -> {manifest_path}")
    return df


# ── Casual Conversations ─────────────────────────────────────
CASUAL_DIR = Path("datasets/casual_conversations")


# ── Build unified manifest (balanced) ────────────────────────
def build_unified_manifest():
    """
    Combine all datasets into one balanced manifest.

    Balancing logic:
      1. Casual Conversations: use EQUAL scripted and unscripted
      2. Total spontaneous = AMI + Casual unscripted + ALLSSTAR spontaneous
      3. Total read = LibriSpeech + Casual scripted + ALLSSTAR read
      4. Cap LibriSpeech so total read ≈ total spontaneous
    """
    from sklearn.model_selection import train_test_split

    # ── Load all sources ─────────────────────────────────────
    sources = {}

    # ALLSSTAR
    allsstar_manifest = Path("datasets/allsstar/manifest_local.csv")
    if allsstar_manifest.exists():
        df = pd.read_csv(allsstar_manifest)
        df = df[df["filepath"].apply(os.path.exists)]
        df = df[["filepath", "filename", "label", "label_int"]].copy()
        df["source"] = "allsstar"
        sources["allsstar"] = df
        print(f"ALLSSTAR:    {len(df)} ({(df.label_int==1).sum()} read, {(df.label_int==0).sum()} spont)")
    else:
        print("ALLSSTAR: not found (run download_allsstar.py first)")

    # LibriSpeech
    libri_manifest = LIBRI_DIR / "manifest.csv"
    if libri_manifest.exists():
        df = pd.read_csv(libri_manifest)
        df = df[df["filepath"].apply(os.path.exists)]
        df = df[["filepath", "filename", "label", "label_int"]].copy()
        df["source"] = "librispeech"
        sources["librispeech"] = df
        print(f"LibriSpeech: {len(df)} (all read)")

    # AMI
    ami_manifest = AMI_DIR / "manifest.csv"
    if ami_manifest.exists():
        df = pd.read_csv(ami_manifest)
        df = df[df["filepath"].apply(os.path.exists)]
        df = df[["filepath", "filename", "label", "label_int"]].copy()
        df["source"] = "ami"
        sources["ami"] = df
        print(f"AMI:         {len(df)} (all spontaneous)")

    # Casual Conversations (has both scripted and unscripted)
    casual_manifest = CASUAL_DIR / "manifest.csv"
    casual_scripted = pd.DataFrame()
    casual_unscripted = pd.DataFrame()
    if casual_manifest.exists():
        df = pd.read_csv(casual_manifest)
        df = df[df["filepath"].apply(os.path.exists)]
        casual_scripted = df[df["label_int"] == 1][["filepath", "filename", "label", "label_int"]].copy()
        casual_scripted["source"] = "casual_scripted"
        casual_unscripted = df[df["label_int"] == 0][["filepath", "filename", "label", "label_int"]].copy()
        casual_unscripted["source"] = "casual_unscripted"
        print(f"Casual Conv: {len(df)} ({len(casual_scripted)} scripted, {len(casual_unscripted)} unscripted)")

    if not sources and casual_scripted.empty and casual_unscripted.empty:
        print("ERROR: No datasets found! Download at least one first.")
        sys.exit(1)

    # ── Balance: equal scripted and unscripted from Casual Conversations ──
    n_casual_each = min(len(casual_scripted), len(casual_unscripted))
    if n_casual_each > 0:
        casual_scripted = casual_scripted.sample(n=n_casual_each, random_state=42)
        casual_unscripted = casual_unscripted.sample(n=n_casual_each, random_state=42)
        print(f"\nBalanced Casual Conv: {n_casual_each} scripted + {n_casual_each} unscripted")

    # ── Count spontaneous (before balancing LibriSpeech) ─────
    n_spont = 0
    if "ami" in sources:
        n_spont += len(sources["ami"])
    n_spont += len(casual_unscripted)
    if "allsstar" in sources:
        n_spont += (sources["allsstar"]["label_int"] == 0).sum()

    n_read_other = len(casual_scripted)
    if "allsstar" in sources:
        n_read_other += (sources["allsstar"]["label_int"] == 1).sum()

    # Cap LibriSpeech so total read ≈ total spontaneous
    target_libri = max(0, n_spont - n_read_other)
    if "librispeech" in sources:
        libri = sources["librispeech"]
        if len(libri) > target_libri and target_libri > 0:
            libri = libri.sample(n=target_libri, random_state=42)
            sources["librispeech"] = libri
            print(f"Capped LibriSpeech to {target_libri} (to match spontaneous count)")
        elif target_libri == 0:
            # Still include some LibriSpeech for diversity
            cap = min(5000, len(libri))
            libri = libri.sample(n=cap, random_state=42)
            sources["librispeech"] = libri
            print(f"Capped LibriSpeech to {cap}")

    # ── Combine all ──────────────────────────────────────────
    all_dfs = list(sources.values())
    if not casual_scripted.empty:
        all_dfs.append(casual_scripted)
    if not casual_unscripted.empty:
        all_dfs.append(casual_unscripted)

    combined = pd.concat(all_dfs, ignore_index=True)

    total_read = (combined["label_int"] == 1).sum()
    total_spont = (combined["label_int"] == 0).sum()

    print(f"\n{'='*50}")
    print(f"FINAL COMBINED DATASET")
    print(f"{'='*50}")
    print(f"Total:       {len(combined)}")
    print(f"  Read:        {total_read}")
    print(f"  Spontaneous: {total_spont}")
    print(f"  Ratio:       {total_read/max(total_spont,1):.2f}")
    print(f"\nBy source:")
    for src in combined["source"].unique():
        sub = combined[combined["source"] == src]
        print(f"  {src:<20s}: {len(sub):>6d} ({(sub.label_int==1).sum()} read, {(sub.label_int==0).sum()} spont)")

    # ── Stratified train/val/test split ──────────────────────
    train_df, test_df = train_test_split(
        combined, test_size=0.15, random_state=42, stratify=combined["label_int"])
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=42, stratify=train_df["label_int"])

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    final = pd.concat([train_df, val_df, test_df], ignore_index=True)

    manifest_path = Path("datasets/manifest_unified.csv")
    final.to_csv(manifest_path, index=False)

    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Saved: {manifest_path}")
    return final


def main():
    parser = argparse.ArgumentParser(description="Download datasets for wav2vec2 training")
    parser.add_argument("--skip-libri", action="store_true")
    parser.add_argument("--skip-ami", action="store_true")
    parser.add_argument("--manifest-only", action="store_true", help="Only build unified manifest")
    args = parser.parse_args()

    if not args.manifest_only:
        if not args.skip_libri:
            download_librispeech()
        if not args.skip_ami:
            download_ami()

    # Check if Casual Conversations has been processed
    casual_manifest = CASUAL_DIR / "manifest.csv"
    if not casual_manifest.exists():
        print("\n" + "="*60)
        print("NOTE: Casual Conversations not yet processed.")
        print("="*60)
        print("Run download_casual_conversations.py separately:")
        print("  python download_casual_conversations.py --links ccv2_links.txt")
        print("Then re-run: python download_datasets.py --manifest-only")
        print()

    build_unified_manifest()


if __name__ == "__main__":
    main()
