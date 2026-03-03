"""
Data Audit & Manifest Generation
=================================
Walks the ALLSSTAR dataset (2676=spontaneous, 2677=read), catalogs every
.wav and .TextGrid, extracts metadata from filenames, measures audio
duration, and produces a master manifest CSV + summary statistics.

Usage:
    python -m src.data.build_manifest --config configs/config.yaml
"""

import os
import re
import argparse
import hashlib
from pathlib import Path
from collections import Counter

import yaml
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# ---------- filename parser ----------
# Pattern: ALL_{ID}_{GENDER}_{L1}_{L2}_{TASK}.wav
FILENAME_RE = re.compile(
    r"^ALL_(\d{3})_([FM])_([A-Z]{2,4})_([A-Z]{2,4})_([A-Z0-9]{2,4})\.(wav|TextGrid)$",
    re.IGNORECASE,
)


def parse_filename(fname: str) -> dict | None:
    """Extract metadata fields from an ALLSSTAR filename."""
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    return {
        "speaker_id": m.group(1),
        "gender": m.group(2),
        "l1": m.group(3).upper(),
        "l2": m.group(4).upper(),
        "task": m.group(5).upper(),
        "ext": m.group(6).lower(),
    }


def get_wav_duration(path: str) -> float:
    """Return duration in seconds; -1 on error."""
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return -1.0


def build_manifest(cfg: dict) -> pd.DataFrame:
    """Scan data directories and build the manifest DataFrame."""
    data_root = Path(cfg["paths"]["data_root"])
    spont_dir = data_root / cfg["paths"]["spontaneous_dir"]
    read_dir = data_root / cfg["paths"]["read_dir"]

    records = []
    errors = []

    for label, top_dir in [("spontaneous", spont_dir), ("read", read_dir)]:
        if not top_dir.exists():
            print(f"WARNING: {top_dir} does not exist, skipping")
            continue

        subdirs = sorted([d for d in top_dir.iterdir() if d.is_dir()])
        for subdir in tqdm(subdirs, desc=f"Scanning {label}"):
            wav_files = sorted(subdir.glob("*.wav"))
            tg_files = {f.stem for f in subdir.glob("*.TextGrid")}

            for wav_path in wav_files:
                meta = parse_filename(wav_path.name)
                if meta is None:
                    errors.append(str(wav_path))
                    continue

                duration = get_wav_duration(str(wav_path))
                has_textgrid = meta["speaker_id"] in tg_files or wav_path.stem in tg_files

                records.append(
                    {
                        "filepath": str(wav_path),
                        "filename": wav_path.name,
                        "speaker_id": meta["speaker_id"],
                        "gender": meta["gender"],
                        "l1": meta["l1"],
                        "l2": meta["l2"],
                        "task": meta["task"],
                        "label": label,
                        "label_int": 0 if label == "spontaneous" else 1,
                        "has_textgrid": has_textgrid,
                        "textgrid_path": str(wav_path.with_suffix(".TextGrid")) if has_textgrid else "",
                        "duration_sec": round(duration, 3),
                        "subfolder": subdir.name,
                        "parent_dir": top_dir.name,
                    }
                )

    if errors:
        print(f"\n⚠  {len(errors)} files could not be parsed:")
        for e in errors[:10]:
            print(f"   {e}")

    df = pd.DataFrame(records)
    return df


def speaker_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split by speaker ID, stratified by L1, so all files from one speaker
    stay in the same split.  Adds a 'split' column.
    """
    rng = np.random.RandomState(seed)

    # Get unique speakers with their L1
    speaker_info = df.groupby("speaker_id").agg({"l1": "first"}).reset_index()

    # Stratified split by L1
    splits = {}
    for l1, group in speaker_info.groupby("l1"):
        speakers = group["speaker_id"].tolist()
        rng.shuffle(speakers)
        n = len(speakers)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0
        # Ensure at least 1 in test if enough speakers
        n_test = n - n_train - n_val

        if n_test < 0:
            n_val = n - n_train
            n_test = 0
        if n <= 1:
            # Only 1 speaker: put in train
            n_train, n_val, n_test = n, 0, 0

        for spk in speakers[:n_train]:
            splits[spk] = "train"
        for spk in speakers[n_train : n_train + n_val]:
            splits[spk] = "val"
        for spk in speakers[n_train + n_val :]:
            splits[spk] = "test"

    df = df.copy()
    df["split"] = df["speaker_id"].map(splits)
    return df


def print_summary(df: pd.DataFrame):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("ALLSSTAR DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal files: {len(df)}")
    print(f"Total speakers: {df['speaker_id'].nunique()}")
    print(f"Total duration: {df['duration_sec'].sum() / 3600:.1f} hours")

    print(f"\n--- By Label ---")
    for label, grp in df.groupby("label"):
        print(
            f"  {label:12s}: {len(grp):4d} files, "
            f"{grp['speaker_id'].nunique():3d} speakers, "
            f"{grp['duration_sec'].sum() / 60:.1f} min"
        )

    print(f"\n--- By L1 (accent) ---")
    l1_stats = (
        df.groupby("l1")
        .agg(
            files=("filepath", "count"),
            speakers=("speaker_id", "nunique"),
            duration_min=("duration_sec", lambda x: round(x.sum() / 60, 1)),
        )
        .sort_values("files", ascending=False)
    )
    print(l1_stats.to_string())

    print(f"\n--- By Task ---")
    task_stats = (
        df.groupby("task")
        .agg(
            files=("filepath", "count"),
            duration_min=("duration_sec", lambda x: round(x.sum() / 60, 1)),
        )
        .sort_values("files", ascending=False)
    )
    print(task_stats.to_string())

    if "split" in df.columns:
        print(f"\n--- By Split ---")
        for split, grp in df.groupby("split"):
            print(
                f"  {split:5s}: {len(grp):4d} files, "
                f"{grp['speaker_id'].nunique():3d} speakers, "
                f"L1s: {sorted(grp['l1'].unique())}"
            )

    print(f"\nTextGrid coverage: {df['has_textgrid'].sum()}/{len(df)} "
          f"({100*df['has_textgrid'].mean():.1f}%)")
    print(f"Duration range: {df['duration_sec'].min():.1f}s - {df['duration_sec'].max():.1f}s")
    print(f"Mean duration: {df['duration_sec'].mean():.1f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build ALLSSTAR data manifest")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Building manifest...")
    df = build_manifest(cfg)

    if df.empty:
        print("ERROR: No files found. Check your config paths.")
        return

    # Apply speaker-stratified splits
    split_cfg = cfg.get("splits", {})
    df = speaker_stratified_split(
        df,
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        seed=split_cfg.get("random_seed", 42),
    )

    # Save manifest
    out_path = Path(cfg["paths"]["data_root"]) / cfg["paths"]["manifest_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nManifest saved to: {out_path}")

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
