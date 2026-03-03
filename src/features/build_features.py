"""
Feature Extraction Runner
===========================
Reads the manifest CSV, windows every audio file, extracts prosodic features
per window, and saves a feature matrix for XGBoost/LightGBM training.

Usage:
    python -m src.features.build_features --config configs/config.yaml
"""

import os
import argparse
import traceback
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.audio_utils import load_audio, window_audio
from src.features.extract_features import extract_all_features


def process_one_file(
    row: pd.Series,
    cfg: dict,
) -> list[dict]:
    """Extract features from all valid windows of one audio file."""
    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]

    audio, sr = load_audio(
        row["filepath"],
        target_sr=audio_cfg["sample_rate"],
        max_duration=audio_cfg.get("max_duration_sec", 120),
    )

    windows = window_audio(
        audio, sr,
        window_sec=audio_cfg["window_sec"],
        hop_sec=audio_cfg["hop_sec"],
        min_speech_ratio=audio_cfg["min_speech_ratio"],
        energy_threshold=audio_cfg["vad_energy_threshold"],
    )

    results = []
    for i, win in enumerate(windows):
        if not win.is_valid:
            continue

        feats = extract_all_features(
            win.audio, sr,
            f0_min=feat_cfg["f0_min"],
            f0_max=feat_cfg["f0_max"],
            n_mfcc=feat_cfg["n_mfcc"],
            energy_threshold=audio_cfg["vad_energy_threshold"],
        )

        # Add metadata
        feats["source_file"] = row["filepath"]
        feats["speaker_id"] = row["speaker_id"]
        feats["l1"] = row["l1"]
        feats["gender"] = row["gender"]
        feats["task"] = row["task"]
        feats["label"] = row["label"]
        feats["label_int"] = row["label_int"]
        feats["split"] = row["split"]
        feats["window_idx"] = i
        feats["window_start_sec"] = win.start_sec
        feats["window_end_sec"] = win.end_sec
        feats["speech_ratio"] = win.speech_ratio

        results.append(feats)

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract features from audio windows")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--workers", type=int, default=1, help="Not used (sequential for stability)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load manifest
    data_root = Path(cfg["paths"]["data_root"])
    manifest_path = data_root / cfg["paths"]["manifest_csv"]
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(df)} files")

    # Process each file
    all_features = []
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            file_feats = process_one_file(row, cfg)
            all_features.extend(file_feats)
        except Exception as e:
            errors.append({"file": row["filepath"], "error": str(e)})
            if len(errors) <= 5:
                traceback.print_exc()

    print(f"\nExtracted {len(all_features)} feature vectors from {len(df)} files")
    if errors:
        print(f"  {len(errors)} files failed")

    # Save feature matrix
    feat_df = pd.DataFrame(all_features)
    out_dir = data_root / cfg["paths"]["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "window_features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"Features saved to: {out_path}")

    # Print summary
    meta_cols = [
        "source_file", "speaker_id", "l1", "gender", "task",
        "label", "label_int", "split", "window_idx",
        "window_start_sec", "window_end_sec", "speech_ratio",
    ]
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Window counts by label: {feat_df['label'].value_counts().to_dict()}")
    print(f"Window counts by split: {feat_df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
