"""
PyTorch Dataset for Wav2Vec2 Fine-Tuning
==========================================
Loads audio windows on-the-fly, returning raw waveforms suitable for
Wav2Vec2/HuBERT feature extractors.

Supports accent-weighted sampling for overrepresenting target populations.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import yaml

from src.data.audio_utils import load_audio, window_audio


class SpeechWindowDataset(Dataset):
    """
    Dataset that yields (waveform, label) pairs from pre-windowed audio.

    Each item is one 5-second window from an audio file.
    Windows are computed on-the-fly from the manifest to save disk space.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        cfg: dict,
        split: str = "train",
        max_windows_per_file: int = 50,
    ):
        self.cfg = cfg
        self.audio_cfg = cfg["audio"]
        self.split = split
        self.max_windows_per_file = max_windows_per_file

        # Filter to split
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        print(f"[{split}] {len(self.df)} files")

        # Pre-compute window index (file_idx, window_idx) pairs
        self._build_window_index()

    def _build_window_index(self):
        """Build an index mapping global_idx → (file_row_idx, window_position)."""
        self.window_index = []  # List of (file_idx, window_start_sample)

        sr = self.audio_cfg["sample_rate"]
        window_samples = int(self.audio_cfg["window_sec"] * sr)
        hop_samples = int(self.audio_cfg["hop_sec"] * sr)

        for file_idx, row in self.df.iterrows():
            duration = row["duration_sec"]
            total_samples = int(duration * sr)

            start = 0
            win_count = 0
            while start < total_samples and win_count < self.max_windows_per_file:
                end = start + window_samples
                if end - start < window_samples // 2:
                    break
                self.window_index.append((file_idx, start))
                start += hop_samples
                win_count += 1

        print(f"  [{self.split}] {len(self.window_index)} windows indexed")

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        file_idx, start_sample = self.window_index[idx]
        row = self.df.iloc[file_idx] if file_idx < len(self.df) else self.df.loc[file_idx]

        sr = self.audio_cfg["sample_rate"]
        window_samples = int(self.audio_cfg["window_sec"] * sr)

        # Load audio
        audio, _ = load_audio(
            row["filepath"],
            target_sr=sr,
            max_duration=self.audio_cfg.get("max_duration_sec", 120),
        )

        # Extract window
        chunk = audio[start_sample : start_sample + window_samples]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode="constant")

        # Check if window has enough speech
        from src.data.audio_utils import compute_speech_ratio
        speech_ratio = compute_speech_ratio(chunk, sr, self.audio_cfg["vad_energy_threshold"])

        label = int(row["label_int"])

        return {
            "input_values": torch.tensor(chunk, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
            "speech_ratio": speech_ratio,
        }


class PrecomputedFeatureDataset(Dataset):
    """
    Dataset for XGBoost/LightGBM: loads pre-extracted feature vectors.
    """

    def __init__(self, features_df: pd.DataFrame, split: str = "train"):
        self.split = split
        df = features_df[features_df["split"] == split].reset_index(drop=True)

        # Separate features from metadata
        meta_cols = [
            "source_file", "speaker_id", "l1", "gender", "task",
            "label", "label_int", "split", "window_idx",
            "window_start_sec", "window_end_sec", "speech_ratio",
        ]
        feature_cols = [c for c in df.columns if c not in meta_cols]

        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df["label_int"].values.astype(np.int64)
        self.meta = df[meta_cols]
        self.feature_names = feature_cols

        print(f"[{split}] {len(self.X)} samples, {len(feature_cols)} features")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_accent_sampler(
    manifest_df: pd.DataFrame,
    window_index: list,
    accent_weights: dict,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that oversamples under-represented
    or target-population accents.
    """
    weights = []
    for file_idx, _ in window_index:
        row = manifest_df.iloc[file_idx] if file_idx < len(manifest_df) else manifest_df.loc[file_idx]
        l1 = row["l1"]
        w = accent_weights.get(l1, 1.0)
        weights.append(w)

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler
