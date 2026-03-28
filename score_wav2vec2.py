"""
Run wav2vec2 on audio files and add scores to feature CSVs.
============================================================
Uses PyTorch GPU model for scoring.

Usage:
    python score_wav2vec2.py --features datasets/features_allsstar.csv --model checkpoints_biased/biased_wav2vec2_best.pt
"""

import argparse
import os
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm


class BiasedWav2Vec2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x):
        outputs = self.encoder(x).last_hidden_state
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled)


def score_file(filepath, model, device, sr=16000, window_sec=5.0, threshold=0.65):
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    window = int(window_sec * sr)
    if len(audio) < window:
        return 0.0, 0.0, 0.0

    p_reads = []
    for start in range(0, len(audio) - window + 1, window):
        chunk = torch.tensor(audio[start:start+window], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(chunk)
        p_read = torch.sigmoid(logits).item()
        p_reads.append(p_read)

    if not p_reads:
        return 0.0, 0.0, 0.0

    read_count = sum(1 for p in p_reads if p >= threshold)
    return (
        round(read_count / len(p_reads), 4),
        round(float(np.mean(p_reads)), 4),
        round(float(np.max(p_reads)), 4),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--model", default="checkpoints_combined/wav2vec2_combined_best.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = BiasedWav2Vec2().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model: {args.model}")

    for feat_path in args.features:
        df = pd.read_csv(feat_path)
        if "filepath" not in df.columns:
            print(f"  SKIP {feat_path}: no filepath column")
            continue

        has_audio = df["filepath"].apply(os.path.exists)
        print(f"\n{feat_path}: {len(df)} rows, {has_audio.sum()} with audio on disk")

        if has_audio.sum() == 0:
            df["wav2vec2_read_ratio"] = 0
            df["wav2vec2_mean_p_read"] = 0
            df["wav2vec2_max_p_read"] = 0
            df.to_csv(feat_path, index=False)
            print(f"  No audio files found, wrote zeros")
            continue

        ratios, means, maxes = [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {feat_path}"):
            fp = row["filepath"]
            if os.path.exists(fp):
                r, m, mx = score_file(fp, model, device)
            else:
                r, m, mx = 0.0, 0.0, 0.0
            ratios.append(r)
            means.append(m)
            maxes.append(mx)

        df["wav2vec2_read_ratio"] = ratios
        df["wav2vec2_mean_p_read"] = means
        df["wav2vec2_max_p_read"] = maxes
        df.to_csv(feat_path, index=False)
        print(f"  Saved with wav2vec2 scores")


if __name__ == "__main__":
    main()
