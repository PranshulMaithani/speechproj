"""
Company Data Adaptation Pipeline
====================================
Handles the domain adaptation workflow for company interview recordings:

  1. Run trained model on unlabeled company data → pseudo-labels
  2. Sort by confidence → identify samples needing manual review
  3. Generate labeling suggestions for manual review
  4. Fine-tune model on manually labeled + high-confidence pseudo-labeled data
  5. Re-iterate

Usage:
    # Step 1: Generate pseudo-labels
    python -m src.inference.adapt_company --step pseudo_label --company-dir path/to/company/wavs

    # Step 2: After manual labeling, fine-tune
    python -m src.inference.adapt_company --step finetune --labels path/to/labels.csv
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import asdict

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.audio_utils import load_audio, window_audio
from src.inference.predict import (
    predict_file, load_wav2vec2_model, load_xgboost_model, format_report,
)
from src.models.train_wav2vec2 import SpeechClassifier, collate_fn, train_one_epoch, evaluate


# ============================================================
# Step 1: Pseudo-Labeling
# ============================================================

def generate_pseudo_labels(cfg: dict, company_dir: str, mode: str = "wav2vec2"):
    """
    Run trained model on company data, produce pseudo-labels sorted by confidence.
    """
    data_root = Path(cfg["paths"]["data_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model(s)
    wav2vec2_model = None
    xgb_model, xgb_scaler = None, None

    if mode in ("wav2vec2", "ensemble"):
        wav2vec2_model = load_wav2vec2_model(cfg, device)
    if mode in ("xgboost", "ensemble"):
        xgb_model, xgb_scaler = load_xgboost_model(cfg)

    # Find all wav files
    company_path = Path(company_dir)
    audio_files = sorted(company_path.glob("*.wav"))
    print(f"Found {len(audio_files)} company audio files in {company_path}")

    # Predict each file
    results = []
    for fpath in tqdm(audio_files, desc="Pseudo-labeling"):
        result = predict_file(
            str(fpath), cfg,
            wav2vec2_model=wav2vec2_model,
            xgb_model=xgb_model,
            xgb_scaler=xgb_scaler,
            device=device,
            mode=mode,
        )
        results.append({
            "filepath": str(fpath),
            "filename": fpath.name,
            "predicted_label": result.overall_label,
            "confidence": result.overall_confidence,
            "read_ratio": result.read_ratio,
            "duration_sec": result.duration_sec,
            "n_segments": len(result.segments),
            "n_windows": len(result.window_predictions),
        })

    # Create pseudo-label DataFrame
    pseudo_df = pd.DataFrame(results)
    pseudo_df = pseudo_df.sort_values("confidence", ascending=True)  # Low confidence first

    # Categorize by confidence
    adapt_cfg = cfg["training"]["adaptation"]
    high_thresh = adapt_cfg["pseudo_label_threshold"]
    low_thresh = cfg["inference"]["confidence_threshold_low"]

    pseudo_df["category"] = "medium"
    pseudo_df.loc[pseudo_df["confidence"] >= high_thresh, "category"] = "high_confidence"
    pseudo_df.loc[pseudo_df["confidence"] < low_thresh, "category"] = "needs_review"

    # Add manual_label column (empty — to be filled by human)
    pseudo_df["manual_label"] = ""

    # Save
    out_dir = data_root / "outputs" / "company_adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)

    pseudo_path = out_dir / "pseudo_labels.csv"
    pseudo_df.to_csv(pseudo_path, index=False)

    # Save full prediction results for reference
    full_results_path = out_dir / "full_predictions.json"
    all_full = []
    for fpath in tqdm(audio_files, desc="Full predictions"):
        result = predict_file(
            str(fpath), cfg,
            wav2vec2_model=wav2vec2_model,
            xgb_model=xgb_model,
            xgb_scaler=xgb_scaler,
            device=device,
            mode=mode,
        )
        all_full.append(asdict(result))
    with open(full_results_path, "w") as f:
        json.dump(all_full, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"PSEUDO-LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files: {len(pseudo_df)}")
    print(f"")
    print(f"By predicted label:")
    print(f"  Read:        {(pseudo_df['predicted_label'] == 'read').sum()}")
    print(f"  Spontaneous: {(pseudo_df['predicted_label'] == 'spontaneous').sum()}")
    print(f"")
    print(f"By confidence category:")
    print(f"  High confidence (>={high_thresh:.0%}): {(pseudo_df['category'] == 'high_confidence').sum()}")
    print(f"  Medium:       {(pseudo_df['category'] == 'medium').sum()}")
    print(f"  Needs review (<{low_thresh:.0%}):  {(pseudo_df['category'] == 'needs_review').sum()}")
    print(f"")
    print(f"Pseudo-labels saved to: {pseudo_path}")
    print(f"Full predictions saved to: {full_results_path}")
    print(f"")
    print(f"NEXT STEPS:")
    print(f"  1. Open {pseudo_path}")
    print(f"  2. Fill in the 'manual_label' column (read/spontaneous) for 'needs_review' rows")
    print(f"  3. Optionally verify some 'high_confidence' rows")
    print(f"  4. Run: python -m src.inference.adapt_company --step finetune --labels {pseudo_path}")

    return pseudo_df


# ============================================================
# Step 2: Domain Adaptation Fine-Tuning
# ============================================================

class CompanyDataset(Dataset):
    """Dataset for company audio with manual + pseudo labels."""

    def __init__(self, labels_df: pd.DataFrame, cfg: dict):
        self.df = labels_df.reset_index(drop=True)
        self.cfg = cfg
        self.audio_cfg = cfg["audio"]

        sr = self.audio_cfg["sample_rate"]
        self.window_samples = int(self.audio_cfg["window_sec"] * sr)
        self.hop_samples = int(self.audio_cfg["hop_sec"] * sr)

        # Build window index
        self.window_index = []
        for file_idx, row in self.df.iterrows():
            audio, _ = load_audio(
                row["filepath"],
                target_sr=sr,
                max_duration=self.audio_cfg.get("max_duration_sec", 120),
            )
            total_samples = len(audio)

            start = 0
            while start < total_samples:
                end = min(start + self.window_samples, total_samples)
                if end - start < self.window_samples // 2:
                    break
                self.window_index.append((file_idx, start))
                start += self.hop_samples

        print(f"Company dataset: {len(self.df)} files, {len(self.window_index)} windows")

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        file_idx, start_sample = self.window_index[idx]
        row = self.df.iloc[file_idx]

        sr = self.audio_cfg["sample_rate"]
        audio, _ = load_audio(row["filepath"], target_sr=sr)

        chunk = audio[start_sample : start_sample + self.window_samples]
        if len(chunk) < self.window_samples:
            chunk = np.pad(chunk, (0, self.window_samples - len(chunk)), mode="constant")

        # Label: use manual_label if available, else predicted_label
        label_str = row.get("manual_label", "") or row.get("predicted_label", "spontaneous")
        label = 1 if label_str == "read" else 0

        return {
            "input_values": torch.tensor(chunk, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
            "speech_ratio": 1.0,  # Placeholder
        }


def finetune_on_company_data(cfg: dict, labels_path: str):
    """
    Fine-tune the ALLSSTAR-trained Wav2Vec2 model on company data.

    Uses:
      - All high-confidence pseudo-labeled samples
      - All manually labeled samples
      - Low learning rate + heavy regularization to prevent catastrophic forgetting
    """
    data_root = Path(cfg["paths"]["data_root"])
    adapt_cfg = cfg["training"]["adaptation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    labels_df = pd.read_csv(labels_path)
    print(f"Loaded labels: {len(labels_df)} files")

    # Filter to usable samples
    # Use samples that have manual labels OR high-confidence pseudo-labels
    usable = labels_df[
        (labels_df["manual_label"].notna() & (labels_df["manual_label"] != "")) |
        (labels_df["category"] == "high_confidence")
    ].copy()

    # For high-confidence samples without manual label, use predicted label
    mask = usable["manual_label"].isna() | (usable["manual_label"] == "")
    usable.loc[mask, "manual_label"] = usable.loc[mask, "predicted_label"]

    print(f"Usable samples: {len(usable)}")
    print(f"  Manually labeled: {(labels_df['manual_label'].notna() & (labels_df['manual_label'] != '')).sum()}")
    print(f"  High-confidence pseudo-labeled: {mask.sum()}")
    print(f"  Label distribution: {usable['manual_label'].value_counts().to_dict()}")

    if len(usable) < 10:
        print("ERROR: Too few usable samples for fine-tuning. Label more data.")
        return

    # Split: 80% train, 20% val
    usable = usable.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(usable))
    train_df = usable.iloc[:split_idx]
    val_df = usable.iloc[split_idx:]

    # Create datasets
    train_dataset = CompanyDataset(train_df, cfg)
    val_dataset = CompanyDataset(val_df, cfg)

    batch_size = 4  # Small batch for 8GB VRAM
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Load pre-trained model
    model = load_wav2vec2_model(cfg, device)

    # Freeze more layers during adaptation
    freeze_layers = adapt_cfg["freeze_layers"]
    for i, layer in enumerate(model.encoder.encoder.layers):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing {freeze_layers} layers: {trainable:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=adapt_cfg["learning_rate"],
        weight_decay=0.01,
    )

    num_steps = len(train_loader) * adapt_cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * num_steps), num_steps,
    )

    # Train
    ckpt_dir = data_root / cfg["paths"]["checkpoints_dir"]
    best_f1 = 0.0

    print(f"\nFine-tuning: {adapt_cfg['num_epochs']} epochs, lr={adapt_cfg['learning_rate']}")

    for epoch in range(1, adapt_cfg["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{adapt_cfg['num_epochs']}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
        )

        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device,
        )

        print(f"Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, ckpt_dir / "wav2vec2_adapted.pt")
            print(f"  Saved adapted model (F1={val_f1:.4f})")

    print(f"\nAdaptation complete. Best val F1: {best_f1:.4f}")
    print(f"Adapted model: {ckpt_dir / 'wav2vec2_adapted.pt'}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Company data adaptation")
    parser.add_argument("--step", type=str, required=True,
                        choices=["pseudo_label", "finetune"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--company-dir", type=str, default=None,
                        help="Path to company .wav files (for pseudo_label step)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labeled CSV (for finetune step)")
    parser.add_argument("--mode", type=str, default="wav2vec2",
                        choices=["wav2vec2", "xgboost", "ensemble"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.step == "pseudo_label":
        if not args.company_dir:
            print("ERROR: --company-dir required for pseudo_label step")
            return
        generate_pseudo_labels(cfg, args.company_dir, mode=args.mode)

    elif args.step == "finetune":
        if not args.labels:
            print("ERROR: --labels required for finetune step")
            return
        finetune_on_company_data(cfg, args.labels)


if __name__ == "__main__":
    main()
