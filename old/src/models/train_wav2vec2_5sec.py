"""
Wav2Vec2 Fine-Tuning for Read vs Spontaneous Speech Classification
=====================================================================
Fine-tunes a pre-trained Wav2Vec2 (or WavLM) encoder with a classification
head on 5-second audio windows.

Designed for:
  - Local GPU (8GB VRAM) for small-batch training / inference
  - Kaggle TPU for full training runs

Usage:
    python -m src.models.train_wav2vec2 --config configs/config.yaml
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from src.data.dataset import SpeechWindowDataset, build_accent_sampler


class SpeechClassifier(nn.Module):
    """
    Wav2Vec2/WavLM encoder + classification head for binary classification.

    Architecture:
      - Pre-trained Wav2Vec2 encoder (optionally partially frozen)
      - Mean pooling over time dimension
      - Linear(768, hidden) → ReLU → Dropout → Linear(hidden, 2)
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        hidden_size: int = 256,
        num_labels: int = 2,
        dropout: float = 0.3,
        freeze_layers: int = 6,
    ):
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size  # 768 for base

        # Freeze feature extractor (CNN) always
        self.encoder.feature_extractor._freeze_parameters()

        # Freeze first N transformer layers
        if freeze_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Model: {model_name}")
        print(f"  Trainable: {trainable:,} / {total:,} params "
              f"({100*trainable/total:.1f}%)")
        print(f"  Frozen layers: {freeze_layers}, Hidden: {hidden_size}")

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            input_values: (batch, seq_len) raw waveform at 16kHz
            attention_mask: (batch, seq_len) optional

        Returns:
            logits: (batch, num_labels)
        """
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, time, 768)

        # Mean pooling over time
        if attention_mask is not None:
            # Create mask for hidden states (different length due to CNN)
            mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            hidden_states = hidden_states * mask.unsqueeze(-1)
            pooled = hidden_states.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)  # (batch, 768)

        logits = self.classifier(pooled)  # (batch, 2)
        return logits

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        """Downsample attention mask to match encoder output length."""
        output_lengths = self.encoder._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).to(torch.long)
        batch_size = attention_mask.shape[0]
        mask = torch.zeros(
            batch_size, feature_vector_length,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        for i in range(batch_size):
            mask[i, : output_lengths[i]] = 1
        return mask


def collate_fn(batch):
    """Custom collator: pad waveforms and stack labels."""
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_values": input_values, "labels": labels}


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler=None):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision
            with torch.cuda.amp.autocast():
                logits = model(input_values)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_values)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Run evaluation."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_values)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train(cfg: dict):
    """Full training loop."""
    data_root = Path(cfg["paths"]["data_root"])
    w2v_cfg = cfg["training"]["wav2vec2"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load manifest
    manifest_path = data_root / cfg["paths"]["manifest_csv"]
    manifest_df = pd.read_csv(manifest_path)

    # Create datasets
    train_dataset = SpeechWindowDataset(manifest_df, cfg, split="train")
    val_dataset = SpeechWindowDataset(manifest_df, cfg, split="val")

    # Accent-weighted sampler
    accent_weights = cfg.get("accent_weights", {})
    train_sampler = build_accent_sampler(
        train_dataset.df, train_dataset.window_index, accent_weights
    )

    # DataLoaders
    batch_size = w2v_cfg["batch_size"]
    # Reduce batch size for 8GB VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 10:
            batch_size = min(batch_size, 8)
            print(f"Reduced batch size to {batch_size} for {vram_gb:.0f}GB VRAM")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Model
    model = SpeechClassifier(
        model_name=w2v_cfg["model_name"],
        hidden_size=w2v_cfg["hidden_size"],
        num_labels=2,
        dropout=w2v_cfg["dropout"],
        freeze_layers=w2v_cfg["freeze_layers"],
    ).to(device)

    # Class weights for loss
    train_labels = [train_dataset.df.iloc[fi]["label_int"] for fi, _ in train_dataset.window_index]
    label_counts = pd.Series(train_labels).value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_labels) / (2 * c) for c in label_counts],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights.cpu().tolist()}")

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=w2v_cfg["learning_rate"],
        weight_decay=w2v_cfg["weight_decay"],
    )

    # Scheduler
    num_training_steps = len(train_loader) * w2v_cfg["num_epochs"]
    num_warmup_steps = int(num_training_steps * w2v_cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps,
    )

    # Mixed precision
    use_fp16 = w2v_cfg.get("fp16", True) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # Training loop
    ckpt_dir = data_root / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    patience = w2v_cfg["patience"]
    patience_counter = 0
    history = []

    print(f"\nStarting training: {w2v_cfg['num_epochs']} epochs, "
          f"batch_size={batch_size}, lr={w2v_cfg['learning_rate']}")
    print(f"Training steps: {num_training_steps}, Warmup: {num_warmup_steps}")

    for epoch in range(1, w2v_cfg["num_epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{w2v_cfg['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler,
        )

        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device,
        )

        print(f"\nTrain — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "config": w2v_cfg,
            }, ckpt_dir / "wav2vec2_best.pt")
            print(f"  ✓ New best model (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save training history
    with open(ckpt_dir / "wav2vec2_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(ckpt_dir / "wav2vec2_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = SpeechWindowDataset(manifest_df, cfg, split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device,
    )

    print(f"\nTest — Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['spontaneous', 'read'])}")

    # Save test results
    test_results = {
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "best_val_f1": round(best_f1, 4),
        "best_epoch": checkpoint["epoch"],
    }
    with open(ckpt_dir / "wav2vec2_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nAll artifacts saved to: {ckpt_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
