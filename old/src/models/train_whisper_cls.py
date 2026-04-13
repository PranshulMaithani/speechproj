"""
Whisper Medium Fine-Tuning for Read vs Spontaneous Speech Classification
=========================================================================
Uses Whisper encoder's last hidden layer as features + MLP classification head.

Architecture:
    - Whisper Medium encoder (frozen CNN + partially frozen transformer layers)
    - Last hidden state mean-pooled over time → (batch, 1024)
    - MLP: Linear(1024, 512) → GELU → Dropout → Linear(512, 256) → GELU → Dropout → Linear(256, 2)

Key difference from Wav2Vec2:
    - Whisper takes log-mel spectrograms (80 mel bins), NOT raw waveforms
    - Feature extraction handled by WhisperFeatureExtractor
    - Encoder dim is 1024 for medium (vs 768 for wav2vec2-base)

Usage:
    python -m src.models.train_whisper_cls --config configs/config.yaml

Output:
    checkpoints/whisper_medium_best.pt
    checkpoints/whisper_medium_history.json
    checkpoints/whisper_medium_results.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import (
    WhisperModel,
    WhisperFeatureExtractor,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

import librosa


# ============================================================
# Logger
# ============================================================

LOGGER = logging.getLogger("train_whisper_cls")


def setup_logger(log_path: Path) -> logging.Logger:
    logger = LOGGER
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ============================================================
# Model
# ============================================================

class WhisperClassifier(nn.Module):
    """
    Whisper Medium encoder + MLP classification head.

    Whisper encoder produces frame-level embeddings of dim 1024.
    We mean-pool over the time dimension then pass through an MLP.

    Architecture:
        WhisperEncoder → mean pool → Linear(1024,512) → GELU → Dropout
                      → Linear(512,256) → GELU → Dropout → Linear(256,2)
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-medium",
        hidden_size: int = 512,
        num_labels: int = 2,
        dropout: float = 0.3,
        freeze_layers: int = 6,
    ):
        super().__init__()

        self.whisper = WhisperModel.from_pretrained(model_name)
        enc_dim = self.whisper.config.d_model   # 1024 for medium

        # Freeze CNN always
        for param in self.whisper.encoder.conv1.parameters():
            param.requires_grad = False
        for param in self.whisper.encoder.conv2.parameters():
            param.requires_grad = False
        for param in self.whisper.encoder.embed_positions.parameters():
            param.requires_grad = False

        # Freeze ALL encoder transformer layers — MLP head only
        for layer in self.whisper.encoder.layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze encoder layer norm too
        for param in self.whisper.encoder.layer_norm.parameters():
            param.requires_grad = False

        # Freeze decoder entirely
        for param in self.whisper.decoder.parameters():
            param.requires_grad = False

        # MLP head — two hidden layers
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_labels),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        LOGGER.info(f"Model: {model_name}")
        LOGGER.info(f"  Encoder dim:  {enc_dim}")
        LOGGER.info(f"  Trainable:    {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        LOGGER.info(f"  Frozen encoder layers: {freeze_layers}")

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (batch, 80, time_frames) log-mel spectrogram
                            time_frames = window_sec * 100 (100 frames/sec for Whisper)

        Returns:
            logits: (batch, num_labels)
        """
        # Run encoder only — decoder not needed for classification
        encoder_out = self.whisper.encoder(input_features)
        hidden      = encoder_out.last_hidden_state     # (batch, time, 1024)

        # Mean pool over time dimension
        pooled  = hidden.mean(dim=1)                    # (batch, 1024)
        logits  = self.classifier(pooled)               # (batch, 2)
        return logits


# ============================================================
# Dataset
# ============================================================

class WhisperSpeechDataset(Dataset):
    """
    Dataset that yields (log_mel_spectrogram, label) pairs.

    Each item is one 5-second window from an audio file.
    Windows computed on-the-fly from manifest.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        cfg: dict,
        feature_extractor: WhisperFeatureExtractor,
        split: str = "train",
        max_windows_per_file: int = 50,
    ):
        self.cfg              = cfg
        self.audio_cfg        = cfg["audio"]
        self.feature_extractor = feature_extractor
        self.split            = split
        self.max_windows      = max_windows_per_file

        self.sr          = self.audio_cfg["sample_rate"]
        self.window_sec  = self.audio_cfg["window_sec"]
        self.hop_sec     = self.audio_cfg["hop_sec"]
        self.window_samp = int(self.window_sec * self.sr)
        self.hop_samp    = int(self.hop_sec    * self.sr)

        self.df = manifest_df[
            manifest_df["split"] == split
        ].reset_index(drop=True)
        LOGGER.info(f"[{split}] {len(self.df)} files")

        self._build_window_index()

    def _build_window_index(self):
        self.window_index = []
        for idx, row in self.df.iterrows():
            try:
                duration = float(row["duration"])
                if duration != duration:   # nan check
                    duration = self.window_sec
            except Exception:
                duration = self.window_sec

            total_samp = int(min(duration, self.audio_cfg.get("max_duration_sec", 120)) * self.sr)
            start      = 0
            count      = 0
            while start < total_samp and count < self.max_windows:
                self.window_index.append((idx, start))
                start += self.hop_samp
                count += 1

        LOGGER.info(f"  [{self.split}] {len(self.window_index)} windows indexed")

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        file_idx, start_samp = self.window_index[idx]
        row = self.df.iloc[file_idx]

        # Load audio with librosa
        offset_sec = start_samp / self.sr
        try:
            audio, _ = librosa.load(
                row["filepath"],
                sr=self.sr,
                mono=True,
                offset=offset_sec,
                duration=self.window_sec,
            )
        except Exception:
            audio = np.zeros(self.window_samp, dtype=np.float32)

        # Pad if needed
        if len(audio) < self.window_samp:
            audio = np.pad(audio, (0, self.window_samp - len(audio)))

        # Extract Whisper log-mel spectrogram
        # WhisperFeatureExtractor returns (1, 80, time_frames)
        features = self.feature_extractor(
            audio,
            sampling_rate=self.sr,
            return_tensors="pt",
        ).input_features.squeeze(0)   # (80, time_frames)

        label = int(row["label_int"])

        return {
            "input_features": features,
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# Collator
# ============================================================

def collate_fn(batch):
    input_features = torch.stack([item["input_features"] for item in batch])
    labels         = torch.stack([item["labels"]          for item in batch])
    return {"input_features": input_features, "labels": labels}


# ============================================================
# Training utilities
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, scaler=None):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        feats  = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(feats)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(feats)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        feats  = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        logits = model(feats)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ============================================================
# Accent sampler (same logic as dataset.py)
# ============================================================

def build_accent_sampler(manifest_df, window_index, accent_weights):
    weights = []
    for file_idx, _ in window_index:
        row = manifest_df.iloc[file_idx]
        l1  = row.get("l1", "unknown") if hasattr(row, "get") else "unknown"
        w   = accent_weights.get(str(l1), 1.0)
        weights.append(w)
    wt = torch.DoubleTensor(weights)
    return WeightedRandomSampler(wt, num_samples=len(wt), replacement=True)


# ============================================================
# Main training loop
# ============================================================

def train(cfg: dict):
    data_root  = Path(cfg["paths"]["data_root"])
    w_cfg      = cfg["training"]["whisper"]
    ckpt_dir   = data_root / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(ckpt_dir / "whisper_medium_train.log")

    best_ckpt_path  = ckpt_dir / "whisper_medium_best.pt"
    history_path    = ckpt_dir / "whisper_medium_history.json"
    results_path    = ckpt_dir / "whisper_medium_results.json"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU:  {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Load manifest
    manifest_path = data_root / cfg["paths"]["manifest_csv"]
    manifest_df   = pd.read_csv(manifest_path)
    logger.info(f"Manifest: {manifest_path}  ({len(manifest_df)} rows)")

    # Feature extractor
    model_name = w_cfg["model_name"]
    logger.info(f"Loading feature extractor: {model_name}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    # Datasets
    train_dataset = WhisperSpeechDataset(manifest_df, cfg, feature_extractor, split="train")
    val_dataset   = WhisperSpeechDataset(manifest_df, cfg, feature_extractor, split="val")

    # Batch size — Whisper medium needs more VRAM than Wav2Vec2
    batch_size = w_cfg.get("batch_size", 4)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 10:
            batch_size = min(batch_size, 16)
            logger.info(f"Batch size capped at {batch_size} for {vram_gb:.0f}GB VRAM")
        if vram_gb < 6:
            batch_size = min(batch_size, 2)

    # Accent sampler
    accent_weights = cfg.get("accent_weights", {})
    train_sampler  = build_accent_sampler(
        train_dataset.df, train_dataset.window_index, accent_weights
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Model
    model = WhisperClassifier(
        model_name=model_name,
        hidden_size=w_cfg.get("hidden_size", 512),
        num_labels=2,
        dropout=w_cfg.get("dropout", 0.3),
        freeze_layers=w_cfg.get("freeze_layers", 6),
    ).to(device)

    # Class weights
    train_labels  = [train_dataset.df.iloc[fi]["label_int"] for fi, _ in train_dataset.window_index]
    label_counts  = pd.Series(train_labels).value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_labels) / (2 * c) for c in label_counts],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Class weights: {class_weights.cpu().tolist()}")

    # Optimizer — higher LR than WavLM since Whisper encoder is being used differently
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=w_cfg.get("learning_rate", 1e-4),
        weight_decay=w_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    num_epochs         = w_cfg.get("num_epochs", 10)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps   = int(num_training_steps * w_cfg.get("warmup_ratio", 0.10))
    scheduler          = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps,
    )

    # Mixed precision
    use_fp16 = w_cfg.get("fp16", True) and torch.cuda.is_available()
    scaler   = torch.amp.GradScaler("cuda") if use_fp16 else None
    logger.info(f"Mixed precision: {use_fp16}")

    patience = w_cfg.get("patience", 3)
    best_f1  = 0.0
    patience_counter = 0
    history  = []

    logger.info(f"Starting training: {num_epochs} epochs  "
                f"batch={batch_size}  lr={w_cfg.get('learning_rate', 1e-4)}")
    logger.info(f"Steps: {num_training_steps}  Warmup: {num_warmup_steps}")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"{'='*60}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*60}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler,
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device,
        )

        logger.info(f"Train — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f}")
        logger.info(f"Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss":   val_loss,   "val_acc":   val_acc,   "val_f1":   val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":           val_f1,
                "val_acc":          val_acc,
                "config":           w_cfg,
            }, best_ckpt_path)
            logger.info(f"  New best saved (F1={val_f1:.4f}): {best_ckpt_path.name}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Test evaluation
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)

    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = WhisperSpeechDataset(manifest_df, cfg, feature_extractor, split="test")
    test_loader  = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    test_loss, test_acc, test_f1, test_preds, test_labels, _ = evaluate(
        model, test_loader, criterion, device,
    )

    logger.info(f"Test — Acc: {test_acc:.4f}  F1: {test_f1:.4f}")
    logger.info(
        "\n" + classification_report(test_labels, test_preds,
                                     target_names=["spontaneous", "read"])
    )

    results = {
        "test_accuracy": round(test_acc, 4),
        "test_f1":       round(test_f1, 4),
        "best_val_f1":   round(best_f1, 4),
        "best_epoch":    checkpoint["epoch"],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"All artifacts saved to: {ckpt_dir}")
    return model


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Whisper Medium classifier for read vs spontaneous speech"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Validate whisper config section exists
    if "whisper" not in cfg.get("training", {}):
        raise ValueError(
            "Missing 'whisper' section under 'training' in config.yaml.\n"
            "Add this to your config:\n\n"
            "  training:\n"
            "    whisper:\n"
            "      model_name: \"openai/whisper-medium\"\n"
            "      freeze_layers: 6\n"
            "      hidden_size: 512\n"
            "      dropout: 0.3\n"
            "      batch_size: 4\n"
            "      learning_rate: 0.0001\n"
            "      warmup_ratio: 0.10\n"
            "      num_epochs: 10\n"
            "      patience: 3\n"
            "      weight_decay: 0.01\n"
            "      fp16: true\n"
        )

    train(cfg)


if __name__ == "__main__":
    main()