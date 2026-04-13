"""
Biased Wav2Vec2 — Single-Neuron Read vs Spontaneous Classifier
================================================================
Single sigmoid output: P(read). Only classify as "reading" if confidence
exceeds a high threshold (default 0.65), so spontaneous speech is the
default assumption and only high-confidence segments get flagged.

Usage:
    python train_biased_wav2vec2.py
"""

import os
import sys
import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import Wav2Vec2Model, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# Add old/ to path so we can reuse the data pipeline
sys.path.insert(0, str(Path(__file__).parent / "old"))
from old.src.data.dataset import SpeechWindowDataset, build_accent_sampler


# ── Config ──────────────────────────────────────────────────────────────
READ_THRESHOLD = 0.65  # Only classify as "read" if P(read) > this
CONFIG_PATH = Path("old/configs/config.yaml")
CHECKPOINT_DIR = Path("checkpoints_biased")


class BiasedSpeechClassifier(nn.Module):
    """
    Wav2Vec2 encoder + single-neuron classification head.

    Output is a single logit → sigmoid → P(read).
    Spontaneous = 0, Read = 1.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        hidden_size: int = 256,
        dropout: float = 0.3,
        freeze_layers: int = 6,
    ):
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size  # 768

        # Freeze CNN feature extractor
        self.encoder.feature_extractor._freeze_parameters()

        # Freeze first N transformer layers
        if freeze_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Single-neuron classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # Single output
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Model: {model_name}")
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        print(f"  Frozen layers: {freeze_layers}, Hidden: {hidden_size}")
        print(f"  Output: 1 neuron (sigmoid), Read threshold: {READ_THRESHOLD}")

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, time, 768)

        if attention_mask is not None:
            mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            hidden_states = hidden_states * mask.unsqueeze(-1)
            pooled = hidden_states.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        logit = self.classifier(pooled)  # (batch, 1)
        return logit.squeeze(-1)  # (batch,)

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
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
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_values": input_values, "labels": labels}


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device).float()  # BCEWithLogitsLoss needs float

        optimizer.zero_grad()

        if scaler is not None:
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
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= READ_THRESHOLD).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().astype(int))

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=READ_THRESHOLD):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device).float()

        logits = model(input_values)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().astype(int))
        all_probs.extend(probs)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train():
    # Load config from old pipeline
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    w2v_cfg = cfg["training"]["wav2vec2"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")

    # Load manifest (lives under old/outputs/)
    data_root = Path(cfg["paths"]["data_root"])
    manifest_path = data_root / "old" / cfg["paths"]["manifest_csv"]
    manifest_df = pd.read_csv(manifest_path)
    print(f"Manifest: {len(manifest_df)} files")

    # Filter out rows whose audio files don't exist on this machine
    # (e.g. GigaSpeech data that was only on company laptop)
    exists_mask = manifest_df["filepath"].apply(os.path.exists)
    if not exists_mask.all():
        n_missing = (~exists_mask).sum()
        print(f"Dropping {n_missing} files not found on this machine")
        manifest_df = manifest_df[exists_mask].reset_index(drop=True)
        print(f"Remaining: {len(manifest_df)} files")

    # Datasets
    train_dataset = SpeechWindowDataset(manifest_df, cfg, split="train")
    val_dataset = SpeechWindowDataset(manifest_df, cfg, split="val")

    # Accent-weighted sampler
    accent_weights = cfg.get("accent_weights", {})
    train_sampler = build_accent_sampler(
        train_dataset.df, train_dataset.window_index, accent_weights
    )

    # Batch size — auto-reduce for low VRAM
    batch_size = w2v_cfg["batch_size"]
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 10:
            batch_size = min(batch_size, 8)
            print(f"Reduced batch to {batch_size} for {vram_gb:.0f}GB VRAM")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Model
    model = BiasedSpeechClassifier(
        model_name=w2v_cfg["model_name"],
        hidden_size=w2v_cfg["hidden_size"],
        dropout=w2v_cfg["dropout"],
        freeze_layers=w2v_cfg["freeze_layers"],
    ).to(device)

    # Class weights for loss — use pos_weight for BCEWithLogitsLoss
    train_labels = [train_dataset.df.iloc[fi]["label_int"] for fi, _ in train_dataset.window_index]
    n_spontaneous = sum(1 for l in train_labels if l == 0)
    n_read = sum(1 for l in train_labels if l == 1)
    # pos_weight = n_negative / n_positive (weight for the "read" class)
    pos_weight = torch.tensor([n_spontaneous / max(n_read, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Class balance: {n_spontaneous} spontaneous, {n_read} read")
    print(f"pos_weight (read): {pos_weight.item():.3f}")

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

    # Checkpoint dir
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    patience = w2v_cfg["patience"]
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"BIASED WAV2VEC2 TRAINING")
    print(f"Read threshold: {READ_THRESHOLD} (only flag if P(read) > {READ_THRESHOLD})")
    print(f"Epochs: {w2v_cfg['num_epochs']}, Batch: {batch_size}, LR: {w2v_cfg['learning_rate']}")
    print(f"Training steps: {num_training_steps}, Warmup: {num_warmup_steps}")
    print(f"{'='*60}")

    for epoch in range(1, w2v_cfg["num_epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{w2v_cfg['num_epochs']}")
        print(f"{'='*60}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler,
        )

        val_loss, val_acc, val_f1, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device,
        )

        print(f"\nTrain — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Show confidence distribution
        read_probs = val_probs[val_labels == 1]
        spont_probs = val_probs[val_labels == 0]
        if len(read_probs) > 0:
            print(f"  Read  P(read): mean={read_probs.mean():.3f}, "
                  f"median={np.median(read_probs):.3f}, >thresh={(read_probs >= READ_THRESHOLD).mean():.1%}")
        if len(spont_probs) > 0:
            print(f"  Spont P(read): mean={spont_probs.mean():.3f}, "
                  f"median={np.median(spont_probs):.3f}, >thresh={(spont_probs >= READ_THRESHOLD).mean():.1%}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "threshold": READ_THRESHOLD,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "read_threshold": READ_THRESHOLD,
                "config": w2v_cfg,
            }, CHECKPOINT_DIR / "biased_wav2vec2_best.pt")
            print(f"  -> New best model (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save history
    with open(CHECKPOINT_DIR / "biased_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final test evaluation ──
    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*60}")

    checkpoint = torch.load(CHECKPOINT_DIR / "biased_wav2vec2_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = SpeechWindowDataset(manifest_df, cfg, split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device,
    )

    print(f"\nTest — Acc: {test_acc:.4f}, F1: {test_f1:.4f} (threshold={READ_THRESHOLD})")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['spontaneous', 'read'])}")

    # Show how different thresholds would perform
    print("\n--- Threshold sensitivity ---")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        t_preds = (test_probs >= t).astype(int)
        t_acc = accuracy_score(test_labels, t_preds)
        t_f1 = f1_score(test_labels, t_preds, zero_division=0)
        marker = " <-- current" if abs(t - READ_THRESHOLD) < 0.01 else ""
        print(f"  threshold={t:.2f}: acc={t_acc:.4f}, f1={t_f1:.4f}{marker}")

    # Save results
    test_results = {
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "best_val_f1": round(best_f1, 4),
        "best_epoch": checkpoint["epoch"],
        "read_threshold": READ_THRESHOLD,
    }
    with open(CHECKPOINT_DIR / "biased_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nAll artifacts saved to: {CHECKPOINT_DIR}")
    return model


if __name__ == "__main__":
    train()
