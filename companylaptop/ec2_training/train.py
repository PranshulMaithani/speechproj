"""
Train wav2vec2 classifier on combined datasets (EC2 GPU).
=========================================================
Trains BiasedSpeechClassifier on ALLSSTAR + LibriSpeech + AMI + Casual Conversations.

Usage:
    python train.py
    python train.py --epochs 12 --batch-size 16 --lr 1e-5
    python train.py --manifest data/manifest_unified.csv
"""

import argparse
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import Wav2Vec2Model, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────
SAMPLE_RATE = 16000
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)  # 80,000
READ_THRESHOLD = 0.65


# ── Dataset ─────────────────────────────────────────────────
class CombinedWindowDataset(Dataset):
    """Loads audio from manifest, creates 5-sec non-overlapping windows."""

    def __init__(self, manifest_df):
        self.sr = SAMPLE_RATE
        self.window = WINDOW_SAMPLES
        self.df = manifest_df.reset_index(drop=True)
        self.window_index = []

        print("Building window index...")
        skipped = 0
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
            fp = row["filepath"]
            if not os.path.exists(fp):
                skipped += 1
                continue
            try:
                duration = librosa.get_duration(path=fp)
                n_samples = int(duration * self.sr)
                n_windows = max(1, n_samples // self.window)
                for w in range(n_windows):
                    self.window_index.append((idx, w * self.window))
            except Exception:
                skipped += 1
                continue

        print(f"  {len(self.window_index)} windows from {len(self.df)} files (skipped {skipped})")

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, i):
        file_idx, start_sample = self.window_index[i]
        row = self.df.iloc[file_idx]

        offset_sec = start_sample / self.sr
        audio, _ = librosa.load(row["filepath"], sr=self.sr, mono=True,
                                offset=offset_sec, duration=WINDOW_SEC)

        if len(audio) < self.window:
            audio = np.pad(audio, (0, self.window - len(audio)))
        else:
            audio = audio[:self.window]

        return {
            "input_values": torch.tensor(audio, dtype=torch.float32),
            "labels": torch.tensor(row["label_int"], dtype=torch.long),
        }


# ── Model ───────────────────────────────────────────────────
class BiasedSpeechClassifier(nn.Module):
    """wav2vec2 encoder + classification head. Output: single logit → P(read)."""

    def __init__(self, model_name="facebook/wav2vec2-base", hidden_size=256,
                 dropout=0.3, freeze_layers=6):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size  # 768

        self.encoder.feature_extractor._freeze_parameters()

        if freeze_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Model: {model_name}")
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        print(f"  Frozen layers: {freeze_layers}, Hidden: {hidden_size}")

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask)
            hidden_states = hidden_states * mask.unsqueeze(-1)
            pooled = hidden_states.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        return self.classifier(pooled).squeeze(-1)

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        output_lengths = self.encoder._get_feat_extract_output_lengths(
            attention_mask.sum(-1)).to(torch.long)
        batch_size = attention_mask.shape[0]
        mask = torch.zeros(batch_size, feature_vector_length,
                           dtype=attention_mask.dtype, device=attention_mask.device)
        for i in range(batch_size):
            mask[i, :output_lengths[i]] = 1
        return mask


def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ── Training ────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training"):
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device).float()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits).detach().cpu().numpy() >= READ_THRESHOLD).astype(int)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy().astype(int))

    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, zero_division=0))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device).float()

        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend((probs >= READ_THRESHOLD).astype(int))
        all_labels.extend(y.cpu().numpy().astype(int))
        all_probs.extend(probs)

    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, zero_division=0),
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


# ── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train wav2vec2 on combined datasets")
    _script_dir = str(Path(__file__).resolve().parent)
    parser.add_argument("--manifest", type=str, default=os.path.join(_script_dir, "data", "manifest_unified.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--freeze-layers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-per-dataset", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name}, VRAM: {vram:.1f} GB")
        # Auto-adjust batch size for low VRAM
        if vram < 12 and args.batch_size > 4:
            print(f"  Low VRAM — reducing batch_size to 4")
            args.batch_size = 4

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    if os.path.exists(args.manifest):
        combined = pd.read_csv(args.manifest)
    else:
        print(f"Manifest not found: {args.manifest}")
        print("Run download_datasets.py first to build it.")
        return

    # Filter to existing files
    combined = combined[combined["filepath"].apply(os.path.exists)].reset_index(drop=True)
    combined = combined[combined["label_int"].isin([0, 1])].reset_index(drop=True)

    print(f"\nDataset: {len(combined)} files")
    print(f"  Read:        {(combined.label_int==1).sum()}")
    print(f"  Spontaneous: {(combined.label_int==0).sum()}")
    if "source" in combined.columns:
        print(f"\nBy source:")
        for src in combined["source"].unique():
            sub = combined[combined["source"] == src]
            print(f"  {src}: {len(sub)} ({(sub.label_int==1).sum()} read, {(sub.label_int==0).sum()} spont)")

    # Split — use existing splits if available
    if "split" in combined.columns and combined["split"].nunique() >= 3:
        train_df = combined[combined["split"] == "train"]
        val_df = combined[combined["split"] == "val"]
        test_df = combined[combined["split"] == "test"]
    else:
        train_df, test_df = train_test_split(
            combined, test_size=0.15, random_state=42, stratify=combined["label_int"])
        train_df, val_df = train_test_split(
            train_df, test_size=0.15, random_state=42, stratify=train_df["label_int"])

    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build datasets
    train_dataset = CombinedWindowDataset(train_df)
    val_dataset = CombinedWindowDataset(val_df)
    test_dataset = CombinedWindowDataset(test_df)

    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    # Model
    print(f"\nBuilding model (freeze_layers={args.freeze_layers})...")
    model = BiasedSpeechClassifier(freeze_layers=args.freeze_layers).to(device)

    # Resume from checkpoint
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"  Resumed at epoch {start_epoch}, val_f1={ck.get('val_f1', '?')}")

    # Loss with class balancing
    train_labels = [train_dataset.df.iloc[fi]["label_int"] for fi, _ in train_dataset.window_index]
    n_spont = sum(1 for l in train_labels if l == 0)
    n_read = sum(1 for l in train_labels if l == 1)
    pos_weight = torch.tensor([n_spont / max(n_read, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Class balance: {n_spont} spont, {n_read} read, pos_weight={pos_weight.item():.3f}")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=0.01)
    num_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * 0.1), num_steps)
    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0.0
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"TRAINING: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler)

        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device)

        elapsed = time.time() - epoch_start
        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  Time: {elapsed/60:.1f} min")

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_f1": round(train_f1, 4), "val_loss": round(val_loss, 4),
            "val_f1": round(val_f1, 4), "time_min": round(elapsed/60, 1),
        })

        # Save history each epoch (in case of crash)
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "read_threshold": READ_THRESHOLD,
            }, output_dir / "wav2vec2_best.pt")
            print(f"  -> New best (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping")
                break

    total_time = (time.time() - start_time) / 60
    print(f"\nTotal training time: {total_time:.1f} min")

    # ── Final test ──
    print(f"\n{'='*60}")
    print("FINAL TEST (best model)")
    print(f"{'='*60}")
    ck = torch.load(output_dir / "wav2vec2_best.pt", map_location=device)
    model.load_state_dict(ck["model_state_dict"])

    _, test_acc, test_f1, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device)

    print(f"\nTest — Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(classification_report(test_labels, test_preds,
                                target_names=["spontaneous", "read"]))

    # Threshold sweep
    print("Threshold Sensitivity:")
    for t in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        t_preds = (test_probs >= t).astype(int)
        t_f1 = f1_score(test_labels, t_preds, zero_division=0)
        marker = " <-- default" if t == READ_THRESHOLD else ""
        print(f"  threshold={t:.2f}: f1={t_f1:.4f}{marker}")

    # Save results
    results = {
        "test_acc": round(test_acc, 4), "test_f1": round(test_f1, 4),
        "best_val_f1": round(best_f1, 4), "best_epoch": ck["epoch"],
        "total_time_min": round(total_time, 1),
        "read_threshold": READ_THRESHOLD,
        "config": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "freeze_layers": args.freeze_layers,
            "datasets": args.manifest,
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  wav2vec2_best.pt  — best checkpoint")
    print(f"  results.json      — test metrics")
    print(f"  history.json      — training history")


if __name__ == "__main__":
    main()
