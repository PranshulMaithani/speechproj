"""
Train Biased Wav2Vec2 on Combined Datasets (ALLSSTAR + LibriSpeech + AMI)
==========================================================================
Same biased single-neuron architecture, now trained on more diverse data.

Usage:
    python train_wav2vec2_combined.py
"""

import os
import json
from pathlib import Path

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
import librosa


# ── Config ──────────────────────────────────────────────────────────────
READ_THRESHOLD = 0.65
CHECKPOINT_DIR = Path("checkpoints_combined")
WINDOW_SEC = 5.0
SAMPLE_RATE = 16000
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)

# Training hyperparams
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 3
FREEZE_LAYERS = 6
MAX_PER_DATASET = 5000  # Max files per dataset to keep balanced


# ── Dataset ─────────────────────────────────────────────────────────────
class CombinedWindowDataset(Dataset):
    """Loads audio files and returns 5-sec windows with labels."""

    def __init__(self, manifest_df):
        self.sr = SAMPLE_RATE
        self.window = WINDOW_SAMPLES

        # Build window index: (row_idx, window_start_sample)
        self.df = manifest_df.reset_index(drop=True)
        self.window_index = []

        print("Building window index...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
            filepath = row["filepath"]
            try:
                info = librosa.get_duration(path=filepath)
                n_samples = int(info * self.sr)
                n_windows = max(1, n_samples // self.window)
                for w in range(n_windows):
                    self.window_index.append((idx, w * self.window))
            except Exception:
                continue

        print(f"  {len(self.window_index)} windows from {len(self.df)} files")

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, i):
        file_idx, start_sample = self.window_index[i]
        row = self.df.iloc[file_idx]

        # Load audio chunk
        offset_sec = start_sample / self.sr
        audio, _ = librosa.load(row["filepath"], sr=self.sr, mono=True,
                                offset=offset_sec, duration=WINDOW_SEC)

        # Pad if too short
        if len(audio) < self.window:
            audio = np.pad(audio, (0, self.window - len(audio)))
        else:
            audio = audio[:self.window]

        return {
            "input_values": torch.tensor(audio, dtype=torch.float32),
            "labels": torch.tensor(row["label_int"], dtype=torch.long),
        }


# ── Model ───────────────────────────────────────────────────────────────
class BiasedSpeechClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", hidden_size=256,
                 dropout=0.3, freeze_layers=6):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze CNN feature extractor
        self.encoder.feature_extractor._freeze_parameters()

        # Freeze first N transformer layers
        for i, layer in enumerate(self.encoder.encoder.layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, input_values):
        outputs = self.encoder(input_values).last_hidden_state
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


# ── Training ────────────────────────────────────────────────────────────
def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


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

    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, zero_division=0)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device).float()

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


# ── Main ────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # Load all three manifests
    dfs = []

    # ALLSSTAR
    allsstar = pd.read_csv("old/outputs/manifest_expanded.csv")
    allsstar = allsstar[allsstar["filepath"].apply(os.path.exists)].reset_index(drop=True)
    allsstar = allsstar[["filepath", "filename", "label", "label_int"]].copy()
    allsstar["source"] = "allsstar"
    print(f"ALLSSTAR: {len(allsstar)} files ({(allsstar.label_int==1).sum()} read, {(allsstar.label_int==0).sum()} spont)")
    dfs.append(allsstar)

    # LibriSpeech
    libri = pd.read_csv("datasets/librispeech/manifest.csv")
    libri = libri[libri["filepath"].apply(os.path.exists)].reset_index(drop=True)
    if len(libri) > MAX_PER_DATASET:
        libri = libri.sample(n=MAX_PER_DATASET, random_state=42)
    libri = libri[["filepath", "filename", "label", "label_int"]].copy()
    libri["source"] = "librispeech"
    print(f"LibriSpeech: {len(libri)} files (all read)")
    dfs.append(libri)

    # AMI
    ami = pd.read_csv("datasets/ami/manifest.csv")
    ami = ami[ami["filepath"].apply(os.path.exists)].reset_index(drop=True)
    if len(ami) > MAX_PER_DATASET:
        ami = ami.sample(n=MAX_PER_DATASET, random_state=42)
    ami = ami[["filepath", "filename", "label", "label_int"]].copy()
    ami["source"] = "ami"
    print(f"AMI: {len(ami)} files (all spontaneous)")
    dfs.append(ami)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} files")
    print(f"  Read: {(combined.label_int==1).sum()}, Spontaneous: {(combined.label_int==0).sum()}")

    # Train/val/test split (stratified)
    train_df, test_df = train_test_split(combined, test_size=0.15, random_state=42, stratify=combined["label_int"])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["label_int"])

    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build datasets
    train_dataset = CombinedWindowDataset(train_df)
    val_dataset = CombinedWindowDataset(val_df)
    test_dataset = CombinedWindowDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Model
    model = BiasedSpeechClassifier(freeze_layers=FREEZE_LAYERS).to(device)

    # Loss with class balancing
    train_labels = [train_dataset.df.iloc[fi]["label_int"] for fi, _ in train_dataset.window_index]
    n_spont = sum(1 for l in train_labels if l == 0)
    n_read = sum(1 for l in train_labels if l == 1)
    pos_weight = torch.tensor([n_spont / max(n_read, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Class balance: {n_spont} spont, {n_read} read, pos_weight={pos_weight.item():.3f}")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * WARMUP_RATIO), num_steps)
    scaler = torch.cuda.amp.GradScaler()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"TRAINING: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"{'='*60}")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_f1": train_f1,
                         "val_loss": val_loss, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "read_threshold": READ_THRESHOLD,
            }, CHECKPOINT_DIR / "wav2vec2_combined_best.pt")
            print(f"  -> New best (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final test
    print(f"\n{'='*60}")
    print("FINAL TEST")
    print(f"{'='*60}")
    ck = torch.load(CHECKPOINT_DIR / "wav2vec2_combined_best.pt", map_location=device)
    model.load_state_dict(ck["model_state_dict"])

    _, test_acc, test_f1, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    print(f"\nTest — Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=["spontaneous", "read"]))

    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        t_preds = (test_probs >= t).astype(int)
        t_f1 = f1_score(test_labels, t_preds, zero_division=0)
        marker = " <--" if abs(t - READ_THRESHOLD) < 0.01 else ""
        print(f"  threshold={t:.2f}: f1={t_f1:.4f}{marker}")

    with open(CHECKPOINT_DIR / "results.json", "w") as f:
        json.dump({"test_acc": round(test_acc, 4), "test_f1": round(test_f1, 4),
                    "best_val_f1": round(best_f1, 4), "best_epoch": ck["epoch"]}, f, indent=2)

    print(f"\nSaved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
