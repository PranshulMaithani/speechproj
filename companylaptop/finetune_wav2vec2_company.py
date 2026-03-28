"""
Finetune wav2vec2 on company data (cheating vs not cheating).
==============================================================
Loads pretrained combined wav2vec2, finetunes on company audio.
Works on CPU (slow but doable with ~500 files).

Usage:
    python finetune_wav2vec2_company.py --features features_company.csv --checkpoint wav2vec2_combined_best.pt
    python finetune_wav2vec2_company.py --features features_company.csv --checkpoint wav2vec2_combined_best.pt --epochs 5 --batch-size 4
"""

import argparse
import json
import os
import time
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


# ── Config ──────────────────────────────────────────────────────
SAMPLE_RATE = 16000
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)


# ── Dataset ─────────────────────────────────────────────────────
class CompanyAudioDataset(Dataset):
    """Loads audio from filepaths in CSV, creates 5-sec windows."""

    def __init__(self, df):
        self.sr = SAMPLE_RATE
        self.window = WINDOW_SAMPLES
        self.df = df.reset_index(drop=True)
        self.window_index = []

        print("Building window index...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
            fp = row["filepath"]
            if not os.path.exists(fp):
                continue
            try:
                duration = librosa.get_duration(path=fp)
                n_samples = int(duration * self.sr)
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


# ── Model ───────────────────────────────────────────────────────
class CompanyWav2Vec2(nn.Module):
    def __init__(self, freeze_layers=10):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Freeze CNN feature extractor
        self.encoder.feature_extractor._freeze_parameters()

        # Freeze first N transformer layers (wav2vec2-base has 12)
        # Only train last (12 - freeze_layers) layers + classifier
        for i, layer in enumerate(self.encoder.encoder.layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, input_values):
        outputs = self.encoder(input_values).last_hidden_state
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ── Training ────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training"):
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device).float()

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits).detach().cpu().numpy() >= 0.5).astype(int)
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

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend((probs >= 0.5).astype(int))
        all_labels.extend(y.cpu().numpy().astype(int))
        all_probs.extend(probs)

    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, zero_division=0),
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


# ── Export to ONNX ──────────────────────────────────────────────
def export_onnx(model, output_dir):
    model.eval()
    model_cpu = model.to("cpu")

    onnx_path = output_dir / "wav2vec2_company.onnx"
    dummy = torch.zeros(1, WINDOW_SAMPLES)

    print(f"\nExporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model_cpu, dummy, str(onnx_path),
        opset_version=14,
        input_names=["input_values"],
        output_names=["logit"],
        dynamic_axes={"input_values": {0: "batch_size"}, "logit": {0: "batch_size"}},
        do_constant_folding=True,
    )

    # Quantize
    quant_path = output_dir / "wav2vec2_company_quant.onnx"
    print(f"Quantizing to INT8: {quant_path}")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(onnx_path), str(quant_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"  {quant_path.stat().st_size / 1e6:.1f} MB")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, WINDOW_SAMPLES).astype(np.float32)
    output = sess.run(["logit"], {"input_values": test_input})
    print(f"  Inference test OK — output shape: {output[0].shape}")

    return quant_path


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Finetune wav2vec2 on company data")
    parser.add_argument("--features", required=True, help="Features CSV with filepath and label_int columns")
    parser.add_argument("--checkpoint", default=None, help="Pretrained wav2vec2 checkpoint (.pt) to start from")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--freeze-layers", type=int, default=10, help="Freeze first N transformer layers (of 12)")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="checkpoints_company_w2v")
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"NOTE: CPU training — this will take 2-3 hours for ~500 files")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from CSV
    df = pd.read_csv(args.features)
    df = df[df["label_int"].isin([0, 1])].reset_index(drop=True)
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    print(f"\nLoaded {len(df)} labelled files with audio on disk")
    print(f"  Cheating (1):     {(df['label_int']==1).sum()}")
    print(f"  Not cheating (0): {(df['label_int']==0).sum()}")

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_int"])
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

    # Build datasets
    train_dataset = CompanyAudioDataset(train_df)
    test_dataset = CompanyAudioDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # Model
    print(f"\nBuilding model (freeze_layers={args.freeze_layers})...")
    model = CompanyWav2Vec2(freeze_layers=args.freeze_layers).to(device)

    # Load pretrained weights if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading pretrained checkpoint: {args.checkpoint}")
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Load encoder + classifier weights from checkpoint
        state = ck["model_state_dict"] if "model_state_dict" in ck else ck
        model.load_state_dict(state, strict=False)
        print(f"  Loaded (epoch={ck.get('epoch', '?')}, val_f1={ck.get('val_f1', '?')})")

    # Class balancing
    train_labels = [train_dataset.df.iloc[fi]["label_int"] for fi, _ in train_dataset.window_index]
    n_neg = sum(1 for l in train_labels if l == 0)
    n_pos = sum(1 for l in train_labels if l == 1)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Class balance: {n_neg} not-cheating, {n_pos} cheating, pos_weight={pos_weight.item():.3f}")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=0.01)
    num_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * 0.1), num_steps)

    # Train
    best_f1 = 0.0
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"TRAINING: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, test_loader, criterion, device)

        elapsed = time.time() - epoch_start
        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  Time: {elapsed/60:.1f} min")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_f1": train_f1,
                        "val_loss": val_loss, "val_f1": val_f1, "time_min": round(elapsed/60, 1)})

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
            }, output_dir / "wav2vec2_company_best.pt")
            print(f"  -> New best (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping")
                break

    total_time = (time.time() - start_time) / 60
    print(f"\nTotal training time: {total_time:.1f} min")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final test with best model
    print(f"\n{'='*60}")
    print("FINAL TEST")
    print(f"{'='*60}")
    ck = torch.load(output_dir / "wav2vec2_company_best.pt", map_location=device)
    model.load_state_dict(ck["model_state_dict"])

    _, test_acc, test_f1, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device)
    print(f"\nTest — Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=["not cheating", "cheating"]))

    # Threshold sweep
    for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
        t_preds = (test_probs >= t).astype(int)
        t_f1 = f1_score(test_labels, t_preds, zero_division=0)
        marker = " <--" if t == 0.50 else ""
        print(f"  threshold={t:.2f}: f1={t_f1:.4f}{marker}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({"test_acc": round(test_acc, 4), "test_f1": round(test_f1, 4),
                    "best_val_f1": round(best_f1, 4), "best_epoch": ck["epoch"],
                    "total_time_min": round(total_time, 1)}, f, indent=2)

    # Export ONNX
    if not args.no_export:
        try:
            quant_path = export_onnx(model, output_dir)
            print(f"\nDone! Use: {quant_path}")
        except Exception as e:
            print(f"\nONNX export failed: {e}")
            print("Run manually later or use the .pt checkpoint")
    else:
        print(f"\nDone! Checkpoint: {output_dir / 'wav2vec2_company_best.pt'}")


if __name__ == "__main__":
    main()
