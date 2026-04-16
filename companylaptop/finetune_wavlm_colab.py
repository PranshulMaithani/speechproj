"""
WavLM Fine-Tuning for Scripted vs Spontaneous Speech Detection
==============================================================
Run this on Google Colab (free T4 GPU is enough).

What it does:
  1. Downloads LibriSpeech-clean-100 + AMI from HuggingFace (auto, no upload needed)
  2. Reads AllStar 2676 (spontaneous) + 2677 (scripted) and CasualConversations
     from paths you configure below
  3. Slices every long file into 10-second windows (5-second hop)
  4. Concatenates consecutive short AMI/LibriSpeech clips from the same speaker
     until they reach 10 seconds (no silence padding)
  5. Caps each class at MAX_WINDOWS_PER_CLASS to keep training balanced
  6. Fine-tunes WavLM-base-plus:
       - Feature extractor (CNN): always frozen
       - Transformer layers 0-5:  frozen
       - Transformer layers 6-11: fine-tuned (lr = WAVLM_LR)
       - Classification head:     fine-tuned (lr = HEAD_LR)
  7. Saves ONLY the fine-tuned WavLM encoder (no classification head)
     in HuggingFace format → drop-in replacement for the extraction pipeline

After training download:
    output/wavlm_finetuned/   (the encoder weights, ~350 MB)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths to your uploaded data ───────────────────────────────────────────────
# On Colab with Google Drive mounted at /content/drive/MyDrive:
ALLSTAR_SPONT_DIR = "/content/drive/MyDrive/speech_data/2676"      # spontaneous
ALLSTAR_SCRIPT_DIR = "/content/drive/MyDrive/speech_data/2677"     # scripted
CASUAL_CONV_DIR = "/content/drive/MyDrive/speech_data/casual_conversations"  # has audio_scripted/ and audio_nonscripted/
CASUAL_CONV_MANIFEST = "/content/drive/MyDrive/speech_data/casual_conversations/manifest.csv"

# Where to save the fine-tuned model
OUTPUT_DIR = "/content/drive/MyDrive/speech_data/wavlm_finetuned"

# ── Dataset settings ──────────────────────────────────────────────────────────
WINDOW_SEC = 10          # seconds per training window
HOP_SEC    = 5           # hop between windows (overlap = WINDOW_SEC - HOP_SEC)
MIN_WIN_SEC = 8          # discard windows shorter than this
MAX_WINDOWS_PER_CLASS = 25_000   # cap to keep train balanced + fit in Colab RAM
VAL_FRACTION = 0.10      # fraction of windowed data held out for validation
                          # (uses CasualConversations only for validation — closest to Mettl domain)

# ── Model settings ────────────────────────────────────────────────────────────
WAVLM_MODEL = "microsoft/wavlm-base-plus"
FREEZE_LAYERS = 6        # freeze bottom N transformer layers (0 = freeze nothing)
WAVLM_LR  = 5e-5         # learning rate for fine-tuned WavLM transformer layers
HEAD_LR   = 1e-4         # learning rate for classification head
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 200
NUM_EPOCHS   = 6
BATCH_SIZE   = 8         # per-GPU; T4 (16GB) handles 8 comfortably with 10s clips
GRAD_ACCUM   = 4         # effective batch = BATCH_SIZE × GRAD_ACCUM = 32

SR = 16000               # WavLM expects 16kHz mono

# ─────────────────────────────────────────────────────────────────────────────
# 0. INSTALL DEPENDENCIES (uncomment on first Colab run)
# ─────────────────────────────────────────────────────────────────────────────
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "-q",
#                 "transformers", "datasets", "soundfile", "librosa",
#                 "tqdm", "scikit-learn", "torch", "numpy", "pandas",
#                 "soundfile", "resampy"])

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, json, warnings, random
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from transformers import (
    AutoFeatureExtractor,
    WavLMModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOWNLOAD AMI + LIBRISPEECH (skipped if already done)
# ─────────────────────────────────────────────────────────────────────────────

def download_and_save_hf(dataset_name, config, split, out_dir, label, label_int,
                          min_sec=3.0, max_sec=30.0, max_samples=15_000):
    """Download a HuggingFace dataset, save wavs + manifest. Skips if .done exists."""
    out_dir = Path(out_dir)
    done = out_dir / ".done"
    manifest_path = out_dir / "manifest.csv"
    if done.exists():
        print(f"  {out_dir.name}: already downloaded, loading manifest.")
        return pd.read_csv(manifest_path)

    from datasets import load_dataset, Audio as HFAudio
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {dataset_name} / {config} / {split} …")
    ds = load_dataset(dataset_name, config, split=split, trust_remote_code=True)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=SR))
    print(f"  {len(ds)} utterances found, filtering {min_sec}–{max_sec}s …")

    rows = []
    skipped = 0
    for i in tqdm(range(len(ds)), desc=f"  {out_dir.name}"):
        if len(rows) >= max_samples:
            break
        item = ds[i]
        audio = item["audio"]["array"].astype(np.float32)
        dur   = len(audio) / SR
        if dur < min_sec or dur > max_sec:
            skipped += 1
            continue
        fname = f"{out_dir.name}_{i:07d}.wav"
        fpath = out_dir / fname
        if not fpath.exists():
            sf.write(str(fpath), audio, SR)
        rows.append({
            "filepath":   str(fpath.resolve()),
            "filename":   fname,
            "source":     out_dir.name,
            "label":      label,
            "label_int":  label_int,
            "duration_sec": round(dur, 3),
            "speaker_id": str(item.get("speaker_id", i // 10)),  # fallback group
        })

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    done.touch()
    print(f"  Saved {len(df)} utterances ({skipped} skipped). Manifest: {manifest_path}")
    return df


def get_external_manifests():
    """Download AMI and LibriSpeech, return manifests."""
    ami_df  = download_and_save_hf(
        "edinburghcstr/ami", "ihm", "train",
        out_dir="/content/datasets/ami",
        label="spontaneous", label_int=0,
        min_sec=3.0, max_sec=25.0, max_samples=12_000,
    )
    libri_df = download_and_save_hf(
        "openslr/librispeech_asr", "clean", "train.100",
        out_dir="/content/datasets/librispeech",
        label="scripted", label_int=1,
        min_sec=3.0, max_sec=25.0, max_samples=12_000,
    )
    return ami_df, libri_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD WINDOWED CLIP LIST
#    Long files  → 10s windows with HOP_SEC overlap
#    Short clips → concatenate consecutive clips from same speaker until ≥10s
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_16k(path):
    """Load any audio file as 16kHz mono float32 numpy array."""
    try:
        y, sr = sf.read(str(path), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)           # stereo → mono
        if sr != SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
        return y.astype(np.float32)
    except Exception as e:
        print(f"  WARN load_audio {path}: {e}")
        return None


def slice_long_clip(filepath, label_int, win_samples, hop_samples, min_samples):
    """
    Slice a single long audio file into overlapping windows.
    Returns list of (numpy_array, label_int).
    """
    y = load_audio_16k(filepath)
    if y is None or len(y) < min_samples:
        return []
    windows = []
    start = 0
    while start + win_samples <= len(y):
        windows.append((y[start : start + win_samples], label_int))
        start += hop_samples
    # last partial window: keep if long enough
    tail = y[start:]
    if len(tail) >= min_samples:
        # pad to exactly win_samples with reflected audio
        pad = win_samples - len(tail)
        tail = np.concatenate([tail, tail[:pad][::-1]])
        windows.append((tail, label_int))
    return windows


def concat_short_clips(manifest_df, label_int, win_samples, min_samples):
    """
    For short-utterance datasets (AMI, LibriSpeech), concatenate consecutive
    clips from the same speaker until we have ≥ win_samples, then yield one window.
    No silence padding — consecutive clips from the same speaker are semantically
    consistent and this preserves natural prosody boundaries.
    """
    windows = []
    # Group by speaker_id; within group, process in file order
    for spk_id, grp in manifest_df.groupby("speaker_id"):
        buffer = np.array([], dtype=np.float32)
        for _, row in grp.iterrows():
            y = load_audio_16k(row["filepath"])
            if y is None:
                continue
            buffer = np.concatenate([buffer, y])
            # emit as many full windows as possible
            while len(buffer) >= win_samples:
                windows.append((buffer[:win_samples], label_int))
                buffer = buffer[win_samples:]   # no hop for short-clip concat — each window distinct
        # leftover: keep if long enough
        if len(buffer) >= min_samples:
            pad = win_samples - len(buffer)
            buf_padded = np.concatenate([buffer, buffer[:pad][::-1]])
            windows.append((buf_padded, label_int))
    return windows


def build_windows_from_long_folder(folder_path, label_int, win_samples, hop_samples, min_samples):
    """Walk a folder of long WAV files and slice into windows."""
    folder = Path(folder_path)
    wavs = sorted(folder.rglob("*.wav")) + sorted(folder.rglob("*.flac"))
    windows = []
    for wav in tqdm(wavs, desc=f"  windowing {folder.name}"):
        windows.extend(slice_long_clip(wav, label_int, win_samples, hop_samples, min_samples))
    return windows


def build_windows_from_manifest_long(manifest_df, label_int, win_samples, hop_samples, min_samples):
    """Same as above but driven by a manifest CSV."""
    windows = []
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df),
                       desc=f"  windowing manifest (label={label_int})"):
        windows.extend(slice_long_clip(row["filepath"], label_int,
                                       win_samples, hop_samples, min_samples))
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# 4. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SpeechWindowDataset(Dataset):
    """
    Each item: (audio_tensor [win_samples], label_int)
    The feature extractor (padding/normalisation) is applied in __getitem__
    so it can run in parallel DataLoader workers.
    """
    def __init__(self, windows, feature_extractor):
        # windows: list of (np.ndarray, int)
        self.windows = windows
        self.fe = feature_extractor

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        audio, label = self.windows[idx]
        # WavLM feature extractor: normalises + converts to input_values tensor
        inputs = self.fe(audio, sampling_rate=SR, return_tensors="pt",
                         padding=False)
        input_values = inputs.input_values.squeeze(0)   # [win_samples]
        return input_values, torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    input_values = torch.stack([x[0] for x in batch])   # [B, T]
    labels       = torch.stack([x[1] for x in batch])   # [B]
    return input_values, labels


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────────────────────────────────────

class WavLMClassifier(nn.Module):
    """WavLM encoder + mean-pool + linear head for binary classification."""

    def __init__(self, wavlm_model_name, freeze_layers=6):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)

        # Freeze feature extractor (CNN) — always
        for p in self.wavlm.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.wavlm.feature_projection.parameters():
            p.requires_grad = False

        # Freeze bottom N transformer layers
        for i, layer in enumerate(self.wavlm.encoder.layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        hidden = self.wavlm.config.hidden_size   # 768 for base-plus
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_values):
        # input_values: [B, T]
        out = self.wavlm(input_values)
        # Mean-pool over time → [B, 768]
        pooled = out.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)   # [B]
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def get_param_groups(model):
    """Differential learning rates: higher for head, lower for WavLM layers."""
    wavlm_params = [p for p in model.wavlm.parameters() if p.requires_grad]
    head_params  = list(model.classifier.parameters())
    return [
        {"params": wavlm_params, "lr": WAVLM_LR},
        {"params": head_params,  "lr": HEAD_LR},
    ]


def run_epoch(model, loader, optimizer=None, scheduler=None, scaler=None,
              desc="train"):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    criterion = nn.BCEWithLogitsLoss()

    pbar = tqdm(loader, desc=desc, leave=False)
    step = 0
    for input_values, labels in pbar:
        input_values = input_values.to(DEVICE)
        labels       = labels.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(input_values)
            loss   = criterion(logits, labels)
            if training:
                loss = loss / GRAD_ACCUM

        if training:
            scaler.scale(loss).backward()
            step += 1
            if step % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * (GRAD_ACCUM if training else 1)   # un-scale
        preds = (torch.sigmoid(logits.detach()) > 0.5).cpu().numpy().astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().astype(int))
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")

    f1  = f1_score(all_labels, all_preds, zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = total_loss / len(loader)
    return avg_loss, f1, acc


def train(train_windows, val_windows, feature_extractor):
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"  Train windows: {len(train_windows)}")
    print(f"  Val windows:   {len(val_windows)}")
    print(f"  Train scripted:    {sum(1 for _,l in train_windows if l==1)}")
    print(f"  Train spontaneous: {sum(1 for _,l in train_windows if l==0)}")
    print(f"{'='*60}\n")

    random.shuffle(train_windows)

    train_ds = SpeechWindowDataset(train_windows, feature_extractor)
    val_ds   = SpeechWindowDataset(val_windows,   feature_extractor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True,
                              collate_fn=collate_fn)

    model = WavLMClassifier(WAVLM_MODEL, freeze_layers=FREEZE_LAYERS).to(DEVICE)

    param_groups = get_param_groups(model)
    optimizer    = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_f1 = 0.0
    history = []
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_f1, tr_acc = run_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]"
        )
        vl_loss, vl_f1, vl_acc = run_epoch(
            model, val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]"
        )

        row = dict(epoch=epoch,
                   tr_loss=round(tr_loss,4), tr_f1=round(tr_f1,4), tr_acc=round(tr_acc,4),
                   vl_loss=round(vl_loss,4), vl_f1=round(vl_f1,4), vl_acc=round(vl_acc,4))
        history.append(row)
        print(f"Epoch {epoch:02d}  "
              f"train  loss={tr_loss:.4f}  F1={tr_f1:.4f}  acc={tr_acc:.4f}  |  "
              f"val    loss={vl_loss:.4f}  F1={vl_f1:.4f}  acc={vl_acc:.4f}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            # Save ONLY the WavLM encoder — this is what the embedding pipeline uses
            model.wavlm.save_pretrained(str(out))
            feature_extractor.save_pretrained(str(out))
            print(f"  ✓ Saved best model (val F1={vl_f1:.4f}) → {out}")

    # Save training history
    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {out}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    win_samples = int(WINDOW_SEC * SR)     # 160,000
    hop_samples = int(HOP_SEC    * SR)     #  80,000
    min_samples = int(MIN_WIN_SEC * SR)    # 128,000

    feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)

    # ── Load / download data ──────────────────────────────────────────────────
    print("\n[1/4] Loading data sources...")

    # AllStar 2677 — scripted long responses (label=1)
    print("\nAllStar 2677 (scripted):")
    allstar_script_wins = build_windows_from_long_folder(
        ALLSTAR_SCRIPT_DIR, label_int=1,
        win_samples=win_samples, hop_samples=hop_samples, min_samples=min_samples
    )
    print(f"  {len(allstar_script_wins)} windows from 2677")

    # AllStar 2676 — spontaneous long responses (label=0)
    print("\nAllStar 2676 (spontaneous):")
    allstar_spont_wins = build_windows_from_long_folder(
        ALLSTAR_SPONT_DIR, label_int=0,
        win_samples=win_samples, hop_samples=hop_samples, min_samples=min_samples
    )
    print(f"  {len(allstar_spont_wins)} windows from 2676")

    # CasualConversations — split by folder name (scripted / non-scripted)
    print("\nCasualConversations:")
    cc_df = pd.read_csv(CASUAL_CONV_MANIFEST)
    cc_script_df = cc_df[cc_df["label"].isin(["read", "scripted"])]
    cc_spont_df  = cc_df[cc_df["label"].isin(["spontaneous", "nonscripted"])]

    cc_script_wins = build_windows_from_manifest_long(
        cc_script_df, label_int=1,
        win_samples=win_samples, hop_samples=hop_samples, min_samples=min_samples
    )
    cc_spont_wins = build_windows_from_manifest_long(
        cc_spont_df, label_int=0,
        win_samples=win_samples, hop_samples=hop_samples, min_samples=min_samples
    )
    print(f"  {len(cc_script_wins)} scripted windows, {len(cc_spont_wins)} spontaneous windows")

    # AMI — short utterances → concatenate per speaker
    print("\nAMI (download if needed):")
    ami_df, libri_df = get_external_manifests()
    ami_wins = concat_short_clips(ami_df, label_int=0,
                                   win_samples=win_samples, min_samples=min_samples)
    print(f"  {len(ami_wins)} AMI windows")

    # LibriSpeech — short utterances → concatenate per speaker
    print("\nLibriSpeech (already downloaded):")
    libri_wins = concat_short_clips(libri_df, label_int=1,
                                     win_samples=win_samples, min_samples=min_samples)
    print(f"  {len(libri_wins)} LibriSpeech windows")

    # ── Combine and cap ───────────────────────────────────────────────────────
    # AllStar is most domain-similar to Mettl → always include ALL AllStar windows.
    # Fill remaining slots up to MAX_WINDOWS_PER_CLASS from CC + AMI/LibriSpeech.
    print("\n[2/4] Combining and balancing...")

    remaining_s = max(0, MAX_WINDOWS_PER_CLASS - len(allstar_script_wins))
    extra_s = cc_script_wins + libri_wins
    random.shuffle(extra_s)
    all_scripted = allstar_script_wins + extra_s[:remaining_s]

    remaining_p = max(0, MAX_WINDOWS_PER_CLASS - len(allstar_spont_wins))
    extra_p = cc_spont_wins + ami_wins
    random.shuffle(extra_p)
    all_spontaneous = allstar_spont_wins + extra_p[:remaining_p]

    print(f"  Scripted:    {len(all_scripted)}"
          f"  (AllStar={len(allstar_script_wins)}, other={len(all_scripted)-len(allstar_script_wins)})")
    print(f"  Spontaneous: {len(all_spontaneous)}"
          f"  (AllStar={len(allstar_spont_wins)}, other={len(all_spontaneous)-len(allstar_spont_wins)})")

    # ── Train / val split ─────────────────────────────────────────────────────
    # Simple random split: VAL_FRACTION from each class.
    print("\n[3/4] Splitting train / val...")

    def split_val(wins, val_frac):
        wins = list(wins)
        random.shuffle(wins)
        n_val = max(50, int(len(wins) * val_frac))
        return wins[n_val:], wins[:n_val]   # (train, val)

    train_s, val_s = split_val(all_scripted,    VAL_FRACTION)
    train_p, val_p = split_val(all_spontaneous, VAL_FRACTION)

    train_windows = train_s + train_p
    val_windows   = val_s   + val_p

    print(f"  Train: {len(train_windows)}  (scripted={len(train_s)}, spont={len(train_p)})")
    print(f"  Val:   {len(val_windows)}    (scripted={len(val_s)}, spont={len(val_p)})")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print("\n[4/4] Fine-tuning WavLM...")
    history = train(train_windows, val_windows, feature_extractor)

    print("\n✓ Done! Fine-tuned encoder saved to:", OUTPUT_DIR)
    print("\nNext steps:")
    print("  1. Download the wavlm_finetuned/ folder from Drive")
    print("  2. In train_models_playground.ipynb, change:")
    print('       WavLMModel.from_pretrained("microsoft/wavlm-base-plus")')
    print('     to:')
    print('       WavLMModel.from_pretrained("path/to/wavlm_finetuned")')
    print("  3. Run the extraction pipeline (it uses the same 10s window approach)")


if __name__ == "__main__":
    main()
