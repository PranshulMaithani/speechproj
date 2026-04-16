"""
WavLM Fine-Tuning for Scripted vs Spontaneous Speech Detection
==============================================================
Run this on Google Colab or Kaggle (free T4 GPU is enough).

Memory strategy:
  - Long-file sources (AllStar, CasualConversations): LAZY -- only store
    (filepath, start_sample, end_sample, label) in RAM; audio loaded per-batch.
  - Short-clip sources (AMI, LibriSpeech, CommonVoice): EAGER -- small enough
    to concatenate and hold in RAM after per-source cap.
  This keeps peak RAM under ~4 GB regardless of how many AllStar windows exist.

After training, download:
    /kaggle/working/wavlm_finetuned/   (encoder weights, ~350 MB)
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

ALLSTAR_SPONT_DIR     = "/kaggle/input/datasets/pranshulmaithani/2676-zip/2676"
ALLSTAR_SCRIPT_DIR    = "/kaggle/input/datasets/pranshulmaithani/2677dataset/2677"
CASUAL_CONV_DIR       = "/kaggle/input/datasets/pranshulmaithani/casual-conversations/casual_conversations/casual_conversations"
COMMONVOICE_SPONT_DIR = "/kaggle/input/datasets/pranshulmaithani/commonvoicespont/commonvoicespont/sps-corpus-3.0-2026-03-09-en"
OUTPUT_DIR            = "/kaggle/working/wavlm_finetuned"

WINDOW_SEC  = 10
HOP_SEC     = 5
MIN_WIN_SEC = 8
# Cap on in-memory (short-clip) windows per source -- lazy sources are uncapped
MAX_INMEM_PER_SOURCE = 3_000
USE_HF_DATASETS      = True    # stream AMI + LibriSpeech from HuggingFace
VAL_FRACTION         = 0.10

WAVLM_MODEL   = "microsoft/wavlm-base-plus"
FREEZE_LAYERS = 6
WAVLM_LR      = 5e-5
HEAD_LR       = 1e-4
WEIGHT_DECAY  = 1e-2
WARMUP_STEPS  = 200
NUM_EPOCHS    = 6
BATCH_SIZE    = 8
GRAD_ACCUM    = 4

SR = 16000

# =============================================================================
# 0. INSTALL DEPENDENCIES (uncomment on first run)
# =============================================================================
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "-q",
#                 "transformers", "datasets", "soundfile", "librosa",
#                 "tqdm", "scikit-learn", "torch", "numpy", "pandas", "resampy"])

# =============================================================================
# 1. IMPORTS
# =============================================================================
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
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# 2. AUDIO UTILITIES
# =============================================================================

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_audio_16k(path):
    try:
        y, sr = sf.read(str(path), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
        return y.astype(np.float32)
    except Exception as e:
        print(f"  WARN load_audio {path}: {e}")
        return None


# =============================================================================
# 3. LAZY WINDOWS  (long-file sources: AllStar, CasualConversations)
#    Each item is a 4-tuple: (filepath_str, start_sample, end_sample, label_int)
#    Audio is loaded from disk only in __getitem__ -- zero RAM for audio arrays.
# =============================================================================

def get_audio_length_16k(path):
    """Return number of samples at 16 kHz without loading the full file."""
    try:
        info = sf.info(str(path))
        if info.samplerate == SR:
            return info.frames
        return int(info.frames * SR / info.samplerate)
    except Exception:
        return None


def slice_long_clip_lazy(filepath, label_int, win_samples, hop_samples, min_samples):
    """Return lazy index tuples (filepath, start, end, label) -- no audio loaded."""
    n = get_audio_length_16k(filepath)
    if n is None or n < min_samples:
        return []
    windows = []
    start = 0
    while start + win_samples <= n:
        windows.append((str(filepath), start, start + win_samples, label_int))
        start += hop_samples
    # last partial window: reflect-pad if long enough
    tail_start = start
    if n - tail_start >= min_samples:
        windows.append((str(filepath), n - win_samples, n, label_int))
    return windows


def build_lazy_windows_from_folder(folder_path, label_int, win_samples, hop_samples, min_samples):
    folder = Path(folder_path)
    files  = sorted(folder.rglob("*.wav")) + sorted(folder.rglob("*.flac"))
    print(f"  indexing {folder.name} ({len(files)} files) ...", flush=True)
    windows = []
    for f in files:
        windows.extend(slice_long_clip_lazy(f, label_int, win_samples, hop_samples, min_samples))
    return windows


def build_lazy_windows_from_manifest(manifest_df, label_int, win_samples, hop_samples, min_samples):
    print(f"  indexing manifest ({len(manifest_df)} files, label={label_int}) ...", flush=True)
    windows = []
    for _, row in manifest_df.iterrows():
        windows.extend(slice_long_clip_lazy(row["filepath"], label_int,
                                            win_samples, hop_samples, min_samples))
    return windows


# =============================================================================
# 4. EAGER WINDOWS  (short-clip sources: AMI, LibriSpeech, CommonVoice)
#    These are small enough to hold in RAM after per-source caps.
# =============================================================================

def concat_short_clips(manifest_df, label_int, win_samples, min_samples, max_windows=None):
    """Concatenate per-speaker clips into full windows. Returns list of (np.array, label)."""
    windows = []
    for spk_id, grp in manifest_df.groupby("speaker_id"):
        if max_windows and len(windows) >= max_windows:
            break
        buffer = np.array([], dtype=np.float32)
        for _, row in grp.iterrows():
            y = load_audio_16k(row["filepath"])
            if y is None:
                continue
            buffer = np.concatenate([buffer, y])
            while len(buffer) >= win_samples:
                if max_windows and len(windows) >= max_windows:
                    break
                windows.append((buffer[:win_samples], label_int))
                buffer = buffer[win_samples:]
        if len(buffer) >= min_samples:
            if not max_windows or len(windows) < max_windows:
                pad = win_samples - len(buffer)
                windows.append((np.concatenate([buffer, buffer[:pad][::-1]]), label_int))
    return windows


def stream_hf_to_windows(dataset_name, config, split, label_int,
                          win_samples, min_samples, max_windows=3_000,
                          min_sec=3.0, max_sec=25.0):
    """Stream HuggingFace dataset into eager windows. Nothing written to disk."""
    from datasets import load_dataset, Audio as HFAudio
    short_name = dataset_name.split("/")[-1]
    print(f"  Streaming {short_name} ({config}) -- target {max_windows} windows ...")
    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=SR))

    speaker_buffers = defaultdict(lambda: np.array([], dtype=np.float32))
    windows = []

    for i, item in enumerate(ds):
        if len(windows) >= max_windows:
            break
        try:
            audio = item["audio"]["array"].astype(np.float32)
        except Exception:
            continue
        dur = len(audio) / SR
        if dur < min_sec or dur > max_sec:
            continue
        spk = str(item.get("speaker_id", i // 10))
        buf = np.concatenate([speaker_buffers[spk], audio])
        while len(buf) >= win_samples and len(windows) < max_windows:
            windows.append((buf[:win_samples], label_int))
            buf = buf[win_samples:]
        speaker_buffers[spk] = buf

    for buf in speaker_buffers.values():
        if len(windows) >= max_windows:
            break
        if len(buf) >= min_samples:
            pad = win_samples - len(buf)
            windows.append((np.concatenate([buf, buf[:pad][::-1]]), label_int))

    print(f"  -> {len(windows)} windows (nothing written to disk)")
    return windows


# =============================================================================
# 5. COMMONVOICE SPONTANEOUS (local folder)
# =============================================================================

def load_commonvoice_spont():
    cv_dir       = Path(COMMONVOICE_SPONT_DIR)
    manifest_tsv = cv_dir / "ss-corpus-en.tsv"
    reported_tsv = cv_dir / "ss-reported-audios-en.tsv"
    audios_dir   = cv_dir / "audios"

    if not manifest_tsv.exists():
        print(f"  WARN: CV Spont manifest not found -- skipping.")
        return pd.DataFrame(columns=["filepath", "filename", "source",
                                     "label", "label_int", "duration_sec", "speaker_id"])

    print(f"  Loading CommonVoice Spontaneous ...")
    df = pd.read_csv(manifest_tsv, sep="\t")

    if reported_tsv.exists():
        reported = set(pd.read_csv(reported_tsv, sep="\t", header=None)[0].astype(str))
        before = len(df)
        df = df[~df["audio_file"].astype(str).isin(reported)].reset_index(drop=True)
        print(f"  Excluded {before - len(df)} reported clips")

    df["filepath"]     = df["audio_file"].apply(lambda fn: str(audios_dir / fn))
    df["filename"]     = df["audio_file"].astype(str)
    df["source"]       = "cv_spont"
    df["label"]        = "spontaneous"
    df["label_int"]    = 0
    df["duration_sec"] = (df["duration_ms"] / 1000.0).round(3)
    df["speaker_id"]   = df["client_id"].astype(str)

    exists_mask = df["filepath"].apply(lambda p: Path(p).exists())
    if (~exists_mask).sum():
        print(f"  WARN: {(~exists_mask).sum()} files missing from disk -- dropping.")
    df = df[exists_mask].reset_index(drop=True)
    df = df[(df["duration_sec"] >= 3.0) & (df["duration_sec"] <= 22.0)].reset_index(drop=True)
    print(f"  {len(df)} clips loaded")
    return df[["filepath", "filename", "source", "label", "label_int",
               "duration_sec", "speaker_id"]]


# =============================================================================
# 6. PYTORCH DATASET  (handles both lazy tuples and eager numpy arrays)
# =============================================================================

class SpeechWindowDataset(Dataset):
    """
    Accepts a mixed list of:
      - Lazy items:  (filepath_str, start_sample, end_sample, label_int)
      - Eager items: (np.ndarray, label_int)
    Lazy items are loaded from disk in __getitem__ so audio never sits in RAM.
    """
    def __init__(self, windows, feature_extractor):
        self.windows = windows
        self.fe = feature_extractor

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        if isinstance(item[0], str):
            # Lazy: load slice from disk
            filepath, start, end, label = item
            y = load_audio_16k(filepath)
            if y is None:
                y = np.zeros(end - start, dtype=np.float32)
            audio = y[start:end]
            # Reflect-pad if file ended early
            if len(audio) < end - start:
                need = (end - start) - len(audio)
                audio = np.concatenate([audio, audio[:need][::-1]])
        else:
            # Eager: already a numpy array
            audio, label = item

        inputs = self.fe(audio, sampling_rate=SR, return_tensors="pt", padding=False)
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch])


# =============================================================================
# 7. MODEL
# =============================================================================

class WavLMClassifier(nn.Module):
    def __init__(self, wavlm_model_name, freeze_layers=6):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)
        for p in self.wavlm.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.wavlm.feature_projection.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.wavlm.encoder.layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        hidden = self.wavlm.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_values):
        pooled = self.wavlm(input_values).last_hidden_state.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


# =============================================================================
# 8. TRAINING
# =============================================================================

def get_param_groups(model):
    return [
        {"params": [p for p in model.wavlm.parameters() if p.requires_grad], "lr": WAVLM_LR},
        {"params": list(model.classifier.parameters()), "lr": HEAD_LR},
    ]


_criterion = nn.BCEWithLogitsLoss()


def run_epoch(model, loader, optimizer=None, scheduler=None, scaler=None, desc="train"):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    accum_step = 0
    n_batches  = len(loader)
    log_every  = max(1, n_batches // 5)   # print ~5 updates per epoch

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for step, (input_values, labels) in enumerate(loader, 1):
            input_values = input_values.to(DEVICE)
            labels       = labels.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(input_values)
                loss   = _criterion(logits, labels)
                if training:
                    loss = loss / GRAD_ACCUM
            if training:
                scaler.scale(loss).backward()
                accum_step += 1
                if accum_step % GRAD_ACCUM == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * (GRAD_ACCUM if training else 1)
            preds = (torch.sigmoid(logits.detach()) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))
            if step % log_every == 0 or step == n_batches:
                print(f"  {desc}  [{step}/{n_batches}]  loss={total_loss/step:.4f}",
                      flush=True)
        if training and accum_step % GRAD_ACCUM != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(loader), f1, acc


def train(train_windows, val_windows, feature_extractor):
    print(f"\n{'='*60}")
    print(f"TRAINING  ({len(train_windows)} train / {len(val_windows)} val windows)")
    lazy  = sum(1 for w in train_windows if isinstance(w[0], str))
    eager = len(train_windows) - lazy
    print(f"  Lazy (disk): {lazy}  |  Eager (RAM): {eager}")
    print(f"  Scripted:    {sum(1 for w in train_windows if w[-1]==1)}")
    print(f"  Spontaneous: {sum(1 for w in train_windows if w[-1]==0)}")
    print(f"{'='*60}\n")

    random.shuffle(train_windows)
    train_ds = SpeechWindowDataset(train_windows, feature_extractor)
    val_ds   = SpeechWindowDataset(val_windows,   feature_extractor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model     = WavLMClassifier(WAVLM_MODEL, freeze_layers=FREEZE_LAYERS).to(DEVICE)
    optimizer = torch.optim.AdamW(get_param_groups(model), weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_f1 = 0.0
    history = []
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_f1, tr_acc = run_epoch(model, train_loader, optimizer, scheduler, scaler,
                                            desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]")
        vl_loss, vl_f1, vl_acc = run_epoch(model, val_loader,
                                            desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]")
        history.append(dict(epoch=epoch,
                            tr_loss=round(tr_loss,4), tr_f1=round(tr_f1,4), tr_acc=round(tr_acc,4),
                            vl_loss=round(vl_loss,4), vl_f1=round(vl_f1,4), vl_acc=round(vl_acc,4)))
        print(f"Epoch {epoch:02d}  "
              f"train loss={tr_loss:.4f} F1={tr_f1:.4f} acc={tr_acc:.4f}  |  "
              f"val   loss={vl_loss:.4f} F1={vl_f1:.4f} acc={vl_acc:.4f}")
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            model.wavlm.save_pretrained(str(out))
            feature_extractor.save_pretrained(str(out))
            print(f"  Saved best model (val F1={vl_f1:.4f}) -> {out}")

    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val F1: {best_val_f1:.4f}  |  Model saved to: {out}")
    return history


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    win_samples = int(WINDOW_SEC  * SR)   # 160,000
    hop_samples = int(HOP_SEC     * SR)   #  80,000
    min_samples = int(MIN_WIN_SEC * SR)   # 128,000

    feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    print("\n[1/4] Indexing / loading data sources...")

    # -- CasualConversations: lazy, infer label from subfolder name -----------
    print("\nCasualConversations (lazy):")
    cc_root   = Path(CASUAL_CONV_DIR)
    cc_script_rows, cc_spont_rows = [], []
    for f in cc_root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXTS:
            continue
        parts_lower = " ".join(p.lower() for p in f.parts)
        if any(kw in parts_lower for kw in ("nonscript", "non_script", "spontaneous", "nonscripted")):
            cc_spont_rows.append({"filepath": str(f), "filename": f.name, "speaker_id": "cc"})
        elif any(kw in parts_lower for kw in ("scripted", "audio_scripted", "read")):
            cc_script_rows.append({"filepath": str(f), "filename": f.name, "speaker_id": "cc"})

    if not cc_script_rows and not cc_spont_rows:
        subdirs = sorted(set(f.parent.name for f in cc_root.rglob("*")
                             if f.is_file() and f.suffix.lower() in AUDIO_EXTS))
        print(f"  WARN: no CC files matched label keywords. Subdirs: {subdirs[:20]}")
        print(f"  Continuing without CC.")
    else:
        print(f"  Found {len(cc_script_rows)} scripted, {len(cc_spont_rows)} spontaneous files")

    cc_script_df = pd.DataFrame(cc_script_rows) if cc_script_rows else pd.DataFrame(
        columns=["filepath", "filename", "speaker_id"])
    cc_spont_df  = pd.DataFrame(cc_spont_rows)  if cc_spont_rows  else pd.DataFrame(
        columns=["filepath", "filename", "speaker_id"])

    cc_script_wins = build_lazy_windows_from_manifest(
        cc_script_df, label_int=1, win_samples=win_samples,
        hop_samples=hop_samples, min_samples=min_samples)
    cc_spont_wins = build_lazy_windows_from_manifest(
        cc_spont_df, label_int=0, win_samples=win_samples,
        hop_samples=hop_samples, min_samples=min_samples)
    print(f"  {len(cc_script_wins)} scripted lazy windows, {len(cc_spont_wins)} spont lazy windows")

    # -- AllStar: lazy --------------------------------------------------------
    print("\nAllStar 2677 (scripted, lazy):")
    allstar_script_wins = build_lazy_windows_from_folder(
        ALLSTAR_SCRIPT_DIR, label_int=1,
        win_samples=win_samples, hop_samples=hop_samples, min_samples=min_samples)
    print(f"  {len(allstar_script_wins)} lazy windows")

    print("\nAllStar 2676 (spontaneous, lazy):")
    allstar_spont_wins = build_lazy_windows_from_folder(
        ALLSTAR_SPONT_DIR, label_int=0,
        win_samples=win_samples, hop_samples=hop_samples, min_samples=min_samples)
    print(f"  {len(allstar_spont_wins)} lazy windows")

    # -- AMI + LibriSpeech: streaming, eager, capped --------------------------
    if USE_HF_DATASETS:
        print("\nAMI + LibriSpeech (streaming -- no disk writes):")
        ami_wins = stream_hf_to_windows(
            "edinburghcstr/ami", "ihm", "train", label_int=0,
            win_samples=win_samples, min_samples=min_samples,
            max_windows=MAX_INMEM_PER_SOURCE)
        libri_wins = stream_hf_to_windows(
            "openslr/librispeech_asr", "clean", "train.100", label_int=1,
            win_samples=win_samples, min_samples=min_samples,
            max_windows=MAX_INMEM_PER_SOURCE)
    else:
        print("\nSkipping AMI + LibriSpeech (USE_HF_DATASETS=False)")
        ami_wins, libri_wins = [], []

    # -- CommonVoice Spontaneous: eager, capped --------------------------------
    print("\nCommonVoice Spontaneous (local, eager):")
    cv_spont_df   = load_commonvoice_spont()
    cv_spont_wins = concat_short_clips(cv_spont_df, label_int=0,
                                        win_samples=win_samples, min_samples=min_samples,
                                        max_windows=MAX_INMEM_PER_SOURCE)
    print(f"  {len(cv_spont_wins)} windows")

    # -- Combine: AllStar (lazy, all) + equal sample per extra source ---------
    print("\n[2/4] Combining and balancing...")

    def equal_sample(sources, n_total):
        """Sample n_total/len(sources) items from each non-empty source."""
        active = [s for s in sources if s]
        if not active:
            return []
        per = n_total // len(active)
        out = []
        for s in active:
            out.extend(random.sample(s, min(per, len(s))))
        return out

    # Scripted: AllStar + (CC + Libri)
    n_extra_s   = MAX_INMEM_PER_SOURCE * 2   # budget for extra scripted sources
    extra_s     = equal_sample([cc_script_wins, libri_wins], n_extra_s)
    all_scripted = allstar_script_wins + extra_s

    # Spontaneous: AllStar + (CC + AMI + CV)
    n_extra_p   = MAX_INMEM_PER_SOURCE * 3   # budget for extra spont sources
    extra_p     = equal_sample([cc_spont_wins, ami_wins, cv_spont_wins], n_extra_p)
    all_spontaneous = allstar_spont_wins + extra_p

    lazy_s  = sum(1 for w in all_scripted    if isinstance(w[0], str))
    lazy_p  = sum(1 for w in all_spontaneous if isinstance(w[0], str))
    eager_s = len(all_scripted)    - lazy_s
    eager_p = len(all_spontaneous) - lazy_p
    print(f"  Scripted:    {len(all_scripted)}"
          f"  (lazy={lazy_s}, eager={eager_s})")
    print(f"  Spontaneous: {len(all_spontaneous)}"
          f"  (lazy={lazy_p}, eager={eager_p})")
    print(f"  Eager RAM estimate: ~{(eager_s+eager_p)*0.64/1024:.1f} GB")

    # -- Train / val split ----------------------------------------------------
    print("\n[3/4] Splitting train / val...")

    def split_val(wins, val_frac):
        wins = list(wins)
        random.shuffle(wins)
        n_val = max(50, int(len(wins) * val_frac))
        return wins[n_val:], wins[:n_val]

    train_s, val_s = split_val(all_scripted,    VAL_FRACTION)
    train_p, val_p = split_val(all_spontaneous, VAL_FRACTION)
    train_windows  = train_s + train_p
    val_windows    = val_s   + val_p
    print(f"  Train: {len(train_windows)}  Val: {len(val_windows)}")

    # -- Fine-tune ------------------------------------------------------------
    print("\n[4/4] Fine-tuning WavLM...")
    train(train_windows, val_windows, feature_extractor)
    print("\nDone! Download wavlm_finetuned/ from the Output tab.")


if __name__ == "__main__":
    main()
