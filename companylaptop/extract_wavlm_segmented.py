"""
Segmented WavLM Embedding Extraction
=====================================
Replaces the whole-audio mean-pool (768-dim) with 10-second window
statistical pooling (mean + std = 1536-dim).

Run this LOCALLY after downloading the fine-tuned WavLM encoder from Colab.
If you don't have the fine-tuned model yet, point WAVLM_MODEL_PATH to
"microsoft/wavlm-base-plus" to use the pretrained baseline first.

Usage:
    python extract_wavlm_segmented.py --audio_dir audios2 --out audios2_wavlm_seg.csv
    python extract_wavlm_segmented.py --audio_dir audios4 --out audios4_wavlm_seg.csv
    python extract_wavlm_segmented.py --audio_dir audios5 --out audios5_wavlm_seg.csv

Output CSV columns:
    filename, wavlm_mean_0 … wavlm_mean_767, wavlm_std_0 … wavlm_std_767
    → 1 + 768 + 768 = 1537 columns total
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change WAVLM_MODEL_PATH once you have the fine-tuned weights
# ─────────────────────────────────────────────────────────────────────────────

# Before Colab fine-tuning: use pretrained baseline
WAVLM_MODEL_PATH = "microsoft/wavlm-base-plus"

# After downloading from Colab: use fine-tuned weights
# WAVLM_MODEL_PATH = r"path\to\wavlm_finetuned"

WINDOW_SEC  = 10      # seconds per window
HOP_SEC     = 5       # hop (5s overlap between consecutive windows)
MIN_WIN_SEC = 4       # discard windows shorter than this
SR          = 16000
AUDIO_EXTS  = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# ─────────────────────────────────────────────────────────────────────────────

def load_audio_16k(path):
    try:
        y, sr = sf.read(str(path), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
        return y.astype(np.float32)
    except Exception as e:
        print(f"  WARN load {path.name}: {e}")
        return None


def get_windows(y, win_samples, hop_samples, min_samples):
    """Slice audio into overlapping windows. Returns list of np.arrays."""
    if len(y) < min_samples:
        # Too short even for one window: return as-is (model handles variable length)
        return [y]
    windows = []
    start = 0
    while start + win_samples <= len(y):
        windows.append(y[start : start + win_samples])
        start += hop_samples
    # Last tail: keep if long enough
    tail = y[start:]
    if len(tail) >= min_samples:
        windows.append(tail)
    # Edge case: audio shorter than one full window but longer than min
    if not windows:
        windows.append(y)
    return windows


@torch.no_grad()
def embed_windows(windows, model, feature_extractor, device):
    """
    Run WavLM on each window, mean-pool over time → list of 768-dim vectors.
    Windows are processed individually (no padding issues, no batching needed
    since audio length varies for the last window).
    """
    embeddings = []
    for win in windows:
        inputs = feature_extractor(win, sampling_rate=SR, return_tensors="pt",
                                   padding=False)
        input_values = inputs.input_values.to(device)
        out = model(input_values)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()  # [768]
        embeddings.append(emb)
    return np.array(embeddings)   # [N_windows, 768]


def extract_folder(audio_dir, out_csv, model, feature_extractor, device):
    audio_dir = Path(audio_dir).resolve()
    if not audio_dir.exists():
        print(f"ERROR: {audio_dir} does not exist")
        return

    audio_files = sorted(f for f in audio_dir.rglob("*")
                         if f.suffix.lower() in AUDIO_EXTS and f.is_file())
    print(f"\nExtracting from {audio_dir.name}: {len(audio_files)} files")
    print(f"  Model: {WAVLM_MODEL_PATH}")
    print(f"  Window: {WINDOW_SEC}s, hop: {HOP_SEC}s")
    print(f"  Output: {out_csv}")

    win_samples = int(WINDOW_SEC * SR)
    hop_samples = int(HOP_SEC    * SR)
    min_samples = int(MIN_WIN_SEC * SR)

    rows = []
    n_windows_total = 0

    for fp in tqdm(audio_files, desc=audio_dir.name):
        y = load_audio_16k(fp)
        if y is None:
            continue

        windows = get_windows(y, win_samples, hop_samples, min_samples)
        embs    = embed_windows(windows, model, feature_extractor, device)  # [N, 768]

        n_windows_total += len(embs)

        # Statistical pooling across windows
        mean_emb = embs.mean(axis=0)    # [768]
        std_emb  = embs.std(axis=0)     # [768] — consistency signal

        row = {"filename": fp.name}
        for i, v in enumerate(mean_emb): row[f"wavlm_mean_{i}"] = round(float(v), 6)
        for i, v in enumerate(std_emb):  row[f"wavlm_std_{i}"]  = round(float(v), 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  Saved {len(df)} files ({n_windows_total} total windows) → {out_csv}")
    print(f"  Feature shape per file: {len(df.columns)-1} dims (768 mean + 768 std)")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Folder with audio files")
    parser.add_argument("--out",       required=True, help="Output CSV path")
    parser.add_argument("--model",     default=WAVLM_MODEL_PATH,
                        help="HuggingFace model name or local path to fine-tuned encoder")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading WavLM from: {args.model}")
    fe    = AutoFeatureExtractor.from_pretrained(args.model)
    model = WavLMModel.from_pretrained(args.model).eval().to(device)

    extract_folder(args.audio_dir, args.out, model, fe, device)


if __name__ == "__main__":
    main()
