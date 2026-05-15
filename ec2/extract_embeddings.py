"""Extract WavLM + Whisper embeddings for ALL audios + ALL augmentations, once.

Output: a single .npz file that downstream training iterations load instead
of re-running the encoders. Re-running with the same gt.csv + augs is a no-op;
adding new audio files or a new aug only extracts the missing combinations.

For each row in gt.csv and each aug in `--augs`:
    - load waveform from audio_npy/<npy_filename>
    - apply augmentation (no-op for 'orig')
    - WavLM-base-plus mean-pool        -> (768,)
    - Whisper-medium encoder mean-pool -> (1024,)

Augmentations:
    orig    no-op (always included)
    noise   AddGaussianSNR(10-25 dB)
    pitch   PitchShift +-2 semitones
    speed   TimeStretch 0.92-1.08
    gain    Gain +-9 dB
    air     AirAbsorption
    vtlp    Vocal Tract Length Perturbation (alpha in 0.9-1.1)
    combo   stochastic mix of (noise/pitch/speed/gain), each p=0.4-0.6
    rir     (optional) ApplyImpulseResponse  -- needs --rir_dir
    bgnoise (optional) AddBackgroundNoise    -- needs --noise_dir
    codec   (optional) Mp3Compression 32-64 kbps  -- if installed

Run:
    python extract_embeddings.py \
        --data_dir /path/to/upload \
        --out_path /path/to/embeddings_cache.npz \
        --augs orig,noise,pitch,speed,gain,air,vtlp,combo
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (WavLMModel, WhisperFeatureExtractor, WhisperModel)

WAVLM_ID = "microsoft/wavlm-base-plus"
WHISPER_ID = "openai/whisper-medium"
WAVLM_CHUNK_SEC = 20.0
WHISPER_CHUNK_SEC = 30.0
TARGET_SR = 16000
WAVLM_DIM = 768
WHISPER_DIM = 1024


# ----------------------------------------------------------------------------
# VTLP (custom; audiomentations doesn't include it).
# ----------------------------------------------------------------------------

def _vtlp_warp(x: np.ndarray, sr: int, alpha: float) -> np.ndarray:
    """Frequency-warping VTLP (Jaitly & Hinton 2013).

    Warps magnitude bins of an STFT by factor `alpha` (typically 0.9-1.1),
    preserves phase, ISTFTs back. Frequencies below `f_b = 4800*min(alpha,1)`
    Hz scale linearly; frequencies above are squeezed to keep f_max fixed."""
    n_fft = 512
    hop = 128
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    phase = np.angle(S)
    n_bins = mag.shape[0]
    f_max = sr / 2.0
    f_b = 4800.0 * min(alpha, 1.0)
    if f_b >= f_max:
        f_b = 0.8 * f_max
    new_mag = np.zeros_like(mag)
    for i in range(n_bins):
        f = i * f_max / (n_bins - 1)
        if f <= f_b:
            f_new = f * alpha
        else:
            f_new = f_max - (f_max - alpha * f_b) * (f_max - f) / (f_max - f_b)
        j = int(round(f_new * (n_bins - 1) / f_max))
        if 0 <= j < n_bins:
            new_mag[j] += mag[i]
    S_new = new_mag * np.exp(1j * phase)
    return librosa.istft(S_new, hop_length=hop, length=len(x)).astype(np.float32)


class VTLP:
    """audiomentations-compatible callable.  Call: aug(samples=wav, sample_rate=sr)."""

    def __init__(self, min_alpha: float = 0.9, max_alpha: float = 1.1, p: float = 1.0):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.p = p

    def __call__(self, *, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if np.random.random() > self.p:
            return samples
        alpha = float(np.random.uniform(self.min_alpha, self.max_alpha))
        return _vtlp_warp(samples, sample_rate, alpha)


# ----------------------------------------------------------------------------
# Augmenter pipelines.
# ----------------------------------------------------------------------------

def build_augmenters(names: list[str], rir_dir: Path | None, noise_dir: Path | None,
                     log: logging.Logger) -> dict[str, object]:
    """Map of aug_name -> callable. 'orig' is identity and handled by the caller."""
    from audiomentations import (
        AddGaussianSNR, AirAbsorption, Compose, Gain, PitchShift, TimeStretch,
    )

    available: dict[str, object] = {
        "noise": Compose([AddGaussianSNR(min_snr_db=10.0, max_snr_db=25.0, p=1.0)]),
        "pitch": Compose([PitchShift(min_semitones=-2, max_semitones=2, p=1.0)]),
        "speed": Compose([TimeStretch(min_rate=0.92, max_rate=1.08, p=1.0,
                                      leave_length_unchanged=False)]),
        "gain":  Compose([Gain(min_gain_db=-9.0, max_gain_db=9.0, p=1.0)]),
        "air":   Compose([AirAbsorption(p=1.0)]),
        "vtlp":  VTLP(min_alpha=0.9, max_alpha=1.1, p=1.0),
        "combo": Compose([
            AddGaussianSNR(min_snr_db=10.0, max_snr_db=25.0, p=0.6),
            PitchShift(min_semitones=-1, max_semitones=1, p=0.4),
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.4, leave_length_unchanged=False),
            Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.4),
        ]),
    }
    if rir_dir is not None and rir_dir.exists():
        from audiomentations import ApplyImpulseResponse
        available["rir"] = Compose([ApplyImpulseResponse(ir_path=str(rir_dir), p=1.0)])
        log.info("RIR aug enabled from %s", rir_dir)
    if noise_dir is not None and noise_dir.exists():
        from audiomentations import AddBackgroundNoise
        available["bgnoise"] = Compose([AddBackgroundNoise(
            sounds_path=str(noise_dir), min_snr_db=5.0, max_snr_db=20.0, p=1.0)])
        log.info("Background-noise aug enabled from %s", noise_dir)
    try:
        from audiomentations import Mp3Compression
        available["codec"] = Compose([Mp3Compression(min_bitrate=32, max_bitrate=64, p=1.0)])
    except Exception as e:
        log.info("codec aug unavailable (%s)", e)

    chosen: dict[str, object] = {}
    for n in names:
        if n == "orig":
            continue
        if n in available:
            chosen[n] = available[n]
        else:
            log.warning("aug '%s' not available -- ignoring (have: %s)",
                        n, sorted(available.keys()))
    log.info("active augmenters (%d): %s", len(chosen), list(chosen.keys()))
    return chosen


# ----------------------------------------------------------------------------
# Encoder forward passes.
# ----------------------------------------------------------------------------

@torch.no_grad()
def extract_wavlm_meanpool(wav: np.ndarray, model: WavLMModel,
                           device: torch.device,
                           layers: list[str]) -> dict[str, np.ndarray]:
    """Return {layer_tag: (768,)} for each requested layer.

    layer tags:
        'last' -> last_hidden_state
        '<int>' -> model.encoder hidden_states[<int>]; for WavLM-base-plus
                   valid indices are 0 (embedding output) ... 12 (last layer).
                   '9' is the commonly best layer for paralinguistic tasks.
    """
    chunk = int(WAVLM_CHUNK_SEC * TARGET_SR)
    chunks = [wav] if len(wav) <= chunk else [wav[i:i + chunk] for i in range(0, len(wav), chunk)]
    pools: dict[str, list[np.ndarray]] = {l: [] for l in layers}
    weights: list[int] = []
    for c in chunks:
        if len(c) < TARGET_SR * 0.4:
            continue
        x = torch.from_numpy(c).float().unsqueeze(0).to(device)
        out = model(x, output_hidden_states=True)
        weights.append(out.last_hidden_state.shape[1])
        for l in layers:
            if l == "last":
                layer_out = out.last_hidden_state
            else:
                layer_out = out.hidden_states[int(l)]
            pools[l].append(layer_out.mean(dim=1).squeeze(0).cpu().numpy())
    if not weights:
        zeros = np.zeros(model.config.hidden_size, dtype=np.float32)
        return {l: zeros.copy() for l in layers}
    W = np.asarray(weights, dtype=np.float32)
    result: dict[str, np.ndarray] = {}
    for l in layers:
        P = np.stack(pools[l], axis=0)
        result[l] = (P * (W[:, None] / W.sum())).sum(axis=0).astype(np.float32)
    return result


@torch.no_grad()
def extract_whisper_meanpool(wav: np.ndarray, model: WhisperModel,
                             feat: WhisperFeatureExtractor,
                             device: torch.device) -> np.ndarray:
    chunk = int(WHISPER_CHUNK_SEC * TARGET_SR)
    chunks = [wav] if len(wav) <= chunk else [wav[i:i + chunk] for i in range(0, len(wav), chunk)]
    pooled = []
    for c in chunks:
        if len(c) < TARGET_SR * 0.4:
            continue
        feats = feat(c, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(device)
        enc = model.encoder(feats).last_hidden_state
        pooled.append(enc.mean(dim=1).squeeze(0).cpu().numpy())
    if not pooled:
        return np.zeros(model.config.d_model, dtype=np.float32)
    return np.stack(pooled, axis=0).mean(axis=0).astype(np.float32)


# ----------------------------------------------------------------------------
# Cache I/O.
# ----------------------------------------------------------------------------

def load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def save_cache(path: Path, filenames: np.ndarray, aug_names: list[str],
               wavlm_layers: list[str],
               wavlm: dict[tuple[str, str], np.ndarray],
               whisper: dict[str, np.ndarray]) -> None:
    out: dict[str, np.ndarray] = {
        "filenames": filenames,
        "aug_names": np.array(aug_names, dtype=object),
        "wavlm_layers": np.array(wavlm_layers, dtype=object),
    }
    for layer in wavlm_layers:
        for a in aug_names:
            out[f"wavlm_{layer}_{a}"] = wavlm[(layer, a)]
    for a in aug_names:
        out[f"whisper_{a}"] = whisper[a]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out_dir / "extract_log.txt", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="folder containing gt.csv and audio_npy/")
    ap.add_argument("--out_path", required=True,
                    help="path to write embeddings_cache.npz")
    ap.add_argument("--augs",
                    default="orig,noise,pitch,speed,gain,air,vtlp,combo",
                    help="comma-separated aug names; 'orig' is always included")
    ap.add_argument("--wavlm_layers", default="last,9",
                    help="comma-separated WavLM layer tags to extract. "
                         "'last' = last_hidden_state. Integers = hidden_states[i]. "
                         "Layer 9 is commonly best for paralinguistics.")
    ap.add_argument("--rir_dir", default="")
    ap.add_argument("--noise_dir", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="ignore existing cache, re-extract from scratch")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_path = Path(args.out_path)
    log = setup_logging(out_path.parent)
    log.info("data_dir = %s", data_dir)
    log.info("out_path = %s", out_path)

    requested = [a.strip() for a in args.augs.split(",") if a.strip()]
    if "orig" not in requested:
        requested = ["orig"] + requested
    seen: set[str] = set()
    requested = [a for a in requested if not (a in seen or seen.add(a))]
    log.info("requested aug_names = %s", requested)

    layers = [l.strip() for l in args.wavlm_layers.split(",") if l.strip()]
    if "last" not in layers:
        layers = ["last"] + layers
    seen.clear()
    layers = [l for l in layers if not (l in seen or seen.add(l))]
    log.info("wavlm_layers = %s", layers)

    gt = pd.read_csv(data_dir / "gt.csv")
    filenames_cur = gt["npy_filename"].to_numpy().astype(str)
    log.info("gt rows = %d", len(filenames_cur))

    rir_dir = Path(args.rir_dir) if args.rir_dir else None
    noise_dir = Path(args.noise_dir) if args.noise_dir else None
    augs_callable = build_augmenters(requested, rir_dir, noise_dir, log)

    aug_names = ["orig"] + [a for a in requested if a != "orig" and a in augs_callable]
    log.info("aug_names (after availability check) = %s", aug_names)

    cache = None if args.force else load_cache(out_path)

    # wavlm_data keyed by (layer, aug); whisper_data keyed by aug.
    wavlm_data: dict[tuple[str, str], np.ndarray] = {}
    whisper_data: dict[str, np.ndarray] = {}

    if cache is not None:
        cached_filenames = cache["filenames"].astype(str)
        cached_augs = list(cache["aug_names"].astype(str))
        # Backward-compat: a cache written by the old single-layer code has no
        # 'wavlm_layers' key and stores 'wavlm_<aug>' (which was effectively the
        # last layer). Treat that as if it were layer='last'.
        if "wavlm_layers" in cache:
            cached_layers = list(cache["wavlm_layers"].astype(str))
        else:
            cached_layers = ["last"]
            log.info("cache is old-format (no wavlm_layers key); treating as layer='last'")
            for a in cached_augs:
                old_key = f"wavlm_{a}"
                new_key = f"wavlm_last_{a}"
                if old_key in cache and new_key not in cache:
                    cache[new_key] = cache[old_key]
        log.info("cache: %d filenames, %d augs (%s), %d layers (%s)",
                 len(cached_filenames), len(cached_augs), cached_augs,
                 len(cached_layers), cached_layers)
        f2c = {f: i for i, f in enumerate(cached_filenames)}

        for layer in layers:
            for a in aug_names:
                wavlm_data[(layer, a)] = np.zeros((len(filenames_cur), WAVLM_DIM),
                                                  dtype=np.float32)
                key = f"wavlm_{layer}_{a}"
                if layer in cached_layers and a in cached_augs and key in cache:
                    src = cache[key]
                    for new_idx, fn in enumerate(filenames_cur):
                        if fn in f2c:
                            wavlm_data[(layer, a)][new_idx] = src[f2c[fn]]

        for a in aug_names:
            whisper_data[a] = np.zeros((len(filenames_cur), WHISPER_DIM), dtype=np.float32)
            key = f"whisper_{a}"
            if a in cached_augs and key in cache:
                src = cache[key]
                for new_idx, fn in enumerate(filenames_cur):
                    if fn in f2c:
                        whisper_data[a][new_idx] = src[f2c[fn]]
    else:
        log.info("no existing cache (or --force); cold extraction")
        for layer in layers:
            for a in aug_names:
                wavlm_data[(layer, a)] = np.zeros((len(filenames_cur), WAVLM_DIM),
                                                  dtype=np.float32)
        for a in aug_names:
            whisper_data[a] = np.zeros((len(filenames_cur), WHISPER_DIM), dtype=np.float32)

    # Per file, determine which augs need any encoder work. Then for each such
    # aug, figure out which wavlm layers + whisper still need extraction.
    # We load the wav once per file and apply each aug once per file.
    by_file: dict[str, set[str]] = {}
    for layer in layers:
        for a in aug_names:
            zero_mask = (np.abs(wavlm_data[(layer, a)]).sum(axis=1) == 0)
            for idx in np.where(zero_mask)[0]:
                by_file.setdefault(filenames_cur[idx], set()).add(a)
    for a in aug_names:
        zero_mask = (np.abs(whisper_data[a]).sum(axis=1) == 0)
        for idx in np.where(zero_mask)[0]:
            by_file.setdefault(filenames_cur[idx], set()).add(a)

    log.info("files needing extraction = %d", len(by_file))
    if not by_file:
        save_cache(out_path, filenames_cur, aug_names, layers, wavlm_data, whisper_data)
        log.info("Cache up-to-date. Saved (possibly trimmed to current gt) to %s", out_path)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device = %s", device)
    log.info("Loading WavLM (%s)...", WAVLM_ID)
    wavlm = WavLMModel.from_pretrained(WAVLM_ID).to(device).eval()
    log.info("Loading Whisper (%s)...", WHISPER_ID)
    whisper = WhisperModel.from_pretrained(WHISPER_ID).to(device).eval()
    whisper_feat = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

    npy_dir = data_dir / "audio_npy"
    fn_to_row = {fn: i for i, fn in enumerate(filenames_cur)}

    t0 = time.time()
    for fn, augs_for_file in tqdm(list(by_file.items()), desc="extract", unit="audio"):
        try:
            wav = np.load(npy_dir / fn).astype(np.float32, copy=False)
        except Exception as e:
            log.warning("load failed for %s: %s -- leaving zeros", fn, e)
            continue
        row_idx = fn_to_row[fn]
        for a in sorted(augs_for_file):
            if a == "orig":
                wav_a = wav
            else:
                try:
                    wav_a = augs_callable[a](samples=wav, sample_rate=TARGET_SR)
                except Exception as e:
                    log.warning("aug %s failed on %s: %s -- using orig", a, fn, e)
                    wav_a = wav
                if wav_a.dtype != np.float32:
                    wav_a = wav_a.astype(np.float32, copy=False)
            # Identify which wavlm layers are still zero at this (layer, aug).
            layers_needed = [l for l in layers
                             if np.abs(wavlm_data[(l, a)][row_idx]).sum() == 0]
            if layers_needed:
                res = extract_wavlm_meanpool(wav_a, wavlm, device, layers_needed)
                for l, emb in res.items():
                    wavlm_data[(l, a)][row_idx] = emb
            if np.abs(whisper_data[a][row_idx]).sum() == 0:
                whisper_data[a][row_idx] = extract_whisper_meanpool(
                    wav_a, whisper, whisper_feat, device)

    log.info("extraction done in %.1f s", time.time() - t0)
    save_cache(out_path, filenames_cur, aug_names, layers, wavlm_data, whisper_data)
    log.info("Saved cache to %s  (filenames=%d, augs=%d, layers=%d)",
             out_path, len(filenames_cur), len(aug_names), len(layers))

    del wavlm, whisper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
