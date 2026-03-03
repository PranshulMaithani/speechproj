"""
Portable Inference Script — Read vs Spontaneous Speech
========================================================
Designed for deployment on CPU-only machines (no GPU required).

MODES (in order of speed on CPU):
  --mode xgboost        Fastest   (~seconds per file) — uses handcrafted features
  --mode onnx           Fast      (~10-30s per file)  — quantized Wav2Vec2 INT8
  --mode pytorch        Medium    (~1-3 min per file) — full PyTorch on CPU
  --mode ensemble       Accurate  — combines onnx/pytorch + xgboost

SETUP on laptop:
  1. Copy this project folder (or at minimum: src/, configs/, checkpoints/, venv/)
  2. pip install -r requirements_inference.txt
  3. python infer.py --audio path/to/wavs/ --mode onnx

SUPPORTS: .wav .mp3 .m4a .webm .ogg .flac .aac .wma (requires ffmpeg for non-wav)
"""

import os
import sys
import json
import pickle
import argparse
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict

import yaml
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm import tqdm

# Suppress noisy warnings from librosa/audioread/transformers
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Supported audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".aac", ".wma", ".mp4"}


# ============================================================
# Config
# ============================================================

def load_cfg(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================
# Audio utilities (standalone, no src/ import needed if copied)
# ============================================================

def load_audio(path: str, target_sr: int = 16000, max_duration: float = 120.0):
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_duration)
    # Peak-normalize so VAD works regardless of codec/format
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio


def simple_vad(audio: np.ndarray, sr: int = 16000) -> float:
    """Return fraction of audio that is speech (0..1), using relative threshold."""
    import librosa
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
    rms_max = float(np.max(rms))
    if rms_max < 1e-8:
        return 0.0
    noise_floor = np.percentile(rms, 10)
    signal_peak = np.percentile(rms, 95)
    thresh = noise_floor + (signal_peak - noise_floor) * 0.15
    thresh = max(thresh, rms_max * 0.05)
    return float(np.mean(rms > thresh))


def make_windows(audio: np.ndarray, sr: int, window_sec: float = 5.0, hop_sec: float = 2.5):
    """Yield (chunk, start_sec, end_sec, speech_ratio) tuples."""
    win_samp = int(window_sec * sr)
    hop_samp = int(hop_sec * sr)
    start = 0
    while start < len(audio):
        end = start + win_samp
        chunk = audio[start:end]
        if len(chunk) < win_samp // 2:
            break
        if len(chunk) < win_samp:
            chunk = np.pad(chunk, (0, win_samp - len(chunk)))
        speech_ratio = simple_vad(chunk, sr)
        yield chunk, start / sr, end / sr, speech_ratio
        start += hop_samp


# ============================================================
# Model backends
# ============================================================

class ONNXBackend:
    def __init__(self, onnx_path: str, num_threads: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if num_threads > 0:
            opts.inter_op_num_threads = num_threads
            opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        print(f"[ONNX] Loaded: {Path(onnx_path).name}")

    def predict_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """waveforms: (B, 80000) → probs: (B, 2)"""
        logits = self.sess.run(["logits"], {"input_values": waveforms.astype(np.float32)})[0]
        probs = _softmax(logits)
        return probs


class PyTorchBackend:
    def __init__(self, ckpt_path: str, cfg: dict, num_threads: int = 0):
        import torch
        from src.models.train_wav2vec2 import SpeechClassifier

        if num_threads > 0:
            torch.set_num_threads(num_threads)

        device = torch.device("cpu")
        w2v_cfg = cfg["training"]["wav2vec2"]
        self.model = SpeechClassifier(
            model_name=w2v_cfg["model_name"],
            hidden_size=w2v_cfg["hidden_size"],
            num_labels=2,
            dropout=0.0,
            freeze_layers=0,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._torch = torch
        print(f"[PyTorch CPU] Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    def predict_batch(self, waveforms: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            t = self._torch.tensor(waveforms, dtype=self._torch.float32)
            logits = self.model(t)
            probs = self._torch.nn.functional.softmax(logits, dim=-1).numpy()
        return probs


class XGBoostBackend:
    def __init__(self, model_path: str, scaler_path: str, feat_cfg: dict, audio_cfg: dict):
        import xgboost as xgb
        from src.features.extract_features import extract_all_features

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.feat_cfg = feat_cfg
        self.audio_cfg = audio_cfg
        self._extract = extract_all_features
        print(f"[XGBoost CPU] Loaded: {Path(model_path).name}")

    def predict_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """Process each window individually (XGBoost is fast, no batching needed)."""
        probs_list = []
        for wav in waveforms:
            feats = self._extract(
                wav, 16000,
                f0_min=self.feat_cfg["f0_min"],
                f0_max=self.feat_cfg["f0_max"],
                n_mfcc=self.feat_cfg["n_mfcc"],
                energy_threshold=self.audio_cfg["vad_energy_threshold"],
            )
            fv = np.array([feats[k] for k in sorted(feats.keys())], dtype=np.float32)
            fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
            fv_scaled = self.scaler.transform(fv.reshape(1, -1))
            probs_list.append(self.model.predict_proba(fv_scaled)[0])
        return np.array(probs_list)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================
# Core prediction logic
# ============================================================

def predict_file(
    audio_path: str,
    backends: list,
    backend_weights: list[float],
    cfg: dict,
    batch_size: int = 8,
) -> dict:
    audio_cfg = cfg["audio"]
    inf_cfg = cfg["inference"]
    sr = audio_cfg["sample_rate"]
    min_speech = audio_cfg["min_speech_ratio"]

    # Load audio (handles wav, m4a, mp3, etc.)
    audio = load_audio(audio_path, target_sr=sr, max_duration=audio_cfg.get("max_duration_sec", 120))
    total_duration = len(audio) / sr

    # Make windows
    all_windows = list(make_windows(
        audio, sr,
        window_sec=audio_cfg["window_sec"],
        hop_sec=audio_cfg["hop_sec"],
    ))

    if not all_windows:
        return _empty_result(audio_path, total_duration)

    chunks = np.stack([w[0] for w in all_windows])  # (N, 80000)
    starts = [w[1] for w in all_windows]
    ends   = [w[2] for w in all_windows]
    speech_ratios = [w[3] for w in all_windows]

    # Batch inference through each backend
    combined_probs = None
    for backend, weight in zip(backends, backend_weights):
        probs = []
        for i in range(0, len(chunks), batch_size):
            probs.append(backend.predict_batch(chunks[i : i + batch_size]))
        probs = np.concatenate(probs, axis=0)  # (N, 2)
        if combined_probs is None:
            combined_probs = weight * probs
        else:
            combined_probs += weight * probs

    # Build window predictions
    window_preds = []
    for i, (start, end, speech_ratio) in enumerate(zip(starts, ends, speech_ratios)):
        if speech_ratio < min_speech:
            label, conf = "silence", 1.0
        else:
            pred_cls = int(np.argmax(combined_probs[i]))
            label = "spontaneous" if pred_cls == 0 else "read"
            conf = float(combined_probs[i][pred_cls])
        window_preds.append({
            "start_sec": round(start, 2),
            "end_sec":   round(end, 2),
            "label":     label,
            "confidence": round(conf, 3),
            "speech_ratio": round(speech_ratio, 3),
        })

    # Temporal smoothing on speaking windows
    speaking_idx = [i for i, wp in enumerate(window_preds) if wp["label"] != "silence"]
    if len(speaking_idx) >= inf_cfg["temporal_smooth_window"]:
        labels_num = np.array([
            0 if window_preds[i]["label"] == "spontaneous" else 1
            for i in speaking_idx
        ])
        smoothed = median_filter(labels_num, size=inf_cfg["temporal_smooth_window"]).astype(int)
        for j, i in enumerate(speaking_idx):
            window_preds[i]["label"] = "spontaneous" if smoothed[j] == 0 else "read"

    # Build segments (merge consecutive same-label windows)
    segments = _merge_segments(window_preds)

    # Overall label
    speaking_windows = [wp for wp in window_preds if wp["label"] != "silence"]
    if not speaking_windows:
        overall_label, overall_conf, read_ratio = "silence", 1.0, 0.0
    else:
        read_count = sum(1 for wp in speaking_windows if wp["label"] == "read")
        read_ratio = read_count / len(speaking_windows)
        overall_label = "read" if read_ratio >= inf_cfg["read_threshold"] else "spontaneous"
        same_label_windows = [wp for wp in speaking_windows if wp["label"] == overall_label]
        overall_conf = float(np.mean([wp["confidence"] for wp in same_label_windows]))

    return {
        "filepath": audio_path,
        "filename": Path(audio_path).name,
        "duration_sec": round(total_duration, 2),
        "overall_label": overall_label,
        "overall_confidence": round(overall_conf, 3),
        "read_ratio": round(read_ratio, 3),
        "cheating_suspected": overall_label == "read",
        "segments": segments,
        "window_predictions": window_preds,
    }


def _merge_segments(window_preds: list[dict]) -> list[dict]:
    if not window_preds:
        return []
    segments = []
    cur = window_preds[0]
    seg_start = cur["start_sec"]
    seg_confs = [cur["confidence"]]
    for wp in window_preds[1:]:
        if wp["label"] != cur["label"]:
            # End the segment at the START of the new window (no overlap)
            seg_end = wp["start_sec"]
            segments.append({
                "start_sec": seg_start,
                "end_sec":   seg_end,
                "duration_sec": round(seg_end - seg_start, 2),
                "label": cur["label"],
                "confidence": round(float(np.mean(seg_confs)), 3),
            })
            seg_start = wp["start_sec"]
            seg_confs = [wp["confidence"]]
            cur = wp
        else:
            seg_confs.append(wp["confidence"])
            cur = wp
    # Final segment ends at last window's end
    segments.append({
        "start_sec": seg_start,
        "end_sec":   cur["end_sec"],
        "duration_sec": round(cur["end_sec"] - seg_start, 2),
        "label": cur["label"],
        "confidence": round(float(np.mean(seg_confs)), 3),
    })
    return segments


def _empty_result(path, duration):
    return {
        "filepath": path, "filename": Path(path).name,
        "duration_sec": round(duration, 2),
        "overall_label": "silence", "overall_confidence": 1.0,
        "read_ratio": 0.0, "cheating_suspected": False,
        "segments": [], "window_predictions": [],
    }


def print_result(r: dict):
    icon = "■ READ  (⚠ cheating suspected)" if r["overall_label"] == "read" else "□ Spontaneous"
    print(f"\n{'='*62}")
    print(f"  {Path(r['filepath']).name}")
    print(f"  {r['duration_sec']}s  →  {icon}")
    print(f"  Confidence: {r['overall_confidence']:.0%}   Read ratio: {r['read_ratio']:.0%}")
    print(f"  Timeline:")
    for s in r["segments"]:
        sym = "■" if s["label"] == "read" else ("□" if s["label"] == "spontaneous" else "·")
        print(f"    {sym} [{s['start_sec']:6.1f}s – {s['end_sec']:5.1f}s]  "
              f"{s['label']:12s}  conf {s['confidence']:.0%}  "
              f"({s['duration_sec']:.0f}s)")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classify read vs spontaneous speech (CPU-friendly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--audio", required=True,
                        help="Path to a .wav/.m4a/.mp3 file, or a folder of audio files")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default="onnx",
                        choices=["onnx", "pytorch", "xgboost", "ensemble"])
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Windows per inference batch (lower = less RAM)")
    parser.add_argument("--threads", type=int, default=0,
                        help="CPU threads (0 = auto-detect)")
    parser.add_argument("--output", default=None,
                        help="Path for output JSON (default: outputs/predictions_<timestamp>.json)")
    parser.add_argument("--csv", action="store_true",
                        help="Also save a summary CSV alongside the JSON")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    ckpt_dir = data_root / cfg["paths"]["checkpoints_dir"]

    threads = args.threads if args.threads > 0 else max(1, os.cpu_count() - 1)
    print(f"CPU threads: {threads}  |  Mode: {args.mode}  |  Batch size: {args.batch_size}")

    # ---- Load backends ----
    backends, weights = [], []

    use_deep = args.mode in ("onnx", "pytorch", "ensemble")
    use_xgb  = args.mode in ("xgboost", "ensemble")

    if use_deep:
        if args.mode in ("onnx", "ensemble"):
            # Prefer quantized ONNX, fall back to fp32, then PyTorch
            quant_path = ckpt_dir / "speech_classifier_quant.onnx"
            fp32_path  = ckpt_dir / "speech_classifier.onnx"
            try:
                if quant_path.exists():
                    backends.append(ONNXBackend(str(quant_path), threads))
                elif fp32_path.exists():
                    backends.append(ONNXBackend(str(fp32_path), threads))
                    print("  (quantized ONNX not found, using fp32 — run export_onnx.py for faster inference)")
                else:
                    raise FileNotFoundError("No ONNX model found")
                weights.append(0.6 if use_xgb else 1.0)
            except Exception as e:
                print(f"  ONNX backend failed ({e}), falling back to PyTorch")
                args.mode = "pytorch" if not use_xgb else "ensemble_pt"
                use_deep = False

        if args.mode in ("pytorch",) or (args.mode == "ensemble" and not backends):
            ckpt_path = ckpt_dir / "wav2vec2_best.pt"
            try:
                backends.append(PyTorchBackend(str(ckpt_path), cfg, threads))
                weights.append(0.6 if use_xgb else 1.0)
            except Exception as e:
                print(f"  PyTorch backend failed: {e}")

    if use_xgb:
        xgb_model_path  = ckpt_dir / "xgboost_baseline.json"
        xgb_scaler_path = ckpt_dir / "xgboost_scaler.pkl"
        try:
            backends.append(XGBoostBackend(
                str(xgb_model_path), str(xgb_scaler_path),
                cfg["features"], cfg["audio"],
            ))
            weights.append(0.4 if len(weights) > 0 else 1.0)
        except Exception as e:
            print(f"  XGBoost backend failed: {e}")

    if not backends:
        print("ERROR: No models loaded. Check that checkpoints exist.")
        sys.exit(1)

    # Normalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    print(f"Loaded {len(backends)} backend(s), weights: {[round(w,2) for w in weights]}")

    # ---- Collect audio files ----
    audio_arg = Path(args.audio)
    if audio_arg.is_file():
        audio_files = [audio_arg]
    elif audio_arg.is_dir():
        audio_files = sorted([
            f for f in audio_arg.iterdir()
            if f.suffix.lower() in AUDIO_EXTS
        ])
        print(f"Found {len(audio_files)} audio files in {audio_arg}")
    else:
        print(f"ERROR: {audio_arg} not found")
        sys.exit(1)

    if not audio_files:
        print("No audio files found. Supported: .wav .mp3 .m4a .webm .ogg .flac")
        sys.exit(1)

    # ---- Run inference ----
    results = []
    t_start = time.perf_counter()

    for fpath in tqdm(audio_files, desc="Processing", unit="file"):
        t0 = time.perf_counter()
        try:
            result = predict_file(
                str(fpath), backends, weights, cfg, batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"\n  ERROR processing {fpath.name}: {e}")
            result = _empty_result(str(fpath), 0.0)
            result["error"] = str(e)
        result["processing_sec"] = round(time.perf_counter() - t0, 2)
        results.append(result)
        print_result(result)

    total_elapsed = time.perf_counter() - t_start

    # ---- Summary ----
    print(f"\n{'='*62}")
    print(f"SUMMARY — {len(results)} files processed in {total_elapsed/60:.1f} min")
    print(f"{'='*62}")
    read_n = sum(1 for r in results if r["overall_label"] == "read")
    spont_n = sum(1 for r in results if r["overall_label"] == "spontaneous")
    print(f"  ■ CHEATING SUSPECTED (read): {read_n}")
    print(f"  □ Spontaneous:               {spont_n}")
    if results:
        avg_conf = np.mean([r["overall_confidence"] for r in results])
        print(f"  Average confidence: {avg_conf:.1%}")
        print(f"  Average time/file:  {total_elapsed/len(results):.1f}s")

    # ---- Save outputs ----
    out_dir = data_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        json_path = Path(args.output)
    else:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"predictions_{ts}.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed JSON saved: {json_path}")

    if args.csv:
        csv_path = json_path.with_suffix(".csv")
        summary_cols = [
            "filename", "duration_sec", "overall_label", "overall_confidence",
            "read_ratio", "cheating_suspected", "processing_sec",
        ]
        pd.DataFrame([
            {k: r.get(k, "") for k in summary_cols} for r in results
        ]).to_csv(csv_path, index=False)
        print(f"Summary CSV saved:   {csv_path}")


if __name__ == "__main__":
    main()
