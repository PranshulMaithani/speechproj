"""
Inference Pipeline — Read vs Spontaneous Speech Detection
============================================================
Takes a .wav file (or folder), runs sliding-window classification,
produces:
  1. Per-window predictions with confidence
  2. Segment-level timeline (grouped consecutive predictions)
  3. Binary overall label with aggregate confidence
  4. JSON report per file

Supports both Wav2Vec2 and XGBoost models, plus ensemble mode.

Usage:
    # Single file
    python -m src.inference.predict --audio path/to/audio.wav --config configs/config.yaml

    # Folder of files
    python -m src.inference.predict --audio path/to/folder/ --config configs/config.yaml

    # Ensemble mode (both models)
    python -m src.inference.predict --audio path/to/audio.wav --mode ensemble
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import yaml
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from tqdm import tqdm

from src.data.audio_utils import load_audio, window_audio, get_speaking_segments
from src.features.extract_features import extract_all_features
from src.models.train_wav2vec2 import SpeechClassifier


# ============================================================
# Data Structures
# ============================================================

@dataclass
class WindowPrediction:
    window_idx: int
    start_sec: float
    end_sec: float
    label: str              # "read" | "spontaneous" | "silence"
    confidence: float       # 0-1
    speech_ratio: float


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    duration_sec: float
    label: str
    confidence: float       # Mean confidence of constituent windows


@dataclass
class PredictionResult:
    filepath: str
    duration_sec: float
    overall_label: str      # "read" | "spontaneous"
    overall_confidence: float
    read_ratio: float       # Fraction of speaking time classified as read
    segments: list           # List of Segment dicts
    window_predictions: list # List of WindowPrediction dicts
    model_used: str          # "wav2vec2" | "xgboost" | "ensemble"


# ============================================================
# Model Loading
# ============================================================

def load_wav2vec2_model(cfg: dict, device: torch.device) -> SpeechClassifier:
    """Load trained Wav2Vec2 classifier from checkpoint."""
    data_root = Path(cfg["paths"]["data_root"])
    ckpt_path = data_root / cfg["paths"]["checkpoints_dir"] / "wav2vec2_best.pt"

    w2v_cfg = cfg["training"]["wav2vec2"]
    model = SpeechClassifier(
        model_name=w2v_cfg["model_name"],
        hidden_size=w2v_cfg["hidden_size"],
        num_labels=2,
        dropout=0.0,  # No dropout at inference
        freeze_layers=0,  # Doesn't matter at inference
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded Wav2Vec2 model from: {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
    return model


def load_xgboost_model(cfg: dict):
    """Load trained XGBoost model and scaler."""
    import xgboost as xgb

    data_root = Path(cfg["paths"]["data_root"])
    ckpt_dir = data_root / cfg["paths"]["checkpoints_dir"]

    model = xgb.XGBClassifier()
    model.load_model(str(ckpt_dir / "xgboost_baseline.json"))

    with open(ckpt_dir / "xgboost_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"Loaded XGBoost model from: {ckpt_dir}")
    return model, scaler


# ============================================================
# Window-Level Prediction
# ============================================================

@torch.no_grad()
def predict_windows_wav2vec2(
    windows: list,
    model: SpeechClassifier,
    device: torch.device,
    batch_size: int = 8,
) -> list[tuple[np.ndarray, float]]:
    """Run Wav2Vec2 on a list of AudioWindow objects.
    Returns list of (probabilities, speech_ratio) per window."""
    results = []

    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i : i + batch_size]
        waveforms = torch.stack([
            torch.tensor(w.audio, dtype=torch.float32) for w in batch_windows
        ]).to(device)

        logits = model(waveforms)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for j, w in enumerate(batch_windows):
            results.append((probs[j], w.speech_ratio))

    return results


def predict_windows_xgboost(
    windows: list,
    model,
    scaler,
    cfg: dict,
) -> list[tuple[np.ndarray, float]]:
    """Run XGBoost on a list of AudioWindow objects.
    Returns list of (probabilities, speech_ratio) per window."""
    feat_cfg = cfg["features"]
    audio_cfg = cfg["audio"]
    results = []

    for w in windows:
        feats = extract_all_features(
            w.audio, w.sr,
            f0_min=feat_cfg["f0_min"],
            f0_max=feat_cfg["f0_max"],
            n_mfcc=feat_cfg["n_mfcc"],
            energy_threshold=audio_cfg["vad_energy_threshold"],
        )
        feat_values = np.array([feats[k] for k in sorted(feats.keys())], dtype=np.float32)
        feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=0.0, neginf=0.0)
        feat_scaled = scaler.transform(feat_values.reshape(1, -1))
        probs = model.predict_proba(feat_scaled)[0]
        results.append((probs, w.speech_ratio))

    return results


# ============================================================
# Timeline Construction
# ============================================================

def build_timeline(
    window_preds: list[WindowPrediction],
    smooth_window: int = 3,
) -> list[Segment]:
    """
    Smooth window predictions and merge consecutive same-label windows
    into segments. Also fills gaps between windows as silence.
    """
    if not window_preds:
        return []

    # Filter out silence-only windows for smoothing
    speaking_preds = [wp for wp in window_preds if wp.label != "silence"]
    silence_preds = [wp for wp in window_preds if wp.label == "silence"]

    if not speaking_preds:
        # All silence
        return [Segment(
            start_sec=window_preds[0].start_sec,
            end_sec=window_preds[-1].end_sec,
            duration_sec=window_preds[-1].end_sec - window_preds[0].start_sec,
            label="silence",
            confidence=1.0,
        )]

    # Temporal smoothing: median filter on speaking windows
    if len(speaking_preds) >= smooth_window:
        labels_numeric = np.array([
            0 if wp.label == "spontaneous" else 1 for wp in speaking_preds
        ])
        smoothed = median_filter(labels_numeric, size=smooth_window).astype(int)
        for i, wp in enumerate(speaking_preds):
            wp.label = "spontaneous" if smoothed[i] == 0 else "read"

    # Re-merge all predictions (speaking + silence) sorted by time
    all_preds = sorted(speaking_preds + silence_preds, key=lambda wp: wp.start_sec)

    # Merge consecutive same-label windows into segments
    segments = []
    current_label = all_preds[0].label
    seg_start = all_preds[0].start_sec
    seg_confidences = [all_preds[0].confidence]

    for i in range(1, len(all_preds)):
        if all_preds[i].label != current_label:
            # End segment at the START of the new window (avoids overlap)
            seg_end = all_preds[i].start_sec
            segments.append(Segment(
                start_sec=round(seg_start, 2),
                end_sec=round(seg_end, 2),
                duration_sec=round(seg_end - seg_start, 2),
                label=current_label,
                confidence=round(float(np.mean(seg_confidences)), 3),
            ))
            current_label = all_preds[i].label
            seg_start = all_preds[i].start_sec
            seg_confidences = [all_preds[i].confidence]
        else:
            seg_confidences.append(all_preds[i].confidence)

    # Final segment
    seg_end = all_preds[-1].end_sec
    segments.append(Segment(
        start_sec=round(seg_start, 2),
        end_sec=round(seg_end, 2),
        duration_sec=round(seg_end - seg_start, 2),
        label=current_label,
        confidence=round(float(np.mean(seg_confidences)), 3),
    ))

    return segments


# ============================================================
# Main Prediction
# ============================================================

def predict_file(
    audio_path: str,
    cfg: dict,
    wav2vec2_model=None,
    xgb_model=None,
    xgb_scaler=None,
    device=None,
    mode: str = "wav2vec2",
) -> PredictionResult:
    """
    Run full prediction pipeline on a single audio file.

    Args:
        audio_path: Path to .wav file
        cfg: Config dict
        wav2vec2_model: Loaded Wav2Vec2 model (if mode includes wav2vec2)
        xgb_model: Loaded XGBoost model (if mode includes xgboost)
        xgb_scaler: Loaded scaler for XGBoost
        device: torch device
        mode: "wav2vec2" | "xgboost" | "ensemble"

    Returns:
        PredictionResult with all outputs
    """
    audio_cfg = cfg["audio"]
    inf_cfg = cfg["inference"]

    # Load and window audio
    audio, sr = load_audio(
        audio_path,
        target_sr=audio_cfg["sample_rate"],
        max_duration=audio_cfg.get("max_duration_sec", 120),
    )
    total_duration = len(audio) / sr

    windows = window_audio(
        audio, sr,
        window_sec=audio_cfg["window_sec"],
        hop_sec=audio_cfg["hop_sec"],
        min_speech_ratio=0.05,  # Lower threshold — we classify silence separately
        energy_threshold=audio_cfg["vad_energy_threshold"],
    )

    if not windows:
        return PredictionResult(
            filepath=audio_path,
            duration_sec=total_duration,
            overall_label="silence",
            overall_confidence=1.0,
            read_ratio=0.0,
            segments=[],
            window_predictions=[],
            model_used=mode,
        )

    # Get predictions per window
    if mode in ("wav2vec2", "ensemble") and wav2vec2_model is not None:
        w2v_results = predict_windows_wav2vec2(windows, wav2vec2_model, device)
    else:
        w2v_results = None

    if mode in ("xgboost", "ensemble") and xgb_model is not None:
        xgb_results = predict_windows_xgboost(windows, xgb_model, xgb_scaler, cfg)
    else:
        xgb_results = None

    # Build window predictions
    window_preds = []
    min_speech = audio_cfg["min_speech_ratio"]

    for i, win in enumerate(windows):
        # Determine probability
        if mode == "ensemble" and w2v_results and xgb_results:
            # Weighted average: 60% Wav2Vec2, 40% XGBoost
            probs = 0.6 * w2v_results[i][0] + 0.4 * xgb_results[i][0]
        elif w2v_results:
            probs = w2v_results[i][0]
        elif xgb_results:
            probs = xgb_results[i][0]
        else:
            continue

        speech_ratio = win.speech_ratio

        if speech_ratio < min_speech:
            label = "silence"
            confidence = 1.0
        else:
            pred_class = int(np.argmax(probs))
            label = "spontaneous" if pred_class == 0 else "read"
            confidence = float(probs[pred_class])

        window_preds.append(WindowPrediction(
            window_idx=i,
            start_sec=round(win.start_sec, 2),
            end_sec=round(win.end_sec, 2),
            label=label,
            confidence=round(confidence, 3),
            speech_ratio=round(speech_ratio, 3),
        ))

    # Build timeline segments
    segments = build_timeline(
        window_preds,
        smooth_window=inf_cfg["temporal_smooth_window"],
    )

    # Compute overall label
    speaking_windows = [wp for wp in window_preds if wp.label != "silence"]
    if not speaking_windows:
        overall_label = "silence"
        overall_confidence = 1.0
        read_ratio = 0.0
    else:
        read_count = sum(1 for wp in speaking_windows if wp.label == "read")
        read_ratio = read_count / len(speaking_windows)
        overall_label = "read" if read_ratio >= inf_cfg["read_threshold"] else "spontaneous"

        # Confidence = how strongly the windows agree
        if overall_label == "read":
            overall_confidence = float(np.mean([
                wp.confidence for wp in speaking_windows if wp.label == "read"
            ]))
        else:
            overall_confidence = float(np.mean([
                wp.confidence for wp in speaking_windows if wp.label == "spontaneous"
            ]))

    return PredictionResult(
        filepath=audio_path,
        duration_sec=round(total_duration, 2),
        overall_label=overall_label,
        overall_confidence=round(overall_confidence, 3),
        read_ratio=round(read_ratio, 3),
        segments=[asdict(s) for s in segments],
        window_predictions=[asdict(wp) for wp in window_preds],
        model_used=mode,
    )


def format_report(result: PredictionResult) -> str:
    """Format a human-readable report from prediction result."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"File: {Path(result.filepath).name}")
    lines.append(f"Duration: {result.duration_sec}s")
    lines.append(f"{'='*60}")
    lines.append(f"")
    lines.append(f"VERDICT: {result.overall_label.upper()}")
    lines.append(f"Confidence: {result.overall_confidence:.1%}")
    lines.append(f"Read ratio: {result.read_ratio:.1%} of speaking time")
    lines.append(f"Model: {result.model_used}")
    lines.append(f"")
    lines.append(f"--- TIMELINE ---")

    for seg in result.segments:
        marker = "■" if seg["label"] == "read" else ("□" if seg["label"] == "spontaneous" else "·")
        lines.append(
            f"  {marker} [{seg['start_sec']:6.1f}s - {seg['end_sec']:6.1f}s] "
            f"{seg['label']:12s} (conf: {seg['confidence']:.0%}, "
            f"dur: {seg['duration_sec']:.1f}s)"
        )

    lines.append(f"")
    lines.append(f"--- WINDOW DETAILS ({len(result.window_predictions)} windows) ---")
    for wp in result.window_predictions:
        lines.append(
            f"  [{wp['start_sec']:6.1f}s - {wp['end_sec']:6.1f}s] "
            f"{wp['label']:12s} conf={wp['confidence']:.2f} "
            f"speech={wp['speech_ratio']:.0%}"
        )

    return "\n".join(lines)


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Predict read vs spontaneous speech")
    parser.add_argument("--audio", type=str, required=True, help="Path to .wav file or folder")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--mode", type=str, default="wav2vec2",
                        choices=["wav2vec2", "xgboost", "ensemble"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    wav2vec2_model = None
    xgb_model, xgb_scaler = None, None

    if args.mode in ("wav2vec2", "ensemble"):
        wav2vec2_model = load_wav2vec2_model(cfg, device)

    if args.mode in ("xgboost", "ensemble"):
        xgb_model, xgb_scaler = load_xgboost_model(cfg)

    # Collect audio files
    audio_path = Path(args.audio)
    if audio_path.is_file():
        audio_files = [str(audio_path)]
    elif audio_path.is_dir():
        audio_files = sorted([str(f) for f in audio_path.glob("*.wav")])
        print(f"Found {len(audio_files)} .wav files in {audio_path}")
    else:
        print(f"ERROR: {audio_path} not found")
        return

    # Run predictions
    all_results = []
    for fpath in tqdm(audio_files, desc="Predicting"):
        result = predict_file(
            fpath, cfg,
            wav2vec2_model=wav2vec2_model,
            xgb_model=xgb_model,
            xgb_scaler=xgb_scaler,
            device=device,
            mode=args.mode,
        )
        all_results.append(result)

        # Print report
        print(format_report(result))
        print()

    # Save results
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(cfg["paths"]["data_root"]) / "outputs" / "predictions.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(all_results)} files")
        print(f"{'='*60}")
        read_count = sum(1 for r in all_results if r.overall_label == "read")
        spont_count = sum(1 for r in all_results if r.overall_label == "spontaneous")
        print(f"  Read (cheating suspected): {read_count}")
        print(f"  Spontaneous: {spont_count}")
        print(f"  Average confidence: {np.mean([r.overall_confidence for r in all_results]):.1%}")


if __name__ == "__main__":
    main()
