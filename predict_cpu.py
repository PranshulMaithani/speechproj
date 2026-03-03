"""
Portable CPU Inference — Read vs Spontaneous Speech Classifier
=================================================================
This is a SELF-CONTAINED script for running inference on a CPU-only
"potato laptop". It uses ONNX Runtime (no PyTorch/transformers needed).

Requirements (install on the laptop):
    pip install onnxruntime librosa soundfile numpy scipy tqdm

Files to copy to the laptop:
    1. This script (predict_cpu.py)
    2. checkpoints/speech_classifier_quant.onnx  (95 MB)
    3. configs/config.yaml

Usage:
    # Single file (wav, mp3, m4a, flac — anything librosa can read)
    python predict_cpu.py --audio interview.wav

    # Folder of files
    python predict_cpu.py --audio company_recordings/

    # With custom output path
    python predict_cpu.py --audio company_recordings/ --output results.json

    # Verbose mode (print per-window details)
    python predict_cpu.py --audio interview.wav --verbose

Output:
    - Console report per file with verdict + timeline
    - JSON file with full results (default: outputs/cpu_predictions.json)
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field

import numpy as np
import librosa
import onnxruntime as ort
from scipy.ndimage import median_filter
from tqdm import tqdm


# ============================================================
# Configuration (hardcoded defaults — no YAML dependency needed)
# ============================================================

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "window_sec": 5.0,
    "hop_sec": 2.5,
    "min_speech_ratio": 0.20,
    "vad_energy_threshold": 0.01,
    "max_duration_sec": 120,
    "temporal_smooth_window": 3,
    "read_threshold": 0.50,
}


# ============================================================
# Data Structures
# ============================================================

@dataclass
class WindowPrediction:
    window_idx: int
    start_sec: float
    end_sec: float
    label: str
    confidence: float
    speech_ratio: float


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    duration_sec: float
    label: str
    confidence: float


@dataclass
class PredictionResult:
    filepath: str
    filename: str
    duration_sec: float
    overall_label: str
    overall_confidence: float
    read_ratio: float
    segments: list
    window_predictions: list
    processing_time_sec: float


# ============================================================
# Audio Loading (supports wav, mp3, m4a, flac, ogg, etc.)
# ============================================================

def load_audio(path: str, sr: int = 16000, max_duration: float = 120.0) -> tuple:
    """Load audio file, convert to mono, resample."""
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=max_duration)
    return audio, sr


# ============================================================
# Voice Activity Detection
# ============================================================

def simple_vad(audio: np.ndarray, sr: int, energy_threshold: float = 0.01) -> np.ndarray:
    """Energy-based VAD. Returns boolean frame-level mask."""
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]

    # Adaptive threshold: use median of non-silent frames as anchor
    # This handles very quiet recordings (common in phone/laptop mic)
    p30 = np.percentile(rms, 30)
    p70 = np.percentile(rms, 70)
    adaptive_thresh = p30 + 0.1 * (p70 - p30)  # Low bar — biased toward detecting speech
    dynamic_thresh = max(min(energy_threshold, adaptive_thresh), np.percentile(rms, 10))
    speech_mask = rms > dynamic_thresh

    # Smooth short gaps
    min_frames = int(0.1 * sr / 256)
    smoothed = speech_mask.copy()
    in_speech = False
    gap_start = 0
    for i in range(len(smoothed)):
        if smoothed[i]:
            if in_speech and (i - gap_start) < min_frames:
                smoothed[gap_start:i] = True
            in_speech = True
        else:
            if in_speech:
                gap_start = i
                in_speech = False
    return smoothed


def compute_speech_ratio(audio: np.ndarray, sr: int, threshold: float = 0.01) -> float:
    """Fraction of audio that contains speech."""
    mask = simple_vad(audio, sr, threshold)
    return float(mask.sum()) / max(len(mask), 1)


# ============================================================
# Windowing
# ============================================================

def window_audio(audio: np.ndarray, sr: int, cfg: dict) -> list:
    """Segment audio into overlapping windows. Returns list of (chunk, start_sec, end_sec, speech_ratio)."""
    window_samples = int(cfg["window_sec"] * sr)
    hop_samples = int(cfg["hop_sec"] * sr)
    total = len(audio)
    threshold = cfg["vad_energy_threshold"]

    windows = []
    start = 0
    while start < total:
        end = min(start + window_samples, total)
        chunk = audio[start:end]

        if len(chunk) < window_samples // 2:
            break
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))

        speech_ratio = compute_speech_ratio(chunk, sr, threshold)
        windows.append((chunk, start / sr, end / sr, speech_ratio))
        start += hop_samples

    return windows


# ============================================================
# ONNX Inference
# ============================================================

class ONNXClassifier:
    """Lightweight ONNX Runtime wrapper for the speech classifier."""

    def __init__(self, model_path: str):
        # Use CPU only — works on any machine
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.inter_op_num_threads = 2

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Warmup
        dummy = np.zeros((1, 80000), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

    def predict_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of waveforms.

        Args:
            waveforms: (batch, samples) float32 array

        Returns:
            probabilities: (batch, 2) — [p_spontaneous, p_read]
        """
        logits = self.session.run(
            [self.output_name],
            {self.input_name: waveforms.astype(np.float32)},
        )[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return probs


# ============================================================
# Timeline Construction
# ============================================================

def build_timeline(window_preds: list, smooth_window: int = 3) -> list:
    """Smooth and merge consecutive same-label windows into segments."""
    if not window_preds:
        return []

    speaking = [wp for wp in window_preds if wp.label != "silence"]
    silence = [wp for wp in window_preds if wp.label == "silence"]

    if not speaking:
        return [Segment(
            start_sec=window_preds[0].start_sec,
            end_sec=window_preds[-1].end_sec,
            duration_sec=window_preds[-1].end_sec - window_preds[0].start_sec,
            label="silence", confidence=1.0,
        )]

    # Temporal smoothing via median filter
    if len(speaking) >= smooth_window:
        labels_num = np.array([0 if wp.label == "spontaneous" else 1 for wp in speaking])
        smoothed = median_filter(labels_num, size=smooth_window).astype(int)
        for i, wp in enumerate(speaking):
            wp.label = "spontaneous" if smoothed[i] == 0 else "read"

    # Merge
    all_preds = sorted(speaking + silence, key=lambda wp: wp.start_sec)
    segments = []
    cur_label = all_preds[0].label
    seg_start = all_preds[0].start_sec
    seg_confs = [all_preds[0].confidence]

    for i in range(1, len(all_preds)):
        if all_preds[i].label != cur_label:
            seg_end = all_preds[i].start_sec
            segments.append(Segment(
                start_sec=round(seg_start, 2),
                end_sec=round(seg_end, 2),
                duration_sec=round(seg_end - seg_start, 2),
                label=cur_label,
                confidence=round(float(np.mean(seg_confs)), 3),
            ))
            cur_label = all_preds[i].label
            seg_start = all_preds[i].start_sec
            seg_confs = [all_preds[i].confidence]
        else:
            seg_confs.append(all_preds[i].confidence)

    seg_end = all_preds[-1].end_sec
    segments.append(Segment(
        start_sec=round(seg_start, 2),
        end_sec=round(seg_end, 2),
        duration_sec=round(seg_end - seg_start, 2),
        label=cur_label,
        confidence=round(float(np.mean(seg_confs)), 3),
    ))

    return segments


# ============================================================
# Main Prediction Pipeline
# ============================================================

def predict_file(audio_path: str, classifier: ONNXClassifier, cfg: dict, batch_size: int = 8) -> PredictionResult:
    """Full prediction for one audio file."""
    t0 = time.perf_counter()

    # Load & window
    audio, sr = load_audio(audio_path, sr=cfg["sample_rate"], max_duration=cfg["max_duration_sec"])
    total_duration = len(audio) / sr
    windows = window_audio(audio, sr, cfg)

    if not windows:
        return PredictionResult(
            filepath=audio_path, filename=Path(audio_path).name,
            duration_sec=round(total_duration, 2), overall_label="silence",
            overall_confidence=1.0, read_ratio=0.0, segments=[], window_predictions=[],
            processing_time_sec=round(time.perf_counter() - t0, 2),
        )

    # Batch inference
    all_probs = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        waveforms = np.stack([w[0] for w in batch])
        probs = classifier.predict_batch(waveforms)
        all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)

    # Build window predictions
    min_speech = cfg["min_speech_ratio"]
    window_preds = []

    for i, (chunk, start_sec, end_sec, speech_ratio) in enumerate(windows):
        if speech_ratio < min_speech:
            label, confidence = "silence", 1.0
        else:
            pred_class = int(np.argmax(all_probs[i]))
            label = "spontaneous" if pred_class == 0 else "read"
            confidence = float(all_probs[i, pred_class])

        window_preds.append(WindowPrediction(
            window_idx=i,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            label=label,
            confidence=round(confidence, 3),
            speech_ratio=round(speech_ratio, 3),
        ))

    # Timeline
    segments = build_timeline(window_preds, smooth_window=cfg["temporal_smooth_window"])

    # Overall verdict
    speaking_wins = [wp for wp in window_preds if wp.label != "silence"]
    if not speaking_wins:
        overall_label, overall_confidence, read_ratio = "silence", 1.0, 0.0
    else:
        read_count = sum(1 for wp in speaking_wins if wp.label == "read")
        read_ratio = read_count / len(speaking_wins)
        overall_label = "read" if read_ratio >= cfg["read_threshold"] else "spontaneous"

        matching = [wp.confidence for wp in speaking_wins if wp.label == overall_label]
        overall_confidence = float(np.mean(matching)) if matching else 0.5

    elapsed = time.perf_counter() - t0

    return PredictionResult(
        filepath=audio_path,
        filename=Path(audio_path).name,
        duration_sec=round(total_duration, 2),
        overall_label=overall_label,
        overall_confidence=round(overall_confidence, 3),
        read_ratio=round(read_ratio, 3),
        segments=[asdict(s) for s in segments],
        window_predictions=[asdict(wp) for wp in window_preds],
        processing_time_sec=round(elapsed, 2),
    )


# ============================================================
# Reporting
# ============================================================

def format_report(result: PredictionResult, verbose: bool = False) -> str:
    """Human-readable report."""
    lines = []
    lines.append(f"{'='*65}")
    lines.append(f"  File: {result.filename}")
    lines.append(f"  Duration: {result.duration_sec}s | Processed in: {result.processing_time_sec}s")
    lines.append(f"{'='*65}")

    verdict_marker = "!! READING DETECTED !!" if result.overall_label == "read" else "OK — Spontaneous"
    lines.append(f"  VERDICT: {verdict_marker}")
    lines.append(f"  Confidence: {result.overall_confidence:.1%}")
    lines.append(f"  Read ratio: {result.read_ratio:.1%} of speaking time")
    lines.append(f"")
    lines.append(f"  --- TIMELINE ---")

    for seg in result.segments:
        if seg["label"] == "read":
            marker, color = "██", ""
        elif seg["label"] == "spontaneous":
            marker, color = "░░", ""
        else:
            marker, color = "··", ""

        lines.append(
            f"    {marker} [{seg['start_sec']:6.1f}s - {seg['end_sec']:6.1f}s] "
            f"{seg['label']:12s} conf={seg['confidence']:.0%}  "
            f"({seg['duration_sec']:.1f}s)"
        )

    if verbose:
        lines.append(f"")
        lines.append(f"  --- WINDOWS ({len(result.window_predictions)}) ---")
        for wp in result.window_predictions:
            lines.append(
                f"    [{wp['start_sec']:6.1f}s-{wp['end_sec']:6.1f}s] "
                f"{wp['label']:12s} conf={wp['confidence']:.2f} "
                f"speech={wp['speech_ratio']:.0%}"
            )

    lines.append("")
    return "\n".join(lines)


def format_summary_table(results: list) -> str:
    """Summary table for batch results."""
    lines = []
    lines.append(f"\n{'='*85}")
    lines.append(f"  BATCH SUMMARY — {len(results)} files")
    lines.append(f"{'='*85}")
    lines.append(f"  {'Filename':<40s} {'Verdict':<14s} {'Conf':>6s} {'Read%':>6s} {'Time':>6s}")
    lines.append(f"  {'-'*40} {'-'*14} {'-'*6} {'-'*6} {'-'*6}")

    for r in results:
        flag = "** READ **" if r.overall_label == "read" else "spontaneous"
        lines.append(
            f"  {r.filename:<40s} {flag:<14s} "
            f"{r.overall_confidence:5.0%} {r.read_ratio:5.0%} "
            f"{r.processing_time_sec:5.1f}s"
        )

    read_count = sum(1 for r in results if r.overall_label == "read")
    spont_count = sum(1 for r in results if r.overall_label == "spontaneous")
    total_time = sum(r.processing_time_sec for r in results)
    avg_conf = np.mean([r.overall_confidence for r in results])

    lines.append(f"  {'-'*78}")
    lines.append(f"  Read (cheating suspected): {read_count}")
    lines.append(f"  Spontaneous (OK):          {spont_count}")
    lines.append(f"  Avg confidence:            {avg_conf:.1%}")
    lines.append(f"  Total processing time:     {total_time:.1f}s")
    lines.append(f"{'='*85}")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def find_audio_files(path: Path) -> list:
    """Find all supported audio files in a path."""
    AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".webm"}
    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        files = []
        for ext in AUDIO_EXTS:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(set(str(f) for f in files))
    return []


def main():
    parser = argparse.ArgumentParser(
        description="CPU Inference — Read vs Spontaneous Speech Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_cpu.py --audio interview.wav
  python predict_cpu.py --audio recordings/ --output results.json
  python predict_cpu.py --audio recordings/ --verbose
        """,
    )
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file or folder of audio files")
    parser.add_argument("--model", type=str, default="checkpoints/speech_classifier_quant.onnx",
                        help="Path to ONNX model (default: checkpoints/speech_classifier_quant.onnx)")
    parser.add_argument("--output", type=str, default="outputs/cpu_predictions.json",
                        help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Inference batch size (lower = less RAM)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-window details")
    parser.add_argument("--read-threshold", type=float, default=0.50,
                        help="Fraction of speaking windows to flag as read (default: 0.50)")
    args = parser.parse_args()

    # Find audio files
    audio_path = Path(args.audio)
    audio_files = find_audio_files(audio_path)

    if not audio_files:
        print(f"ERROR: No audio files found at '{args.audio}'")
        print(f"Supported formats: wav, mp3, m4a, flac, ogg, wma, aac, webm")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s)")

    # Load model
    model_path = args.model
    if not Path(model_path).exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        model_path = str(script_dir / args.model)
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at '{args.model}'")
        print(f"Copy speech_classifier_quant.onnx to checkpoints/")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model_size_mb = Path(model_path).stat().st_size / 1e6
    print(f"Model size: {model_size_mb:.0f} MB")
    classifier = ONNXClassifier(model_path)
    print(f"Model loaded, ready for inference\n")

    # Config
    cfg = DEFAULT_CONFIG.copy()
    cfg["read_threshold"] = args.read_threshold

    # Run
    results = []
    for fpath in tqdm(audio_files, desc="Processing", disable=len(audio_files) == 1):
        try:
            result = predict_file(fpath, classifier, cfg, batch_size=args.batch_size)
            results.append(result)
            print(format_report(result, verbose=args.verbose))
        except Exception as e:
            print(f"ERROR processing {fpath}: {e}")

    # Summary for batch
    if len(results) > 1:
        print(format_summary_table(results))

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
