"""
Portable CPU Inference — Biased Single-Neuron Read vs Spontaneous Classifier
===============================================================================
Self-contained script for the company laptop. Uses ONNX Runtime only
(no PyTorch/transformers needed).

The biased model outputs a single logit -> sigmoid -> P(read).
Only flags as "reading" when P(read) exceeds the threshold (default 0.65).
This means: if the model isn't SURE it's reading, it defaults to spontaneous.

Requirements:
    pip install onnxruntime librosa soundfile numpy scipy tqdm

Optional (better silence detection):
    pip install torch torchaudio   # for Silero VAD

Files needed:
    1. This script
    2. biased_wav2vec2_quant.onnx (from checkpoints_biased/ or HuggingFace)

Usage:
    python predict_biased.py --audio interview.wav
    python predict_biased.py --audio recordings/
    python predict_biased.py --audio interview.wav --threshold 0.70
    python predict_biased.py --audio interview.wav --output results.json
    python predict_biased.py --audio interview.wav --verbose
    python predict_biased.py --audio interview.wav --no-silero
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import librosa
import onnxruntime as ort
from scipy.ndimage import median_filter
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG = {
    "sample_rate":            16000,
    "window_sec":             5.0,
    "min_speech_ratio":       0.20,
    "vad_energy_threshold":   0.01,
    "max_duration_sec":       120,
    "temporal_smooth_window": 3,
    "read_threshold":         0.65,   # P(read) must exceed this to flag
    "min_segment_sec":        3.0,
    "vad_merge_gap_sec":      1.0,
}

# Default model location — next to this script or in checkpoints/
DEFAULT_MODEL = "biased_wav2vec2_quant.onnx"

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".webm", ".mp4"}


# ============================================================
# Audio Loading
# ============================================================

def load_audio(path: str, sr: int = 16000, max_duration: float = 120.0) -> np.ndarray:
    """Load audio file, convert to mono, resample, peak-normalize."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(path, sr=sr, mono=True, duration=max_duration)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio


# ============================================================
# Voice Activity Detection
# ============================================================

def load_silero_vad():
    """Load Silero VAD model. Returns (model, get_speech_timestamps_fn)."""
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def get_speech_segments_silero(audio, sr, vad_model, get_ts_fn,
                                min_silence_ms=300, min_speech_ms=250):
    """Run Silero VAD. Returns list of {start, end} dicts in seconds."""
    import torch
    audio_t = torch.from_numpy(audio).float()
    timestamps = get_ts_fn(
        audio_t, vad_model,
        sampling_rate=sr,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        return_seconds=True,
    )
    return [{"start": float(t["start"]), "end": float(t["end"])} for t in timestamps]


def get_speech_segments_rms(audio, sr, energy_threshold=0.01):
    """RMS-based VAD fallback. Returns list of {start, end} dicts in seconds."""
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
    hop = 256

    p30 = np.percentile(rms, 30)
    p90 = np.percentile(rms, 90)
    dynamic_range = p90 - p30
    if dynamic_range < 0.001:
        thresh = max(energy_threshold, 0.002)
    else:
        thresh = max(p30 + 0.2 * dynamic_range, 0.002)

    speech_mask = rms > thresh
    segments = []
    in_speech = False
    seg_start = 0

    for i, is_speech in enumerate(speech_mask):
        if is_speech and not in_speech:
            seg_start = i
            in_speech = True
        elif not is_speech and in_speech:
            segments.append({
                "start": round(seg_start * hop / sr, 3),
                "end":   round(i * hop / sr, 3),
            })
            in_speech = False
    if in_speech:
        segments.append({
            "start": round(seg_start * hop / sr, 3),
            "end":   round(len(audio) / sr, 3),
        })

    return segments


# ============================================================
# Windowing
# ============================================================

def adaptive_hop(window_sec, floor_sec=2.5, ratio=0.4):
    return max(floor_sec, window_sec * ratio)


def make_vad_gated_windows(audio, sr, speech_segments, window_samples,
                            hop_sec, merge_gap_sec=1.0):
    """Window only over VAD-confirmed speech regions."""
    # Merge close segments
    merged = []
    for seg in speech_segments:
        if merged and (seg["start"] - merged[-1]["end"]) < merge_gap_sec:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    win_samp = int(window_samples)
    hop_samp = int(hop_sec * sr)
    windows = []

    for seg in merged:
        seg_start_samp = int(seg["start"] * sr)
        seg_end_samp = int(seg["end"] * sr)
        seg_audio = audio[seg_start_samp:seg_end_samp]

        if len(seg_audio) < win_samp // 2:
            chunk = np.pad(seg_audio, (0, win_samp - len(seg_audio)))
            windows.append((chunk, seg["start"], seg["end"]))
            continue

        pos = 0
        while pos < len(seg_audio):
            chunk = seg_audio[pos : pos + win_samp]
            if len(chunk) < win_samp:
                chunk = np.pad(chunk, (0, win_samp - len(chunk)))
            start_sec = seg["start"] + pos / sr
            end_sec = seg["start"] + (pos + win_samp) / sr
            windows.append((chunk, start_sec, end_sec))
            pos += hop_samp
            if pos + win_samp // 4 >= len(seg_audio):
                break

    return windows


# ============================================================
# ONNX Inference — Single Neuron (Sigmoid)
# ============================================================

class BiasedONNXClassifier:
    """
    ONNX wrapper for the biased single-neuron model.
    Output is a single logit per window -> sigmoid -> P(read).
    """

    def __init__(self, model_path: str):
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

        # Get window size from ONNX model shape
        model_shape = self.session.get_inputs()[0].shape
        self.window_samples = None
        if isinstance(model_shape, (list, tuple)) and len(model_shape) >= 2:
            dim = model_shape[-1]
            if isinstance(dim, int) and dim > 0:
                self.window_samples = dim

        # Fallback to 5sec @ 16kHz
        if self.window_samples is None:
            self.window_samples = 80000

        # Warmup
        dummy = np.zeros((1, self.window_samples), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

    def predict_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """
        Args:
            waveforms: (B, window_samples) float32
        Returns:
            p_read: (B,) — probability of "read" per window
        """
        logits = self.session.run(
            [self.output_name],
            {self.input_name: waveforms.astype(np.float32)},
        )[0]
        # Single neuron -> sigmoid
        return 1.0 / (1.0 + np.exp(-logits.flatten()))


# ============================================================
# Segment Construction
# ============================================================

def merge_segments(window_preds):
    """Merge consecutive windows with the same label into segments."""
    if not window_preds:
        return []

    segments = []
    cur = window_preds[0]
    seg_start = cur["start_sec"]
    seg_confs = [cur["confidence"]]

    for wp in window_preds[1:]:
        if wp["label"] != cur["label"]:
            seg_end = wp["start_sec"]
            segments.append({
                "start_sec":    round(seg_start, 2),
                "end_sec":      round(seg_end, 2),
                "duration_sec": round(seg_end - seg_start, 2),
                "label":        cur["label"],
                "confidence":   round(float(np.mean(seg_confs)), 3),
            })
            seg_start = wp["start_sec"]
            seg_confs = [wp["confidence"]]
            cur = wp
        else:
            seg_confs.append(wp["confidence"])
            cur = wp

    segments.append({
        "start_sec":    round(seg_start, 2),
        "end_sec":      round(cur["end_sec"], 2),
        "duration_sec": round(cur["end_sec"] - seg_start, 2),
        "label":        cur["label"],
        "confidence":   round(float(np.mean(seg_confs)), 3),
    })
    return segments


def enforce_min_segment_length(segments, min_sec=3.0):
    """Merge segments shorter than min_sec into their neighbor."""
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if seg["duration_sec"] < min_sec and seg["label"] not in ("silence",):
                if out:
                    out[-1]["end_sec"] = seg["end_sec"]
                    out[-1]["duration_sec"] = round(
                        out[-1]["end_sec"] - out[-1]["start_sec"], 2)
                    changed = True
                elif i + 1 < len(segments):
                    segments[i + 1]["start_sec"] = seg["start_sec"]
                    segments[i + 1]["duration_sec"] = round(
                        segments[i + 1]["end_sec"] - segments[i + 1]["start_sec"], 2)
                    changed = True
                else:
                    out.append(seg)
            else:
                out.append(seg)
            i += 1
        segments = out
    return segments


def empty_result(path, duration):
    return {
        "filepath":            path,
        "filename":            Path(path).name,
        "duration_sec":        round(duration, 2),
        "overall_label":       "silence",
        "overall_confidence":  1.0,
        "read_ratio":          0.0,
        "read_threshold_used": 0.0,
        "cheating_suspected":  False,
        "segments":            [],
        "window_predictions":  [],
        "processing_time_sec": 0.0,
    }


# ============================================================
# Main Prediction Pipeline
# ============================================================

def predict_file(audio_path, classifier, cfg, vad_model=None,
                  get_ts_fn=None, batch_size=4):
    sr = cfg["sample_rate"]
    window_samples = classifier.window_samples
    window_sec = window_samples / sr
    smooth_window = cfg["temporal_smooth_window"]
    threshold = cfg["read_threshold"]

    t0 = time.perf_counter()

    audio = load_audio(audio_path, sr=sr, max_duration=cfg["max_duration_sec"])
    total_duration = len(audio) / sr

    # VAD
    if vad_model is not None:
        try:
            speech_segments = get_speech_segments_silero(audio, sr, vad_model, get_ts_fn)
        except Exception as e:
            print(f"  [VAD] Silero failed ({e}), using RMS fallback")
            speech_segments = None
    else:
        speech_segments = None

    if not speech_segments:
        speech_segments = get_speech_segments_rms(audio, sr, cfg["vad_energy_threshold"])

    if not speech_segments:
        result = empty_result(audio_path, total_duration)
        result["processing_time_sec"] = round(time.perf_counter() - t0, 2)
        result["read_threshold_used"] = threshold
        return result

    # Windowing
    hop_sec = adaptive_hop(window_sec)
    all_windows = make_vad_gated_windows(
        audio, sr, speech_segments, window_samples, hop_sec,
        merge_gap_sec=cfg["vad_merge_gap_sec"],
    )

    if not all_windows:
        result = empty_result(audio_path, total_duration)
        result["processing_time_sec"] = round(time.perf_counter() - t0, 2)
        result["read_threshold_used"] = threshold
        return result

    chunks = np.stack([w[0] for w in all_windows])
    starts = [w[1] for w in all_windows]
    ends = [w[2] for w in all_windows]

    # Batch inference
    all_probs = []
    for i in range(0, len(chunks), batch_size):
        all_probs.append(classifier.predict_batch(chunks[i : i + batch_size]))
    p_read = np.concatenate(all_probs, axis=0)  # P(read) per window

    # Per-window predictions using threshold
    window_preds = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        pr = float(p_read[i])

        if pr >= threshold:
            label = "read"
            confidence = round(pr, 3)
        else:
            label = "spontaneous"
            confidence = round(1.0 - pr, 3)

        window_preds.append({
            "window_idx": i,
            "start_sec":  round(start, 2),
            "end_sec":    round(end, 2),
            "label":      label,
            "confidence": confidence,
            "p_read":     round(pr, 3),  # raw P(read) for debugging
        })

    # Temporal smoothing
    voting_idx = [i for i, wp in enumerate(window_preds)
                  if wp["label"] in ("spontaneous", "read")]
    if len(voting_idx) >= smooth_window:
        labels_num = np.array([
            0 if window_preds[i]["label"] == "spontaneous" else 1
            for i in voting_idx
        ])
        smoothed = median_filter(labels_num, size=smooth_window).astype(int)
        for j, i in enumerate(voting_idx):
            new_label = "spontaneous" if smoothed[j] == 0 else "read"
            if new_label != window_preds[i]["label"]:
                window_preds[i]["label"] = new_label
                pr = window_preds[i]["p_read"]
                window_preds[i]["confidence"] = round(
                    pr if new_label == "read" else 1.0 - pr, 3
                )

    # Segments
    segments = merge_segments(window_preds)
    segments = enforce_min_segment_length(segments, min_sec=cfg["min_segment_sec"])

    # Overall label
    speaking = [wp for wp in window_preds if wp["label"] in ("spontaneous", "read")]
    if not speaking:
        result = empty_result(audio_path, total_duration)
        result["processing_time_sec"] = round(time.perf_counter() - t0, 2)
        result["read_threshold_used"] = threshold
        return result

    read_count = sum(1 for wp in speaking if wp["label"] == "read")
    read_ratio = read_count / len(speaking)
    # File-level: if majority of windows are read, flag as read
    overall_label = "read" if read_ratio >= 0.5 else "spontaneous"
    same_label = [wp for wp in speaking if wp["label"] == overall_label]
    overall_conf = float(np.mean([wp["confidence"] for wp in same_label]))

    return {
        "filepath":            audio_path,
        "filename":            Path(audio_path).name,
        "duration_sec":        round(total_duration, 2),
        "overall_label":       overall_label,
        "overall_confidence":  round(overall_conf, 3),
        "read_ratio":          round(read_ratio, 3),
        "read_threshold_used": threshold,
        "cheating_suspected":  overall_label == "read",
        "segments":            segments,
        "window_predictions":  window_preds,
        "processing_time_sec": round(time.perf_counter() - t0, 2),
    }


# ============================================================
# Reporting
# ============================================================

def format_report(result, verbose=False):
    lines = []
    lines.append(f"{'='*65}")
    lines.append(f"  File: {result['filename']}")
    lines.append(f"  Duration: {result['duration_sec']}s  |  "
                 f"Processed in: {result['processing_time_sec']}s")
    lines.append(f"  Threshold: {result['read_threshold_used']}")
    lines.append(f"{'='*65}")

    verdict = ("!! READING DETECTED !!" if result["overall_label"] == "read"
               else "OK -- Spontaneous")
    lines.append(f"  VERDICT: {verdict}")
    lines.append(f"  Confidence: {result['overall_confidence']:.1%}")
    lines.append(f"  Read ratio: {result['read_ratio']:.1%} of speaking time")
    lines.append("")
    lines.append("  --- TIMELINE ---")

    for seg in result["segments"]:
        marker = "##" if seg["label"] == "read" else ".."
        lines.append(
            f"    {marker} [{seg['start_sec']:6.1f}s - {seg['end_sec']:6.1f}s] "
            f"{seg['label']:12s} conf={seg['confidence']:.0%}  "
            f"({seg['duration_sec']:.1f}s)"
        )

    if verbose:
        lines.append("")
        lines.append(f"  --- WINDOWS ({len(result['window_predictions'])}) ---")
        for wp in result["window_predictions"]:
            lines.append(
                f"    [{wp['start_sec']:6.1f}s-{wp['end_sec']:6.1f}s] "
                f"{wp['label']:12s} conf={wp['confidence']:.2f}  "
                f"P(read)={wp['p_read']:.3f}"
            )

    lines.append("")
    return "\n".join(lines)


def format_summary_table(results):
    lines = []
    lines.append(f"\n{'='*85}")
    lines.append(f"  BATCH SUMMARY -- {len(results)} files  |  threshold={results[0]['read_threshold_used']}")
    lines.append(f"{'='*85}")
    lines.append(f"  {'Filename':<40s} {'Verdict':<14s} {'Conf':>6s} {'Read%':>6s} {'Time':>6s}")
    lines.append(f"  {'-'*40} {'-'*14} {'-'*6} {'-'*6} {'-'*6}")

    for r in results:
        flag = "** READ **" if r["overall_label"] == "read" else "spontaneous"
        lines.append(
            f"  {r['filename']:<40s} {flag:<14s} "
            f"{r['overall_confidence']:5.0%} {r['read_ratio']:5.0%} "
            f"{r['processing_time_sec']:5.1f}s"
        )

    read_n = sum(1 for r in results if r["overall_label"] == "read")
    spont_n = sum(1 for r in results if r["overall_label"] == "spontaneous")
    total_t = sum(r["processing_time_sec"] for r in results)

    lines.append(f"  {'-'*78}")
    lines.append(f"  Read (cheating suspected): {read_n}")
    lines.append(f"  Spontaneous (OK):          {spont_n}")
    lines.append(f"  Total processing time:     {total_t:.1f}s")
    lines.append(f"{'='*85}")
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def find_audio_files(path):
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        files = []
        for ext in AUDIO_EXTS:
            files.extend(p.rglob(f"*{ext}"))
            files.extend(p.rglob(f"*{ext.upper()}"))
        return sorted(set(str(f) for f in files))
    return []


def resolve_model_path(model_arg):
    """Find the ONNX model file."""
    candidate = Path(model_arg)
    if candidate.exists():
        return str(candidate)

    # Check next to this script
    script_dir = Path(__file__).parent
    for loc in [
        script_dir / model_arg,
        script_dir / "checkpoints" / model_arg,
        script_dir / "checkpoints_biased" / model_arg,
    ]:
        if loc.exists():
            return str(loc)

    return model_arg


def main():
    parser = argparse.ArgumentParser(
        description="Biased CPU Inference -- Read vs Spontaneous (single-neuron model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The biased model uses a SINGLE sigmoid neuron. It only flags "reading"
when P(read) exceeds the threshold. Below the threshold = spontaneous.

This means: fewer false positives for reading detection.
Tune --threshold higher (e.g. 0.75) for even stricter detection.

Examples:
  python predict_biased.py --audio interview.wav
  python predict_biased.py --audio interview.wav --threshold 0.70
  python predict_biased.py --audio recordings/ --output results.json
  python predict_biased.py --audio interview.wav --verbose
        """,
    )
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file or folder")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Path to ONNX model (default: {DEFAULT_MODEL})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["read_threshold"],
                        help=f"P(read) threshold to flag as reading (default: {DEFAULT_CONFIG['read_threshold']})")
    parser.add_argument("--output", type=str, default="predictions_biased.json",
                        help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-window details")
    parser.add_argument("--no-silero", action="store_true",
                        help="Use RMS VAD instead of Silero")
    args = parser.parse_args()

    # Find audio files
    audio_files = find_audio_files(args.audio)
    if not audio_files:
        print(f"ERROR: No audio files found at '{args.audio}'")
        sys.exit(1)
    print(f"Found {len(audio_files)} audio file(s)")

    # Resolve model
    model_path = resolve_model_path(args.model)
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: '{args.model}'")
        print(f"Download from HuggingFace: Pransfrance/speechproj-models")
        print(f"  File: biased/biased_wav2vec2_quant.onnx")
        sys.exit(1)

    model_mb = Path(model_path).stat().st_size / 1e6

    # Build config with user's threshold
    cfg = DEFAULT_CONFIG.copy()
    cfg["read_threshold"] = args.threshold

    # Load classifier
    classifier = BiasedONNXClassifier(model_path)
    window_sec = classifier.window_samples / cfg["sample_rate"]
    hop_sec = adaptive_hop(window_sec)

    print(f"Model:     {Path(model_path).name} ({model_mb:.0f} MB)")
    print(f"Window:    {window_sec:.1f}s | Hop: {hop_sec:.1f}s")
    print(f"Threshold: {args.threshold} (P(read) must exceed this to flag)")
    print(f"Output:    single neuron -> sigmoid -> P(read)")

    # Load Silero VAD
    vad_model, get_ts_fn = None, None
    if not args.no_silero:
        try:
            vad_model, get_ts_fn = load_silero_vad()
            print("Silero VAD loaded")
        except Exception:
            print("Silero VAD unavailable -- using RMS VAD")
    else:
        print("Using RMS VAD (--no-silero)")

    print()

    # Run inference
    results = []
    for fpath in tqdm(audio_files, desc="Processing", disable=len(audio_files) == 1):
        try:
            result = predict_file(
                fpath, classifier, cfg,
                vad_model=vad_model,
                get_ts_fn=get_ts_fn,
                batch_size=args.batch_size,
            )
            results.append(result)
            print(format_report(result, verbose=args.verbose))
        except Exception as e:
            print(f"ERROR processing {fpath}: {e}")

    if len(results) > 1:
        print(format_summary_table(results))

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
