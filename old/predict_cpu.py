"""
Portable CPU Inference — Read vs Spontaneous Speech Classifier
=================================================================
This is a SELF-CONTAINED script for running inference on a CPU-only
"potato laptop". It uses ONNX Runtime (no PyTorch/transformers needed).

Requirements (install on the laptop):
    pip install onnxruntime librosa soundfile numpy scipy tqdm

Optional (strongly recommended — much better silence detection):
    pip install torch torchaudio   # for Silero VAD
    # Silero model (~2MB) is downloaded automatically on first run

Files to copy to the laptop:
    1. This script (predict_cpu.py)
    2. One or more ONNX model files, for example:
       - checkpoints/speech_classifier_quant.onnx                   (5sec wav2vec2)
       - checkpoints/speech_classifier_wav2vec2_7_5sec_quant.onnx
       - checkpoints/speech_classifier_wav2vec2_10sec_quant.onnx
       - checkpoints/speech_classifier_wav2vec2_12_5sec_quant.onnx
       - checkpoints/speech_classifier_wav2vec2_15sec_quant.onnx
       - checkpoints/speech_classifier_wavlm_5sec_quant.onnx

Usage:
    python predict_cpu.py --audio interview.wav
    python predict_cpu.py --audio company_recordings/
    python predict_cpu.py --audio company_recordings/ --output results.json
    python predict_cpu.py --audio interview.wav --verbose
    python predict_cpu.py --audio interview.wav --model 10sec
    python predict_cpu.py --audio interview.wav --model wavlm
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
    "read_threshold":         0.45,   # tuned on company data
    "min_conf":               0.0,    # disabled — was hurting recall
    "min_segment_sec":        3.0,
    "vad_merge_gap_sec":      1.0,
}

MODEL_ALIAS_TO_FILE = {
    # 5sec wav2vec2 (production)
    "5sec":             "speech_classifier_quant.onnx",
    "wav2vec2":         "speech_classifier_quant.onnx",
    "wav2vec2_5sec":    "speech_classifier_quant.onnx",
    # 7.5sec
    "7sec":             "speech_classifier_wav2vec2_7_5sec_quant.onnx",
    "7.5":              "speech_classifier_wav2vec2_7_5sec_quant.onnx",
    "7.5sec":           "speech_classifier_wav2vec2_7_5sec_quant.onnx",
    "7_5sec":           "speech_classifier_wav2vec2_7_5sec_quant.onnx",
    "wav2vec2_7sec":    "speech_classifier_wav2vec2_7_5sec_quant.onnx",
    "wav2vec2_7_5sec":  "speech_classifier_wav2vec2_7_5sec_quant.onnx",
    # 10sec
    "10":               "speech_classifier_wav2vec2_10sec_quant.onnx",
    "10sec":            "speech_classifier_wav2vec2_10sec_quant.onnx",
    "wav2vec2_10sec":   "speech_classifier_wav2vec2_10sec_quant.onnx",
    # 12.5sec
    "12.5":             "speech_classifier_wav2vec2_12_5sec_quant.onnx",
    "12.5sec":          "speech_classifier_wav2vec2_12_5sec_quant.onnx",
    "12_5sec":          "speech_classifier_wav2vec2_12_5sec_quant.onnx",
    "wav2vec2_12_5sec": "speech_classifier_wav2vec2_12_5sec_quant.onnx",
    # 15sec
    "15":               "speech_classifier_wav2vec2_15sec_quant.onnx",
    "15sec":            "speech_classifier_wav2vec2_15sec_quant.onnx",
    "wav2vec2_15sec":   "speech_classifier_wav2vec2_15sec_quant.onnx",
    # WavLM
    "wavlm":            "speech_classifier_wavlm_5sec_quant.onnx",
    "wavlm5sec":        "speech_classifier_wavlm_5sec_quant.onnx",
    "wavlm_5sec":       "speech_classifier_wavlm_5sec_quant.onnx",
    "whisper":        "speech_classifier_whisper_medium_quant.onnx",
    "whisper_medium": "speech_classifier_whisper_medium_quant.onnx",
}

# Exact filename -> window size mapping (checked before pattern matching)
WINDOW_SEC_MAP = {
    "speech_classifier_quant.onnx":                   5.0,
    "speech_classifier_wav2vec2_7_5sec_quant.onnx":   7.5,
    "speech_classifier_wav2vec2_10sec_quant.onnx":   10.0,
    "speech_classifier_wav2vec2_12_5sec_quant.onnx": 12.5,
    "speech_classifier_wav2vec2_15sec_quant.onnx":   15.0,
    "speech_classifier_wavlm_5sec_quant.onnx":        5.0,
}


# ============================================================
# Audio Loading
# ============================================================

def load_audio(path: str, sr: int = 16000, max_duration: float = 120.0) -> np.ndarray:
    """Load audio file, convert to mono, resample. Returns waveform array."""
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


def get_speech_segments_silero(
    audio: np.ndarray,
    sr: int,
    vad_model,
    get_ts_fn,
    min_silence_ms: int = 300,
    min_speech_ms: int = 250,
) -> list[dict]:
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


def get_speech_segments_rms(
    audio: np.ndarray,
    sr: int,
    energy_threshold: float = 0.01,
) -> list[dict]:
    """
    RMS-based VAD using a global threshold from the full recording.
    Returns list of {start, end} dicts in seconds.
    """
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

def adaptive_hop(window_sec: float, floor_sec: float = 2.5, ratio: float = 0.4) -> float:
    """Compute hop size scaling with window, floored at floor_sec.

    window_sec ->  hop_sec
         5.0   ->  2.5
         7.5   ->  3.0
        10.0   ->  4.0
        12.5   ->  5.0
        15.0   ->  6.0
    """
    return max(floor_sec, window_sec * ratio)


def make_vad_gated_windows(
    audio: np.ndarray,
    sr: int,
    speech_segments: list[dict],
    window_samples: int,
    hop_sec: float,
    merge_gap_sec: float = 1.0,
) -> list[tuple]:
    """
    Return (chunk, start_sec, end_sec) tuples windowed ONLY over
    VAD-confirmed speech regions. Silence is never passed to the model.
    """
    merged = []
    for seg in speech_segments:
        if merged and (seg["start"] - merged[-1]["end"]) < merge_gap_sec:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    win_samp = int(window_samples)
    hop_samp = int(hop_sec * sr)
    windows  = []

    for seg in merged:
        seg_start_samp = int(seg["start"] * sr)
        seg_end_samp   = int(seg["end"]   * sr)
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
            end_sec   = seg["start"] + (pos + win_samp) / sr
            windows.append((chunk, start_sec, end_sec))
            pos += hop_samp
            if pos + win_samp // 4 >= len(seg_audio):
                break

    return windows


# ============================================================
# ONNX Inference
# ============================================================

class ONNXClassifier:
    """Lightweight ONNX Runtime wrapper for the speech classifier."""

    def __init__(self, model_path: str, window_samples: int):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.inter_op_num_threads = 2

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name    = self.session.get_inputs()[0].name
        self.output_name   = self.session.get_outputs()[0].name

        # Prefer the static ONNX input length if available.
        # This avoids mismatches like configured 120000 vs model-required 80000.
        model_shape = self.session.get_inputs()[0].shape
        model_samples = None
        if isinstance(model_shape, (list, tuple)) and len(model_shape) >= 2:
            dim = model_shape[-1]
            if isinstance(dim, int) and dim > 0:
                model_samples = dim

        self.window_samples = int(model_samples or window_samples)

        # Warmup with correct window size
        dummy = np.zeros((1, self.window_samples), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

    def predict_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """
        Args:
            waveforms: (B, window_samples) float32

        Returns:
            probs: (B, 2) — [p_spontaneous, p_read]
        """
        logits = self.session.run(
            [self.output_name],
            {self.input_name: waveforms.astype(np.float32)},
        )[0]
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_l / exp_l.sum(axis=-1, keepdims=True)
    
class WhisperONNXClassifier:
    """ONNX classifier for Whisper-based model — takes log-mel spectrograms."""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 4

        self.session     = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Warmup
        dummy = np.zeros((1, 80, 3000), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})
        print(f"Whisper ONNX loaded: {Path(model_path).name}")

    def _extract_features(self, waveforms: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Convert raw waveforms to Whisper log-mel spectrograms."""
        import librosa
        features = []
        for wav in waveforms:
            mel = librosa.feature.melspectrogram(
                y=wav.astype(np.float32), sr=sr,
                n_mels=80, n_fft=400, hop_length=160, win_length=400,
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            # Whisper expects 3000 frames — pad or trim
            if log_mel.shape[1] < 3000:
                log_mel = np.pad(log_mel, ((0,0),(0, 3000 - log_mel.shape[1])))
            else:
                log_mel = log_mel[:, :3000]
            features.append(log_mel)
        return np.stack(features).astype(np.float32)   # (B, 80, 3000)

    def predict_batch(self, waveforms: np.ndarray) -> np.ndarray:
        features = self._extract_features(waveforms)
        logits   = self.session.run(
            [self.output_name],
            {self.input_name: features},
        )[0]
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_l / exp_l.sum(axis=-1, keepdims=True)


# ============================================================
# Segment Construction
# ============================================================

def _merge_segments(window_preds: list[dict]) -> list[dict]:
    """Merge consecutive windows with the same label into segments."""
    if not window_preds:
        return []

    segments  = []
    cur       = window_preds[0]
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


def enforce_min_segment_length(segments: list[dict], min_sec: float = 3.0) -> list[dict]:
    """Merge segments shorter than min_sec into their neighbor."""
    changed = True
    while changed:
        changed = False
        out = []
        i   = 0
        while i < len(segments):
            seg = segments[i]
            if seg["duration_sec"] < min_sec and seg["label"] not in ("silence", "uncertain"):
                if out:
                    out[-1]["end_sec"]      = seg["end_sec"]
                    out[-1]["duration_sec"] = round(
                        out[-1]["end_sec"] - out[-1]["start_sec"], 2)
                    changed = True
                elif i + 1 < len(segments):
                    segments[i + 1]["start_sec"]    = seg["start_sec"]
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


def _empty_result(path: str, duration: float) -> dict:
    return {
        "filepath":            path,
        "filename":            Path(path).name,
        "duration_sec":        round(duration, 2),
        "overall_label":       "silence",
        "overall_confidence":  1.0,
        "read_ratio":          0.0,
        "cheating_suspected":  False,
        "segments":            [],
        "window_predictions":  [],
        "processing_time_sec": 0.0,
    }


# ============================================================
# Main Prediction Pipeline
# ============================================================

def predict_file(
    audio_path: str,
    classifier: ONNXClassifier,
    cfg: dict,
    vad_model=None,
    get_ts_fn=None,
    batch_size: int = 4,
) -> dict:
    sr             = cfg["sample_rate"]
    window_samples = int(cfg.get("window_samples", cfg["window_sec"] * sr))
    window_sec     = window_samples / sr
    min_conf       = cfg["min_conf"]
    smooth_window  = cfg["temporal_smooth_window"]
    read_threshold = cfg["read_threshold"]

    t0 = time.perf_counter()

    audio = load_audio(audio_path, sr=sr, max_duration=cfg["max_duration_sec"])
    total_duration = len(audio) / sr

    # --- VAD ---
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
        result = _empty_result(audio_path, total_duration)
        result["processing_time_sec"] = round(time.perf_counter() - t0, 2)
        return result

    # --- Windowing ---
    hop_sec = adaptive_hop(window_sec)
    all_windows = make_vad_gated_windows(
        audio, sr, speech_segments, window_samples, hop_sec,
        merge_gap_sec=cfg["vad_merge_gap_sec"],
    )

    if not all_windows:
        result = _empty_result(audio_path, total_duration)
        result["processing_time_sec"] = round(time.perf_counter() - t0, 2)
        return result

    chunks = np.stack([w[0] for w in all_windows])
    starts = [w[1] for w in all_windows]
    ends   = [w[2] for w in all_windows]

    # --- Inference ---
    all_probs = []
    for i in range(0, len(chunks), batch_size):
        all_probs.append(classifier.predict_batch(chunks[i : i + batch_size]))
    probs = np.concatenate(all_probs, axis=0)

    # --- Per-window predictions ---
    window_preds = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        pred_cls = int(np.argmax(probs[i]))
        conf     = float(probs[i][pred_cls])
        if min_conf > 0 and conf < min_conf:
            label = "uncertain"
        else:
            label = "spontaneous" if pred_cls == 0 else "read"
        window_preds.append({
            "start_sec":  round(start, 2),
            "end_sec":    round(end, 2),
            "label":      label,
            "confidence": round(conf, 3),
        })

    # --- Temporal smoothing ---
    voting_idx = [
        i for i, wp in enumerate(window_preds)
        if wp["label"] in ("spontaneous", "read")
    ]
    if len(voting_idx) >= smooth_window:
        labels_num = np.array([
            0 if window_preds[i]["label"] == "spontaneous" else 1
            for i in voting_idx
        ])
        smoothed = median_filter(labels_num, size=smooth_window).astype(int)
        for j, i in enumerate(voting_idx):
            window_preds[i]["label"] = "spontaneous" if smoothed[j] == 0 else "read"

    # --- Segments ---
    segments = _merge_segments(window_preds)
    segments = enforce_min_segment_length(segments, min_sec=cfg["min_segment_sec"])

    # --- Overall label ---
    speaking = [wp for wp in window_preds if wp["label"] in ("spontaneous", "read")]
    if not speaking:
        result = _empty_result(audio_path, total_duration)
        result["processing_time_sec"] = round(time.perf_counter() - t0, 2)
        return result

    read_count    = sum(1 for wp in speaking if wp["label"] == "read")
    read_ratio    = read_count / len(speaking)
    overall_label = "read" if read_ratio >= read_threshold else "spontaneous"
    same_label    = [wp for wp in speaking if wp["label"] == overall_label]
    overall_conf  = float(np.mean([wp["confidence"] for wp in same_label]))

    return {
        "filepath":            audio_path,
        "filename":            Path(audio_path).name,
        "duration_sec":        round(total_duration, 2),
        "overall_label":       overall_label,
        "overall_confidence":  round(overall_conf, 3),
        "read_ratio":          round(read_ratio, 3),
        "cheating_suspected":  overall_label == "read",
        "segments":            segments,
        "window_predictions":  window_preds,
        "processing_time_sec": round(time.perf_counter() - t0, 2),
    }


# ============================================================
# Reporting
# ============================================================

def format_report(result: dict, verbose: bool = False) -> str:
    lines = []
    lines.append(f"{'='*65}")
    lines.append(f"  File: {result['filename']}")
    lines.append(f"  Duration: {result['duration_sec']}s  |  "
                 f"Processed in: {result['processing_time_sec']}s")
    lines.append(f"{'='*65}")

    verdict = ("!! READING DETECTED !!" if result["overall_label"] == "read"
               else "OK — Spontaneous")
    lines.append(f"  VERDICT: {verdict}")
    lines.append(f"  Confidence: {result['overall_confidence']:.1%}")
    lines.append(f"  Read ratio: {result['read_ratio']:.1%} of speaking time")
    lines.append("")
    lines.append("  --- TIMELINE ---")

    for seg in result["segments"]:
        marker = "██" if seg["label"] == "read" else ("░░" if seg["label"] == "spontaneous" else "··")
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
                f"{wp['label']:12s} conf={wp['confidence']:.2f}"
            )

    lines.append("")
    return "\n".join(lines)


def format_summary_table(results: list[dict]) -> str:
    lines = []
    lines.append(f"\n{'='*85}")
    lines.append(f"  BATCH SUMMARY — {len(results)} files")
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

    read_n   = sum(1 for r in results if r["overall_label"] == "read")
    spont_n  = sum(1 for r in results if r["overall_label"] == "spontaneous")
    avg_conf = np.mean([r["overall_confidence"] for r in results])
    total_t  = sum(r["processing_time_sec"] for r in results)

    lines.append(f"  {'-'*78}")
    lines.append(f"  Read (cheating suspected): {read_n}")
    lines.append(f"  Spontaneous (OK):          {spont_n}")
    lines.append(f"  Avg confidence:            {avg_conf:.1%}")
    lines.append(f"  Total processing time:     {total_t:.1f}s")
    lines.append(f"{'='*85}")
    return "\n".join(lines)


# ============================================================
# CLI helpers
# ============================================================

def find_audio_files(path: Path) -> list[str]:
    AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".webm"}
    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        files = []
        for ext in AUDIO_EXTS:
            files.extend(path.rglob(f"*{ext}"))
            files.extend(path.rglob(f"*{ext.upper()}"))
        return sorted(set(str(f) for f in files))
    return []


def infer_window_sec_from_model_name(model_path: str, fallback: float = 5.0) -> float:
    """Infer window size from model filename. Exact map checked first, then patterns."""
    name = Path(model_path).name.lower()

    # Check exact filename map first — most reliable
    for fname, window in WINDOW_SEC_MAP.items():
        if fname.lower() == name:
            return window

    # Pattern matching — ORDER MATTERS (specific before generic)
    if "12_5sec" in name or "12.5sec" in name:
        return 12.5
    if "15sec" in name:
        return 15.0
    if "10sec" in name:
        return 10.0
    if "7_5sec" in name:      # must check before plain "7sec"
        return 7.5
    if "7sec" in name:
        return 7.5
    if "wavlm" in name:
        return 5.0
    if "5sec" in name:
        return 5.0

    return fallback


def resolve_model_path(model_arg: str) -> str:
    """Resolve alias or path string to an existing ONNX file path."""
    # Direct path
    candidate = Path(model_arg)
    if candidate.exists():
        return str(candidate)

    # Alias lookup
    alias_key = model_arg.lower().strip()
    if alias_key in MODEL_ALIAS_TO_FILE:
        for base in (Path("checkpoints"), Path(__file__).parent / "checkpoints"):
            p = base / MODEL_ALIAS_TO_FILE[alias_key]
            if p.exists():
                return str(p)

    # Relative to script directory
    script_relative = Path(__file__).parent / model_arg
    if script_relative.exists():
        return str(script_relative)

    return model_arg


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CPU Inference — Read vs Spontaneous Speech Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model aliases:
  5sec / wav2vec2          ->  speech_classifier_quant.onnx              (production)
  7sec / 7.5sec / 7_5sec  ->  speech_classifier_wav2vec2_7_5sec_quant.onnx
  10sec                    ->  speech_classifier_wav2vec2_10sec_quant.onnx
  12.5sec / 12_5sec        ->  speech_classifier_wav2vec2_12_5sec_quant.onnx
  15sec                    ->  speech_classifier_wav2vec2_15sec_quant.onnx
  wavlm / wavlm_5sec       ->  speech_classifier_wavlm_5sec_quant.onnx

Examples:
  python predict_cpu.py --audio interview.wav
  python predict_cpu.py --audio interview.wav --model 10sec
  python predict_cpu.py --audio interview.wav --model 7_5sec
  python predict_cpu.py --audio interview.wav --model wavlm
  python predict_cpu.py --audio recordings/ --output results.json --no-silero
        """,
    )
    parser.add_argument("--audio",          type=str, required=True,
                        help="Path to audio file or folder")
    parser.add_argument("--model",          type=str,
                        default="checkpoints/speech_classifier_quant.onnx",
                        help="Path or alias (default: 5sec wav2vec2)")
    parser.add_argument("--window-sec",     type=float, default=None,
                        help="Override window size in seconds (inferred from filename by default)")
    parser.add_argument("--output",         type=str,
                        default="outputs/cpu_predictions.json")
    parser.add_argument("--batch-size",     type=int, default=4)
    parser.add_argument("--read-threshold", type=float, default=None,
                        help="Override read_threshold (default: 0.45)")
    parser.add_argument("--verbose",        action="store_true",
                        help="Print per-window details")
    parser.add_argument("--no-silero",      action="store_true",
                        help="Use RMS VAD instead of Silero")
    parser.add_argument("--model-type", choices=["wav2vec2", "whisper"],
                    default=None,
                    help="Force model type (auto-detected from filename by default)")
    args = parser.parse_args()

    # --- Find audio files ---
    audio_files = find_audio_files(Path(args.audio))
    if not audio_files:
        print(f"ERROR: No audio files found at '{args.audio}'")
        sys.exit(1)
    print(f"Found {len(audio_files)} audio file(s)")

    # --- Resolve model ---
    model_path = resolve_model_path(args.model)
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: '{args.model}'")
        print("Run with --help to see available aliases")
        sys.exit(1)

    # --- Build config ---
    cfg = DEFAULT_CONFIG.copy()
    if args.read_threshold is not None:
        cfg["read_threshold"] = args.read_threshold

    # Window size: explicit flag > exact filename map > pattern matching > default
    if args.window_sec is not None:
        cfg["window_sec"] = args.window_sec
    else:
        cfg["window_sec"] = infer_window_sec_from_model_name(
            model_path, fallback=DEFAULT_CONFIG["window_sec"]
        )

    window_samples = int(cfg["window_sec"] * cfg["sample_rate"])
    model_mb = Path(model_path).stat().st_size / 1e6
# Auto-detect model type from filename if not specified
    model_name_lower = Path(model_path).name.lower()
    model_type = args.model_type
    if model_type is None:
        model_type = "whisper" if "whisper" in model_name_lower else "wav2vec2"

    if model_type == "whisper":
        classifier = WhisperONNXClassifier(model_path)
        cfg["window_samples"] = int(cfg["window_sec"] * cfg["sample_rate"])
    else:
        classifier = ONNXClassifier(model_path, window_samples=window_samples)
        cfg["window_samples"] = int(classifier.window_samples)

    print(f"Model type: {model_type}")
    effective_window_sec = cfg["window_samples"] / cfg["sample_rate"]

    print(f"Model:     {Path(model_path).name}  ({model_mb:.0f} MB)")
    print(f"Window:    {effective_window_sec:.2f}s  |  "
          f"Hop: {adaptive_hop(effective_window_sec):.2f}s  |  "
          f"Samples: {cfg['window_samples']}")
    if cfg["window_samples"] != window_samples:
        print(f"Note: requested {window_samples} samples from --window-sec/name, "
              f"but ONNX expects {cfg['window_samples']}; using ONNX shape.")
    print(f"Threshold: {cfg['read_threshold']}")

    print("Model loaded")

    # --- Load Silero VAD ---
    vad_model, get_ts_fn = None, None
    if not args.no_silero:
        try:
            vad_model, get_ts_fn = load_silero_vad()
            print("Silero VAD loaded")
        except Exception as e:
            print(f"Silero VAD unavailable ({e}) — using RMS VAD")
    else:
        print("Using RMS VAD (--no-silero)")

    print()

    # --- Run inference ---
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

    # --- Save JSON ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    
    main()