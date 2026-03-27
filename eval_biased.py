"""
Evaluation / Inference for Biased Wav2Vec2 (single-neuron)
============================================================
Runs inference on audio files and outputs JSON in the same format as the
old pipeline (segments, window_predictions, overall_label, etc.).

Supports both PyTorch and ONNX backends.

Usage:
    # Single file
    python eval_biased.py path/to/audio.wav

    # Directory of files
    python eval_biased.py path/to/folder/

    # With ONNX (faster, no GPU needed)
    python eval_biased.py path/to/audio.wav --backend onnx

    # Custom threshold
    python eval_biased.py path/to/audio.wav --threshold 0.70

    # Evaluate on test split from manifest
    python eval_biased.py --test-split
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "old"))

SAMPLE_RATE = 16000
WINDOW_SEC = 5.0
HOP_SEC = 2.5
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SEC * SAMPLE_RATE)
DEFAULT_THRESHOLD = 0.65
MIN_SPEECH_RATIO = 0.20

CKPT_DIR = Path("checkpoints_biased")
ONNX_PATH = CKPT_DIR / "biased_wav2vec2_quant.onnx"
PT_PATH = CKPT_DIR / "biased_wav2vec2_best.pt"
OUTPUT_DIR = Path("outputs_biased")


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
    cheating_suspected: bool
    segments: List[Segment]
    window_predictions: List[WindowPrediction]


def load_audio(path: str) -> np.ndarray:
    """Load audio file at 16kHz mono."""
    import librosa
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=120)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio


def compute_speech_ratio(audio_chunk: np.ndarray) -> float:
    """Simple energy-based speech ratio."""
    import librosa
    rms = librosa.feature.rms(y=audio_chunk, frame_length=512, hop_length=256)[0]
    rms_max = float(np.max(rms))
    if rms_max < 1e-8:
        return 0.0
    noise_floor = np.percentile(rms, 10)
    signal_peak = np.percentile(rms, 95)
    thresh = noise_floor + (signal_peak - noise_floor) * 0.15
    thresh = max(thresh, rms_max * 0.05)
    return float((rms > thresh).mean())


class ONNXBackend:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        """Returns P(read) for each window."""
        logits = self.session.run(["logit"], {"input_values": windows.astype(np.float32)})[0]
        return 1.0 / (1.0 + np.exp(-logits))  # sigmoid


class PyTorchBackend:
    def __init__(self, model_path: str):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        from train_biased_wav2vec2 import BiasedSpeechClassifier
        w2v_cfg = checkpoint["config"]
        self.model = BiasedSpeechClassifier(
            model_name=w2v_cfg["model_name"],
            hidden_size=w2v_cfg["hidden_size"],
            dropout=0.0,
            freeze_layers=0,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        import torch
        x = torch.tensor(windows, dtype=torch.float32).to(self.device)
        logits = self.model(x)
        return torch.sigmoid(logits).cpu().numpy()


def predict_file(audio_path: str, backend, threshold: float) -> PredictionResult:
    """Run inference on a single audio file."""
    audio = load_audio(audio_path)
    duration_sec = len(audio) / SAMPLE_RATE

    # Create windows
    windows = []
    window_meta = []
    idx = 0
    start = 0
    while start < len(audio):
        end = start + WINDOW_SAMPLES
        chunk = audio[start:end]
        if len(chunk) < WINDOW_SAMPLES // 2:
            break
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))

        speech_ratio = compute_speech_ratio(chunk)
        window_meta.append({
            "idx": idx,
            "start_sec": round(start / SAMPLE_RATE, 2),
            "end_sec": round(min(end, len(audio)) / SAMPLE_RATE, 2),
            "speech_ratio": round(speech_ratio, 3),
        })
        windows.append(chunk)
        idx += 1
        start += HOP_SAMPLES

    if not windows:
        return PredictionResult(
            filepath=audio_path,
            filename=Path(audio_path).name,
            duration_sec=round(duration_sec, 2),
            overall_label="silence",
            overall_confidence=1.0,
            read_ratio=0.0,
            cheating_suspected=False,
            segments=[Segment(0.0, round(duration_sec, 2), round(duration_sec, 2), "silence", 1.0)],
            window_predictions=[],
        )

    # Batch inference
    windows_arr = np.array(windows, dtype=np.float32)
    probs = backend.predict_batch(windows_arr)  # P(read) per window
    if probs.ndim > 1:
        probs = probs.squeeze(-1)

    # Build window predictions
    win_preds = []
    for i, meta in enumerate(window_meta):
        p_read = float(probs[i])
        sr = meta["speech_ratio"]

        if sr < MIN_SPEECH_RATIO:
            label = "silence"
            confidence = 1.0
        elif p_read >= threshold:
            label = "read"
            confidence = round(p_read, 3)
        else:
            label = "spontaneous"
            confidence = round(1.0 - p_read, 3)

        win_preds.append(WindowPrediction(
            window_idx=meta["idx"],
            start_sec=meta["start_sec"],
            end_sec=meta["end_sec"],
            label=label,
            confidence=confidence,
            speech_ratio=sr,
        ))

    # Merge consecutive same-label windows into segments
    segments = []
    current_label = win_preds[0].label
    seg_start = win_preds[0].start_sec
    seg_confs = [win_preds[0].confidence]

    for wp in win_preds[1:]:
        if wp.label == current_label:
            seg_confs.append(wp.confidence)
        else:
            seg_end = wp.start_sec
            segments.append(Segment(
                start_sec=seg_start,
                end_sec=round(seg_end, 2),
                duration_sec=round(seg_end - seg_start, 2),
                label=current_label,
                confidence=round(float(np.mean(seg_confs)), 3),
            ))
            current_label = wp.label
            seg_start = wp.start_sec
            seg_confs = [wp.confidence]

    # Close last segment
    seg_end = win_preds[-1].end_sec
    segments.append(Segment(
        start_sec=seg_start,
        end_sec=round(seg_end, 2),
        duration_sec=round(seg_end - seg_start, 2),
        label=current_label,
        confidence=round(float(np.mean(seg_confs)), 3),
    ))

    # Overall stats (only over speaking windows)
    speaking = [wp for wp in win_preds if wp.label != "silence"]
    if speaking:
        read_count = sum(1 for wp in speaking if wp.label == "read")
        read_ratio = round(read_count / len(speaking), 3)
        if read_ratio > 0.5:
            overall_label = "read"
            overall_confidence = round(read_ratio, 3)
        else:
            overall_label = "spontaneous"
            overall_confidence = round(1.0 - read_ratio, 3)
    else:
        overall_label = "silence"
        overall_confidence = 1.0
        read_ratio = 0.0

    return PredictionResult(
        filepath=audio_path,
        filename=Path(audio_path).name,
        duration_sec=round(duration_sec, 2),
        overall_label=overall_label,
        overall_confidence=overall_confidence,
        read_ratio=read_ratio,
        cheating_suspected=(overall_label == "read"),
        segments=segments,
        window_predictions=win_preds,
    )


def eval_test_split(backend, threshold: float):
    """Evaluate on the test split from the manifest."""
    import pandas as pd
    import yaml
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    with open("old/configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["paths"]["data_root"])
    manifest_path = data_root / "old" / cfg["paths"]["manifest_csv"]
    df = pd.read_csv(manifest_path)

    # Filter to existing files + test split
    df = df[df["filepath"].apply(os.path.exists)]
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test set: {len(test_df)} files")

    results = []
    all_true, all_pred = [], []

    for i, row in test_df.iterrows():
        result = predict_file(row["filepath"], backend, threshold)
        results.append(result)

        true_label = "read" if row["label_int"] == 1 else "spontaneous"
        all_true.append(row["label_int"])
        pred_int = 1 if result.overall_label == "read" else 0
        all_pred.append(pred_int)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_df)} files...")

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, zero_division=0)
    print(f"\nFile-level results (threshold={threshold}):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"\n{classification_report(all_true, all_pred, target_names=['spontaneous', 'read'])}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Eval biased wav2vec2")
    parser.add_argument("input", nargs="?", help="Audio file or directory")
    parser.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--test-split", action="store_true", help="Evaluate on manifest test split")
    args = parser.parse_args()

    # Load backend
    if args.backend == "onnx":
        if not ONNX_PATH.exists():
            print(f"ONNX model not found at {ONNX_PATH}. Run export_biased_onnx.py first.")
            sys.exit(1)
        print(f"Backend: ONNX ({ONNX_PATH})")
        backend = ONNXBackend(str(ONNX_PATH))
    else:
        print(f"Backend: PyTorch ({PT_PATH})")
        import torch
        backend = PyTorchBackend(str(PT_PATH))

    print(f"Read threshold: {args.threshold}")

    # Test split mode
    if args.test_split:
        results = eval_test_split(backend, args.threshold)
        output_path = args.output or str(OUTPUT_DIR / "test_predictions.json")
    else:
        if not args.input:
            parser.error("Provide an audio file/directory or use --test-split")

        input_path = Path(args.input)
        if input_path.is_file():
            audio_files = [str(input_path)]
        elif input_path.is_dir():
            exts = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".aac", ".wma", ".mp4"}
            audio_files = sorted(
                str(p) for p in input_path.iterdir() if p.suffix.lower() in exts
            )
            print(f"Found {len(audio_files)} audio files")
        else:
            print(f"Not found: {input_path}")
            sys.exit(1)

        results = []
        for i, fpath in enumerate(audio_files):
            t0 = time.perf_counter()
            result = predict_file(fpath, backend, args.threshold)
            elapsed = time.perf_counter() - t0
            flag = " ** CHEATING" if result.cheating_suspected else ""
            print(f"  [{i+1}/{len(audio_files)}] {result.filename}: "
                  f"{result.overall_label} (conf={result.overall_confidence:.3f}, "
                  f"read_ratio={result.read_ratio:.3f}) [{elapsed:.1f}s]{flag}")
            results.append(result)

        output_path = args.output or str(OUTPUT_DIR / "predictions.json")

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
