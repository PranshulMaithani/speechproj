"""
Step 4: Run inference on new audio files.
Transcribes with Whisper, extracts features, runs XGBoost, outputs JSON results.

Usage:
    python -m approach1.predict --audio path/to/audio_or_folder
    python -m approach1.predict --audio audios_new/ --model outputs/approach1/models/xgboost_combined.pkl
    python -m approach1.predict --audio single_file.wav
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import whisper

from approach1.config import MODEL_DIR, OUTPUT_DIR, SAMPLE_RATE, MAX_DURATION_SEC, WHISPER_MODEL
from approach1.text_features import extract_text_features
from approach1.acoustic_features import extract_all_acoustic_features


def load_model(model_path: Path = None):
    """Load trained XGBoost model."""
    if model_path is None:
        # Try combined first, then text-only
        for tag in ["combined", "text", "acoustic"]:
            candidate = MODEL_DIR / f"xgboost_{tag}.pkl"
            if candidate.exists():
                model_path = candidate
                break
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"No trained model found. Run training first: python -m approach1.train")

    with open(model_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded model from {model_path}")
    print(f"  Features: {len(data['feature_names'])} ({data['feature_names'][0]}...)")
    return data["model"], data["feature_names"]


def predict_single(
    audio_path: Path,
    whisper_model,
    xgb_model,
    feature_names: list[str],
) -> dict:
    """Run full pipeline on a single audio file."""
    t0 = time.time()

    # ── Transcribe ────────────────────────────────────────────────────
    result = whisper_model.transcribe(
        str(audio_path), language="en", word_timestamps=True,
        condition_on_previous_text=True, fp16=False,
    )
    transcript = result["text"].strip()
    duration_sec = result["segments"][-1]["end"] if result.get("segments") else None

    # ── Extract text features ─────────────────────────────────────────
    text_feats = extract_text_features(transcript, duration_sec)

    # ── Extract acoustic features ─────────────────────────────────────
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION_SEC)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    if duration_sec is None:
        duration_sec = len(audio) / sr
    acoustic_feats = extract_all_acoustic_features(audio, sr)

    # ── Build feature vector ──────────────────────────────────────────
    combined = {}
    combined.update({f"text__{k}": v for k, v in text_feats.items()})
    combined.update({f"acoustic__{k}": v for k, v in acoustic_feats.items()})

    X = np.array([[combined.get(f, 0.0) for f in feature_names]], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Predict ───────────────────────────────────────────────────────
    prob = xgb_model.predict_proba(X)[0]
    pred_label = int(xgb_model.predict(X)[0])
    confidence = float(prob[pred_label])

    # Determine verdict
    if prob[1] >= 0.6:
        verdict = "CHEATING"
    elif prob[1] >= 0.4:
        verdict = "UNCERTAIN"
    else:
        verdict = "GENUINE"

    elapsed = time.time() - t0

    return {
        "file": audio_path.name,
        "path": str(audio_path),
        "verdict": verdict,
        "cheating_probability": round(float(prob[1]), 4),
        "confidence": round(confidence, 4),
        "transcript": transcript,
        "duration_sec": round(duration_sec, 2) if duration_sec else None,
        "inference_time_sec": round(elapsed, 2),
        # Key features for interpretability
        "key_signals": {
            "type_token_ratio": round(text_feats.get("type_token_ratio", 0), 4),
            "filler_rate": round(text_feats.get("filler_rate", 0), 4),
            "complex_word_rate": round(text_feats.get("complex_word_rate", 0), 4),
            "repetition_rate": round(text_feats.get("repetition_rate_bigram", 0), 4),
            "self_reference_rate": round(text_feats.get("self_reference_rate", 0), 4),
            "words_per_minute": round(text_feats.get("words_per_minute", 0), 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Predict cheating on audio files")
    parser.add_argument("--audio", required=True, help="Audio file or directory to process")
    parser.add_argument("--model", default=None, help="Path to trained model .pkl")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL, help="Whisper model size")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if audio_path.is_file():
        audio_files = [audio_path]
    elif audio_path.is_dir():
        audio_files = sorted(audio_path.rglob("*.wav"))
        if not audio_files:
            audio_files = sorted(audio_path.rglob("*.m4a")) + sorted(audio_path.rglob("*.mp3"))
        print(f"Found {len(audio_files)} audio files in {audio_path}")
    else:
        print(f"ERROR: {audio_path} not found")
        return

    # Load models
    model_path = Path(args.model) if args.model else None
    xgb_model, feature_names = load_model(model_path)

    print(f"Loading Whisper '{args.whisper_model}'...")
    whisper_model = whisper.load_model(args.whisper_model)

    # Run predictions
    results = []
    for i, f in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] {f.name}")
        try:
            res = predict_single(f, whisper_model, xgb_model, feature_names)
            results.append(res)
            print(f"  Verdict: {res['verdict']} (P(cheat)={res['cheating_probability']:.3f})")
            print(f"  Key: TTR={res['key_signals']['type_token_ratio']:.3f}, "
                  f"fillers={res['key_signals']['filler_rate']:.3f}, "
                  f"complex={res['key_signals']['complex_word_rate']:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"file": f.name, "error": str(e)})

    # Save results
    out_path = Path(args.output) if args.output else OUTPUT_DIR / "predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    # Summary
    verdicts = [r.get("verdict", "ERROR") for r in results]
    print(f"\nSummary: {len(results)} files processed")
    print(f"  CHEATING:  {verdicts.count('CHEATING')}")
    print(f"  UNCERTAIN: {verdicts.count('UNCERTAIN')}")
    print(f"  GENUINE:   {verdicts.count('GENUINE')}")
    print(f"  ERRORS:    {verdicts.count('ERROR')}")


if __name__ == "__main__":
    main()
