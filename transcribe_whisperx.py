"""
Transcribe audio files using WhisperX (verbatim + word timestamps).
==================================================================
Uses Whisper large-v2 for transcription + wav2vec2 forced alignment
for accurate word-level timestamps needed for pause analysis.

Usage:
    # Transcribe ALLSSTAR
    python transcribe_whisperx.py --manifest old/outputs/manifest_expanded.csv --output datasets/transcripts_allsstar.json

    # Single file
    python transcribe_whisperx.py --audio path/to/audio.wav
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import whisperx
from tqdm import tqdm


def load_models(device="cuda", compute_type="float16"):
    """Load WhisperX transcription + alignment models."""
    print("Loading WhisperX large-v2...")
    model = whisperx.load_model("large-v2", device=device, compute_type=compute_type,
                                language="en")

    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    mem_gb = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"  Models loaded. GPU mem: {mem_gb:.1f}GB")
    return model, model_a, metadata


def transcribe_audio(audio_path, model, model_a, metadata, device="cuda"):
    """
    Transcribe a single audio file. Returns dict with:
    - text: full transcript
    - words: list of {word, start, end} dicts
    - duration_sec: audio duration
    """
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000

    if duration < 0.5:
        return {"text": "", "words": [], "duration_sec": round(duration, 2)}

    # Transcribe
    result = model.transcribe(audio, batch_size=8, language="en")

    if not result.get("segments"):
        return {"text": "", "words": [], "duration_sec": round(duration, 2)}

    # Align for word-level timestamps
    aligned = whisperx.align(
        result["segments"], model_a, metadata, audio,
        device=device, return_char_alignments=False
    )

    # Collect all words
    all_words = []
    all_text = []
    for seg in aligned.get("segments", []):
        all_text.append(seg.get("text", "").strip())
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                all_words.append({
                    "word": w["word"],
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })

    return {
        "text": " ".join(all_text),
        "words": all_words,
        "duration_sec": round(duration, 2),
    }


def transcribe_manifest(manifest_path, output_path, device="cuda", max_files=None):
    """Transcribe all files in a manifest CSV."""
    df = pd.read_csv(manifest_path)
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    if max_files:
        df = df.head(max_files)

    print(f"Transcribing {len(df)} files from {manifest_path}")

    model, model_a, metadata = load_models(device)

    # Load existing results if resuming
    results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                results = json.load(f)
            print(f"  Resuming: {len(results)} already done")
        except Exception:
            pass

    errors = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        filepath = row["filepath"]

        # Skip if already transcribed
        if filepath in results:
            continue

        try:
            t0 = time.perf_counter()
            result = transcribe_audio(filepath, model, model_a, metadata, device)
            elapsed = time.perf_counter() - t0

            result["filepath"] = filepath
            result["filename"] = row.get("filename", Path(filepath).name)
            result["label"] = row.get("label", "")
            result["label_int"] = int(row.get("label_int", -1))
            result["source"] = row.get("source", "")
            result["processing_time_sec"] = round(elapsed, 2)

            results[filepath] = result

            # Save every 10 files
            if (len(results)) % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\n  ERROR on {Path(filepath).name}: {e}")
            errors.append({"filepath": filepath, "error": str(e)})

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(results)} transcribed, {len(errors)} errors")
    print(f"Output: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Transcribe with WhisperX")
    parser.add_argument("--manifest", type=str, help="Manifest CSV to transcribe")
    parser.add_argument("--audio", type=str, help="Single audio file")
    parser.add_argument("--output", type=str, default="datasets/transcripts_allsstar.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    if args.audio:
        model, model_a, metadata = load_models(args.device)
        t0 = time.perf_counter()
        result = transcribe_audio(args.audio, model, model_a, metadata, args.device)
        elapsed = time.perf_counter() - t0

        print(f"\nFile: {args.audio}")
        print(f"Duration: {result['duration_sec']}s | Processed in: {elapsed:.1f}s")
        print(f"\nTranscript:\n  {result['text'][:300]}")
        print(f"\nWords ({len(result['words'])}):")
        for w in result["words"][:20]:
            print(f"  [{w['start']:6.2f}s - {w['end']:6.2f}s] {w['word']}")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    elif args.manifest:
        transcribe_manifest(args.manifest, args.output, args.device, args.max_files)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
