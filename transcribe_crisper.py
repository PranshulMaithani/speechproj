"""
Transcribe audio files using CrisperWhisper (verbatim + word timestamps).
==========================================================================
Preserves disfluencies (um, uh, false starts, repetitions) and gives
accurate word-level timestamps for pause analysis.

Usage:
    # Transcribe ALLSSTAR
    python transcribe_crisper.py --manifest old/outputs/manifest_expanded.csv --output datasets/transcripts_allsstar.json

    # Transcribe LibriSpeech (skip — already has transcripts, but can add timestamps)
    python transcribe_crisper.py --manifest datasets/librispeech/manifest.csv --output datasets/transcripts_libri.json

    # Transcribe any audio file/folder
    python transcribe_crisper.py --audio path/to/audio.wav --output result.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm


def load_crisper_whisper(device="cuda"):
    """Load CrisperWhisper model."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    model_name = "nyrahealth/CrisperWhisper"
    print(f"Loading CrisperWhisper ({model_name})...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    mem_gb = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"  Model loaded. GPU mem: {mem_gb:.1f}GB")
    return model, processor


def transcribe_audio(audio_path, model, processor, device="cuda", sr=16000):
    """
    Transcribe a single audio file. Returns dict with:
    - text: full verbatim transcript
    - words: list of {word, start, end} dicts
    - language: detected language
    """
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    if len(audio) == 0:
        return {"text": "", "words": [], "language": "en", "duration_sec": 0}

    duration = len(audio) / sr

    # For long audio, process in chunks of 30 seconds (Whisper's window)
    # CrisperWhisper handles this internally but we need to manage memory
    chunk_size = 30 * sr  # 30 seconds
    all_words = []
    all_text_parts = []

    for chunk_start in range(0, len(audio), chunk_size):
        chunk = audio[chunk_start:chunk_start + chunk_size]
        if len(chunk) < sr:  # Skip chunks shorter than 1 second
            continue

        offset_sec = chunk_start / sr

        inputs = processor(
            chunk, sampling_rate=sr, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device, dtype=torch.float16)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                return_timestamps=True,
                language="en",
                task="transcribe",
            )

        # Decode with timestamps
        output = processor.batch_decode(
            predicted_ids, skip_special_tokens=False, output_offsets=True
        )

        if output and len(output) > 0:
            decoded = output[0]

            # Extract text
            text_clean = processor.decode(predicted_ids[0], skip_special_tokens=True)
            all_text_parts.append(text_clean.strip())

            # Extract word-level timestamps from offsets
            if isinstance(decoded, dict) and "offsets" in decoded:
                for offset_info in decoded["offsets"]:
                    word = offset_info.get("text", "").strip()
                    if not word:
                        continue
                    start = offset_info.get("timestamp", (0, 0))
                    if isinstance(start, (list, tuple)) and len(start) == 2:
                        w_start, w_end = float(start[0]) + offset_sec, float(start[1]) + offset_sec
                    else:
                        w_start, w_end = 0, 0
                    all_words.append({
                        "word": word,
                        "start": round(w_start, 3),
                        "end": round(w_end, 3),
                    })

    full_text = " ".join(all_text_parts)
    return {
        "text": full_text,
        "words": all_words,
        "language": "en",
        "duration_sec": round(duration, 2),
    }


def transcribe_manifest(manifest_path, output_path, device="cuda", max_files=None):
    """Transcribe all files in a manifest CSV."""
    df = pd.read_csv(manifest_path)

    # Filter to existing files
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    if max_files:
        df = df.head(max_files)

    print(f"Transcribing {len(df)} files from {manifest_path}")

    model, processor = load_crisper_whisper(device)

    results = {}
    errors = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        filepath = row["filepath"]
        try:
            t0 = time.perf_counter()
            result = transcribe_audio(filepath, model, processor, device)
            elapsed = time.perf_counter() - t0

            result["filepath"] = filepath
            result["filename"] = row.get("filename", Path(filepath).name)
            result["label"] = row.get("label", "")
            result["label_int"] = int(row.get("label_int", -1))
            result["source"] = row.get("source", "")
            result["processing_time_sec"] = round(elapsed, 2)

            results[filepath] = result

            if (i + 1) % 20 == 0:
                # Save intermediate results
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


def transcribe_single(audio_path, device="cuda"):
    """Transcribe a single file and print results."""
    model, processor = load_crisper_whisper(device)

    t0 = time.perf_counter()
    result = transcribe_audio(audio_path, model, processor, device)
    elapsed = time.perf_counter() - t0

    print(f"\nFile: {audio_path}")
    print(f"Duration: {result['duration_sec']}s | Processed in: {elapsed:.1f}s")
    print(f"\nTranscript:\n  {result['text']}")
    print(f"\nWords ({len(result['words'])}):")
    for w in result["words"][:30]:
        print(f"  [{w['start']:6.2f}s - {w['end']:6.2f}s] {w['word']}")
    if len(result["words"]) > 30:
        print(f"  ... and {len(result['words']) - 30} more")

    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe with CrisperWhisper")
    parser.add_argument("--manifest", type=str, help="Manifest CSV to transcribe")
    parser.add_argument("--audio", type=str, help="Single audio file or directory")
    parser.add_argument("--output", type=str, default="datasets/transcripts.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    if args.audio:
        p = Path(args.audio)
        if p.is_file():
            result = transcribe_single(str(p), args.device)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        elif p.is_dir():
            exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
            files = sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in exts)
            print(f"Found {len(files)} audio files in {p}")
            # Create temporary manifest
            import csv
            tmp_manifest = Path(args.output).parent / "tmp_manifest.csv"
            with open(tmp_manifest, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filepath", "filename", "label", "label_int", "source"])
                for fp in files:
                    w.writerow([fp, Path(fp).name, "", -1, "custom"])
            transcribe_manifest(str(tmp_manifest), args.output, args.device, args.max_files)
            tmp_manifest.unlink()
    elif args.manifest:
        transcribe_manifest(args.manifest, args.output, args.device, args.max_files)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
