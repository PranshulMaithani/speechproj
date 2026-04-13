"""
Step 1: Transcribe all audio files using Whisper.
Saves transcripts as JSON (text + word-level timestamps + language).

Usage:
    python -m approach1.transcribe
    python -m approach1.transcribe --model medium  # use larger model
    python -m approach1.transcribe --audio-dir audios2  # only one dir
"""
import argparse
import json
import time
from pathlib import Path

import whisper
import numpy as np

from approach1.config import (
    AUDIO_DIRS, TRANSCRIPTS_DIR, WHISPER_MODEL, SAMPLE_RATE, MAX_DURATION_SEC,
    SPEECHTRY_ROOT,
)


def transcribe_file(model, audio_path: Path) -> dict:
    """Transcribe a single audio file. Returns dict with text, segments, language."""
    result = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        condition_on_previous_text=True,
        fp16=False,  # CPU-safe
    )
    return {
        "file": audio_path.name,
        "path": str(audio_path),
        "text": result["text"].strip(),
        "language": result.get("language", "en"),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "words": seg.get("words", []),
            }
            for seg in result.get("segments", [])
        ],
    }


def find_all_audio_files(audio_dirs: dict = None) -> list[Path]:
    """Find all .wav files in the audio directories."""
    if audio_dirs is None:
        audio_dirs = AUDIO_DIRS
    files = []
    for name, dir_path in audio_dirs.items():
        if not dir_path.exists():
            print(f"  WARNING: {dir_path} does not exist, skipping")
            continue
        wav_files = sorted(dir_path.rglob("*.wav"))
        print(f"  Found {len(wav_files)} wav files in {name}/")
        files.extend(wav_files)
    return files


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument("--model", default=WHISPER_MODEL, help="Whisper model size")
    parser.add_argument("--audio-dir", default=None, help="Only transcribe this dir (audios2 or audios4)")
    parser.add_argument("--force", action="store_true", help="Re-transcribe even if transcript exists")
    args = parser.parse_args()

    # Resolve audio dirs
    if args.audio_dir:
        if args.audio_dir not in AUDIO_DIRS:
            print(f"ERROR: unknown audio dir '{args.audio_dir}'. Choose from: {list(AUDIO_DIRS.keys())}")
            return
        dirs = {args.audio_dir: AUDIO_DIRS[args.audio_dir]}
    else:
        dirs = AUDIO_DIRS

    print(f"Loading Whisper '{args.model}' model...")
    model = whisper.load_model(args.model)
    print(f"  Model loaded. Device: {model.device}")

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = find_all_audio_files(dirs)
    print(f"\nTotal files to transcribe: {len(audio_files)}")

    done = 0
    skipped = 0
    errors = []
    t0 = time.time()

    for i, audio_path in enumerate(audio_files):
        # Use relative path as key for transcript filename
        # e.g., audios2/candidateid/candidateid_25.wav -> audios2__candidateid__candidateid_25.json
        rel = audio_path.relative_to(SPEECHTRY_ROOT)  # relative to speechtry/
        out_name = str(rel).replace("/", "__").replace("\\", "__").replace(".wav", ".json")
        out_path = TRANSCRIPTS_DIR / out_name

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            result = transcribe_file(model, audio_path)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            done += 1

            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(audio_files) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(audio_files)}] {audio_path.name} -> {len(result['text'])} chars "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        except Exception as e:
            errors.append((str(audio_path), str(e)))
            print(f"  [{i+1}/{len(audio_files)}] ERROR on {audio_path.name}: {e}")

    elapsed = time.time() - t0
    print(f"\nDone. Transcribed: {done}, Skipped (already done): {skipped}, Errors: {len(errors)}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if errors:
        err_path = TRANSCRIPTS_DIR / "_errors.json"
        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Error details saved to {err_path}")


if __name__ == "__main__":
    main()
