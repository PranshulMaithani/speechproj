"""Transcribe a batch's audio -> <batch>_transcripts.json   (COMPANY LAPTOP ONLY).

Step that runs BEFORE neural_baseline_prep.py for any batch whose handcrafted
text features you want computed from transcripts (rather than a precomputed
<batch>_features.csv). It walks companylaptop/<batch>/ and writes

    companylaptop/<batch>_transcripts.json   =   { "<raw_filename>": "<text>", ... }

keyed by the ORIGINAL on-disk filename (e.g. "John123_25.wav") -- exactly the
key neural_baseline_prep.py looks up (it matches transcripts by src.name). prep
then turns each transcript into feat_* numbers via text_features.py; the
transcript TEXT and the real filename never leave this machine. Only the numeric
feat_* columns (in the anonymized gt.csv) and the .npy are uploaded.

Uses faster-whisper (the CPU transcriber already in companylaptop/requirements
.txt). Incremental: re-running skips files already in the json (so you can stop
and resume, or add a few new files). Pass --force to re-transcribe everything.

Run (transcribe audios7 with the medium model on CPU):
    python companylaptop/transcribe_batch.py --batch audios7

    # faster / slower-but-better model, or GPU if available:
    python companylaptop/transcribe_batch.py --batch audios7 --model small
    python companylaptop/transcribe_batch.py --batch audios7 --model large-v3 \
        --device cuda --compute_type float16

Then run neural_baseline_prep.py -- it auto-discovers audios7 and computes the
feat_* columns from this json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Matches ACCEPT_EXT in neural_baseline_prep.py so the two scripts see the
# exact same set of files.
ACCEPT_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


def transcribe_one(model, audio_path: Path, language: str, vad: bool) -> str:
    """Return the full transcript text for one file (segments joined)."""
    segments, _info = model.transcribe(
        str(audio_path),
        language=(language or None),
        vad_filter=vad,
        condition_on_previous_text=False,  # avoids run-on hallucination across silences
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True,
                    help="batch name / folder under --audio_root, e.g. audios7")
    ap.add_argument("--audio_root", default="",
                    help="folder containing <batch>/ (default: this script's dir)")
    ap.add_argument("--model", default="medium",
                    help="faster-whisper model: tiny/base/small/medium/large-v3. "
                         "medium = good quality on CPU; small/base = faster; "
                         "large-v3 = best (slow on CPU, use --device cuda).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--compute_type", default="int8",
                    help="int8 (CPU, default) / float16 (GPU) / int8_float16.")
    ap.add_argument("--language", default="en",
                    help="force language code (default en); empty = auto-detect.")
    ap.add_argument("--no_vad", action="store_true",
                    help="disable the silence (VAD) filter.")
    ap.add_argument("--force", action="store_true",
                    help="re-transcribe everything, ignoring the existing json.")
    args = ap.parse_args()

    audio_root = Path(args.audio_root) if args.audio_root else Path(__file__).resolve().parent
    batch_dir = audio_root / args.batch
    out_path = audio_root / f"{args.batch}_transcripts.json"

    if not batch_dir.is_dir():
        print(f"ERROR: batch dir not found: {batch_dir}", file=sys.stderr)
        return 1

    files = sorted(f for f in batch_dir.rglob("*")
                   if f.is_file() and f.suffix.lower() in ACCEPT_EXT)
    if not files:
        print(f"ERROR: no audio files ({sorted(ACCEPT_EXT)}) under {batch_dir}",
              file=sys.stderr)
        return 1

    existing: dict[str, str] = {}
    if out_path.exists() and not args.force:
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
    todo = [f for f in files if f.name not in existing]
    print(f"{args.batch}: {len(files)} audio files, {len(existing)} already "
          f"transcribed, {len(todo)} to do -> {out_path.name}")
    if not todo:
        print("Nothing to transcribe. (use --force to redo)")
        return 0

    # Import here so --help works without the dependency installed.
    from faster_whisper import WhisperModel
    print(f"Loading faster-whisper '{args.model}' on {args.device} "
          f"({args.compute_type}) ...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    transcripts = dict(existing)
    t0 = time.time()
    n_empty = 0
    for f in tqdm(todo, desc=f"transcribe {args.batch}", unit="file"):
        try:
            text = transcribe_one(model, f, args.language, vad=not args.no_vad)
        except Exception as e:
            print(f"  WARN: failed on {f.name}: {e} -- storing empty", file=sys.stderr)
            text = ""
        if not text:
            n_empty += 1
        transcripts[f.name] = text
        # Save after every file so a crash/stop loses nothing (resume-safe).
        out_path.write_text(json.dumps(transcripts, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print(f"done in {time.time() - t0:.1f}s. wrote {len(transcripts)} entries "
          f"({n_empty} empty) to {out_path}")
    print("KEEP THIS LOCAL. Next: python companylaptop/neural_baseline_prep.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
