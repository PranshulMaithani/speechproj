#!/usr/bin/env python3
"""Delete non-target-question audio to save disk space. SAFE: dry-run by default.

New batches arrive nested as:
    audios<N>/<ciid>/<ciid>_<q>.wav      for q = 1 .. 27

The per-audio cheating system only uses Q25/26/27, so every other question's file is
dead weight on disk. This script deletes them -- but ONLY when you pass --apply.

  * DRY-RUN (default): reports, per batch, how many files and how many MB WOULD be
    freed, and a few example paths. Deletes NOTHING.
  * --apply: actually deletes the non-kept files.
  * Files whose question id cannot be parsed from the name are NEVER touched.
  * --keep sets the questions to KEEP (default 25,26,27).

This is intentionally a separate, explicit tool (not silent in the training run)
because deletion is irreversible. train_pipeline.py can invoke it with --apply only
when you set prune_questions: true in the config.

Run:
    python companylaptop/prune_questions.py --batch audios8              # preview
    python companylaptop/prune_questions.py --batch audios8 --apply      # delete
    python companylaptop/prune_questions.py --all                        # preview all
    python companylaptop/prune_questions.py --all --apply --keep 25,26,27
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ACCEPT_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
_QID_TAIL_RE = re.compile(r"_(\d{1,3})$")
BATCH_DIR_RE = re.compile(r"^audios\d+$")


def qid_of(stem: str) -> int | None:
    """Trailing question id from a '<cid>_<qid>' stem (e.g. ciid_25 -> 25)."""
    m = _QID_TAIL_RE.search(stem)
    return int(m.group(1)) if m else None


def discover_batches(audio_root: Path) -> list[str]:
    """Every audios<N>/ folder with a sibling <batch>GT.csv, numerically ordered."""
    out = [d.name for d in audio_root.glob("audios*")
           if d.is_dir() and BATCH_DIR_RE.match(d.name)
           and (audio_root / f"{d.name}GT.csv").exists()]
    out.sort(key=lambda b: (int(b[6:]) if b[6:].isdigit() else 10**9, b))
    return out


def plan_batch(batch_dir: Path, keep: set[int]) -> tuple[list[Path], int, int, int]:
    """Return (files_to_delete, n_kept, n_unparseable, bytes_to_free) for one batch.
    Unparseable-qid files are counted separately and never scheduled for deletion."""
    to_delete: list[Path] = []
    n_kept = n_unparseable = 0
    bytes_free = 0
    for f in batch_dir.rglob("*"):
        if not (f.is_file() and f.suffix.lower() in ACCEPT_EXT):
            continue
        q = qid_of(f.stem)
        if q is None:
            n_unparseable += 1
            continue
        if q in keep:
            n_kept += 1
        else:
            to_delete.append(f)
            try:
                bytes_free += f.stat().st_size
            except OSError:
                pass
    return to_delete, n_kept, n_unparseable, bytes_free


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune non-Q25/26/27 audio (dry-run by default)")
    ap.add_argument("--audio_root", default="",
                    help="folder holding audios<N>/ (default: this script's dir, companylaptop/)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch", help="one batch, e.g. audios8")
    g.add_argument("--all", action="store_true", help="every discovered audios<N>/")
    ap.add_argument("--keep", default="25,26,27", help="comma qids to KEEP (default 25,26,27)")
    ap.add_argument("--apply", action="store_true",
                    help="ACTUALLY delete (without this it is a dry run and deletes nothing)")
    args = ap.parse_args()

    audio_root = Path(args.audio_root) if args.audio_root else Path(__file__).resolve().parent
    keep = {int(x) for x in args.keep.split(",") if x.strip().isdigit()}
    if not keep:
        print("ERROR: --keep parsed to an empty set", file=sys.stderr)
        return 2
    batches = discover_batches(audio_root) if args.all else [args.batch]

    mode = "APPLY (deleting)" if args.apply else "DRY-RUN (nothing deleted)"
    print(f"prune_questions :: keep={sorted(keep)}  root={audio_root}  mode={mode}")
    total_del = total_bytes = 0
    for b in batches:
        bd = audio_root / b
        if not bd.is_dir():
            print(f"  {b}: MISSING dir {bd} -- skipping")
            continue
        to_delete, n_kept, n_unparse, bytes_free = plan_batch(bd, keep)
        print(f"  {b}: keep={n_kept}  delete={len(to_delete)}  unparseable(kept)={n_unparse}"
              f"  -> {bytes_free/1e6:.1f} MB")
        for f in to_delete[:3]:
            print(f"      e.g. {f.relative_to(audio_root)}")
        if args.apply:
            freed = 0
            for f in to_delete:
                try:
                    freed += f.stat().st_size
                    f.unlink()
                except OSError as e:
                    print(f"      WARN could not delete {f}: {e}")
            print(f"      deleted {len(to_delete)} files, freed {freed/1e6:.1f} MB")
        total_del += len(to_delete)
        total_bytes += bytes_free

    verb = "Deleted" if args.apply else "Would delete"
    print(f"{verb} {total_del} files across {len(batches)} batch(es) "
          f"(~{total_bytes/1e6:.1f} MB).")
    if not args.apply and total_del:
        print("Re-run with --apply to actually delete. (Irreversible.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
