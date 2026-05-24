#!/usr/bin/env python3
"""Backfill the missing feat_n_words_asr column in gt.csv.

================================================================================
WHY
================================================================================
domain_diagnose.py section 0 showed:
  feat_n_words_asr  nan_rate_A=0.0000  nan_rate_B=1.0000  delta=1.0000

i.e. every audios6 row has NaN for feat_n_words_asr, while every audios2/4/5
row has a value. The feature was originally computed in
companylaptop/cheating_detection_v3.ipynb and never made it into the EC2
extraction code (ec2/full_text_features.py), so re-extracting on EC2 for
audios6 leaves the column missing.

This script counts words from each row's transcript (len(text.split())) and
writes the result into gt.csv under feat_n_words_asr. Idempotent: skips rows
that already have a numeric value unless --force.

================================================================================
WHERE TO RUN THIS  (PII CONSTRAINT)
================================================================================
RUN THIS ON THE COMPANY LAPTOP, NOT ON EC2.

Transcripts contain raw candidate speech and are PII-gated -- they must not
leave the laptop. gt.csv with G_NNNNN anonymized filenames IS allowed on EC2.

Workflow:
  1. On LAPTOP: run this script. It modifies gt.csv in place.
  2. From LAPTOP -> EC2: scp the updated gt.csv only. Leave transcripts.json
     on the laptop.

Easier alternative: don't backfill at all. The v2 training scripts already
include feat_n_words_asr in HIGH_KS_FEATURES, so --drop_high_ks_features=true
removes it from training. The feature was all-NaN on audios6 anyway (silently
filled to 0.0 during training), so dropping it loses nothing on client B.
Only run this backfill if you specifically want to keep word-count as a
training signal on client A.

================================================================================
EXACT RUN COMMANDS  (run on LAPTOP)
================================================================================

# Default: backfill ONLY audios6 (the one batch with the gap)
python ec2/backfill_n_words_asr.py \\
    --data_dir <LAPTOP_DATA_DIR> \\
    --transcripts <LAPTOP_DATA_DIR>/transcripts.json \\
    --batches audios6

# Then upload ONLY the updated gt.csv to EC2:
#   scp <LAPTOP_DATA_DIR>/gt.csv ubuntu@<ec2>:/home/ubuntu/nn/data/gt.csv

# Multiple batches at once (comma-separated)
python ec2/backfill_n_words_asr.py \\
    --data_dir <UPLOAD> \\
    --transcripts <UPLOAD>/transcripts.json \\
    --batches audios6,audios5

# All rows where feat_n_words_asr is NaN (any batch)
python ec2/backfill_n_words_asr.py \\
    --data_dir <UPLOAD> \\
    --transcripts <UPLOAD>/transcripts.json \\
    --batches ""

# Recompute even where a value is already present (audit / consistency check)
python ec2/backfill_n_words_asr.py \\
    --data_dir <UPLOAD> \\
    --transcripts <UPLOAD>/transcripts.json \\
    --batches audios6 \\
    --force

# Multiple transcripts files (allstar_transcripts.json + per-batch ones)
python ec2/backfill_n_words_asr.py \\
    --data_dir <UPLOAD> \\
    --transcripts <UPLOAD>/transcripts.json,<UPLOAD>/allstar_transcripts.json \\
    --batches audios6

================================================================================
TRANSCRIPTS FILE FORMAT (any one of these will work)
================================================================================
  a) {npy_filename: {"text": str, "words": [...]}}     (ec2/allstar_prep.py)
  b) {npy_filename: str}                                (plain mapping)
  c) {npy_filename: {"transcript": str}}                (alternative key)

================================================================================
SAFETY
================================================================================
* Writes gt.csv.backup before modifying gt.csv. Restore with:
    cp gt.csv.backup gt.csv
* If --transcripts cannot find a row's transcript, the row is skipped and
  logged. Existing value (if any) is left untouched.
* No model retraining is needed after this -- the v2 training scripts will
  just pick up the populated column on next run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_transcripts(paths: list[Path]) -> dict[str, str]:
    """Merge one or more transcripts JSON files into {npy_filename: text}.
    Last-one-wins on duplicate keys."""
    out: dict[str, str] = {}
    for p in paths:
        if not p.exists():
            print(f"WARNING: transcripts file not found: {p}", file=sys.stderr)
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        n_added = 0
        for k, v in data.items():
            text = _extract_text(v)
            if text is None:
                continue
            out[str(k)] = text
            n_added += 1
        print(f"  loaded {n_added}/{len(data)} transcripts from {p}")
    return out


def _extract_text(v) -> str | None:
    """Accept {'text': str, 'words': [...]} | {'transcript': str} | str."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for key in ("text", "transcript", "asr_text"):
            if key in v and isinstance(v[key], str):
                return v[key]
    return None


def _word_count(text: str) -> int:
    """len(text.split()) -- same definition as companylaptop's
    cheating_detection_v3.ipynb feat_n_words_asr."""
    return len(text.split()) if isinstance(text, str) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path,
                    help="folder containing gt.csv")
    ap.add_argument("--transcripts", required=True,
                    help="comma-separated paths to transcripts JSON files")
    ap.add_argument("--batches", default="audios6",
                    help="comma-separated batches to backfill; "
                         "'' = every batch where feat_n_words_asr is NaN")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing non-NaN values too")
    ap.add_argument("--column", default="feat_n_words_asr",
                    help="column name to backfill (default feat_n_words_asr)")
    ap.add_argument("--no_backup", action="store_true",
                    help="skip writing gt.csv.backup")
    args = ap.parse_args()

    gt_path = args.data_dir / "gt.csv"
    if not gt_path.exists():
        print(f"ERROR: {gt_path} not found", file=sys.stderr)
        return 1

    transcript_paths = [Path(p.strip()) for p in args.transcripts.split(",")
                        if p.strip()]
    if not transcript_paths:
        print("ERROR: --transcripts is empty", file=sys.stderr)
        return 1

    target_batches = [b.strip() for b in args.batches.split(",") if b.strip()]

    # ---- load
    print(f"loading gt.csv from {gt_path}")
    gt = pd.read_csv(gt_path)
    gt["batch"] = gt["batch"].astype(str)
    print(f"  gt: {len(gt)} rows, {len(gt.columns)} cols")

    if "npy_filename" not in gt.columns:
        print("ERROR: gt.csv has no npy_filename column", file=sys.stderr)
        return 1

    print(f"loading transcripts from {len(transcript_paths)} file(s)")
    transcripts = _load_transcripts(transcript_paths)
    print(f"  total unique transcripts: {len(transcripts)}")

    # ---- pick rows to update
    if target_batches:
        batch_mask = gt["batch"].isin(target_batches).to_numpy()
        print(f"targeting batches={target_batches}: {batch_mask.sum()} rows")
    else:
        batch_mask = np.ones(len(gt), dtype=bool)
        print(f"targeting all batches: {batch_mask.sum()} rows")

    if args.column not in gt.columns:
        print(f"  column {args.column} is new -- adding it")
        gt[args.column] = np.nan

    if args.force:
        target_mask = batch_mask
        print(f"  --force: will overwrite {target_mask.sum()} rows")
    else:
        existing = pd.to_numeric(gt[args.column], errors="coerce")
        nan_mask = existing.isna().to_numpy()
        target_mask = batch_mask & nan_mask
        print(f"  rows where {args.column} is NaN AND in target batches: "
              f"{target_mask.sum()}")

    if target_mask.sum() == 0:
        print("nothing to do -- exit")
        return 0

    # ---- backfill
    new_values = np.full(len(gt), np.nan, dtype=np.float64)
    n_filled = 0
    n_missing_transcript = 0
    n_empty_text = 0
    missing_examples: list[str] = []

    for i in np.where(target_mask)[0]:
        npy = str(gt.iloc[i]["npy_filename"])
        text = transcripts.get(npy)
        if text is None:
            n_missing_transcript += 1
            if len(missing_examples) < 5:
                missing_examples.append(npy)
            continue
        wc = _word_count(text)
        if wc == 0:
            n_empty_text += 1
        new_values[i] = float(wc)
        n_filled += 1

    print(f"\n  filled         : {n_filled}")
    print(f"  empty text     : {n_empty_text}  (counted as 0; train will see 0)")
    print(f"  missing in JSON: {n_missing_transcript}")
    if missing_examples:
        print(f"    examples (first 5): {missing_examples}")

    # ---- write
    if n_filled == 0:
        print("WARNING: 0 rows filled (no matching transcripts). gt.csv unchanged.")
        return 1

    if not args.no_backup:
        backup = gt_path.with_suffix(".csv.backup")
        shutil.copy2(gt_path, backup)
        print(f"\n  wrote backup -> {backup}")

    # Only update rows where new_values is non-NaN, leave others as-is.
    update_idx = ~np.isnan(new_values)
    gt.loc[update_idx, args.column] = new_values[update_idx]
    gt.to_csv(gt_path, index=False)
    print(f"  wrote gt.csv  -> {gt_path}")

    # Quick sanity print
    after = pd.to_numeric(gt[args.column], errors="coerce")
    for b in sorted(gt["batch"].unique()):
        m = (gt["batch"] == b).to_numpy()
        v = after[m]
        nan_rate = float(v.isna().mean())
        mean_wc = float(v.dropna().mean()) if v.notna().any() else float("nan")
        print(f"    batch={b:12s}  n={m.sum():5d}  "
              f"nan_rate={nan_rate:.3f}  mean_words={mean_wc:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
