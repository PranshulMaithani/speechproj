#!/usr/bin/env python3
"""Build a transferable sample package of AUGMENTED audio -- as NPY, coded ids.

PII boundary (read this):
  * This writes ONLY .npy waveforms (the same numpy format the embedding
    pipeline already uses) named with ENCODED ids (group_id, e.g. G_00123).
  * NO .wav files and NO real candidate ids are ever written here.
  * Run it wherever gt.csv + audio_npy/ live (EC2 or the company laptop).
  * Transfer the resulting zip + your LOCAL cid_mapping.json to the listening
    machine, then run make_listen_wavs.py THERE to turn npy -> wav and
    de-anonymize. The wav files only ever exist locally.

What it stores: for each selected candidate x question x aug version, one npy
    <batch>__<group_id>__<region>__q<qid>__<aug>.npy
plus a manifest.csv (coded). Augmenters are the EXACT ones from
extract_embeddings.py (build_augmenters), so what you eventually hear matches
what the model was trained on.

Selection (defaults): for each batch, 5 candidates that have ALL of Q25/26/27;
audios6 is split 5 IND + 5 PHP. 8 versions each (orig + 7 augs):
    single-region batch -> 5 x 3 x 8 = 120 npy
    audios6             -> 10 x 3 x 8 = 240 npy

Run:
    python ec2/make_sample_package.py \
        --data_dir /home/ubuntu/nn/data \
        --out_zip  /home/ubuntu/nn/runs/aug_samples.zip
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Exact augmenters used for training (and TARGET_SR). Same dir as this script.
from extract_embeddings import TARGET_SR, build_augmenters

DEFAULT_AUGS = "orig,noise,pitch,speed,gain,air,vtlp,combo"


def _norm_region(r) -> str:
    r = str(r).strip().upper()
    if r in ("NAN", "NONE", ""):
        return "NA"
    if r.startswith("CASUAL"):
        return "CASUAL"
    return r


def _stable_seed(*parts: str) -> int:
    """Process-independent seed from strings (Python's hash() is salted)."""
    h = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def pick_candidates(sub: pd.DataFrame, questions: list[int], per_batch: int,
                    region_split: bool, base_seed: int, batch: str,
                    log: logging.Logger) -> list[tuple[str, str]]:
    """Return [(group_id, region)] selected for this batch."""
    qset = set(questions)
    by_g_q = sub.groupby("group_id")["question_id"].apply(
        lambda s: {int(x) for x in s})
    cand_region = sub.groupby("group_id")["region"].first().map(_norm_region).to_dict() \
        if "region" in sub.columns else {}
    eligible = [g for g, qs in by_g_q.items() if qset.issubset(qs)]
    if not eligible:
        log.warning("[%s] no candidate has all questions %s", batch, sorted(qset))
        return []

    def take(pool: list[str], n: int, seed: int) -> list[str]:
        pool = sorted(pool)
        random.Random(seed).shuffle(pool)
        return pool[:n]

    chosen: list[tuple[str, str]] = []
    if region_split:
        regions = sorted({cand_region.get(g, "NA") for g in eligible})
        for reg in regions:
            pool = [g for g in eligible if cand_region.get(g, "NA") == reg]
            picked = take(pool, per_batch, _stable_seed(str(base_seed), batch, reg))
            if len(picked) < per_batch:
                log.warning("[%s/%s] only %d candidates available (wanted %d)",
                            batch, reg, len(picked), per_batch)
            chosen += [(g, reg) for g in picked]
    else:
        picked = take(eligible, per_batch, _stable_seed(str(base_seed), batch))
        if len(picked) < per_batch:
            log.warning("[%s] only %d candidates available (wanted %d)",
                        batch, len(picked), per_batch)
        chosen = [(g, cand_region.get(g, "NA")) for g in picked]
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with gt.csv + audio_npy/")
    ap.add_argument("--out_zip", required=True)
    ap.add_argument("--batches", default="audios2,audios4,audios5,audios6")
    ap.add_argument("--region_split_batches", default="audios6",
                    help="batches sampled per-region (e.g. audios6 -> 5 IND + 5 PHP)")
    ap.add_argument("--per_batch", type=int, default=5,
                    help="candidates per batch (per region for region-split batches)")
    ap.add_argument("--questions", default="25,26,27")
    ap.add_argument("--augs", default=DEFAULT_AUGS,
                    help="comma-separated; 'orig' is always included")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("pkg")

    data_dir = Path(args.data_dir)
    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    npy_dir = data_dir / "audio_npy"

    batches = [b.strip() for b in args.batches.split(",") if b.strip()]
    region_split = {b.strip() for b in args.region_split_batches.split(",") if b.strip()}
    questions = [int(q) for q in args.questions.split(",") if q.strip()]
    augs = [a.strip() for a in args.augs.split(",") if a.strip()]
    if "orig" not in augs:
        augs = ["orig"] + augs
    # de-dup, keep order
    seen: set[str] = set()
    augs = [a for a in augs if not (a in seen or seen.add(a))]

    gt = pd.read_csv(data_dir / "gt.csv")
    for col in ("group_id", "question_id", "batch", "npy_filename"):
        if col not in gt.columns:
            log.error("gt.csv missing required column '%s'", col)
            return 1
    gt["batch"] = gt["batch"].astype(str)
    gt["group_id"] = gt["group_id"].astype(str)
    if "region" not in gt.columns:
        gt["region"] = "NA"

    aug_callables = build_augmenters([a for a in augs if a != "orig"], None, None, log)

    # ---- select candidates per batch
    selected: list[tuple[str, str, str]] = []  # (batch, group_id, region)
    for batch in batches:
        sub = gt[gt["batch"] == batch]
        if sub.empty:
            log.warning("batch '%s' has no rows in gt.csv -- skipping", batch)
            continue
        cands = pick_candidates(sub, questions, args.per_batch,
                                batch in region_split, args.seed, batch, log)
        for g, reg in cands:
            selected.append((batch, g, reg))
        log.info("[%s] selected %d candidates: %s", batch, len(cands),
                 [g for g, _ in cands])

    if not selected:
        log.error("no candidates selected -- check --batches / --questions / gt.csv")
        return 1

    # ---- build the package
    manifest_buf = io.StringIO()
    writer = csv.writer(manifest_buf)
    writer.writerow(["npy_in_zip", "batch", "group_id", "question_id", "region",
                     "label", "aug", "src_npy_filename", "sample_rate", "n_samples"])

    n_written = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for batch, g, reg in selected:
            sub = gt[(gt["batch"] == batch) & (gt["group_id"] == g)]
            for qid in questions:
                row = sub[sub["question_id"] == qid]
                if row.empty:
                    log.warning("[%s/%s] missing Q%d -- skipping", batch, g, qid)
                    continue
                row = row.iloc[0]
                src_name = str(row["npy_filename"])
                label = row["label"] if "label" in row else ""
                try:
                    wav = np.load(npy_dir / src_name).astype(np.float32, copy=False)
                except Exception as e:
                    log.warning("load failed for %s: %s -- skipping", src_name, e)
                    continue
                for aug in augs:
                    if aug == "orig":
                        wav_a = wav
                    else:
                        # deterministic, order-independent per (file, aug)
                        s = _stable_seed(src_name, aug)
                        np.random.seed(s)
                        random.seed(s)
                        try:
                            wav_a = aug_callables[aug](samples=wav, sample_rate=TARGET_SR)
                        except Exception as e:
                            log.warning("aug %s failed on %s: %s -- using orig",
                                        aug, src_name, e)
                            wav_a = wav
                    wav_a = np.asarray(wav_a, dtype=np.float32)
                    name = f"npy/{batch}__{g}__{reg}__q{qid}__{aug}.npy"
                    buf = io.BytesIO()
                    np.save(buf, wav_a)
                    zf.writestr(name, buf.getvalue())
                    writer.writerow([name, batch, g, qid, reg, label, aug,
                                     src_name, TARGET_SR, len(wav_a)])
                    n_written += 1
        zf.writestr("manifest.csv", manifest_buf.getvalue())

    log.info("wrote %s", out_zip)
    log.info("  candidates=%d  npy files=%d  augs=%s", len(selected), n_written, augs)
    log.info("CONTAINS: npy + coded group_ids only. No wav, no real ids.")
    log.info("Next: copy %s + your LOCAL cid_mapping.json to the listening "
             "machine, then run make_listen_wavs.py there.", out_zip.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
