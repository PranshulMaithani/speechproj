#!/usr/bin/env python3
"""LOCAL listen-builder: turn the npy sample package into a wav tree (real ids).

Run this on the listening machine, in (or pointing at) the folder that has BOTH
  - aug_samples.zip      (from ec2/make_sample_package.py: npy + coded ids)
  - cid_mapping.json     (your LOCAL real_cid -> G_NNNNN map; never uploaded)

It unzips the npy, converts each to wav, and de-anonymizes the coded group_id
back to the real candidate id. The wav files are created ONLY here, locally --
they are never uploaded anywhere.

Output tree (default ./samples):
    samples/
      a2sample/<real_id>/<real_id>_q25_orig.wav
                         <real_id>_q25_noise.wav   ... (8 augs)
                         <real_id>_q26_*.wav  <real_id>_q27_*.wav
      a4sample/...  a5sample/...
      a6sample_IND/<real_id>/...   a6sample_PHP/<real_id>/...   (region-split)
    samples/manifest_real.csv   (local index incl. real ids)

So per candidate: 3 questions x 8 augs = 24 wav. Listen across the 8 augs of a
question to judge whether an augmentation is reasonable or destroys the speech.

Deps: numpy, soundfile  (pip install soundfile)

Run:
    python make_listen_wavs.py --zip aug_samples.zip --cid_mapping cid_mapping.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: need soundfile to write wav. Install:  pip install soundfile",
          file=sys.stderr)
    sys.exit(1)


def _batch_tag(batch: str, region: str) -> str:
    """audios2 -> a2sample ; audios6 + region -> a6sample_IND / a6sample_PHP."""
    base = f"a{batch[len('audios'):]}sample" if batch.startswith("audios") else f"{batch}sample"
    if batch == "audios6" and region and region != "NA":
        return f"{base}_{region}"
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path, help="aug_samples.zip")
    ap.add_argument("--cid_mapping", required=True, type=Path,
                    help="cid_mapping.json (real_cid -> G_NNNNN); kept local")
    ap.add_argument("--out_dir", default=Path("samples"), type=Path)
    ap.add_argument("--keep_coded", action="store_true",
                    help="name folders by the coded group_id instead of the real id "
                         "(skip de-anonymization)")
    args = ap.parse_args()

    if not args.zip.exists():
        print(f"ERROR: zip not found: {args.zip}", file=sys.stderr)
        return 1

    # real_cid -> gid  ==> invert to gid -> real_cid
    gid_to_real: dict[str, str] = {}
    if not args.keep_coded:
        if not args.cid_mapping.exists():
            print(f"ERROR: cid_mapping not found: {args.cid_mapping}\n"
                  f"  (pass --keep_coded to skip de-anonymization)", file=sys.stderr)
            return 1
        with args.cid_mapping.open("r", encoding="utf-8") as f:
            real_to_gid = json.load(f)
        gid_to_real = {gid: real for real, gid in real_to_gid.items()}

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    real_rows: list[dict] = []
    n_wav = 0
    n_missing_map = 0

    with zipfile.ZipFile(args.zip, "r") as zf:
        names = set(zf.namelist())
        if "manifest.csv" not in names:
            print("ERROR: zip has no manifest.csv -- is this a make_sample_package.py zip?",
                  file=sys.stderr)
            return 1
        rows = list(csv.DictReader(io.StringIO(zf.read("manifest.csv").decode("utf-8"))))
        for r in rows:
            zip_name = r["npy_in_zip"]
            if zip_name not in names:
                print(f"  WARN: {zip_name} listed in manifest but missing from zip")
                continue
            gid = r["group_id"]
            region = r.get("region", "NA")
            sr = int(r.get("sample_rate") or 16000)
            qid = r["question_id"]
            aug = r["aug"]
            batch = r["batch"]

            if args.keep_coded:
                disp_id = gid
            else:
                disp_id = gid_to_real.get(gid)
                if disp_id is None:
                    disp_id = gid  # fall back to coded if not in the map
                    n_missing_map += 1

            wav = np.load(io.BytesIO(zf.read(zip_name)), allow_pickle=False)
            wav = np.asarray(wav, dtype=np.float32)

            cand_dir = out_dir / _batch_tag(batch, region) / str(disp_id)
            cand_dir.mkdir(parents=True, exist_ok=True)
            wav_path = cand_dir / f"{disp_id}_q{qid}_{aug}.wav"
            sf.write(str(wav_path), wav, sr)
            n_wav += 1
            real_rows.append({
                "wav_path": str(wav_path.relative_to(out_dir)),
                "batch": batch, "real_id": disp_id, "group_id": gid,
                "question_id": qid, "region": region, "aug": aug,
                "label": r.get("label", ""), "sample_rate": sr,
            })

    # local index (includes real ids -> never upload this either)
    idx_path = out_dir / "manifest_real.csv"
    with idx_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(real_rows[0].keys()) if real_rows else
                           ["wav_path", "batch", "real_id", "group_id",
                            "question_id", "region", "aug", "label", "sample_rate"])
        w.writeheader()
        w.writerows(real_rows)

    print(f"wrote {n_wav} wav under {out_dir}/  (index: {idx_path.name})")
    if n_missing_map:
        print(f"  note: {n_missing_map} npy had no real-id match in cid_mapping.json "
              f"(named by coded group_id instead)")
    print("These wav are LOCAL only -- do not upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
