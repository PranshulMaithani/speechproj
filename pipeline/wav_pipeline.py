#!/usr/bin/env python3
"""End-to-end WAV -> trained models pipeline (self-contained, PII-preserving).

Start point: raw `.wav` files. End point: trained + evaluated models. This
orchestrates the existing, tested stage scripts in order so a full retrain from
audio is ONE command -- while keeping the PII flow intact (real candidate IDs are
anonymised to encoded group IDs, the mapping dictionary stays local, and only the
anonymised npy + gt.csv move downstream; raw wavs never do).

    STAGE 1  features   companylaptop/extract_features_batch.py   (per batch)
             wav -> rich word-timestamped transcript + the 55 handcrafted feat_*
             (faster-whisper on CPU here; wav_pipeline_gpu.py runs it on GPU)
    STAGE 2  prep       companylaptop/neural_baseline_prep.py
             wav -> ENCODED CID + npy + gt.csv, mapping kept in local/cid_mapping.json
             (auto-discovers every audios<N>/ that has a sibling <batch>GT.csv)
    STAGE 3  embed      ec2/extract_embeddings.py
             npy -> WavLM + Whisper embedding cache (incremental)
    STAGE 4  train      ec2/train_pipeline.py
             cache + gt.csv -> variant x seed models + summaries + threshold books

WHERE TO PUT YOUR DATA (the modular convention -- no code edits to add a batch)
------------------------------------------------------------------------------
Drop, side by side, inside `companylaptop/`:

    companylaptop/audios<N>/<realCID>_<qid>.wav      the raw audio (flat folder)
    companylaptop/audios<N>GT.csv                    columns: filename,label[,region]

Then run this script. `discover_batches()` inside prep (and this orchestrator)
picks up any new `audios<N>/` that has a matching `<batch>GT.csv`, so adding
audios8, audios9, ... needs ZERO code changes -- just the folder + its GT csv.

Single-region batch with no region column? add its default in
`neural_baseline_prep.DEFAULT_REGION_BY_BATCH`. Mixed-region batch? put a
`region` column in its GT csv.

OUTPUTS
-------
    data/neural_prep_out/upload/gt.csv                anonymised labels + feat_*
    data/neural_prep_out/upload/audio_npy/*.npy       anonymised waveforms
    data/neural_prep_out/local/cid_mapping.json       real CID -> encoded (LOCAL ONLY)
    <cache>                                            embeddings_cache_base.npz
    <out_dir>/models/<variant>/seed_<seed>/...         trained model bundles
    <out_dir>/summary/{per_run,summary_mean_std,best_models}.csv + threshold books

Run (full, CPU transcription):
    python pipeline/wav_pipeline.py \
        --batches auto \
        --out_dir data/runs/retrain_from_wav \
        --train_config ec2/configs/train_job.example.yaml

Run only some stages (e.g. re-train without re-transcribing):
    python pipeline/wav_pipeline.py --stages embed,train --out_dir ...

For faster GPU transcription use the sibling script `wav_pipeline_gpu.py`
(identical, but Stage 1 runs faster-whisper on the GPU).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Repo layout: this file lives in <repo>/pipeline/, the stage scripts in
# <repo>/companylaptop and <repo>/ec2. Everything is resolved from the repo root
# so the script works regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
COMPANYLAPTOP = REPO_ROOT / "companylaptop"
EC2 = REPO_ROOT / "ec2"
PREP_OUT = REPO_ROOT / "data" / "neural_prep_out"      # neural_baseline_prep's OUTPUT_ROOT
UPLOAD_DIR = PREP_OUT / "upload"                        # gt.csv + audio_npy/ land here
DEFAULT_CACHE = UPLOAD_DIR / "embeddings_cache_base.npz"
DEFAULT_AUGS = "orig,noise,pitch,speed,gain,air,vtlp,combo"
DEFAULT_LAYERS = "last,9"
ALL_STAGES = ["features", "prep", "embed", "train"]


def discover_batches(audio_root: Path) -> list[str]:
    """Return every `audios<N>/` folder under audio_root that has a sibling
    `<batch>GT.csv`, numerically ordered. Mirrors neural_baseline_prep's own
    discovery so 'drop a folder + its GT csv and re-run' works with no code edit."""
    out = []
    for d in audio_root.glob("audios*"):
        if d.is_dir() and (audio_root / f"{d.name}GT.csv").exists():
            out.append(d.name)
    out.sort(key=lambda b: (int(b[len("audios"):]) if b[len("audios"):].isdigit() else 1_000_000, b))
    return out


def run(cmd: list[str], desc: str, dry_run: bool) -> None:
    """Run one stage as a subprocess from the repo root, streaming its output.
    Raises SystemExit if the stage fails so the pipeline stops loudly rather than
    silently training on stale data. --dry_run prints the command and returns."""
    printable = " ".join(str(c) for c in cmd)
    print(f"\n{'=' * 78}\n[wav_pipeline] {desc}\n$ {printable}\n{'=' * 78}", flush=True)
    if dry_run:
        print("[wav_pipeline] --dry_run: not executing.", flush=True)
        return
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise SystemExit(f"[wav_pipeline] STAGE FAILED ({desc}) with exit code {r.returncode}")


def build_argparser(default_device: str = "cpu",
                    default_compute_type: str = "int8") -> argparse.ArgumentParser:
    """CLI for the pipeline. `default_device`/`default_compute_type` are the only
    knobs the GPU sibling overrides -- everything else is shared."""
    ap = argparse.ArgumentParser(description="WAV -> trained models pipeline")
    # data selection
    ap.add_argument("--audio_root", default=str(COMPANYLAPTOP),
                    help="folder holding audios<N>/ + audios<N>GT.csv "
                         "(default: companylaptop/, the convention prep also uses)")
    ap.add_argument("--batches", default="auto",
                    help="'auto' = every audios<N>/ with a sibling GT csv; or a "
                         "comma list e.g. audios7,audios8 to transcribe just those")
    # stages
    ap.add_argument("--stages", default=",".join(ALL_STAGES),
                    help=f"comma subset of {ALL_STAGES} to run (in this order)")
    ap.add_argument("--force", action="store_true",
                    help="pass --force to the features + embed stages (recompute)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print each stage's command without executing")
    # stage 1 (transcription) -- device/compute_type defaulted per-script
    ap.add_argument("--transcribe_device", default=default_device, choices=["cpu", "cuda"])
    ap.add_argument("--compute_type", default=default_compute_type,
                    help="faster-whisper compute type (cpu: int8; gpu: float16)")
    ap.add_argument("--transcribe_model", default="",
                    help="OVERRIDE the transcription model. Leave EMPTY to keep the "
                         "same model as audios2..6 (changing it diverges the features).")
    # stage 3 (embeddings)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE),
                    help="embeddings_cache .npz to build/extend (base = wavlm-base-plus + whisper-medium)")
    ap.add_argument("--augs", default=DEFAULT_AUGS)
    ap.add_argument("--wavlm_layers", default=DEFAULT_LAYERS)
    ap.add_argument("--wavlm_id", default="")     # empty -> extract_embeddings default (base-plus)
    ap.add_argument("--whisper_id", default="")   # empty -> extract_embeddings default (medium)
    # stage 4 (training)
    ap.add_argument("--data_dir", default=str(UPLOAD_DIR),
                    help="gt.csv + audio_npy/ dir consumed by embed + train "
                         "(default: prep's upload/ output)")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "data" / "runs" / "retrain_from_wav"),
                    help="train_pipeline output dir (models + summaries + threshold books)")
    ap.add_argument("--train_config", default="",
                    help="YAML job file passed to train_pipeline.py (see "
                         "ec2/configs/train_job.example.yaml); CLI below still applies")
    return ap


def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the requested stages in order. Returns a process exit code."""
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; valid: {ALL_STAGES}")
    audio_root = Path(args.audio_root)
    if args.batches.strip().lower() == "auto":
        batches = discover_batches(audio_root)
    else:
        batches = [b.strip() for b in args.batches.split(",") if b.strip()]
    print(f"[wav_pipeline] repo_root = {REPO_ROOT}")
    print(f"[wav_pipeline] audio_root = {audio_root}")
    print(f"[wav_pipeline] batches    = {batches or '(none found)'}")
    print(f"[wav_pipeline] stages     = {stages}")
    print(f"[wav_pipeline] transcription = {args.transcribe_device} / {args.compute_type}")
    print("[wav_pipeline] PII: raw wavs stay put; prep anonymises CID -> encoded id, "
          "keeps local/cid_mapping.json, and only npy + gt.csv move downstream.")

    # ---- STAGE 1: transcription + 55 handcrafted features (per batch) ----
    if "features" in stages:
        if not batches:
            raise SystemExit("no batches found for the features stage "
                             "(need companylaptop/audios<N>/ + audios<N>GT.csv)")
        for b in batches:
            cmd = [sys.executable, str(COMPANYLAPTOP / "extract_features_batch.py"),
                   "--batch", b, "--audio_root", str(audio_root),
                   "--device", args.transcribe_device, "--compute_type", args.compute_type]
            if args.transcribe_model.strip():
                cmd += ["--model", args.transcribe_model.strip()]
            if args.force:
                cmd += ["--force"]
            run(cmd, f"STAGE 1 features/transcribe :: {b}", args.dry_run)

    # ---- STAGE 2: anonymise -> encoded CID + npy + gt.csv (all batches) ----
    if "prep" in stages:
        # neural_baseline_prep.py takes no CLI args -- it auto-discovers batches
        # under companylaptop/ and writes to data/neural_prep_out/. So the
        # --audio_root convention must be companylaptop/ for this stage.
        run([sys.executable, str(COMPANYLAPTOP / "neural_baseline_prep.py")],
            "STAGE 2 prep/anonymise (CID -> encoded id + npy + gt.csv)", args.dry_run)

    # ---- STAGE 3: WavLM + Whisper embeddings (incremental) ----
    if "embed" in stages:
        cmd = [sys.executable, str(EC2 / "extract_embeddings.py"),
               "--data_dir", args.data_dir, "--out_path", args.cache,
               "--augs", args.augs, "--wavlm_layers", args.wavlm_layers]
        if args.wavlm_id.strip():
            cmd += ["--wavlm_id", args.wavlm_id.strip()]
        if args.whisper_id.strip():
            cmd += ["--whisper_id", args.whisper_id.strip()]
        if args.force:
            cmd += ["--force"]
        run(cmd, "STAGE 3 embed (WavLM + Whisper cache)", args.dry_run)

    # ---- STAGE 4: train the variant x seed grid ----
    if "train" in stages:
        cmd = [sys.executable, str(EC2 / "train_pipeline.py"),
               "--data_dir", args.data_dir, "--cache", args.cache,
               "--out_dir", args.out_dir, "--do_extract", "false"]  # embeddings already built
        if args.train_config.strip():
            cmd += ["--config", args.train_config.strip()]
        run(cmd, "STAGE 4 train (variant x seed + threshold books)", args.dry_run)

    print(f"\n[wav_pipeline] DONE. models + summaries under {args.out_dir}")
    print("[wav_pipeline] evaluate any of them with: "
          f"python ec2/evaluate_models.py --models_root {args.out_dir}/models "
          f"--data_dir {args.data_dir} --cache {args.cache} --out_dir <eval_dir>")
    return 0


def main() -> int:
    return run_pipeline(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
