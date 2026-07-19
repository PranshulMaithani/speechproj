#!/usr/bin/env python3
"""Unified retraining pipeline -- extract (if needed) -> train grid x seeds ->
save every model -> mean/std summary -> per-variant threshold workbooks.

This is the CANONICAL "retrain everything" entry point. It consolidates what
used to be a manual chain (extract_embeddings.py, then
neural_baseline_multiseed.py, then evaluate). neural_baseline_multiseed.py is
kept untouched for the historical record; new retraining should use this file.

WHAT IT DOES (stages, each skippable)
-------------------------------------
  0. EXTRACT (guarded, laptop-only). Prints a per-batch coverage table -- which
     audio is present on disk, which rows are already in the cache, which augs
     / WavLM layers the cache covers. Then, ONLY if audio is present AND
     something the job needs is missing, it shells out to extract_embeddings.py
     (which itself extracts only the missing (file, aug, layer) combos). On EC2
     there is no audio (PII boundary -- audio stays on the laptop), so this
     auto-skips and the existing cache is used. A genuinely missing dataset is
     reported LOUDLY, not silently dropped.
  1. LOAD gt + cache once (seed-independent).
  2. SPLITS per seed -> splits/seed_<seed>.json ledger.
  3. TRAIN every (variant x seed): aug-expand train, fit scaler + PCA, seed init
     with derive_model_seed(seed, variant), train_one, then export the full
     artifact folder + BOTH val and test predictions.
  4. AGGREGATE -> per_run.csv, summary_mean_std.csv (ranked), best_models.csv.
  5. THRESHOLD WORKBOOKS -> one universal xlsx PER VARIANT: a seed_<n> sheet
     (val / test / combined precision-recall-F1 at every threshold) for each of
     the 5 seeds, plus a `robustness` sheet that lines all 5 seeds up with
     cross-seed mean_F1 / min_F1 / std_F1 and the recommended thresholds
     (best-average and most-robust). This is how you pick an operating
     threshold that holds across splits.

CONFIG: a YAML job file (plug-and-play -- copy, tweak, run) whose fields any CLI
flag overrides. See configs/train_job.example.yaml. Everything is configurable:
train / test / auxiliary (train_only) batches, augs, the variant grid (explicit
names OR archs x layers x pca), PCA variance, seeds, min_duration, etc.

ARTIFACT CONTRACT (so evaluate_models.py can pick any model up)
---------------------------------------------------------------
  <out_dir>/models/<variant>/seed_<seed>/
      model.pt  scaler.joblib  pca.joblib(optional)  inference_meta.json
      predictions.csv (TEST)   predictions_val.csv (VAL)   metrics.json
inference_meta.json fully describes the feature order, so the tester needs no
model-specific code.

Run (train-only on EC2; cache already present):
  python ec2/train_pipeline.py --config ec2/configs/train_job.example.yaml \
      --out_dir /home/ubuntu/nn/runs/retrain_20pct

Run (full, on the laptop, will extract any missing embeddings first):
  python ec2/train_pipeline.py --config ec2/configs/train_job.example.yaml \
      --do_extract auto --out_dir D:/.../runs/retrain_20pct
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from _data_pipeline import (assert_no_group_leak, build_splits, compute_metrics,
                            extract_region_metrics, load_cache_reindexed,
                            load_gt_and_filter, per_client_standardize,
                            resolve_use_augs, setup_logging)
from neural_baseline_train import (ARCH_CONFIG, GRAD_CLIP_NORM, LABEL_SMOOTHING,
                                   PCA_VARIANTS, MLP, make_variants, train_one)
from neural_baseline_multiseed import derive_model_seed
# the threshold sweep table (P/R/F1/confusion at every threshold) is already
# implemented and used for the finalised eval -- reuse it verbatim.
from evaluate_final_models import threshold_sweep

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
COMPANYLAPTOP = REPO_ROOT / "companylaptop"        # extract_features_batch + neural_baseline_prep
PREP_UPLOAD = REPO_ROOT / "data" / "neural_prep_out" / "upload"   # prep writes gt.csv + audio_npy here
DEFAULT_OUT = REPO_ROOT / "data" / "runs" / "train"

# Canonical aug set used ONLY for cold-start extraction (no cache yet, use_augs:
# all). Matches extract_embeddings.build_augmenters; extract drops any that the
# audiomentations install can't provide (e.g. codec). Once a cache exists,
# resolve_use_augs reads the real list from it instead.
DEFAULT_AUG_SET = ["noise", "pitch", "speed", "gain", "air", "vtlp", "combo", "codec"]

# ----------------------------------------------------------------------------
# Config: YAML defaults <- overridden by any explicitly-passed CLI flag.
# ----------------------------------------------------------------------------

DEFAULTS = {
    "data_dir": str(PREP_UPLOAD),        # gt.csv + audio_npy/ (prep's output; override for EC2)
    "cache": "", "out_dir": str(DEFAULT_OUT),
    "audio_subdir": "audio_npy",         # where extract reads waveforms
    # --- STAGE I: optional wav ingest (auto-detect audios<N>/ -> transcribe -> npy + gt.csv) ---
    "ingest": "auto",                    # auto (run if new folders/features) | true | false
    "audio_root": "",                    # "" -> companylaptop/ ; holds audios<N>/ + audios<N>GT.csv
    "transcribe_device": "cpu",          # cpu | cuda  (cuda = faster transcription)
    "transcribe_compute_type": "int8",   # faster-whisper compute type (cuda: float16)
    "transcribe_model": "",              # "" keeps the SAME model as audios2..6 (feature parity)
    "retranscribe": False,               # re-transcribe every discovered batch (--force)
    "keep_questions": "25,26,27",         # only these qids -> npy -> embeddings ('all' = every question)
    "prune_questions": False,            # DELETE non-keep wavs to save space (irreversible; opt-in)
    # --- embeddings ---
    "do_extract": "auto",                # auto | true | false
    "wavlm_id": "microsoft/wavlm-base-plus",
    "whisper_id": "openai/whisper-medium",
    "train_batches": ["audios2", "audios4", "audios5", "audios6"],
    "test_batches": [],                  # [] -> 20pct ; e.g. [audios6] -> a6
    "test_region_filter": "",
    "train_only_batches": ["casual"],    # auxiliary; forced into train only
    "use_augs": "all",                   # 'all' | comma list | '' (none)
    "seeds": [42, 43, 44, 45, 46],
    "variants": "all",                   # 'all' | explicit names | 'grid'
    "archs": list(ARCH_CONFIG.keys()),   # used when variants == 'grid'
    "layers": ["last", "9"],             # used when variants == 'grid'
    "pca": [n for n, _ in PCA_VARIANTS], # used when variants == 'grid'
    "min_duration": 30.0,
    "use_text_features": True,
    "per_client_standardize": False,
    "class_balance": "sampler",
    "batch_size": 64, "epochs": 60, "lr": 1e-3,
    "thr_step": 0.01,
}


def _as_list(v):
    """Coerce a config value into a clean list of strings.

    Accepts a Python list/tuple (from YAML) or a comma-separated string (from a
    CLI flag) and returns a whitespace-trimmed, empty-dropped list. Returns None
    for None so callers can tell 'not set' apart from 'set to empty'.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [s.strip() for s in str(v).split(",") if s.strip()]


def load_config() -> dict:
    """Build the resolved job config: DEFAULTS <- YAML (--config) <- CLI flags.

    Precedence (low to high): the DEFAULTS dict, then any keys in the YAML job
    file, then any CLI flag the user explicitly passed (argparse default is None,
    so None means 'not set' and YAML/defaults win). After merging, list fields
    (batches/archs/layers/pca) are normalised via _as_list, seeds to ints, and
    the numeric/boolean fields are cast. Errors out if data_dir/out_dir are
    missing. Returns the fully-typed config dict used by the rest of the run.
    """
    ap = argparse.ArgumentParser(description="Unified retraining pipeline")
    ap.add_argument("--config", default="", help="YAML job file (optional)")
    # every field is also a CLI flag (default None -> 'not set' -> YAML wins)
    for key in DEFAULTS:
        ap.add_argument(f"--{key}", default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        import yaml
        with open(args.config) as f:
            y = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in y.items() if k in DEFAULTS})
    for key in DEFAULTS:                       # CLI overrides YAML/defaults
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v

    # normalise types
    for key in ("train_batches", "test_batches", "train_only_batches",
                "archs", "layers", "pca"):
        cfg[key] = _as_list(cfg[key]) or []
    cfg["seeds"] = [int(s) for s in _as_list(cfg["seeds"])]
    if isinstance(cfg["variants"], str) and cfg["variants"] not in ("all", "grid"):
        cfg["variants"] = _as_list(cfg["variants"])
    for key in ("min_duration", "lr", "thr_step"):
        cfg[key] = float(cfg[key])
    for key in ("batch_size", "epochs"):
        cfg[key] = int(cfg[key])
    for key in ("use_text_features", "per_client_standardize", "retranscribe",
                "prune_questions"):
        cfg[key] = str(cfg[key]).lower() in ("1", "true", "yes")
    cfg["test_region_filter"] = (str(cfg["test_region_filter"]).strip() or None)
    if not cfg["out_dir"]:
        ap.error("--out_dir is required (in YAML or CLI)")
    return cfg


# ----------------------------------------------------------------------------
# Variant grid.
# ----------------------------------------------------------------------------

def build_grid(cfg: dict, log) -> list[tuple[str, str, str, float | None]]:
    """Return [(variant_name, arch, wavlm_layer, pca_val)]. variants:
        'all'      -> the canonical 30 (make_variants)
        'grid'     -> archs x layers x pca cross product from the config
        [names...] -> filter the canonical 30 to those names."""
    pca_map = dict(PCA_VARIANTS)                      # name -> value
    canonical = make_variants()
    v = cfg["variants"]
    if v == "all":
        grid = canonical
    elif v == "grid":
        grid = []
        for arch in cfg["archs"]:
            if arch not in ARCH_CONFIG:
                log.error("unknown arch '%s' (have %s)", arch, list(ARCH_CONFIG)); continue
            for layer in cfg["layers"]:
                tag = layer if layer == "last" else f"l{layer}"
                for pname in cfg["pca"]:
                    if pname not in pca_map:
                        log.error("unknown pca '%s' (have %s)", pname, list(pca_map)); continue
                    grid.append((f"{arch}_{tag}_{pname}", arch, layer, pca_map[pname]))
    else:                                            # explicit names
        want = set(v)
        grid = [g for g in canonical if g[0] in want]
        missing = want - {g[0] for g in canonical}
        if missing:
            log.error("unknown variant name(s): %s. valid: %s",
                      sorted(missing), [g[0] for g in canonical])
    if not grid:
        raise SystemExit("empty variant grid -- check 'variants'/'archs'/'layers'/'pca'")
    log.info("variant grid (%d): %s", len(grid), [g[0] for g in grid])
    return grid


# ----------------------------------------------------------------------------
# Stage 0: coverage report + guarded extraction.
# ----------------------------------------------------------------------------

def coverage_and_extract(cfg: dict, gt_all: pd.DataFrame, cache_path: Path,
                         layers_needed: list[str], use_augs: list[str], log) -> None:
    """Stage 0: print a data/cache coverage report, then extract only if needed.

    For every batch the job touches (train + test + train_only) it logs gt_rows /
    audio-files-on-disk / rows-already-in-cache and a status: OK(cache) when the
    batch is fully cached, `extract` when its audio is present but uncached,
    `partial`, `NO DATA` (no audio and not cached -> flagged loudly), or MISSING
    (no gt rows at all). It also reports which requested augs/WavLM layers and how
    many rows are absent from the cache.

    Extraction policy (respects the PII boundary -- audio is laptop-only):
      * do_extract=false               -> never extract, use the cache as-is.
      * cache already covers everything -> nothing to do.
      * audio dir absent               -> cannot extract; warn (expected on EC2)
                                          unless do_extract=true, which errors.
      * audio present and something is
        missing                        -> shell out to extract_embeddings.py,
                                          which itself only computes the still-
                                          missing (file, aug, layer) cells.
    Mutates nothing; side effect is the (possibly refreshed) cache file on disk.
    """
    data_dir = Path(cfg["data_dir"])
    npy_dir = data_dir / cfg["audio_subdir"]
    do_extract = str(cfg["do_extract"]).lower()
    batches = sorted(set(cfg["train_batches"]) | set(cfg["test_batches"])
                     | set(cfg["train_only_batches"]))
    need_augs = ["orig"] + use_augs

    # cache header (cheap): filenames present + augs/layers covered
    cached_files: set[str] = set()
    cached_augs: list[str] = []
    cached_layers: list[str] = []
    if cache_path.exists():
        c = np.load(cache_path, allow_pickle=True)
        cached_files = set(c["filenames"].astype(str).tolist())
        cached_augs = list(c["aug_names"].astype(str)) if "aug_names" in c else []
        cached_layers = (list(c["wavlm_layers"].astype(str))
                         if "wavlm_layers" in c else ["last"])

    log.info("=" * 78)
    log.info("STAGE 0  DATA / CACHE COVERAGE")
    log.info("  audio dir : %s  (%s)", npy_dir,
             "present" if npy_dir.exists() else "ABSENT -- audio stays on the laptop")
    log.info("  cache     : %s  (%s)", cache_path,
             "present" if cache_path.exists() else "ABSENT -- nothing cached yet")
    log.info("  cache augs: %s", cached_augs or "(none)")
    log.info("  cache lyrs: %s", cached_layers or "(none)")
    log.info("  %-10s %8s %8s %8s %10s   %s", "batch", "gt_rows", "audio", "cached", "status", "")
    fn_all = gt_all["npy_filename"].astype(str)
    any_missing_data = False
    for b in batches:
        rows = gt_all[gt_all["batch"].astype(str) == b]
        if len(rows) == 0:
            log.warning("  %-10s %8s %8s %8s %10s   <-- NO ROWS in gt.csv for this batch!",
                        b, 0, "-", "-", "MISSING")
            any_missing_data = True
            continue
        fns = rows["npy_filename"].astype(str).tolist()
        n_audio = sum(1 for fn in fns if (npy_dir / fn).exists()) if npy_dir.exists() else 0
        n_cached = sum(1 for fn in fns if fn in cached_files)
        if n_cached == len(fns):
            status = "OK(cache)"
        elif n_audio == len(fns):
            status = "extract"
        elif n_audio > 0:
            status = "partial"
        else:
            status = "NO DATA"
            any_missing_data = True
        log.info("  %-10s %8d %8d %8d %10s", b, len(fns), n_audio, n_cached, status)

    miss_augs = [a for a in need_augs if a not in cached_augs] if cached_augs else need_augs
    miss_layers = [l for l in layers_needed if l not in cached_layers] if cached_layers else layers_needed
    uncached_files = int((~fn_all.isin(cached_files)).sum()) if cached_files else len(fn_all)
    log.info("  needed augs missing from cache  : %s", miss_augs or "(none)")
    log.info("  needed layers missing from cache: %s", miss_layers or "(none)")
    log.info("  gt rows not yet in cache        : %d / %d", uncached_files, len(fn_all))

    if do_extract == "false":
        log.info("  do_extract=false -> skipping extraction (using cache as-is).")
        return

    needs_work = bool(miss_augs or miss_layers or uncached_files)
    if not needs_work:
        log.info("  cache already covers every needed file/aug/layer -> no extraction.")
        return

    if not npy_dir.exists():
        msg = ("audio dir absent -> CANNOT extract. On EC2 this is expected "
               "(audio is laptop-only); training will use whatever the cache has. "
               "If rows above show NO DATA, those will fail to load in stage 1.")
        if do_extract == "true":
            log.error("  do_extract=true but %s", msg)
            raise SystemExit("extraction forced but no audio present")
        log.warning("  %s", msg)
        return

    # audio present and something is missing -> let extract_embeddings.py do the
    # incremental work (it only computes the still-zero (file, aug, layer) cells).
    cmd = [sys.executable, str(THIS_DIR / "extract_embeddings.py"),
           "--data_dir", str(data_dir), "--out_path", str(cache_path),
           "--augs", ",".join(need_augs), "--wavlm_layers", ",".join(layers_needed),
           "--wavlm_id", cfg["wavlm_id"], "--whisper_id", cfg["whisper_id"]]
    log.info("  -> running extraction (incremental): %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    log.info("  extraction step complete.")


# ----------------------------------------------------------------------------
# Feature assembly (per layer) + best-state prediction.
# ----------------------------------------------------------------------------

def _concat(wavlm, whisper, feats):
    """Concatenate the three feature blocks into the model's input matrix:
    [ WavLM mean-pool | Whisper-encoder mean-pool | feat_* ] along axis 1.
    feats may be None (WavLM+Whisper only). Returns float32."""
    parts = [wavlm, whisper]
    if feats is not None:
        parts.append(feats)
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_per_layer(splits, wavlm_cache, whisper_cache, feat_arr, y_full,
                    use_augs, layers, log) -> dict:
    """Per layer: aug-expanded TRAIN (orig + each aug), val/test (orig only), and
    a StandardScaler fit on the expanded train. Same recipe as the finalised
    pipeline; rebuilt per seed because the split (hence the scaler) changes."""
    tr, va, te = splits["train"], splits["val"], splits["test"]
    out: dict[str, dict] = {}
    for layer in layers:
        from sklearn.preprocessing import StandardScaler
        X_orig = _concat(wavlm_cache[(layer, "orig")], whisper_cache["orig"], feat_arr)
        Xtr_blocks, ytr_blocks = [X_orig[tr]], [y_full[tr]]
        for a in use_augs:
            Xtr_blocks.append(_concat(wavlm_cache[(layer, a)], whisper_cache[a], feat_arr)[tr])
            ytr_blocks.append(y_full[tr])
        X_train = np.concatenate(Xtr_blocks, axis=0).astype(np.float32)
        y_train = np.concatenate(ytr_blocks, axis=0).astype(np.int64)
        scaler = StandardScaler().fit(X_train)
        out[layer] = {
            "X_train_s": scaler.transform(X_train).astype(np.float32), "y_train": y_train,
            "X_val_s": scaler.transform(X_orig[va]).astype(np.float32),
            "X_test_s": scaler.transform(X_orig[te]).astype(np.float32),
            "scaler": scaler,
        }
    return out


@torch.no_grad()
def predict_state(state: dict, X: np.ndarray, in_dim: int, hidden, dropout,
                  device) -> np.ndarray:
    """Rebuild the MLP from a saved state_dict and return sigmoid probabilities
    for X (batched at 512). Used to score val and test from the SAME best-epoch
    weights train_one exported, so predictions_val.csv / predictions.csv are
    exactly what the saved model.pt produces (BatchNorm in eval mode)."""
    model = MLP(in_dim, hidden=tuple(hidden), dropout=dropout).to(device)
    model.load_state_dict(state)
    model.eval()
    out = []
    for i in range(0, len(X), 512):
        x = torch.from_numpy(X[i:i + 512].astype(np.float32)).to(device)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out) if len(X) else np.array([], dtype=np.float32)


def _pred_df(gt: pd.DataFrame, idx: np.ndarray, p: np.ndarray, split: str) -> pd.DataFrame:
    """Assemble a per-row predictions frame for the rows `idx`: the identifying
    columns (npy_filename/group_id/question_id/batch/region, whichever exist), a
    `split` tag ('val' or 'test'), the integer label, and the model's pred_score.
    This is what predictions.csv / predictions_val.csv hold and what the
    threshold workbook (and evaluate_models.py) read back."""
    keep = [c for c in ("npy_filename", "group_id", "question_id", "batch", "region")
            if c in gt.columns]
    df = gt.iloc[idx][keep].copy()
    df.insert(0, "split", split)
    df["label"] = gt.iloc[idx]["label"].to_numpy().astype(int)
    df["pred_score"] = np.round(p, 6)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Stage 5: per-variant universal threshold workbook.
# ----------------------------------------------------------------------------

def build_threshold_workbook(var_dir: Path, variant: str, seeds: list[int],
                             thr_step: float, log) -> dict:
    """One xlsx per variant: a seed_<n> sheet (val/test/combined P-R-F1 at each
    threshold) per seed + a `robustness` sheet lining the seeds up with cross-seed
    mean/min/std F1 and recommended thresholds. Returns a rec dict for best_models."""
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, thr_step), 4)
    robo = pd.DataFrame({"threshold": grid})
    per_seed_sheets: dict[str, pd.DataFrame] = {}
    best_rows = []
    for seed in seeds:
        sd = var_dir / f"seed_{seed}"
        ptest = sd / "predictions.csv"
        pval = sd / "predictions_val.csv"
        if not ptest.exists():
            continue
        te = pd.read_csv(ptest)
        va = pd.read_csv(pval) if pval.exists() else te.iloc[0:0]
        comb = pd.concat([va, te], ignore_index=True)
        sw_v = threshold_sweep(va["label"].to_numpy(), va["pred_score"].to_numpy(), thr_step) if len(va) else None
        sw_t = threshold_sweep(te["label"].to_numpy(), te["pred_score"].to_numpy(), thr_step)
        sw_c = threshold_sweep(comb["label"].to_numpy(), comb["pred_score"].to_numpy(), thr_step)
        sheet = pd.DataFrame({"threshold": sw_c["threshold"]})
        for tag, sw in (("val", sw_v), ("test", sw_t), ("comb", sw_c)):
            if sw is None:
                continue
            for col in ("precision", "recall", "f1"):
                sheet[f"{tag}_{col}"] = sw[col].to_numpy()
        per_seed_sheets[f"seed_{seed}"] = sheet
        robo[f"s{seed}_comb_f1"] = sw_c["f1"].to_numpy()
        # this seed's own best thresholds (where you'd operate per split)
        best_rows.append({
            "seed": seed,
            "best_thr_val": float(sw_v.loc[sw_v["f1"].idxmax(), "threshold"]) if sw_v is not None else float("nan"),
            "best_thr_test": float(sw_t.loc[sw_t["f1"].idxmax(), "threshold"]),
            "best_thr_comb": float(sw_c.loc[sw_c["f1"].idxmax(), "threshold"]),
            "best_f1_comb": float(sw_c["f1"].max()),
        })
    if not per_seed_sheets:
        log.warning("[%s] no per-seed predictions found -> no threshold workbook", variant)
        return {}

    f1_cols = [c for c in robo.columns if c.endswith("_comb_f1")]
    robo["mean_f1"] = robo[f1_cols].mean(axis=1).round(4)
    robo["min_f1"] = robo[f1_cols].min(axis=1).round(4)
    robo["std_f1"] = robo[f1_cols].std(axis=1, ddof=0).round(4)
    thr_best_avg = float(robo.loc[robo["mean_f1"].idxmax(), "threshold"])
    thr_robust = float(robo.loc[robo["min_f1"].idxmax(), "threshold"])
    rec = {
        "variant": variant, "n_seeds": len(per_seed_sheets),
        "thr_best_avg": thr_best_avg,
        "mean_f1_at_best_avg": float(robo["mean_f1"].max()),
        "thr_robust": thr_robust,
        "min_f1_at_robust": float(robo["min_f1"].max()),
    }
    summary = pd.DataFrame(best_rows)
    rec_df = pd.DataFrame([{
        "metric": "recommended (max mean F1)", "threshold": thr_best_avg,
        "value": rec["mean_f1_at_best_avg"]},
        {"metric": "recommended (max min F1 = most robust)", "threshold": thr_robust,
         "value": rec["min_f1_at_robust"]}])

    xlsx = var_dir / "threshold_sweep.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            robo.round(4).to_excel(xw, sheet_name="robustness", index=False)
            rec_df.to_excel(xw, sheet_name="recommended", index=False)
            summary.round(4).to_excel(xw, sheet_name="per_seed_best", index=False)
            for name, sdf in per_seed_sheets.items():
                sdf.round(4).to_excel(xw, sheet_name=name[:31], index=False)
        log.info("[%s] threshold workbook -> %s  (best-avg thr=%.2f, robust thr=%.2f)",
                 variant, xlsx, thr_best_avg, thr_robust)
    except Exception as e:
        log.warning("[%s] could not write threshold xlsx (%s); install openpyxl", variant, e)
    return rec


# ----------------------------------------------------------------------------
# Stage I: optional wav ingest (auto-detect folders -> transcribe -> npy + gt).
# ----------------------------------------------------------------------------

def discover_batches(audio_root: Path) -> list[str]:
    """Every `audios<N>/` under audio_root that has a sibling `<batch>GT.csv`,
    numerically ordered. Mirrors neural_baseline_prep's own discovery so 'drop a
    folder + its GT csv and re-run' works with zero code edits."""
    out = []
    for d in sorted(audio_root.glob("audios*")):
        if d.is_dir() and (audio_root / f"{d.name}GT.csv").exists():
            out.append(d.name)
    out.sort(key=lambda b: (int(b[6:]) if b[6:].isdigit() else 10**9, b))
    return out


def _repo_run(cmd: list[str], desc: str, log, env: dict | None = None) -> None:
    """Run one ingest sub-stage as a subprocess from the repo root, and stop the
    whole pipeline loudly if it fails (so we never train on stale/partial data).
    `env` (if given) fully replaces the child environment -- pass {**os.environ, ...}."""
    log.info("-" * 70)
    log.info("[ingest] %s", desc)
    log.info("[ingest] $ %s", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if r.returncode != 0:
        raise SystemExit(f"[ingest] sub-stage FAILED ({desc}) exit code {r.returncode}")


def ingest_stage(cfg: dict, log) -> bool:
    """STAGE I -- turn raw wavs into the anonymised gt.csv + npy the rest of the
    pipeline needs, only when there is something new to do.

    Auto-detects `audios<N>/` folders (each with a `<batch>GT.csv`) under
    `audio_root` (default companylaptop/), transcribes any batch missing its
    `<batch>_features.csv` (faster-whisper, CPU or GPU per `transcribe_device`),
    then runs neural_baseline_prep to anonymise CID -> encoded id, write
    `<PREP_UPLOAD>/{gt.csv, audio_npy/}` and keep `local/cid_mapping.json`.

    ingest = false -> never run (train from existing gt/cache).
    ingest = true  -> always run prep (transcribe only the batches missing features).
    ingest = auto  -> run only if a discovered batch is new (not in gt.csv) or is
                      missing its features csv; otherwise skip.

    PII: raw wavs never move downstream; only npy + gt.csv (anonymised) do, and the
    real-id mapping stays local. Returns True if it (re)built gt/npy under
    PREP_UPLOAD (so main() reads from there)."""
    mode = str(cfg["ingest"]).lower()
    if mode == "false":
        log.info("[ingest] ingest=false -> training from existing gt/cache")
        return False
    audio_root = Path(cfg["audio_root"]) if str(cfg["audio_root"]).strip() else COMPANYLAPTOP
    discovered = discover_batches(audio_root)
    if not discovered:
        log.info("[ingest] no audios<N>/ + <batch>GT.csv under %s -> nothing to ingest "
                 "(training from existing gt/cache)", audio_root)
        return False

    gt_path = PREP_UPLOAD / "gt.csv"
    existing: set[str] = set()
    if gt_path.exists():
        try:
            existing = set(pd.read_csv(gt_path, usecols=["batch"])["batch"].astype(str))
        except Exception:
            existing = set()
    need_feats = [b for b in discovered if not (audio_root / f"{b}_features.csv").exists()]
    new_batches = [b for b in discovered if b not in existing]
    retr = bool(cfg["retranscribe"])
    log.info("=" * 78)
    log.info("STAGE I  INGEST (auto-detect -> transcribe -> anonymise -> npy + gt)")
    log.info("[ingest] audio_root=%s  discovered=%s", audio_root, discovered)
    log.info("[ingest] new(not in gt)=%s  missing_features=%s  retranscribe=%s",
             new_batches, need_feats, retr)

    if mode == "auto" and not new_batches and not need_feats and not retr:
        log.info("[ingest] auto: nothing new -> skip transcription/prep, use existing gt")
        return False

    import os
    keep = str(cfg["keep_questions"]).strip() or "25,26,27"
    prune = bool(cfg["prune_questions"])
    # pass the question filter down so ONLY these qids get transcribed + npy'd ->
    # embeddings (audios<N>/<ciid>/<ciid>_<q>.wav is nested; both sub-scripts rglob).
    sub_env = {**os.environ, "KEEP_QUESTIONS": keep}
    log.info("[ingest] PII: raw wavs stay put; CID -> encoded id + local cid_mapping.json; "
             "only npy + gt.csv move downstream.")
    log.info("[ingest] keep_questions=%s  prune_questions=%s", keep, prune)
    to_transcribe = discovered if retr else need_feats

    # optional: DELETE non-keep wavs first (irreversible; only when prune_questions:true)
    if prune:
        for b in to_transcribe:
            _repo_run([sys.executable, str(COMPANYLAPTOP / "prune_questions.py"),
                       "--batch", b, "--audio_root", str(audio_root),
                       "--keep", keep, "--apply"],
                      f"PRUNE non-keep questions (delete) :: {b}", log)

    for b in to_transcribe:
        cmd = [sys.executable, str(COMPANYLAPTOP / "extract_features_batch.py"),
               "--batch", b, "--audio_root", str(audio_root),
               "--device", str(cfg["transcribe_device"]),
               "--compute_type", str(cfg["transcribe_compute_type"])]
        if str(cfg["transcribe_model"]).strip():
            cmd += ["--model", str(cfg["transcribe_model"]).strip()]
        if retr:
            cmd += ["--force"]
        _repo_run(cmd, f"transcribe + 55 features :: {b}  ({cfg['transcribe_device']})",
                  log, env=sub_env)
    # prep takes no CLI args -- it auto-discovers batches under companylaptop/ and
    # writes the anonymised gt.csv + npy to PREP_UPLOAD (KEEP_QUESTIONS via env).
    _repo_run([sys.executable, str(COMPANYLAPTOP / "neural_baseline_prep.py")],
              "anonymise -> encoded id + npy + gt.csv (only kept questions)", log, env=sub_env)
    return True


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main() -> int:
    """Drive the pipeline end to end: resolve config; STAGE I optionally ingests raw
    wavs (auto-detect folders -> transcribe -> anonymise -> npy + gt.csv); then the
    coverage/extract pre-flight (stage 0), load gt+cache (1), per-seed splits (2),
    train every variant x seed exporting full artifacts + val/test predictions (3),
    aggregate into per_run/summary/best_models (4), and per-variant threshold
    workbooks (5). Returns a process exit code (0 on success)."""
    cfg = load_config()
    out_dir = Path(cfg["out_dir"])
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)

    log = setup_logging(out_dir / "log_train.txt", name="pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "a6" if cfg["test_batches"] else "20pct"
    log.info("UNIFIED TRAIN PIPELINE   device=%s  mode=%s", device, mode)
    with (out_dir / "config_resolved.json").open("w") as f:
        json.dump(cfg, f, indent=2, default=str)
    log.info("resolved config -> %s", out_dir / "config_resolved.json")

    # ---- STAGE I: optional wav ingest (auto-detect + transcribe + anonymise) ----
    ran_ingest = ingest_stage(cfg, log)
    data_dir = PREP_UPLOAD if ran_ingest else Path(cfg["data_dir"])
    cfg["data_dir"] = str(data_dir)          # keep cfg consistent (Stage 0 reads it)
    cache_path = Path(cfg["cache"]) if cfg["cache"] else (data_dir / "embeddings_cache.npz")
    log.info("data_dir = %s", data_dir)
    log.info("cache    = %s", cache_path)

    # variant grid + the WavLM layers it needs
    grid = build_grid(cfg, log)
    layers_needed = sorted({g[2] for g in grid}, key=lambda l: (l != "last", l))
    if "last" not in layers_needed:
        layers_needed = ["last"] + layers_needed

    # raw gt (pre-min_duration) for the coverage table
    gt_raw = pd.read_csv(data_dir / "gt.csv")
    gt_raw["batch"] = gt_raw["batch"].astype(str)
    # what augs Stage 0 should make sure the cache has. If a cache exists,
    # resolve 'all' against it; on a cold start, 'all' means the canonical set
    # (extract trims to what's installed). An explicit list is used as-is.
    if cache_path.exists():
        use_augs_planned = resolve_use_augs(cfg["use_augs"], cache_path, log)
    elif str(cfg["use_augs"]).strip().lower() == "all":
        use_augs_planned = list(DEFAULT_AUG_SET)
    else:
        use_augs_planned = _as_list(cfg["use_augs"]) or []

    # ---- STAGE 0 ----
    coverage_and_extract(cfg, gt_raw, cache_path, layers_needed, use_augs_planned, log)

    # ---- STAGE 1: load gt + cache ----
    log.info("=" * 78); log.info("STAGE 1  LOAD")
    gt = load_gt_and_filter(data_dir, cfg["min_duration"], log)
    if len(gt) == 0:
        log.error("all rows filtered out"); return 1
    use_augs = resolve_use_augs(cfg["use_augs"], cache_path, log)
    log.info("use_augs (%d): %s", len(use_augs), use_augs)
    filenames_cur = gt["npy_filename"].to_numpy().astype(str)
    wavlm_cache, whisper_cache = load_cache_reindexed(
        cache_path, filenames_cur, ["orig"] + use_augs, layers_needed, log)
    feat_cols = [c for c in gt.columns if c.startswith("feat_")]
    if feat_cols and cfg["use_text_features"]:
        feat_arr = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
        log.info("handcrafted feats: %d cols", len(feat_cols))
    else:
        feat_arr = None
    if cfg["per_client_standardize"]:
        feat_arr = per_client_standardize(wavlm_cache, whisper_cache, gt, log, feat_arr=feat_arr)
    y_full = gt["label"].to_numpy().astype(np.int64)
    gids = gt["group_id"].astype(str).to_numpy() if "group_id" in gt.columns else filenames_cur
    wavlm_dim = int(wavlm_cache[(layers_needed[0], "orig")].shape[1])
    whisper_dim = int(whisper_cache["orig"].shape[1])

    # ---- STAGE 2: splits per seed ----
    log.info("=" * 78); log.info("STAGE 2  SPLITS (mode=%s)", mode)
    splits_by_seed, manifest_rows = {}, []
    for seed in cfg["seeds"]:
        sp = build_splits(gt, train_batches=cfg["train_batches"],
                          test_batches=cfg["test_batches"] or None,
                          test_region_filter=cfg["test_region_filter"],
                          train_only_batches=cfg["train_only_batches"],
                          seed=seed, log=log, fewshot_frac=0.0)
        assert_no_group_leak(gt, sp)
        splits_by_seed[seed] = sp
        tr, va, te = sp["train"], sp["val"], sp["test"]
        led = {"seed": seed, "mode": mode,
               "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
               "train_candidates": sorted(set(gids[tr].tolist())),
               "val_candidates": sorted(set(gids[va].tolist())),
               "test_candidates": sorted(set(gids[te].tolist()))}
        with (out_dir / "splits" / f"seed_{seed}.json").open("w") as f:
            json.dump(led, f, indent=2)
        manifest_rows.append({k: led[k] for k in ("seed", "mode", "n_train", "n_val", "n_test")})

    # ---- STAGE 3: train grid x seeds ----
    log.info("=" * 78)
    log.info("STAGE 3  TRAIN  (%d variants x %d seeds = %d models)",
             len(grid), len(cfg["seeds"]), len(grid) * len(cfg["seeds"]))
    per_run_rows = []
    for seed in cfg["seeds"]:
        sp = splits_by_seed[seed]
        tr, va, te = sp["train"], sp["val"], sp["test"]
        per_layer = build_per_layer(sp, wavlm_cache, whisper_cache, feat_arr,
                                    y_full, use_augs, layers_needed, log)
        for variant, arch_name, layer, pca_var in grid:
            arch = ARCH_CONFIG[arch_name]
            lp = per_layer[layer]
            Xtr_s, ytr = lp["X_train_s"], lp["y_train"]
            Xva_s, Xte_s = lp["X_val_s"], lp["X_test_s"]
            ms = derive_model_seed(seed, variant)
            if pca_var is None:
                Xt, Xv, Xs, pca = Xtr_s, Xva_s, Xte_s, None
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=pca_var, svd_solver="full", random_state=ms).fit(Xtr_s)
                Xt, Xv, Xs = (pca.transform(Xtr_s).astype(np.float32),
                              pca.transform(Xva_s).astype(np.float32),
                              pca.transform(Xte_s).astype(np.float32))
            in_dim = Xt.shape[1]
            torch.manual_seed(ms); torch.cuda.manual_seed_all(ms); np.random.seed(ms)
            result, _ = train_one(
                Xt, ytr, Xv, y_full[va], Xs, y_full[te], in_dim=in_dim,
                batch_size=cfg["batch_size"], epochs=cfg["epochs"], lr=cfg["lr"],
                wd=arch["weight_decay"], device=device, log=log,
                tag=f"s{seed}/{variant}", hidden=tuple(arch["hidden"]),
                dropout=arch["dropout"], class_balance=cfg["class_balance"],
                export_model=True)
            m = result.test_metrics

            # predictions for BOTH val and test from the best-epoch weights
            sd = out_dir / "models" / variant / f"seed_{seed}"
            sd.mkdir(parents=True, exist_ok=True)
            val_p = predict_state(result.model_state, Xv, in_dim, arch["hidden"], arch["dropout"], device)
            test_p = predict_state(result.model_state, Xs, in_dim, arch["hidden"], arch["dropout"], device)
            _pred_df(gt, te, test_p, "test").to_csv(sd / "predictions.csv", index=False)
            _pred_df(gt, va, val_p, "val").to_csv(sd / "predictions_val.csv", index=False)

            torch.save(result.model_state, sd / "model.pt")
            joblib.dump(lp["scaler"], sd / "scaler.joblib")
            if pca is not None:
                joblib.dump(pca, sd / "pca.joblib")
            with (sd / "metrics.json").open("w") as f:
                json.dump({"variant": variant, "seed": seed, "model_seed": ms,
                           "mode": mode, "in_dim": in_dim, "augs_used": use_augs,
                           "best_val_f1": result.best_val_f1,
                           "best_epoch": result.best_epoch, "test": m}, f, indent=2)
            with (sd / "inference_meta.json").open("w") as f:
                json.dump({
                    "variant": variant, "arch": arch_name, "wavlm_layer": layer,
                    "hidden": list(arch["hidden"]), "dropout": arch["dropout"],
                    "weight_decay": arch["weight_decay"], "pca": pca_var, "in_dim": in_dim,
                    "use_augs": use_augs, "use_text_features": cfg["use_text_features"],
                    "feat_cols": feat_cols if (feat_cols and cfg["use_text_features"]) else [],
                    "class_balance": cfg["class_balance"], "label_smoothing": LABEL_SMOOTHING,
                    "grad_clip": GRAD_CLIP_NORM, "seed": seed, "model_seed": ms, "mode": mode,
                    "best_epoch": result.best_epoch, "best_val_f1": result.best_val_f1,
                    "best_f1_threshold": m["best_f1"]["threshold"],
                    "wavlm_dim": wavlm_dim, "whisper_dim": whisper_dim, "target_sr": 16000,
                    "feature_order": (f"concat[ wavlm[{layer}]({wavlm_dim}) | "
                                      f"whisper({whisper_dim}) | feat_*"
                                      f"({len(feat_cols) if feat_cols and cfg['use_text_features'] else 0}) ]"
                                      f" -> scaler -> {'pca -> ' if pca is not None else ''}MLP -> sigmoid"),
                }, f, indent=2)

            rap = m.get("recall_at_precision", {})
            ind, php = extract_region_metrics(m, "IND"), extract_region_metrics(m, "PHP")
            per_run_rows.append({
                "variant": variant, "arch": arch_name, "wavlm_layer": layer,
                "pca": "full" if pca_var is None else f"{int(pca_var*100)}",
                "seed": seed, "model_seed": ms, "mode": mode, "in_dim": in_dim,
                "n_augs": len(use_augs), "val_f1": round(result.best_val_f1, 4),
                "best_epoch": result.best_epoch, "test_auc": round(m["auc"], 4),
                "test_ap": round(m["ap"], 4), "test_f1@0.5": round(m["thr0.5"]["f1"], 4),
                "test_best_f1": round(m["best_f1"]["f1"], 4),
                "test_best_thr": round(m["best_f1"]["threshold"], 3),
                "recall@p80": round(rap.get("p80", {}).get("recall", float("nan")), 4),
                "recall@p90": round(rap.get("p90", {}).get("recall", float("nan")), 4),
                "recall@p95": round(rap.get("p95", {}).get("recall", float("nan")), 4),
                "ind_r@p90": round(ind["p90"], 4), "php_r@p90": round(php["p90"], 4),
            })
            log.info("[s%d/%s] best_f1=%.4f auc=%.3f r@p90=%.4f -> %s",
                     seed, variant, m["best_f1"]["f1"], m["auc"],
                     rap.get("p90", {}).get("recall", float("nan")), sd)

    # ---- STAGE 4: aggregate ----
    log.info("=" * 78); log.info("STAGE 4  AGGREGATE")
    per_run = pd.DataFrame(per_run_rows)
    per_run.to_csv(out_dir / "summary" / "per_run.csv", index=False)
    metric_cols = ["test_best_f1", "recall@p80", "recall@p90", "recall@p95",
                   "test_auc", "test_ap", "val_f1", "ind_r@p90", "php_r@p90"]
    agg_rows = []
    for (variant, arch_name, layer, pca), g in per_run.groupby(
            ["variant", "arch", "wavlm_layer", "pca"], sort=False):
        row = {"variant": variant, "arch": arch_name, "wavlm_layer": layer, "pca": pca,
               "mode": mode, "n_seeds": int(len(g)),
               "seeds": ",".join(map(str, sorted(g["seed"]))),
               "in_dim": int(g["in_dim"].iloc[0]), "n_augs": int(g["n_augs"].iloc[0]),
               "best_f1_seed": int(g.loc[g["test_best_f1"].idxmax(), "seed"])}
        for c in metric_cols:
            row[f"avg_{c}"] = round(float(g[c].mean()), 4)
            row[f"std_{c}"] = round(float(g[c].std(ddof=0)), 4)
            row[f"best_{c}"] = round(float(g[c].max()), 4)
        agg_rows.append(row)
    summary = (pd.DataFrame(agg_rows)
               .sort_values("avg_test_best_f1", ascending=False).reset_index(drop=True))
    summary.to_csv(out_dir / "summary" / "summary_mean_std.csv", index=False)

    # ---- STAGE 5: per-variant threshold workbooks + best_models ----
    log.info("=" * 78); log.info("STAGE 5  THRESHOLD WORKBOOKS")
    recs = {}
    for variant in summary["variant"]:
        rec = build_threshold_workbook(out_dir / "models" / variant, variant,
                                       cfg["seeds"], cfg["thr_step"], log)
        if rec:
            recs[variant] = rec
    best = summary.copy()
    best["thr_robust"] = best["variant"].map(lambda v: recs.get(v, {}).get("thr_robust", float("nan")))
    best["min_f1_at_robust"] = best["variant"].map(lambda v: recs.get(v, {}).get("min_f1_at_robust", float("nan")))
    best["thr_best_avg"] = best["variant"].map(lambda v: recs.get(v, {}).get("thr_best_avg", float("nan")))
    best["recommended"] = ""
    if len(best):
        best.loc[0, "recommended"] = "TOP by avg_test_best_f1"
    best_cols = ["variant", "arch", "wavlm_layer", "pca", "n_seeds",
                 "avg_test_best_f1", "std_test_best_f1", "best_test_best_f1",
                 "avg_recall@p90", "avg_test_auc", "thr_robust", "min_f1_at_robust",
                 "thr_best_avg", "recommended"]
    best[[c for c in best_cols if c in best.columns]].to_csv(
        out_dir / "summary" / "best_models.csv", index=False)

    manifest = pd.DataFrame(manifest_rows)
    try:
        with pd.ExcelWriter(out_dir / "summary" / "training_summary.xlsx", engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="mean_std", index=False)
            best[[c for c in best_cols if c in best.columns]].to_excel(xw, sheet_name="best_models", index=False)
            per_run.to_excel(xw, sheet_name="per_run", index=False)
            manifest.to_excel(xw, sheet_name="split_manifest", index=False)
        log.info("wrote %s", out_dir / "summary" / "training_summary.xlsx")
    except Exception as e:
        log.warning("could not write training_summary.xlsx (%s); CSVs saved", e)

    log.info("=" * 78)
    log.info("DONE. models -> %s/models/<variant>/seed_<seed>/", out_dir)
    log.info("Top variants by avg test best-F1:\n%s",
             summary[["variant", "avg_test_best_f1", "std_test_best_f1",
                      "avg_recall@p90"]].head(8).to_string(index=False))
    log.info("Pick a model dir and score it on any data with: "
             "python ec2/evaluate_models.py --models_root %s/models --data_dir ... "
             "--test_batch <batch>", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
