#!/usr/bin/env python3
"""Generic model tester -- point it at a folder of saved models, score them all.

Companion to train_pipeline.py. train_pipeline.py writes one self-describing
artifact folder per (variant, seed):

    <variant>/seed_<seed>/
        model.pt  scaler.joblib  pca.joblib(optional)  inference_meta.json
        predictions.csv (TEST)   predictions_val.csv (VAL)   metrics.json

This script DISCOVERS every such folder under --models_root, reloads the weights,
rebuilds the ORIG features for the rows you want to test on (using each model's
OWN stored feature order from inference_meta.json), predicts, and produces a full
metric + threshold sweep. No model-specific code -- if a folder has model.pt +
inference_meta.json, it can be tested.

Two evaluation modes
--------------------
  (A) default            : score each model on ITS OWN test set (the run's
                           predictions.csv) and verify the reloaded weights
                           reproduce the saved pred_score (~0 diff).
  (B) --test_batch X[,Y] : score every discovered model on a fresh held-out
                           batch (e.g. audios7) that none of them trained on, to
                           check generalisation. Features are built from gt.csv +
                           the embedding cache; scored against the batch labels.

Outputs (in --out_dir)
----------------------
  per_model/<model_id>/predictions.csv      per-row scores + decisions at key thresholds
  per_model/<model_id>/threshold_sweep.csv  P/R/F1/confusion at every threshold
  per_model/<model_id>/metrics.json         full compute_metrics dump
  eval_summary.csv                          one row per model (AUC/AP/F1/recall@p*)
  eval_summary.xlsx                         summary + per-model sweeps + per-region

Run (own test sets):
  python ec2/evaluate_models.py \
      --models_root /home/ubuntu/nn/runs/retrain_20pct/models \
      --data_dir    /home/ubuntu/nn/data \
      --out_dir     /home/ubuntu/nn/runs/retrain_20pct/eval_own

Run (validity on a new batch, only the two finalised picks):
  python ec2/evaluate_models.py \
      --models_root /home/ubuntu/nn/runs/retrain_20pct/models \
      --models "default_last_pca98/seed_42,tiny_l9_pca95/seed_44" \
      --data_dir /home/ubuntu/nn/data --test_batch audios7 \
      --out_dir  /home/ubuntu/nn/runs/eval_audios7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _data_pipeline import compute_metrics, load_gt_and_filter, setup_logging
from neural_baseline_train import ARCH_CONFIG
# Reuse the exact artifact loader/predictor + sweep helpers that produced the
# finalised numbers, so a model scores here identically to its training run.
from aug_ablation2 import load_and_predict
from evaluate_final_models import (threshold_sweep, region_breakdown,
                                   build_features, _rap)


def discover_models(models_root: Path, filters: list[str], log) -> list[dict]:
    """Walk models_root and return one descriptor per testable model folder.

    A folder is "testable" if it contains BOTH model.pt and inference_meta.json.
    `filters` (comma list of substrings) keeps only models whose id contains one
    of them -- e.g. "default_last_pca98" or "seed_42". The model id is the folder
    path relative to models_root with separators flattened to '/', so a typical
    id is "default_last_pca98/seed_42".

    Returns dicts: {id, dir, meta} sorted by id. meta is the parsed
    inference_meta.json (carries layer, feat_cols, hidden, dropout, in_dim, ...).
    """
    out = []
    for meta_path in sorted(models_root.rglob("inference_meta.json")):
        var_dir = meta_path.parent
        if not (var_dir / "model.pt").exists():
            continue
        mid = var_dir.relative_to(models_root).as_posix()
        if filters and not any(f in mid for f in filters):
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            log.warning("skip %s: bad inference_meta.json (%s)", mid, e)
            continue
        out.append({"id": mid, "dir": var_dir, "meta": meta})
    log.info("discovered %d testable model(s) under %s%s", len(out), models_root,
             f" matching {filters}" if filters else "")
    return out


def arch_for(meta: dict) -> dict:
    """Resolve the {hidden, dropout, weight_decay} arch dict for a model.

    load_and_predict reads hidden/dropout straight from inference_meta.json, so
    this is mainly a fallback for older metas: prefer the meta's own hidden/
    dropout, else look the arch name up in ARCH_CONFIG, else use 'default'.
    """
    if "hidden" in meta and "dropout" in meta:
        return {"hidden": tuple(meta["hidden"]), "dropout": float(meta["dropout"]),
                "weight_decay": float(meta.get("weight_decay", 0.0))}
    return ARCH_CONFIG.get(meta.get("arch", "default"), ARCH_CONFIG["default"])


def decision_columns(meta_df: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                     m: dict, stored_thr: float, override_thr: float | None
                     ) -> pd.DataFrame:
    """Build the per-row predictions table with hard decisions at the thresholds
    that matter: the model's stored best-F1 threshold, 0.5, the data-derived
    best-F1 threshold, and (optionally) a user --threshold override.

    `m` is the compute_metrics dict for (y, p); `stored_thr` is the threshold
    baked into inference_meta.json at train time.
    """
    def col(name):
        return meta_df[name].astype(str).to_numpy() if name in meta_df.columns else ""
    best_thr = m["best_f1"]["threshold"]
    out = pd.DataFrame({
        "npy_filename": col("npy_filename"), "group_id": col("group_id"),
        "batch": col("batch"), "region": col("region"),
        "label": y, "pred_score": np.round(p, 6),
        "pred@stored_thr": (p >= stored_thr).astype(int),
        "pred@best_f1_thr": (p >= best_thr).astype(int),
        "pred@0.5": (p >= 0.5).astype(int),
    })
    if override_thr is not None:
        out["pred@override"] = (p >= override_thr).astype(int)
    if "pred_score" in meta_df.columns:          # own-test mode: the saved score
        out["saved_pred_score"] = np.round(meta_df["pred_score"].to_numpy(float), 6)
    return out


def evaluate_model(spec: dict, eval_df: pd.DataFrame, cache_path: Path,
                   own_mode: bool, override_thr: float | None, thr_step: float,
                   device, out_dir: Path, sheets: dict, log) -> dict:
    """Score ONE discovered model and write its artifacts; return a summary row.

    Steps:
      1. read layer + feat_order from the model's meta;
      2. rebuild ORIG features (wavlm[layer] | whisper | feat_*) for eval_df's
         rows via build_features, in the model's stored feat_order;
      3. load_and_predict (reloads scaler+pca+model.pt, checks in_dim);
      4. in own-test mode, verify the reload reproduces the saved pred_score;
      5. compute_metrics + threshold_sweep + region_breakdown;
      6. dump predictions.csv / threshold_sweep.csv / metrics.json and stash the
         sweep + region sheets for the workbook.
    """
    mid, var_dir, meta = spec["id"], spec["dir"], spec["meta"]
    layer = str(meta.get("wavlm_layer", "last"))
    feat_order = meta.get("feat_cols") or None
    stored_thr = float(meta.get("best_f1_threshold", 0.5))
    arch = arch_for(meta)

    y = eval_df["label"].to_numpy().astype(int)
    X, _ = build_features(eval_df, layer, cache_path, log, feat_order)
    p, _ = load_and_predict(var_dir, X, arch, device, log)

    max_diff = float("nan")
    if own_mode and "pred_score" in eval_df.columns:
        saved = eval_df["pred_score"].to_numpy(float)
        if len(saved) == len(p):
            max_diff = float(np.max(np.abs(p - saved)))
            lvl = log.info if max_diff < 1e-3 else log.warning
            lvl("[%s] reload max|recon-saved| = %.2e (<1e-3 => exact)", mid, max_diff)

    m = compute_metrics(y, p)
    md = out_dir / "per_model" / mid.replace("/", "__")
    md.mkdir(parents=True, exist_ok=True)
    preds = decision_columns(eval_df, y, p, m, stored_thr, override_thr)
    preds.to_csv(md / "predictions.csv", index=False)
    sweep = threshold_sweep(y, p, thr_step)
    sweep.to_csv(md / "threshold_sweep.csv", index=False)
    with (md / "metrics.json").open("w") as f:
        json.dump(m, f, indent=2)
    sheets[f"{mid}_sweep"[:31]] = sweep
    rb = region_breakdown(eval_df, y, p)
    if len(rb):
        sheets[f"{mid}_region"[:31]] = rb

    log.info("[%s] n=%d pos=%d AUC=%.4f AP=%.4f best_f1=%.4f@%.2f r@p90=%.4f",
             mid, m["n"], m["n_pos"], m["auc"], m["ap"], m["best_f1"]["f1"],
             m["best_f1"]["threshold"], _rap(m, "p90"))
    return {
        "model_id": mid, "variant": meta.get("variant", ""), "seed": meta.get("seed", ""),
        "mode": meta.get("mode", ""), "used_weights": True,
        "repro_max_abs_diff": (round(max_diff, 8) if max_diff == max_diff else float("nan")),
        "n": m["n"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
        "prevalence": round(m["n_pos"] / max(m["n"], 1), 4),
        "auc": round(m["auc"], 4), "ap": round(m["ap"], 4),
        "best_f1": round(m["best_f1"]["f1"], 4), "best_thr": round(m["best_f1"]["threshold"], 3),
        "stored_thr": round(stored_thr, 3),
        "f1@stored_thr": round(_f1_at(y, p, stored_thr), 4),
        "f1@0.5": round(m["thr0.5"]["f1"], 4),
        "recall@p80": round(_rap(m, "p80"), 4), "recall@p90": round(_rap(m, "p90"), 4),
        "recall@p95": round(_rap(m, "p95"), 4),
        "thr@p90": round(_rap(m, "p90", "threshold"), 3),
    }


def _f1_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    """Binary F1 of predictions (p >= thr) against y. Small local helper so the
    summary can report F1 at the model's stored operating threshold."""
    from sklearn.metrics import f1_score
    return float(f1_score(y, (p >= thr).astype(int), zero_division=0))


def main() -> int:
    """Parse args, discover models, build the eval set (own test sets or a fresh
    --test_batch), score every model, and write the combined summary + workbook."""
    ap = argparse.ArgumentParser(description="Score a folder of saved models")
    ap.add_argument("--models_root", required=True,
                    help="folder to search recursively for <model>/inference_meta.json")
    ap.add_argument("--data_dir", required=True, help="folder with gt.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache", default="",
                    help="embeddings_cache.npz (defaults to <data_dir>/embeddings_cache.npz)")
    ap.add_argument("--models", default="",
                    help="comma list of substrings; keep only models whose id "
                         "contains one (e.g. 'default_last_pca98,seed_42'). Empty=all.")
    ap.add_argument("--test_batch", default="",
                    help="comma batch(es) to score every model on as a fresh "
                         "held-out set (mode B). Empty = each model's own test set.")
    ap.add_argument("--min_duration", type=float, default=30.0,
                    help="(test_batch mode) drop eval rows shorter than this")
    ap.add_argument("--threshold", type=float, default=None,
                    help="optional decision threshold to also report per row")
    ap.add_argument("--thr_step", type=float, default=0.01,
                    help="threshold sweep granularity (default 0.01 -> 101 rows)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")
    models_root = Path(args.models_root)
    filters = [f.strip() for f in args.models.split(",") if f.strip()]
    test_batches = [b.strip() for b in args.test_batch.split(",") if b.strip()]

    log = setup_logging(out_dir / "log_eval.txt", name="eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s  models_root=%s  test_batch=%s",
             device, models_root, test_batches or "(own test sets)")

    models = discover_models(models_root, filters, log)
    if not models:
        log.error("no testable models under %s (need model.pt + inference_meta.json)", models_root)
        return 1

    # mode B: one shared eval frame (the held-out batch) reused by every model
    gt_eval = None
    if test_batches:
        gt = load_gt_and_filter(data_dir, args.min_duration, log)
        gt_eval = gt[gt["batch"].astype(str).isin(test_batches)].reset_index(drop=True)
        if len(gt_eval) == 0:
            log.error("no labelled rows for batch(es) %s after min_duration=%.1f",
                      test_batches, args.min_duration)
            return 1
        log.info("eval batch(es) %s: %d rows (pos=%d)", test_batches, len(gt_eval),
                 int((gt_eval["label"] == 1).sum()))

    summary_rows, sheets = [], {}
    for spec in models:
        log.info("=" * 70); log.info("MODEL %s", spec["id"])
        if test_batches:
            eval_df, own_mode = gt_eval, False
        else:
            pred_path = spec["dir"] / "predictions.csv"
            if not pred_path.exists():
                log.error("[%s] no predictions.csv for own-test mode -- skip", spec["id"])
                continue
            eval_df, own_mode = pd.read_csv(pred_path), True
        summary_rows.append(evaluate_model(
            spec, eval_df, cache_path, own_mode, args.threshold, args.thr_step,
            device, out_dir, sheets, log))

    if not summary_rows:
        log.error("nothing scored."); return 1
    summary = pd.DataFrame(summary_rows).sort_values("best_f1", ascending=False).reset_index(drop=True)
    summary.to_csv(out_dir / "eval_summary.csv", index=False)
    log.info("=" * 70); log.info("SUMMARY:\n%s", summary.to_string(index=False))

    try:
        with pd.ExcelWriter(out_dir / "eval_summary.xlsx", engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="summary", index=False)
            for name, sdf in sheets.items():
                sdf.to_excel(xw, sheet_name=name[:31], index=False)
        log.info("wrote %s (%d sheets)", out_dir / "eval_summary.xlsx", 1 + len(sheets))
    except Exception as e:
        log.warning("could not write xlsx (%s); CSVs are saved. pip install openpyxl", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
