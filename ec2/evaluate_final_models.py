#!/usr/bin/env python3
"""Evaluate the TWO finalised models from their stored weights -> one workbook.

The two finalised (model x protocol) pairings, with exported artifacts:
    Model 1 : default_last_pca98 (+ casual train-only)   on the 20pct split
              run dir default: m1_casual_20pct/default_last_pca98/
    Model 2 : tiny_l9_pca95 (no casual)                  on the a6 split
              run dir default: m2_nocasual_a6/tiny_l9_pca95/

Two modes:

  (A) default -- evaluate each model on ITS OWN test set (the run's
      predictions.csv). Reloads model.pt + scaler + pca, predicts, and VERIFIES
      the reproduced probabilities equal the saved pred_score (~0 diff).

  (B) --test_batch audios7 -- evaluate BOTH finalised models on a NEW held-out
      batch (e.g. audios7) that neither trained on, to check validity /
      generalisation. Builds ORIG features for that batch from gt.csv + cache,
      loads the SAME stored weights, predicts, scores against the batch labels.
      (No saved pred_score to verify against -- it's fresh data.)

Either way it computes detailed metrics, a full threshold sweep, and a
per-region breakdown, written to per-row CSVs + one xlsx of sheets.

Outputs (in --out_dir), <tag> = model1_20pct / model2_a6 (mode A) or
<model>_on_<batch> (mode B):
  predictions_<tag>.csv          per-row predictions (+ decisions at key thresholds)
  final_eval_summary.csv         both models side by side
  evaluate_final_models.xlsx     sheets: summary, <tag>_preds, <tag>_sweep,
                                 <tag>_by_region (per model)

Run (own test sets):
  python ec2/evaluate_final_models.py \
      --data_dir  /home/ubuntu/nn/data \
      --cache     /home/ubuntu/nn/data/embeddings_cache.npz \
      --runs_root /home/ubuntu/nn/runs \
      --out_dir   /home/ubuntu/nn/runs/final_eval

Run (validity on the new audios7 batch):
  python ec2/evaluate_final_models.py \
      --data_dir  /home/ubuntu/nn/data \
      --cache     /home/ubuntu/nn/data/embeddings_cache.npz \
      --runs_root /home/ubuntu/nn/runs \
      --out_dir   /home/ubuntu/nn/runs/final_eval_audios7 \
      --test_batch audios7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _data_pipeline import (compute_metrics, load_cache_reindexed,
                            load_gt_and_filter, setup_logging)
from neural_baseline_train import ARCH_CONFIG
# Reuse the exact artifact loader + predictor used to anchor the headline numbers.
from aug_ablation2 import load_and_predict
from aug_ablation import _l9


def _rap(m: dict, key: str, field: str = "recall") -> float:
    v = m.get("recall_at_precision", {}).get(key, {})
    return v.get(field, float("nan")) if isinstance(v, dict) else float("nan")


def threshold_sweep(y: np.ndarray, p: np.ndarray, step: float) -> pd.DataFrame:
    """Confusion + precision/recall/f1/specificity/accuracy at every threshold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    rows = []
    for t in np.round(np.arange(0.0, 1.0 + 1e-9, step), 4):
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append({
            "threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "specificity": round(spec, 4),
            "accuracy": round((tp + tn) / n, 4) if n else 0.0,
            "n_pred_pos": tp + fp,
        })
    return pd.DataFrame(rows)


def region_breakdown(df: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    if "region" not in df.columns:
        return pd.DataFrame()
    reg = df["region"].fillna("UNK").astype(str).to_numpy()
    rows = []
    for r in sorted(set(reg)):
        mask = reg == r
        yr, pr = y[mask], p[mask]
        n = int(mask.sum())
        pos = int((yr == 1).sum())
        row = {"region": r, "n": n, "n_pos": pos, "n_neg": n - pos}
        if 0 < pos < n:
            m = compute_metrics(yr, pr)
            row.update({
                "auc": round(m["auc"], 4), "ap": round(m["ap"], 4),
                "best_f1": round(m["best_f1"]["f1"], 4),
                "best_thr": round(m["best_f1"]["threshold"], 3),
                "recall@p90": round(_rap(m, "p90"), 4),
                "recall@p95": round(_rap(m, "p95"), 4),
            })
        else:
            for c in ("auc", "ap", "best_f1", "best_thr", "recall@p90", "recall@p95"):
                row[c] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_features(df: pd.DataFrame, layer: str, cache_path: Path, log,
                   feat_order: list[str] | None):
    """ORIG features (wavlm[layer] + whisper + feat_*) for the rows in df, keyed
    by npy_filename. feat columns follow feat_order (the model's training order,
    from inference_meta.json) so the vector matches what the scaler/pca expect;
    a feature missing from df is filled with 0.0."""
    filenames = df["npy_filename"].astype(str).to_numpy()
    wavlm_cache, whisper_cache = load_cache_reindexed(
        cache_path, filenames, ["orig"], [layer], log)
    parts = [wavlm_cache[(layer, "orig")], whisper_cache["orig"]]
    if feat_order is None:
        feat_order = [c for c in df.columns if c.startswith("feat_")]
    if feat_order:
        feat = (df.reindex(columns=feat_order)
                  .apply(pd.to_numeric, errors="coerce").fillna(0.0))
        parts.append(feat.to_numpy().astype(np.float32))
    X = np.concatenate(parts, axis=1).astype(np.float32)
    nz = int((np.abs(X).sum(1) == 0).sum())
    if nz:
        log.error("  %d/%d rows have ALL-ZERO features -- those npy are not in the "
                  "cache (extract embeddings for this batch first).", nz, len(X))
    return X, len(feat_order)


def evaluate_and_record(tag: str, title: str, spec: dict, meta_df: pd.DataFrame,
                        y: np.ndarray, p: np.ndarray, used_weights: bool,
                        max_diff: float, eval_label: str, out_dir: Path,
                        sheets: dict, summary_rows: list, thr_step: float, log):
    m = compute_metrics(y, p)
    n, npos, nneg = m["n"], m["n_pos"], m["n_neg"]
    log.info("[%s] eval=%s  n=%d pos=%d (%.1f%%)  AUC=%.4f AP=%.4f best_f1=%.4f@%.2f",
             tag, eval_label, n, npos, 100 * npos / max(n, 1), m["auc"], m["ap"],
             m["best_f1"]["f1"], m["best_f1"]["threshold"])
    log.info("[%s] recall@ p50=%.3f p80=%.3f p85=%.3f p90=%.3f p95=%.3f", tag,
             _rap(m, "p50"), _rap(m, "p80"), _rap(m, "p85"), _rap(m, "p90"), _rap(m, "p95"))

    best_thr = m["best_f1"]["threshold"]
    def col(name):
        return meta_df[name].astype(str) if name in meta_df.columns else ""
    preds_out = pd.DataFrame({
        "npy_filename": col("npy_filename"), "group_id": col("group_id"),
        "question_id": meta_df["question_id"] if "question_id" in meta_df.columns else "",
        "batch": col("batch"), "region": col("region"),
        "label": y, "pred_score": np.round(p, 6),
        "pred@best_thr": (p >= best_thr).astype(int),
        "pred@0.5": (p >= 0.5).astype(int),
        "pred@p90_thr": (p >= _rap(m, "p90", "threshold")).astype(int),
    })
    if "pred_score" in meta_df.columns:
        preds_out["saved_pred_score"] = np.round(meta_df["pred_score"].to_numpy(float), 6)
    preds_out.to_csv(out_dir / f"predictions_{tag}.csv", index=False)
    sheets[f"{tag}_preds"] = preds_out
    sheets[f"{tag}_sweep"] = threshold_sweep(y, p, thr_step)
    rb = region_breakdown(meta_df, y, p)
    if len(rb):
        sheets[f"{tag}_by_region"] = rb

    summary_rows.append({
        "model": spec["key"], "title": title, "eval_set": eval_label,
        "protocol": spec["protocol"], "variant": spec["variant"],
        "run_dir": spec["run_dir"], "used_weights": used_weights,
        "repro_max_abs_diff": (round(max_diff, 8) if max_diff == max_diff else float("nan")),
        "n_test": n, "n_pos": npos, "n_neg": nneg,
        "prevalence": round(npos / max(n, 1), 4),
        "auc": round(m["auc"], 4), "ap": round(m["ap"], 4),
        "best_f1": round(m["best_f1"]["f1"], 4), "best_thr": round(best_thr, 3),
        "f1@0.5": round(m["thr0.5"]["f1"], 4),
        "precision@0.5": round(m["thr0.5"].get("precision", float("nan")), 4),
        "recall@0.5": round(m["thr0.5"].get("recall", float("nan")), 4),
        "recall@p50": round(_rap(m, "p50"), 4), "recall@p80": round(_rap(m, "p80"), 4),
        "recall@p85": round(_rap(m, "p85"), 4), "recall@p90": round(_rap(m, "p90"), 4),
        "recall@p95": round(_rap(m, "p95"), 4),
        "thr@p80": round(_rap(m, "p80", "threshold"), 3),
        "thr@p90": round(_rap(m, "p90", "threshold"), 3),
        "thr@p95": round(_rap(m, "p95", "threshold"), 3),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache", default="",
                    help="embeddings_cache.npz (defaults to <data_dir>/embeddings_cache.npz)")
    ap.add_argument("--runs_root", default="/home/ubuntu/nn/runs")
    ap.add_argument("--m1_dir", default="m1_casual_20pct",
                    help="run dir of finalised Model 1 @ 20pct")
    ap.add_argument("--m2_dir", default="m2_nocasual_a6",
                    help="run dir of finalised Model 2 @ a6")
    ap.add_argument("--test_batch", default="",
                    help="comma-separated batch(es) to evaluate BOTH models on as "
                         "a fresh held-out set (e.g. audios7). Empty = each model "
                         "on its own predictions.csv test set.")
    ap.add_argument("--min_duration", type=float, default=30.0,
                    help="(test_batch mode) drop eval rows shorter than this; "
                         "default 30 matches how the models were trained.")
    ap.add_argument("--thr_step", type=float, default=0.01,
                    help="threshold sweep granularity (default 0.01 -> 101 rows)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")
    runs_root = Path(args.runs_root)
    test_batches = [b.strip() for b in args.test_batch.split(",") if b.strip()]

    log = setup_logging(out_dir / "log_final_eval.txt", name="final_eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s  runs_root=%s  cache=%s  test_batch=%s",
             device, runs_root, cache_path, test_batches or "(own test sets)")

    models = [
        {"key": "model1_20pct", "title": "Model 1: default_last_pca98 (+casual) @ 20pct",
         "arch": "default", "layer": "last", "variant": "default_last_pca98",
         "run_dir": args.m1_dir, "protocol": "20pct"},
        {"key": "model2_a6", "title": "Model 2: tiny_l9_pca95 @ a6",
         "arch": "tiny", "layer": _l9(), "variant": "tiny_l9_pca95",
         "run_dir": args.m2_dir, "protocol": "a6"},
    ]

    summary_rows: list[dict] = []
    sheets: dict[str, pd.DataFrame] = {}

    # In test_batch mode, load gt once and subset to the eval batch(es).
    gt_eval = None
    if test_batches:
        gt = load_gt_and_filter(data_dir, args.min_duration, log)
        gt_eval = gt[gt["batch"].astype(str).isin(test_batches)].copy()
        gt_eval = gt_eval[gt_eval["label"].isin([0, 1])].reset_index(drop=True)
        if len(gt_eval) == 0:
            log.error("No labelled rows for batch(es) %s after min_duration=%.1f. "
                      "Check the batch name / labels / cache.", test_batches, args.min_duration)
            return 1
        log.info("eval batch(es) %s: %d labelled rows (pos=%d) after min_duration=%.1f",
                 test_batches, len(gt_eval), int((gt_eval["label"] == 1).sum()), args.min_duration)

    for spec in models:
        key, title = spec["key"], spec["title"]
        var_dir = runs_root / spec["run_dir"] / spec["variant"]
        meta_path = var_dir / "inference_meta.json"
        meta0 = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        feat_order = meta0.get("feat_cols") or None
        arch = ARCH_CONFIG[spec["arch"]]
        log.info("=" * 70)
        log.info("%s", title)

        if not (var_dir / "model.pt").exists() and test_batches:
            log.error("[%s] no model.pt at %s -- cannot evaluate a new batch without "
                      "the weights. Re-export that run with --export_artifacts true.",
                      key, var_dir)
            continue

        if test_batches:
            # ---- mode B: fresh held-out batch, same weights ----
            eval_label = "+".join(test_batches)
            tag = f"{key}_on_{'_'.join(test_batches)}"
            meta_df = gt_eval
            y = gt_eval["label"].to_numpy().astype(int)
            X, _ = build_features(meta_df, spec["layer"], cache_path, log, feat_order)
            p, _ = load_and_predict(var_dir, X, arch, device, log)
            used_weights, max_diff = True, float("nan")
        else:
            # ---- mode A: own predictions.csv test set ----
            eval_label = f"own test ({spec['protocol']})"
            tag = key
            pred_path = var_dir / "predictions.csv"
            if not pred_path.exists():
                log.error("[%s] missing %s -- skipping.", key, pred_path)
                continue
            meta_df = pd.read_csv(pred_path)
            if "label" not in meta_df.columns or "npy_filename" not in meta_df.columns:
                log.error("[%s] predictions.csv lacks label/npy_filename -- skipping", key)
                continue
            y = meta_df["label"].to_numpy().astype(int)
            X, _ = build_features(meta_df, spec["layer"], cache_path, log, feat_order)
            if (var_dir / "model.pt").exists():
                p, _ = load_and_predict(var_dir, X, arch, device, log)
                saved = (meta_df["pred_score"].to_numpy(float)
                         if "pred_score" in meta_df.columns else None)
                max_diff = (float(np.max(np.abs(p - saved)))
                            if saved is not None and len(saved) == len(p) else float("nan"))
                lvl = log.info if (max_diff != max_diff or max_diff < 1e-3) else log.warning
                lvl("[%s] reproduced from model.pt: max|recon-saved| = %.2e (<1e-3 => exact)",
                    key, max_diff)
                used_weights = True
            else:
                log.warning("[%s] no model.pt -- using saved pred_score (weights not re-run).", key)
                p = meta_df["pred_score"].to_numpy(float)
                used_weights, max_diff = False, float("nan")

        evaluate_and_record(tag, title, spec, meta_df, y, p, used_weights, max_diff,
                            eval_label, out_dir, sheets, summary_rows, args.thr_step, log)

    if not summary_rows:
        log.error("No models evaluated. Check --runs_root/--m1_dir/--m2_dir.")
        return 1

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "final_eval_summary.csv", index=False)
    log.info("=" * 70)
    log.info("SUMMARY:\n%s", summary.to_string(index=False))

    xlsx = out_dir / "evaluate_final_models.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="summary", index=False)
            for name, sdf in sheets.items():
                sdf.to_excel(xw, sheet_name=name[:31], index=False)
        log.info("wrote %s (%d sheets: summary + %d)", xlsx, 1 + len(sheets), len(sheets))
    except Exception as e:
        log.warning("could not write xlsx (%s); CSVs are saved. pip install openpyxl", e)

    if test_batches:
        log.info("READING: this is the SAME finalised weights scored on %s (data "
                 "neither model trained on). Compare auc/ap/recall@p90 here vs the "
                 "models' own-test numbers -- a big drop = the model does not "
                 "generalise to this batch; similar = it holds up.", test_batches)
    else:
        log.info("READING: repro_max_abs_diff ~0 confirms the stored weights "
                 "reproduce the saved predictions. <tag>_sweep gives precision/"
                 "recall at each threshold -- pick the row at your target precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
