#!/usr/bin/env python3
"""Evaluate the TWO finalised models from their stored weights -> one workbook.

The two finalised (model x protocol) pairings, with exported artifacts:
    Model 1 : default_last_pca98 (+ casual train-only)   on the 20pct split
              run dir default: m1_casual_20pct/default_last_pca98/
    Model 2 : tiny_l9_pca95 (no casual)                  on the a6 split
              run dir default: m2_nocasual_a6/tiny_l9_pca95/

For each, this:
  1. reads the run's predictions.csv -- the model's exact test set (it is
     gt.iloc[test_idx] + pred_score, so it carries label / batch / region /
     group_id / feat_* already),
  2. rebuilds the ORIG test features (wavlm[layer] + whisper + feat_*) from the
     cache keyed by those npy_filenames,
  3. LOADS the stored model.pt + scaler.joblib + pca.joblib and predicts -- i.e.
     it genuinely uses the finalised weights -- and VERIFIES the reproduced
     probabilities equal the saved pred_score (max|diff| should be ~0),
  4. computes detailed metrics, a full threshold sweep, and a per-region
     breakdown.

Outputs (in --out_dir):
  predictions_model1_20pct.csv   per-row predictions (+ decisions at key thresholds)
  predictions_model2_a6.csv
  evaluate_final_models.xlsx     sheets:
       summary                   both models side by side (AUC/AP/best-F1/
                                 recall@p50..95/threshold + reproduction check)
       model1_20pct_preds        per-row predictions
       model1_20pct_sweep        threshold 0..1: tp/fp/tn/fn/prec/rec/f1/...
       model1_20pct_by_region    per-region metrics
       model2_a6_preds / _sweep / _by_region

Run:
  python ec2/evaluate_final_models.py \
      --data_dir  /home/ubuntu/nn/data \
      --cache     /home/ubuntu/nn/data/embeddings_cache.npz \
      --runs_root /home/ubuntu/nn/runs \
      --out_dir   /home/ubuntu/nn/runs/final_eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _data_pipeline import (compute_metrics, load_cache_reindexed, setup_logging)
from neural_baseline_train import ARCH_CONFIG
# Reuse the exact artifact loader + predictor used to anchor the headline numbers.
from aug_ablation2 import load_and_predict, _rp
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


def build_test_features(df: pd.DataFrame, layer: str, cache_path: Path, log):
    """ORIG test features (wavlm[layer] + whisper + feat_*) for the exact rows in
    a run's predictions.csv, keyed by npy_filename -- same feature order as train."""
    filenames = df["npy_filename"].astype(str).to_numpy()
    wavlm_cache, whisper_cache = load_cache_reindexed(
        cache_path, filenames, ["orig"], [layer], log)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    parts = [wavlm_cache[(layer, "orig")], whisper_cache["orig"]]
    if feat_cols:
        feat = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        parts.append(feat.to_numpy().astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32), len(feat_cols)


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
    ap.add_argument("--thr_step", type=float, default=0.01,
                    help="threshold sweep granularity (default 0.01 -> 101 rows)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")
    runs_root = Path(args.runs_root)

    log = setup_logging(out_dir / "log_final_eval.txt", name="final_eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s  runs_root=%s  cache=%s", device, runs_root, cache_path)

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

    for spec in models:
        key, title = spec["key"], spec["title"]
        log.info("=" * 70)
        log.info("%s", title)
        var_dir = runs_root / spec["run_dir"] / spec["variant"]
        pred_path = var_dir / "predictions.csv"
        if not pred_path.exists():
            log.error("[%s] missing %s -- skipping. Re-export with --export_artifacts "
                      "and --dump? Need predictions.csv.", key, pred_path)
            continue
        df = pd.read_csv(pred_path)
        if "label" not in df.columns or "npy_filename" not in df.columns:
            log.error("[%s] predictions.csv lacks label/npy_filename -- skipping", key)
            continue
        y = df["label"].to_numpy().astype(int)

        # ORIG test features for these exact rows, then load weights + predict.
        X_test_orig, n_feat = build_test_features(df, spec["layer"], cache_path, log)
        arch = ARCH_CONFIG[spec["arch"]]
        used_weights = (var_dir / "model.pt").exists()
        if used_weights:
            p, meta = load_and_predict(var_dir, X_test_orig, arch, device, log)
            saved = df["pred_score"].to_numpy(float) if "pred_score" in df.columns else None
            max_diff = float(np.max(np.abs(p - saved))) if saved is not None and len(saved) == len(p) else float("nan")
            lvl = log.info if (max_diff != max_diff or max_diff < 1e-3) else log.warning
            lvl("[%s] reproduced from model.pt: max|recon-saved pred_score| = %.2e "
                "(<1e-3 => exact)", key, max_diff)
        else:
            log.warning("[%s] no model.pt at %s -- falling back to saved pred_score "
                        "(metrics still valid, weights not re-run).", key, var_dir)
            p = df["pred_score"].to_numpy(float)
            meta, max_diff = {}, float("nan")

        m = compute_metrics(y, p)
        n, npos, nneg = m["n"], m["n_pos"], m["n_neg"]
        log.info("[%s] n=%d  pos=%d (%.1f%%)  AUC=%.4f AP=%.4f  best_f1=%.4f@%.2f",
                 key, n, npos, 100 * npos / max(n, 1), m["auc"], m["ap"],
                 m["best_f1"]["f1"], m["best_f1"]["threshold"])
        log.info("[%s] recall@  p50=%.3f p80=%.3f p85=%.3f p90=%.3f p95=%.3f", key,
                 _rap(m, "p50"), _rap(m, "p80"), _rap(m, "p85"),
                 _rap(m, "p90"), _rap(m, "p95"))

        best_thr = m["best_f1"]["threshold"]
        preds_out = pd.DataFrame({
            "npy_filename": df["npy_filename"].astype(str),
            "group_id": df["group_id"].astype(str) if "group_id" in df.columns else "",
            "question_id": df["question_id"] if "question_id" in df.columns else "",
            "batch": df["batch"].astype(str) if "batch" in df.columns else "",
            "region": df["region"].astype(str) if "region" in df.columns else "",
            "label": y,
            "pred_score": np.round(p, 6),
            "saved_pred_score": (np.round(df["pred_score"].to_numpy(float), 6)
                                 if "pred_score" in df.columns else np.nan),
            "pred@best_thr": (p >= best_thr).astype(int),
            "pred@0.5": (p >= 0.5).astype(int),
            "pred@p90_thr": (p >= _rap(m, "p90", "threshold")).astype(int),
        })
        preds_out.to_csv(out_dir / f"predictions_{key}.csv", index=False)
        sheets[f"{key}_preds"] = preds_out
        sheets[f"{key}_sweep"] = threshold_sweep(y, p, args.thr_step)
        rb = region_breakdown(df, y, p)
        if len(rb):
            sheets[f"{key}_by_region"] = rb

        summary_rows.append({
            "model": key, "title": title, "protocol": spec["protocol"],
            "variant": spec["variant"], "run_dir": spec["run_dir"],
            "used_weights": used_weights,
            "repro_max_abs_diff": (round(max_diff, 8) if max_diff == max_diff else float("nan")),
            "n_test": n, "n_pos": npos, "n_neg": nneg,
            "prevalence": round(npos / max(n, 1), 4),
            "auc": round(m["auc"], 4), "ap": round(m["ap"], 4),
            "best_f1": round(m["best_f1"]["f1"], 4),
            "best_thr": round(best_thr, 3),
            "f1@0.5": round(m["thr0.5"]["f1"], 4),
            "precision@0.5": round(m["thr0.5"].get("precision", float("nan")), 4),
            "recall@0.5": round(m["thr0.5"].get("recall", float("nan")), 4),
            "recall@p50": round(_rap(m, "p50"), 4),
            "recall@p80": round(_rap(m, "p80"), 4),
            "recall@p85": round(_rap(m, "p85"), 4),
            "recall@p90": round(_rap(m, "p90"), 4),
            "recall@p95": round(_rap(m, "p95"), 4),
            "thr@p80": round(_rap(m, "p80", "threshold"), 3),
            "thr@p90": round(_rap(m, "p90", "threshold"), 3),
            "thr@p95": round(_rap(m, "p95", "threshold"), 3),
        })

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

    log.info("READING: 'summary' compares both finalised models at every operating "
             "point; repro_max_abs_diff ~0 confirms the stored weights reproduce "
             "the saved predictions. <model>_sweep gives the precision/recall "
             "trade-off at each threshold; pick the row at your target precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
