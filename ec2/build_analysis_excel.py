#!/usr/bin/env python3
"""Build the 2-model analysis workbook (6 sheets) from full-prediction dumps.

Consumes the full_predictions.csv files written by neural_baseline_train_v2.py
when run with --dump_full_predictions true. Because those dumps come from the
exact trained model, the numbers here reproduce the original runs exactly.

For EACH model (2 total) it writes 3 sheets:
  <label>_pred     per-file predictions on ALL mercer rows (train+val+test),
                   with the 20%-split prob/membership AND the a6-split
                   prob/membership side by side.
  <label>_sweep    threshold sweep 0.00..1.00 step 0.01 over ALL rows:
                   a6 {fp,fn,tp,tn,prec,recall} and 20% {fp,fn,tp,tn,prec,recall}
  <label>_sweepT   same sweep but TEST rows only (each model's own test set)
=> 6 sheets total.

Each model needs TWO runs (same variant, two split protocols):
  --modelN_20pct_run : a Mode-A run (--test_batches "" -> 60/20/20 split)
  --modelN_a6_run    : a Mode-B run (--test_batches audios6)
both launched with --dump_full_predictions true. We read
  <run>/<variant>/full_predictions.csv

Mercer audios only (batch in audios2/4/5/6); casual/ALLSTAR rows are excluded
from the sheets even if a model trained on them.

--------------------------------------------------------------------------------
EXAMPLE
--------------------------------------------------------------------------------
python ec2/build_analysis_excel.py \
    --model1_variant default_last_pca98 \
    --model1_label    m1_default_last_pca98_casual \
    --model1_20pct_run /home/ubuntu/nn/runs/m1_casual_20pct \
    --model1_a6_run    /home/ubuntu/nn/runs/m1_casual_a6 \
    --model2_variant tiny_l9_pca95 \
    --model2_label    m2_tiny_l9_pca95 \
    --model2_20pct_run /home/ubuntu/nn/runs/m2_nocasual_20pct \
    --model2_a6_run    /home/ubuntu/nn/runs/m2_nocasual_a6 \
    --out /home/ubuntu/nn/runs/analysis_2models.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MERCER_BATCHES = {"audios2", "audios4", "audios5", "audios6"}
THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def _load_full_pred(run_dir: Path, variant: str) -> pd.DataFrame:
    p = run_dir / variant / "full_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}\n  -> run neural_baseline_train_v2.py on this config "
            f"with --dump_full_predictions true")
    df = pd.read_csv(p)
    df["batch"] = df["batch"].astype(str)
    df["npy_filename"] = df["npy_filename"].astype(str)
    need = {"npy_filename", "batch", "label", "split", "pred_prob"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing columns {missing}")
    return df


def _mercer_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["batch"].isin(MERCER_BATCHES)].reset_index(drop=True)


def _audios_folder(batch: str) -> str:
    # 'audios6' -> '6'
    return batch.replace("audios", "") if batch.startswith("audios") else batch


def confusion_at(p: np.ndarray, y: np.ndarray, t: float) -> dict:
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return {"fp": fp, "fn": fn, "tp": tp, "tn": tn,
            "precision": round(prec, 4) if prec == prec else np.nan,
            "recall": round(rec, 4) if rec == rec else np.nan}


def sweep_table(p20: np.ndarray, y20: np.ndarray,
                pa6: np.ndarray, ya6: np.ndarray) -> pd.DataFrame:
    """One row per threshold. a6 block then 20% block."""
    rows = []
    for t in THRESHOLDS:
        a6 = confusion_at(pa6, ya6, t)
        p20c = confusion_at(p20, y20, t)
        rows.append({
            "threshold": t,
            "a6_fp": a6["fp"], "a6_fn": a6["fn"], "a6_tp": a6["tp"],
            "a6_tn": a6["tn"], "a6_precision": a6["precision"], "a6_recall": a6["recall"],
            "split20_fp": p20c["fp"], "split20_fn": p20c["fn"], "split20_tp": p20c["tp"],
            "split20_tn": p20c["tn"], "split20_precision": p20c["precision"],
            "split20_recall": p20c["recall"],
        })
    return pd.DataFrame(rows)


def build_model_sheets(variant: str, run_20pct: Path, run_a6: Path,
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (pred_sheet, sweep_all, sweep_testonly) for one model."""
    df20 = _mercer_only(_load_full_pred(run_20pct, variant))
    dfa6 = _mercer_only(_load_full_pred(run_a6, variant))

    # Align on npy_filename. label/batch must agree across the two runs.
    m = df20.merge(
        dfa6[["npy_filename", "split", "pred_prob"]],
        on="npy_filename", how="inner", suffixes=("_20", "_a6"))
    n_lost = max(len(df20), len(dfa6)) - len(m)
    if n_lost:
        print(f"  WARNING: {n_lost} rows didn't align between the two runs "
              f"(min_duration / batch set differs?)", file=sys.stderr)

    # ---- Sheet 1: per-file predictions
    pred = pd.DataFrame({
        "Filename": m["npy_filename"],
        "real_id": "",  # blank for the user's VLOOKUP
        "GT": np.where(m["label"].to_numpy() == 1, "yes", "no"),
        "audios_folder": m["batch"].map(_audios_folder),
        "split_20%_test": m["split_20"],
        "Prediction_prob_20%_test": m["pred_prob_20"].round(6),
        "split_a6_test": m["split_a6"],
        "Prediction_prob_a6_test": m["pred_prob_a6"].round(6),
    })

    y = m["label"].to_numpy().astype(int)
    p20 = m["pred_prob_20"].to_numpy().astype(float)
    pa6 = m["pred_prob_a6"].to_numpy().astype(float)

    # ---- Sheet 2: sweep over ALL rows (train+val+test)
    sweep_all = sweep_table(p20, y, pa6, y)

    # ---- Sheet 3: sweep over TEST rows only (each model's own test set)
    test20 = (m["split_20"] == "test").to_numpy()
    testa6 = (m["split_a6"] == "test").to_numpy()
    sweep_test = sweep_table(p20[test20], y[test20], pa6[testa6], y[testa6])

    return pred, sweep_all, sweep_test


def _safe_sheet_name(name: str, suffix: str) -> str:
    # Excel sheet names: <=31 chars, no : \ / ? * [ ]
    base = "".join(c for c in name if c not in ':\\/?*[]')
    full = f"{base}_{suffix}"
    if len(full) > 31:
        base = base[:31 - len(suffix) - 1]
        full = f"{base}_{suffix}"
    return full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1_variant", required=True)
    ap.add_argument("--model1_label", required=True)
    ap.add_argument("--model1_20pct_run", required=True, type=Path)
    ap.add_argument("--model1_a6_run", required=True, type=Path)
    ap.add_argument("--model2_variant", required=True)
    ap.add_argument("--model2_label", required=True)
    ap.add_argument("--model2_20pct_run", required=True, type=Path)
    ap.add_argument("--model2_a6_run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    models = [
        (args.model1_label, args.model1_variant, args.model1_20pct_run, args.model1_a6_run),
        (args.model2_label, args.model2_variant, args.model2_20pct_run, args.model2_a6_run),
    ]

    # Pick whichever xlsx engine is installed.
    engine = None
    for cand in ("openpyxl", "xlsxwriter"):
        try:
            __import__(cand)
            engine = cand
            break
        except ImportError:
            continue
    if engine is None:
        print("ERROR: need an xlsx writer. Install one:\n"
              "    pip install openpyxl", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine=engine) as xw:
        for label, variant, run20, runa6 in models:
            print(f"[{label}] variant={variant}")
            print(f"  20% run: {run20}")
            print(f"  a6  run: {runa6}")
            pred, sweep_all, sweep_test = build_model_sheets(variant, run20, runa6)
            pred.to_excel(xw, sheet_name=_safe_sheet_name(label, "pred"), index=False)
            sweep_all.to_excel(xw, sheet_name=_safe_sheet_name(label, "sweep"), index=False)
            sweep_test.to_excel(xw, sheet_name=_safe_sheet_name(label, "sweepT"), index=False)
            n_test20 = int((pred["split_20%_test"] == "test").sum())
            n_testa6 = int((pred["split_a6_test"] == "test").sum())
            print(f"  rows={len(pred)}  20%-test={n_test20}  a6-test={n_testa6}")

    print(f"wrote {args.out}  (6 sheets: 3 per model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
