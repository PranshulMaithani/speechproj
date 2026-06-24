#!/usr/bin/env python3
"""Exhaustive aug-COMBINATION ablation, SINGLE seed, one finalised model.

What it does
------------
For ONE chosen model (Model 1 or Model 2) on a FROZEN, pre-decided split, it
trains and scores EVERY combination of augmentations of size 3, 4, 5 and 6
(out of the 8 in the cache) -- C(8,3)+C(8,4)+C(8,5)+C(8,6) = 56+70+56+28 = 210
combinations -- then ranks them by test best-F1 so you can read off the single
best aug recipe.

How it differs from the other aug scripts
-----------------------------------------
  aug_strategy_20pct.py : 5-seed split-reshuffled search (singles/LOO/greedy).
  aug_ablation.py       : frozen split, leave-one-out / greedy, multi-seed avg.
  THIS one             : ONE seed, ONE model, and an EXHAUSTIVE sweep over the
                          mid-size combinations (3-6) rather than a heuristic.
                          No averaging -- it is a single decisive draw on a
                          fixed split, the cheapest way to compare every
                          mid-size recipe head-to-head.

The two models (architecture + its pre-decided split)
-----------------------------------------------------
  Model 1: default / wavlm 'last' / pca98 / + casual (train-only)
           split = 20pct (Mode A, 60/20/20 candidate-wise on audios2/4/5/6)
  Model 2: tiny    / wavlm  l9    / pca95 / no casual
           split = a6   (Mode B, test = ALL audios6; train/val from the rest)

"Pre-decided split" = a single fixed --seed drives build_splits (with
--min_duration 30, seed 42 reproduces the finalised split). The split is built
ONCE and reused for all 210 combinations, so the ONLY thing differing between
combinations is which augs expand the training set. Model init is also fixed
(derive_model_seed(seed, variant)) so it is a clean paired comparison.

Outputs (in --out_dir, with <model> = m1 / m2)
----------------------------------------------
  aug_combo_<model>_raw.csv          one row per combination: test best-F1, auc,
                                     ap, recall@p80/85/90/95, val F1, threshold
  aug_combo_<model>_leaderboard.csv  same rows ranked by test best-F1 (+ rank)
  aug_combo_<model>_by_size.csv      per size (3/4/5/6): count + mean/median/max
                                     test best-F1 + the best subset of that size
  aug_combo_<model>.xlsx             leaderboard / by_size / raw as sheets
  splits/<model>_seed<seed>.json     the frozen split's candidate membership

Run (Model 1, sizes 3-6, seed 42):
  python ec2/aug_combo_ablation.py \
      --data_dir /home/ubuntu/nn/data \
      --cache    /home/ubuntu/nn/data/embeddings_cache.npz \
      --out_dir  /home/ubuntu/nn/runs/aug_combo_m1 \
      --model 1 --seed 42

Run (Model 2 on its a6 split):
  python ec2/aug_combo_ablation.py ... --out_dir .../aug_combo_m2 --model 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _data_pipeline import (assert_no_group_leak, build_splits,
                            load_cache_reindexed, load_gt_and_filter,
                            resolve_use_augs, setup_logging)
from neural_baseline_train import ARCH_CONFIG, train_one
from neural_baseline_multiseed import derive_model_seed
from aug_ablation import TRAIN_BATCHES, _l9, build_xy


def _rp(rap: dict, k: str, field: str = "recall") -> float:
    """Pull recall (or 'threshold') at a fixed-precision operating point out of
    compute_metrics' recall_at_precision block; NaN if that point is absent."""
    v = rap.get(k, {})
    return v.get(field, float("nan")) if isinstance(v, dict) else float("nan")


def cfg_name(augs, all_augs: list[str]) -> str:
    """Canonical '+'-joined name for an aug subset, ordered as in the cache's aug
    list (so {combo,noise} and {noise,combo} print the same stable label)."""
    s = set(augs)
    return "+".join(a for a in all_augs if a in s)


def model_spec(model: str) -> dict:
    """Resolve '1'/'2' to that finalised model's architecture + pre-decided split.

    Returns {tag, label, variant, arch, layer, pca, train_only, test_batches}.
    Model 1 -> default/last/pca98 + casual on the 20pct split (test_batches None);
    Model 2 -> tiny/l9/pca95, no casual, on the a6 split (test_batches=[audios6]).
    """
    if str(model) == "1":
        return {"tag": "m1", "label": "m1_default_last_pca98_casual",
                "variant": "default_last_pca98", "arch": "default", "layer": "last",
                "pca": 0.98, "train_only": ["casual"], "test_batches": None}
    if str(model) == "2":
        return {"tag": "m2", "label": "m2_tiny_l9_pca95",
                "variant": "tiny_l9_pca95", "arch": "tiny", "layer": _l9(),
                "pca": 0.95, "train_only": [], "test_batches": ["audios6"]}
    raise SystemExit(f"--model must be 1 or 2, got {model!r}")


def main() -> int:
    """Parse args, build the one frozen split for the chosen model, enumerate all
    size-3..6 aug combinations, train+score each on a single seed, and write the
    ranked leaderboard + per-size summary + workbook."""
    ap = argparse.ArgumentParser(description="Exhaustive size-3..6 aug-combination "
                                             "ablation, single seed, one model")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache", default="",
                    help="embeddings_cache.npz (defaults to <data_dir>/embeddings_cache.npz)")
    ap.add_argument("--model", default="1", choices=["1", "2"],
                    help="1 = default_last_pca98 (+casual) @ 20pct; "
                         "2 = tiny_l9_pca95 @ a6")
    ap.add_argument("--seed", type=int, default=42,
                    help="single seed driving BOTH the frozen split and the model "
                         "init (seed 42 + min_duration 30 reproduces the finalised split)")
    ap.add_argument("--sizes", default="3,4,5,6",
                    help="comma list of combination sizes to enumerate (default 3,4,5,6)")
    ap.add_argument("--min_duration", type=float, default=30.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class_balance", default="sampler",
                    choices=["sampler", "pos_weight", "both", "none"])
    args = ap.parse_args()

    spec = model_spec(args.model)
    sizes = sorted({int(s) for s in args.sizes.split(",") if s.strip()})
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")
    arch = ARCH_CONFIG[spec["arch"]]

    log = setup_logging(out_dir / f"log_aug_combo_{spec['tag']}.txt", name="combo")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("EXHAUSTIVE AUG-COMBINATION ABLATION (single seed)")
    log.info("model=%s (%s)  arch=%s layer=%s pca=%s train_only=%s  split=%s",
             args.model, spec["variant"], spec["arch"], spec["layer"], spec["pca"],
             spec["train_only"], "a6" if spec["test_batches"] else "20pct")
    log.info("seed=%d  sizes=%s  min_duration=%.1f  device=%s",
             args.seed, sizes, args.min_duration, device)

    # ---- gt + cache (the model's single layer + all augs) ----
    gt = load_gt_and_filter(data_dir, args.min_duration, log)
    if len(gt) == 0:
        log.error("all rows filtered out"); return 1
    y_full = gt["label"].to_numpy().astype(np.int64)
    filenames_cur = gt["npy_filename"].to_numpy().astype(str)
    gids = gt["group_id"].astype(str).to_numpy() if "group_id" in gt.columns else filenames_cur
    all_augs = resolve_use_augs("all", cache_path, log)
    log.info("augs in cache (%d): %s", len(all_augs), all_augs)
    bad = [k for k in sizes if k > len(all_augs)]
    if bad:
        log.warning("sizes %s exceed #augs (%d) -> skipped", bad, len(all_augs))
        sizes = [k for k in sizes if k <= len(all_augs)]
    if not sizes:
        log.error("no valid combination sizes"); return 1

    layer = spec["layer"]
    wavlm_cache, whisper_cache = load_cache_reindexed(
        cache_path, filenames_cur, ["orig"] + all_augs, [layer], log)
    feat_cols = [c for c in gt.columns if c.startswith("feat_")]
    if feat_cols:
        feat_arr = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
    else:
        feat_arr = None

    def concat(w, wh, f):
        """Stack wavlm | whisper | feat_* into the 1847-d input row block."""
        parts = [w, wh]
        if f is not None:
            parts.append(f)
        return np.concatenate(parts, axis=1).astype(np.float32)

    X_by_aug = {"orig": concat(wavlm_cache[(layer, "orig")], whisper_cache["orig"], feat_arr)}
    for a in all_augs:
        X_by_aug[a] = concat(wavlm_cache[(layer, a)], whisper_cache[a], feat_arr)

    # ---- the ONE frozen, pre-decided split (reused for every combination) ----
    splits = build_splits(gt, train_batches=TRAIN_BATCHES,
                          test_batches=spec["test_batches"], test_region_filter=None,
                          train_only_batches=spec["train_only"], seed=args.seed,
                          log=log, fewshot_frac=0.0)
    assert_no_group_leak(gt, splits)
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    log.info("frozen split: train=%d val=%d test=%d (test candidates=%d)",
             len(train_idx), len(val_idx), len(test_idx),
             len(set(gids[test_idx].tolist())))
    led = {"model": spec["label"], "variant": spec["variant"], "seed": args.seed,
           "split": "a6" if spec["test_batches"] else "20pct",
           "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
           "n_test": int(len(test_idx)),
           "train_candidates": sorted(set(gids[train_idx].tolist())),
           "val_candidates": sorted(set(gids[val_idx].tolist())),
           "test_candidates": sorted(set(gids[test_idx].tolist()))}
    with (out_dir / "splits" / f"{spec['tag']}_seed{args.seed}.json").open("w") as f:
        json.dump(led, f, indent=2)

    # ---- enumerate every size-k combination and score it ----
    combos: list[list[str]] = []
    for k in sizes:
        combos += [list(c) for c in combinations(all_augs, k)]
    log.info("enumerating %d combinations: %s",
             len(combos), {k: math.comb(len(all_augs), k) for k in sizes})

    model_seed = derive_model_seed(args.seed, spec["variant"])  # fixed init (paired)
    rows = []
    for i, augs in enumerate(combos, 1):
        name = cfg_name(augs, all_augs)
        ordered = [a for a in all_augs if a in set(augs)]
        Xtr, ytr, Xva, Xte = build_xy(
            X_by_aug, train_idx, val_idx, test_idx, y_full, ordered,
            spec["pca"], split_seed=args.seed)
        in_dim = Xtr.shape[1]
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
        np.random.seed(model_seed)
        result, _ = train_one(
            Xtr, ytr, Xva, y_full[val_idx], Xte, y_full[test_idx], in_dim=in_dim,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            wd=arch["weight_decay"], device=device, log=log,
            tag=f"{spec['tag']}/{name}", hidden=tuple(arch["hidden"]),
            dropout=arch["dropout"], class_balance=args.class_balance)
        m = result.test_metrics
        rap = m.get("recall_at_precision", {})
        rows.append({
            "rank": 0, "aug_combo": name, "n_augs": len(augs),
            "test_best_f1": round(m["best_f1"]["f1"], 4),
            "test_best_thr": round(m["best_f1"]["threshold"], 3),
            "test_f1@0.5": round(m["thr0.5"]["f1"], 4),
            "test_auc": round(m["auc"], 4), "test_ap": round(m["ap"], 4),
            "val_f1": round(result.best_val_f1, 4),
            "test_recall@p80": round(_rp(rap, "p80"), 4),
            "test_recall@p85": round(_rp(rap, "p85"), 4),
            "test_recall@p90": round(_rp(rap, "p90"), 4),
            "test_recall@p95": round(_rp(rap, "p95"), 4),
            "in_dim": in_dim,
        })
        if i % 20 == 0 or i == len(combos):
            log.info("  ... %d/%d combinations done (last %s: best_f1=%.4f)",
                     i, len(combos), name, m["best_f1"]["f1"])

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / f"aug_combo_{spec['tag']}_raw.csv", index=False)

    # ---- leaderboard: best test F1 first; ties -> fewer augs, then higher auc ----
    lb = (raw.drop(columns=["rank"])
          .sort_values(["test_best_f1", "n_augs", "test_auc"],
                       ascending=[False, True, False]).reset_index(drop=True))
    lb.insert(0, "rank", lb.index + 1)
    lb.to_csv(out_dir / f"aug_combo_{spec['tag']}_leaderboard.csv", index=False)

    # ---- per-size summary: does adding more augs help? ----
    by_size_rows = []
    for k, g in raw.groupby("n_augs", sort=True):
        best = g.loc[g["test_best_f1"].idxmax()]
        by_size_rows.append({
            "n_augs": int(k), "n_combos": int(len(g)),
            "mean_test_best_f1": round(float(g["test_best_f1"].mean()), 4),
            "median_test_best_f1": round(float(g["test_best_f1"].median()), 4),
            "max_test_best_f1": round(float(g["test_best_f1"].max()), 4),
            "best_combo": best["aug_combo"],
            "best_combo_recall@p90": round(float(best["test_recall@p90"]), 4),
        })
    by_size = pd.DataFrame(by_size_rows)
    by_size.to_csv(out_dir / f"aug_combo_{spec['tag']}_by_size.csv", index=False)

    try:
        with pd.ExcelWriter(out_dir / f"aug_combo_{spec['tag']}.xlsx", engine="openpyxl") as xw:
            lb.to_excel(xw, sheet_name="leaderboard", index=False)
            by_size.to_excel(xw, sheet_name="by_size", index=False)
            raw.drop(columns=["rank"]).to_excel(xw, sheet_name="raw", index=False)
        log.info("wrote %s", out_dir / f"aug_combo_{spec['tag']}.xlsx")
    except Exception as e:
        log.warning("could not write xlsx (%s); CSVs saved. pip install openpyxl", e)

    log.info("=" * 70)
    log.info("TOP 15 COMBINATIONS (%s, seed %d):", spec["variant"], args.seed)
    show = ["rank", "aug_combo", "n_augs", "test_best_f1", "test_auc",
            "test_recall@p90", "test_recall@p95", "val_f1"]
    log.info("\n%s", lb[show].head(15).to_string(index=False))
    log.info("PER-SIZE:\n%s", by_size.to_string(index=False))
    top = lb.iloc[0]
    log.info("=" * 70)
    log.info("BEST COMBO: {%s}  (%d augs)  test best-F1=%.4f auc=%.3f r@p90=%.4f",
             top["aug_combo"], int(top["n_augs"]), top["test_best_f1"],
             top["test_auc"], top["test_recall@p90"])
    log.info("NOTE: single seed on a frozen split -- this ranks recipes head-to-head "
             "but is one draw. Re-confirm the winner across seeds with "
             "aug_strategy_20pct.py / aug_ablation.py before finalising.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
