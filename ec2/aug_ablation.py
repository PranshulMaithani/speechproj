#!/usr/bin/env python3
"""Augmentation ablation for the 2 finalised models (leave-one-out + none).

Goal: attribute which of the 7 augmentations HELP vs HURT each model, without
running the full 2^7 = 128 subset grid (combinatorial, and drowned in seed
noise on small candidate-wise test sets).

Design (decided with the user):
  * Only the 2 chosen models -- no 30-variant sweep.
        Model 1: default / wavlm 'last' / pca98 / + casual (train-only)
        Model 2: tiny    / wavlm  l9    / pca95 / no casual
  * BOTH split protocols:
        20pct : Mode A 60/20/20 candidate-wise on audios2/4/5/6
        a6    : Mode B, test = all audios6 (no region filter)
  * Aug configs (9): 'all', 'drop_<aug>' for each aug, and 'none'.
        LOO reading: removing an aug and test F1 DROPS  => that aug HELPS.
                     removing an aug and test F1 RISES  => that aug HURTS.
  * SPLIT IS FROZEN. build_splits() uses a single fixed --split_seed for every
    config + seed, so the exact same candidates sit in train/val/test
    everywhere. The aug set is the ONLY thing that differs between configs.
  * Only the TRAINING seed varies (MLP init, sampler order, dropout). We run
    --seeds of them and average, so a true aug effect can be told apart from
    init luck. An aug only "helps/hurts" if |delta vs all| exceeds the seed
    wobble (std).

This is a SEPARATE experiment from the finalised runs: it deliberately reseeds
per training run and does NOT reproduce the headline numbers bit-for-bit. It
reuses train_one / ARCH_CONFIG from neural_baseline_train so the model and
training loop are identical.

Modes (--mode):
  loo     leave-one-out: 'all', 'drop_<aug>' x7, 'none' (9 configs). Cleanest
          per-aug help/hurt signal, lowest selection bias. Default.
  greedy  forward selection: start from none, repeatedly add the aug that most
          improves mean VAL F1 (report test). ~O(n^2) configs, finds a good
          subset + main interactions cheaply. Selection on val, never test.
  both    run loo then greedy.

Outputs (in --out_dir):
  aug_ablation_raw.csv      one row per (model, split, phase, aug_config, seed);
                            includes test_recall@p80/85/90/95 (recall at fixed
                            precision -- the operating points that move between
                            no-aug and all-aug even when f1/auc barely change)
  aug_ablation_summary.csv  LOO only: per (model, split, aug_config) mean/std of
                            test F1 + val F1, delta vs 'all', verdict, AND
                            mean + delta_vs_all for recall@p80/85/90/95
  aug_greedy_path.csv       greedy only: the forward-selection path (step,
                            chosen_aug, selected_set, mean val/test F1)

Run:
  python ec2/aug_ablation.py \
      --data_dir /home/ubuntu/nn/data \
      --cache    /home/ubuntu/nn/data/embeddings_cache.npz \
      --out_dir  /home/ubuntu/nn/runs/aug_ablation \
      --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _data_pipeline import (WAVLM_LAYERS, assert_no_group_leak, build_splits,
                            load_cache_reindexed, load_gt_and_filter,
                            resolve_use_augs, setup_logging)
from neural_baseline_train import ARCH_CONFIG, train_one

# audios2/4/5/6 are the mercer batches both protocols train on.
TRAIN_BATCHES = ["audios2", "audios4", "audios5", "audios6"]


def _l9() -> object:
    """The non-'last' WavLM layer key (e.g. 9) used by model 2 (tiny_l9)."""
    non_last = [l for l in WAVLM_LAYERS if l != "last"]
    if not non_last:
        raise RuntimeError(f"no non-'last' layer in WAVLM_LAYERS={WAVLM_LAYERS}")
    return non_last[0]


def build_xy(X_by_aug: dict, train_idx, val_idx, test_idx, y_full,
             use_augs: list[str], pca_var: float | None, split_seed: int):
    """Expand TRAIN with the given augs (val/test stay orig), fit scaler + PCA
    on the aug-expanded train only, return (Xtr, ytr, Xva, Xte). Deterministic
    in the aug set -- independent of the training seed."""
    X_orig = X_by_aug["orig"]
    X_train_blocks = [X_orig[train_idx]]
    y_train_blocks = [y_full[train_idx]]
    for a in use_augs:
        X_train_blocks.append(X_by_aug[a][train_idx])
        y_train_blocks.append(y_full[train_idx])
    X_train = np.concatenate(X_train_blocks, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train_blocks, axis=0).astype(np.int64)
    X_val = X_orig[val_idx]
    X_test = X_orig[test_idx]

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)
    if pca_var is not None:
        pca = PCA(n_components=pca_var, svd_solver="full", random_state=split_seed)
        pca.fit(Xtr)
        Xtr = pca.transform(Xtr).astype(np.float32)
        Xva = pca.transform(Xva).astype(np.float32)
        Xte = pca.transform(Xte).astype(np.float32)
    return Xtr, y_train, Xva, Xte


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache", default="",
                    help="embeddings_cache.npz (defaults to <data_dir>/embeddings_cache.npz)")
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated TRAINING seeds; split stays fixed across all")
    ap.add_argument("--split_seed", type=int, default=42,
                    help="single fixed seed for build_splits -> identical partition everywhere")
    ap.add_argument("--min_duration", type=float, default=30.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class_balance", default="sampler",
                    choices=["sampler", "pos_weight", "both", "none"])
    ap.add_argument("--mode", default="loo", choices=["loo", "greedy", "both"],
                    help="loo = leave-one-out (9 configs: all, drop_<aug> x7, none); "
                         "greedy = forward selection on val F1 (reports test); "
                         "both = run loo then greedy.")
    ap.add_argument("--greedy_tol", type=float, default=0.0,
                    help="greedy adds an aug only if mean val F1 improves by MORE "
                         "than this (default 0.0 = any improvement).")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")

    log = setup_logging(out_dir / "log_ablation.txt", name="ablation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s  seeds=%s  split_seed=%d  min_duration=%.1f",
             device, seeds, args.split_seed, args.min_duration)

    models = [
        {"label": "m1_default_last_pca98_casual", "arch": "default",
         "layer": "last", "pca": 0.98, "train_only": ["casual"]},
        {"label": "m2_tiny_l9_pca95", "arch": "tiny",
         "layer": _l9(), "pca": 0.95, "train_only": []},
    ]
    split_protocols = [
        {"name": "20pct", "test_batches": None},
        {"name": "a6",    "test_batches": ["audios6"]},
    ]

    raw_rows: list[dict] = []
    path_rows: list[dict] = []  # greedy forward-selection path

    for mspec in models:
        label = mspec["label"]
        arch_name = mspec["arch"]
        layer = mspec["layer"]
        pca_var = mspec["pca"]
        train_only = mspec["train_only"]
        arch = ARCH_CONFIG[arch_name]

        for sp in split_protocols:
            split_name = sp["name"]
            tag0 = f"{label}/{split_name}"
            log.info("=" * 70)
            log.info("MODEL %s  arch=%s layer=%s pca=%s train_only=%s  SPLIT=%s",
                     label, arch_name, layer, pca_var, train_only, split_name)

            # ---- gt + min_duration (fresh per model/split; train_only differs)
            gt = load_gt_and_filter(data_dir, args.min_duration, log)
            if len(gt) == 0:
                log.error("[%s] all rows filtered out", tag0)
                return 1
            y_full = gt["label"].to_numpy().astype(np.int64)
            filenames_cur = gt["npy_filename"].to_numpy().astype(str)

            # ---- augs available in the cache (the 7), validated
            all_augs = resolve_use_augs("all", cache_path, log)
            log.info("[%s] augs in cache: %s", tag0, all_augs)

            # ---- cache for THIS model's single layer + all augs
            aug_names_needed = ["orig"] + all_augs
            wavlm_cache, whisper_cache = load_cache_reindexed(
                cache_path, filenames_cur, aug_names_needed, [layer], log)

            # ---- handcrafted feats (use_text_features always true here)
            feat_cols = [c for c in gt.columns if c.startswith("feat_")]
            if feat_cols:
                feat_block = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                feat_arr = feat_block.to_numpy().astype(np.float32)
            else:
                feat_arr = None

            def concat(w, wh, f):
                parts = [w, wh]
                if f is not None:
                    parts.append(f)
                return np.concatenate(parts, axis=1).astype(np.float32)

            X_by_aug = {"orig": concat(wavlm_cache[(layer, "orig")],
                                       whisper_cache["orig"], feat_arr)}
            for a in all_augs:
                X_by_aug[a] = concat(wavlm_cache[(layer, a)], whisper_cache[a], feat_arr)

            # ---- FROZEN split (fixed split_seed, never varies with train seed)
            splits = build_splits(
                gt, train_batches=TRAIN_BATCHES,
                test_batches=sp["test_batches"],
                test_region_filter=None,
                train_only_batches=train_only,
                seed=args.split_seed, log=log, fewshot_frac=0.0)
            assert_no_group_leak(gt, splits)
            train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
            log.info("[%s] split frozen: train=%d val=%d test=%d",
                     tag0, len(train_idx), len(val_idx), len(test_idx))

            def eval_config(use_augs: list[str], cfg_name: str, phase: str
                            ) -> tuple[float, float, float]:
                """Train one aug config across all seeds (split frozen). Appends a
                per-seed row to raw_rows; returns (mean_val_f1, mean_test_best_f1,
                std_test_best_f1). Features depend only on the aug set, so they're
                built once and reused across seeds."""
                Xtr, ytr, Xva, Xte = build_xy(
                    X_by_aug, train_idx, val_idx, test_idx, y_full,
                    use_augs, pca_var, args.split_seed)
                in_dim = Xtr.shape[1]
                vfs, tfs = [], []
                for seed in seeds:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                    tag = f"{tag0}/{cfg_name}/s{seed}"
                    result, _ = train_one(
                        Xtr, ytr, Xva, y_full[val_idx], Xte, y_full[test_idx],
                        in_dim=in_dim, batch_size=args.batch_size, epochs=args.epochs,
                        lr=args.lr, wd=arch["weight_decay"], device=device, log=log,
                        tag=tag, hidden=tuple(arch["hidden"]), dropout=arch["dropout"],
                        class_balance=args.class_balance)
                    m = result.test_metrics
                    vfs.append(result.best_val_f1)
                    tfs.append(m["best_f1"]["f1"])
                    rap = m.get("recall_at_precision", {})

                    def _rp(k: str) -> float:
                        v = rap.get(k, {})
                        return v.get("recall", float("nan")) if isinstance(v, dict) \
                            else float("nan")

                    raw_rows.append({
                        "model": label, "split": split_name, "phase": phase,
                        "aug_config": cfg_name, "n_augs": len(use_augs),
                        "seed": seed,
                        "val_f1": round(result.best_val_f1, 4),
                        "test_f1@0.5": round(m["thr0.5"]["f1"], 4),
                        "test_best_f1": round(m["best_f1"]["f1"], 4),
                        "test_best_thr": round(m["best_f1"]["threshold"], 3),
                        "test_auc": round(m["auc"], 4),
                        "test_ap": round(m["ap"], 4),
                        # recall at fixed precision -- the operating points that
                        # actually move between no-aug and all-aug even when
                        # f1/auc barely change.
                        "test_recall@p80": round(_rp("p80"), 4),
                        "test_recall@p85": round(_rp("p85"), 4),
                        "test_recall@p90": round(_rp("p90"), 4),
                        "test_recall@p95": round(_rp("p95"), 4),
                        "n_test": m["n"],
                    })
                    log.info("[%s] val_f1=%.4f test_best_f1=%.4f f1@.5=%.4f auc=%.3f",
                             tag, result.best_val_f1, m["best_f1"]["f1"],
                             m["thr0.5"]["f1"], m["auc"])
                return float(np.mean(vfs)), float(np.mean(tfs)), float(np.std(tfs))

            # ---- LOO: all, drop_<aug> x7, none
            if args.mode in ("loo", "both"):
                configs: list[tuple[str, list[str]]] = [("all", list(all_augs))]
                for a in all_augs:
                    configs.append((f"drop_{a}", [x for x in all_augs if x != a]))
                configs.append(("none", []))
                for cfg_name, use_augs in configs:
                    eval_config(use_augs, cfg_name, "loo")

            # ---- Greedy forward selection (select on val, report test)
            if args.mode in ("greedy", "both"):
                selected: list[str] = []
                remaining = list(all_augs)
                best_val, base_test, _ = eval_config([], "greedy_base_none", "greedy")
                log.info("[%s] greedy start (none): val=%.4f test=%.4f",
                         tag0, best_val, base_test)
                step = 0
                while remaining:
                    step += 1
                    trials = []  # (mean_val, mean_test, aug)
                    for a in remaining:
                        cand = selected + [a]
                        cname = "greedy_try__" + "+".join(cand)
                        mv, mt, _ = eval_config(cand, cname, f"greedy_trial_step{step}")
                        trials.append((mv, mt, a))
                    trials.sort(key=lambda t: t[0], reverse=True)
                    bv, bt, ba = trials[0]
                    if bv > best_val + args.greedy_tol:
                        selected.append(ba)
                        remaining.remove(ba)
                        best_val = bv
                        path_rows.append({
                            "model": label, "split": split_name, "step": step,
                            "chosen_aug": ba, "selected_set": "+".join(selected),
                            "mean_val_f1": round(bv, 4),
                            "mean_test_best_f1": round(bt, 4),
                        })
                        log.info("[%s] greedy step %d: +%s  set={%s}  val=%.4f test=%.4f",
                                 tag0, step, ba, "+".join(selected), bv, bt)
                    else:
                        log.info("[%s] greedy stop at step %d: best trial val=%.4f "
                                 "does not beat current %.4f + tol=%.4f",
                                 tag0, step, bv, best_val, args.greedy_tol)
                        break
                log.info("[%s] greedy FINAL selected set: {%s}",
                         tag0, "+".join(selected) if selected else "(none)")

    raw = pd.DataFrame(raw_rows)
    raw_path = out_dir / "aug_ablation_raw.csv"
    raw.to_csv(raw_path, index=False)
    log.info("wrote %s (%d rows)", raw_path, len(raw))

    # ---- LOO summary: mean/std per (model, split, aug_config), delta vs 'all'
    loo_raw = raw[raw["phase"] == "loo"] if "phase" in raw.columns else raw.iloc[0:0]
    if len(loo_raw):
        summary = _summarise(loo_raw)
        sum_path = out_dir / "aug_ablation_summary.csv"
        summary.to_csv(sum_path, index=False)
        log.info("wrote %s", sum_path)
        for (model, split), g in summary.groupby(["model", "split"], sort=False):
            log.info("=" * 70)
            log.info("LOO ABLATION  model=%s  split=%s  (test_best_f1, mean over %d seeds)",
                     model, split, len(seeds))
            g = g.sort_values("delta_test_best_f1")
            cols = ["aug_config", "mean_test_best_f1", "std_test_best_f1",
                    "delta_test_best_f1", "mean_val_f1"]
            for extra in ("mean_test_recall@p90", "delta_test_recall@p90",
                          "mean_test_recall@p95", "delta_test_recall@p95"):
                if extra in g.columns:
                    cols.append(extra)
            cols.append("verdict")
            log.info("\n%s", g[cols].to_string(index=False))
        log.info("=" * 70)
        log.info("LOO READING: 'drop_X' with NEGATIVE delta => removing X hurt => "
                 "X HELPS. POSITIVE delta => removing X helped => X HURTS. Only "
                 "trust deltas whose |.| exceeds the 'all' seed std.")

    # ---- Greedy forward-selection path
    if path_rows:
        gp = pd.DataFrame(path_rows)
        gp_path = out_dir / "aug_greedy_path.csv"
        gp.to_csv(gp_path, index=False)
        log.info("wrote %s", gp_path)
        for (model, split), g in gp.groupby(["model", "split"], sort=False):
            log.info("=" * 70)
            log.info("GREEDY PATH  model=%s  split=%s  (forward selection on val F1)",
                     model, split)
            log.info("\n%s", g[["step", "chosen_aug", "selected_set",
                                 "mean_val_f1", "mean_test_best_f1"]].to_string(index=False))
        log.info("=" * 70)
        log.info("GREEDY READING: each row is an aug ADDED because it improved mean "
                 "val F1. The last selected_set is the chosen subset; its "
                 "mean_test_best_f1 is the held-out estimate (selection was on val).")
    return 0


def _summarise(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split), g in raw.groupby(["model", "split"], sort=False):
        rp_cols = ["test_recall@p80", "test_recall@p85",
                   "test_recall@p90", "test_recall@p95"]
        agg = g.groupby("aug_config", sort=False).agg(
            n_augs=("n_augs", "first"),
            mean_test_best_f1=("test_best_f1", "mean"),
            std_test_best_f1=("test_best_f1", "std"),
            mean_test_f1_05=("test_f1@0.5", "mean"),
            mean_val_f1=("val_f1", "mean"),
            std_val_f1=("val_f1", "std"),
            mean_test_auc=("test_auc", "mean"),
            **{f"mean_{c}": (c, "mean") for c in rp_cols if c in g.columns},
        ).reset_index()
        base = agg.loc[agg["aug_config"] == "all"]
        base_f1 = float(base["mean_test_best_f1"].iloc[0]) if len(base) else float("nan")
        base_std = float(base["std_test_best_f1"].iloc[0]) if len(base) else float("nan")
        base_rp = {c: (float(base[f"mean_{c}"].iloc[0])
                       if (len(base) and f"mean_{c}" in base.columns) else float("nan"))
                   for c in rp_cols}
        for _, r in agg.iterrows():
            delta = r["mean_test_best_f1"] - base_f1
            # significant only if the move beats the baseline seed wobble
            sig = (not np.isnan(base_std)) and abs(delta) > base_std
            if r["aug_config"] in ("all", "none"):
                verdict = "baseline" if r["aug_config"] == "all" else "no-aug floor"
            elif not sig:
                verdict = "~neutral"
            elif delta < 0:
                verdict = "HELPS (removing hurt)"
            else:
                verdict = "HURTS (removing helped)"
            row = {
                "model": model, "split": split,
                "aug_config": r["aug_config"], "n_augs": int(r["n_augs"]),
                "mean_test_best_f1": round(r["mean_test_best_f1"], 4),
                "std_test_best_f1": round(r["std_test_best_f1"], 4),
                "delta_test_best_f1": round(delta, 4),
                "mean_test_f1@0.5": round(r["mean_test_f1_05"], 4),
                "mean_val_f1": round(r["mean_val_f1"], 4),
                "std_val_f1": round(r["std_val_f1"], 4),
                "mean_test_auc": round(r["mean_test_auc"], 4),
                "verdict": verdict,
            }
            # recall@precision means + deltas vs 'all' -- where the no-aug vs
            # all-aug difference actually shows up.
            for c in rp_cols:
                mc = f"mean_{c}"
                if mc in agg.columns:
                    row[mc] = round(r[mc], 4)
                    row[f"delta_{c}"] = round(r[mc] - base_rp[c], 4)
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.exit(main())
