#!/usr/bin/env python3
"""Augmentation STRATEGY SEARCH for Model 1 on the 20pct protocol.

Question this answers: out of the 8 audio augmentations in the cache
(noise, pitch, speed, gain, air, vtlp, combo, codec), which SUBSET gives the
best-performing Model 1 -- and how confident are we it's not a lucky split?

How this differs from aug_ablation.py / aug_ablation2.py
--------------------------------------------------------
Those two FREEZE the split (one --split_seed) and vary only the training seed,
so they measure the aug effect with split variance held out. Useful, but they
do not tell you how a subset generalises across DIFFERENT splits, and they only
run leave-one-out (good for "which aug helps", not for "what's the best
combination").

This script instead:
  * Focuses on ONE model x ONE protocol: Model 1 = default / wavlm 'last' /
    pca98 / + casual (train-only), on the 20pct protocol (Mode A: 60/20/20
    StratifiedGroupKFold, candidate-disjoint, over audios2/4/5/6).
  * RESHUFFLES THE SPLIT every seed. Each seed draws a fresh 20pct split (new
    train/val/test candidates) -- the honest "average performance" the
    presentation needs. Metrics are mean +/- std over --seeds.
  * Searches the 8-aug space in three phases instead of one LOO pass:
        Phase S  singles : none + each aug alone        -> marginal lift of each aug
        Phase L  LOO      : all  + all_minus_<aug>       -> each aug's value in context
        Phase G  greedy   : forward-select on VAL F1     -> BUILD the best subset
    then a final LEADERBOARD ranks every distinct subset tried by mean test
    best-F1 (+/- std) so you can read off the winner.

Statistical design (so the ranking is trustworthy, not seed luck)
-----------------------------------------------------------------
PAIRED comparison within each seed:
  * the split is a deterministic function of `seed` alone -> every aug config
    sees the IDENTICAL candidates in train/val/test for a given seed.
  * the MLP init / sampler is seeded with derive_model_seed(seed,
    "default_last_pca98") -- the SAME init for every config at that seed.
  => for a fixed seed the ONLY thing that differs between configs is the
     aug-expanded training data. Across the 5 seeds we get 5 independent
     (split, init) draws, so std over seeds is the real spread. A subset only
     "wins" if its mean lead over `none`/`all` exceeds that seed std.

Greedy selects on VALIDATION F1 only (never test) -> no test-set selection
bias; the leaderboard's test column is then an honest held-out estimate.

Memoization: every config is keyed by its frozenset of augs and trained ONCE.
Greedy step 1 reuses the singles, LOO 'all' is reused by greedy, etc., so the
three phases overlap for free and identical subsets always report identical
numbers.

This is a SEPARATE experiment from the finalised runs: it deliberately reseeds
and does NOT reproduce the headline numbers bit-for-bit (those came from one
slot in the 30-variant sweep). It reuses train_one / ARCH_CONFIG / build_xy so
the model and training loop are byte-identical to the real pipeline.

Outputs (in --out_dir)
----------------------
  aug_strategy_raw.csv        one row per (aug_config, seed): val/test F1, auc,
                              ap, recall@p80/85/90/95, split + init seeds
  aug_strategy_singles.csv    Phase S: per-aug marginal lift over 'none'
  aug_strategy_loo.csv        Phase L: per-aug delta vs 'all' + help/hurt verdict
  aug_strategy_greedy.csv     Phase G: the forward-selection path
  aug_strategy_leaderboard.csv  every distinct subset, ranked by mean test best-F1
  aug_strategy.xlsx           all of the above as sheets
  splits/seed_<seed>.json     per-seed candidate membership (reproducibility)

Run (5-seed search, GPU recommended):
  python ec2/aug_strategy_20pct.py \
      --data_dir /home/ubuntu/nn/data \
      --cache    /home/ubuntu/nn/data/embeddings_cache.npz \
      --out_dir  /home/ubuntu/nn/runs/aug_strategy_20pct \
      --seeds 42,43,44,45,46

Reproduce one subset at one seed:
  ... --seeds 44 --only_subset noise,combo   (trains just that config)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _data_pipeline import (assert_no_group_leak, build_splits,
                            load_cache_reindexed, load_gt_and_filter,
                            resolve_use_augs, setup_logging)
from neural_baseline_train import ARCH_CONFIG, train_one
from neural_baseline_multiseed import derive_model_seed
from aug_ablation import TRAIN_BATCHES, build_xy

# Model 1's identity. Kept as a constant so init seeding is config-independent
# (same init for every aug subset at a given data seed -> paired comparison).
MODEL_VARIANT = "default_last_pca98"
_RP_KEYS = ["p80", "p85", "p90", "p95"]


# ----------------------------------------------------------------------------
# Naming.
# ----------------------------------------------------------------------------

def cfg_label(augs, all_augs: list[str]) -> str:
    """Human-readable canonical name for a subset (used as the memo key's label
    and in every CSV). 'none' / 'all' / 'only_<a>' / 'all_minus_<a>' / 'a+b'."""
    s = set(augs)
    if not s:
        return "none"
    if s == set(all_augs):
        return "all"
    if len(s) == 1:
        return f"only_{next(iter(s))}"
    if len(s) == len(all_augs) - 1:
        missing = (set(all_augs) - s).pop()
        return f"all_minus_{missing}"
    return "+".join(a for a in all_augs if a in s)  # stable order


def _rp(rap: dict, k: str) -> float:
    v = rap.get(k, {})
    return v.get("recall", float("nan")) if isinstance(v, dict) else float("nan")


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="folder containing gt.csv (+ feat_* cols) and the cache")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache", default="",
                    help="embeddings_cache.npz (defaults to <data_dir>/embeddings_cache.npz)")
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated seeds. EACH seed reshuffles the 20pct "
                         "split AND seeds init -> metrics are mean +/- std over them.")
    ap.add_argument("--train_batches", default=",".join(TRAIN_BATCHES))
    ap.add_argument("--train_only_batches", default="casual",
                    help="batches forced into TRAIN only (Model 1 used 'casual'). "
                         "Empty to disable. Augs are applied to these rows too.")
    ap.add_argument("--min_duration", type=float, default=30.0,
                    help="drop rows shorter than this (match the finalised run: 30s)")
    # Model 1 architecture knobs (defaults ARE Model 1; exposed for reuse).
    ap.add_argument("--arch", default="default", choices=list(ARCH_CONFIG.keys()))
    ap.add_argument("--wavlm_layer", default="last")
    ap.add_argument("--pca", type=float, default=0.98,
                    help="PCA explained-variance ratio; <=0 disables PCA")
    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class_balance", default="sampler",
                    choices=["sampler", "pos_weight", "both", "none"])
    # search control
    ap.add_argument("--phases", default="singles,loo,greedy",
                    help="comma list from {singles,loo,greedy}. Leaderboard always "
                         "ranks whatever was evaluated.")
    ap.add_argument("--greedy_tol", type=float, default=0.0,
                    help="greedy adds an aug only if mean VAL F1 improves by MORE "
                         "than this (default 0.0 = any improvement).")
    ap.add_argument("--only_subset", default="",
                    help="debug: train ONLY this comma-subset (e.g. 'noise,combo') "
                         "over the seeds and exit. Overrides --phases.")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    phases = {p.strip() for p in args.phases.split(",") if p.strip()}
    train_batches = [b.strip() for b in args.train_batches.split(",") if b.strip()]
    train_only = [b.strip() for b in args.train_only_batches.split(",") if b.strip()]
    pca_var = args.pca if args.pca and args.pca > 0 else None
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")
    arch = ARCH_CONFIG[args.arch]

    log = setup_logging(out_dir / "log_aug_strategy.txt", name="augstrat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("AUGMENTATION STRATEGY SEARCH -- Model 1 / 20pct")
    log.info("device=%s seeds=%s", device, seeds)
    log.info("model: arch=%s wavlm_layer=%s pca=%s  variant=%s",
             args.arch, args.wavlm_layer, pca_var, MODEL_VARIANT)
    log.info("train_batches=%s  train_only=%s  min_duration=%.1f",
             train_batches, train_only, args.min_duration)

    # ---- gt + min_duration (seed-independent; load once)
    gt = load_gt_and_filter(data_dir, args.min_duration, log)
    if len(gt) == 0:
        log.error("all rows filtered out (check --min_duration / gt labels)")
        return 1
    y_full = gt["label"].to_numpy().astype(np.int64)
    filenames_cur = gt["npy_filename"].to_numpy().astype(str)
    gids_full = (gt["group_id"].astype(str).to_numpy()
                 if "group_id" in gt.columns else filenames_cur)
    log.info("label dist: %s   batch dist: %s",
             gt["label"].value_counts().to_dict(), gt["batch"].value_counts().to_dict())

    # ---- augs available in the cache (the 8), validated against it
    all_augs = resolve_use_augs("all", cache_path, log)
    log.info("augs in cache (%d): %s", len(all_augs), all_augs)
    if not all_augs:
        log.error("no augmentations in cache -- nothing to ablate")
        return 1

    # ---- cache for THIS model's single layer + all augs (seed-independent)
    layer = args.wavlm_layer
    aug_names_needed = ["orig"] + all_augs
    wavlm_cache, whisper_cache = load_cache_reindexed(
        cache_path, filenames_cur, aug_names_needed, [layer], log)

    feat_cols = [c for c in gt.columns if c.startswith("feat_")]
    if feat_cols:
        feat_block = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feat_arr = feat_block.to_numpy().astype(np.float32)
        log.info("handcrafted feats: %d cols", len(feat_cols))
    else:
        feat_arr = None
        log.info("no feat_* columns found -- WavLM+Whisper only")

    def concat(w, wh, f):
        parts = [w, wh]
        if f is not None:
            parts.append(f)
        return np.concatenate(parts, axis=1).astype(np.float32)

    # X_by_aug[a] = full-gt-order feature matrix for aug `a` (indexed by split idx).
    X_by_aug = {"orig": concat(wavlm_cache[(layer, "orig")],
                               whisper_cache["orig"], feat_arr)}
    for a in all_augs:
        X_by_aug[a] = concat(wavlm_cache[(layer, a)], whisper_cache[a], feat_arr)

    # ---- precompute the per-seed 20pct splits ONCE; write the reproducibility
    #      ledger. Every aug config at seed S reuses splits_by_seed[S].
    splits_by_seed: dict[int, dict] = {}
    manifest_rows: list[dict] = []
    for seed in seeds:
        sp = build_splits(gt, train_batches=train_batches,
                          test_batches=None,             # None -> 20pct Mode A
                          test_region_filter=None,
                          train_only_batches=train_only,
                          seed=seed, log=log, fewshot_frac=0.0)
        assert_no_group_leak(gt, sp)
        splits_by_seed[seed] = sp
        tr, va, te = sp["train"], sp["val"], sp["test"]
        tr_c = sorted(set(gids_full[tr].tolist()))
        va_c = sorted(set(gids_full[va].tolist()))
        te_c = sorted(set(gids_full[te].tolist()))
        ledger = {
            "seed": seed, "mode": "20pct",
            "train_batches": train_batches, "train_only_batches": train_only,
            "n_train_rows": int(len(tr)), "n_val_rows": int(len(va)),
            "n_test_rows": int(len(te)),
            "n_train_cand": len(tr_c), "n_val_cand": len(va_c), "n_test_cand": len(te_c),
            "train_candidates": tr_c, "val_candidates": va_c, "test_candidates": te_c,
            "test_files": filenames_cur[te].tolist(),
        }
        with (out_dir / "splits" / f"seed_{seed}.json").open("w") as f:
            json.dump(ledger, f, indent=2)
        manifest_rows.append({k: ledger[k] for k in (
            "seed", "mode", "n_train_rows", "n_val_rows", "n_test_rows",
            "n_train_cand", "n_val_cand", "n_test_cand")})
        log.info("[seed %d] split: train %d rows/%d cand | val %d/%d | test %d/%d",
                 seed, len(tr), len(tr_c), len(va), len(va_c), len(te), len(te_c))

    # ---- memoized per-config evaluator ------------------------------------
    # key = frozenset(augs). Value = aggregates dict. Trained ONCE per subset.
    raw_rows: list[dict] = []
    memo: dict[frozenset, dict] = {}

    def eval_config(use_augs: list[str]) -> dict:
        """Train this subset over all seeds (split reshuffled per seed, init paired
        per seed). Returns aggregates; memoized by the aug set so the three phases
        overlap for free."""
        key = frozenset(use_augs)
        if key in memo:
            return memo[key]
        name = cfg_label(use_augs, all_augs)
        ordered = [a for a in all_augs if a in key]  # stable order for logging
        vfs, tfs, aucs, aps = [], [], [], []
        rp_acc = {k: [] for k in _RP_KEYS}
        for seed in seeds:
            sp = splits_by_seed[seed]
            train_idx, val_idx, test_idx = sp["train"], sp["val"], sp["test"]
            Xtr, ytr, Xva, Xte = build_xy(
                X_by_aug, train_idx, val_idx, test_idx, y_full,
                ordered, pca_var, split_seed=seed)
            in_dim = Xtr.shape[1]
            # paired init: SAME init for every config at this seed.
            ms = derive_model_seed(seed, MODEL_VARIANT)
            torch.manual_seed(ms)
            torch.cuda.manual_seed_all(ms)
            np.random.seed(ms)
            tag = f"{name}/s{seed}"
            result, _ = train_one(
                Xtr, ytr, Xva, y_full[val_idx], Xte, y_full[test_idx],
                in_dim=in_dim, batch_size=args.batch_size, epochs=args.epochs,
                lr=args.lr, wd=arch["weight_decay"], device=device, log=log,
                tag=tag, hidden=tuple(arch["hidden"]), dropout=arch["dropout"],
                class_balance=args.class_balance)
            m = result.test_metrics
            rap = m.get("recall_at_precision", {})
            vfs.append(result.best_val_f1)
            tfs.append(m["best_f1"]["f1"])
            aucs.append(m["auc"])
            aps.append(m["ap"])
            for k in _RP_KEYS:
                rp_acc[k].append(_rp(rap, k))
            raw_rows.append({
                "aug_config": name, "n_augs": len(key),
                "augs": "+".join(ordered) if ordered else "",
                "seed": seed, "model_seed": ms, "in_dim": in_dim,
                "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "val_f1": round(result.best_val_f1, 4),
                "best_epoch": result.best_epoch,
                "test_f1@0.5": round(m["thr0.5"]["f1"], 4),
                "test_best_f1": round(m["best_f1"]["f1"], 4),
                "test_best_thr": round(m["best_f1"]["threshold"], 3),
                "test_auc": round(m["auc"], 4), "test_ap": round(m["ap"], 4),
                "test_recall@p80": round(_rp(rap, "p80"), 4),
                "test_recall@p85": round(_rp(rap, "p85"), 4),
                "test_recall@p90": round(_rp(rap, "p90"), 4),
                "test_recall@p95": round(_rp(rap, "p95"), 4),
            })
            log.info("[%s] val_f1=%.4f test_best_f1=%.4f auc=%.3f r@p90=%.4f",
                     tag, result.best_val_f1, m["best_f1"]["f1"], m["auc"],
                     _rp(rap, "p90"))
        agg = {
            "aug_config": name, "augs": ordered, "n_augs": len(key),
            "mean_val_f1": float(np.mean(vfs)), "std_val_f1": float(np.std(vfs)),
            "mean_test_best_f1": float(np.mean(tfs)),
            "std_test_best_f1": float(np.std(tfs)),
            "best_test_best_f1": float(np.max(tfs)),
            "best_seed": int(seeds[int(np.argmax(tfs))]),
            "mean_test_auc": float(np.mean(aucs)),
            "mean_test_ap": float(np.mean(aps)),
            **{f"mean_test_recall@{k}": float(np.nanmean(rp_acc[k])) for k in _RP_KEYS},
        }
        memo[key] = agg
        log.info(">> [%s] n_augs=%d  mean_test_best_f1=%.4f +/- %.4f  mean_val_f1=%.4f",
                 name, len(key), agg["mean_test_best_f1"],
                 agg["std_test_best_f1"], agg["mean_val_f1"])
        return agg

    # ---- debug shortcut: single subset then exit -------------------------
    if args.only_subset.strip():
        sub = [a.strip() for a in args.only_subset.split(",")
               if a.strip() and a.strip() in all_augs]
        log.info("only_subset mode: training {%s} over seeds %s", "+".join(sub), seeds)
        eval_config(sub)
        pd.DataFrame(raw_rows).to_csv(out_dir / "aug_strategy_raw.csv", index=False)
        log.info("wrote aug_strategy_raw.csv; exiting (only_subset).")
        return 0

    # ---- always anchor the two ends so every phase has its baselines ------
    none_agg = eval_config([])
    all_agg = eval_config(list(all_augs))

    # ---- Phase S: singles -------------------------------------------------
    singles_rows: list[dict] = []
    if "singles" in phases:
        log.info("#" * 70)
        log.info("PHASE S -- singles (each aug alone vs none)")
        for a in all_augs:
            agg = eval_config([a])
            singles_rows.append({
                "aug": a,
                "mean_test_best_f1": round(agg["mean_test_best_f1"], 4),
                "std_test_best_f1": round(agg["std_test_best_f1"], 4),
                "lift_vs_none": round(agg["mean_test_best_f1"]
                                      - none_agg["mean_test_best_f1"], 4),
                "mean_val_f1": round(agg["mean_val_f1"], 4),
                "mean_test_recall@p90": round(agg["mean_test_recall@p90"], 4),
                "mean_test_recall@p95": round(agg["mean_test_recall@p95"], 4),
            })

    # ---- Phase L: leave-one-out from all ----------------------------------
    loo_rows: list[dict] = []
    if "loo" in phases:
        log.info("#" * 70)
        log.info("PHASE L -- leave-one-out from all")
        base_f1 = all_agg["mean_test_best_f1"]
        base_std = all_agg["std_test_best_f1"]
        for a in all_augs:
            agg = eval_config([x for x in all_augs if x != a])
            delta = agg["mean_test_best_f1"] - base_f1   # drop_a minus all
            sig = abs(delta) > base_std
            if not sig:
                verdict = "~neutral"
            elif delta < 0:
                verdict = "HELPS (removing hurt)"
            else:
                verdict = "HURTS (removing helped)"
            loo_rows.append({
                "dropped_aug": a, "aug_config": agg["aug_config"],
                "mean_test_best_f1": round(agg["mean_test_best_f1"], 4),
                "std_test_best_f1": round(agg["std_test_best_f1"], 4),
                "delta_vs_all": round(delta, 4), "verdict": verdict,
                "mean_val_f1": round(agg["mean_val_f1"], 4),
                "mean_test_recall@p90": round(agg["mean_test_recall@p90"], 4),
                "delta_recall@p90_vs_all": round(
                    agg["mean_test_recall@p90"] - all_agg["mean_test_recall@p90"], 4),
            })

    # ---- Phase G: greedy forward selection (select on VAL, report test) ---
    greedy_rows: list[dict] = []
    if "greedy" in phases:
        log.info("#" * 70)
        log.info("PHASE G -- greedy forward selection on mean VAL F1")
        selected: list[str] = []
        remaining = list(all_augs)
        best_val = none_agg["mean_val_f1"]
        greedy_rows.append({
            "step": 0, "chosen_aug": "(start=none)", "selected_set": "",
            "mean_val_f1": round(best_val, 4),
            "mean_test_best_f1": round(none_agg["mean_test_best_f1"], 4),
            "delta_val_vs_prev": 0.0})
        step = 0
        while remaining:
            step += 1
            trials = []  # (mean_val, mean_test, aug)
            for a in remaining:
                agg = eval_config(selected + [a])
                trials.append((agg["mean_val_f1"], agg["mean_test_best_f1"], a))
            trials.sort(key=lambda t: t[0], reverse=True)
            bv, bt, ba = trials[0]
            if bv > best_val + args.greedy_tol:
                selected.append(ba)
                remaining.remove(ba)
                greedy_rows.append({
                    "step": step, "chosen_aug": ba,
                    "selected_set": "+".join(selected),
                    "mean_val_f1": round(bv, 4),
                    "mean_test_best_f1": round(bt, 4),
                    "delta_val_vs_prev": round(bv - best_val, 4)})
                log.info("[greedy] step %d: +%s  set={%s}  val=%.4f test=%.4f",
                         step, ba, "+".join(selected), bv, bt)
                best_val = bv
            else:
                log.info("[greedy] stop at step %d: best trial val=%.4f does not "
                         "beat current %.4f + tol=%.4f", step, bv, best_val, args.greedy_tol)
                break
        log.info("[greedy] FINAL selected set: {%s}",
                 "+".join(selected) if selected else "(none)")

    # ---- LEADERBOARD: every distinct subset, ranked by mean test best-F1 --
    lb_rows = []
    none_f1 = none_agg["mean_test_best_f1"]
    for agg in memo.values():
        lb_rows.append({
            "aug_config": agg["aug_config"],
            "augs": "+".join(agg["augs"]) if agg["augs"] else "",
            "n_augs": agg["n_augs"],
            "mean_test_best_f1": round(agg["mean_test_best_f1"], 4),
            "std_test_best_f1": round(agg["std_test_best_f1"], 4),
            "best_test_best_f1": round(agg["best_test_best_f1"], 4),
            "best_seed": agg["best_seed"],
            "lift_vs_none": round(agg["mean_test_best_f1"] - none_f1, 4),
            "mean_val_f1": round(agg["mean_val_f1"], 4),
            "mean_test_auc": round(agg["mean_test_auc"], 4),
            "mean_test_ap": round(agg["mean_test_ap"], 4),
            "mean_test_recall@p80": round(agg["mean_test_recall@p80"], 4),
            "mean_test_recall@p85": round(agg["mean_test_recall@p85"], 4),
            "mean_test_recall@p90": round(agg["mean_test_recall@p90"], 4),
            "mean_test_recall@p95": round(agg["mean_test_recall@p95"], 4),
        })
    # rank: higher mean test best-F1, then lower std, then fewer augs (simpler)
    leaderboard = (pd.DataFrame(lb_rows)
                   .sort_values(["mean_test_best_f1", "std_test_best_f1", "n_augs"],
                                ascending=[False, True, True])
                   .reset_index(drop=True))
    leaderboard.insert(0, "rank", leaderboard.index + 1)

    # ---- write everything --------------------------------------------------
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_dir / "aug_strategy_raw.csv", index=False)
    leaderboard.to_csv(out_dir / "aug_strategy_leaderboard.csv", index=False)
    singles = pd.DataFrame(singles_rows)
    if len(singles):
        singles = singles.sort_values("lift_vs_none", ascending=False).reset_index(drop=True)
        singles.to_csv(out_dir / "aug_strategy_singles.csv", index=False)
    loo = pd.DataFrame(loo_rows)
    if len(loo):
        loo = loo.sort_values("delta_vs_all").reset_index(drop=True)
        loo.to_csv(out_dir / "aug_strategy_loo.csv", index=False)
    greedy = pd.DataFrame(greedy_rows)
    if len(greedy):
        greedy.to_csv(out_dir / "aug_strategy_greedy.csv", index=False)
    manifest = pd.DataFrame(manifest_rows)

    try:
        with pd.ExcelWriter(out_dir / "aug_strategy.xlsx", engine="openpyxl") as xw:
            leaderboard.to_excel(xw, sheet_name="leaderboard", index=False)
            if len(singles):
                singles.to_excel(xw, sheet_name="singles", index=False)
            if len(loo):
                loo.to_excel(xw, sheet_name="loo", index=False)
            if len(greedy):
                greedy.to_excel(xw, sheet_name="greedy_path", index=False)
            raw.to_excel(xw, sheet_name="raw", index=False)
            manifest.to_excel(xw, sheet_name="split_manifest", index=False)
        log.info("wrote %s", out_dir / "aug_strategy.xlsx")
    except Exception as e:
        log.warning("could not write xlsx (%s); CSVs are saved. pip install openpyxl", e)

    # ---- headline to the log ----------------------------------------------
    log.info("=" * 70)
    log.info("LEADERBOARD (top 12 of %d distinct subsets; mean over %d seeds):",
             len(leaderboard), len(seeds))
    show = ["rank", "aug_config", "n_augs", "mean_test_best_f1", "std_test_best_f1",
            "lift_vs_none", "mean_val_f1", "mean_test_recall@p90", "mean_test_recall@p95"]
    log.info("\n%s", leaderboard[show].head(12).to_string(index=False))
    top = leaderboard.iloc[0]
    log.info("=" * 70)
    log.info("BEST SUBSET: {%s}  ->  mean test best-F1 = %.4f +/- %.4f  "
             "(none=%.4f, all=%.4f)", top["augs"] or "none",
             top["mean_test_best_f1"], top["std_test_best_f1"],
             none_f1, all_agg["mean_test_best_f1"])
    log.info("READING: pick the highest mean_test_best_f1 whose lead over 'none'/'all' "
             "exceeds std_test_best_f1 (otherwise it's seed noise -- prefer the "
             "simpler subset). singles.csv = each aug's marginal lift; loo.csv = "
             "value-in-context (drop_X negative delta => X HELPS); greedy_path.csv = "
             "the val-selected build of the best subset (test column is held out).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
