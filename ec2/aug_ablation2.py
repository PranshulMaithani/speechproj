#!/usr/bin/env python3
"""aug_ablation2 -- ablation ANCHORED to the exact original trained models.

Why this exists (vs aug_ablation.py):
  aug_ablation.py reseeds and 5-seed-averages EVERY config, including 'all'.
  Its 'all' is therefore a FRESH estimate -- it does not reproduce the headline
  sheet number (the headline came from one training run whose init RNG depended
  on its slot in the 30-variant sweep). That is exactly the 48->30 gap the user
  saw for M2/a6: same candidates, different training luck.

  aug_ablation2.py adds an ANCHOR: for each model x protocol it LOADS the
  original exported artifacts (model.pt + scaler.joblib + pca.joblib +
  inference_meta.json) and runs real inference on the test rows. That anchor
  row IS the exact headline number (e.g. M2/a6 recall@p90 = 0.48), reproduced
  by predicting with the original weights -- not copied from a CSV.

  It first PROVES the split is the identical candidate partition (the frozen
  split's test group_ids must equal the original predictions.csv group_ids,
  and the reconstructed test probabilities must equal the saved pred_score),
  then runs the same LOO + greedy ablation on that frozen split. Every delta in
  the workbook is read against BOTH the true anchor 'all' and the fresh 5-seed
  'all', so the augmentation effect is separated from training luck.

Split identity (confirmed with the user): original runs used --seed 42 and
--min_duration 30 (defaults), so build_splits(seed=42) reproduces the same
candidates. ALLSTAR (2676/2677) being train_only in the original run does not
change val/test (train_only rows never enter val/test); it only adds rows to
TRAIN, which is baked into the saved weights and so does not affect the anchor.

Outputs (in --out_dir):
  aug_ablation2_anchor.csv   one row per (model, split): EXACT original metrics
                             from load+predict, + split/pred-match verification
  aug_ablation2_raw.csv      one row per (model, split, phase, aug_config, seed)
                             -- the fresh ablation runs on the frozen split
  aug_ablation2_summary.csv  LOO: mean/std per config + delta vs fresh 'all'
                             AND delta vs the ORIGINAL anchor 'all'
  aug_greedy2_path.csv       greedy forward-selection path
  aug_ablation2.xlsx         all of the above as sheets (anchor / loo / greedy / raw)

Run:
  python ec2/aug_ablation2.py \
      --data_dir /home/ubuntu/nn/data \
      --cache    /home/ubuntu/nn/data/embeddings_cache.npz \
      --out_dir  /home/ubuntu/nn/runs/aug_ablation2 \
      --runs_root /home/ubuntu/nn/runs \
      --seeds 42,43,44,45,46 \
      --mode both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from _data_pipeline import (assert_no_group_leak, build_splits, compute_metrics,
                            load_cache_reindexed, load_gt_and_filter,
                            resolve_use_augs, setup_logging)
from neural_baseline_train import ARCH_CONFIG, MLP, train_one
from aug_ablation import TRAIN_BATCHES, _l9, _summarise, build_xy


def _rp(rap: dict, k: str) -> float:
    """recall at fixed precision out of compute_metrics' recall_at_precision."""
    v = rap.get(k, {})
    return v.get("recall", float("nan")) if isinstance(v, dict) else float("nan")


@torch.no_grad()
def load_and_predict(var_dir: Path, X_test_orig: np.ndarray, arch: dict,
                     device: torch.device, log) -> tuple[np.ndarray, dict]:
    """Load the original exported model + scaler + pca and predict on the
    ORIGINAL (un-augmented) test features. Reproduces the headline pred_score
    exactly (BatchNorm in eval mode uses the saved running stats)."""
    meta_path = var_dir / "inference_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    scaler = joblib.load(var_dir / "scaler.joblib")
    Xte = scaler.transform(X_test_orig).astype(np.float32)
    pca_path = var_dir / "pca.joblib"
    if pca_path.exists():
        pca = joblib.load(pca_path)
        Xte = pca.transform(Xte).astype(np.float32)

    in_dim = Xte.shape[1]
    meta_in = meta.get("in_dim")
    if meta_in is not None and int(meta_in) != in_dim:
        raise RuntimeError(
            f"in_dim mismatch loading {var_dir.name}: features give {in_dim} but "
            f"inference_meta.json says {meta_in}. Feature order / cache mismatch.")
    hidden = tuple(meta.get("hidden", arch["hidden"]))
    dropout = float(meta.get("dropout", arch["dropout"]))

    model = MLP(in_dim, hidden=hidden, dropout=dropout).to(device)
    state = torch.load(var_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    out = []
    for i in range(0, len(Xte), 512):
        x = torch.from_numpy(Xte[i:i + 512]).to(device)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    p = np.concatenate(out) if out else np.array([], dtype=np.float32)
    log.info("  loaded %s: hidden=%s dropout=%.2f in_dim=%d  (use_augs=%s)",
             var_dir.name, hidden, dropout, in_dim, meta.get("use_augs", "?"))
    return p, meta


def verify_split(gt: pd.DataFrame, test_idx: np.ndarray, var_dir: Path,
                 test_p: np.ndarray, log) -> dict:
    """Prove the frozen split's test set == the original run's test set, and
    that our reconstructed probabilities == the saved pred_score."""
    res = {"test_groups_match": "NO_FILE", "pred_max_abs_diff": float("nan"),
           "n_test": int(len(test_idx))}
    pred_path = var_dir / "predictions.csv"
    if not pred_path.exists():
        log.warning("  [verify] %s has no predictions.csv -- cannot verify", var_dir)
        return res
    orig = pd.read_csv(pred_path)
    frozen_groups = set(gt.iloc[test_idx]["group_id"].astype(str))
    if "group_id" in orig.columns:
        orig_groups = set(orig["group_id"].astype(str))
        match = (frozen_groups == orig_groups)
        res["test_groups_match"] = "YES" if match else "NO"
        if not match:
            log.error("  [verify] TEST CANDIDATES DIFFER from original! "
                      "frozen=%d orig=%d  only_frozen=%s  only_orig=%s",
                      len(frozen_groups), len(orig_groups),
                      sorted(frozen_groups - orig_groups)[:5],
                      sorted(orig_groups - frozen_groups)[:5])
        else:
            log.info("  [verify] test candidates match original (%d groups)",
                     len(frozen_groups))

    # pred_score match (the real proof the whole load->predict path is faithful)
    score_col = next((c for c in ("pred_score", "pred_prob") if c in orig.columns), None)
    if score_col and "npy_filename" in orig.columns and len(test_p) == len(test_idx):
        cur = pd.DataFrame({
            "npy_filename": gt.iloc[test_idx]["npy_filename"].astype(str).to_numpy(),
            "p_recon": test_p,
        })
        merged = cur.merge(orig[["npy_filename", score_col]].astype({"npy_filename": str}),
                           on="npy_filename", how="inner")
        if len(merged):
            d = float(np.max(np.abs(merged["p_recon"].to_numpy()
                                    - merged[score_col].to_numpy())))
            res["pred_max_abs_diff"] = d
            lvl = log.info if d < 1e-3 else log.warning
            lvl("  [verify] pred_score max|recon-orig| = %.2e over %d matched rows "
                "(<1e-3 => exact reproduction)", d, len(merged))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache", default="",
                    help="embeddings_cache.npz (defaults to <data_dir>/embeddings_cache.npz)")
    ap.add_argument("--runs_root", default="/home/ubuntu/nn/runs",
                    help="folder holding the original run dirs")
    ap.add_argument("--m1_a6_dir", default="m1_casual_a6")
    ap.add_argument("--m1_20pct_dir", default="m1_casual_20pct")
    ap.add_argument("--m2_a6_dir", default="m2_nocasual_a6")
    ap.add_argument("--m2_20pct_dir", default="m2_nocasual_20pct")
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated TRAINING seeds for the fresh ablation rows")
    ap.add_argument("--split_seed", type=int, default=42,
                    help="must match the original run's --seed (default 42) so the "
                         "candidate partition is identical")
    ap.add_argument("--min_duration", type=float, default=30.0,
                    help="must match the original run's --min_duration (default 30)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class_balance", default="sampler",
                    choices=["sampler", "pos_weight", "both", "none"])
    ap.add_argument("--mode", default="both", choices=["loo", "greedy", "both"])
    ap.add_argument("--greedy_tol", type=float, default=0.0)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")
    runs_root = Path(args.runs_root)

    log = setup_logging(out_dir / "log_ablation2.txt", name="ablation2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s seeds=%s split_seed=%d min_duration=%.1f",
             device, seeds, args.split_seed, args.min_duration)

    models = [
        {"label": "m1_default_last_pca98_casual", "arch": "default",
         "layer": "last", "pca": 0.98, "train_only": ["casual"],
         "variant": "default_last_pca98",
         "orig_dirs": {"20pct": args.m1_20pct_dir, "a6": args.m1_a6_dir}},
        {"label": "m2_tiny_l9_pca95", "arch": "tiny",
         "layer": _l9(), "pca": 0.95, "train_only": [],
         "variant": "tiny_l9_pca95",
         "orig_dirs": {"20pct": args.m2_20pct_dir, "a6": args.m2_a6_dir}},
    ]
    split_protocols = [
        {"name": "20pct", "test_batches": None},
        {"name": "a6",    "test_batches": ["audios6"]},
    ]

    anchor_rows: list[dict] = []
    raw_rows: list[dict] = []
    path_rows: list[dict] = []

    for mspec in models:
        label = mspec["label"]
        arch = ARCH_CONFIG[mspec["arch"]]
        layer = mspec["layer"]
        pca_var = mspec["pca"]
        train_only = mspec["train_only"]

        for sp in split_protocols:
            split_name = sp["name"]
            tag0 = f"{label}/{split_name}"
            log.info("=" * 70)
            log.info("MODEL %s  arch=%s layer=%s pca=%s train_only=%s  SPLIT=%s",
                     label, mspec["arch"], layer, pca_var, train_only, split_name)

            gt = load_gt_and_filter(data_dir, args.min_duration, log)
            if len(gt) == 0:
                log.error("[%s] all rows filtered out", tag0)
                return 1
            y_full = gt["label"].to_numpy().astype(np.int64)
            filenames_cur = gt["npy_filename"].to_numpy().astype(str)

            all_augs = resolve_use_augs("all", cache_path, log)
            aug_names_needed = ["orig"] + all_augs
            wavlm_cache, whisper_cache = load_cache_reindexed(
                cache_path, filenames_cur, aug_names_needed, [layer], log)

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

            # ---- FROZEN split: identical candidates to the original run ----
            splits = build_splits(
                gt, train_batches=TRAIN_BATCHES,
                test_batches=sp["test_batches"], test_region_filter=None,
                train_only_batches=train_only, seed=args.split_seed,
                log=log, fewshot_frac=0.0)
            assert_no_group_leak(gt, splits)
            train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
            log.info("[%s] split frozen: train=%d val=%d test=%d",
                     tag0, len(train_idx), len(val_idx), len(test_idx))

            # ---- ANCHOR: load the original model, predict, verify ----
            var_dir = runs_root / mspec["orig_dirs"][split_name] / mspec["variant"]
            y_test = y_full[test_idx]
            if (var_dir / "model.pt").exists():
                X_test_orig = X_by_aug["orig"][test_idx]
                test_p, meta = load_and_predict(var_dir, X_test_orig, arch, device, log)
                vinfo = verify_split(gt, test_idx, var_dir, test_p, log)
                am = compute_metrics(y_test, test_p)
                arap = am.get("recall_at_precision", {})
                anchor_rows.append({
                    "model": label, "split": split_name, "source": "ORIGINAL (load+predict)",
                    "orig_dir": str(var_dir),
                    "orig_use_augs": meta.get("use_augs", "?"),
                    "n_test": am["n"], "n_pos": am["n_pos"], "n_neg": am["n_neg"],
                    "best_f1": round(am["best_f1"]["f1"], 4),
                    "best_thr": round(am["best_f1"]["threshold"], 3),
                    "auc": round(am["auc"], 4), "ap": round(am["ap"], 4),
                    "recall@p80": round(_rp(arap, "p80"), 4),
                    "recall@p85": round(_rp(arap, "p85"), 4),
                    "recall@p90": round(_rp(arap, "p90"), 4),
                    "recall@p95": round(_rp(arap, "p95"), 4),
                    "test_groups_match_orig": vinfo["test_groups_match"],
                    "pred_max_abs_diff": (round(vinfo["pred_max_abs_diff"], 8)
                                          if vinfo["pred_max_abs_diff"] == vinfo["pred_max_abs_diff"]
                                          else float("nan")),
                })
                log.info("[%s] ANCHOR (original): best_f1=%.4f auc=%.3f "
                         "recall@p90=%.4f recall@p95=%.4f",
                         tag0, am["best_f1"]["f1"], am["auc"],
                         _rp(arap, "p90"), _rp(arap, "p95"))
            else:
                log.error("[%s] no model.pt at %s -- skipping anchor. Re-run the "
                          "original command with --export_artifacts true.", tag0, var_dir)
                anchor_rows.append({
                    "model": label, "split": split_name, "source": "MISSING model.pt",
                    "orig_dir": str(var_dir)})

            # ---- fresh ablation on the frozen split (5-seed mean) ----
            def eval_config(use_augs: list[str], cfg_name: str, phase: str
                            ) -> tuple[float, float, float]:
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
                        Xtr, ytr, Xva, y_full[val_idx], Xte, y_test,
                        in_dim=in_dim, batch_size=args.batch_size, epochs=args.epochs,
                        lr=args.lr, wd=arch["weight_decay"], device=device, log=log,
                        tag=tag, hidden=tuple(arch["hidden"]), dropout=arch["dropout"],
                        class_balance=args.class_balance)
                    m = result.test_metrics
                    rap = m.get("recall_at_precision", {})
                    vfs.append(result.best_val_f1)
                    tfs.append(m["best_f1"]["f1"])
                    raw_rows.append({
                        "model": label, "split": split_name, "phase": phase,
                        "aug_config": cfg_name, "n_augs": len(use_augs), "seed": seed,
                        "val_f1": round(result.best_val_f1, 4),
                        "test_f1@0.5": round(m["thr0.5"]["f1"], 4),
                        "test_best_f1": round(m["best_f1"]["f1"], 4),
                        "test_best_thr": round(m["best_f1"]["threshold"], 3),
                        "test_auc": round(m["auc"], 4), "test_ap": round(m["ap"], 4),
                        "test_recall@p80": round(_rp(rap, "p80"), 4),
                        "test_recall@p85": round(_rp(rap, "p85"), 4),
                        "test_recall@p90": round(_rp(rap, "p90"), 4),
                        "test_recall@p95": round(_rp(rap, "p95"), 4),
                        "n_test": m["n"],
                    })
                    log.info("[%s] val_f1=%.4f test_best_f1=%.4f auc=%.3f",
                             tag, result.best_val_f1, m["best_f1"]["f1"], m["auc"])
                return float(np.mean(vfs)), float(np.mean(tfs)), float(np.std(tfs))

            if args.mode in ("loo", "both"):
                configs: list[tuple[str, list[str]]] = [("all", list(all_augs))]
                for a in all_augs:
                    configs.append((f"drop_{a}", [x for x in all_augs if x != a]))
                configs.append(("none", []))
                for cfg_name, use_augs in configs:
                    eval_config(use_augs, cfg_name, "loo")

            if args.mode in ("greedy", "both"):
                selected: list[str] = []
                remaining = list(all_augs)
                best_val, base_test, _ = eval_config([], "greedy_base_none", "greedy")
                log.info("[%s] greedy start (none): val=%.4f test=%.4f",
                         tag0, best_val, base_test)
                step = 0
                while remaining:
                    step += 1
                    trials = []
                    for a in remaining:
                        cand = selected + [a]
                        mv, mt, _ = eval_config(cand, "greedy_try__" + "+".join(cand),
                                                f"greedy_trial_step{step}")
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
                            "mean_test_best_f1": round(bt, 4)})
                        log.info("[%s] greedy step %d: +%s set={%s} val=%.4f test=%.4f",
                                 tag0, step, ba, "+".join(selected), bv, bt)
                    else:
                        log.info("[%s] greedy stop at step %d (best val %.4f <= %.4f+tol)",
                                 tag0, step, bv, best_val)
                        break
                log.info("[%s] greedy FINAL set: {%s}",
                         tag0, "+".join(selected) if selected else "(none)")

    # ---- write CSVs ----
    anchor = pd.DataFrame(anchor_rows)
    anchor.to_csv(out_dir / "aug_ablation2_anchor.csv", index=False)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_dir / "aug_ablation2_raw.csv", index=False)
    log.info("wrote anchor (%d rows) + raw (%d rows)", len(anchor), len(raw))

    summary = pd.DataFrame()
    loo_raw = raw[raw["phase"] == "loo"] if ("phase" in raw.columns and len(raw)) else raw.iloc[0:0]
    if len(loo_raw):
        summary = _summarise(loo_raw)
        summary = _attach_anchor_delta(summary, anchor)
        summary.to_csv(out_dir / "aug_ablation2_summary.csv", index=False)
        log.info("wrote aug_ablation2_summary.csv")
        for (model, split), g in summary.groupby(["model", "split"], sort=False):
            log.info("=" * 70)
            log.info("LOO2  model=%s split=%s  (fresh 5-seed mean vs ORIGINAL anchor)",
                     model, split)
            cols = ["aug_config", "mean_test_best_f1", "delta_test_best_f1",
                    "mean_test_recall@p90", "delta_test_recall@p90"]
            cols += [c for c in ("anchor_recall@p90", "delta_recall@p90_vs_anchor")
                     if c in g.columns]
            log.info("\n%s", g.sort_values("delta_test_best_f1")[
                [c for c in cols if c in g.columns]].to_string(index=False))

    gp = pd.DataFrame(path_rows)
    if len(gp):
        gp.to_csv(out_dir / "aug_greedy2_path.csv", index=False)
        log.info("wrote aug_greedy2_path.csv")

    # ---- workbook ----
    xlsx_path = out_dir / "aug_ablation2.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            anchor.to_excel(xw, sheet_name="anchor_original", index=False)
            if len(summary):
                summary.to_excel(xw, sheet_name="loo_summary", index=False)
            if len(gp):
                gp.to_excel(xw, sheet_name="greedy_path", index=False)
            if len(raw):
                raw.to_excel(xw, sheet_name="raw", index=False)
        log.info("wrote %s", xlsx_path)
    except Exception as e:  # openpyxl missing or write error -- CSVs already saved
        log.warning("could not write xlsx (%s). CSVs are in %s. "
                    "pip install openpyxl to enable the workbook.", e, out_dir)

    log.info("=" * 70)
    log.info("READING: 'anchor_original' = exact headline numbers reproduced by "
             "loading the original weights. loo_summary delta_test_best_f1 is vs "
             "the FRESH 5-seed 'all'; delta_*_vs_anchor is vs the ORIGINAL model. "
             "An aug HELPS if dropping it lowers test (negative delta beyond seed std).")
    return 0


def _attach_anchor_delta(summary: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    """Add the ORIGINAL anchor's best_f1 + recall@p{80,85,90,95} to every row,
    plus delta of the fresh config vs that true anchor."""
    if not len(anchor) or "best_f1" not in anchor.columns:
        return summary
    a = anchor.set_index(["model", "split"])
    rp = ["recall@p80", "recall@p85", "recall@p90", "recall@p95"]
    out = summary.copy()
    for col in ["best_f1"] + rp:
        out[f"anchor_{col}"] = [
            float(a.loc[(m, s), col]) if (m, s) in a.index and col in a.columns
            else float("nan")
            for m, s in zip(out["model"], out["split"])]
    # delta of fresh config vs the original anchor (only meaningful for f1 + rp)
    out["delta_f1_vs_anchor"] = (out["mean_test_best_f1"] - out["anchor_best_f1"]).round(4)
    for k in rp:
        mc = f"mean_test_{k}"
        if mc in out.columns:
            out[f"delta_{k}_vs_anchor"] = (out[mc] - out[f"anchor_{k}"]).round(4)
    return out


if __name__ == "__main__":
    sys.exit(main())
