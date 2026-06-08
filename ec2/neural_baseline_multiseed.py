"""Multi-seed architecture comparison -- which variant wins WITHOUT a fluke.

This is a sibling of neural_baseline_train.py with one purpose: run every
variant over N seeds and report mean +/- std per variant, so you pick the best
ARCHITECTURE rather than the best single lucky run. It is a SEPARATE file (not a
flag on the sweep) because it makes a different reproducibility guarantee:

    neural_baseline_train.py / _ablation.py : ONE global seed set once. The RNG
        a variant sees depends on its slot in the sweep, so the only way to
        reproduce a single variant's headline number is to replay the whole
        sweep to that slot. (That is exactly what neural_baseline_ablation.py
        does for the M2 headline.)

    neural_baseline_multiseed.py (this file) : every (variant, seed) is
        SELF-CONTAINED.
          * the split is a deterministic function of `seed` alone
          * the model init / sampler / loader shuffle is a deterministic
            function of `(variant, seed)` -- we reseed torch+numpy right before
            each train_one with model_seed = stable_hash(seed, variant).
        => Re-running with `--seeds <that seed> --variants <that variant>`
           reproduces that one model BIT-FOR-BIT (on CPU; GPU is close), without
           touching any other variant. The model_seed is written into
           metrics.json and per_run.csv so the exact recipe is on record.

How seeds shuffle the split (same `--test_batches` switch as the base script):
    20pct mode (no --test_batches): StratifiedGroupKFold(random_state=seed) over
        ALL train_batches -> a different seed reshuffles EVERY candidate into new
        train/val/test folds. ("for 20 split you can shuffle all of them")
    a6 mode (--test_batches audios6): test = audios6, FIXED for every seed; the
        non-test pool is re-partitioned by StratifiedGroupKFold(random_state=
        seed) so train vs val candidates change per seed while test stays put.
        ("for a6 shuffle train and val set, and keep test set")

Reproducibility ledger (so any model can be re-derived):
    <out_dir>/splits/seed_<seed>.json   per-seed: mode + the exact group_id
        (candidate) membership of train / val / test, and the npy filenames.
    <out_dir>/per_run.csv               one row per (variant, seed): metrics +
        model_seed + split counts. The audit trail of every model file.
    <out_dir>/summary_mean_std.csv      per variant: mean & std across seeds,
        ranked by mean test_best_f1 -> the fluke-proof architecture ranking.
    <out_dir>/multiseed_summary.xlsx    per_run / mean_std / split_manifest.
    With --export_artifacts: <out_dir>/seed_<seed>/<variant>/{model.pt,
        scaler.joblib, pca.joblib, inference_meta.json, predictions.csv,
        metrics.json} -- the downloadable, reproducible model file itself.

Run (5-seed comparison on the a6 protocol, fixed audios6 test):
    python neural_baseline_multiseed.py \\
        --data_dir /home/ubuntu/nn/upload \\
        --out_dir  /home/ubuntu/nn/runs/multiseed_a6 \\
        --cache    /home/ubuntu/nn/upload/embeddings_cache.npz \\
        --train_batches audios2,audios4,audios5,audios6 \\
        --test_batches  audios6 \\
        --seeds 42,43,44,45,46

Run (5-seed comparison on the 20pct protocol, everything reshuffled):
    python neural_baseline_multiseed.py \\
        --data_dir /home/ubuntu/nn/upload \\
        --out_dir  /home/ubuntu/nn/runs/multiseed_20pct \\
        --train_batches audios2,audios4,audios5,audios6 \\
        --seeds 42,43,44,45,46

Reproduce ONE model exactly (e.g. tiny_l9_pca95 at seed 44):
    python neural_baseline_multiseed.py ... --seeds 44 --variants tiny_l9_pca95
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _data_pipeline import (WAVLM_LAYERS, assert_no_group_leak, build_splits,
                            compute_metrics, extract_region_metrics,
                            load_cache_reindexed, load_gt_and_filter,
                            log_split_breakdown, log_variant_prelude,
                            per_client_standardize, per_slice_metrics,
                            resolve_use_augs, setup_logging)
# Reuse the EXACT training logic / arch table from the base sweep so a variant
# here is the same model as in neural_baseline_train.py (only the seeding and
# the multi-seed loop differ).
from neural_baseline_train import (ARCH_CONFIG, GRAD_CLIP_NORM, LABEL_SMOOTHING,
                                   MLP, make_variants, train_one)


# ----------------------------------------------------------------------------
# Per-model seed: stable, order-independent function of (data seed, variant).
# Using a hash (not seed+index) means adding/removing variants from --variants
# never changes any other model's init -- so a single model stays reproducible
# regardless of what else you run alongside it.
# ----------------------------------------------------------------------------

def derive_model_seed(data_seed: int, variant: str) -> int:
    h = hashlib.sha256(f"{data_seed}:{variant}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 0 .. 2**32-1, fits np.random.seed / torch.manual_seed


def _concat(wavlm: np.ndarray, whisper: np.ndarray,
            feats: np.ndarray | None) -> np.ndarray:
    parts = [wavlm, whisper]
    if feats is not None:
        parts.append(feats)
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_per_layer(gt, splits, wavlm_cache, whisper_cache, feat_arr, use_augs,
                    log) -> dict:
    """For each WavLM layer build the aug-expanded train matrix + val/test (orig)
    and the StandardScaler fit on the aug-expanded train. Identical pipeline to
    neural_baseline_train.py, just factored so we can rebuild it per seed (the
    split, hence the scaler, changes every seed)."""
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    y_full = gt["label"].to_numpy().astype(np.int64)
    y_train_base = y_full[train_idx]
    per_layer: dict[str, dict] = {}
    for layer in WAVLM_LAYERS:
        X_orig = _concat(wavlm_cache[(layer, "orig")], whisper_cache["orig"], feat_arr)
        X_by_aug = {"orig": X_orig}
        for a in use_augs:
            X_by_aug[a] = _concat(wavlm_cache[(layer, a)], whisper_cache[a], feat_arr)

        X_train_blocks = [X_orig[train_idx]]
        y_train_blocks = [y_train_base]
        for a in use_augs:
            X_train_blocks.append(X_by_aug[a][train_idx])
            y_train_blocks.append(y_train_base)
        X_train = np.concatenate(X_train_blocks, axis=0).astype(np.float32)
        y_train = np.concatenate(y_train_blocks, axis=0).astype(np.int64)
        X_val = X_orig[val_idx]
        X_test = X_orig[test_idx]

        scaler = StandardScaler().fit(X_train)
        per_layer[layer] = {
            "X_train_s": scaler.transform(X_train).astype(np.float32),
            "y_train": y_train,
            "X_val_s": scaler.transform(X_val).astype(np.float32),
            "X_test_s": scaler.transform(X_test).astype(np.float32),
            "scaler": scaler,
        }
        log.info("[seed-split layer=%s] train=%d (orig=%d + %d augs)  val=%d  test=%d",
                 layer, len(X_train), len(train_idx), len(use_augs),
                 len(X_val), len(X_test))
    return per_layer


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="folder containing gt.csv and audio_npy/")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache", default="",
                    help="path to embeddings_cache.npz. Defaults to <data_dir>/embeddings_cache.npz.")
    ap.add_argument("--train_batches", default="audios2,audios4,audios5,audios6")
    ap.add_argument("--test_batches", default="",
                    help="empty -> 20pct mode (reshuffle ALL candidates per seed). "
                         "Set (e.g. audios6) -> a6 mode (test FIXED, only train/val "
                         "reshuffled per seed).")
    ap.add_argument("--test_region_filter", default="",
                    help="when test_batches set, restrict test to region == this")
    ap.add_argument("--train_only_batches", default="",
                    help="comma-separated batches forced into train only (never "
                         "val/test). EMPTY by default. e.g. 'casual' or '2676,2677'.")
    ap.add_argument("--min_duration", type=float, default=0.0,
                    help="drop rows with duration_sec < this from BOTH train and test")
    ap.add_argument("--use_augs", default="",
                    help="comma-separated aug names from the cache to add to TRAIN "
                         "(val/test always use 'orig'). 'all' = every aug in cache. "
                         "Empty (default) = no augs.")
    ap.add_argument("--use_text_features", default="true",
                    choices=["true", "false"])
    ap.add_argument("--per_client_standardize", default="false",
                    choices=["true", "false"])
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated data seeds. Each seed = one split "
                         "shuffle; per-variant metrics are averaged across them. "
                         "Pass a single seed to reproduce one draw.")
    ap.add_argument("--variants", default="",
                    help="comma-separated variant names to run (e.g. "
                         "tiny_l9_pca95,default_last_pca98). Empty = all 30. "
                         "Restricting this never changes any model's result -- "
                         "each (variant, seed) is independently seeded.")
    ap.add_argument("--export_artifacts", default="false",
                    choices=["true", "false"],
                    help="Save model.pt + scaler.joblib + pca.joblib + "
                         "inference_meta.json per (seed, variant) so the exact "
                         "model file is downloadable and reproducible.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class_balance", default="sampler",
                    choices=["sampler", "pos_weight", "both", "none"])
    args = ap.parse_args()

    use_text_features = (args.use_text_features == "true")
    do_per_client_std = (args.per_client_standardize == "true")
    export_artifacts = (args.export_artifacts == "true")
    train_batches = [b.strip() for b in args.train_batches.split(",") if b.strip()]
    test_batches = [b.strip() for b in args.test_batches.split(",") if b.strip()]
    train_only_batches = [b.strip() for b in args.train_only_batches.split(",") if b.strip()]
    test_region_filter = args.test_region_filter.strip() or None
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    want_variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    mode_tag = "a6" if test_batches else "20pct"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")

    log = setup_logging(out_dir / "log_nn.txt", name="nn")
    log.info("MULTI-SEED architecture comparison")
    log.info("data_dir = %s", data_dir)
    log.info("out_dir  = %s", out_dir)
    log.info("cache    = %s", cache_path)
    log.info("mode     = %s  (test_batches=%s)", mode_tag, test_batches or "<reshuffle all>")
    log.info("seeds    = %s", seeds)
    log.info("use_text_features      = %s", use_text_features)
    log.info("per_client_standardize = %s", do_per_client_std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device   = %s", device)

    # ---- gt + min_duration filter (seed-independent; load once)
    gt = load_gt_and_filter(data_dir, args.min_duration, log)
    if len(gt) == 0:
        log.error("All rows filtered out (check --min_duration / gt labels)")
        return 1

    requested = set(train_batches) | set(test_batches)
    n_unused = int((~gt["batch"].isin(requested)).sum())
    if n_unused:
        log.warning("%d rows are outside train/test args and will be unused", n_unused)
    log.info("label dist : %s", gt["label"].value_counts().to_dict())
    log.info("batch dist : %s", gt["batch"].value_counts().to_dict())

    # ---- augs + cache (seed-independent; load once)
    use_augs = resolve_use_augs(args.use_augs, cache_path, log)
    log.info("use_augs: %s", use_augs)
    aug_names_needed = ["orig"] + use_augs
    filenames_cur = gt["npy_filename"].to_numpy().astype(str)
    wavlm_cache, whisper_cache = load_cache_reindexed(
        cache_path, filenames_cur, aug_names_needed, WAVLM_LAYERS, log)

    feat_cols = [c for c in gt.columns if c.startswith("feat_")]
    if feat_cols and use_text_features:
        feat_block = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feat_arr = feat_block.to_numpy().astype(np.float32)
        log.info("handcrafted feats: %d cols", len(feat_cols))
    else:
        feat_arr = None
        log.info("no feat_* used (use_text_features=%s, n_feat_cols=%d)",
                 use_text_features, len(feat_cols))
    if do_per_client_std:
        feat_arr = per_client_standardize(wavlm_cache, whisper_cache, gt, log,
                                          feat_arr=feat_arr)

    # ---- export-artifact metadata (computed once)
    if export_artifacts:
        exp_wavlm_dim = int(wavlm_cache[(WAVLM_LAYERS[0], "orig")].shape[1])
        exp_whisper_dim = int(whisper_cache["orig"].shape[1])
        try:
            _cmeta = np.load(cache_path, allow_pickle=True)
            exp_wavlm_id = str(_cmeta["wavlm_id"]) if "wavlm_id" in _cmeta else ""
            exp_whisper_id = str(_cmeta["whisper_id"]) if "whisper_id" in _cmeta else ""
        except Exception:
            exp_wavlm_id = exp_whisper_id = ""

    gids_full = (gt["group_id"].astype(str).to_numpy()
                 if "group_id" in gt.columns else filenames_cur)
    y_full = gt["label"].to_numpy().astype(np.int64)

    all_variants = make_variants()
    variants = [v for v in all_variants if not want_variants or v[0] in want_variants]
    if want_variants:
        missing = want_variants - {v[0] for v in all_variants}
        if missing:
            log.error("unknown --variants: %s. valid: %s",
                      sorted(missing), [v[0] for v in all_variants])
            return 1
    log.info("Running %d variant(s) x %d seed(s) = %d models: %s",
             len(variants), len(seeds), len(variants) * len(seeds),
             [v[0] for v in variants])

    per_run_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for seed in seeds:
        log.info("#" * 70)
        log.info("SEED %d  (mode=%s: %s)", seed, mode_tag,
                 "test FIXED, train/val reshuffled" if test_batches
                 else "all candidates reshuffled")

        splits = build_splits(gt, train_batches=train_batches,
                              test_batches=test_batches if test_batches else None,
                              test_region_filter=test_region_filter,
                              train_only_batches=train_only_batches,
                              seed=seed, log=log, fewshot_frac=0.0)
        assert_no_group_leak(gt, splits)
        log_split_breakdown(gt, splits, log)
        train_idx = splits["train"]
        val_idx = splits["val"]
        test_idx = splits["test"]
        y_val = y_full[val_idx]
        y_test = y_full[test_idx]

        # ---- reproducibility ledger: who was train / val / test this seed ----
        tr_c = sorted(set(gids_full[train_idx].tolist()))
        va_c = sorted(set(gids_full[val_idx].tolist()))
        te_c = sorted(set(gids_full[test_idx].tolist()))
        seed_manifest = {
            "seed": seed, "mode": mode_tag,
            "train_batches": train_batches, "test_batches": test_batches,
            "train_only_batches": train_only_batches,
            "n_train_rows": int(len(train_idx)), "n_val_rows": int(len(val_idx)),
            "n_test_rows": int(len(test_idx)),
            "n_train_cand": len(tr_c), "n_val_cand": len(va_c), "n_test_cand": len(te_c),
            "train_candidates": tr_c, "val_candidates": va_c, "test_candidates": te_c,
            "train_files": filenames_cur[train_idx].tolist(),
            "val_files": filenames_cur[val_idx].tolist(),
            "test_files": filenames_cur[test_idx].tolist(),
        }
        with (out_dir / "splits" / f"seed_{seed}.json").open("w") as f:
            json.dump(seed_manifest, f, indent=2)
        manifest_rows.append({k: seed_manifest[k] for k in (
            "seed", "mode", "n_train_rows", "n_val_rows", "n_test_rows",
            "n_train_cand", "n_val_cand", "n_test_cand")})
        log.info("[seed %d] split: train %d rows/%d cand | val %d/%d | test %d/%d",
                 seed, len(train_idx), len(tr_c), len(val_idx), len(va_c),
                 len(test_idx), len(te_c))

        per_layer = build_per_layer(gt, splits, wavlm_cache, whisper_cache,
                                    feat_arr, use_augs, log)

        for variant, arch_name, layer, pca_var in variants:
            arch = ARCH_CONFIG[arch_name]
            model_seed = derive_model_seed(seed, variant)
            log.info("-" * 60)
            log.info("SEED %d  VARIANT %s  (arch=%s layer=%s pca=%s)  model_seed=%d",
                     seed, variant, arch_name, layer, pca_var, model_seed)
            log_variant_prelude(variant, gt, splits, log)
            lp = per_layer[layer]
            X_train_s, y_train = lp["X_train_s"], lp["y_train"]
            X_val_s, X_test_s = lp["X_val_s"], lp["X_test_s"]

            # PCA is deterministic (full SVD); seed only for tie-break parity.
            if pca_var is None:
                Xt, Xv, Xs = X_train_s, X_val_s, X_test_s
                pca = None
            else:
                pca = PCA(n_components=pca_var, svd_solver="full", random_state=model_seed)
                pca.fit(X_train_s)
                Xt = pca.transform(X_train_s).astype(np.float32)
                Xv = pca.transform(X_val_s).astype(np.float32)
                Xs = pca.transform(X_test_s).astype(np.float32)
                log.info("PCA(%.2f) -> %d components", pca_var, Xt.shape[1])
            in_dim = Xt.shape[1]

            # Self-contained init: this (variant, seed) reproduces in isolation.
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)

            result, test_p = train_one(
                Xt, y_train, Xv, y_val, Xs, y_test,
                in_dim=in_dim, batch_size=args.batch_size, epochs=args.epochs,
                lr=args.lr, wd=arch["weight_decay"], device=device, log=log,
                tag=f"s{seed}/{variant}",
                hidden=tuple(arch["hidden"]), dropout=arch["dropout"],
                class_balance=args.class_balance,
                export_model=export_artifacts)

            m = result.test_metrics
            rap = m.get("recall_at_precision", {})
            ind = extract_region_metrics(m, "IND")
            php = extract_region_metrics(m, "PHP")
            log.info("[s%d/%s] best_f1=%.4f auc=%.3f r@p90=%.4f r@p95=%.4f",
                     seed, variant, m["best_f1"]["f1"], m["auc"],
                     rap.get("p90", {}).get("recall", float("nan")),
                     rap.get("p95", {}).get("recall", float("nan")))

            per_run_rows.append({
                "variant": variant, "arch": arch_name, "wavlm_layer": layer,
                "hidden": "x".join(str(h) for h in arch["hidden"]),
                "pca": "full" if pca_var is None else f"{int(pca_var*100)}",
                "seed": seed, "model_seed": model_seed, "mode": mode_tag,
                "n_augs": len(use_augs), "in_dim": in_dim,
                "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "n_train_cand": len(tr_c), "n_val_cand": len(va_c),
                "n_test_cand": len(te_c),
                "val_f1": round(result.best_val_f1, 4),
                "best_epoch": result.best_epoch,
                "test_auc": round(m["auc"], 4), "test_ap": round(m["ap"], 4),
                "test_f1@0.5": round(m["thr0.5"]["f1"], 4),
                "test_best_f1": round(m["best_f1"]["f1"], 4),
                "test_best_thr": round(m["best_f1"]["threshold"], 3),
                "recall@p80": round(rap.get("p80", {}).get("recall", float("nan")), 4),
                "recall@p85": round(rap.get("p85", {}).get("recall", float("nan")), 4),
                "recall@p90": round(rap.get("p90", {}).get("recall", float("nan")), 4),
                "recall@p95": round(rap.get("p95", {}).get("recall", float("nan")), 4),
                "thr@p90": round(rap.get("p90", {}).get("threshold", float("nan")), 3),
                "ind_f1": round(ind["f1"], 4), "ind_r@p90": round(ind["p90"], 4),
                "php_f1": round(php["f1"], 4), "php_r@p90": round(php["p90"], 4),
            })

            # ---- per-model artifacts (predictions + metrics always; weights opt) --
            run_dir = out_dir / f"seed_{seed}" / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            pred_df = gt.iloc[test_idx].copy()
            pred_df["pred_score"] = test_p
            pred_df.to_csv(run_dir / "predictions.csv", index=False)
            with (run_dir / "metrics.json").open("w") as f:
                json.dump({"variant": variant, "arch": arch_name,
                           "wavlm_layer": layer, "pca": pca_var,
                           "seed": seed, "model_seed": model_seed,
                           "mode": mode_tag, "in_dim": in_dim,
                           "augs_used": use_augs,
                           "best_val_f1": result.best_val_f1,
                           "best_epoch": result.best_epoch, "test": m}, f, indent=2)

            if export_artifacts and result.model_state is not None:
                torch.save(result.model_state, run_dir / "model.pt")
                joblib.dump(lp["scaler"], run_dir / "scaler.joblib")
                if pca is not None:
                    joblib.dump(pca, run_dir / "pca.joblib")
                n_feat = len(feat_cols) if (feat_cols and use_text_features) else 0
                with (run_dir / "inference_meta.json").open("w") as f:
                    json.dump({
                        "variant": variant, "arch": arch_name, "wavlm_layer": layer,
                        "hidden": list(arch["hidden"]), "dropout": arch["dropout"],
                        "weight_decay": arch["weight_decay"], "pca": pca_var,
                        "in_dim": in_dim, "use_augs": use_augs,
                        "use_text_features": use_text_features,
                        "feat_cols": feat_cols if (feat_cols and use_text_features) else [],
                        "class_balance": args.class_balance,
                        "label_smoothing": LABEL_SMOOTHING, "grad_clip": GRAD_CLIP_NORM,
                        "seed": seed, "model_seed": model_seed, "mode": mode_tag,
                        "best_epoch": result.best_epoch,
                        "best_val_f1": result.best_val_f1,
                        "best_f1_threshold": m["best_f1"]["threshold"],
                        "wavlm_dim": exp_wavlm_dim, "whisper_dim": exp_whisper_dim,
                        "wavlm_id": exp_wavlm_id, "whisper_id": exp_whisper_id,
                        "target_sr": 16000,
                        "reproduce_cmd": (
                            f"neural_baseline_multiseed.py --seeds {seed} "
                            f"--variants {variant} (+ same data/aug flags)"),
                        "feature_order": (
                            f"concat[ wavlm[{layer}] ({exp_wavlm_dim}) | "
                            f"whisper ({exp_whisper_dim}) | feat_* ({n_feat}) ] "
                            f"-> scaler.transform -> "
                            f"{'pca.transform -> ' if pca is not None else ''}"
                            f"MLP -> sigmoid"),
                    }, f, indent=2)

    # ---- per-run audit trail ----
    per_run = pd.DataFrame(per_run_rows)
    per_run_path = out_dir / "per_run.csv"
    per_run.to_csv(per_run_path, index=False)
    log.info("=" * 70)
    log.info("Wrote %s (%d model runs)", per_run_path, len(per_run))

    # ---- mean +/- std per variant across seeds = fluke-proof ranking ----
    agg_metrics = ["val_f1", "test_auc", "test_ap", "test_f1@0.5",
                   "test_best_f1", "recall@p80", "recall@p85", "recall@p90",
                   "recall@p95"]
    grp = per_run.groupby(["variant", "arch", "wavlm_layer", "hidden", "pca"],
                          sort=False)
    rows = []
    for keys, g in grp:
        row = dict(zip(["variant", "arch", "wavlm_layer", "hidden", "pca"], keys))
        row["n_seeds"] = int(len(g))
        row["seeds"] = ",".join(str(s) for s in sorted(g["seed"].tolist()))
        for col in agg_metrics:
            row[f"{col}_mean"] = round(float(g[col].mean()), 4)
            row[f"{col}_std"] = round(float(g[col].std(ddof=0)), 4)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("test_best_f1_mean", ascending=False)
    summary_path = out_dir / "summary_mean_std.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Wrote %s", summary_path)

    manifest_df = pd.DataFrame(manifest_rows)
    try:
        with pd.ExcelWriter(out_dir / "multiseed_summary.xlsx", engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="mean_std", index=False)
            per_run.to_excel(xw, sheet_name="per_run", index=False)
            manifest_df.to_excel(xw, sheet_name="split_manifest", index=False)
        log.info("Wrote %s", out_dir / "multiseed_summary.xlsx")
    except Exception as e:
        log.warning("could not write multiseed_summary.xlsx (%s); CSVs are saved", e)

    # ---- headline ranking to the log ----
    log.info("=" * 70)
    log.info("ARCHITECTURE RANKING over %d seed(s) %s (mean +/- std):",
             len(seeds), seeds)
    show = ["variant", "n_seeds", "test_best_f1_mean", "test_best_f1_std",
            "recall@p90_mean", "recall@p90_std", "test_auc_mean", "test_auc_std"]
    log.info("\n%s", summary[[c for c in show if c in summary.columns]]
             .to_string(index=False))
    log.info("READING: high *_mean with low *_std = genuinely best architecture "
             "(not a lucky seed). Big *_std = its score is a fluke of the split. "
             "Reproduce any single model with --seeds <seed> --variants <variant>; "
             "its split is in splits/seed_<seed>.json and init seed (model_seed) "
             "in per_run.csv / metrics.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
