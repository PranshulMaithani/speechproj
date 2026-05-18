"""XGBoost head for the per-audio cheating classifier.

Companion to neural_baseline_train.py. Both scripts use _data_pipeline.py
to load gt.csv, apply --min_duration, build the train/val/test split, and
read the embedding cache -- so for a given run with matching CLI args the
two heads see the EXACT same rows and the same WavLM/Whisper/feat_* arrays.

Outputs summary2.csv next to summary1.csv (the NN summary). Each row of
summary2.csv is one variant; columns mirror summary1.csv so you can stack
them for side-by-side comparison on the same data.

Variant matrix:

  Tier A bases (XGBClassifier on a feature subset):
    text_all_xgb           every feat_* column
    text_stylo_xgb         stylometric subset (~15 feats)
    text_top20_xgb         top 20 feat_* by XGB importance on TRAIN
    whisper_xgb            Whisper-medium mean-pool only
    wavlm_xgb              WavLM mean-pool only (per layer)
    everything_xgb         WavLM + Whisper + every feat_*

  Tier B bases (content-invariant feature subsets, XGB):
    text_prosodic_base_xgb  pause/F0/energy/voice-quality + filler/repair/hedge
                            disfluency. Excludes stylometric, formal/AI,
                            perplexity, and content-conditional pause ratios.

  Tier A picks (weighted average of base probabilities, threshold tuned on val):
    tierA_pick1   text_top20 + whisper + wavlm           [0.20, 0.44, 0.36]
    tierA_pick2   text_all   + wavlm                     [0.50, 0.50]
    tierA_pick3   text_top20 + text_stylo + whisper      [0.12, 0.16, 0.72]

  Tier B picks:
    tierB_prosodic_fusion  prosodic_base + whisper + wavlm  [0.34, 0.33, 0.33]

Variants that don't include WavLM run ONCE with wavlm_layer='n/a'. Variants
that do are run once per WavLM layer in {last, 9}, mirroring the NN script.

Class balance flag mirrors the NN script:
    sampler     XGB sample_weight = 1/class_count  (analogue of NN's
                WeightedRandomSampler -- each class has total weight 0.5)
    pos_weight  scale_pos_weight = neg/pos
    both        sample_weight + scale_pos_weight (rarely useful)
    none        natural distribution

Run:
    python xgboost_train.py \\
        --data_dir /path/to/upload \\
        --out_dir  /path/to/results/run1 \\
        --cache    /path/to/embeddings_cache.npz \\
        --train_batches audios2,audios4,audios5,2676,2677 \\
        --test_batches  audios6 \\
        --test_region_filter IND \\
        --train_only_batches 2676,2677 \\
        --min_duration 5.0 \\
        --use_augs noise,pitch,vtlp,combo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from _data_pipeline import (WAVLM_LAYERS, assert_no_group_leak, build_splits,
                            compute_metrics, load_cache_reindexed,
                            load_gt_and_filter, per_slice_metrics,
                            resolve_use_augs, setup_logging, sweep_threshold)


# ----------------------------------------------------------------------------
# Feature subset definitions (names match the new feat_* columns in gt.csv,
# which come from ec2/full_text_features.py and audios6_eval.ipynb).
# ----------------------------------------------------------------------------

FEAT_DISFLUENCY = ["filler_rate", "filler_count", "repetition_rate", "repair_rate",
                   "discourse_marker_rate", "hedge_rate"]
FEAT_STYLOMETRIC = ["ttr", "mattr", "mtld", "complex_word_rate", "avg_word_length",
                    "n_words", "n_unique_words", "avg_sentence_length",
                    "std_sentence_length", "fragment_rate", "n_sentences",
                    "self_ref_rate", "noun_rate", "verb_rate", "adj_rate"]
FEAT_PAUSE = ["pause_mean", "pause_std", "pause_median", "pause_skew",
              "long_pause_rate", "pause_ratio", "n_pauses", "pause_regularity",
              "pause_before_content_ratio", "pause_before_function_ratio",
              "mid_phrase_pause_rate", "words_per_sec", "articulation_rate",
              "initial_pause", "longest_pause"]
FEAT_FORMAL_AI = ["formal_transition_count", "formal_transition_rate",
                  "ai_phrase_count", "ai_phrase_rate"]
FEAT_PROSODIC = ["f0_mean", "f0_std", "f0_range", "f0_skew", "f0_slope",
                 "energy_mean", "energy_std", "speaking_rate_std"]
FEAT_VOICE_Q = ["jitter_local", "shimmer_local", "hnr_mean"]
FEAT_PERPLEXITY = ["mean_perplexity", "burstiness"]
FEAT_SUSPICIOUS = ["suspicious_gap_count", "suspicious_gap_ratio"]

FEAT_ALL_TEXT = (FEAT_DISFLUENCY + FEAT_STYLOMETRIC + FEAT_PAUSE +
                 FEAT_FORMAL_AI + FEAT_PROSODIC + FEAT_VOICE_Q +
                 FEAT_PERPLEXITY + FEAT_SUSPICIOUS)

# Tier B content-invariant subset. Excludes stylometric (word counts, TTR),
# formal/AI phrases, perplexity, suspicious gaps, and content-conditional
# pause ratios -- everything left is acoustic or content-blind disfluency.
FEAT_PROSODIC_BASE = [
    "pause_mean", "pause_std", "pause_median", "pause_skew", "long_pause_rate",
    "pause_ratio", "n_pauses", "pause_regularity", "mid_phrase_pause_rate",
    "words_per_sec", "articulation_rate", "initial_pause", "longest_pause",
    "f0_mean", "f0_std", "f0_range", "f0_skew", "f0_slope",
    "energy_mean", "energy_std", "speaking_rate_std",
    "jitter_local", "shimmer_local", "hnr_mean",
    "filler_rate", "repetition_rate", "repair_rate", "hedge_rate",
]


def _feat_cols(names: list[str], available: set[str]) -> list[str]:
    """Prefix and intersect a feature-name list against gt's feat_* columns."""
    return [f"feat_{n}" for n in names if f"feat_{n}" in available]


# Tier A picks: members reference base-variant names defined in BASE_DEFS below,
# and weights are the published ones from tier A picks (pick1/pick2/pick3 in
# companylaptop/_build_audios6_tier_a.py).
PICK_DEFS: dict[str, dict] = {
    "tierA_pick1": {
        "members": ["text_top20", "whisper", "wavlm"],
        "weights": [0.20, 0.44, 0.36],
        "needs_wavlm": True,
    },
    "tierA_pick2": {
        "members": ["text_all", "wavlm"],
        "weights": [0.50, 0.50],
        "needs_wavlm": True,
    },
    "tierA_pick3": {
        "members": ["text_top20", "text_stylo", "whisper"],
        "weights": [0.12, 0.16, 0.72],
        "needs_wavlm": False,
    },
    "tierB_prosodic_fusion": {
        "members": ["text_prosodic_base", "whisper", "wavlm"],
        "weights": [0.34, 0.33, 0.33],
        "needs_wavlm": True,
    },
}


# ----------------------------------------------------------------------------
# Base-model registry.
# ----------------------------------------------------------------------------

def _make_xgb(n_feats: int, seed: int, scale_pos_weight: float | None) -> xgb.XGBClassifier:
    """Hyperparams copied from companylaptop/_build_audios6_tier_a.py make_xgb."""
    cs = 0.3 if n_feats > 500 else 0.8
    spw = float(scale_pos_weight) if scale_pos_weight is not None else 1.0
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=cs, min_child_weight=3,
        scale_pos_weight=spw,
        eval_metric="logloss", random_state=seed,
        use_label_encoder=False, verbosity=0, n_jobs=-1,
    )


def base_definitions(layer: str, feat_available: set[str], top20_feats: list[str]
                     ) -> dict[str, dict]:
    """Returns {variant_name: {tier, kind, blocks, needs_wavlm}} for the
    given WavLM layer. `blocks` enumerates feature blocks to concat for
    that variant; each block is one of:
        ('wavlm', layer)       -- WavLM mean-pool for the given layer
        ('whisper',)           -- Whisper-medium mean-pool
        ('feat', [col_name…])  -- a list of feat_* column names
    """
    return {
        "text_all_xgb": {
            "tier": "A", "kind": "base", "needs_wavlm": False,
            "blocks": [("feat", _feat_cols(FEAT_ALL_TEXT, feat_available))],
        },
        "text_stylo_xgb": {
            "tier": "A", "kind": "base", "needs_wavlm": False,
            "blocks": [("feat", _feat_cols(FEAT_STYLOMETRIC, feat_available))],
        },
        "text_top20_xgb": {
            "tier": "A", "kind": "base", "needs_wavlm": False,
            "blocks": [("feat", top20_feats)],
        },
        "whisper_xgb": {
            "tier": "A", "kind": "base", "needs_wavlm": False,
            "blocks": [("whisper",)],
        },
        "wavlm_xgb": {
            "tier": "A", "kind": "base", "needs_wavlm": True,
            "blocks": [("wavlm", layer)],
        },
        "everything_xgb": {
            "tier": "A", "kind": "base", "needs_wavlm": True,
            "blocks": [("wavlm", layer), ("whisper",),
                       ("feat", _feat_cols(FEAT_ALL_TEXT, feat_available))],
        },
        "text_prosodic_base_xgb": {
            "tier": "B", "kind": "base", "needs_wavlm": False,
            "blocks": [("feat", _feat_cols(FEAT_PROSODIC_BASE, feat_available))],
        },
    }


# ----------------------------------------------------------------------------
# Feature assembly per row-set.
# ----------------------------------------------------------------------------

def select_block(block: tuple, wavlm_cache: dict, whisper_cache: dict,
                 feat_arr: np.ndarray | None, feat_index: dict[str, int],
                 aug: str) -> np.ndarray:
    """Materialize one block of features for the requested aug."""
    kind = block[0]
    if kind == "wavlm":
        layer = block[1]
        return wavlm_cache[(layer, aug)].astype(np.float32)
    if kind == "whisper":
        return whisper_cache[aug].astype(np.float32)
    if kind == "feat":
        cols = block[1]
        if feat_arr is None or not cols:
            return np.zeros((wavlm_cache[("last", aug)].shape[0], 0), dtype=np.float32)
        col_idx = [feat_index[c] for c in cols]
        return feat_arr[:, col_idx].astype(np.float32)
    raise ValueError(f"unknown block kind: {kind}")


def assemble_variant(blocks: list[tuple], wavlm_cache: dict, whisper_cache: dict,
                     feat_arr: np.ndarray | None, feat_index: dict[str, int],
                     row_idx: np.ndarray, aug: str) -> np.ndarray:
    parts = [select_block(b, wavlm_cache, whisper_cache, feat_arr, feat_index, aug)[row_idx]
             for b in blocks]
    return np.concatenate(parts, axis=1).astype(np.float32)


def aug_expand_train(blocks: list[tuple], wavlm_cache: dict, whisper_cache: dict,
                     feat_arr: np.ndarray | None, feat_index: dict[str, int],
                     train_idx: np.ndarray, y_train_base: np.ndarray,
                     use_augs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build the train matrix from orig + every aug in use_augs."""
    X_blocks = [assemble_variant(blocks, wavlm_cache, whisper_cache, feat_arr,
                                 feat_index, train_idx, "orig")]
    y_blocks = [y_train_base]
    for a in use_augs:
        X_blocks.append(assemble_variant(blocks, wavlm_cache, whisper_cache, feat_arr,
                                         feat_index, train_idx, a))
        y_blocks.append(y_train_base)
    return (np.concatenate(X_blocks, axis=0).astype(np.float32),
            np.concatenate(y_blocks, axis=0).astype(np.int64))


# ----------------------------------------------------------------------------
# Top20 selection (XGB importance on TRAIN, orig only).
# ----------------------------------------------------------------------------

def compute_top20(feat_arr: np.ndarray, feat_cols: list[str],
                  train_idx: np.ndarray, y_train: np.ndarray,
                  seed: int, log) -> list[str]:
    if feat_arr is None or len(feat_cols) == 0:
        return []
    X = feat_arr[train_idx]
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    m = _make_xgb(n_feats=X.shape[1], seed=seed, scale_pos_weight=spw)
    m.fit(X, y_train)
    imp = np.argsort(m.feature_importances_)[::-1][:20]
    top = [feat_cols[i] for i in imp]
    log.info("top20 feat_* by XGB importance (train, orig): %s", top[:10])
    return top


# ----------------------------------------------------------------------------
# Class-balance helper.
# ----------------------------------------------------------------------------

def class_weights_for(y: np.ndarray, mode: str) -> tuple[np.ndarray | None, float | None]:
    """Returns (sample_weight, scale_pos_weight) for the chosen --class_balance."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    spw = neg / max(pos, 1.0)
    if mode == "sampler":
        # per-sample weight = 0.5 / class_count -> total weight 0.5 per class
        w = np.where(y == 1, 0.5 / max(pos, 1.0), 0.5 / max(neg, 1.0)).astype(np.float64)
        return w, None
    if mode == "pos_weight":
        return None, spw
    if mode == "both":
        w = np.where(y == 1, 0.5 / max(pos, 1.0), 0.5 / max(neg, 1.0)).astype(np.float64)
        return w, spw
    if mode == "none":
        return None, None
    raise ValueError(f"unknown --class_balance: {mode}")


# ----------------------------------------------------------------------------
# Per-variant training + scoring.
# ----------------------------------------------------------------------------

def train_one_xgb(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, X_test: np.ndarray,
                  seed: int, class_balance: str, log, tag: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (val_proba, test_proba) for the trained XGB."""
    sw, spw = class_weights_for(y_train, class_balance)
    model = _make_xgb(n_feats=X_train.shape[1], seed=seed, scale_pos_weight=spw)
    log.info("[%s] xgb fit  n_train=%d  in_dim=%d  spw=%s  use_sample_w=%s",
             tag, len(y_train), X_train.shape[1],
             f"{spw:.2f}" if spw is not None else "1.00",
             "yes" if sw is not None else "no")
    model.fit(X_train, y_train, sample_weight=sw)
    val_p = model.predict_proba(X_val)[:, 1]
    test_p = model.predict_proba(X_test)[:, 1]
    return val_p, test_p


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
    ap.add_argument("--train_batches", default="audios2,audios4,audios5,audios6,2676,2677")
    ap.add_argument("--test_batches", default="",
                    help="empty -> 60/20/20 candidate-wise split on train_batches")
    ap.add_argument("--test_region_filter", default="")
    ap.add_argument("--train_only_batches", default="2676,2677",
                    help="batches forced into train; never appear in val/test.")
    ap.add_argument("--min_duration", type=float, default=0.0)
    ap.add_argument("--use_augs", default="",
                    help="comma-separated aug names to add to TRAIN (val/test "
                         "always use 'orig'). 'all' = every aug in cache.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_balance", default="sampler",
                    choices=["sampler", "pos_weight", "both", "none"])
    args = ap.parse_args()

    train_batches = [b.strip() for b in args.train_batches.split(",") if b.strip()]
    test_batches = [b.strip() for b in args.test_batches.split(",") if b.strip()]
    train_only_batches = [b.strip() for b in args.train_only_batches.split(",") if b.strip()]
    test_region_filter = args.test_region_filter.strip() or None

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")

    log = setup_logging(out_dir / "log_xgb.txt", name="xgb")
    log.info("data_dir = %s", data_dir)
    log.info("out_dir  = %s", out_dir)
    log.info("cache    = %s", cache_path)

    np.random.seed(args.seed)

    # ---- Load gt + min_duration filter (identical to NN script via shared helper)
    gt = load_gt_and_filter(data_dir, args.min_duration, log)
    if len(gt) == 0:
        log.error("All rows filtered out (check --min_duration / gt labels)")
        return 1

    requested = set(train_batches) | set(test_batches)
    n_unused = int((~gt["batch"].isin(requested)).sum())
    if n_unused:
        log.warning("%d rows are outside train/test args and will be unused", n_unused)
    log.info("label dist : %s", gt["label"].value_counts().to_dict())
    if "region" in gt.columns and gt["region"].notna().any():
        log.info("region dist: %s", gt["region"].value_counts(dropna=False).to_dict())
    log.info("batch dist : %s", gt["batch"].value_counts().to_dict())

    # ---- Resolve augs against cache
    use_augs = resolve_use_augs(args.use_augs, cache_path, log)
    log.info("use_augs: %s", use_augs)
    aug_names_needed = ["orig"] + use_augs

    # ---- Load cache (both WavLM layers + each aug), aligned to gt order
    filenames_cur = gt["npy_filename"].to_numpy().astype(str)
    wavlm_cache, whisper_cache = load_cache_reindexed(
        cache_path, filenames_cur, aug_names_needed, WAVLM_LAYERS, log)

    # ---- Handcrafted features
    feat_cols = [c for c in gt.columns if c.startswith("feat_")]
    if feat_cols:
        feat_block = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feat_arr = feat_block.to_numpy().astype(np.float32)
        feat_index = {c: i for i, c in enumerate(feat_cols)}
        log.info("handcrafted feats: %d cols", len(feat_cols))
    else:
        feat_arr = None
        feat_index = {}
        log.info("no feat_* columns in gt -- text-based variants will be empty")

    feat_available = set(feat_cols)
    y_full = gt["label"].to_numpy().astype(np.int64)

    # ---- Splits (same builder, same seed -> identical to NN's splits)
    splits = build_splits(gt, train_batches=train_batches,
                          test_batches=test_batches if test_batches else None,
                          test_region_filter=test_region_filter,
                          train_only_batches=train_only_batches,
                          seed=args.seed, log=log)
    assert_no_group_leak(gt, splits)
    for k, idx in splits.items():
        labels = y_full[idx]
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        n_tot = max(len(idx), 1)
        log.info("split %-5s n=%4d  pos=%4d (%.1f%%)  neg=%4d (%.1f%%)  groups=%4d",
                 k, len(idx), n_pos, 100 * n_pos / n_tot, n_neg, 100 * n_neg / n_tot,
                 gt.iloc[idx]["group_id"].nunique())

    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    y_train_base = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]

    # ---- top20 (computed once on the orig train rows, same across both wavlm layers)
    top20_feats = compute_top20(feat_arr, feat_cols, train_idx, y_train_base,
                                args.seed, log) if feat_arr is not None else []

    # ---- Build variant list. Layer-independent variants run once with
    # wavlm_layer="n/a"; layer-dependent variants run once per WavLM layer.
    # base_definitions() is layer-aware so we can pass layer="last" for both
    # paths without affecting the layer-independent ones.
    variant_specs: list[tuple[str, str, dict]] = []   # (variant_label, layer, spec)
    # bases:
    sample_defs = base_definitions("last", feat_available, top20_feats)
    for name, spec in sample_defs.items():
        if spec["needs_wavlm"]:
            for layer in WAVLM_LAYERS:
                lay_tag = layer if layer == "last" else f"l{layer}"
                variant_specs.append((f"{name}_{lay_tag}", layer,
                                      base_definitions(layer, feat_available, top20_feats)[name]))
        else:
            variant_specs.append((name, "n/a", spec))

    # ---- Train all bases first, cache their val/test probabilities so picks
    # (which are weighted averages) can reuse them without retraining.
    base_proba: dict[str, dict[str, np.ndarray]] = {}   # variant_label -> {val, test}
    summary_rows: list[dict] = []

    for variant_label, layer, spec in variant_specs:
        blocks = spec["blocks"]
        n_cols_pre = sum(
            (768 if b[0] == "wavlm" else
             1024 if b[0] == "whisper" else
             len(b[1]) if b[0] == "feat" else 0)
            for b in blocks)
        if n_cols_pre == 0:
            log.warning("[%s] empty feature spec -- skipping", variant_label)
            continue
        log.info("=" * 60)
        log.info("VARIANT %s   tier=%s   kind=%s   wavlm_layer=%s   approx_dim=%d   augs=%s",
                 variant_label, spec["tier"], spec["kind"], layer, n_cols_pre, use_augs)

        X_train, y_train = aug_expand_train(blocks, wavlm_cache, whisper_cache,
                                            feat_arr, feat_index,
                                            train_idx, y_train_base, use_augs)
        X_val = assemble_variant(blocks, wavlm_cache, whisper_cache, feat_arr,
                                 feat_index, val_idx, "orig")
        X_test = assemble_variant(blocks, wavlm_cache, whisper_cache, feat_arr,
                                  feat_index, test_idx, "orig")

        val_p, test_p = train_one_xgb(X_train, y_train, X_val, X_test,
                                      seed=args.seed, class_balance=args.class_balance,
                                      log=log, tag=variant_label)
        base_proba[variant_label] = {"val": val_p, "test": test_p}

        thr, val_f1_at_thr = sweep_threshold(y_val, val_p)
        m = compute_metrics(y_test, test_p)
        log.info("[%s] val_thr=%.2f val_f1@thr=%.3f -> TEST auc=%.3f ap=%.3f best_f1=%.3f@%.2f",
                 variant_label, thr, val_f1_at_thr, m["auc"], m["ap"],
                 m["best_f1"]["f1"], m["best_f1"]["threshold"])

        # persist per-variant artifacts
        var_dir = out_dir / f"xgb_{variant_label}"
        var_dir.mkdir(parents=True, exist_ok=True)
        pred_df = gt.iloc[test_idx].copy()
        pred_df["pred_score"] = test_p
        pred_df.to_csv(var_dir / "predictions.csv", index=False)
        with (var_dir / "metrics.json").open("w") as f:
            json.dump({"variant": variant_label, "tier": spec["tier"],
                       "kind": spec["kind"], "wavlm_layer": layer,
                       "in_dim": int(X_train.shape[1]),
                       "augs_used": use_augs,
                       "train_only_batches": train_only_batches,
                       "min_duration": args.min_duration,
                       "class_balance": args.class_balance,
                       "val_thr_swept": thr, "val_f1_at_thr": val_f1_at_thr,
                       "test": m}, f, indent=2)

        rap = m.get("recall_at_precision", {})
        summary_rows.append({
            "variant": variant_label,
            "tier": spec["tier"],
            "kind": spec["kind"],
            "wavlm_layer": layer,
            "in_dim": int(X_train.shape[1]),
            "n_augs": len(use_augs),
            "val_thr": round(thr, 3),
            "val_f1@thr": round(val_f1_at_thr, 4),
            "test_auc": round(m["auc"], 4),
            "test_ap": round(m["ap"], 4),
            "test_f1@0.5": round(m["thr0.5"]["f1"], 4),
            "test_best_f1": round(m["best_f1"]["f1"], 4),
            "test_best_thr": round(m["best_f1"]["threshold"], 3),
            "test_topk_f1": round(m.get("topk", {}).get("f1", float("nan")), 4),
            "recall@p50": round(rap.get("p50", {}).get("recall", float("nan")), 4),
            "recall@p80": round(rap.get("p80", {}).get("recall", float("nan")), 4),
            "recall@p90": round(rap.get("p90", {}).get("recall", float("nan")), 4),
            "recall@p95": round(rap.get("p95", {}).get("recall", float("nan")), 4),
            "thr@p80": round(rap.get("p80", {}).get("threshold", float("nan")), 3),
            "thr@p90": round(rap.get("p90", {}).get("threshold", float("nan")), 3),
        })

    # ---- Picks: weighted average of base probabilities. For wavlm-dependent
    # picks we emit one row per WavLM layer (using that layer's base scores).
    for pick_name, pick in PICK_DEFS.items():
        layers_to_run = WAVLM_LAYERS if pick["needs_wavlm"] else ["n/a"]
        for layer in layers_to_run:
            # Resolve each member to a concrete variant label that has scores
            # in base_proba. wavlm/everything members are layer-suffixed.
            member_labels: list[str] = []
            for m_name in pick["members"]:
                base_name = f"{m_name}_xgb"
                if base_name in {"wavlm_xgb", "everything_xgb"}:
                    lay_tag = layer if layer == "last" else f"l{layer}"
                    member_labels.append(f"{base_name}_{lay_tag}")
                else:
                    member_labels.append(base_name)
            if not all(lab in base_proba for lab in member_labels):
                miss = [lab for lab in member_labels if lab not in base_proba]
                log.warning("[%s/%s] missing member scores %s -- skipping",
                            pick_name, layer, miss)
                continue
            w = np.array(pick["weights"], dtype=np.float64)
            w = w / w.sum()
            val_fused = sum(w[i] * base_proba[lab]["val"]
                            for i, lab in enumerate(member_labels))
            test_fused = sum(w[i] * base_proba[lab]["test"]
                             for i, lab in enumerate(member_labels))

            variant_label = pick_name if layer == "n/a" else \
                f"{pick_name}_{layer if layer == 'last' else f'l{layer}'}"
            tier = "A" if pick_name.startswith("tierA") else "B"
            log.info("=" * 60)
            log.info("PICK %s   members=%s  weights=%s  layer=%s",
                     variant_label, member_labels, w.tolist(), layer)

            thr, val_f1_at_thr = sweep_threshold(y_val, val_fused)
            m = compute_metrics(y_test, test_fused)
            log.info("[%s] val_thr=%.2f val_f1@thr=%.3f -> TEST auc=%.3f ap=%.3f "
                     "best_f1=%.3f@%.2f", variant_label, thr, val_f1_at_thr,
                     m["auc"], m["ap"], m["best_f1"]["f1"], m["best_f1"]["threshold"])

            var_dir = out_dir / f"xgb_{variant_label}"
            var_dir.mkdir(parents=True, exist_ok=True)
            pred_df = gt.iloc[test_idx].copy()
            pred_df["pred_score"] = test_fused
            pred_df.to_csv(var_dir / "predictions.csv", index=False)
            with (var_dir / "metrics.json").open("w") as f:
                json.dump({"variant": variant_label, "tier": tier,
                           "kind": "pick", "wavlm_layer": layer,
                           "members": member_labels,
                           "weights": w.tolist(),
                           "augs_used": use_augs,
                           "train_only_batches": train_only_batches,
                           "min_duration": args.min_duration,
                           "class_balance": args.class_balance,
                           "val_thr_swept": thr, "val_f1_at_thr": val_f1_at_thr,
                           "test": m}, f, indent=2)

            if "batch" in pred_df.columns:
                m["per_batch"] = per_slice_metrics(y_test, test_fused,
                                                   pred_df["batch"].to_numpy(),
                                                   "batch", log)

            rap = m.get("recall_at_precision", {})
            summary_rows.append({
                "variant": variant_label,
                "tier": tier,
                "kind": "pick",
                "wavlm_layer": layer,
                "in_dim": -1,
                "n_augs": len(use_augs),
                "val_thr": round(thr, 3),
                "val_f1@thr": round(val_f1_at_thr, 4),
                "test_auc": round(m["auc"], 4),
                "test_ap": round(m["ap"], 4),
                "test_f1@0.5": round(m["thr0.5"]["f1"], 4),
                "test_best_f1": round(m["best_f1"]["f1"], 4),
                "test_best_thr": round(m["best_f1"]["threshold"], 3),
                "test_topk_f1": round(m.get("topk", {}).get("f1", float("nan")), 4),
                "recall@p50": round(rap.get("p50", {}).get("recall", float("nan")), 4),
                "recall@p80": round(rap.get("p80", {}).get("recall", float("nan")), 4),
                "recall@p90": round(rap.get("p90", {}).get("recall", float("nan")), 4),
                "recall@p95": round(rap.get("p95", {}).get("recall", float("nan")), 4),
                "thr@p80": round(rap.get("p80", {}).get("threshold", float("nan")), 3),
                "thr@p90": round(rap.get("p90", {}).get("threshold", float("nan")), 3),
            })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary2.csv"
    summary.to_csv(summary_path, index=False)
    log.info("=" * 60)
    log.info("SUMMARY (XGB, summary2.csv):\n%s", summary.to_string(index=False))
    log.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
