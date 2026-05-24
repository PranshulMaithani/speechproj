"""XGBoost head for the per-audio cheating classifier -- v2 (post-diagnostic).

================================================================================
V2 IMPROVEMENTS over xgboost_train.py
================================================================================
Three changes triggered by domain_diagnose.py findings:

  1. --drop_high_ks_features TRUE/FALSE  (default FALSE)
       Drops 18 feat_* columns whose KS statistic between client A and B is
       > 0.13. List = HIGH_KS_FEATURES below. From domain_gap_report.txt §5.

  2. --prior_correction TRUE/FALSE + --target_prior FLOAT  (defaults FALSE / -1)
       Applies Saerens-Latinne-Decaestecker correction to val_p and test_p
       before sweep_threshold + compute_metrics. Adjusts pi_train -> pi_target
       calibration. Monotonic -> test_best_f1 is invariant; what changes is
       test_f1@0.5 and the optimal threshold value. pi_target=-1.0 means
       'estimate from test labels' (oracle; research only).

  3. Retuned tierA_pick1 weights: [0.20, 0.44, 0.36] -> [0.50, 0.30, 0.20].
       Text gets MORE weight (the only domain-invariant feature group per
       diagnostic §3); WavLM/Whisper get LESS (domain AUC 0.99 each, i.e. they
       carry mostly client-id, not cheating signal). tierA_pick2 and pick3
       unchanged; only pick1 is retuned.

================================================================================
EXACT RUN COMMAND -- R6 (drop + prior + retuned-tier-A + fewshot 10%)
================================================================================

    python ec2/xgboost_train_v2.py \\
        --data_dir <UPLOAD> \\
        --out_dir  <RUNS>/r6_drop_prior_fewshot10 \\
        --train_batches audios2,audios4,audios5,audios6 \\
        --train_only_batches "" \\
        --test_batches  audios6 \\
        --test_region_filter IND \\
        --min_duration 5.0 \\
        --use_text_features true \\
        --per_client_standardize false \\
        --fewshot_frac 0.10 \\
        --drop_high_ks_features true \\
        --prior_correction true \\
        --target_prior -1.0

    # Run NN v2 in parallel (separate folder is fine; both write into out_dir):
    # python ec2/neural_baseline_train_v2.py [same flags]

Backfill missing feat_n_words_asr (one-time, before R6):

    python ec2/backfill_n_words_asr.py \\
        --data_dir <UPLOAD> \\
        --transcripts <UPLOAD>/transcripts.json \\
        --batches audios6

================================================================================
Original v1 docstring follows.
================================================================================

Companion to neural_baseline_train.py. Both use _data_pipeline.py for the
same gt filter, splits, cache loader, and metrics, so summary1.csv (NN) and
summary2.csv (XGB) are comparable apples-to-apples on the same rows.

Variants are Tier A only (XGB on different feature subsets, plus three
weighted-average picks). Tier B was removed -- Tier A consistently won in
prior runs.

  Tier A bases (XGBClassifier on the specified feature subset):
    text_all_xgb     every feat_* column
    text_stylo_xgb   stylometric subset (~15 feats)
    text_top20_xgb   top 20 feat_* by XGB importance on TRAIN (orig only)
    whisper_xgb      Whisper-medium mean-pool only (1024d)
    wavlm_xgb        WavLM mean-pool only (768d, per layer)
    everything_xgb   WavLM + Whisper [+ all feat_*]   (per layer)

  Tier A picks (weighted-average of base probabilities, threshold tuned on val):
    tierA_pick1   text_top20 + whisper + wavlm           [0.20, 0.44, 0.36]
    tierA_pick2   text_all   + wavlm                     [0.50, 0.50]
    tierA_pick3   text_top20 + text_stylo + whisper      [0.12, 0.16, 0.72]

Layer-independent variants (no WavLM) run once with wavlm_layer='n/a'.
WavLM-using variants run once per layer in {last, 9}.

--use_text_features (default true): when false, feat_* is excluded from
every variant. Text-only bases (text_all/text_stylo/text_top20) are then
skipped. 'everything' becomes wavlm+whisper. Picks drop their text members
and renormalize over surviving members; a pick with <2 surviving members
is skipped (it would just duplicate a base row).

Class balance flag mirrors the NN script:
    sampler     XGB sample_weight = 0.5/class_count  (each class total
                weight = 0.5; mirrors NN's WeightedRandomSampler)
    pos_weight  scale_pos_weight = neg/pos
    both        sample_weight + scale_pos_weight (rare)
    none        natural distribution

Run:
    python xgboost_train.py \\
        --data_dir /path/to/upload \\
        --out_dir  /path/to/results/run1 \\
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
                            compute_metrics, extract_region_metrics,
                            load_cache_reindexed, load_gt_and_filter,
                            log_split_breakdown, log_variant_prelude,
                            per_client_standardize, per_slice_metrics,
                            resolve_use_augs, setup_logging, sweep_threshold)


# ----------------------------------------------------------------------------
# V2 ADDITIONS: high-KS feature list + prior-correction helper.
# Source: domain_gap_report.txt section 5 (KS > 0.13 between client A and B).
# ----------------------------------------------------------------------------

HIGH_KS_FEATURES: list[str] = [
    "feat_shimmer_local", "feat_speaking_rate_std", "feat_duration_sec",
    "feat_pause_ratio", "feat_initial_pause", "feat_f0_skew",
    "feat_longest_pause", "feat_adj_rate", "feat_discourse_marker_rate",
    "feat_burstiness", "feat_f0_range", "feat_noun_rate", "feat_ttr",
    "feat_pause_std", "feat_avg_word_length", "feat_avg_sentence_length",
    "feat_f0_mean", "feat_pause_mean", "feat_n_words_asr",
]


def _str_to_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "t", "yes", "y")


def drop_high_ks_features(gt: pd.DataFrame, log) -> pd.DataFrame:
    dropped = [c for c in HIGH_KS_FEATURES if c in gt.columns]
    if dropped:
        gt = gt.drop(columns=dropped)
        log.info("[v2] dropped %d high-KS feat_* columns: %s",
                 len(dropped), dropped)
    else:
        log.info("[v2] --drop_high_ks_features=true but no HIGH_KS_FEATURES "
                 "were present in gt")
    return gt


def apply_prior_correction(p: np.ndarray, pi_train: float,
                            pi_target: float) -> np.ndarray:
    """Saerens-Latinne-Decaestecker correction (monotonic; AUC/AP/best_f1
    invariant). Rescales P_train(y=1|x) to P_target(y=1|x)."""
    pi_train = float(np.clip(pi_train, 1e-4, 1 - 1e-4))
    pi_target = float(np.clip(pi_target, 1e-4, 1 - 1e-4))
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    num = p * (pi_target / pi_train)
    den = num + (1.0 - p) * ((1.0 - pi_target) / (1.0 - pi_train))
    return (num / den).astype(np.float32)


# ----------------------------------------------------------------------------
# Feature subset names (without the feat_ prefix). Match the audios6_eval /
# full_text_features.py keys; mapping to columns in gt.csv happens at runtime
# via _feat_cols().
# ----------------------------------------------------------------------------

FEAT_STYLOMETRIC = ["ttr", "mattr", "mtld", "complex_word_rate", "avg_word_length",
                    "n_words", "n_unique_words", "avg_sentence_length",
                    "std_sentence_length", "fragment_rate", "n_sentences",
                    "self_ref_rate", "noun_rate", "verb_rate", "adj_rate"]


def _feat_cols(names: list[str], available: set[str]) -> list[str]:
    """Prefix names with 'feat_' and keep only those present in gt.csv."""
    return [f"feat_{n}" for n in names if f"feat_{n}" in available]


# Pick definitions reference base member names (without the _xgb suffix).
PICK_DEFS: dict[str, dict] = {
    "tierA_pick1": {
        # V2: retuned from [0.20, 0.44, 0.36] -> [0.50, 0.30, 0.20].
        # Text gets the most weight (only domain-invariant feature group per
        # domain_diagnose.py §3); WavLM/Whisper are down-weighted because their
        # domain AUC is ~0.99 (they encode client-id more than cheating signal).
        "members": ["text_top20", "whisper", "wavlm"],
        "weights": [0.50, 0.30, 0.20],
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
}


# ----------------------------------------------------------------------------
# XGB factory + class-balance helper.
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


def class_weights_for(y: np.ndarray, mode: str) -> tuple[np.ndarray | None, float | None]:
    """Returns (sample_weight, scale_pos_weight) for the chosen --class_balance.

    For 'sampler', uses the sklearn 'balanced' formulation:
        w_pos = N / (2 * pos),   w_neg = N / (2 * neg)
    Each class contributes total weight N/2 (so minibatches are class-balanced
    in expectation), and per-sample weights AVERAGE 1.0. The averaging detail
    matters: an earlier version used 0.5 / class_count, which is correct for
    class totals but sub-unit per-sample. XGB has min_child_weight=3 in
    _make_xgb -- with sub-unit weights, no leaf can ever accumulate 3 units of
    hessian, so no splits happen, every tree stays at the base score, and AUC
    sits at 0.5 across every variant in every mode.
    """
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    n = float(len(y))
    spw = neg / max(pos, 1.0)
    if mode in ("sampler", "both"):
        w_pos = n / (2.0 * max(pos, 1.0))
        w_neg = n / (2.0 * max(neg, 1.0))
        w = np.where(y == 1, w_pos, w_neg).astype(np.float64)
        return (w, None) if mode == "sampler" else (w, spw)
    if mode == "pos_weight":
        return None, spw
    if mode == "none":
        return None, None
    raise ValueError(f"unknown --class_balance: {mode}")


def train_one_xgb(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, X_test: np.ndarray,
                  seed: int, class_balance: str, log, tag: str
                  ) -> tuple[np.ndarray, np.ndarray]:
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
# Per-layer feature matrices.
# ----------------------------------------------------------------------------

class LayerMatrices:
    """For a given WavLM layer, holds (aug -> full-row matrix) plus the
    column slices for wavlm/whisper/feat. Variants pick columns out of these
    matrices; aug expansion happens at row level for the train set.

    Column layout (left to right):
        [0 : wavlm_dim)                        wavlm
        [wavlm_dim : wavlm_dim+whisper_dim)    whisper
        [...                       : ...)      feat (only if use_text_features)
    """

    def __init__(self, layer: str, wavlm_cache: dict, whisper_cache: dict,
                 feat_arr: np.ndarray | None, feat_cols: list[str],
                 use_augs: list[str]) -> None:
        self.layer = layer
        self.use_feat = feat_arr is not None
        wd = wavlm_cache[(layer, "orig")].shape[1]
        sd = whisper_cache["orig"].shape[1]
        fd = feat_arr.shape[1] if self.use_feat else 0
        self.wavlm_slice = slice(0, wd)
        self.whisper_slice = slice(wd, wd + sd)
        self.feat_slice = slice(wd + sd, wd + sd + fd) if self.use_feat else None
        self.feat_cols = feat_cols if self.use_feat else []

        def _concat(aug: str) -> np.ndarray:
            parts = [wavlm_cache[(layer, aug)], whisper_cache[aug]]
            if self.use_feat:
                parts.append(feat_arr)
            return np.concatenate(parts, axis=1).astype(np.float32)

        self.X_orig = _concat("orig")
        self.X_augs = {a: _concat(a) for a in use_augs}
        self.use_augs = list(use_augs)

    def feat_col_index(self, feat_col_name: str) -> int:
        """Returns the global column index for a 'feat_<name>' column."""
        assert self.use_feat
        local = self.feat_cols.index(feat_col_name)
        return self.feat_slice.start + local


def build_block_cols(blocks: list[tuple], lm: LayerMatrices) -> np.ndarray:
    """Translate a variant's block-spec to an int array of column indices
    into LayerMatrices.X_orig."""
    parts: list[np.ndarray] = []
    for b in blocks:
        kind = b[0]
        if kind == "wavlm":
            parts.append(np.arange(lm.wavlm_slice.start, lm.wavlm_slice.stop))
        elif kind == "whisper":
            parts.append(np.arange(lm.whisper_slice.start, lm.whisper_slice.stop))
        elif kind == "feat_all":
            if lm.use_feat:
                parts.append(np.arange(lm.feat_slice.start, lm.feat_slice.stop))
        elif kind == "feat_subset":
            if lm.use_feat:
                wanted = [c for c in b[1] if c in lm.feat_cols]
                if wanted:
                    parts.append(np.array([lm.feat_col_index(c) for c in wanted]))
        else:
            raise ValueError(f"unknown block kind: {kind}")
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


def materialize(lm: LayerMatrices, col_idx: np.ndarray, train_idx: np.ndarray,
                val_idx: np.ndarray, test_idx: np.ndarray,
                y_train_base: np.ndarray, expand_train_with_augs: bool
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, y_train, X_val, X_test) with the requested columns.

    expand_train_with_augs=False is for text-only variants: feat_* doesn't
    change with aug, so aug-expanding just duplicates rows and overweights
    them in XGB. WavLM/Whisper-bearing variants pass True.
    """
    X_val = lm.X_orig[val_idx][:, col_idx]
    X_test = lm.X_orig[test_idx][:, col_idx]
    if not expand_train_with_augs or not lm.use_augs:
        return (lm.X_orig[train_idx][:, col_idx], y_train_base, X_val, X_test)
    X_train_blocks = [lm.X_orig[train_idx][:, col_idx]]
    y_train_blocks = [y_train_base]
    for a in lm.use_augs:
        X_train_blocks.append(lm.X_augs[a][train_idx][:, col_idx])
        y_train_blocks.append(y_train_base)
    return (np.concatenate(X_train_blocks, axis=0).astype(np.float32),
            np.concatenate(y_train_blocks, axis=0).astype(np.int64),
            X_val, X_test)


# ----------------------------------------------------------------------------
# Top-20 selection (XGB importance on TRAIN orig).
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
# Variant block specs.
# ----------------------------------------------------------------------------

def base_specs(top20_feats: list[str], stylo_feats: list[str],
               use_text_features: bool) -> dict[str, dict]:
    """Returns {bare_name: {needs_wavlm, needs_text, expand_train, blocks}}.
    Bare names are without the _xgb suffix; the layer suffix is added later
    where needed.
    """
    specs: dict[str, dict] = {
        "whisper": {
            "needs_wavlm": False, "needs_text": False,
            "expand_train": True,
            "blocks": [("whisper",)],
        },
        "wavlm": {
            "needs_wavlm": True, "needs_text": False,
            "expand_train": True,
            "blocks": [("wavlm",)],
        },
        "everything": {
            "needs_wavlm": True, "needs_text": False,
            "expand_train": True,
            "blocks": [("wavlm",), ("whisper",)] +
                      ([("feat_all",)] if use_text_features else []),
        },
    }
    if use_text_features:
        specs.update({
            "text_all": {
                "needs_wavlm": False, "needs_text": True,
                "expand_train": False,
                "blocks": [("feat_all",)],
            },
            "text_stylo": {
                "needs_wavlm": False, "needs_text": True,
                "expand_train": False,
                "blocks": [("feat_subset", stylo_feats)],
            },
            "text_top20": {
                "needs_wavlm": False, "needs_text": True,
                "expand_train": False,
                "blocks": [("feat_subset", top20_feats)],
            },
        })
    return specs


def expand_variants(specs: dict[str, dict]) -> list[tuple[str, str, str, dict]]:
    """Returns [(variant_label, base_name, layer, spec)]. base_name carries
    no layer suffix; variant_label includes layer suffix where needed."""
    out: list[tuple[str, str, str, dict]] = []
    for name, spec in specs.items():
        if spec["needs_wavlm"]:
            for layer in WAVLM_LAYERS:
                tag = layer if layer == "last" else f"l{layer}"
                out.append((f"{name}_xgb_{tag}", name, layer, spec))
        else:
            out.append((f"{name}_xgb", name, "n/a", spec))
    return out


# ----------------------------------------------------------------------------
# Summary-row builder (shared between bases and picks).
# ----------------------------------------------------------------------------

def _summary_row(variant_label: str, tier: str, kind: str, layer: str,
                 in_dim: int, n_augs: int, thr: float, val_f1_at_thr: float,
                 m: dict) -> dict:
    rap = m.get("recall_at_precision", {})
    ind = extract_region_metrics(m, "IND")
    php = extract_region_metrics(m, "PHP")
    return {
        "variant": variant_label, "tier": tier, "kind": kind,
        "wavlm_layer": layer, "in_dim": in_dim, "n_augs": n_augs,
        "val_thr": round(thr, 3), "val_f1@thr": round(val_f1_at_thr, 4),
        "test_auc": round(m["auc"], 4),
        "test_ap": round(m["ap"], 4),
        "test_f1@0.5": round(m["thr0.5"]["f1"], 4),
        "test_best_f1": round(m["best_f1"]["f1"], 4),
        "test_best_thr": round(m["best_f1"]["threshold"], 3),
        "test_topk_f1": round(m.get("topk", {}).get("f1", float("nan")), 4),
        "recall@p50": round(rap.get("p50", {}).get("recall", float("nan")), 4),
        "recall@p80": round(rap.get("p80", {}).get("recall", float("nan")), 4),
        "recall@p85": round(rap.get("p85", {}).get("recall", float("nan")), 4),
        "recall@p90": round(rap.get("p90", {}).get("recall", float("nan")), 4),
        "recall@p95": round(rap.get("p95", {}).get("recall", float("nan")), 4),
        # per-region columns (NaN when the region is absent from test)
        "ind_f1":    round(ind["f1"], 4),
        "ind_r@p80": round(ind["p80"], 4),
        "ind_r@p85": round(ind["p85"], 4),
        "ind_r@p90": round(ind["p90"], 4),
        "ind_r@p95": round(ind["p95"], 4),
        "php_f1":    round(php["f1"], 4),
        "php_r@p80": round(php["p80"], 4),
        "php_r@p85": round(php["p85"], 4),
        "php_r@p90": round(php["p90"], 4),
        "php_r@p95": round(php["p95"], 4),
    }


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--train_batches", default="audios2,audios4,audios5,audios6,2676,2677")
    ap.add_argument("--test_batches", default="")
    ap.add_argument("--test_region_filter", default="")
    ap.add_argument("--train_only_batches", default="2676,2677")
    ap.add_argument("--min_duration", type=float, default=0.0)
    ap.add_argument("--use_augs", default="")
    ap.add_argument("--use_text_features", default="true",
                    choices=["true", "false"],
                    help="Include feat_* handcrafted features. When false, "
                         "training uses WavLM + Whisper only and text-only "
                         "bases are skipped.")
    ap.add_argument("--per_client_standardize", default="false",
                    choices=["true", "false"],
                    help="Center each client's WavLM+Whisper+feat features "
                         "on its own mean/std. Removes first-order client "
                         "shift before training.")
    ap.add_argument("--fewshot_frac", type=float, default=0.0,
                    help="Few-shot client adaptation: carve this fraction of "
                         "test-batch candidates into train (candidate-disjoint).")
    # ---- v2 additions ----
    ap.add_argument("--drop_high_ks_features", default="false",
                    choices=["true", "false"],
                    help="V2: drop 18 hand-crafted features whose KS between "
                         "client A and B is > 0.13 (recording/style artifacts, "
                         "not cheating). List in HIGH_KS_FEATURES.")
    ap.add_argument("--prior_correction", default="false",
                    choices=["true", "false"],
                    help="V2: apply Saerens-Latinne-Decaestecker correction "
                         "to val_p and test_p (and pick fusions). Monotonic, "
                         "so best_f1 is invariant; calibration is what shifts.")
    ap.add_argument("--target_prior", type=float, default=-1.0,
                    help="V2: pi_target for prior correction. -1.0 = estimate "
                         "from test labels (oracle, research only).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_balance", default="sampler",
                    choices=["sampler", "pos_weight", "both", "none"])
    args = ap.parse_args()

    use_text_features = (args.use_text_features == "true")
    do_per_client_std = (args.per_client_standardize == "true")
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
    log.info("use_text_features      = %s", use_text_features)
    log.info("per_client_standardize = %s", do_per_client_std)
    log.info("fewshot_frac           = %.2f", args.fewshot_frac)

    np.random.seed(args.seed)

    # ---- gt + min_duration filter (shared helper guarantees identical
    # filtering to neural_baseline_train.py)
    gt = load_gt_and_filter(data_dir, args.min_duration, log)
    if len(gt) == 0:
        log.error("All rows filtered out (check --min_duration / gt labels)")
        return 1

    # ---- v2: drop high-KS features BEFORE any downstream feature processing
    if _str_to_bool(args.drop_high_ks_features):
        gt = drop_high_ks_features(gt, log)

    requested = set(train_batches) | set(test_batches)
    n_unused = int((~gt["batch"].isin(requested)).sum())
    if n_unused:
        log.warning("%d rows are outside train/test args and will be unused", n_unused)
    log.info("label dist : %s", gt["label"].value_counts().to_dict())
    if "region" in gt.columns and gt["region"].notna().any():
        log.info("region dist (whole gt): %s",
                 gt["region"].value_counts(dropna=False).to_dict())
    log.info("batch dist : %s", gt["batch"].value_counts().to_dict())

    # ---- augs resolved against cache
    use_augs = resolve_use_augs(args.use_augs, cache_path, log)
    log.info("use_augs: %s", use_augs)
    aug_names_needed = ["orig"] + use_augs

    # ---- cache aligned to gt order
    filenames_cur = gt["npy_filename"].to_numpy().astype(str)
    wavlm_cache, whisper_cache = load_cache_reindexed(
        cache_path, filenames_cur, aug_names_needed, WAVLM_LAYERS, log)

    # ---- handcrafted features
    feat_cols = [c for c in gt.columns if c.startswith("feat_")]
    if feat_cols and use_text_features:
        feat_arr_global = (gt[feat_cols].apply(pd.to_numeric, errors="coerce")
                           .fillna(0.0).to_numpy().astype(np.float32))
        log.info("handcrafted feats: %d cols  first10=%s%s",
                 len(feat_cols), feat_cols[:10], " ..." if len(feat_cols) > 10 else "")
    else:
        if not use_text_features:
            log.info("--use_text_features=false  -> feat_* dropped from every variant")
        elif not feat_cols:
            log.warning("no feat_* columns found; running on WavLM+Whisper only")
        feat_arr_global = None
        feat_cols = []

    feat_available = set(feat_cols)
    y_full = gt["label"].to_numpy().astype(np.int64)

    # ---- Per-client standardization (unsupervised; features only)
    if do_per_client_std:
        feat_arr_global = per_client_standardize(wavlm_cache, whisper_cache,
                                                 gt, log,
                                                 feat_arr=feat_arr_global)

    # ---- splits + group-leak check + clear per-region breakdown
    splits = build_splits(gt, train_batches=train_batches,
                          test_batches=test_batches if test_batches else None,
                          test_region_filter=test_region_filter,
                          train_only_batches=train_only_batches,
                          seed=args.seed, log=log,
                          fewshot_frac=args.fewshot_frac)
    assert_no_group_leak(gt, splits)
    log_split_breakdown(gt, splits, log)

    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    y_train_base = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]

    # ---- compute top-20 if text features are on (once on TRAIN orig)
    if feat_arr_global is not None:
        top20_feats = compute_top20(feat_arr_global, feat_cols, train_idx,
                                    y_train_base, args.seed, log)
    else:
        top20_feats = []
    stylo_feats = _feat_cols(FEAT_STYLOMETRIC, feat_available)

    # ---- per-layer feature matrices (built once)
    per_layer = {layer: LayerMatrices(layer, wavlm_cache, whisper_cache,
                                      feat_arr_global, feat_cols, use_augs)
                 for layer in WAVLM_LAYERS}

    # ---- variant catalog
    specs = base_specs(top20_feats, stylo_feats, use_text_features)
    variant_specs = expand_variants(specs)
    log.info("base variants: %s", [v[0] for v in variant_specs])

    # ---- train each base; cache val/test probas so picks reuse them
    base_proba: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict] = []
    test_region_arr = (gt.iloc[test_idx]["region"].fillna("UNK").to_numpy()
                       if "region" in gt.columns else None)

    for variant_label, bare_name, layer, spec in variant_specs:
        lm = per_layer[layer] if layer != "n/a" else per_layer[WAVLM_LAYERS[0]]
        col_idx = build_block_cols(spec["blocks"], lm)
        if len(col_idx) == 0:
            log.warning("[%s] no usable columns (likely text-only with "
                        "--use_text_features=false); skipping", variant_label)
            continue
        log.info("=" * 60)
        log.info("VARIANT %s   kind=base   wavlm_layer=%s   in_dim=%d   "
                 "expand_train=%s   augs=%s",
                 variant_label, layer, len(col_idx),
                 spec["expand_train"], use_augs if spec["expand_train"] else "(none)")
        log_variant_prelude(variant_label, gt, splits, log)

        X_train, y_train, X_val, X_test = materialize(
            lm, col_idx, train_idx, val_idx, test_idx, y_train_base,
            expand_train_with_augs=spec["expand_train"])

        val_p, test_p = train_one_xgb(X_train, y_train, X_val, X_test,
                                      seed=args.seed, class_balance=args.class_balance,
                                      log=log, tag=variant_label)
        # ---- v2: prior correction (monotonic; best_f1 invariant, calibration shifts)
        if _str_to_bool(args.prior_correction):
            pi_train = float(np.mean(y_train))
            pi_target = (float(np.mean(y_test)) if args.target_prior < 0
                          else float(args.target_prior))
            log.info("[%s] [v2] prior_correction: pi_train=%.4f  pi_target=%.4f",
                     variant_label, pi_train, pi_target)
            val_p = apply_prior_correction(val_p, pi_train, pi_target)
            test_p = apply_prior_correction(test_p, pi_train, pi_target)
        base_proba[variant_label] = {"val": val_p, "test": test_p}
        base_proba_by_bare = base_proba  # (alias, just for clarity)

        thr, val_f1_at_thr = sweep_threshold(y_val, val_p)
        m = compute_metrics(y_test, test_p)
        # per-region metrics so the new summary columns can be populated
        if test_region_arr is not None:
            log.info("[%s] per-region test:", variant_label)
            m["per_region"] = per_slice_metrics(y_test, test_p, test_region_arr,
                                                "region", log)

        log.info("[%s] val_thr=%.2f val_f1@thr=%.3f -> TEST auc=%.3f ap=%.3f "
                 "best_f1=%.3f@%.2f",
                 variant_label, thr, val_f1_at_thr, m["auc"], m["ap"],
                 m["best_f1"]["f1"], m["best_f1"]["threshold"])

        var_dir = out_dir / f"xgb_{variant_label}"
        var_dir.mkdir(parents=True, exist_ok=True)
        pred_df = gt.iloc[test_idx].copy()
        pred_df["pred_score"] = test_p
        pred_df.to_csv(var_dir / "predictions.csv", index=False)
        with (var_dir / "metrics.json").open("w") as f:
            json.dump({"variant": variant_label, "tier": "A", "kind": "base",
                       "wavlm_layer": layer, "in_dim": int(X_train.shape[1]),
                       "augs_used": use_augs if spec["expand_train"] else [],
                       "use_text_features": use_text_features,
                       "train_only_batches": train_only_batches,
                       "min_duration": args.min_duration,
                       "class_balance": args.class_balance,
                       "val_thr_swept": thr, "val_f1_at_thr": val_f1_at_thr,
                       "test": m}, f, indent=2)

        summary_rows.append(_summary_row(variant_label, "A", "base", layer,
                                         int(X_train.shape[1]),
                                         len(use_augs) if spec["expand_train"] else 0,
                                         thr, val_f1_at_thr, m))

    # ---- picks
    def _resolve_member_label(bare: str, layer: str) -> str | None:
        if bare in ("wavlm", "everything"):
            tag = layer if layer == "last" else f"l{layer}"
            label = f"{bare}_xgb_{tag}"
        else:
            label = f"{bare}_xgb"
        return label if label in base_proba else None

    for pick_name, pick in PICK_DEFS.items():
        layers_to_run = WAVLM_LAYERS if pick["needs_wavlm"] else ["n/a"]
        for layer in layers_to_run:
            members: list[str] = []
            weights: list[float] = []
            missing: list[str] = []
            for w_i, bare in enumerate(pick["members"]):
                lab = _resolve_member_label(bare, layer)
                if lab is None:
                    missing.append(bare)
                else:
                    members.append(lab)
                    weights.append(pick["weights"][w_i])
            if missing:
                log.info("[%s/%s] dropping missing members %s "
                         "(usually because --use_text_features=false)",
                         pick_name, layer, missing)
            if len(members) < 2:
                log.warning("[%s/%s] only %d viable member -- skipping "
                            "(would just duplicate a base row)",
                            pick_name, layer, len(members))
                continue
            w = np.array(weights, dtype=np.float64)
            w = w / w.sum()
            val_fused = sum(w[i] * base_proba[lab]["val"]
                            for i, lab in enumerate(members))
            test_fused = sum(w[i] * base_proba[lab]["test"]
                             for i, lab in enumerate(members))

            variant_label = pick_name if layer == "n/a" else \
                f"{pick_name}_{layer if layer == 'last' else f'l{layer}'}"
            log.info("=" * 60)
            log.info("PICK %s   members=%s  weights=%s  layer=%s",
                     variant_label, members, w.tolist(), layer)
            log_variant_prelude(variant_label, gt, splits, log)

            thr, val_f1_at_thr = sweep_threshold(y_val, val_fused)
            m = compute_metrics(y_test, test_fused)
            if test_region_arr is not None:
                log.info("[%s] per-region test:", variant_label)
                m["per_region"] = per_slice_metrics(y_test, test_fused,
                                                    test_region_arr, "region", log)
            log.info("[%s] val_thr=%.2f val_f1@thr=%.3f -> TEST auc=%.3f ap=%.3f "
                     "best_f1=%.3f@%.2f", variant_label, thr, val_f1_at_thr,
                     m["auc"], m["ap"], m["best_f1"]["f1"], m["best_f1"]["threshold"])

            var_dir = out_dir / f"xgb_{variant_label}"
            var_dir.mkdir(parents=True, exist_ok=True)
            pred_df = gt.iloc[test_idx].copy()
            pred_df["pred_score"] = test_fused
            pred_df.to_csv(var_dir / "predictions.csv", index=False)
            with (var_dir / "metrics.json").open("w") as f:
                json.dump({"variant": variant_label, "tier": "A", "kind": "pick",
                           "wavlm_layer": layer, "members": members,
                           "weights": w.tolist(),
                           "augs_used": use_augs,
                           "use_text_features": use_text_features,
                           "train_only_batches": train_only_batches,
                           "min_duration": args.min_duration,
                           "class_balance": args.class_balance,
                           "val_thr_swept": thr, "val_f1_at_thr": val_f1_at_thr,
                           "test": m}, f, indent=2)

            summary_rows.append(_summary_row(variant_label, "A", "pick", layer,
                                             -1, len(use_augs), thr,
                                             val_f1_at_thr, m))

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary2.csv"
    summary.to_csv(summary_path, index=False)
    log.info("=" * 60)
    log.info("SUMMARY (XGB, summary2.csv):\n%s", summary.to_string(index=False))
    log.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
