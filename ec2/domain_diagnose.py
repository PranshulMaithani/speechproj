#!/usr/bin/env python3
"""Domain-gap diagnostic: client A (audios2/4/5) vs client B (audios6).

Tells you which of the four failure modes is driving the audios6 F1 collapse:

  (1) LABEL/PRIOR SHIFT   -- cheat-rate differs between clients
                             fix: re-tune threshold only
  (2) DOMAIN GAP          -- feature distributions differ
                             fix: drop high-KS features / re-weight groups
  (3) CALIBRATION ONLY    -- AUC fine, threshold wrong
                             fix: Platt/isotonic on a tiny target sample
  (4) FEATURE-GROUP DRIFT -- one group (wavlm vs whisper vs text) is the culprit
                             fix: down-weight that group in the ensemble

Output: a text report you paste back.

Usage:
    python ec2/domain_diagnose.py --data_dir <UPLOAD> --out domain_gap_report.txt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _data_pipeline import (
    WAVLM_LAYERS,
    load_cache_reindexed,
    load_gt_and_filter,
    setup_logging,
)


CLIENT_A = ("audios2", "audios4", "audios5")
CLIENT_B = ("audios6",)


def _subsample(X: np.ndarray, n: int, rng) -> np.ndarray:
    if len(X) <= n:
        return X
    idx = rng.choice(len(X), n, replace=False)
    return X[idx]


def mmd_rbf_sq(X: np.ndarray, Y: np.ndarray, n_max: int = 500,
               seed: int = 0) -> float:
    """Squared MMD with RBF kernel + median-distance bandwidth.
    Subsamples to <=n_max per side to keep this O(n^2) tractable."""
    from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
    rng = np.random.default_rng(seed)
    X = _subsample(X.astype(np.float32), n_max, rng)
    Y = _subsample(Y.astype(np.float32), n_max, rng)
    if len(X) < 5 or len(Y) < 5:
        return float("nan")
    D = euclidean_distances(np.vstack([X, Y]))
    med = float(np.median(D[D > 0])) if (D > 0).any() else 1.0
    gamma = 1.0 / (2.0 * med * med) if med > 0 else 1.0
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def domain_clf_auc(X: np.ndarray, domain: np.ndarray, seed: int = 0) -> float:
    """5-fold CV AUC of a logistic regression predicting domain from features.
    AUC near 0.5 = invariant; AUC -> 1.0 = perfectly separable."""
    if len(np.unique(domain)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(X, domain):
        pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                              LogisticRegression(max_iter=2000, C=1.0,
                                                 solver="liblinear"))
        pipe.fit(X[tr], domain[tr])
        p = pipe.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(domain[va], p))
    return float(np.mean(aucs))


def fmt_header(title: str) -> str:
    return f"\n{'=' * 80}\n{title}\n{'=' * 80}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out", default="domain_gap_report.txt")
    ap.add_argument("--min_duration", type=float, default=5.0)
    ap.add_argument("--cache_name", default="embeddings_cache.npz")
    args = ap.parse_args()

    log = setup_logging(Path(args.out).with_suffix(".log"), "domain_diagnose")
    gt = load_gt_and_filter(args.data_dir, min_duration=args.min_duration,
                             log=log)
    cache_path = args.data_dir / args.cache_name
    # Need "orig" aug only; both wavlm layers
    wavlm_by_layer_aug, whisper_by_aug = load_cache_reindexed(
        cache_path, gt["npy_filename"].astype(str).to_numpy(),
        aug_names_needed=["orig"], layers_needed=WAVLM_LAYERS, log=log)

    is_a = gt["batch"].isin(CLIENT_A).to_numpy()
    is_b = gt["batch"].isin(CLIENT_B).to_numpy()
    keep = is_a | is_b
    if not keep.any():
        log.error("no rows match client A or B")
        return 1

    gt_kb = gt.loc[keep].reset_index(drop=True)
    domain = gt_kb["batch"].isin(CLIENT_B).astype(int).to_numpy()
    label = gt_kb["label"].astype(int).to_numpy()
    keep_idx = np.where(keep)[0]

    # Slice every feature group down to the kept rows.
    # Mirror the training scripts: feat_* is coerced to numeric and NaN -> 0.0
    # (xgboost_train.py:471, neural_baseline_train.py:387). For wavlm/whisper
    # we also nan_to_num as a safety net in case extraction wrote NaN.
    groups: dict[str, np.ndarray] = {}
    for layer in WAVLM_LAYERS:
        arr = wavlm_by_layer_aug[(layer, "orig")][keep_idx]
        groups[f"wavlm_{layer}"] = np.nan_to_num(arr, nan=0.0,
                                                 posinf=0.0, neginf=0.0)
    arr = whisper_by_aug["orig"][keep_idx]
    groups["whisper"] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    text_cols = [c for c in gt_kb.columns if c.startswith("feat_")]
    nan_rates: dict[str, dict[str, float]] = {}  # group -> {A_rate, B_rate}
    if text_cols:
        raw_text = gt_kb[text_cols].apply(pd.to_numeric, errors="coerce")
        # NaN-rate accounting BEFORE we fill, so we can report it.
        nan_a = raw_text.loc[domain == 0].isna().mean(axis=0)
        nan_b = raw_text.loc[domain == 1].isna().mean(axis=0)
        text_nan_table = pd.DataFrame({
            "feature": text_cols,
            "nan_rate_A": nan_a.values.round(4),
            "nan_rate_B": nan_b.values.round(4),
            "delta": (nan_b.values - nan_a.values).round(4),
        })
        groups["text_feats"] = raw_text.fillna(0.0).to_numpy(dtype=np.float32)
        nan_rates["text_feats_per_feature"] = text_nan_table
    # WavLM/Whisper group-level NaN rate (per-row: any NaN in the row?)
    for layer in WAVLM_LAYERS:
        arr_full = wavlm_by_layer_aug[(layer, "orig")][keep_idx]
        any_nan = np.isnan(arr_full).any(axis=1)
        nan_rates[f"wavlm_{layer}"] = {
            "nan_row_rate_A": float(any_nan[domain == 0].mean()),
            "nan_row_rate_B": float(any_nan[domain == 1].mean()),
        }
    arr_full = whisper_by_aug["orig"][keep_idx]
    any_nan = np.isnan(arr_full).any(axis=1)
    nan_rates["whisper"] = {
        "nan_row_rate_A": float(any_nan[domain == 0].mean()),
        "nan_row_rate_B": float(any_nan[domain == 1].mean()),
    }

    out = open(args.out, "w", encoding="utf-8")
    out.write("DOMAIN GAP DIAGNOSTIC\n")
    out.write(f"client A = {CLIENT_A}\n")
    out.write(f"client B = {CLIENT_B}\n")
    out.write(f"rows kept: {keep.sum()}  (A: {is_a.sum()}, B: {is_b.sum()})\n")

    # === 0. NaN PRESENCE BY GROUP / CLIENT ===
    # Differential NaN rate between clients is itself a domain-shift symptom:
    # if client B can't have a feature computed but client A can, that's audio/
    # transcript-quality drift.
    out.write(fmt_header("0. NaN PRESENCE BY GROUP / CLIENT  "
                         "(extraction failures; surfaces hidden drift)"))
    out.write("Embedding-level (any NaN in row):\n")
    emb_rows = []
    for g in ("wavlm_last", "wavlm_9", "whisper"):
        if g in nan_rates:
            emb_rows.append({"group": g,
                              "nan_row_rate_A": round(nan_rates[g]["nan_row_rate_A"], 4),
                              "nan_row_rate_B": round(nan_rates[g]["nan_row_rate_B"], 4)})
    out.write(pd.DataFrame(emb_rows).to_string(index=False))
    out.write("\n\nText-feature-level (top-15 features by |Delta NaN rate|):\n")
    if "text_feats_per_feature" in nan_rates:
        t = nan_rates["text_feats_per_feature"].copy()
        t["abs_delta"] = t["delta"].abs()
        t = t.sort_values("abs_delta", ascending=False).drop(columns="abs_delta")
        out.write(t.head(15).to_string(index=False))
        out.write("\n\nTotal NaN cells in feat_* matrix:\n")
        n_a_nan = int((t["nan_rate_A"] *
                       int((domain == 0).sum())).sum())
        n_b_nan = int((t["nan_rate_B"] *
                       int((domain == 1).sum())).sum())
        out.write(f"  client A: ~{n_a_nan} NaN cells across all feat_* columns\n")
        out.write(f"  client B: ~{n_b_nan} NaN cells across all feat_* columns\n")
        out.write("\n>>> All NaN cells were filled with 0.0 below (matches training scripts).\n")
        out.write(">>> If a feat_* has Delta NaN rate > 0.10, the feature is unreliable on\n")
        out.write(">>> that client -- consider dropping it before training.\n")
        out.write(">>> NaN at extraction = empty transcript, audio too short, or library\n")
        out.write(">>> missing for that feature; full_text_features.py normally returns 0.0\n")
        out.write(">>> but some corner cases leak through to gt.csv as NaN/string.\n")
    else:
        out.write("(no feat_* columns)\n")

    # === 1. LABEL PRIOR SHIFT ===
    out.write(fmt_header("1. LABEL PRIOR SHIFT  (per-client cheat rate)"))
    rows = []
    for batch in sorted(gt_kb["batch"].unique()):
        m = (gt_kb["batch"] == batch).to_numpy()
        rows.append({"batch": batch, "n": int(m.sum()),
                     "cheat_rate": round(float(label[m].mean()), 4),
                     "n_speakers": int(gt_kb.loc[m, "group_id"].nunique()
                                       if "group_id" in gt_kb.columns else -1)})
    out.write(pd.DataFrame(rows).to_string(index=False))
    a_rate = float(label[domain == 0].mean())
    b_rate = float(label[domain == 1].mean())
    out.write(f"\n\nclient-A cheat_rate: {a_rate:.4f}  (n={int((domain==0).sum())})\n")
    out.write(f"client-B cheat_rate: {b_rate:.4f}  (n={int((domain==1).sum())})\n")
    out.write(f"|delta|            : {abs(a_rate - b_rate):.4f}\n")
    if abs(a_rate - b_rate) > 0.10:
        out.write("\n>>> PRIOR SHIFT likely a factor (|delta| > 0.10).\n")
        out.write(">>> A threshold-only adjustment could close a chunk of the gap.\n")
    else:
        out.write("\n>>> Prior shift is small -- not the dominant cause.\n")

    # === 2. WITHIN CLIENT-B REGION BREAKDOWN ===
    out.write(fmt_header("2. WITHIN-CLIENT-B REGION SHIFT  (audios6 IND vs PHP)"))
    if "region" in gt_kb.columns:
        sub = gt_kb.loc[domain == 1]
        rows = []
        for region in sorted(sub["region"].dropna().astype(str).unique()):
            m = (sub["region"].astype(str) == region).to_numpy()
            y = label[domain == 1][m]
            rows.append({"region": region, "n": int(m.sum()),
                         "cheat_rate": round(float(y.mean()), 4)})
        out.write(pd.DataFrame(rows).to_string(index=False))
    else:
        out.write("(no region column)")
    out.write("\n")

    # === 3. DOMAIN CLASSIFIER AUC PER FEATURE GROUP ===
    out.write(fmt_header("3. DOMAIN CLASSIFIER AUC per FEATURE GROUP"
                         "  (>0.9 = heavy drift; ~0.5 = invariant)"))
    rows = []
    for name, X in groups.items():
        auc = domain_clf_auc(X, domain)
        interp = ("HEAVY drift"   if auc > 0.90 else
                  "moderate drift" if auc > 0.75 else
                  "mild drift"     if auc > 0.60 else
                  "near-invariant")
        rows.append({"group": name, "domain_auc": round(auc, 4),
                     "dim": X.shape[1], "verdict": interp})
    rows.sort(key=lambda r: -r["domain_auc"])
    out.write(pd.DataFrame(rows).to_string(index=False))
    out.write("\n\n>>> Groups with AUC < 0.60 are domain-invariant -- lean on them.\n")
    out.write(">>> Groups with AUC > 0.90 carry mostly client-id, not cheating signal.\n")

    # === 4. MMD PER FEATURE GROUP ===
    out.write(fmt_header("4. MMD^2 (RBF) between client A and B per FEATURE GROUP"))
    rows = []
    for name, X in groups.items():
        m = mmd_rbf_sq(X[domain == 0], X[domain == 1])
        rows.append({"group": name, "mmd2": round(m, 5)})
    rows.sort(key=lambda r: -r["mmd2"])
    out.write(pd.DataFrame(rows).to_string(index=False))
    out.write("\n")

    # === 5. PER-FEATURE KS (text features only) ===
    out.write(fmt_header("5. PER-FEATURE KS  (text features; max|CDF_A - CDF_B|)"))
    if text_cols:
        ks_rows = []
        for col in text_cols:
            v = gt_kb[col].to_numpy()
            if np.unique(v).size < 2:
                continue
            try:
                ks = float(ks_2samp(v[domain == 0], v[domain == 1]).statistic)
            except Exception:
                continue
            ks_rows.append({"feature": col, "ks": round(ks, 4)})
        ks_df = pd.DataFrame(ks_rows).sort_values("ks", ascending=False)
        out.write("TOP-20 MOST SHIFTED (drop or downweight these):\n")
        out.write(ks_df.head(20).to_string(index=False))
        out.write("\n\nTOP-15 MOST INVARIANT (these are your reliable signals):\n")
        out.write(ks_df.tail(15).iloc[::-1].to_string(index=False))
        out.write(f"\n\n  mean KS over all {len(ks_df)} text feats: "
                  f"{ks_df['ks'].mean():.4f}\n")
    else:
        out.write("(no feat_* columns in gt -- skip)\n")

    # === 6. LABEL-CONDITIONAL DRIFT  (is cheat-class drifting more than honest?) ===
    out.write(fmt_header("6. LABEL-CONDITIONAL DOMAIN AUC"
                         "  (run domain clf separately on cheat vs honest rows)"))
    rows = []
    for cls, cls_name in [(1, "cheat=1"), (0, "honest=0")]:
        sel = (label == cls)
        if sel.sum() < 20 or len(np.unique(domain[sel])) < 2:
            continue
        for name, X in groups.items():
            auc = domain_clf_auc(X[sel], domain[sel])
            rows.append({"class": cls_name, "group": name,
                         "domain_auc": round(auc, 4),
                         "n_A": int(((domain == 0) & sel).sum()),
                         "n_B": int(((domain == 1) & sel).sum())})
    if rows:
        out.write(pd.DataFrame(rows).to_string(index=False))
        out.write("\n\n>>> If domain AUC is higher within the CHEAT class than the\n")
        out.write(">>> honest class, the model's cheat representation is more\n")
        out.write(">>> client-specific. That's the worst case: it means cheat 'looks\n")
        out.write(">>> like' something different at each client.\n")
    else:
        out.write("(not enough rows per class to compute)\n")

    # === 7. CALIBRATION-VS-RANKING DIAGNOSTIC  (synthesis) ===
    out.write(fmt_header("7. SYNTHESIS  (what this report concludes)"))
    out.write(
"""Read in this order:

  Section 1 -- if |delta cheat_rate| > 0.10:  PRIOR SHIFT contributes.
  Section 3 -- which group has the highest domain AUC.
  Section 4 -- MMD ranking confirms section 3.
  Section 5 -- pinpoints WHICH text features are client-id surrogates.
  Section 6 -- tells you if the drift is class-conditional (worst case).

Decision matrix:

  prior_shift > 0.10  AND  AUC@0.92 on a6   -> CALIBRATION ONLY (cheapest fix)
                                              -> Platt scaling on 10% target labels.
                                              -> Or shift threshold by |delta prior|.

  domain_auc(wavlm) > 0.9 AND domain_auc(text) < 0.6
                                              -> WAVLM carries client-id.
                                              -> down-weight wavlm in tier A picks.
                                              -> rely on text + whisper.

  domain_auc all > 0.85                       -> EVERY group drifts.
                                              -> only fewshot will save you.
                                              -> per-client-std failed because shift
                                                 is non-linear, not first-moment.

  section 6 cheat-AUC >> honest-AUC          -> CLASS-CONDITIONAL DRIFT.
                                              -> the model's cheat manifold is
                                                 client-specific. No unsupervised
                                                 fix exists. Fewshot is mandatory.

  KS of top-5 text feats > 0.5               -> those features encode client style
                                                 not cheating. Drop them.
""")

    out.close()
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
