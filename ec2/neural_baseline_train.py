"""Per-audio cheating classifier training (Stage 0 of NEURAL_PLAN.md).

Loads pre-extracted frozen embeddings from embeddings_cache.npz (run
extract_embeddings.py once first). For each iteration the script trains a
fresh MLP head on the concatenation of:

    WavLM-base-plus mean-pool (768) + Whisper-medium encoder mean-pool (1024)
    + handcrafted feat_* features from gt.csv

feat_* columns are concatenated BEFORE standardization and PCA, so every
variant (including pca90) trains on a compressed representation that still
contains the text features.

Variant matrix (20 = 2 archs x 2 WavLM layers x 5 PCA settings):

    architectures:
        default = 512 -> 256 -> 128 -> 1, dropout 0.40, wd 5e-4
        tiny    = 128 -> 1,                dropout 0.55, wd 5e-3

    WavLM layer:
        last  encoder output (the standard mean-pool baseline)
        l9    hidden_states[9] -- often best for paralinguistic tasks

    PCA on standardized concat:
        full   no PCA (control)
        pca98  keep 98% variance
        pca95  keep 95% variance
        pca93  keep 93% variance
        pca90  keep 90% variance

Standardizer and PCA are fit per-layer on the (possibly aug-expanded) train
set only and reused across all 5 PCA settings for that layer.

Split modes:
    Mode A (no --test_batches): StratifiedGroupKFold(5) on (train_batches minus
        --train_only_batches). fold0 = test, fold1 = val, folds 2-4 = train.
        --train_only_batches rows are appended to train after the split.
    Mode B (--test_batches set): test = those batches (optionally region-filtered).
        Train pool = (train_batches minus train_only_batches) minus candidates
        leaking into test. Val drawn from same-region subset when possible.
        --train_only_batches rows are appended to train.

ALLSTAR support: by default --train_only_batches=2676,2677 so the ALLSTAR
auxiliary batches segment-split by neural_baseline_prep.py are never placed
in val or test. They contribute supervised acoustic signal to train only.

Optional --use_augs:
    Augmentation expands the TRAINING matrix only; val/test always use 'orig'
    embeddings. Pass 'all' to use every aug present in the cache, or a CSV
    of names (e.g. 'noise,pitch,vtlp'). Empty = no augs.

Optional --min_duration:
    Drops any row with duration_sec < threshold from BOTH train and test
    before splits are built. Reduces label noise from very short clips.

Run:
    python neural_baseline_train.py \\
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
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from _data_pipeline import (WAVLM_LAYERS, assert_no_group_leak, build_splits,
                            compute_metrics, extract_region_metrics,
                            load_cache_reindexed, load_gt_and_filter,
                            log_split_breakdown, log_variant_prelude,
                            per_slice_metrics, resolve_use_augs, setup_logging)

# ----------------------------------------------------------------------------
# Variants.
# ----------------------------------------------------------------------------

ARCH_CONFIG: dict[str, dict] = {
    "default": {"hidden": (512, 256, 128), "dropout": 0.40, "weight_decay": 5e-4},
    "tiny":    {"hidden": (128,),          "dropout": 0.55, "weight_decay": 5e-3},
}

PCA_VARIANTS: list[tuple[str, float | None]] = [
    ("full",  None),
    ("pca98", 0.98),
    ("pca95", 0.95),
    ("pca93", 0.93),
    ("pca90", 0.90),
]

# NOTE: WAVLM_LAYERS is imported from _data_pipeline so the XGB script sees
# the same layer set in the cache loader.

LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0


def make_variants() -> list[tuple[str, str, str, float | None]]:
    """20 (variant_name, arch_name, wavlm_layer, pca_var) tuples =
    2 archs x 2 layers x 5 PCA settings."""
    out = []
    for arch in ARCH_CONFIG.keys():
        for layer in WAVLM_LAYERS:
            layer_tag = layer if layer == "last" else f"l{layer}"
            for pca_name, pca_val in PCA_VARIANTS:
                out.append((f"{arch}_{layer_tag}_{pca_name}", arch, layer, pca_val))
    return out


# ----------------------------------------------------------------------------
# MLP.
# ----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 256, 128), dropout=0.4):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def smoothed_bce_logits(logits: torch.Tensor, target: torch.Tensor,
                        smoothing: float, pos_weight: torch.Tensor) -> torch.Tensor:
    if smoothing > 0:
        target = target * (1.0 - smoothing) + 0.5 * smoothing
    return nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


@dataclass
class TrainResult:
    best_val_f1: float
    best_epoch: int
    test_metrics: dict


def train_one(X_tr, y_tr, X_va, y_va, X_te, y_te, *, in_dim: int, batch_size: int,
              epochs: int, lr: float, wd: float, device: torch.device,
              log: logging.Logger, tag: str,
              hidden: tuple, dropout: float,
              label_smoothing: float = LABEL_SMOOTHING,
              grad_clip: float = GRAD_CLIP_NORM,
              patience: int = 10,
              class_balance: str = "sampler") -> tuple[TrainResult, np.ndarray]:
    """class_balance:
        'sampler'    WeightedRandomSampler so each minibatch is class-balanced
                     in expectation (default; replaces pos_weight)
        'pos_weight' BCE pos_weight = neg/pos, natural shuffling
        'both'       sampler + pos_weight (rare; usually over-corrects)
        'none'       natural distribution, no correction (sanity check)
    """
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    total = max(pos + neg, 1.0)
    if class_balance in ("pos_weight", "both"):
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)
    log.info("[%s] arch hidden=%s dropout=%.2f wd=%.1e ls=%.2f clip=%.1f balance=%s",
             tag, hidden, dropout, wd, label_smoothing, grad_clip, class_balance)
    log.info("[%s] train n=%d  pos=%d (%.1f%%)  neg=%d (%.1f%%)  pw=%.2f  val n=%d  test n=%d  in_dim=%d",
             tag, len(y_tr), int(pos), 100 * pos / total, int(neg), 100 * neg / total,
             pos_weight.item(), len(y_va), len(y_te), in_dim)

    model = MLP(in_dim, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def to_tensor(X, y):
        return TensorDataset(torch.from_numpy(X.astype(np.float32)),
                             torch.from_numpy(y.astype(np.float32)))

    if class_balance in ("sampler", "both") and pos > 0 and neg > 0:
        # per-sample weight = 1 / count_of_its_class -> equal probability for
        # each class in expectation. num_samples=len(y_tr) so an epoch still
        # sees ~one-pass worth of gradients (just with reweighted draws).
        cls_counts = np.array([neg, pos], dtype=np.float64)
        per_sample_w = (1.0 / cls_counts[y_tr.astype(np.int64)]).astype(np.float64)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=per_sample_w.tolist(), num_samples=len(y_tr), replacement=True,
        )
        tr_loader = DataLoader(to_tensor(X_tr, y_tr), batch_size=batch_size,
                               sampler=sampler, drop_last=True)
    else:
        tr_loader = DataLoader(to_tensor(X_tr, y_tr), batch_size=batch_size,
                               shuffle=True, drop_last=True)

    @torch.no_grad()
    def predict(X):
        model.eval()
        out = []
        for i in range(0, len(X), 512):
            x = torch.from_numpy(X[i:i + 512].astype(np.float32)).to(device)
            out.append(torch.sigmoid(model(x)).cpu().numpy())
        return np.concatenate(out)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience_left = patience

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = smoothed_bce_logits(logits, yb, label_smoothing, pos_weight)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item() * xb.size(0)
        sched.step()
        ep_loss /= max(len(y_tr), 1)

        val_p = predict(X_va)
        val_f1 = f1_score(y_va, (val_p >= 0.5).astype(int), zero_division=0)
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            patience_left = patience
        else:
            patience_left -= 1

        if ep % 5 == 0 or ep == 1:
            log.info("[%s] ep %3d  loss %.4f  val_f1 %.4f  best %.4f@%d",
                     tag, ep, ep_loss, val_f1, best_val_f1, best_epoch)
        if patience_left <= 0:
            log.info("[%s] early stop at ep %d (best val_f1 %.4f @ ep %d)",
                     tag, ep, best_val_f1, best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_p = predict(X_te)
    metrics = compute_metrics(y_te, test_p)
    return TrainResult(best_val_f1, best_epoch, metrics), test_p


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
    ap.add_argument("--test_region_filter", default="",
                    help="when test_batches set, restrict test to region == this")
    ap.add_argument("--train_only_batches", default="2676,2677",
                    help="comma-separated batches whose rows are forced into train "
                         "only -- never appear in val/test. Default: ALLSTAR batches.")
    ap.add_argument("--min_duration", type=float, default=0.0,
                    help="drop rows with duration_sec < this from BOTH train and test")
    ap.add_argument("--use_augs", default="",
                    help="comma-separated aug names from the cache to add to TRAIN "
                         "(val/test always use 'orig'). 'all' = every aug in cache. "
                         "Empty (default) = no augs.")
    ap.add_argument("--use_text_features", default="true",
                    choices=["true", "false"],
                    help="Include feat_* handcrafted features in the input "
                         "vector. When false, the MLP is trained on WavLM + "
                         "Whisper only.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_balance", default="sampler",
                    choices=["sampler", "pos_weight", "both", "none"],
                    help="how to handle class imbalance in train. "
                         "'sampler' = WeightedRandomSampler (balanced minibatches; default). "
                         "'pos_weight' = BCE pos_weight=neg/pos with natural shuffling. "
                         "'both' = sampler + pos_weight (over-corrects; rarely needed). "
                         "'none' = natural distribution, no correction.")
    args = ap.parse_args()

    use_text_features = (args.use_text_features == "true")
    train_batches = [b.strip() for b in args.train_batches.split(",") if b.strip()]
    test_batches = [b.strip() for b in args.test_batches.split(",") if b.strip()]
    train_only_batches = [b.strip() for b in args.train_only_batches.split(",") if b.strip()]
    test_region_filter = args.test_region_filter.strip() or None

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")

    log = setup_logging(out_dir / "log_nn.txt", name="nn")
    log.info("data_dir = %s", data_dir)
    log.info("out_dir  = %s", out_dir)
    log.info("cache    = %s", cache_path)
    log.info("use_text_features = %s", use_text_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device   = %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load gt + min_duration filter (shared with xgboost_train.py)
    gt = load_gt_and_filter(data_dir, args.min_duration, log)
    if len(gt) == 0:
        log.error("All rows filtered out (check --min_duration / gt labels)")
        return 1

    # ---- Diagnostics
    requested = set(train_batches) | set(test_batches)
    n_unused = int((~gt["batch"].isin(requested)).sum())
    if n_unused:
        log.warning("%d rows are outside train/test args and will be unused", n_unused)
    log.info("label dist : %s", gt["label"].value_counts().to_dict())
    if "region" in gt.columns and gt["region"].notna().any():
        log.info("region dist: %s", gt["region"].value_counts(dropna=False).to_dict())
    log.info("batch dist : %s", gt["batch"].value_counts().to_dict())

    # ---- Resolve augs (validated against cache)
    use_augs = resolve_use_augs(args.use_augs, cache_path, log)
    log.info("use_augs: %s", use_augs)
    aug_names_needed = ["orig"] + use_augs

    # ---- Load cache (both WavLM layers, plus each aug requested), aligned to gt order
    filenames_cur = gt["npy_filename"].to_numpy().astype(str)
    wavlm_cache, whisper_cache = load_cache_reindexed(
        cache_path, filenames_cur, aug_names_needed, WAVLM_LAYERS, log)
    log.info("Loaded cache: wavlm keys=%s  whisper keys=%s",
             list(wavlm_cache.keys()), list(whisper_cache.keys()))

    # ---- Handcrafted features
    feat_cols = [c for c in gt.columns if c.startswith("feat_")]
    if feat_cols and use_text_features:
        feat_block = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feat_arr = feat_block.to_numpy().astype(np.float32)
        log.info("handcrafted feats: %d cols  first10=%s%s",
                 len(feat_cols), feat_cols[:10], " ..." if len(feat_cols) > 10 else "")
    else:
        feat_arr = None
        if not use_text_features:
            log.info("--use_text_features=false  -> feat_* dropped; "
                     "training on WavLM + Whisper only")
        else:
            log.info("no feat_* columns -- audio embeddings only")

    def concat(wavlm: np.ndarray, whisper: np.ndarray, feats: np.ndarray | None) -> np.ndarray:
        parts = [wavlm, whisper]
        if feats is not None:
            parts.append(feats)
        return np.concatenate(parts, axis=1).astype(np.float32)

    y_full = gt["label"].to_numpy().astype(np.int64)

    # ---- Splits
    splits = build_splits(gt, train_batches=train_batches,
                          test_batches=test_batches if test_batches else None,
                          test_region_filter=test_region_filter,
                          train_only_batches=train_only_batches,
                          seed=args.seed, log=log)
    assert_no_group_leak(gt, splits)
    log_split_breakdown(gt, splits, log)
    log.info("--class_balance=%s will be applied to TRAIN minibatches "
             "(val/test always use natural distribution)", args.class_balance)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    feat_train = feat_arr[train_idx] if feat_arr is not None else None
    y_train_base = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]

    # ---- Per-layer (X_train aug-expanded, X_val, X_test, scaler) ----
    # We pre-build all features per WavLM layer once so the variant loop only
    # has to fit PCA + MLP.
    per_layer: dict[str, dict] = {}
    for layer in WAVLM_LAYERS:
        X_orig = concat(wavlm_cache[(layer, "orig")], whisper_cache["orig"], feat_arr)
        zero_rows = (np.abs(X_orig).sum(axis=1) == 0)
        if zero_rows.any():
            log.warning("[layer=%s] %d rows have all-zero orig embeddings",
                        layer, int(zero_rows.sum()))

        X_train_blocks = [X_orig[train_idx]]
        y_train_blocks = [y_train_base]
        aug_tag_blocks: list[list[str]] = [["orig"] * len(train_idx)]
        for a in use_augs:
            X_a = concat(wavlm_cache[(layer, a)], whisper_cache[a], feat_arr)
            X_train_blocks.append(X_a[train_idx])
            y_train_blocks.append(y_train_base)
            aug_tag_blocks.append([a] * len(train_idx))

        X_train = np.concatenate(X_train_blocks, axis=0).astype(np.float32)
        y_train = np.concatenate(y_train_blocks, axis=0).astype(np.int64)
        aug_tags: list[str] = []
        for blk in aug_tag_blocks:
            aug_tags.extend(blk)
        X_val = X_orig[val_idx]
        X_test = X_orig[test_idx]
        log.info("[layer=%s] train=%d (orig=%d + %d augs x %d)  val=%d  test=%d",
                 layer, len(X_train), len(train_idx), len(use_augs),
                 len(train_idx), len(X_val), len(X_test))

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        per_layer[layer] = {
            "X_train_s": X_train_s, "y_train": y_train,
            "X_val_s": X_val_s, "X_test_s": X_test_s,
            "aug_tags": aug_tags,
        }

    # ---- Iterate variants (20 = 2 archs x 2 wavlm layers x 5 PCAs) ----
    summary_rows: list[dict] = []
    variants = make_variants()
    log.info("Running %d variants: %s", len(variants), [v[0] for v in variants])

    for variant, arch_name, layer, pca_var in variants:
        log.info("=" * 60)
        arch = ARCH_CONFIG[arch_name]
        log.info("VARIANT %s   arch=%s   wavlm_layer=%s   pca=%s   augs=%s",
                 variant, arch_name, layer, pca_var, use_augs)
        log_variant_prelude(variant, gt, splits, log)
        lp = per_layer[layer]
        X_train_s = lp["X_train_s"]
        y_train = lp["y_train"]
        X_val_s = lp["X_val_s"]
        X_test_s = lp["X_test_s"]

        if pca_var is None:
            Xt, Xv, Xs = X_train_s, X_val_s, X_test_s
            in_dim = Xt.shape[1]
        else:
            pca = PCA(n_components=pca_var, svd_solver="full", random_state=args.seed)
            pca.fit(X_train_s)
            Xt = pca.transform(X_train_s).astype(np.float32)
            Xv = pca.transform(X_val_s).astype(np.float32)
            Xs = pca.transform(X_test_s).astype(np.float32)
            in_dim = Xt.shape[1]
            log.info("PCA(%.2f) -> %d components", pca_var, in_dim)

        result, test_p = train_one(
            Xt, y_train, Xv, y_val, Xs, y_test,
            in_dim=in_dim, batch_size=args.batch_size, epochs=args.epochs,
            lr=args.lr, wd=arch["weight_decay"], device=device, log=log, tag=variant,
            hidden=tuple(arch["hidden"]), dropout=arch["dropout"],
            class_balance=args.class_balance)

        var_dir = out_dir / variant
        var_dir.mkdir(parents=True, exist_ok=True)
        pred_df = gt.iloc[test_idx].copy()
        pred_df["pred_score"] = test_p
        pred_df.to_csv(var_dir / "predictions.csv", index=False)

        m = result.test_metrics
        log.info("[%s] TEST n=%d  auc=%.3f  ap=%.3f  thr0.5 f1=%.3f  best_f1=%.3f@%.2f",
                 variant, m["n"], m["auc"], m["ap"], m["thr0.5"]["f1"],
                 m["best_f1"]["f1"], m["best_f1"]["threshold"])
        rap = m.get("recall_at_precision", {})
        log.info("[%s] recall@p  p50=%.3f  p80=%.3f  p90=%.3f  p95=%.3f", variant,
                 rap.get("p50", {}).get("recall", float("nan")),
                 rap.get("p80", {}).get("recall", float("nan")),
                 rap.get("p90", {}).get("recall", float("nan")),
                 rap.get("p95", {}).get("recall", float("nan")))

        if "batch" in pred_df.columns:
            log.info("[%s] per-batch test:", variant)
            m["per_batch"] = per_slice_metrics(y_test, test_p,
                                               pred_df["batch"].to_numpy(), "batch", log)
        if "region" in pred_df.columns and pred_df["region"].notna().any():
            log.info("[%s] per-region test:", variant)
            m["per_region"] = per_slice_metrics(y_test, test_p,
                                                pred_df["region"].fillna("UNK").to_numpy(),
                                                "region", log)

        with (var_dir / "metrics.json").open("w") as f:
            json.dump({"variant": variant, "arch": arch_name,
                       "wavlm_layer": layer,
                       "hidden": list(arch["hidden"]),
                       "dropout": arch["dropout"],
                       "weight_decay": arch["weight_decay"],
                       "label_smoothing": LABEL_SMOOTHING,
                       "grad_clip": GRAD_CLIP_NORM,
                       "pca": pca_var,
                       "augs_used": use_augs,
                       "train_only_batches": train_only_batches,
                       "min_duration": args.min_duration,
                       "class_balance": args.class_balance,
                       "in_dim": in_dim,
                       "best_val_f1": result.best_val_f1,
                       "best_epoch": result.best_epoch,
                       "test": m}, f, indent=2)

        ind = extract_region_metrics(m, "IND")
        php = extract_region_metrics(m, "PHP")
        summary_rows.append({
            "variant": variant, "arch": arch_name,
            "wavlm_layer": layer,
            "hidden": "x".join(str(h) for h in arch["hidden"]),
            "dropout": arch["dropout"], "wd": arch["weight_decay"],
            "pca": "full" if pca_var is None else f"{int(pca_var*100)}",
            "n_augs": len(use_augs), "in_dim": in_dim,
            "val_f1": round(result.best_val_f1, 4),
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
            "thr@p80": round(rap.get("p80", {}).get("threshold", float("nan")), 3),
            "thr@p90": round(rap.get("p90", {}).get("threshold", float("nan")), 3),
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
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary1.csv"
    summary.to_csv(summary_path, index=False)
    log.info("=" * 60)
    log.info("SUMMARY (NN, summary1.csv):\n%s", summary.to_string(index=False))
    log.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
