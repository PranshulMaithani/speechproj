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
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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

# WavLM layer choices. 'last' = last hidden state (encoder output);
# '9' = hidden_states[9], typically the strongest layer for paralinguistics.
WAVLM_LAYERS: list[str] = ["last", "9"]

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
# Logging.
# ----------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ----------------------------------------------------------------------------
# Splits.
# ----------------------------------------------------------------------------

def _kfold_indices(y: np.ndarray, g: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [te for _, te in skf.split(np.zeros(len(y)), y, g)]


def build_splits(gt: pd.DataFrame, train_batches: list[str],
                 test_batches: list[str] | None,
                 test_region_filter: str | None,
                 train_only_batches: list[str],
                 seed: int, log: logging.Logger) -> dict[str, np.ndarray]:
    """train_only_batches: batches whose rows are forced into the train split
    only -- never appear in val/test. Their group_ids are also stripped from
    the candidate-leak check (they're already disjoint by construction)."""
    train_only_set = set(train_only_batches or [])
    train_only_mask = gt["batch"].isin(train_only_set).to_numpy() if train_only_set else np.zeros(len(gt), dtype=bool)
    train_only_idx = np.where(train_only_mask)[0]
    if len(train_only_idx):
        log.info("train_only rows (auxiliary, forced to train): %d (batches=%s)",
                 len(train_only_idx), sorted(train_only_set))

    # The CV/split pool excludes train_only.
    train_pool_batches = [b for b in train_batches if b not in train_only_set]
    pool_mask = gt["batch"].isin(train_pool_batches).to_numpy()

    if not test_batches:
        sub = gt[pool_mask].reset_index().rename(columns={"index": "_orig"})
        if len(sub) == 0:
            raise ValueError(f"No rows match train_batches={train_pool_batches} "
                             f"(after excluding train_only={sorted(train_only_set)})")
        folds = _kfold_indices(sub["label"].to_numpy(), sub["group_id"].to_numpy(),
                               n_splits=5, seed=seed)
        test_local = folds[0]
        val_local = folds[1]
        used = set(test_local) | set(val_local)
        train_local = np.array([i for i in range(len(sub)) if i not in used])
        to_orig = sub["_orig"].to_numpy()
        train_idx = np.concatenate([to_orig[train_local], train_only_idx])
        log.info("Split mode A: 60/20/20 candidate-wise on %s  + %d train_only rows",
                 train_pool_batches, len(train_only_idx))
        return {"train": train_idx,
                "val":   to_orig[val_local],
                "test":  to_orig[test_local]}

    test_mask = gt["batch"].isin(test_batches).to_numpy()
    if test_region_filter:
        if "region" not in gt.columns:
            raise ValueError("region filter requested but gt has no 'region' column")
        test_mask &= (gt["region"].astype(str) == test_region_filter).to_numpy()

    test_idx = np.where(test_mask)[0]
    if len(test_idx) == 0:
        raise ValueError(f"No test rows match batches={test_batches} region={test_region_filter}")

    test_groups = set(gt.iloc[test_idx]["group_id"].tolist())
    eligible_mask = pool_mask & ~gt["group_id"].isin(test_groups).to_numpy() & ~test_mask
    sub = gt[eligible_mask].reset_index().rename(columns={"index": "_orig"})
    if len(sub) == 0:
        raise ValueError(f"No train-pool rows after excluding test groups; "
                         f"train_pool_batches={train_pool_batches}")

    val_local: np.ndarray | None = None
    if test_region_filter and "region" in gt.columns:
        same_region_mask = (sub["region"].astype(str) == test_region_filter).to_numpy()
        if same_region_mask.sum() >= 30 and len(np.unique(sub.loc[same_region_mask, "label"])) == 2:
            sub_r = sub[same_region_mask].reset_index(drop=True)
            r_folds = _kfold_indices(sub_r["label"].to_numpy(),
                                     sub_r["group_id"].to_numpy(), n_splits=5, seed=seed)
            sub_r_to_sub = np.where(same_region_mask)[0]
            val_local = sub_r_to_sub[r_folds[0]]
            log.info("Mode B val: drawn from region=%s subset of train pool (n=%d)",
                     test_region_filter, len(val_local))
        else:
            log.warning("Mode B: region=%s subset too small or single-class -- using mixed val",
                        test_region_filter)

    if val_local is None:
        folds = _kfold_indices(sub["label"].to_numpy(), sub["group_id"].to_numpy(),
                               n_splits=5, seed=seed)
        val_local = folds[0]

    train_local = np.array([i for i in range(len(sub)) if i not in set(val_local)])
    to_orig = sub["_orig"].to_numpy()
    train_idx = np.concatenate([to_orig[train_local], train_only_idx])
    log.info("Split mode B: train=%s test=%s region_filter=%s  + %d train_only rows",
             train_pool_batches, test_batches, test_region_filter, len(train_only_idx))
    return {"train": train_idx,
            "val":   to_orig[val_local],
            "test":  test_idx}


def assert_no_group_leak(gt: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    sets = {k: set(gt.iloc[idx]["group_id"]) for k, idx in splits.items()}
    assert sets["train"].isdisjoint(sets["test"]), "train/test group leak"
    assert sets["train"].isdisjoint(sets["val"]),  "train/val group leak"
    assert sets["val"].isdisjoint(sets["test"]),   "val/test group leak"


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
# Metrics.
# ----------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    out: dict = {}
    out["n"] = int(len(y_true))
    out["n_pos"] = int((y_true == 1).sum())
    out["n_neg"] = int((y_true == 0).sum())

    yhat = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
    out["thr0.5"] = {"precision": float(pr), "recall": float(rc), "f1": float(f1)}

    thrs = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (p >= t).astype(int), zero_division=0) for t in thrs]
    best_t = float(thrs[int(np.argmax(f1s))])
    yhat = (p >= best_t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
    out["best_f1"] = {"threshold": best_t, "precision": float(pr),
                     "recall": float(rc), "f1": float(f1)}

    base_rate = float(np.mean(y_true == 1))
    if base_rate > 0:
        k = max(1, int(round(base_rate * len(y_true))))
        order = np.argsort(-p)
        yhat = np.zeros_like(y_true)
        yhat[order[:k]] = 1
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        out["topk"] = {"k": k, "base_rate": base_rate, "precision": float(pr),
                       "recall": float(rc), "f1": float(f1)}

    try:
        out["auc"] = float(roc_auc_score(y_true, p))
        out["ap"] = float(average_precision_score(y_true, p))
    except ValueError:
        out["auc"] = float("nan")
        out["ap"] = float("nan")

    rap: dict[str, dict] = {}
    try:
        prec, rec, thr = precision_recall_curve(y_true, p)
        prec_t = prec[:-1]
        rec_t = rec[:-1]
        for target in (0.50, 0.80, 0.90, 0.95):
            mask = prec_t >= target
            if mask.any():
                idx = int(np.argmax(np.where(mask, rec_t, -1.0)))
                rap[f"p{int(target * 100)}"] = {
                    "target_precision": float(target),
                    "achieved_precision": float(prec_t[idx]),
                    "recall": float(rec_t[idx]),
                    "threshold": float(thr[idx]),
                }
            else:
                rap[f"p{int(target * 100)}"] = {
                    "target_precision": float(target),
                    "achieved_precision": float("nan"),
                    "recall": 0.0,
                    "threshold": float("nan"),
                }
    except ValueError:
        for target in (0.50, 0.80, 0.90, 0.95):
            rap[f"p{int(target * 100)}"] = {
                "target_precision": float(target),
                "achieved_precision": float("nan"),
                "recall": float("nan"),
                "threshold": float("nan"),
            }
    out["recall_at_precision"] = rap
    return out


def per_slice_metrics(y_true: np.ndarray, p: np.ndarray, slice_vals: np.ndarray,
                      slice_name: str, log: logging.Logger) -> dict:
    res = {}
    for v in pd.Series(slice_vals).dropna().unique():
        m = slice_vals == v
        if m.sum() < 5 or len(np.unique(y_true[m])) < 2:
            continue
        res[str(v)] = compute_metrics(y_true[m], p[m])
        log.info("  [%s=%s] n=%d  auc=%.3f  best_f1=%.3f@%.2f  thr0.5_f1=%.3f",
                 slice_name, v, int(m.sum()), res[str(v)]["auc"],
                 res[str(v)]["best_f1"]["f1"], res[str(v)]["best_f1"]["threshold"],
                 res[str(v)]["thr0.5"]["f1"])
    return res


# ----------------------------------------------------------------------------
# Cache helpers.
# ----------------------------------------------------------------------------

def load_cache_reindexed(cache_path: Path, filenames_cur: np.ndarray,
                         aug_names_needed: list[str], layers_needed: list[str],
                         log: logging.Logger
                         ) -> tuple[dict[tuple[str, str], np.ndarray],
                                    dict[str, np.ndarray]]:
    """Returns (wavlm_by_layer_aug, whisper_by_aug), both aligned to filenames_cur.

    Old caches (no 'wavlm_layers' key) are read as if they held only 'last'.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No embedding cache at {cache_path}. Run extract_embeddings.py first.")
    cache = dict(np.load(cache_path, allow_pickle=True))
    cached_filenames = cache["filenames"].astype(str)
    cached_augs = list(cache["aug_names"].astype(str))
    if "wavlm_layers" in cache:
        cached_layers = list(cache["wavlm_layers"].astype(str))
    else:
        cached_layers = ["last"]
        for a in cached_augs:
            if f"wavlm_{a}" in cache and f"wavlm_last_{a}" not in cache:
                cache[f"wavlm_last_{a}"] = cache[f"wavlm_{a}"]
    log.info("cache: %d filenames, augs=%s, layers=%s",
             len(cached_filenames), cached_augs, cached_layers)

    missing_augs = [a for a in aug_names_needed if a not in cached_augs]
    if missing_augs:
        raise ValueError(
            f"Augs requested but not in cache: {missing_augs}. "
            f"Cache has: {cached_augs}. Re-run extract_embeddings.py.")
    missing_layers = [l for l in layers_needed if l not in cached_layers]
    if missing_layers:
        raise ValueError(
            f"WavLM layers requested but not in cache: {missing_layers}. "
            f"Cache has: {cached_layers}. Re-run extract_embeddings.py with "
            f"--wavlm_layers including those.")

    f2c = {f: i for i, f in enumerate(cached_filenames)}
    missing_files = [f for f in filenames_cur if f not in f2c]
    if missing_files:
        raise ValueError(
            f"{len(missing_files)} gt rows have npy_filename not in the cache. "
            f"First few: {missing_files[:5]}. Re-run extract_embeddings.py.")

    order = np.array([f2c[f] for f in filenames_cur])
    wavlm_out: dict[tuple[str, str], np.ndarray] = {}
    whisper_out: dict[str, np.ndarray] = {}
    for layer in layers_needed:
        for a in aug_names_needed:
            wavlm_out[(layer, a)] = cache[f"wavlm_{layer}_{a}"][order]
    for a in aug_names_needed:
        whisper_out[a] = cache[f"whisper_{a}"][order]
    return wavlm_out, whisper_out


def resolve_use_augs(use_augs_arg: str, cache_path: Path,
                     log: logging.Logger) -> list[str]:
    """Parse --use_augs into a list of aug names that exist in the cache.
    'all' = every aug in the cache except 'orig'. Empty -> []."""
    s = use_augs_arg.strip()
    if not s:
        return []
    cache = dict(np.load(cache_path, allow_pickle=True))
    cached_augs = [a for a in cache["aug_names"].astype(str) if a != "orig"]
    if s.lower() == "all":
        log.info("--use_augs=all -> using every non-orig aug in cache: %s", cached_augs)
        return cached_augs
    wanted = [a.strip() for a in s.split(",") if a.strip() and a.strip() != "orig"]
    missing = [a for a in wanted if a not in cached_augs]
    if missing:
        raise ValueError(f"--use_augs requested {missing} but cache has {cached_augs}. "
                         f"Re-run extract_embeddings.py.")
    return wanted


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

    train_batches = [b.strip() for b in args.train_batches.split(",") if b.strip()]
    test_batches = [b.strip() for b in args.test_batches.split(",") if b.strip()]
    train_only_batches = [b.strip() for b in args.train_only_batches.split(",") if b.strip()]
    test_region_filter = args.test_region_filter.strip() or None

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (data_dir / "embeddings_cache.npz")

    log = setup_logging(out_dir / "log.txt")
    log.info("data_dir = %s", data_dir)
    log.info("out_dir  = %s", out_dir)
    log.info("cache    = %s", cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device   = %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load gt
    gt = pd.read_csv(data_dir / "gt.csv")
    log.info("Loaded gt.csv: %d rows", len(gt))

    gt["label"] = pd.to_numeric(gt["label"], errors="coerce")
    n_before = len(gt)
    gt = gt[gt["label"].isin([0, 1])].copy()
    gt["label"] = gt["label"].astype(int)
    gt = gt.reset_index(drop=True)
    log.info("Filtered to {0,1} labels: %d -> %d", n_before, len(gt))

    # ---- min_duration filter (applies to train + test before splits)
    if args.min_duration > 0.0:
        if "duration_sec" not in gt.columns:
            log.warning("--min_duration set but gt has no duration_sec column; skipping filter")
        else:
            n_before = len(gt)
            gt = gt[pd.to_numeric(gt["duration_sec"], errors="coerce")
                      .fillna(0.0) >= args.min_duration].reset_index(drop=True)
            log.info("min_duration=%.2fs filter: %d -> %d rows",
                     args.min_duration, n_before, len(gt))
            if len(gt) == 0:
                log.error("All rows filtered out by --min_duration")
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
    if feat_cols:
        feat_block = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feat_arr = feat_block.to_numpy().astype(np.float32)
        log.info("handcrafted feats: %d cols  first10=%s%s",
                 len(feat_cols), feat_cols[:10], " ..." if len(feat_cols) > 10 else "")
    else:
        feat_arr = None
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
    for k, idx in splits.items():
        labels = y_full[idx]
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        n_tot = max(len(idx), 1)
        log.info("split %-5s n=%4d  pos=%4d (%.1f%%)  neg=%4d (%.1f%%)  groups=%4d",
                 k, len(idx), n_pos, 100 * n_pos / n_tot,
                 n_neg, 100 * n_neg / n_tot,
                 gt.iloc[idx]["group_id"].nunique())
        if "batch" in gt.columns:
            log.info("split %-5s batches: %s", k,
                     gt.iloc[idx]["batch"].value_counts().to_dict())
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
            "recall@p90": round(rap.get("p90", {}).get("recall", float("nan")), 4),
            "recall@p95": round(rap.get("p95", {}).get("recall", float("nan")), 4),
            "thr@p80": round(rap.get("p80", {}).get("threshold", float("nan")), 3),
            "thr@p90": round(rap.get("p90", {}).get("threshold", float("nan")), 3),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("=" * 60)
    log.info("SUMMARY:\n%s", summary.to_string(index=False))
    log.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
