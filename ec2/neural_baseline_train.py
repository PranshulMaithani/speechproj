"""
EC2 (T4) baseline neural training for per-audio cheating detection.

Pipeline:
    1. Load gt.csv + audio_npy/ produced by companylaptop/neural_baseline_prep.py.
    2. Extract WavLM-base-plus mean-pooled (768d) and Whisper-medium encoder
       mean-pooled (1024d) embeddings. Concatenate -> 1792d. Cache to disk.
    3. Speaker-wise stratified split: 60% train / 20% val / 20% test
       (no group_id leak across splits).
    4. For each variant in {none, pca98, pca95, pca90}: train MLP, evaluate on
       held-out test split, write predictions and metrics.
    5. Print + save a summary table comparing variants.

Run:
    python neural_baseline_train.py \
        --data_dir /path/to/upload \
        --out_dir  /path/to/results \
        --batch_size 64 --epochs 60

The first run does encoder extraction (slow). Subsequent runs reuse the cache
in --out_dir/embeddings_cache.npz.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (AutoFeatureExtractor, WavLMModel, WhisperFeatureExtractor,
                          WhisperModel)

# ----------------------------------------------------------------------------
# Defaults / model identifiers
# ----------------------------------------------------------------------------

WAVLM_ID = "microsoft/wavlm-base-plus"
WHISPER_ID = "openai/whisper-medium"

WAVLM_CHUNK_SEC = 20.0   # WavLM has no fixed length but we chunk to control VRAM
WHISPER_CHUNK_SEC = 30.0  # Whisper is hard-coded to 30s windows
TARGET_SR = 16000

# Architectures. Tuned for the small-sample regime (~1000 labels): heavy
# dropout, larger weight decay, label smoothing, gradient clipping all
# enabled in train_one. The "tiny" variant is a single hidden layer with
# very strong regularization, intended as a hard-to-overfit baseline.
ARCH_CONFIG: dict[str, dict] = {
    "default": {"hidden": (512, 256, 128), "dropout": 0.40, "weight_decay": 5e-4},
    "tiny":    {"hidden": (128,),          "dropout": 0.55, "weight_decay": 5e-3},
}

# (variant_name, pca_variance_or_None, arch_name)
VARIANTS_CONFIG: list[tuple[str, float | None, str]] = [
    ("none",  None, "default"),
    ("pca98", 0.98, "default"),
    ("pca95", 0.95, "default"),
    ("pca90", 0.90, "default"),
    ("tiny",  None, "tiny"),
]
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0


# ----------------------------------------------------------------------------
# Logging
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
# Embedding extraction
# ----------------------------------------------------------------------------

@torch.no_grad()
def extract_wavlm_meanpool(wav: np.ndarray, model: WavLMModel, device: torch.device) -> np.ndarray:
    """Chunk a 16kHz mono waveform, run WavLM, mean-pool last hidden state across all
    frames of all chunks. Returns (768,) float32."""
    chunk = int(WAVLM_CHUNK_SEC * TARGET_SR)
    if len(wav) <= chunk:
        chunks = [wav]
    else:
        chunks = [wav[i:i + chunk] for i in range(0, len(wav), chunk)]
    pooled = []
    weights = []
    for c in chunks:
        if len(c) < TARGET_SR * 0.4:  # under 0.4s = noise
            continue
        x = torch.from_numpy(c).float().unsqueeze(0).to(device)
        out = model(x).last_hidden_state  # (1, T, 768)
        pooled.append(out.mean(dim=1).squeeze(0).cpu().numpy())
        weights.append(out.shape[1])
    if not pooled:
        return np.zeros(model.config.hidden_size, dtype=np.float32)
    P = np.stack(pooled, axis=0)
    W = np.array(weights, dtype=np.float32)
    return (P * (W[:, None] / W.sum())).sum(axis=0).astype(np.float32)


@torch.no_grad()
def extract_whisper_meanpool(wav: np.ndarray, model: WhisperModel,
                             feat: WhisperFeatureExtractor, device: torch.device) -> np.ndarray:
    """Chunk to 30s windows, run Whisper encoder, mean-pool over time across all
    chunks. Returns (1024,) float32 for whisper-medium."""
    chunk = int(WHISPER_CHUNK_SEC * TARGET_SR)
    if len(wav) <= chunk:
        chunks = [wav]
    else:
        chunks = [wav[i:i + chunk] for i in range(0, len(wav), chunk)]
    pooled = []
    for c in chunks:
        if len(c) < TARGET_SR * 0.4:
            continue
        # Whisper feature extractor pads/truncates to 30s automatically.
        feats = feat(c, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(device)
        enc = model.encoder(feats).last_hidden_state  # (1, T_enc, 1024)
        pooled.append(enc.mean(dim=1).squeeze(0).cpu().numpy())
    if not pooled:
        return np.zeros(model.config.d_model, dtype=np.float32)
    return np.stack(pooled, axis=0).mean(axis=0).astype(np.float32)


def extract_all_embeddings(gt: pd.DataFrame, npy_dir: Path, device: torch.device,
                           log: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    log.info("Loading WavLM (%s)...", WAVLM_ID)
    wavlm = WavLMModel.from_pretrained(WAVLM_ID).to(device).eval()
    log.info("Loading Whisper-medium (%s)...", WHISPER_ID)
    whisper = WhisperModel.from_pretrained(WHISPER_ID).to(device).eval()
    whisper_feat = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

    wavlm_dim = wavlm.config.hidden_size
    whisper_dim = whisper.config.d_model
    log.info("WavLM dim = %d, Whisper dim = %d, concat = %d", wavlm_dim, whisper_dim,
             wavlm_dim + whisper_dim)

    wavlm_emb = np.zeros((len(gt), wavlm_dim), dtype=np.float32)
    whisper_emb = np.zeros((len(gt), whisper_dim), dtype=np.float32)

    t0 = time.time()
    for i, row in tqdm(list(gt.iterrows()), desc="extract", unit="audio"):
        path = npy_dir / row["npy_filename"]
        try:
            wav = np.load(path).astype(np.float32, copy=False)
        except Exception as e:
            log.warning("Failed to load %s: %s -- using zeros", path, e)
            continue
        wavlm_emb[i] = extract_wavlm_meanpool(wav, wavlm, device)
        whisper_emb[i] = extract_whisper_meanpool(wav, whisper, whisper_feat, device)

    log.info("Extraction done in %.1f s", time.time() - t0)

    # Free GPU.
    del wavlm, whisper
    torch.cuda.empty_cache()
    return wavlm_emb, whisper_emb


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------

def _kfold_indices(y: np.ndarray, g: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [te for _, te in skf.split(np.zeros(len(y)), y, g)]


def build_splits(gt: pd.DataFrame, train_batches: list[str],
                 test_batches: list[str] | None,
                 test_region_filter: str | None,
                 seed: int = 42, log: logging.Logger | None = None) -> dict[str, np.ndarray]:
    """Build train/val/test row-index arrays into the original gt frame.

    Two modes:
      A) test_batches is None or empty -> filter gt to train_batches, then
         StratifiedGroupKFold(5): fold 0 = test (~20%), fold 1 = val (~20%),
         folds 2-4 = train (~60%). Speaker-disjoint, label-stratified.
      B) test_batches non-empty -> test = gt rows in those batches (optionally
         filtered by region == test_region_filter). Train pool = gt rows in
         train_batches; val is StratifiedGroupKFold(5) fold 0 of train pool;
         remainder is train. Any train-pool row whose group_id appears in test
         is dropped from train+val (defensive against shared candidates).
    """
    log = log or logging.getLogger("train")
    train_mask = gt["batch"].isin(train_batches).to_numpy()

    if not test_batches:
        sub = gt[train_mask].reset_index().rename(columns={"index": "_orig"})
        if len(sub) == 0:
            raise ValueError(f"No rows match train_batches={train_batches}")
        folds = _kfold_indices(sub["label"].to_numpy(), sub["group_id"].to_numpy(),
                               n_splits=5, seed=seed)
        test_local = folds[0]
        val_local = folds[1]
        used = set(test_local) | set(val_local)
        train_local = np.array([i for i in range(len(sub)) if i not in used])
        to_orig = sub["_orig"].to_numpy()
        log.info("Split mode A: 60/20/20 candidate-wise on %s", train_batches)
        return {"train": to_orig[train_local],
                "val":   to_orig[val_local],
                "test":  to_orig[test_local]}

    # Mode B: explicit test batches.
    test_mask = gt["batch"].isin(test_batches).to_numpy()
    if test_region_filter:
        if "region" not in gt.columns:
            raise ValueError("region filter requested but gt has no 'region' column")
        test_mask &= (gt["region"].astype(str) == test_region_filter).to_numpy()

    test_idx = np.where(test_mask)[0]
    if len(test_idx) == 0:
        raise ValueError(f"No test rows match batches={test_batches} region={test_region_filter}")

    test_groups = set(gt.iloc[test_idx]["group_id"].tolist())
    train_pool_mask = train_mask & ~gt["group_id"].isin(test_groups).to_numpy() & ~test_mask
    sub = gt[train_pool_mask].reset_index().rename(columns={"index": "_orig"})
    if len(sub) == 0:
        raise ValueError(f"No train-pool rows after excluding test groups; train_batches={train_batches}")

    # If test is region-filtered, prefer val from same region so early stopping
    # tracks the distribution we actually care about. Fall back to unfiltered
    # val if same-region pool is too small or single-class.
    val_local: np.ndarray | None = None
    if test_region_filter and "region" in gt.columns:
        same_region_mask = (sub["region"].astype(str) == test_region_filter).to_numpy()
        if same_region_mask.sum() >= 30 and len(np.unique(sub.loc[same_region_mask, "label"])) == 2:
            sub_r = sub[same_region_mask].reset_index(drop=True)
            r_folds = _kfold_indices(sub_r["label"].to_numpy(),
                                     sub_r["group_id"].to_numpy(),
                                     n_splits=5, seed=seed)
            sub_r_local = r_folds[0]
            # Map sub_r positions back to positions inside sub.
            sub_r_to_sub = np.where(same_region_mask)[0]
            val_local = sub_r_to_sub[sub_r_local]
            log.info("Mode B val: drawn from region=%s subset of train pool (n=%d)",
                     test_region_filter, len(val_local))
        else:
            log.warning("Mode B: region=%s subset of train pool too small or single-class; "
                        "falling back to mixed-region val (early stopping may track wrong distribution)",
                        test_region_filter)

    if val_local is None:
        folds = _kfold_indices(sub["label"].to_numpy(), sub["group_id"].to_numpy(),
                               n_splits=5, seed=seed)
        val_local = folds[0]

    train_local = np.array([i for i in range(len(sub)) if i not in set(val_local)])
    to_orig = sub["_orig"].to_numpy()
    log.info("Split mode B: train=%s test=%s region_filter=%s",
             train_batches, test_batches, test_region_filter)
    return {"train": to_orig[train_local],
            "val":   to_orig[val_local],
            "test":  test_idx}


def assert_no_group_leak(gt: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    sets = {k: set(gt.iloc[idx]["group_id"]) for k, idx in splits.items()}
    assert sets["train"].isdisjoint(sets["test"]), "train/test group leak"
    assert sets["train"].isdisjoint(sets["val"]),  "train/val group leak"
    assert sets["val"].isdisjoint(sets["test"]),   "val/test group leak"


# ----------------------------------------------------------------------------
# MLP
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
    """BCE-with-logits with symmetric binary label smoothing.
    target=1 -> 1 - smoothing/2,  target=0 -> smoothing/2."""
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
              hidden: tuple = (512, 256, 128), dropout: float = 0.4,
              label_smoothing: float = LABEL_SMOOTHING,
              grad_clip: float = GRAD_CLIP_NORM,
              patience: int = 10) -> tuple["TrainResult", np.ndarray]:
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    log.info("[%s] arch hidden=%s dropout=%.2f wd=%.1e ls=%.2f clip=%.1f",
             tag, hidden, dropout, wd, label_smoothing, grad_clip)
    log.info("[%s] train n=%d (pos=%d/neg=%d, pw=%.2f)  val n=%d  test n=%d  in_dim=%d",
             tag, len(y_tr), int(pos), int(neg), pos_weight.item(), len(y_va), len(y_te), in_dim)

    model = MLP(in_dim, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def to_tensor(X, y):
        return TensorDataset(torch.from_numpy(X.astype(np.float32)),
                             torch.from_numpy(y.astype(np.float32)))

    # drop_last=True so a final batch of size 1 doesn't crash BatchNorm during
    # training. eval/predict path doesn't use BN running-stats updates so it's safe.
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
        ep_loss /= len(y_tr)

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
            log.info("[%s] ep %3d  loss %.4f  val_f1 %.4f  best %.4f@%d", tag, ep, ep_loss,
                     val_f1, best_val_f1, best_epoch)
        if patience_left <= 0:
            log.info("[%s] early stop at ep %d (best val_f1 %.4f @ ep %d)", tag, ep,
                     best_val_f1, best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_p = predict(X_te)
    metrics = compute_metrics(y_te, test_p)
    return TrainResult(best_val_f1=best_val_f1, best_epoch=best_epoch, test_metrics=metrics), test_p


def compute_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    out: dict = {}
    out["n"] = int(len(y_true))
    out["n_pos"] = int((y_true == 1).sum())
    out["n_neg"] = int((y_true == 0).sum())

    # Threshold = 0.5
    yhat = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
    out["thr0.5"] = {"precision": float(pr), "recall": float(rc), "f1": float(f1)}

    # Best-F1 threshold sweep
    thrs = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (p >= t).astype(int), zero_division=0) for t in thrs]
    best_t = float(thrs[int(np.argmax(f1s))])
    yhat = (p >= best_t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
    out["best_f1"] = {"threshold": best_t, "precision": float(pr), "recall": float(rc), "f1": float(f1)}

    # Top-K% rank rule, K matched to base rate
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

    # Recall at fixed precision targets. precision_recall_curve returns
    # arrays where precision[i]/recall[i] correspond to thresholds[i] (and an
    # extra final point precision=1.0/recall=0.0). For each target we pick
    # the operating point with precision >= target that maximises recall.
    rap: dict[str, dict] = {}
    try:
        prec, rec, thr = precision_recall_curve(y_true, p)
        # Drop the trailing (precision=1, recall=0) sentinel that has no threshold.
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
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder containing gt.csv and audio_npy/")
    ap.add_argument("--out_dir", required=True, help="where to write results")
    ap.add_argument("--train_batches", default="audios2,audios4,audios5,audios6",
                    help="comma-separated batch names used for training")
    ap.add_argument("--test_batches", default="",
                    help="comma-separated batch names for held-out test. "
                         "Empty => 80/20 candidate-wise split on train_batches.")
    ap.add_argument("--test_region_filter", default="",
                    help="when test_batches is set, restrict test to rows where "
                         "region == this value (e.g. IND). Empty = no filter.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_extract", action="store_true",
                    help="ignore embeddings_cache.npz and re-extract from audio")
    args = ap.parse_args()

    train_batches = [b.strip() for b in args.train_batches.split(",") if b.strip()]
    test_batches = [b.strip() for b in args.test_batches.split(",") if b.strip()]
    test_region_filter = args.test_region_filter.strip() or None

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(out_dir / "log.txt")
    log.info("data_dir=%s  out_dir=%s", data_dir, out_dir)
    log.info("device count: %d", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("using device: %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gt_path = data_dir / "gt.csv"
    npy_dir = data_dir / "audio_npy"
    gt = pd.read_csv(gt_path)
    log.info("Loaded gt.csv: %d rows", len(gt))

    # Robust label parsing: tolerate strings like "0"/"1"/"" or floats with NaN.
    gt["label"] = pd.to_numeric(gt["label"], errors="coerce")
    n_before = len(gt)
    gt = gt[gt["label"].isin([0, 1])].copy()
    gt["label"] = gt["label"].astype(int)
    gt = gt.reset_index(drop=True)
    log.info("Filtered to labeled rows (label in {0,1}): %d -> %d", n_before, len(gt))

    requested_train = set(train_batches)
    requested_test = set(test_batches)
    in_either = gt["batch"].isin(requested_train | requested_test)
    n_unused = int((~in_either).sum())
    if n_unused:
        log.warning("%d gt rows are not in train_batches or test_batches and will be unused (batches present: %s)",
                    n_unused, sorted(gt.loc[~in_either, "batch"].unique().tolist()))
    log.info("Label dist: %s", gt["label"].value_counts().to_dict())
    if "region" in gt.columns and gt["region"].notna().any():
        log.info("Region dist: %s", gt["region"].value_counts(dropna=False).to_dict())
    if "batch" in gt.columns:
        log.info("Batch dist: %s", gt["batch"].value_counts().to_dict())

    cache_path = out_dir / "embeddings_cache.npz"
    use_cache = cache_path.exists() and not args.force_extract
    if use_cache:
        cache = np.load(cache_path)
        cached_filenames = cache["filenames"].astype(str)
        # Build a lookup so we can re-index the cache to current gt order even
        # if the gt was filtered differently between runs.
        f2idx = {f: i for i, f in enumerate(cached_filenames)}
        cur_files = gt["npy_filename"].to_numpy()
        missing = [f for f in cur_files if f not in f2idx]
        if missing:
            log.warning("Cache is missing %d filenames in current gt; re-extracting fully", len(missing))
            wavlm_emb, whisper_emb = extract_all_embeddings(gt, npy_dir, device, log)
            np.savez(cache_path, wavlm=wavlm_emb, whisper=whisper_emb, filenames=cur_files)
        else:
            order = np.array([f2idx[f] for f in cur_files])
            wavlm_emb = cache["wavlm"][order]
            whisper_emb = cache["whisper"][order]
            log.info("Loaded cached embeddings from %s (reindexed to current gt order)", cache_path)
    else:
        wavlm_emb, whisper_emb = extract_all_embeddings(gt, npy_dir, device, log)
        np.savez(cache_path, wavlm=wavlm_emb, whisper=whisper_emb,
                 filenames=gt["npy_filename"].to_numpy())
        log.info("Saved embedding cache to %s", cache_path)

    # Pull handcrafted features (computed on the company laptop and written
    # into gt.csv as feat_* columns -- text + pause + prosodic + voice quality
    # if the v3 features.csv was used, or text-only via compute_text_features
    # as a fallback). Numeric only; no PII.
    feat_cols = [c for c in gt.columns if c.startswith("feat_")]
    if feat_cols:
        feat_block = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feat_arr = feat_block.to_numpy().astype(np.float32)
        log.info("Handcrafted features detected (%d columns): first 10 = %s%s",
                 len(feat_cols), feat_cols[:10],
                 " ..." if len(feat_cols) > 10 else "")
        X_full = np.concatenate([wavlm_emb, whisper_emb, feat_arr], axis=1).astype(np.float32)
        log.info("Concat shape (wavlm + whisper + feats): %s", X_full.shape)
    else:
        log.info("No feat_* columns in gt.csv -- training on audio embeddings only")
        X_full = np.concatenate([wavlm_emb, whisper_emb], axis=1).astype(np.float32)
        log.info("Concat shape (wavlm + whisper): %s", X_full.shape)
    y_full = gt["label"].to_numpy().astype(np.int64)

    # Sanity: catch all-zero rows (extraction silently failed, returned zeros).
    zero_rows = (np.abs(X_full).sum(axis=1) == 0)
    if zero_rows.any():
        log.warning("%d rows have all-zero embeddings (extraction failed). They will train/test as zeros.",
                    int(zero_rows.sum()))

    # ---- Splits
    splits = build_splits(gt, train_batches=train_batches,
                          test_batches=test_batches if test_batches else None,
                          test_region_filter=test_region_filter,
                          seed=args.seed, log=log)
    assert_no_group_leak(gt, splits)
    for k, idx in splits.items():
        labels = y_full[idx]
        log.info("split %-5s n=%4d  pos=%4d  neg=%4d  unique_groups=%4d",
                 k, len(idx), int((labels == 1).sum()), int((labels == 0).sum()),
                 gt.iloc[idx]["group_id"].nunique())

    summary_rows = []

    # Per-feature standardization, fit ONLY on train. Applied to all splits.
    # Important for an MLP on raw frozen-encoder embeddings (different feature
    # scales = unstable optimization). PCA is fit on the standardized train.
    scaler = StandardScaler().fit(X_full[splits["train"]])
    X_full_std = scaler.transform(X_full).astype(np.float32)

    for variant, pca_var, arch_name in VARIANTS_CONFIG:
        log.info("=" * 60)
        arch = ARCH_CONFIG[arch_name]
        log.info("VARIANT: %s  arch=%s  pca=%s", variant, arch_name, pca_var)

        if pca_var is None:
            X = X_full_std
            in_dim = X.shape[1]
            pca = None
        else:
            pca = PCA(n_components=pca_var, svd_solver="full", random_state=args.seed)
            pca.fit(X_full_std[splits["train"]])
            X = pca.transform(X_full_std).astype(np.float32)
            in_dim = X.shape[1]
            log.info("PCA(%.2f) -> %d components", pca_var, in_dim)

        Xt, Xv, Xs = X[splits["train"]], X[splits["val"]], X[splits["test"]]
        yt, yv, ys = y_full[splits["train"]], y_full[splits["val"]], y_full[splits["test"]]

        result, test_p = train_one(Xt, yt, Xv, yv, Xs, ys,
                                   in_dim=in_dim, batch_size=args.batch_size,
                                   epochs=args.epochs, lr=args.lr,
                                   wd=arch["weight_decay"],
                                   hidden=tuple(arch["hidden"]),
                                   dropout=arch["dropout"],
                                   device=device, log=log, tag=variant)

        # Save per-variant
        var_dir = out_dir / variant
        var_dir.mkdir(parents=True, exist_ok=True)
        pred_df = gt.iloc[splits["test"]].copy()
        pred_df["pred_score"] = test_p
        pred_df.to_csv(var_dir / "predictions.csv", index=False)

        m = result.test_metrics
        log.info("[%s] TEST  n=%d  auc=%.3f  ap=%.3f  thr0.5 f1=%.3f  best_f1=%.3f@%.2f",
                 variant, m["n"], m["auc"], m["ap"], m["thr0.5"]["f1"],
                 m["best_f1"]["f1"], m["best_f1"]["threshold"])
        if "topk" in m:
            log.info("[%s] TEST  topk(k=%d, base=%.2f) f1=%.3f p=%.3f r=%.3f",
                     variant, m["topk"]["k"], m["topk"]["base_rate"],
                     m["topk"]["f1"], m["topk"]["precision"], m["topk"]["recall"])

        # Per-batch slice
        if "batch" in pred_df.columns:
            log.info("[%s] per-batch on test:", variant)
            m["per_batch"] = per_slice_metrics(ys, test_p,
                                               pred_df["batch"].to_numpy(),
                                               "batch", log)
        # Per-region slice
        if "region" in pred_df.columns and pred_df["region"].notna().any():
            log.info("[%s] per-region on test:", variant)
            m["per_region"] = per_slice_metrics(ys, test_p,
                                                pred_df["region"].fillna("UNK").to_numpy(),
                                                "region", log)

        with (var_dir / "metrics.json").open("w") as f:
            json.dump({"variant": variant,
                       "arch": arch_name,
                       "hidden": list(arch["hidden"]),
                       "dropout": arch["dropout"],
                       "weight_decay": arch["weight_decay"],
                       "label_smoothing": LABEL_SMOOTHING,
                       "grad_clip": GRAD_CLIP_NORM,
                       "pca": pca_var,
                       "best_val_f1": result.best_val_f1,
                       "best_epoch": result.best_epoch,
                       "in_dim": in_dim,
                       "test": m}, f, indent=2)

        rap = m.get("recall_at_precision", {})
        log.info("[%s] recall@precision  p50=%.3f  p80=%.3f  p90=%.3f  p95=%.3f", variant,
                 rap.get("p50", {}).get("recall", float("nan")),
                 rap.get("p80", {}).get("recall", float("nan")),
                 rap.get("p90", {}).get("recall", float("nan")),
                 rap.get("p95", {}).get("recall", float("nan")))

        summary_rows.append({
            "variant": variant,
            "arch": arch_name,
            "hidden": "x".join(str(h) for h in arch["hidden"]),
            "dropout": arch["dropout"],
            "wd": arch["weight_decay"],
            "in_dim": in_dim,
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
    log.info("Wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
