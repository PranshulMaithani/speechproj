"""Per-audio cheating classifier training -- ABLATION COPY of neural_baseline_train.py.

This is a byte-faithful copy of the sweep PLUS an in-sweep augmentation ablation.
Why a copy (not aug_ablation2.py): reproducing a finalised variant's HEADLINE
number requires the exact RNG state it had at its slot in the 30-variant sweep.
aug_ablation2.py reseeds per config, so even at seed 42 it does not reproduce
(the 48->30 gap). Here the full sweep runs unchanged, so when it reaches a
--ablate_variants target the RNG is exactly the headline state; we snapshot it,
train 'all' (= reproduces the sheet number), then each drop_<aug>/none config
from that SAME state, then restore the post-train RNG so later variants stay
byte-identical. Run this with the SAME command/flags as the variant's original
run (e.g. the m2_nocasual command + --use_augs all --ablate_variants
tiny_l9_pca95). Extra outputs: ablation_sweep.csv / .xlsx.

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

import joblib
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
                            per_client_standardize, per_slice_metrics,
                            resolve_use_augs, setup_logging)

# ----------------------------------------------------------------------------
# Ablation (this is the copy of neural_baseline_train.py): when the sweep
# reaches a target variant it ALSO trains every leave-one-out aug config from
# the SAME pre-train RNG state, so the 'all' config bit-reproduces the headline
# sheet number and every drop_<aug>/none config is compared from the identical
# initialization. The main sweep RNG is saved/restored around these extra runs
# so later variants are unperturbed and still reproduce.
# ----------------------------------------------------------------------------

def _snapshot_rng(device: torch.device) -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if device.type == "cuda":
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _restore_rng(s: dict, device: torch.device) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    if device.type == "cuda" and "cuda" in s:
        torch.cuda.set_rng_state_all(s["cuda"])


def _build_cfg_xy(X_by_aug: dict, train_idx, val_idx, test_idx, y_train_base,
                  use_augs: list[str], pca_var: float | None, seed: int):
    """Expand TRAIN with use_augs (val/test stay orig), fit scaler + PCA on the
    aug-expanded train only -- identical pipeline to the main sweep, so the 'all'
    config reproduces the variant's headline features exactly."""
    X_orig = X_by_aug["orig"]
    Xtr_blocks = [X_orig[train_idx]]
    ytr_blocks = [y_train_base]
    for a in use_augs:
        Xtr_blocks.append(X_by_aug[a][train_idx])
        ytr_blocks.append(y_train_base)
    X_train = np.concatenate(Xtr_blocks, axis=0).astype(np.float32)
    y_tr = np.concatenate(ytr_blocks, axis=0).astype(np.int64)
    X_val = X_orig[val_idx]
    X_test = X_orig[test_idx]
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)
    if pca_var is not None:
        pca = PCA(n_components=pca_var, svd_solver="full", random_state=seed)
        pca.fit(Xtr)
        Xtr = pca.transform(Xtr).astype(np.float32)
        Xva = pca.transform(Xva).astype(np.float32)
        Xte = pca.transform(Xte).astype(np.float32)
    return Xtr, y_tr, Xva, Xte


# ----------------------------------------------------------------------------
# Variants.
# ----------------------------------------------------------------------------

ARCH_CONFIG: dict[str, dict] = {
    "default": {"hidden": (512, 256, 128), "dropout": 0.40, "weight_decay": 5e-4},
    "tiny":    {"hidden": (128,),          "dropout": 0.55, "weight_decay": 5e-3},
    # 'linear' is logistic-regression-on-WavLM+Whisper: no hidden layer, no
    # nonlinearity. Heavy L2. Sanity baseline -- if it matches 'tiny', the
    # data is essentially linear in this feature space and the MLP nonlinearity
    # is just memorizing client artifacts.
    "linear":  {"hidden": (),              "dropout": 0.00, "weight_decay": 1e-2},
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
    # name -> predictions on full-data matrix for that aug (orig + each aug used).
    extra_preds: dict[str, np.ndarray] | None = None
    # best-epoch MLP weights (cpu tensors); only set when export_model=True.
    model_state: dict | None = None


def train_one(X_tr, y_tr, X_va, y_va, X_te, y_te, *, in_dim: int, batch_size: int,
              epochs: int, lr: float, wd: float, device: torch.device,
              log: logging.Logger, tag: str,
              hidden: tuple, dropout: float,
              label_smoothing: float = LABEL_SMOOTHING,
              grad_clip: float = GRAD_CLIP_NORM,
              patience: int = 10,
              class_balance: str = "sampler",
              extra_predicts: dict[str, np.ndarray] | None = None,
              export_model: bool = False,
              ) -> tuple[TrainResult, np.ndarray]:
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
    extra_p = ({name: predict(X) for name, X in extra_predicts.items()}
               if extra_predicts else None)
    model_state = ({k: v.cpu() for k, v in best_state.items()}
                   if (export_model and best_state is not None) else None)
    return TrainResult(best_val_f1, best_epoch, metrics, extra_preds=extra_p,
                       model_state=model_state), test_p


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
    ap.add_argument("--per_client_standardize", default="false",
                    choices=["true", "false"],
                    help="Per-client feature standardization (centers each "
                         "client's WavLM+Whisper+feat on its own mean/std "
                         "using features only). Removes first-order client "
                         "shift -- the main reason audios6 F1 collapses 19pt "
                         "from the 20%% split.")
    ap.add_argument("--fewshot_frac", type=float, default=0.0,
                    help="Few-shot adaptation: fraction of test-batch "
                         "candidates carved into train (candidate-disjoint, "
                         "rows removed from test). Use with --test_batches "
                         "set (Mode B). e.g. 0.20 puts 20%% of audios6 "
                         "candidates into train and tests on the other 80%%.")
    ap.add_argument("--dump_full_predictions", default="false",
                    choices=["true", "false"],
                    help="For every variant, predict on ALL rows (train+val+"
                         "test, orig embeddings) with the trained model and write "
                         "<variant>/full_predictions.csv (npy_filename, group_id, "
                         "question_id, batch, region, label, split, pred_prob). "
                         "Used by build_analysis_excel.py. Reuses the exact "
                         "trained model so test rows equal predictions.csv.")
    ap.add_argument("--export_artifacts", default="false",
                    choices=["true", "false"],
                    help="For every variant, save the trained MLP weights "
                         "(model.pt), the fitted StandardScaler (scaler.joblib), "
                         "the PCA (pca.joblib, omitted for pca=full) and an "
                         "inference_meta.json into <out_dir>/<variant>/. Lets you "
                         "download a variant and run inference offline. Re-run "
                         "the SAME full sweep with this flag to export the exact "
                         "weights behind your finalised numbers.")
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
    ap.add_argument("--ablate", default="true", choices=["true", "false"],
                    help="When a target variant is reached in the sweep, ALSO "
                         "train every leave-one-out aug config (all, drop_<aug> "
                         "x7, none) from the SAME pre-train RNG state. The 'all' "
                         "config reproduces that variant's headline number; the "
                         "rest isolate each aug's effect. Needs --use_augs all.")
    ap.add_argument("--ablate_variants", default="tiny_l9_pca95",
                    help="comma-separated variant names to ablate. Only reproduces "
                         "a variant's sheet number when THIS run uses that "
                         "variant's original flags (e.g. run with the m2_nocasual "
                         "command to ablate tiny_l9_pca95; the m1_casual command "
                         "to ablate default_last_pca98).")
    args = ap.parse_args()

    use_text_features = (args.use_text_features == "true")
    do_per_client_std = (args.per_client_standardize == "true")
    dump_full = (args.dump_full_predictions == "true")
    export_artifacts = (args.export_artifacts == "true")
    ablate_on = (args.ablate == "true")
    ablate_set = {v.strip() for v in args.ablate_variants.split(",") if v.strip()}
    _vmap = {v[0]: v[2] for v in make_variants()}
    target_layers = {_vmap[v] for v in ablate_set if v in _vmap}
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
    log.info("use_text_features      = %s", use_text_features)
    log.info("per_client_standardize = %s", do_per_client_std)
    log.info("fewshot_frac           = %.2f", args.fewshot_frac)
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

    # ---- Per-client feature standardization (unsupervised; uses features
    # only). Best applied before splits so the model never sees client-drift
    # variance.
    if do_per_client_std:
        feat_arr = per_client_standardize(wavlm_cache, whisper_cache, gt, log,
                                          feat_arr=feat_arr)

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
                          seed=args.seed, log=log,
                          fewshot_frac=args.fewshot_frac)
    assert_no_group_leak(gt, splits)
    log_split_breakdown(gt, splits, log)
    log.info("--class_balance=%s will be applied to TRAIN minibatches "
             "(val/test always use natural distribution)", args.class_balance)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    # Per-row split label for the full-prediction dump (train/val/test/unused).
    split_label = np.array(["unused"] * len(gt), dtype=object)
    split_label[train_idx] = "train"
    split_label[val_idx] = "val"
    split_label[test_idx] = "test"
    feat_train = feat_arr[train_idx] if feat_arr is not None else None
    y_train_base = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]

    # ---- Per-layer (X_train aug-expanded, X_val, X_test, scaler) ----
    # We pre-build all features per WavLM layer once so the variant loop only
    # has to fit PCA + MLP.
    per_layer: dict[str, dict] = {}
    for layer in WAVLM_LAYERS:
        # X_by_aug[name] = full-data concat(wavlm[name], whisper[name], feats)
        # for every aug requested. Built once per layer and reused for both
        # the train aug-expansion and the dump_full per-aug prediction.
        X_by_aug: dict[str, np.ndarray] = {
            "orig": concat(wavlm_cache[(layer, "orig")], whisper_cache["orig"], feat_arr)
        }
        for a in use_augs:
            X_by_aug[a] = concat(wavlm_cache[(layer, a)], whisper_cache[a], feat_arr)

        X_orig = X_by_aug["orig"]
        zero_rows = (np.abs(X_orig).sum(axis=1) == 0)
        if zero_rows.any():
            log.warning("[layer=%s] %d rows have all-zero orig embeddings",
                        layer, int(zero_rows.sum()))

        X_train_blocks = [X_orig[train_idx]]
        y_train_blocks = [y_train_base]
        aug_tag_blocks: list[list[str]] = [["orig"] * len(train_idx)]
        for a in use_augs:
            X_train_blocks.append(X_by_aug[a][train_idx])
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
            "scaler": scaler,  # needed by --export_artifacts
        }
        # Keep the UNSCALED per-aug full matrices for layers an ablation target
        # uses, so the ablation branch can re-expand train with each aug subset
        # and refit scaler/PCA per config (build_cfg_xy).
        if ablate_on and layer in target_layers:
            per_layer[layer]["X_by_aug"] = X_by_aug
        # For --dump_full_predictions: ALL rows, ONE matrix PER aug, through
        # the SAME scaler. Used by build_analysis_excel to score augmented
        # copies of train rows -- if the model classifies aug copies correctly
        # too, it generalises within the aug neighbourhood (not just memorising
        # clean train rows).
        if dump_full:
            per_layer[layer]["X_all_by_aug_s"] = {
                name: scaler.transform(X).astype(np.float32)
                for name, X in X_by_aug.items()
            }

    # ---- Common metadata for --export_artifacts (computed once) ----
    if export_artifacts:
        exp_wavlm_dim = int(wavlm_cache[(WAVLM_LAYERS[0], "orig")].shape[1])
        exp_whisper_dim = int(whisper_cache["orig"].shape[1])
        try:
            _cmeta = np.load(cache_path, allow_pickle=True)
            exp_wavlm_id = str(_cmeta["wavlm_id"]) if "wavlm_id" in _cmeta else ""
            exp_whisper_id = str(_cmeta["whisper_id"]) if "whisper_id" in _cmeta else ""
        except Exception:
            exp_wavlm_id = exp_whisper_id = ""
        log.info("export_artifacts ON: wavlm_dim=%d whisper_dim=%d feat=%d",
                 exp_wavlm_dim, exp_whisper_dim,
                 len(feat_cols) if (feat_cols and use_text_features) else 0)

    # ---- Iterate variants (20 = 2 archs x 2 wavlm layers x 5 PCAs) ----
    summary_rows: list[dict] = []
    ablation_rows: list[dict] = []
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
            X_all_by_aug = lp["X_all_by_aug_s"] if dump_full else None
        else:
            pca = PCA(n_components=pca_var, svd_solver="full", random_state=args.seed)
            pca.fit(X_train_s)
            Xt = pca.transform(X_train_s).astype(np.float32)
            Xv = pca.transform(X_val_s).astype(np.float32)
            Xs = pca.transform(X_test_s).astype(np.float32)
            in_dim = Xt.shape[1]
            log.info("PCA(%.2f) -> %d components", pca_var, in_dim)
            X_all_by_aug = (
                {name: pca.transform(X).astype(np.float32)
                 for name, X in lp["X_all_by_aug_s"].items()}
                if dump_full else None)

        # Snapshot RNG right BEFORE the main training so every ablation config
        # for this variant can start from the identical initialization.
        rng0 = _snapshot_rng(device)
        result, test_p = train_one(
            Xt, y_train, Xv, y_val, Xs, y_test,
            in_dim=in_dim, batch_size=args.batch_size, epochs=args.epochs,
            lr=args.lr, wd=arch["weight_decay"], device=device, log=log, tag=variant,
            hidden=tuple(arch["hidden"]), dropout=arch["dropout"],
            class_balance=args.class_balance,
            extra_predicts=X_all_by_aug,
            export_model=export_artifacts)

        var_dir = out_dir / variant
        var_dir.mkdir(parents=True, exist_ok=True)
        pred_df = gt.iloc[test_idx].copy()
        pred_df["pred_score"] = test_p
        pred_df.to_csv(var_dir / "predictions.csv", index=False)

        # ---- full-data prediction dump (train+val+test, orig + each aug).
        # Same trained model -> test rows here equal predictions.csv exactly.
        # pred_prob          = predictions on the ORIGINAL (un-augmented) row
        # pred_prob_<aug>    = predictions on the <aug>-augmented version of
        #                      the SAME row, through the same scaler / PCA.
        # Lets the analysis workbook check whether train rows that score 0
        # FPs on orig are still classified correctly under aug perturbation
        # (= generalises within aug neighbourhood) or only on the clean copy
        # (= memorising clean train features).
        if dump_full and result.extra_preds is not None:
            full_df = pd.DataFrame({
                "npy_filename": filenames_cur,
                "group_id": (gt["group_id"].astype(str).to_numpy()
                             if "group_id" in gt.columns else ""),
                "question_id": (gt["question_id"].astype(str).to_numpy()
                                if "question_id" in gt.columns else ""),
                "batch": gt["batch"].astype(str).to_numpy(),
                "region": (gt["region"].astype(str).to_numpy()
                           if "region" in gt.columns else ""),
                "label": y_full,
                "split": split_label,
                "pred_prob": result.extra_preds["orig"],
            })
            for aug_name in use_augs:
                if aug_name in result.extra_preds:
                    full_df[f"pred_prob_{aug_name}"] = result.extra_preds[aug_name]
            full_df.to_csv(var_dir / "full_predictions.csv", index=False)
            n_aug_cols = sum(1 for c in full_df.columns if c.startswith("pred_prob_"))
            log.info("[%s] wrote full_predictions.csv (%d rows: %s; "
                     "pred_prob + %d aug cols)",
                     variant, len(full_df),
                     full_df["split"].value_counts().to_dict(),
                     n_aug_cols)

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

        # ---- export trained artifacts for offline inference / download
        if export_artifacts and result.model_state is not None:
            torch.save(result.model_state, var_dir / "model.pt")
            joblib.dump(lp["scaler"], var_dir / "scaler.joblib")
            if pca_var is not None:
                joblib.dump(pca, var_dir / "pca.joblib")
            n_feat = len(feat_cols) if (feat_cols and use_text_features) else 0
            with (var_dir / "inference_meta.json").open("w") as f:
                json.dump({
                    "variant": variant, "arch": arch_name, "wavlm_layer": layer,
                    "hidden": list(arch["hidden"]), "dropout": arch["dropout"],
                    "weight_decay": arch["weight_decay"], "pca": pca_var,
                    "in_dim": in_dim, "use_augs": use_augs,
                    "use_text_features": use_text_features,
                    "feat_cols": feat_cols if (feat_cols and use_text_features) else [],
                    "class_balance": args.class_balance,
                    "label_smoothing": LABEL_SMOOTHING,
                    "best_epoch": result.best_epoch,
                    "best_val_f1": result.best_val_f1,
                    "best_f1_threshold": m["best_f1"]["threshold"],
                    "thr0.5_f1": m["thr0.5"]["f1"],
                    "wavlm_dim": exp_wavlm_dim, "whisper_dim": exp_whisper_dim,
                    "wavlm_id": exp_wavlm_id, "whisper_id": exp_whisper_id,
                    "target_sr": 16000,
                    "feature_order": (
                        f"concat[ wavlm[{layer}] ({exp_wavlm_dim}) | "
                        f"whisper ({exp_whisper_dim}) | feat_* ({n_feat}) ] "
                        f"-> scaler.transform -> "
                        f"{'pca.transform -> ' if pca_var is not None else ''}"
                        f"MLP -> sigmoid; predict 1 if prob >= best_f1_threshold"),
                }, f, indent=2)
            log.info("[%s] exported model.pt + scaler.joblib%s + inference_meta.json",
                     variant, " + pca.joblib" if pca_var is not None else "")

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

        # ---- ablation branch: extra LOO aug configs for target variants ----
        # Runs only when this variant is a target and the sweep used >=2 augs
        # (i.e. --use_augs all). Each config trains from rng0 (the exact pre-
        # train state of the headline run), so 'all' reproduces the sheet number
        # and drop_<aug>/none differ only by the aug set. rng1 (post-main-train)
        # is restored afterwards so the rest of the sweep is byte-identical.
        if ablate_on and variant in ablate_set and "X_by_aug" in lp and len(use_augs) >= 2:
            rng1 = _snapshot_rng(device)
            Xby = lp["X_by_aug"]
            abl = list(use_augs)
            cfgs = [("all", abl)]
            cfgs += [(f"drop_{a}", [x for x in abl if x != a]) for a in abl]
            cfgs += [("none", [])]
            log.info("[%s] ABLATION: %d aug configs from the SAME pre-train RNG "
                     "(split + train rows frozen; 'all' reproduces the headline)",
                     variant, len(cfgs))
            for cfg_name, cfg_augs in cfgs:
                Xtr, ytr, Xva, Xte = _build_cfg_xy(
                    Xby, train_idx, val_idx, test_idx, y_train_base,
                    cfg_augs, pca_var, args.seed)
                _restore_rng(rng0, device)  # identical init to the headline run
                res_c, _ = train_one(
                    Xtr, ytr, Xva, y_val, Xte, y_test,
                    in_dim=Xtr.shape[1], batch_size=args.batch_size,
                    epochs=args.epochs, lr=args.lr, wd=arch["weight_decay"],
                    device=device, log=log, tag=f"ABL/{variant}/{cfg_name}",
                    hidden=tuple(arch["hidden"]), dropout=arch["dropout"],
                    class_balance=args.class_balance)
                mc = res_c.test_metrics
                rc = mc.get("recall_at_precision", {})
                ablation_rows.append({
                    "variant": variant, "aug_config": cfg_name,
                    "n_augs": len(cfg_augs), "in_dim": int(Xtr.shape[1]),
                    "val_f1": round(res_c.best_val_f1, 4),
                    "test_best_f1": round(mc["best_f1"]["f1"], 4),
                    "test_best_thr": round(mc["best_f1"]["threshold"], 3),
                    "test_f1@0.5": round(mc["thr0.5"]["f1"], 4),
                    "test_auc": round(mc["auc"], 4), "test_ap": round(mc["ap"], 4),
                    "recall@p80": round(rc.get("p80", {}).get("recall", float("nan")), 4),
                    "recall@p85": round(rc.get("p85", {}).get("recall", float("nan")), 4),
                    "recall@p90": round(rc.get("p90", {}).get("recall", float("nan")), 4),
                    "recall@p95": round(rc.get("p95", {}).get("recall", float("nan")), 4),
                    "n_test": mc["n"],
                })
                log.info("[ABL %s/%s] test_best_f1=%.4f auc=%.3f r@p90=%.4f r@p95=%.4f",
                         variant, cfg_name, mc["best_f1"]["f1"], mc["auc"],
                         rc.get("p90", {}).get("recall", float("nan")),
                         rc.get("p95", {}).get("recall", float("nan")))
            _restore_rng(rng1, device)  # continue the sweep unperturbed

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary1.csv"
    summary.to_csv(summary_path, index=False)
    log.info("=" * 60)
    log.info("SUMMARY (NN, summary1.csv):\n%s", summary.to_string(index=False))
    log.info("Wrote %s", summary_path)

    # ---- ablation output: per (variant, aug_config) + delta vs that variant's
    #      'all' row (the headline-reproducing config) ----
    if ablation_rows:
        abl = pd.DataFrame(ablation_rows)
        dcols = ["test_best_f1", "recall@p80", "recall@p85", "recall@p90", "recall@p95"]
        base = (abl[abl["aug_config"] == "all"]
                .set_index("variant")[dcols].rename(columns=lambda c: f"all_{c}"))
        abl = abl.join(base, on="variant")
        for c in dcols:
            abl[f"delta_{c}_vs_all"] = (abl[c] - abl[f"all_{c}"]).round(4)
        abl_path = out_dir / "ablation_sweep.csv"
        abl.to_csv(abl_path, index=False)
        log.info("Wrote %s (%d rows)", abl_path, len(abl))
        try:
            with pd.ExcelWriter(out_dir / "ablation_sweep.xlsx", engine="openpyxl") as xw:
                abl.to_excel(xw, sheet_name="ablation", index=False)
                summary.to_excel(xw, sheet_name="sweep_summary", index=False)
            log.info("Wrote %s", out_dir / "ablation_sweep.xlsx")
        except Exception as e:
            log.warning("could not write ablation_sweep.xlsx (%s); CSV is saved", e)
        for variant, g in abl.groupby("variant", sort=False):
            log.info("=" * 60)
            log.info("ABLATION variant=%s  (delta vs its OWN 'all' = headline repro)",
                     variant)
            cols = ["aug_config", "test_best_f1", "delta_test_best_f1_vs_all",
                    "recall@p90", "delta_recall@p90_vs_all",
                    "recall@p95", "delta_recall@p95_vs_all"]
            log.info("\n%s", g[[c for c in cols if c in g.columns]].to_string(index=False))
        log.info("READING: the 'all' row should match this variant's row in "
                 "summary1.csv (and your original sheet). 'drop_<aug>' below 'all' "
                 "=> that aug HELPS; above => it HURTS. All configs share the "
                 "headline init, so the gap is purely the aug set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
