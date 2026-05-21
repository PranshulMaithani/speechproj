"""Shared data-pipeline helpers used by both neural_baseline_train.py and
xgboost_train.py.

The point of this module is that both training scripts MUST see the same
filtered gt.csv, the same train/val/test row indices, and the same
WavLM/Whisper/feat_* arrays for a given run. Otherwise summary1.csv (NN)
and summary2.csv (XGB) are not comparable on the same data.

What lives here:
  - WAVLM_LAYERS              : layers required from the cache (last, 9)
  - setup_logging             : tee log to file + stdout
  - _kfold_indices            : StratifiedGroupKFold helper
  - build_splits              : Mode A / Mode B split builder
  - assert_no_group_leak      : assert train/val/test groups disjoint
  - load_cache_reindexed      : align cache rows to gt order
  - resolve_use_augs          : parse --use_augs against cache
  - compute_metrics           : F1/AP/AUC + recall@precision
  - per_slice_metrics         : same metrics over a slice column
  - load_gt_and_filter        : read gt.csv, drop bad labels, apply
                                --min_duration, return reset DataFrame
  - assemble_features         : per-WavLM-layer (X_train_aug, X_val, X_test,
                                aug_tags) builder, used by both heads
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold


# WavLM layers expected in the cache and used by every variant
# (extract_embeddings.py must have been run with --wavlm_layers last,9).
WAVLM_LAYERS: list[str] = ["last", "9"]


# ----------------------------------------------------------------------------
# Logging.
# ----------------------------------------------------------------------------

def setup_logging(log_path: Path, name: str = "train") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
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
                 seed: int, log: logging.Logger,
                 fewshot_frac: float = 0.0,
                 fewshot_batches: list[str] | None = None
                 ) -> dict[str, np.ndarray]:
    """train_only_batches: batches whose rows are forced into the train split
    only -- never appear in val/test. Their group_ids are also stripped from
    the candidate-leak check (they're already disjoint by construction).

    fewshot_frac > 0 enables few-shot client adaptation: that fraction of
    candidates from fewshot_batches (default: the test_batches) is moved out
    of test and into train, candidate-disjoint with the remaining test set.
    This is the realistic deployment scenario -- you label a small batch
    from a new client before going live. Only applies in Mode B.
    """
    train_only_set = set(train_only_batches or [])
    train_only_mask = gt["batch"].isin(train_only_set).to_numpy() if train_only_set else np.zeros(len(gt), dtype=bool)
    train_only_idx = np.where(train_only_mask)[0]
    if len(train_only_idx):
        log.info("train_only rows (auxiliary, forced to train): %d (batches=%s)",
                 len(train_only_idx), sorted(train_only_set))

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

    # Few-shot client adaptation: carve a fraction of the test-batch candidates
    # back into train (candidate-disjoint). Their rows are removed from test.
    fewshot_idx = np.array([], dtype=np.int64)
    if fewshot_frac > 0.0:
        fb_set = set(fewshot_batches) if fewshot_batches else set(test_batches)
        fewshot_mask = gt["batch"].isin(fb_set).to_numpy() & test_mask
        if fewshot_mask.any():
            cands = gt.loc[fewshot_mask, "group_id"].unique().tolist()
            rng = np.random.default_rng(seed)
            rng.shuffle(cands)
            n_pick = max(1, int(round(len(cands) * fewshot_frac)))
            adapt_cands = set(cands[:n_pick])
            adapt_mask = fewshot_mask & gt["group_id"].isin(adapt_cands).to_numpy()
            test_mask = test_mask & ~adapt_mask
            fewshot_idx = np.where(adapt_mask)[0]
            log.info("fewshot_frac=%.2f  -> %d adapt-train rows from %d/%d "
                     "candidates of %s (removed from test)",
                     fewshot_frac, len(fewshot_idx), n_pick, len(cands),
                     sorted(fb_set))
        else:
            log.warning("fewshot_frac=%.2f but fewshot_batches=%s has no rows "
                        "intersecting test_mask; not adapting", fewshot_frac, fb_set)

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
    train_idx = np.concatenate([to_orig[train_local], train_only_idx, fewshot_idx])
    log.info("Split mode B: train=%s test=%s region_filter=%s  + %d train_only "
             "rows + %d fewshot-adapt rows",
             train_pool_batches, test_batches, test_region_filter,
             len(train_only_idx), len(fewshot_idx))
    return {"train": train_idx,
            "val":   to_orig[val_local],
            "test":  test_idx}


def assert_no_group_leak(gt: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    sets = {k: set(gt.iloc[idx]["group_id"]) for k, idx in splits.items()}
    assert sets["train"].isdisjoint(sets["test"]), "train/test group leak"
    assert sets["train"].isdisjoint(sets["val"]),  "train/val group leak"
    assert sets["val"].isdisjoint(sets["test"]),   "val/test group leak"


# ----------------------------------------------------------------------------
# gt loading + min_duration filter.
# ----------------------------------------------------------------------------

# Recognized region values. Anything else collapses to OTHER for bucket logs.
_REGION_VALUES = {"IND", "PHP"}
# Batches whose rows count as "ALLSTAR" in the candidate-bucket breakdown.
ALLSTAR_BATCH_NAMES = {"2676", "2677"}

# Which batches belong to which client. The audios2/4/5 set is one production
# client; audios6 is a different one (different mic chain, candidate pool,
# question pool, and is the only batch with PHP region). Per-client
# standardization centers each client's features on its own mean/std so the
# downstream model sees client-invariant inputs.
CLIENT_MAP = {
    "audios2": "A",
    "audios4": "A",
    "audios5": "A",
    "audios6": "B",
    "2676": "ALLSTAR",
    "2677": "ALLSTAR",
}


def load_gt_and_filter(data_dir: Path, min_duration: float, log: logging.Logger
                       ) -> pd.DataFrame:
    gt = pd.read_csv(data_dir / "gt.csv")
    log.info("Loaded gt.csv: %d rows", len(gt))

    # CRITICAL: force batch to a string column. pandas auto-infers numeric
    # values like "2676" as int64 (and partially-numeric concats become
    # object with mixed int/str entries). Either case makes
    # gt["batch"].isin({"2676","2677"}) silently return all-False, which
    # makes --train_only_batches and --train_batches stop working for
    # ALLSTAR. Symptom: train numbers identical with/without ALLSTAR.
    if "batch" in gt.columns:
        gt["batch"] = gt["batch"].astype(str)

    gt["label"] = pd.to_numeric(gt["label"], errors="coerce")
    n_before = len(gt)
    gt = gt[gt["label"].isin([0, 1])].copy()
    gt["label"] = gt["label"].astype(int)
    gt = gt.reset_index(drop=True)
    log.info("Filtered to {0,1} labels: %d -> %d", n_before, len(gt))

    if min_duration > 0.0:
        if "duration_sec" not in gt.columns:
            log.warning("--min_duration set but gt has no duration_sec column; skipping filter")
        else:
            n_before = len(gt)
            gt = gt[pd.to_numeric(gt["duration_sec"], errors="coerce")
                      .fillna(0.0) >= min_duration].reset_index(drop=True)
            log.info("min_duration=%.2fs filter: %d -> %d rows",
                     min_duration, n_before, len(gt))

    # Quick presence diagnostic so the user can see ALLSTAR was actually
    # loaded before training starts.
    if "batch" in gt.columns:
        as_n = int(gt["batch"].isin(ALLSTAR_BATCH_NAMES).sum())
        as_groups = (gt[gt["batch"].isin(ALLSTAR_BATCH_NAMES)]["group_id"].nunique()
                     if "group_id" in gt.columns else -1)
        log.info("ALLSTAR rows present in gt: n=%d  unique speakers=%d "
                 "(batch in %s)", as_n, as_groups, sorted(ALLSTAR_BATCH_NAMES))
    return gt


def _bucket_for_row(batch: str, region: str | float) -> str:
    """One of: ALLSTAR, IND, PHP, OTHER. Used for candidate-count breakdowns
    so the user can see at a glance how many candidates each split has from
    each source."""
    if batch in ALLSTAR_BATCH_NAMES:
        return "ALLSTAR"
    r = str(region) if region is not None and region == region else ""  # NaN-safe
    if r.upper() in _REGION_VALUES:
        return r.upper()
    return "OTHER"


def bucket_breakdown(gt: pd.DataFrame, row_idx: np.ndarray) -> dict[str, dict[str, int]]:
    """Returns {bucket: {audios, candidates, pos, neg}} for the given rows."""
    if len(row_idx) == 0:
        return {}
    sub = gt.iloc[row_idx].reset_index(drop=True)
    batches = sub["batch"].astype(str).to_numpy()
    regions = (sub["region"].to_numpy() if "region" in sub.columns
               else np.array([""] * len(sub), dtype=object))
    buckets = np.array([_bucket_for_row(batches[i], regions[i])
                        for i in range(len(sub))])
    has_gid = "group_id" in sub.columns
    out: dict[str, dict[str, int]] = {}
    for b in ("IND", "PHP", "ALLSTAR", "OTHER"):
        mask = buckets == b
        if not mask.any():
            continue
        bucket_rows = sub[mask]
        out[b] = {
            "audios": int(mask.sum()),
            "candidates": int(bucket_rows["group_id"].nunique()) if has_gid else -1,
            "pos": int((bucket_rows["label"] == 1).sum()),
            "neg": int((bucket_rows["label"] == 0).sum()),
        }
    return out


def per_client_standardize(wavlm_cache: dict, whisper_cache: dict,
                           gt: pd.DataFrame, log: logging.Logger,
                           feat_arr: np.ndarray | None = None) -> np.ndarray | None:
    """Center each client's features (WavLM + Whisper + optionally feat_*) on
    its own mean / std, computed from ALL of that client's rows. Uses
    FEATURES only -- no labels touched -- so this is a legitimate
    unsupervised domain-adaptation step.

    Rationale: audios2/4/5 are one client; audios6 is a different one
    (different mic, codec, candidate pool, region). WavLM/Whisper mean-pools
    sit at different absolute locations in feature space per client. Without
    standardization, the model learns "which client is this" as a confound.
    With it, the model sees zero-mean unit-variance features in every
    client's coordinate system -- the threshold-tuning bottleneck observed
    on a6 (AUC 0.92, F1 0.70) shrinks because the score scale becomes
    comparable across clients.

    Mutates the cache dicts and feat_arr in place; returns feat_arr.
    """
    client = gt["batch"].astype(str).map(CLIENT_MAP).fillna("UNK")
    log.info("per-client standardize: %s",
             {k: int(v) for k, v in client.value_counts().items()})
    for c in sorted(client.unique()):
        rows = np.where((client == c).to_numpy())[0]
        if len(rows) < 5:
            log.warning("  skipping client=%s -- only %d rows", c, len(rows))
            continue
        for key in list(wavlm_cache.keys()):
            arr = wavlm_cache[key]
            m = arr[rows].mean(axis=0)
            s = arr[rows].std(axis=0) + 1e-6
            arr[rows] = (arr[rows] - m) / s
            wavlm_cache[key] = arr
        for key in list(whisper_cache.keys()):
            arr = whisper_cache[key]
            m = arr[rows].mean(axis=0)
            s = arr[rows].std(axis=0) + 1e-6
            arr[rows] = (arr[rows] - m) / s
            whisper_cache[key] = arr
        if feat_arr is not None:
            m = feat_arr[rows].mean(axis=0)
            s = feat_arr[rows].std(axis=0) + 1e-6
            feat_arr[rows] = (feat_arr[rows] - m) / s
        log.info("  client=%-8s rows=%5d  centered (wavlm + whisper%s)",
                 c, len(rows), " + feat" if feat_arr is not None else "")
    return feat_arr


def _format_bucket_breakdown(b: dict[str, dict[str, int]]) -> str:
    if not b:
        return "(empty)"
    parts = []
    for name in ("IND", "PHP", "ALLSTAR", "OTHER"):
        if name in b:
            d = b[name]
            parts.append(f"{name}: audios={d['audios']} cands={d['candidates']} "
                         f"(pos={d['pos']}/neg={d['neg']})")
    return "  |  ".join(parts)


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
        for target in (0.50, 0.80, 0.85, 0.90, 0.95):
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
        for target in (0.50, 0.80, 0.85, 0.90, 0.95):
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


def log_split_breakdown(gt: pd.DataFrame, splits: dict[str, np.ndarray],
                        log: logging.Logger) -> None:
    """Detailed split printout. For each of train/val/test prints:
        - total n, pos, neg, unique candidates
        - audios per batch
        - audios per region
        - audios + candidates per bucket (IND / PHP / ALLSTAR / OTHER)
    Use this right after assert_no_group_leak so 'only N PHP audios in val'
    surprises and 'ALLSTAR silently dropped' bugs are immediately visible.
    """
    y_full = gt["label"].to_numpy().astype(np.int64)
    has_region = "region" in gt.columns
    for k, idx in splits.items():
        labels = y_full[idx]
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        n_tot = max(len(idx), 1)
        log.info("split %-5s n=%5d  pos=%5d (%.1f%%)  neg=%5d (%.1f%%)  candidates=%5d",
                 k, len(idx), n_pos, 100 * n_pos / n_tot,
                 n_neg, 100 * n_neg / n_tot,
                 gt.iloc[idx]["group_id"].nunique())
        if "batch" in gt.columns:
            bcounts = gt.iloc[idx]["batch"].value_counts().to_dict()
            log.info("  batches: %s", bcounts)
        if has_region:
            sub = gt.iloc[idx]
            region_series = sub["region"].astype("object")
            seen = sorted([str(r) for r in region_series.dropna().unique()])
            for region in seen:
                rmask = (region_series == region).to_numpy()
                ry = labels[rmask]
                log.info("  region=%-7s n=%5d  pos=%5d  neg=%5d",
                         region, int(rmask.sum()),
                         int((ry == 1).sum()), int((ry == 0).sum()))
            n_nan = int(region_series.isna().sum())
            if n_nan:
                log.info("  region=NaN     n=%5d  (typically train_only "
                         "auxiliary rows -- e.g. ALLSTAR)", n_nan)
        # candidate-bucket breakdown (the IND/PHP/ALLSTAR view the user
        # asked for, expressed in candidates rather than audios so it
        # matches how the speakers are counted)
        b = bucket_breakdown(gt, idx)
        log.info("  buckets: %s", _format_bucket_breakdown(b))


def log_variant_prelude(tag: str, gt: pd.DataFrame, splits: dict[str, np.ndarray],
                        log: logging.Logger) -> None:
    """One-line bucket summary printed right before each model fits. Cheap;
    every model gets it so a run's log is self-explanatory."""
    parts = []
    for k in ("train", "val", "test"):
        if k not in splits:
            continue
        b = bucket_breakdown(gt, splits[k])
        cells = []
        for name in ("IND", "PHP", "ALLSTAR", "OTHER"):
            if name in b:
                d = b[name]
                cells.append(f"{name}={d['candidates']}c/{d['audios']}a")
        parts.append(f"{k}[n={len(splits[k])}] " + " ".join(cells))
    log.info("[%s] data:  %s", tag, "    ".join(parts))


def extract_region_metrics(m: dict, region: str) -> dict[str, float]:
    """Pulls best_f1.f1 and recall@p{80,85,90,95} for a single region out of
    the per_region block built by per_slice_metrics. Returns NaN-filled keys
    if the region is absent."""
    pr = (m.get("per_region") or {}).get(region) or {}
    rap = pr.get("recall_at_precision") or {}
    return {
        "f1":  float(pr.get("best_f1", {}).get("f1", float("nan"))),
        "p80": float(rap.get("p80", {}).get("recall", float("nan"))),
        "p85": float(rap.get("p85", {}).get("recall", float("nan"))),
        "p90": float(rap.get("p90", {}).get("recall", float("nan"))),
        "p95": float(rap.get("p95", {}).get("recall", float("nan"))),
    }


def sweep_threshold(y_val: np.ndarray, p_val: np.ndarray,
                    lo: float = 0.05, hi: float = 0.95, n: int = 91) -> tuple[float, float]:
    """Pick the threshold that maximizes F1 on (y_val, p_val)."""
    thrs = np.linspace(lo, hi, n)
    f1s = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in thrs]
    i = int(np.argmax(f1s))
    return float(thrs[i]), float(f1s[i])
