"""
EC2 (T4) baseline training WITH waveform-level data augmentation.

Same pipeline as neural_baseline_train.py, with one key difference: for every
training audio, multiple augmented copies are produced (Gaussian SNR noise,
pitch shift, time stretch / speed perturb, gain, air absorption, optional MUSAN
background, optional RIR convolution, optional codec round-trip), embeddings
are extracted for each, and the MLP trains on the expanded set. Validation
and test stay strictly on un-augmented (orig) audio.

Reuses helpers from neural_baseline_train.py via sibling import:
    MLP, smoothed_bce_logits, train_one, compute_metrics, per_slice_metrics,
    build_splits, assert_no_group_leak, extract_wavlm_meanpool,
    extract_whisper_meanpool, ARCH_CONFIG, VARIANTS_CONFIG,
    LABEL_SMOOTHING, GRAD_CLIP_NORM.

Run:
    python neural_baseline_train_aug.py --data_dir <upload> --out_dir <results> \
        --train_batches audios2,audios4,audios5 --test_batches audios6 \
        --augs noise,pitch,speed,gain,air,combo

Optional external resources:
    --rir_dir   directory of RIR .wav files (enables 'rir' aug)
    --noise_dir directory of MUSAN-style noise .wavs (enables 'bgnoise' aug)

The aug embeddings cache (embeddings_aug_cache.npz) is keyed on
(npy_filename, aug_name) and is reused across runs unless --force_extract.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Sibling import from the base script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neural_baseline_train import (  # noqa: E402
    ARCH_CONFIG, GRAD_CLIP_NORM, LABEL_SMOOTHING, VARIANTS_CONFIG,
    WAVLM_ID, WHISPER_ID, TARGET_SR,
    assert_no_group_leak, build_splits, compute_metrics,
    extract_wavlm_meanpool, extract_whisper_meanpool, per_slice_metrics,
    train_one, setup_logging,
)
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    WavLMModel, WhisperFeatureExtractor, WhisperModel,
)

# ----------------------------------------------------------------------------
# Augmentation pipelines (audiomentations).
# ----------------------------------------------------------------------------

def build_augmenters(names: list[str], rir_dir: Path | None,
                     noise_dir: Path | None, log: logging.Logger
                     ) -> dict[str, "object"]:
    """Return {aug_name -> callable(samples=wav, sample_rate=sr) -> wav}.

    'orig' is special-cased in the extraction loop and not in this dict."""
    from audiomentations import (
        AddGaussianSNR, AirAbsorption, Compose, Gain, PitchShift, TimeStretch,
    )
    available: dict[str, object] = {
        "noise":  Compose([AddGaussianSNR(min_snr_db=10.0, max_snr_db=25.0, p=1.0)]),
        "pitch":  Compose([PitchShift(min_semitones=-2, max_semitones=2, p=1.0)]),
        "speed":  Compose([TimeStretch(min_rate=0.92, max_rate=1.08, p=1.0,
                                       leave_length_unchanged=False)]),
        "gain":   Compose([Gain(min_gain_db=-9.0, max_gain_db=9.0, p=1.0)]),
        "air":    Compose([AirAbsorption(p=1.0)]),
        "combo":  Compose([
            AddGaussianSNR(min_snr_db=10.0, max_snr_db=25.0, p=0.6),
            PitchShift(min_semitones=-1, max_semitones=1, p=0.4),
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.4, leave_length_unchanged=False),
            Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.4),
        ]),
    }

    if rir_dir is not None and rir_dir.exists():
        from audiomentations import ApplyImpulseResponse
        available["rir"] = Compose([ApplyImpulseResponse(ir_path=str(rir_dir), p=1.0)])
        log.info("RIR augmentation enabled from %s", rir_dir)
    if noise_dir is not None and noise_dir.exists():
        from audiomentations import AddBackgroundNoise
        available["bgnoise"] = Compose([
            AddBackgroundNoise(sounds_path=str(noise_dir),
                               min_snr_db=5.0, max_snr_db=20.0, p=1.0)])
        log.info("Background-noise augmentation enabled from %s", noise_dir)

    try:
        from audiomentations import Mp3Compression
        available["codec"] = Compose([Mp3Compression(min_bitrate=32, max_bitrate=64, p=1.0)])
    except Exception as e:
        log.info("Codec augmentation unavailable (%s); skipping", e)

    chosen = {}
    for n in names:
        if n in available:
            chosen[n] = available[n]
        else:
            log.warning("Requested aug '%s' not available -- ignoring (have: %s)",
                        n, sorted(available.keys()))
    log.info("Active augmenters (%d): %s", len(chosen), list(chosen.keys()))
    return chosen


# ----------------------------------------------------------------------------
# Augmented extraction
# ----------------------------------------------------------------------------

def extract_orig_for_all(gt: pd.DataFrame, npy_dir: Path,
                         wavlm: WavLMModel, whisper: WhisperModel,
                         whisper_feat: WhisperFeatureExtractor,
                         device: torch.device, log: logging.Logger
                         ) -> tuple[np.ndarray, np.ndarray]:
    wavlm_dim = wavlm.config.hidden_size
    whisper_dim = whisper.config.d_model
    wavlm_emb = np.zeros((len(gt), wavlm_dim), dtype=np.float32)
    whisper_emb = np.zeros((len(gt), whisper_dim), dtype=np.float32)
    for i, row in tqdm(list(gt.iterrows()), desc="orig", unit="audio"):
        try:
            wav = np.load(npy_dir / row["npy_filename"]).astype(np.float32, copy=False)
        except Exception as e:
            log.warning("Load failed for %s: %s -- zeros", row["npy_filename"], e)
            continue
        wavlm_emb[i] = extract_wavlm_meanpool(wav, wavlm, device)
        whisper_emb[i] = extract_whisper_meanpool(wav, whisper, whisper_feat, device)
    return wavlm_emb, whisper_emb


def extract_augmented_for_train(gt: pd.DataFrame, train_idx: np.ndarray,
                                npy_dir: Path, augs: dict,
                                wavlm: WavLMModel, whisper: WhisperModel,
                                whisper_feat: WhisperFeatureExtractor,
                                device: torch.device, log: logging.Logger
                                ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each aug, return (wavlm_emb_train, whisper_emb_train) aligned with
    train_idx ordering. Skips 'orig' (caller already has those)."""
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    wavlm_dim = wavlm.config.hidden_size
    whisper_dim = whisper.config.d_model
    for aug_name, aug_fn in augs.items():
        log.info("Augmenting train (%s) -- %d audios", aug_name, len(train_idx))
        wavlm_buf = np.zeros((len(train_idx), wavlm_dim), dtype=np.float32)
        whisper_buf = np.zeros((len(train_idx), whisper_dim), dtype=np.float32)
        for k, i in enumerate(tqdm(train_idx, desc=aug_name, unit="audio")):
            row = gt.iloc[i]
            try:
                wav = np.load(npy_dir / row["npy_filename"]).astype(np.float32, copy=False)
            except Exception as e:
                log.warning("Load failed for %s aug=%s: %s -- zeros",
                            row["npy_filename"], aug_name, e)
                continue
            try:
                wav_aug = aug_fn(samples=wav, sample_rate=TARGET_SR)
            except Exception as e:
                log.warning("Aug '%s' failed on %s: %s -- using orig",
                            aug_name, row["npy_filename"], e)
                wav_aug = wav
            if wav_aug.dtype != np.float32:
                wav_aug = wav_aug.astype(np.float32, copy=False)
            wavlm_buf[k] = extract_wavlm_meanpool(wav_aug, wavlm, device)
            whisper_buf[k] = extract_whisper_meanpool(wav_aug, whisper, whisper_feat, device)
        results[aug_name] = (wavlm_buf, whisper_buf)
    return results


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_batches", default="audios2,audios4,audios5,audios6")
    ap.add_argument("--test_batches", default="")
    ap.add_argument("--test_region_filter", default="")
    ap.add_argument("--augs", default="noise,pitch,speed,gain,air,combo",
                    help="comma-separated names of augmentations to apply")
    ap.add_argument("--rir_dir", default="", help="optional RIR wav dir; enables 'rir'")
    ap.add_argument("--noise_dir", default="", help="optional noise wav dir; enables 'bgnoise'")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_extract", action="store_true")
    args = ap.parse_args()

    train_batches = [b.strip() for b in args.train_batches.split(",") if b.strip()]
    test_batches = [b.strip() for b in args.test_batches.split(",") if b.strip()]
    test_region_filter = args.test_region_filter.strip() or None
    aug_names = [a.strip() for a in args.augs.split(",") if a.strip()]
    rir_dir = Path(args.rir_dir) if args.rir_dir else None
    noise_dir = Path(args.noise_dir) if args.noise_dir else None

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "log.txt")
    log.info("[AUG] data_dir=%s out_dir=%s", data_dir, out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gt = pd.read_csv(data_dir / "gt.csv")
    gt["label"] = pd.to_numeric(gt["label"], errors="coerce")
    n_before = len(gt)
    gt = gt[gt["label"].isin([0, 1])].copy()
    gt["label"] = gt["label"].astype(int)
    gt = gt.reset_index(drop=True)
    log.info("Loaded gt: %d -> %d labeled rows", n_before, len(gt))

    requested = set(train_batches) | set(test_batches)
    n_unused = int((~gt["batch"].isin(requested)).sum())
    if n_unused:
        log.warning("%d gt rows are in batches outside train/test args and will be unused",
                    n_unused)

    # Splits first -- we only augment training rows.
    splits = build_splits(gt, train_batches=train_batches,
                          test_batches=test_batches if test_batches else None,
                          test_region_filter=test_region_filter,
                          seed=args.seed, log=log)
    assert_no_group_leak(gt, splits)
    for k, idx in splits.items():
        labels = gt.iloc[idx]["label"].to_numpy()
        log.info("split %-5s n=%4d  pos=%4d  neg=%4d  groups=%4d",
                 k, len(idx), int((labels == 1).sum()), int((labels == 0).sum()),
                 gt.iloc[idx]["group_id"].nunique())

    # Load encoders.
    log.info("Loading encoders...")
    wavlm = WavLMModel.from_pretrained(WAVLM_ID).to(device).eval()
    whisper = WhisperModel.from_pretrained(WHISPER_ID).to(device).eval()
    whisper_feat = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

    # Aug definitions.
    augs = build_augmenters(aug_names, rir_dir, noise_dir, log)
    if not augs:
        log.warning("No augmentations selected -- this run is equivalent to non-aug baseline")

    # Cache: orig embeddings + per-aug train embeddings, single npz.
    cache_path = out_dir / "embeddings_aug_cache.npz"
    npy_dir = data_dir / "audio_npy"

    train_idx = splits["train"]
    rebuild = args.force_extract or not cache_path.exists()
    if not rebuild:
        try:
            cache = np.load(cache_path, allow_pickle=True)
            cached_files = cache["filenames"].astype(str)
            cur_files = gt["npy_filename"].to_numpy()
            cur_train_files = gt.iloc[train_idx]["npy_filename"].to_numpy()
            cached_train_files = cache["train_filenames"].astype(str)
            cached_aug_names = list(cache["aug_names"].astype(str))
            need_extract = (
                set(cur_files) - set(cached_files)
                or set(cur_train_files) - set(cached_train_files)
                or set(augs.keys()) - set(cached_aug_names)
            )
            if need_extract:
                log.info("Cache exists but doesn't cover current request; rebuilding")
                rebuild = True
        except Exception as e:
            log.warning("Cache load failed (%s); rebuilding", e)
            rebuild = True

    if rebuild:
        t0 = time.time()
        wavlm_orig, whisper_orig = extract_orig_for_all(
            gt, npy_dir, wavlm, whisper, whisper_feat, device, log)
        aug_results = extract_augmented_for_train(
            gt, train_idx, npy_dir, augs, wavlm, whisper, whisper_feat, device, log)
        log.info("Total extraction time %.1f s", time.time() - t0)

        save_kwargs = {
            "filenames": gt["npy_filename"].to_numpy(),
            "train_filenames": gt.iloc[train_idx]["npy_filename"].to_numpy(),
            "wavlm_orig": wavlm_orig,
            "whisper_orig": whisper_orig,
            "aug_names": np.array(list(augs.keys()), dtype=object),
        }
        for n, (w, h) in aug_results.items():
            save_kwargs[f"wavlm_{n}"] = w
            save_kwargs[f"whisper_{n}"] = h
        np.savez(cache_path, **save_kwargs)
        log.info("Saved aug cache to %s", cache_path)
    else:
        cache = np.load(cache_path, allow_pickle=True)
        cached_files = cache["filenames"].astype(str)
        f2idx = {f: i for i, f in enumerate(cached_files)}
        order_all = np.array([f2idx[f] for f in gt["npy_filename"].to_numpy()])
        wavlm_orig = cache["wavlm_orig"][order_all]
        whisper_orig = cache["whisper_orig"][order_all]
        cached_train_files = cache["train_filenames"].astype(str)
        f2idx_t = {f: i for i, f in enumerate(cached_train_files)}
        order_tr = np.array([f2idx_t[f] for f in gt.iloc[train_idx]["npy_filename"].to_numpy()])
        aug_results = {}
        for n in augs.keys():
            aug_results[n] = (cache[f"wavlm_{n}"][order_tr],
                              cache[f"whisper_{n}"][order_tr])
        log.info("Loaded aug cache from %s", cache_path)

    # Free encoders (no longer needed).
    del wavlm, whisper, whisper_feat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build handcrafted-feature block from gt (constant per source audio across augs).
    feat_cols = [c for c in gt.columns if c.startswith("feat_")]
    if feat_cols:
        feat_block = gt[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feat_arr = feat_block.to_numpy().astype(np.float32)
        log.info("Handcrafted features detected (%d). NOTE: feat values are constant "
                 "per source audio across all augs (no re-derivation).", len(feat_cols))
    else:
        feat_arr = None
        log.info("No feat_* columns; running on audio embeddings only")

    def concat_feats(wavlm_e: np.ndarray, whisper_e: np.ndarray,
                     hand_e: np.ndarray | None) -> np.ndarray:
        parts = [wavlm_e, whisper_e]
        if hand_e is not None:
            parts.append(hand_e)
        return np.concatenate(parts, axis=1).astype(np.float32)

    # Orig features for ALL rows (used for val + test, and as the "orig" copy of train).
    X_orig = concat_feats(wavlm_orig, whisper_orig, feat_arr)
    y_full = gt["label"].to_numpy().astype(np.int64)

    # Build expanded training matrix: one orig copy + one copy per aug.
    train_feat = feat_arr[train_idx] if feat_arr is not None else None
    X_train_blocks = [concat_feats(wavlm_orig[train_idx], whisper_orig[train_idx], train_feat)]
    y_train_blocks = [y_full[train_idx]]
    aug_tags = ["orig"] * len(train_idx)
    for n, (w, h) in aug_results.items():
        X_train_blocks.append(concat_feats(w, h, train_feat))
        y_train_blocks.append(y_full[train_idx])
        aug_tags.extend([n] * len(train_idx))
    X_train_aug = np.concatenate(X_train_blocks, axis=0).astype(np.float32)
    y_train_aug = np.concatenate(y_train_blocks, axis=0).astype(np.int64)
    log.info("Augmented train: %d rows (= %d orig + %d augs x %d). aug counts: %s",
             len(X_train_aug), len(train_idx), len(augs), len(train_idx),
             pd.Series(aug_tags).value_counts().to_dict())

    X_val = X_orig[splits["val"]]
    y_val = y_full[splits["val"]]
    X_test = X_orig[splits["test"]]
    y_test = y_full[splits["test"]]

    # Standardizer fit on (augmented) train only -- captures aug-induced variance.
    scaler = StandardScaler().fit(X_train_aug)
    X_train_aug_std = scaler.transform(X_train_aug).astype(np.float32)
    X_val_std = scaler.transform(X_val).astype(np.float32)
    X_test_std = scaler.transform(X_test).astype(np.float32)

    summary_rows = []

    for variant, pca_var, arch_name in VARIANTS_CONFIG:
        log.info("=" * 60)
        arch = ARCH_CONFIG[arch_name]
        log.info("VARIANT: %s  arch=%s  pca=%s  (with augs=%s)", variant, arch_name,
                 pca_var, list(augs.keys()))

        if pca_var is None:
            Xt, Xv, Xs = X_train_aug_std, X_val_std, X_test_std
            in_dim = Xt.shape[1]
        else:
            pca = PCA(n_components=pca_var, svd_solver="full", random_state=args.seed)
            pca.fit(X_train_aug_std)
            Xt = pca.transform(X_train_aug_std).astype(np.float32)
            Xv = pca.transform(X_val_std).astype(np.float32)
            Xs = pca.transform(X_test_std).astype(np.float32)
            in_dim = Xt.shape[1]
            log.info("PCA(%.2f) -> %d components", pca_var, in_dim)

        result, test_p = train_one(Xt, y_train_aug, Xv, y_val, Xs, y_test,
                                   in_dim=in_dim, batch_size=args.batch_size,
                                   epochs=args.epochs, lr=args.lr,
                                   wd=arch["weight_decay"],
                                   hidden=tuple(arch["hidden"]),
                                   dropout=arch["dropout"],
                                   device=device, log=log, tag=f"{variant}_aug")

        var_dir = out_dir / variant
        var_dir.mkdir(parents=True, exist_ok=True)
        pred_df = gt.iloc[splits["test"]].copy()
        pred_df["pred_score"] = test_p
        pred_df.to_csv(var_dir / "predictions.csv", index=False)

        m = result.test_metrics
        log.info("[%s] TEST  n=%d  auc=%.3f  ap=%.3f  thr0.5 f1=%.3f  best_f1=%.3f@%.2f",
                 variant, m["n"], m["auc"], m["ap"], m["thr0.5"]["f1"],
                 m["best_f1"]["f1"], m["best_f1"]["threshold"])
        rap = m.get("recall_at_precision", {})
        log.info("[%s] recall@p  p50=%.3f  p80=%.3f  p90=%.3f  p95=%.3f", variant,
                 rap.get("p50", {}).get("recall", float("nan")),
                 rap.get("p80", {}).get("recall", float("nan")),
                 rap.get("p90", {}).get("recall", float("nan")),
                 rap.get("p95", {}).get("recall", float("nan")))

        if "batch" in pred_df.columns:
            log.info("[%s] per-batch on test:", variant)
            m["per_batch"] = per_slice_metrics(y_test, test_p, pred_df["batch"].to_numpy(),
                                               "batch", log)
        if "region" in pred_df.columns and pred_df["region"].notna().any():
            log.info("[%s] per-region on test:", variant)
            m["per_region"] = per_slice_metrics(y_test, test_p,
                                                pred_df["region"].fillna("UNK").to_numpy(),
                                                "region", log)

        with (var_dir / "metrics.json").open("w") as f:
            json.dump({"variant": variant, "arch": arch_name,
                       "hidden": list(arch["hidden"]),
                       "dropout": arch["dropout"],
                       "weight_decay": arch["weight_decay"],
                       "label_smoothing": LABEL_SMOOTHING,
                       "grad_clip": GRAD_CLIP_NORM,
                       "pca": pca_var, "augs": list(augs.keys()),
                       "best_val_f1": result.best_val_f1,
                       "best_epoch": result.best_epoch,
                       "in_dim": in_dim, "test": m}, f, indent=2)

        summary_rows.append({
            "variant": variant, "arch": arch_name,
            "hidden": "x".join(str(h) for h in arch["hidden"]),
            "dropout": arch["dropout"], "wd": arch["weight_decay"],
            "n_augs": len(augs), "in_dim": in_dim,
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
    log.info("SUMMARY (with augs):\n%s", summary.to_string(index=False))
    log.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
