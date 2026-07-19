#!/usr/bin/env python3
"""Self-contained INFERENCE: raw audios -> per-model cheat probability + decision.

Runs the whole chain from scratch on arbitrary audio folders using already-trained
model bundles, and writes one Excel workbook (a sheet per model) into results/.

    wav -> npy (16 kHz) -> transcript -> 55 features -> WavLM+Whisper embeddings
        -> per model: build the 1847-d vector -> scaler -> pca -> MLP -> sigmoid
        -> apply that model's threshold.txt -> CHEAT (1) / GENUINE (0)

FOLDER LAYOUT (create these under inference/)
--------------------------------------------
    inference/audios/    your audio, ANY names. Each immediate child dir = one
                         "audio group" (e.g. audios1/, teamA/, ...). Inside a group
                         there may be candidate subfolders OR direct wavs -- we
                         RECURSE and process only the *_25 / *_26 / *_27 wavs.
                         Loose wavs directly under audios/ form the group "(root)".
    inference/models/    one subfolder per model, ANY name (model1/, best_a6/, ...).
                         Each must contain the exported bundle + a threshold file:
                             model.pt  scaler.joblib  inference_meta.json
                             pca.joblib   (only if the model used PCA)
                             threshold.txt   (a single float, e.g. 0.74)
    inference/data/      intermediate cache (npy, transcript, features, embeddings)
                         so multiple models reuse the same features/embeddings and
                         re-runs are fast.
    inference/results/   the output Excel workbook.

DECISION: probability >= threshold  ->  result = 1 (CHEAT), else 0 (GENUINE).
The probability is the model's cheat probability (sigmoid output).

Run:
    python inference/run_inference.py
    python inference/run_inference.py --transcribe_device cuda --transcribe_compute_type float16
    python inference/run_inference.py --force        # recompute features + embeddings

Notes:
  * Heavy deps (torch, transformers, faster-whisper, spaCy, parselmouth) are imported
    lazily inside the compute stages, so folder/model discovery works without them.
  * No labels are needed and nothing is uploaded -- this is local inference. Real
    audio filenames are used as-is (no anonymisation).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPANYLAPTOP = REPO_ROOT / "companylaptop"
EC2 = REPO_ROOT / "ec2"
INFERENCE_DIR = Path(__file__).resolve().parent

ACCEPT_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
_QID_TAIL_RE = re.compile(r"_(\d{1,3})$")


# ----------------------------------------------------------------------------
# Discovery (pure; no heavy deps).
# ----------------------------------------------------------------------------

def qid_of(stem: str) -> int | None:
    m = _QID_TAIL_RE.search(stem)
    return int(m.group(1)) if m else None


def discover_audio(audios_root: Path, keep: set[int]) -> list[dict]:
    """Every wav under audios_root whose question id is in `keep`, tagged with its
    audio group (the first path component under audios_root; '(root)' for loose
    wavs). Recurses into candidate subfolders."""
    rows = []
    for f in sorted(audios_root.rglob("*")):
        if not (f.is_file() and f.suffix.lower() in ACCEPT_EXT):
            continue
        q = qid_of(f.stem)
        if q is None or q not in keep:
            continue
        rel = f.relative_to(audios_root)
        group = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        rows.append({"group": group, "name": f.name, "stem": f.stem, "qid": q,
                     "path": f, "key": str(rel.with_suffix("")).replace("\\", "__").replace("/", "__")})
    return rows


def discover_models(models_root: Path, log) -> list[dict]:
    """Each immediate subfolder of models_root that has model.pt + inference_meta.json
    + threshold.txt is a model. Reads the meta and the threshold float."""
    out = []
    for d in sorted(p for p in models_root.iterdir() if p.is_dir()):
        meta_p, model_p, thr_p = d / "inference_meta.json", d / "model.pt", d / "threshold.txt"
        missing = [n for n, p in (("model.pt", model_p), ("inference_meta.json", meta_p),
                                  ("threshold.txt", thr_p)) if not p.exists()]
        if missing:
            log(f"  SKIP model '{d.name}' -- missing {missing}")
            continue
        try:
            meta = json.loads(meta_p.read_text())
            thr = float(thr_p.read_text().strip().split()[0])
        except Exception as e:
            log(f"  SKIP model '{d.name}' -- bad meta/threshold ({e})")
            continue
        out.append({"id": d.name, "dir": d, "meta": meta, "threshold": thr})
        log(f"  model '{d.name}': layer={meta.get('wavlm_layer')} in_dim={meta.get('in_dim')} "
            f"pca={'yes' if (d / 'pca.joblib').exists() else 'no'} threshold={thr}")
    return out


# ----------------------------------------------------------------------------
# Stage A: transcripts + 55 handcrafted features (reuses extract_features_batch).
# ----------------------------------------------------------------------------

def compute_features(wavs: list[Path], data_dir: Path, device: str, compute_type: str,
                     model_name: str, force: bool, log) -> dict[str, dict]:
    """Transcribe + compute the 55 feat_* for the wav list (via the training
    scripts' own functions), returning {filename: {feature: value}}."""
    sys.path.insert(0, str(COMPANYLAPTOP))
    from extract_features_batch import (transcribe_batch, extract_features,  # type: ignore
                                        WHISPER_TRANSCRIBE_MODEL)
    trans_path = data_dir / "transcripts.json"
    feat_path = data_dir / "features.csv"
    mdl = model_name or WHISPER_TRANSCRIBE_MODEL
    log(f"[features] transcribe ({device}/{compute_type}, model={mdl}) + 55 feat_* "
        f"for {len(wavs)} wavs")
    transcribe_batch(wavs, trans_path, mdl, device, compute_type, force)
    extract_features(wavs, trans_path, feat_path, force)
    df = pd.read_csv(feat_path)
    key = "filename" if "filename" in df.columns else df.columns[0]
    return {str(r[key]): {c: r[c] for c in df.columns if c != key}
            for _, r in df.iterrows()}


# ----------------------------------------------------------------------------
# Stage B: WavLM (last + 9) + Whisper embeddings, cached per wav.
# ----------------------------------------------------------------------------

def compute_embeddings(audio: list[dict], data_dir: Path, force: bool, log) -> dict[str, dict]:
    """{key: {'wavlm': {'last':768,'9':768}, 'whisper':1024}} for each wav, cached
    to data/emb/<key>.npz so re-runs and extra models are free."""
    import librosa
    import torch
    sys.path.insert(0, str(EC2))
    from extract_embeddings import (extract_wavlm_meanpool, extract_whisper_meanpool,  # type: ignore
                                    WAVLM_ID_DEFAULT, WHISPER_ID_DEFAULT, TARGET_SR)
    from transformers import WavLMModel, WhisperModel, WhisperFeatureExtractor
    emb_dir = data_dir / "emb"
    npy_dir = data_dir / "npy"
    emb_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    todo = [a for a in audio if force or not (emb_dir / f"{a['key']}.npz").exists()]
    out: dict[str, dict] = {}
    if todo:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"[embed] loading WavLM({WAVLM_ID_DEFAULT}) + Whisper({WHISPER_ID_DEFAULT}) "
            f"on {device}; {len(todo)} wavs to encode")
        wavlm = WavLMModel.from_pretrained(WAVLM_ID_DEFAULT).to(device).eval()
        whisper = WhisperModel.from_pretrained(WHISPER_ID_DEFAULT).to(device).eval()
        wfeat = WhisperFeatureExtractor.from_pretrained(WHISPER_ID_DEFAULT)
        for i, a in enumerate(todo, 1):
            y, _ = librosa.load(str(a["path"]), sr=TARGET_SR, mono=True)
            y = y.astype(np.float32)
            np.save(npy_dir / f"{a['key']}.npy", y)                 # the 'npy' step
            wl = extract_wavlm_meanpool(y, wavlm, device, ["last", "9"])
            wh = extract_whisper_meanpool(y, whisper, wfeat, device)
            np.savez(emb_dir / f"{a['key']}.npz",
                     wavlm_last=wl["last"], wavlm_9=wl["9"], whisper=wh)
            if i % 20 == 0 or i == len(todo):
                log(f"[embed]   {i}/{len(todo)} encoded")
    for a in audio:
        z = np.load(emb_dir / f"{a['key']}.npz")
        out[a["key"]] = {"wavlm": {"last": z["wavlm_last"], "9": z["wavlm_9"]},
                         "whisper": z["whisper"]}
    return out


# ----------------------------------------------------------------------------
# Stage C: build the model's feature vector + predict.
# ----------------------------------------------------------------------------

def _feat_value(feats_row: dict, col: str) -> float:
    """gt.csv columns are 'feat_<name>'; features.csv has bare '<name>'. Look up
    prefix-robustly; missing -> 0.0 (matches training's zero-fill)."""
    for k in (col, col[5:] if col.startswith("feat_") else "feat_" + col):
        if k in feats_row and feats_row[k] == feats_row[k]:   # not NaN
            try:
                return float(feats_row[k])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def build_X(audio: list[dict], emb: dict, feats: dict, meta: dict) -> np.ndarray:
    """(n_wavs, in_dim) ORIG feature matrix in the model's training order:
    concat[ wavlm[layer] | whisper | feat_* ]."""
    layer = str(meta.get("wavlm_layer", "last"))
    feat_cols = meta.get("feat_cols") or []
    rows = []
    for a in audio:
        e = emb[a["key"]]
        parts = [e["wavlm"][layer], e["whisper"]]
        if feat_cols:
            frow = feats.get(a["name"], {})
            parts.append(np.array([_feat_value(frow, c) for c in feat_cols], dtype=np.float32))
        rows.append(np.concatenate(parts).astype(np.float32))
    return np.vstack(rows)


def predict(model_dir: Path, X: np.ndarray, meta: dict) -> np.ndarray:
    """Load scaler (+ pca) + MLP weights and return the cheat probability per row."""
    import joblib
    import torch
    sys.path.insert(0, str(EC2))
    from neural_baseline_train import MLP  # type: ignore
    scaler = joblib.load(model_dir / "scaler.joblib")
    Xs = scaler.transform(X).astype(np.float32)
    if (model_dir / "pca.joblib").exists():
        Xs = joblib.load(model_dir / "pca.joblib").transform(Xs).astype(np.float32)
    in_dim = Xs.shape[1]
    meta_in = meta.get("in_dim")
    if meta_in is not None and int(meta_in) != in_dim:
        raise RuntimeError(f"in_dim mismatch for {model_dir.name}: features give {in_dim}, "
                           f"meta says {meta_in} (feature order / cache mismatch)")
    hidden = tuple(meta.get("hidden", [512, 256, 128]))
    dropout = float(meta.get("dropout", 0.4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim, hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(Xs), 512):
            xb = torch.from_numpy(Xs[i:i + 512]).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Self-contained audio -> model inference")
    ap.add_argument("--inference_dir", default=str(INFERENCE_DIR),
                    help="folder holding audios/ data/ models/ results/ (default: this dir)")
    ap.add_argument("--keep_questions", default="25,26,27",
                    help="only these question ids are scored ('all' = every question)")
    ap.add_argument("--transcribe_device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--transcribe_compute_type", default="int8",
                    help="faster-whisper compute type (cuda: float16)")
    ap.add_argument("--transcribe_model", default="",
                    help="override transcription model ('' keeps the training default)")
    ap.add_argument("--force", action="store_true",
                    help="recompute features + embeddings even if cached")
    args = ap.parse_args()

    root = Path(args.inference_dir)
    audios_root = root / "audios"
    data_dir = root / "data"
    models_root = root / "models"
    results_dir = root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    def log(m): print(m, flush=True)

    kq = args.keep_questions.strip().lower()
    keep = set(range(1, 1000)) if kq in ("all", "") else {int(x) for x in kq.split(",") if x.strip().isdigit()}
    log("=" * 74)
    log(f"INFERENCE  root={root}")
    log(f"  audios={audios_root}  models={models_root}  results={results_dir}")

    if not audios_root.is_dir():
        log(f"ERROR: no audios/ dir at {audios_root}"); return 1
    if not models_root.is_dir():
        log(f"ERROR: no models/ dir at {models_root}"); return 1

    audio = discover_audio(audios_root, keep)
    log(f"audio: {len(audio)} wavs (qids {sorted(keep)[:5]}...) across groups "
        f"{sorted(set(a['group'] for a in audio))}")
    log("models:")
    models = discover_models(models_root, log)
    if not audio:
        log("ERROR: no matching wavs (need *_25/_26/_27 under audios/)"); return 1
    if not models:
        log("ERROR: no valid models (need model.pt + inference_meta.json + threshold.txt)"); return 1

    # Stage A + B (shared across all models)
    feats = compute_features([a["path"] for a in audio], data_dir, args.transcribe_device,
                             args.transcribe_compute_type, args.transcribe_model, args.force, log)
    emb = compute_embeddings(audio, data_dir, args.force, log)

    # Stage C + D: per model -> one sheet
    sheets: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for m in models:
        log("-" * 60)
        log(f"[predict] model '{m['id']}'  threshold={m['threshold']}")
        X = build_X(audio, emb, feats, m["meta"])
        try:
            p = predict(m["dir"], X, m["meta"])
        except Exception as e:
            log(f"  ERROR scoring '{m['id']}': {e}"); continue
        thr = m["threshold"]
        result = (p >= thr).astype(int)
        df = pd.DataFrame({
            "audio_group": [a["group"] for a in audio],
            "audio_name": [a["name"] for a in audio],
            "question": [a["qid"] for a in audio],
            "path": [str(a["path"]) for a in audio],
            "probability": np.round(p, 6),
            "threshold": thr,
            "result": result,
            "decision": np.where(result == 1, "CHEAT", "GENUINE"),
        }).sort_values(["audio_group", "audio_name"]).reset_index(drop=True)
        sheets[m["id"][:31]] = df
        df.to_csv(results_dir / f"inference_{m['id']}.csv", index=False)
        n = len(df); nc = int(result.sum())
        summary_rows.append({"model": m["id"], "threshold": thr, "n_audios": n,
                             "n_cheat": nc, "n_genuine": n - nc,
                             "mean_probability": round(float(p.mean()), 4)})
        log(f"  scored {n} audios: {nc} CHEAT / {n - nc} GENUINE  (mean prob {p.mean():.3f})")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "inference_summary.csv", index=False)
    xlsx = results_dir / "inference_results.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="summary", index=False)
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name, index=False)
        log("=" * 74)
        log(f"wrote {xlsx}  (summary + {len(sheets)} model sheet(s))")
    except Exception as e:
        log(f"could not write xlsx ({e}); per-model CSVs are in {results_dir}. pip install openpyxl")
    log(f"SUMMARY:\n{summary.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
