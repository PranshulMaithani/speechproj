"""ALLSTAR data prep, runs on EC2 (no PII concerns -- public corpus).

Steps:
  1. Download 2676.zip + 2677.zip from the HF model repo and unzip.
  2. Walk subfolders, parse speaker + task code from each filename.
  3. Random 3 x 40-60s non-overlapping segments per audio.
  4. Transcribe each segment with faster-whisper (medium, word-timestamps on).
  5. Compute the full audios6_eval feature set (55 cols, feat_*) per segment.
  6. Save segment npys into the SAME upload/audio_npy/ folder as mercer-mettl,
     append rows to the SAME upload/gt.csv. ALLSTAR group_ids live in their
     own AS_NNNNN namespace and cannot collide with mercer-mettl G_NNNNN.

Labels are derived from the task code suffix of each filename:
    ST1..ST4, LPP, NWS  -> 1 (read = cheat-analog)
    QNA, HT1, HT2, DHR  -> 0 (spontaneous = genuine-analog)
Unmapped codes are skipped with a warning.

Embedding extraction is NOT done here -- ec2/extract_embeddings.py is
incremental and will pick up the new ALLSTAR rows on its next run.

Run:
    python ec2/allstar_prep.py --out_dir /opt/data/upload

Outputs (under --out_dir):
    audio_npy/<AS_NNNNN>_<qid>.npy        the 40-60s segment waveforms
    gt.csv                                  ALLSTAR rows appended
    allstar_transcripts.json                {npy_filename: transcript dict}
    allstar_speaker_mapping.json            {spkID: AS_NNNNN}
    allstar_prep_log.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_text_features import compute_all_features, ALL_FEATURES  # noqa: E402

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

HF_REPO_ID = "Pransfrance/speechproj-models"
HF_REPO_TYPE = "model"
ALLSTAR_ZIPS = ["2676.zip", "2677.zip"]
ALLSTAR_BATCHES = ["2676", "2677"]

ALLSTAR_TASK_LABELS: dict[str, int] = {
    "ST1": 1, "ST2": 1, "ST3": 1, "ST4": 1,
    "LPP": 1,
    "NWS": 1,
    "QNA": 0,
    "HT1": 0, "HT2": 0,
    "DHR": 0,
}
ALLSTAR_REGION = "ALLSTAR"
ALLSTAR_FILENAME_RE = re.compile(
    r"^ALL_(\d+)_[MFmf]_[A-Za-z]+_[A-Za-z]+_([A-Za-z0-9]+)",
)
ALLSTAR_QID_START = 9000

SEG_MIN_SEC = 40.0
SEG_MAX_SEC = 60.0
SEGMENTS_PER_AUDIO = 3
SEG_SEED = 1337

TARGET_SR = 16000
ACCEPT_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

WHISPER_MODEL_DEFAULT = "medium"  # for faster-whisper transcription


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("allstar_prep")
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
# Download + unzip
# ----------------------------------------------------------------------------

def download_and_unzip(raw_root: Path, log: logging.Logger) -> None:
    """Populate raw_root/2676/ and raw_root/2677/ from the HF repo."""
    from huggingface_hub import hf_hub_download

    raw_root.mkdir(parents=True, exist_ok=True)
    for zip_name in ALLSTAR_ZIPS:
        batch_name = zip_name.replace(".zip", "")
        target_dir = raw_root / batch_name
        if target_dir.exists() and any(target_dir.iterdir()):
            log.info("Skip %s (already populated at %s)", zip_name, target_dir)
            continue
        log.info("Downloading %s from %s ...", zip_name, HF_REPO_ID)
        zip_path = hf_hub_download(
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
            filename=zip_name, local_dir=str(raw_root),
        )
        log.info("Unzipping %s -> %s", zip_path, target_dir)
        target_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target_dir)
        # Flatten if zip extracted to a single nested folder with the same name.
        entries = [p for p in target_dir.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir() and entries[0].name == batch_name:
            inner = entries[0]
            for p in inner.iterdir():
                shutil.move(str(p), str(target_dir / p.name))
            inner.rmdir()
        try:
            os.remove(zip_path)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Segment sampling (deterministic with a seeded RNG)
# ----------------------------------------------------------------------------

def sample_segments(total_sec: float, n_max: int, min_sec: float, max_sec: float,
                    rng: np.random.Generator) -> list[tuple[float, float]]:
    if total_sec < min_sec:
        return []
    n_fit = min(n_max, int(total_sec // min_sec))
    if n_fit <= 0:
        return []
    bucket = total_sec / n_fit
    segs: list[tuple[float, float]] = []
    for i in range(n_fit):
        b0 = i * bucket
        b1 = b0 + bucket
        cap = min(max_sec, bucket)
        if cap < min_sec:
            continue
        dur = float(rng.uniform(min_sec, cap))
        start = float(rng.uniform(b0, b1 - dur))
        segs.append((start, dur))
    return segs


# ----------------------------------------------------------------------------
# Speaker mapping (AS_NNNNN namespace, persisted)
# ----------------------------------------------------------------------------

def load_speaker_map(path: Path) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_speaker_map(m: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def assign_as_gid(speaker_raw: str, m: dict[str, str]) -> str:
    if speaker_raw in m:
        return m[speaker_raw]
    gid = f"AS_{len(m) + 1:05d}"
    m[speaker_raw] = gid
    return gid


# ----------------------------------------------------------------------------
# Transcription (faster-whisper)
# ----------------------------------------------------------------------------

def transcribe_segment(model, wav: np.ndarray) -> dict:
    """Return {"text": str, "words": [{word, start, end}, ...]} via faster-whisper."""
    # faster-whisper accepts a numpy float32 mono wav at TARGET_SR.
    segs, info = model.transcribe(
        wav, language="en", beam_size=1, word_timestamps=True,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 100},
    )
    parts: list[str] = []
    words: list[dict] = []
    for seg in segs:
        parts.append(seg.text)
        if seg.words:
            for w in seg.words:
                words.append({"word": (w.word or "").strip(),
                              "start": float(round(w.start, 3)),
                              "end":   float(round(w.end, 3))})
    return {"text": " ".join(parts).strip(),
            "words": words,
            "duration_sec": float(round(info.duration, 2))}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="upload folder. Same one mercer-mettl prep wrote to. "
                         "ALLSTAR rows append to its gt.csv and npys land in "
                         "audio_npy/ alongside mercer-mettl ones.")
    ap.add_argument("--download_dest", default="",
                    help="where to cache the unzipped 2676/2677. Default: <out_dir>/.allstar_raw")
    ap.add_argument("--whisper_model", default=WHISPER_MODEL_DEFAULT,
                    help="faster-whisper model size (tiny|base|small|medium|large-v3)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--compute_type", default="float16",
                    help="faster-whisper compute_type: float16 (cuda), int8 (cpu), float32 (cpu)")
    ap.add_argument("--skip_download", action="store_true",
                    help="assume 2676/ and 2677/ already exist under --download_dest")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_dir = out_dir / "audio_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "gt.csv"
    transcripts_path = out_dir / "allstar_transcripts.json"
    speaker_map_path = out_dir / "allstar_speaker_mapping.json"
    log_path = out_dir / "allstar_prep_log.txt"
    log = setup_logging(log_path)

    log.info("out_dir = %s", out_dir)
    log.info("gt_path = %s   (will be created if absent, otherwise appended)", gt_path)

    raw_root = Path(args.download_dest) if args.download_dest else (out_dir / ".allstar_raw")
    raw_root.mkdir(parents=True, exist_ok=True)
    if args.skip_download:
        log.info("--skip_download set; expecting %s and %s under %s",
                 ALLSTAR_BATCHES[0], ALLSTAR_BATCHES[1], raw_root)
    else:
        download_and_unzip(raw_root, log)

    speaker_map = load_speaker_map(speaker_map_path)
    log.info("speaker map: %d existing entries", len(speaker_map))

    transcripts: dict[str, dict] = {}
    if transcripts_path.exists():
        transcripts = json.loads(transcripts_path.read_text(encoding="utf-8"))
        log.info("loaded %d existing transcripts", len(transcripts))

    # Load existing gt for incremental skip.
    existing_gt: pd.DataFrame | None = None
    existing_npy_filenames: set[str] = set()
    if gt_path.exists():
        existing_gt = pd.read_csv(gt_path)
        existing_npy_filenames = set(existing_gt["npy_filename"].astype(str).tolist())
        log.info("existing gt.csv has %d rows; %d unique npy_filename",
                 len(existing_gt), len(existing_npy_filenames))

    # Decide qid starting point: max of existing ALLSTAR question_ids (if any) + 1
    next_qid = ALLSTAR_QID_START
    if existing_gt is not None and len(existing_gt):
        as_mask = existing_gt["batch"].astype(str).isin(ALLSTAR_BATCHES)
        if as_mask.any():
            next_qid = max(int(existing_gt.loc[as_mask, "question_id"].max()) + 1, ALLSTAR_QID_START)
    log.info("ALLSTAR question_id will start at %d", next_qid)

    # Load faster-whisper.
    log.info("Loading faster-whisper (%s, device=%s, compute_type=%s)...",
             args.whisper_model, args.device, args.compute_type)
    from faster_whisper import WhisperModel
    fw_model = WhisperModel(args.whisper_model, device=args.device,
                            compute_type=args.compute_type)

    rng = np.random.default_rng(SEG_SEED)

    new_rows: list[dict] = []
    n_audios = 0
    n_segs_written = 0
    skipped_unknown_task: dict[str, int] = {}
    skipped_bad_name = 0
    skipped_too_short = 0
    t0 = time.time()

    for batch in ALLSTAR_BATCHES:
        batch_dir = raw_root / batch
        if not batch_dir.exists():
            log.warning("Batch dir missing: %s -- skipping", batch_dir)
            continue
        audio_files = sorted(
            f for f in batch_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in ACCEPT_EXT
        )
        log.info("[%s] %d audios under %s", batch, len(audio_files), batch_dir)

        for src in tqdm(audio_files, desc=batch, unit="file"):
            n_audios += 1
            m = ALLSTAR_FILENAME_RE.match(src.stem)
            if not m:
                skipped_bad_name += 1
                continue
            speaker_raw = m.group(1)
            task_code = m.group(2).upper()
            label = ALLSTAR_TASK_LABELS.get(task_code, -1)
            if label not in (0, 1):
                skipped_unknown_task[task_code] = skipped_unknown_task.get(task_code, 0) + 1
                continue
            gid = assign_as_gid(speaker_raw, speaker_map)

            try:
                wav, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
            except Exception as e:
                log.warning("Failed load %s: %s", src, e)
                continue
            wav = wav.astype(np.float32, copy=False)
            total_sec = len(wav) / TARGET_SR
            if total_sec < SEG_MIN_SEC:
                skipped_too_short += 1
                continue

            segs = sample_segments(total_sec, SEGMENTS_PER_AUDIO, SEG_MIN_SEC, SEG_MAX_SEC, rng)
            for (start, dur) in segs:
                s0 = int(start * TARGET_SR)
                s1 = int((start + dur) * TARGET_SR)
                seg_wav = wav[s0:s1]
                if len(seg_wav) < TARGET_SR * SEG_MIN_SEC * 0.95:
                    continue

                qid = next_qid
                next_qid += 1
                out_name = f"{gid}_{qid}.npy"
                if out_name in existing_npy_filenames:
                    # Defensive: shouldn't happen due to qid counter, but skip if so.
                    continue
                out_path = npy_dir / out_name
                np.save(out_path, seg_wav, allow_pickle=False)
                n_segs_written += 1

                # Transcribe + features
                tr = transcribe_segment(fw_model, seg_wav)
                transcripts[out_name] = tr
                feats = compute_all_features(out_path, tr["text"], tr["words"])

                row = {
                    "group_id": gid,
                    "question_id": qid,
                    "batch": batch,
                    "label": label,
                    "region": ALLSTAR_REGION,
                    "duration_sec": round(len(seg_wav) / TARGET_SR, 3),
                    "npy_filename": out_name,
                }
                row.update(feats)
                new_rows.append(row)

                # Periodic flush so a crash doesn't lose work.
                if len(new_rows) % 100 == 0:
                    _flush(out_dir, gt_path, existing_gt, new_rows, transcripts,
                           transcripts_path, speaker_map, speaker_map_path, log)

    elapsed = time.time() - t0
    log.info("done in %.1f s. audios=%d  segments_written=%d  "
             "skipped_bad_name=%d  skipped_too_short=%d  skipped_unknown_task=%s",
             elapsed, n_audios, n_segs_written, skipped_bad_name,
             skipped_too_short, dict(sorted(skipped_unknown_task.items())))

    # Final flush.
    _flush(out_dir, gt_path, existing_gt, new_rows, transcripts,
           transcripts_path, speaker_map, speaker_map_path, log)

    log.info("Wrote / updated %s", gt_path)
    log.info("Transcripts -> %s", transcripts_path)
    log.info("Speaker map -> %s", speaker_map_path)
    log.info("NEXT STEP: run ec2/extract_embeddings.py to pick up ALLSTAR rows in the cache.")
    return 0


def _flush(out_dir: Path, gt_path: Path, existing_gt: pd.DataFrame | None,
           new_rows: list[dict], transcripts: dict, transcripts_path: Path,
           speaker_map: dict, speaker_map_path: Path, log: logging.Logger) -> None:
    if not new_rows:
        return
    df_new = pd.DataFrame(new_rows)
    if existing_gt is not None and len(existing_gt):
        # Align columns: use union, fill missing with NaN/0 as appropriate
        combined = pd.concat([existing_gt, df_new], ignore_index=True, sort=False)
    else:
        combined = df_new

    # Column order: metadata first, feat_* sorted, anything else last
    meta = ["group_id", "question_id", "batch", "label", "region",
            "duration_sec", "npy_filename"]
    feat = sorted(c for c in combined.columns if c.startswith("feat_"))
    other = [c for c in combined.columns if c not in meta and c not in feat]
    combined = combined[[c for c in meta if c in combined.columns] + feat + other]

    combined.to_csv(gt_path, index=False)
    transcripts_path.write_text(json.dumps(transcripts, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    save_speaker_map(speaker_map, speaker_map_path)


if __name__ == "__main__":
    sys.exit(main())
