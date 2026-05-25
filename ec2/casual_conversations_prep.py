"""Casual Conversations data prep, runs on EC2 (public corpus, no PII concerns).

A clean public read-vs-spontaneous corpus -- exactly the distinction the cheating
detector learns. Used as its OWN batch (batch='casual'), participating in
train/val/test splits like audios2/4/5.

Steps (mirror allstar_prep.py):
  1. Download casual_conversations.zip from the HF model repo and unzip.
  2. Walk all *.wav under it (any nesting depth), parse speaker/type from name.
  3. Segment each clip (default 2 x 30-60s; whole-clip fallback if shorter).
  4. Transcribe each segment with faster-whisper (medium, word-timestamps on).
  5. Compute the full audios6_eval feature set (55 cols, feat_*) per segment.
  6. Save segment npys into the SAME upload/audio_npy/ folder, append rows to the
     SAME upload/gt.csv. Casual group_ids live in their own CC_NNNNN namespace
     and cannot collide with mercer-mettl G_NNNNN or ALLSTAR AS_NNNNN.

Labels are derived from the filename type token:
    scripted     -> 1   (read-aloud = cheat-analog)
    nonscripted  -> 0   (spontaneous = genuine-analog)

Filename format expected (e.g. 0017_english_nonscripted_4.wav):
    <speaker>_<language>_<scripted|nonscripted>_<clip>.wav

Embedding extraction is NOT done here -- ec2/extract_embeddings.py is
incremental and will pick up the new casual rows on its next run.

================================================================================
RUN
================================================================================
    python ec2/casual_conversations_prep.py --out_dir /home/ubuntu/nn/data

Then extract embeddings (picks up the new rows):
    python ec2/extract_embeddings.py --data_dir /home/ubuntu/nn/data

Then train treating 'casual' as a normal batch, e.g. its own train+test:
    python ec2/neural_baseline_train_v2.py \\
        --data_dir /home/ubuntu/nn/data \\
        --out_dir  /home/ubuntu/nn/runs/casual_selftest \\
        --train_batches casual \\
        --train_only_batches "" \\
        --test_batches "" \\
        --min_duration 5.0 \\
        --use_text_features true
    # (no --test_batches => Mode A: StratifiedGroupKFold on 'casual' alone,
    #  speaker-disjoint, scripted-vs-nonscripted classification)

Outputs (under --out_dir):
    audio_npy/<CC_NNNNN>_<qid>.npy        the segment waveforms
    gt.csv                                  casual rows appended
    casual_transcripts.json                 {npy_filename: transcript dict}
    casual_speaker_mapping.json             {speaker_raw: CC_NNNNN}
    casual_prep_log.txt

================================================================================
PII NOTE
================================================================================
Casual Conversations is a public corpus -- safe to process on EC2. The
transcripts written here (casual_transcripts.json) are of public audio, so
unlike mercer-mettl transcripts they are NOT PII-gated. gt.csv stays
anonymized (CC_NNNNN), as always.
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
CASUAL_ZIP = "casual_conversations.zip"
CASUAL_BATCH = "casual"
CASUAL_REGION = "CASUAL"

# scripted = reading aloud = cheat-analog; nonscripted = spontaneous = genuine.
CASUAL_TYPE_LABELS: dict[str, int] = {
    "scripted": 1,
    "nonscripted": 0,
}

# 0017_english_nonscripted_4.wav -> (speaker, language, type, clip)
CASUAL_FILENAME_RE = re.compile(
    r"^(\d+)_([A-Za-z]+)_(scripted|nonscripted)_(\d+)$",
    re.IGNORECASE,
)
CASUAL_QID_START = 8000

# Segment params. Casual clips are utterance-sized (~30-120s), so we keep the
# segment count low and fall back to whole-clip for short ones.
SEG_MIN_SEC = 30.0
SEG_MAX_SEC = 60.0
SEGMENTS_PER_AUDIO = 2
SEG_SEED = 1337
WHOLE_CLIP_FLOOR = 5.0   # clips shorter than SEG_MIN but >= this -> one whole seg

TARGET_SR = 16000
ACCEPT_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

WHISPER_MODEL_DEFAULT = "medium"

# Only keep these languages (text features + whisper are English). Empty = all.
DEFAULT_LANGUAGES = {"english"}


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("casual_prep")
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

def download_and_unzip(raw_root: Path, log: logging.Logger) -> Path:
    """Populate raw_root/casual/ from the HF repo. Returns the dir with wavs."""
    from huggingface_hub import hf_hub_download

    raw_root.mkdir(parents=True, exist_ok=True)
    target_dir = raw_root / CASUAL_BATCH
    if target_dir.exists() and any(target_dir.rglob("*.wav")):
        log.info("Skip download (already populated at %s)", target_dir)
        return target_dir

    log.info("Downloading %s from %s ...", CASUAL_ZIP, HF_REPO_ID)
    zip_path = hf_hub_download(
        repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        filename=CASUAL_ZIP, local_dir=str(raw_root),
    )
    log.info("Unzipping %s -> %s", zip_path, target_dir)
    target_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_dir)
    try:
        os.remove(zip_path)
    except Exception:
        pass
    return target_dir


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
# Speaker mapping (CC_NNNNN namespace, persisted)
# ----------------------------------------------------------------------------

def load_speaker_map(path: Path) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_speaker_map(m: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def assign_cc_gid(speaker_raw: str, m: dict[str, str]) -> str:
    """One group_id per speaker so ALL their clips (scripted + nonscripted)
    stay in the same fold -- prevents speaker-identity leakage across split."""
    if speaker_raw in m:
        return m[speaker_raw]
    gid = f"CC_{len(m) + 1:05d}"
    m[speaker_raw] = gid
    return gid


# ----------------------------------------------------------------------------
# Transcription (faster-whisper)
# ----------------------------------------------------------------------------

def transcribe_segment(model, wav: np.ndarray) -> dict:
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
                         "Casual rows append to its gt.csv; npys land in "
                         "audio_npy/ alongside the rest.")
    ap.add_argument("--download_dest", default="",
                    help="where to cache the unzipped data. "
                         "Default: <out_dir>/.casual_raw")
    ap.add_argument("--whisper_model", default=WHISPER_MODEL_DEFAULT,
                    help="faster-whisper model size (tiny|base|small|medium|large-v3)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--compute_type", default="float16",
                    help="faster-whisper compute_type: float16 (cuda), int8/float32 (cpu)")
    ap.add_argument("--skip_download", action="store_true",
                    help="assume the unzipped data already exists under --download_dest")
    ap.add_argument("--languages", default="english",
                    help="comma-separated languages to keep (matches filename "
                         "token). Empty = keep all. Default: english.")
    ap.add_argument("--seg_min_sec", type=float, default=SEG_MIN_SEC)
    ap.add_argument("--seg_max_sec", type=float, default=SEG_MAX_SEC)
    ap.add_argument("--segments_per_audio", type=int, default=SEGMENTS_PER_AUDIO)
    ap.add_argument("--max_files", type=int, default=0,
                    help="cap number of source clips processed (0 = all). "
                         "Useful for a quick smoke test.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_dir = out_dir / "audio_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "gt.csv"
    transcripts_path = out_dir / "casual_transcripts.json"
    speaker_map_path = out_dir / "casual_speaker_mapping.json"
    log_path = out_dir / "casual_prep_log.txt"
    log = setup_logging(log_path)

    languages = {s.strip().lower() for s in args.languages.split(",") if s.strip()}
    log.info("out_dir = %s", out_dir)
    log.info("gt_path = %s   (created if absent, otherwise appended)", gt_path)
    log.info("languages filter = %s", languages or "(all)")
    log.info("segments: %d x [%.0f, %.0f]s  (whole-clip fallback >= %.0fs)",
             args.segments_per_audio, args.seg_min_sec, args.seg_max_sec,
             WHOLE_CLIP_FLOOR)

    raw_root = Path(args.download_dest) if args.download_dest else (out_dir / ".casual_raw")
    raw_root.mkdir(parents=True, exist_ok=True)
    if args.skip_download:
        data_dir = raw_root / CASUAL_BATCH
        log.info("--skip_download set; expecting wavs under %s", data_dir)
    else:
        data_dir = download_and_unzip(raw_root, log)

    speaker_map = load_speaker_map(speaker_map_path)
    log.info("speaker map: %d existing entries", len(speaker_map))

    transcripts: dict[str, dict] = {}
    if transcripts_path.exists():
        transcripts = json.loads(transcripts_path.read_text(encoding="utf-8"))
        log.info("loaded %d existing transcripts", len(transcripts))

    existing_gt: pd.DataFrame | None = None
    existing_npy_filenames: set[str] = set()
    if gt_path.exists():
        existing_gt = pd.read_csv(gt_path)
        existing_npy_filenames = set(existing_gt["npy_filename"].astype(str).tolist())
        log.info("existing gt.csv has %d rows; %d unique npy_filename",
                 len(existing_gt), len(existing_npy_filenames))

    # qid starting point: max existing casual question_id + 1, else CASUAL_QID_START
    next_qid = CASUAL_QID_START
    if existing_gt is not None and len(existing_gt):
        cc_mask = existing_gt["batch"].astype(str) == CASUAL_BATCH
        if cc_mask.any():
            next_qid = max(int(existing_gt.loc[cc_mask, "question_id"].max()) + 1,
                           CASUAL_QID_START)
    log.info("casual question_id will start at %d", next_qid)

    log.info("Loading faster-whisper (%s, device=%s, compute_type=%s)...",
             args.whisper_model, args.device, args.compute_type)
    from faster_whisper import WhisperModel
    fw_model = WhisperModel(args.whisper_model, device=args.device,
                            compute_type=args.compute_type)

    rng = np.random.default_rng(SEG_SEED)

    audio_files = sorted(
        f for f in data_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in ACCEPT_EXT
    )
    if args.max_files > 0:
        audio_files = audio_files[:args.max_files]
    log.info("found %d audio files under %s", len(audio_files), data_dir)

    new_rows: list[dict] = []
    n_audios = 0
    n_segs_written = 0
    skipped_bad_name = 0
    skipped_lang = 0
    skipped_too_short = 0
    label_counts = {0: 0, 1: 0}
    t0 = time.time()

    for src in tqdm(audio_files, desc="casual", unit="file"):
        n_audios += 1
        m = CASUAL_FILENAME_RE.match(src.stem)
        if not m:
            skipped_bad_name += 1
            continue
        speaker_raw = m.group(1)
        language = m.group(2).lower()
        clip_type = m.group(3).lower()
        if languages and language not in languages:
            skipped_lang += 1
            continue
        label = CASUAL_TYPE_LABELS.get(clip_type, -1)
        if label not in (0, 1):
            skipped_bad_name += 1
            continue
        gid = assign_cc_gid(speaker_raw, speaker_map)

        try:
            wav, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
        except Exception as e:
            log.warning("Failed load %s: %s", src, e)
            continue
        wav = wav.astype(np.float32, copy=False)
        total_sec = len(wav) / TARGET_SR
        if total_sec < WHOLE_CLIP_FLOOR:
            skipped_too_short += 1
            continue

        segs = sample_segments(total_sec, args.segments_per_audio,
                               args.seg_min_sec, args.seg_max_sec, rng)
        if not segs:
            # clip between WHOLE_CLIP_FLOOR and seg_min_sec -> use the whole clip
            segs = [(0.0, min(total_sec, args.seg_max_sec))]

        for (start, dur) in segs:
            s0 = int(start * TARGET_SR)
            s1 = int((start + dur) * TARGET_SR)
            seg_wav = wav[s0:s1]
            if len(seg_wav) < TARGET_SR * WHOLE_CLIP_FLOOR:
                continue

            qid = next_qid
            next_qid += 1
            out_name = f"{gid}_{qid}.npy"
            if out_name in existing_npy_filenames:
                continue
            out_path = npy_dir / out_name
            np.save(out_path, seg_wav, allow_pickle=False)
            n_segs_written += 1

            tr = transcribe_segment(fw_model, seg_wav)
            transcripts[out_name] = tr
            feats = compute_all_features(out_path, tr["text"], tr["words"])

            row = {
                "group_id": gid,
                "question_id": qid,
                "batch": CASUAL_BATCH,
                "label": label,
                "region": CASUAL_REGION,
                "duration_sec": round(len(seg_wav) / TARGET_SR, 3),
                "npy_filename": out_name,
                # extra provenance columns (harmless; ignored by training)
                "cc_type": clip_type,
                "cc_language": language,
            }
            row.update(feats)
            new_rows.append(row)
            label_counts[label] += 1

            if len(new_rows) % 100 == 0:
                _flush(out_dir, gt_path, existing_gt, new_rows, transcripts,
                       transcripts_path, speaker_map, speaker_map_path, log)

    elapsed = time.time() - t0
    log.info("done in %.1f s. audios=%d  segments_written=%d  "
             "label0(nonscripted)=%d  label1(scripted)=%d  "
             "skipped_bad_name=%d  skipped_lang=%d  skipped_too_short=%d",
             elapsed, n_audios, n_segs_written, label_counts[0], label_counts[1],
             skipped_bad_name, skipped_lang, skipped_too_short)
    log.info("unique speakers (CC group_ids): %d", len(speaker_map))

    _flush(out_dir, gt_path, existing_gt, new_rows, transcripts,
           transcripts_path, speaker_map, speaker_map_path, log)

    log.info("Wrote / updated %s", gt_path)
    log.info("Transcripts -> %s", transcripts_path)
    log.info("Speaker map -> %s", speaker_map_path)
    log.info("NEXT STEP: run ec2/extract_embeddings.py to pick up casual rows in the cache.")
    return 0


def _flush(out_dir: Path, gt_path: Path, existing_gt: pd.DataFrame | None,
           new_rows: list[dict], transcripts: dict, transcripts_path: Path,
           speaker_map: dict, speaker_map_path: Path, log: logging.Logger) -> None:
    if not new_rows:
        return
    df_new = pd.DataFrame(new_rows)
    if existing_gt is not None and len(existing_gt):
        combined = pd.concat([existing_gt, df_new], ignore_index=True, sort=False)
    else:
        combined = df_new

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
