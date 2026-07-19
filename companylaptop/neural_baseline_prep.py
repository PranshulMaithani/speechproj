"""
Company-laptop data prep for neural-baseline training on EC2.

Layout matches the existing audios6_eval pipeline:
    companylaptop/
        audios2/<filename>.wav             <-- flat folder
        audios2GT.csv                       <-- columns: filename, label, [region]
        audios2_features.csv                <-- handcrafted numeric features (optional)
        audios2_transcripts.json            <-- {filename: transcript} (fallback)
        ... same for audios3..audios7 ...

Adding a new batch (audios7, audios8, ...): drop its audios<N>/ folder and an
audios<N>GT.csv next to this script and re-run. discover_batches() finds any
audios<N>/ that has a matching GT csv, so no code edit is needed. (Mixed-region
batches must carry a region column in their GT csv; see DEFAULT_REGION_BY_BATCH.)

Output (under OUTPUT_ROOT):
    upload/audio_npy/<group_id>_<question_id>.npy   <-- upload to EC2
    upload/gt.csv                                    <-- upload to EC2
    local/cid_mapping.json                           <-- KEEP LOCAL
    local/prep_log.txt                               <-- KEEP LOCAL

Labels come from <batch>GT.csv (label column, mapped via TEXT_LABEL_MAP).
Region comes from the GT csv if present, otherwise defaults to IND for
audios2..audios5 (entirely IND per project knowledge); audios6 is mixed and
relies on its GT csv to provide region.

Features:
    1. If <batch>_features.csv exists, all numeric columns get prefixed with
       'feat_' and merged into gt.csv.
    2. Otherwise, if <batch>_transcripts.json exists, text features are
       computed locally via text_features.compute_text_features and merged in.
    3. Otherwise no feat_* columns for that batch.
Transcripts are NEVER written to gt.csv. Real candidate IDs are NEVER
written to gt.csv (they are anonymized to G_NNNNN; the cid -> group mapping
stays in local/cid_mapping.json and is never uploaded).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_features import TEXT_FEATURE_NAMES, compute_text_features

# ----------------------------------------------------------------------------
# Configuration. Edit if your layout differs from the audios6_eval defaults.
# ----------------------------------------------------------------------------

# AUDIO_ROOT defaults to the directory this script lives in, which is
# `companylaptop/` per the existing project layout. All audios2..audios6
# folders and <batch>GT.csv files are siblings of this script.
AUDIO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = AUDIO_ROOT.parent / "data" / "neural_prep_out"

# Baseline batches always considered. main() also AUTO-DISCOVERS any other
# audios<N>/ folder that has a sibling <batch>GT.csv (see discover_batches), so a
# new batch -- audios7, audios8, ... -- is picked up with zero code changes. A
# name listed here but missing on disk is simply skipped with a warning.
BATCHES = ["audios2", "audios3", "audios4", "audios5", "audios6", "audios7"]

# Mirrors LABEL_MAP in audios6_eval.ipynb so the same GT csvs Just Work.
TEXT_LABEL_MAP: dict[str, int] = {
    "read": 1, "cheating": 1, "reading": 1, "scripted": 1, "yes": 1, "1": 1,
    "spontaneous": 0, "not cheating": 0, "not_cheating": 0, "no": 0, "0": 0,
    "genuine": 0,
}

# Batches that are entirely one region -- their GT csv may omit a region column,
# so we fall back to this. Mixed-region batches (audios6, and audios7 unless you
# add it here) are intentionally omitted and MUST carry a region column in their
# GT csv; rows without one get region=None (handled downstream).
DEFAULT_REGION_BY_BATCH: dict[str, str] = {
    "audios2": "IND",
    "audios3": "IND",
    "audios4": "IND",
    "audios5": "IND",
    # audios6 omitted: mixed IND+PHP, rely on its GT.
    # audios7 omitted: provide its region in audios7GT.csv, or add a default here
    #         (e.g. "audios7": "IND") if it is single-region with no column.
}

# NOTE: ALLSTAR ingestion (folders 2676/2677) has been moved to
# ec2/allstar_prep.py so transcription + feature extraction run on the T4
# instead of the company laptop. ALLSTAR is a public corpus -- no PII concern
# about doing the work on EC2. This script now handles mercer-mettl batches
# only.

TARGET_SR = 16000
ACCEPT_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_DURATION_SEC: float | None = 90.0

CID_QID_RE = re.compile(r"^(.+)_(\d{1,3})$")


def _keep_questions() -> set[int] | None:
    """Which question ids get converted to npy. The per-audio system uses Q25/26/27,
    so only those become npy (and thus embeddings) -- the rest are skipped. Override
    with env KEEP_QUESTIONS="all" (keep every question) or e.g. "25,26,27"."""
    import os
    raw = os.environ.get("KEEP_QUESTIONS", "25,26,27").strip().lower()
    if raw in ("", "all", "none"):
        return None
    return {int(x) for x in raw.split(",") if x.strip().isdigit()}


KEEP_QUESTIONS = _keep_questions()


# ----------------------------------------------------------------------------
# Stats / logging
# ----------------------------------------------------------------------------

@dataclass
class PrepStats:
    seen: int = 0
    written: int = 0
    skipped_bad_name: int = 0
    skipped_other_q: int = 0
    skipped_load_error: int = 0
    skipped_already_done: int = 0
    skipped_dup: int = 0


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ----------------------------------------------------------------------------
# Per-batch loaders
# ----------------------------------------------------------------------------

def parse_label(raw) -> int:
    """Map a GT csv label cell to {0, 1, -1}. -1 = unparseable / unlabeled."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return -1
    if isinstance(raw, (int, np.integer)) and int(raw) in (0, 1):
        return int(raw)
    if isinstance(raw, float) and raw in (0.0, 1.0):
        return int(raw)
    s = str(raw).strip().lower()
    return TEXT_LABEL_MAP.get(s, -1)


def _pick(cols: list[str], names: tuple[str, ...]) -> str | None:
    """Case-insensitive column pick from `cols` matching any of `names`."""
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def load_gt_csv(path: Path, log: logging.Logger) -> dict[str, dict]:
    """Return {filename -> {label: int, region: str|None}}. Empty dict if csv missing."""
    if not path.exists():
        log.info("No GT csv at %s -- batch will have no labels", path.name)
        return {}
    df = pd.read_csv(path)
    fn_col = _pick(list(df.columns), ("filename", "file", "name"))
    lbl_col = _pick(list(df.columns), ("label", "class", "cheating", "gt", "label_int", "ground_truth"))
    rgn_col = _pick(list(df.columns), ("region", "country", "locale", "origin", "nationality"))
    if fn_col is None or lbl_col is None:
        log.warning("GT %s missing filename/label columns -- ignoring", path.name)
        return {}
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        fn = str(r[fn_col]).strip()
        if not fn or fn.lower() == "nan":
            continue
        lab = parse_label(r[lbl_col])
        reg = None
        if rgn_col is not None:
            v = r[rgn_col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                v = str(v).strip()
                if v and v.lower() not in ("nan", "none", "unknown", ""):
                    reg = v.upper()
        out[fn] = {"label": lab, "region": reg}
    label_dist = pd.Series([v["label"] for v in out.values()]).value_counts().to_dict()
    log.info("Loaded GT %s: %d rows  label dist=%s  region_col=%s",
             path.name, len(out), label_dist, rgn_col)
    return out


def load_features_csv(path: Path, log: logging.Logger) -> pd.DataFrame | None:
    """Load <batch>_features.csv keyed by filename. All numeric columns are
    prefixed with 'feat_'. Returns DataFrame with `filename` + feat_* columns,
    or None if the file is missing / has no usable columns."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    fn_col = _pick(list(df.columns), ("filename", "file", "name"))
    if fn_col is None:
        log.warning("features csv %s has no filename column -- skipping", path.name)
        return None
    if fn_col != "filename":
        df = df.rename(columns={fn_col: "filename"})
    df["filename"] = df["filename"].astype(str).str.strip()
    feat_cols: list[str] = []
    for c in list(df.columns):
        if c == "filename":
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().any():
            df[c] = coerced
            feat_cols.append(c)
    if not feat_cols:
        log.warning("features csv %s has no numeric columns -- skipping", path.name)
        return None
    rename_map = {c: (c if c.startswith("feat_") else f"feat_{c}") for c in feat_cols}
    df = df[["filename"] + feat_cols].rename(columns=rename_map)
    df = df.drop_duplicates(subset=["filename"], keep="first")
    log.info("Loaded features %s: %d rows  %d numeric cols", path.name, len(df), len(feat_cols))
    return df


def load_transcripts_json(path: Path, log: logging.Logger) -> dict[str, str] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning("Failed to read %s: %s", path.name, e)
        return None
    if not isinstance(data, dict):
        log.warning("%s is not a {filename: transcript} dict -- skipping", path.name)
        return None
    log.info("Loaded transcripts %s: %d entries", path.name, len(data))
    return {str(k).strip(): ("" if v is None else str(v)) for k, v in data.items()}


# ----------------------------------------------------------------------------
# CID mapping
# ----------------------------------------------------------------------------

def load_cid_mapping(path: Path) -> dict[str, str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cid_mapping(mapping: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)


def assign_group_id(cid: str, mapping: dict[str, str]) -> str:
    if cid in mapping:
        return mapping[cid]
    next_idx = len(mapping) + 1
    gid = f"G_{next_idx:05d}"
    mapping[cid] = gid
    return gid


def parse_cid_qid(stem: str) -> tuple[str, int] | None:
    m = CID_QID_RE.match(stem)
    if not m:
        return None
    try:
        return m.group(1), int(m.group(2))
    except ValueError:
        return None


BATCH_DIR_RE = re.compile(r"^audios\d+$")


def _batch_sort_key(name: str) -> tuple[int, int | str]:
    """Order audios<N> numerically (audios2 < audios7 < audios10); anything
    non-numeric sorts after, lexically, so the order is always deterministic."""
    digits = re.sub(r"\D", "", name)
    return (0, int(digits)) if digits else (1, name)


def discover_batches(root: Path, explicit: list[str], log: logging.Logger) -> list[str]:
    """Final batch list = the explicit baseline UNION every audios<N>/ folder on
    disk that has a sibling <batch>GT.csv. This makes a new batch (audios7,
    audios8, ...) Just Work: drop the folder + its GT csv and re-run, no edit
    needed. Requiring the GT csv avoids picking up stray/incomplete folders.
    An explicit name with no folder is kept (the main loop warns + skips it)."""
    found: set[str] = set(explicit)
    for d in sorted(root.iterdir(), key=lambda p: p.name):
        if not (d.is_dir() and BATCH_DIR_RE.match(d.name)):
            continue
        if d.name in found:
            continue
        if (root / f"{d.name}GT.csv").exists():
            found.add(d.name)
            log.info("auto-discovered batch '%s' (has %sGT.csv)", d.name, d.name)
        else:
            log.info("skip folder '%s' -- no %sGT.csv, not treated as a batch",
                     d.name, d.name)
    return sorted(found, key=_batch_sort_key)


# ----------------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------------

def load_and_resample(src: Path, target_sr: int, max_dur: float | None) -> np.ndarray | None:
    try:
        y, _ = librosa.load(str(src), sr=target_sr, mono=True)
    except Exception:
        return None
    if max_dur is not None:
        max_samples = int(target_sr * max_dur)
        if len(y) > max_samples:
            y = y[:max_samples]
    return y.astype(np.float32, copy=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    upload_dir = OUTPUT_ROOT / "upload"
    local_dir = OUTPUT_ROOT / "local"
    npy_dir = upload_dir / "audio_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(local_dir / "prep_log.txt")
    log.info("AUDIO_ROOT  = %s", AUDIO_ROOT)
    log.info("OUTPUT_ROOT = %s", OUTPUT_ROOT)
    log.info("TARGET_SR   = %d Hz, MAX_DURATION_SEC = %s", TARGET_SR, MAX_DURATION_SEC)
    log.info("KEEP_QUESTIONS = %s  (only these qids -> npy -> embeddings; set env "
             "KEEP_QUESTIONS=all to keep every question)",
             sorted(KEEP_QUESTIONS) if KEEP_QUESTIONS else "ALL")

    if not AUDIO_ROOT.exists():
        log.error("AUDIO_ROOT does not exist: %s", AUDIO_ROOT)
        return 1

    batches = discover_batches(AUDIO_ROOT, BATCHES, log)
    log.info("BATCHES (explicit + auto-discovered) = %s", batches)

    cid_mapping_path = local_dir / "cid_mapping.json"
    cid_mapping = load_cid_mapping(cid_mapping_path)
    log.info("Loaded cid mapping: %d existing entries", len(cid_mapping))

    rows: list[dict] = []
    seen: dict[tuple[str, int], Path] = {}
    stats = PrepStats()

    n_with_features = 0
    n_with_transcripts = 0
    n_no_text = 0

    for batch in batches:
        batch_dir = AUDIO_ROOT / batch
        if not batch_dir.exists():
            log.warning("Missing batch dir: %s -- skipping", batch_dir)
            continue

        gt_lookup = load_gt_csv(AUDIO_ROOT / f"{batch}GT.csv", log)
        feat_df = load_features_csv(AUDIO_ROOT / f"{batch}_features.csv", log)
        feat_lookup: dict[str, dict] | None = None
        if feat_df is not None:
            feat_lookup = feat_df.set_index("filename").to_dict(orient="index")

        transcripts = None
        if feat_lookup is None:
            transcripts = load_transcripts_json(AUDIO_ROOT / f"{batch}_transcripts.json", log)

        audio_files = sorted(
            f for f in batch_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in ACCEPT_EXT
        )
        log.info("%s: %d audio files in %s", batch, len(audio_files), batch_dir)

        for src in tqdm(audio_files, desc=batch, unit="file"):
            stats.seen += 1
            parsed = parse_cid_qid(src.stem)
            if parsed is None:
                stats.skipped_bad_name += 1
                log.warning("Could not parse cid/qid from '%s' -- skipping", src.name)
                continue
            cid, qid = parsed
            # Only Q25/26/27 (default) become npy -> embeddings; skip the rest.
            if KEEP_QUESTIONS is not None and qid not in KEEP_QUESTIONS:
                stats.skipped_other_q += 1
                continue
            gid = assign_group_id(cid, cid_mapping)
            out_name = f"{gid}_{qid}.npy"
            out_path = npy_dir / out_name
            key = (gid, qid)

            # Label + region from GT csv (filename keyed). Falls back to -1 / default region.
            gt_entry = gt_lookup.get(src.name, {})
            label = gt_entry.get("label", -1)
            region = gt_entry.get("region")
            if region is None:
                region = DEFAULT_REGION_BY_BATCH.get(batch)

            # Collision: same candidate + question seen earlier in this run.
            if key in seen:
                stats.skipped_dup += 1
                log.warning("Duplicate %s qid=%d (prev=%s, now=%s) -- keeping first",
                            gid, qid, seen[key].name, src.name)
                continue
            seen[key] = src

            if out_path.exists():
                stats.skipped_already_done += 1
            else:
                y = load_and_resample(src, TARGET_SR, MAX_DURATION_SEC)
                if y is None or len(y) < TARGET_SR * 0.5:
                    stats.skipped_load_error += 1
                    log.warning("Failed/too-short load: %s", src)
                    continue
                np.save(out_path, y, allow_pickle=False)
                stats.written += 1

            try:
                n_samples = int(np.load(out_path, mmap_mode="r").shape[0])
            except Exception:
                n_samples = 0
            duration = n_samples / TARGET_SR if n_samples else 0.0

            row = {
                "group_id": gid,
                "question_id": qid,
                "batch": batch,
                "label": label,
                "region": region,
                "duration_sec": round(duration, 3),
                "npy_filename": out_name,
            }

            # Feature attachment, in priority order:
            #   1. precomputed <batch>_features.csv (numeric columns, feat_*)
            #   2. local compute_text_features over <batch>_transcripts.json
            if feat_lookup is not None and src.name in feat_lookup:
                row.update(feat_lookup[src.name])
                n_with_features += 1
            elif transcripts is not None and src.name in transcripts:
                row.update(compute_text_features(transcripts[src.name]))
                n_with_transcripts += 1
            else:
                n_no_text += 1

            rows.append(row)

    if not rows:
        log.error("No rows produced. Check AUDIO_ROOT/BATCHES and that GT csvs exist.")
        return 2

    gt = pd.DataFrame(rows)

    # Defensive: scrub anything that should never be uploaded.
    for forbidden in ("transcript", "transcript_text", "candidate_id_real", "candidate_id"):
        if forbidden in gt.columns:
            log.error("Forbidden column '%s' present -- dropping before save", forbidden)
            gt = gt.drop(columns=[forbidden])

    # Predictable column ordering: metadata first, feat_* next, anything else last.
    meta_order = ["group_id", "question_id", "batch", "label", "region",
                  "duration_sec", "npy_filename"]
    feat_cols = sorted(c for c in gt.columns if c.startswith("feat_"))
    other = [c for c in gt.columns if c not in meta_order and not c.startswith("feat_")]
    gt = gt[[c for c in meta_order if c in gt.columns] + feat_cols + other]

    gt_path = upload_dir / "gt.csv"
    gt.to_csv(gt_path, index=False)
    save_cid_mapping(cid_mapping, cid_mapping_path)

    log.info("---- summary ----")
    log.info("seen                   = %d", stats.seen)
    log.info("written                = %d", stats.written)
    log.info("already done (kept)    = %d", stats.skipped_already_done)
    log.info("skipped (other question)= %d", stats.skipped_other_q)
    log.info("skipped (bad name)     = %d", stats.skipped_bad_name)
    log.info("skipped (load fail)    = %d", stats.skipped_load_error)
    log.info("skipped (duplicates)   = %d", stats.skipped_dup)
    log.info("unique candidates      = %d", len(cid_mapping))
    log.info("rows in gt.csv         = %d", len(gt))
    log.info("rows w/ features.csv   = %d", n_with_features)
    log.info("rows w/ transcripts JS = %d", n_with_transcripts)
    log.info("rows w/o any feats     = %d", n_no_text)
    log.info("feat_* columns written = %d", len(feat_cols))

    log.info("label distribution     = %s", gt["label"].value_counts(dropna=False).to_dict())
    by_batch_lbl = gt.groupby("batch")["label"].value_counts().unstack(fill_value=0)
    log.info("per-batch x label\n%s", by_batch_lbl.to_string())
    if gt["region"].notna().any():
        by_region_lbl = (gt.dropna(subset=["region"])
                          .groupby("region")["label"].value_counts().unstack(fill_value=0))
        log.info("per-region x label\n%s", by_region_lbl.to_string())
        by_batch_region = (gt.dropna(subset=["region"])
                            .groupby(["batch", "region"]).size().unstack(fill_value=0))
        log.info("per-batch x region\n%s", by_batch_region.to_string())

    log.info("UPLOAD this folder to EC2: %s", upload_dir)
    log.info("DO NOT UPLOAD: %s", local_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
