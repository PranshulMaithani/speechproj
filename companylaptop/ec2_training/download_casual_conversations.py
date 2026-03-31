"""
Download & process Meta Casual Conversations v2 (English only).
================================================================
Downloads CCv2 zip parts one at a time, extracts ONLY English
scripted/nonscripted MP4s, converts to WAV audio, deletes zip.

Filename pattern inside zips:
    {participant_id}_{language}_{scripted|nonscripted}_{index}.mp4
    Example: 5230_english_scripted_0.mp4
             5230_english_nonscripted_3.mp4

We only keep: *_english_scripted_*.mp4 and *_english_nonscripted_*.mp4

Usage:
    # Download from links file (one part at a time to save disk)
    python download_casual_conversations.py --links ccv2_links.txt

    # Process already-downloaded zips
    python download_casual_conversations.py --zip-dir /path/to/zips

    # Process already-extracted mp4s
    python download_casual_conversations.py --mp4-dir /path/to/mp4s

    # Limit number of files per class
    python download_casual_conversations.py --links ccv2_links.txt --max-per-class 3000
"""

import argparse
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SR = 16000
# Use absolute paths to avoid cwd issues
_SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = _SCRIPT_DIR / "data" / "casual_conversations"
AUDIO_DIR = OUT_DIR / "audio_scripted"
AUDIO_DIR_UNSCRIPTED = OUT_DIR / "audio_nonscripted"
MP4_TEMP = OUT_DIR / "mp4_temp"

# Pattern to match English files
ENGLISH_PATTERN = re.compile(r"(\d+)_english_(scripted|nonscripted)_(\d+)\.mp4$", re.IGNORECASE)
# Pattern to match ANY language file
ANY_LANG_PATTERN = re.compile(r"(\d+)_([a-zA-Z]+)_(scripted|nonscripted)_(\d+)\.mp4$", re.IGNORECASE)


def log(msg, indent=0):
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


def get_wav_counts():
    """Get current WAV counts on disk."""
    s = len(list(AUDIO_DIR.glob("*.wav"))) if AUDIO_DIR.exists() else 0
    n = len(list(AUDIO_DIR_UNSCRIPTED.glob("*.wav"))) if AUDIO_DIR_UNSCRIPTED.exists() else 0
    return s, n


def parse_links_file(links_path):
    """Parse the CCv2 download links file."""
    links = {}
    with open(links_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("file_name"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                fname = parts[0].strip()
                url = parts[1].strip()
                # CCv2_part_*.zip (videos) + CCv2_samples.zip. Skip CCv2_frames_part_* (images only)
                if fname == "CCv2_samples.zip" or (fname.startswith("CCv2_part_") and fname.endswith(".zip")):
                    links[fname] = url
    return links


def inspect_zip(zip_path):
    """Log detailed contents of a zip file — languages, counts, sizes."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            all_names = zf.namelist()
            log(f"ZIP CONTENTS: {zip_path.name}", indent=1)
            log(f"Total files inside: {len(all_names)}", indent=2)

            # Categorize by language
            lang_counts = Counter()
            lang_scripted = Counter()
            lang_nonscripted = Counter()
            non_mp4 = []

            for name in all_names:
                basename = Path(name).name
                match = ANY_LANG_PATTERN.match(basename)
                if match:
                    lang = match.group(2).lower()
                    stype = match.group(3).lower()
                    lang_counts[lang] += 1
                    if stype == "scripted":
                        lang_scripted[lang] += 1
                    else:
                        lang_nonscripted[lang] += 1
                elif basename:  # skip directory entries
                    non_mp4.append(basename)

            # Print language breakdown
            if lang_counts:
                log(f"Languages found:", indent=2)
                for lang in sorted(lang_counts, key=lang_counts.get, reverse=True):
                    s = lang_scripted.get(lang, 0)
                    n = lang_nonscripted.get(lang, 0)
                    marker = " <--- THIS IS WHAT WE WANT" if lang == "english" else ""
                    log(f"{lang:<20s}: {lang_counts[lang]:>4d} total ({s} scripted, {n} nonscripted){marker}", indent=3)

            if non_mp4:
                log(f"Non-MP4 files: {len(non_mp4)} (e.g. {non_mp4[:3]})", indent=2)

            # English details
            english_files = [n for n in all_names if ENGLISH_PATTERN.search(Path(n).name)]
            if english_files:
                log(f"", indent=0)
                log(f"ENGLISH FILES TO EXTRACT: {len(english_files)}", indent=1)
                for ef in english_files[:10]:
                    info = zf.getinfo(ef)
                    size_mb = info.file_size / 1e6
                    log(f"{Path(ef).name} ({size_mb:.1f} MB)", indent=3)
                if len(english_files) > 10:
                    log(f"... and {len(english_files) - 10} more", indent=3)
            else:
                log(f"NO ENGLISH FILES in this zip!", indent=1)

            return english_files

    except zipfile.BadZipFile:
        log(f"ERROR: {zip_path.name} is corrupted!", indent=1)
        return []
    except Exception as e:
        log(f"ERROR inspecting {zip_path.name}: {e}", indent=1)
        return []


def extract_english_from_zip(zip_path, mp4_out_dir):
    """Extract only English scripted/nonscripted MP4s from a zip."""
    mp4_out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            all_names = zf.namelist()
            english_files = [n for n in all_names if ENGLISH_PATTERN.search(Path(n).name)]

            if not english_files:
                return extracted

            for name in tqdm(english_files, desc=f"    Extracting"):
                basename = Path(name).name
                out_path = mp4_out_dir / basename

                if out_path.exists():
                    extracted.append(out_path)
                    continue

                try:
                    data = zf.read(name)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    extracted.append(out_path)
                except Exception as e:
                    log(f"Error extracting {name}: {e}", indent=2)

    except zipfile.BadZipFile:
        log(f"ERROR: {zip_path.name} is corrupted. Re-download it.", indent=1)
    except Exception as e:
        log(f"ERROR processing {zip_path.name}: {e}", indent=1)

    log(f"Extracted {len(extracted)} English MP4s", indent=2)
    return extracted


def mp4_to_wav(mp4_path, wav_path):
    """Extract audio from MP4 to 16kHz mono WAV using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-i", str(mp4_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(SR), "-ac", "1",
            "-y", "-loglevel", "error",
            str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except Exception:
        return False


def get_duration(wav_path):
    """Get duration of a WAV file."""
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return info.duration
    except Exception:
        try:
            import librosa
            return librosa.get_duration(path=str(wav_path))
        except Exception:
            return 0


def convert_mp4s(mp4_dir, max_per_class=None):
    """Convert English MP4s to WAV."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR_UNSCRIPTED.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(Path(mp4_dir).glob("*.mp4"))
    if not mp4_files:
        return

    converted = 0
    skipped_limit = 0
    skipped_duration = 0
    skipped_ffmpeg = 0

    for mp4 in tqdm(mp4_files, desc="    MP4 -> WAV"):
        match = ENGLISH_PATTERN.match(mp4.name)
        if not match:
            continue

        script_type = match.group(2)

        # Check max per class
        if max_per_class:
            s, n = get_wav_counts()
            if script_type == "scripted" and s >= max_per_class:
                skipped_limit += 1
                continue
            if script_type == "nonscripted" and n >= max_per_class:
                skipped_limit += 1
                continue

        wav_dir = AUDIO_DIR if script_type == "scripted" else AUDIO_DIR_UNSCRIPTED
        wav_path = wav_dir / (mp4.stem + ".wav")

        if not wav_path.exists():
            if not mp4_to_wav(mp4, wav_path):
                skipped_ffmpeg += 1
                continue

        duration = get_duration(wav_path)
        if duration < 3.0:
            wav_path.unlink(missing_ok=True)
            skipped_duration += 1
            continue

        converted += 1

    s, n = get_wav_counts()
    log(f"Conversion done: {converted} new WAVs", indent=2)
    if skipped_limit:
        log(f"Skipped (class limit reached): {skipped_limit}", indent=2)
    if skipped_duration:
        log(f"Skipped (too short <3s): {skipped_duration}", indent=2)
    if skipped_ffmpeg:
        log(f"Skipped (ffmpeg failed): {skipped_ffmpeg}", indent=2)
    log(f"Total on disk: {s} scripted, {n} nonscripted", indent=2)


def build_ccv2_manifest():
    """Build manifest from all existing WAV files on disk."""
    rows = []
    for wav_dir, script_type, label, label_int in [
        (AUDIO_DIR, "scripted", "read", 1),
        (AUDIO_DIR_UNSCRIPTED, "nonscripted", "spontaneous", 0),
    ]:
        if not wav_dir.exists():
            continue
        for wav in sorted(wav_dir.glob("*.wav")):
            match = re.match(r"(\d+)_english_", wav.name)
            pid = match.group(1) if match else ""
            duration = get_duration(wav)
            if duration < 3.0:
                continue
            rows.append({
                "filepath": str(wav.resolve()),
                "filename": wav.name,
                "source": "casual_conversations",
                "label": label,
                "label_int": label_int,
                "script_type": script_type,
                "duration_sec": round(duration, 2),
                "speaker_id": pid,
            })

    df = pd.DataFrame(rows)
    manifest_path = OUT_DIR / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    scripted_count = (df["label_int"] == 1).sum()
    nonscripted_count = (df["label_int"] == 0).sum()
    log(f"")
    log(f"{'='*60}")
    log(f"FINAL MANIFEST: {manifest_path}")
    log(f"{'='*60}")
    log(f"Total:       {len(df)}")
    log(f"Scripted:    {scripted_count} (label=read)")
    log(f"Nonscripted: {nonscripted_count} (label=spontaneous)")
    if "duration_sec" in df.columns and len(df) > 0:
        log(f"Duration:    {df['duration_sec'].sum()/3600:.1f} hours")
    return df


def process_one_zip(zip_path, max_per_class, keep_zips, keep_mp4s):
    """Full pipeline for one zip: inspect -> extract -> convert -> cleanup."""
    log(f"")
    log(f"{'─'*60}")

    # Inspect
    english_files = inspect_zip(zip_path)

    if not english_files:
        if not keep_zips:
            log(f"Deleting {zip_path.name} (no English files)...", indent=1)
            zip_path.unlink(missing_ok=True)
        return

    # Extract
    extracted = extract_english_from_zip(zip_path, MP4_TEMP)

    # Delete zip
    if not keep_zips:
        size_gb = zip_path.stat().st_size / 1e9 if zip_path.exists() else 0
        log(f"Deleting {zip_path.name} ({size_gb:.1f} GB freed)", indent=1)
        zip_path.unlink(missing_ok=True)

    # Convert
    if extracted:
        convert_mp4s(MP4_TEMP, max_per_class)

    # Delete MP4s
    if not keep_mp4s:
        n_deleted = 0
        for f in MP4_TEMP.glob("*.mp4"):
            f.unlink()
            n_deleted += 1
        if n_deleted:
            log(f"Cleaned up {n_deleted} MP4 temp files", indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Download & process Meta Casual Conversations v2 (English only)")
    parser.add_argument("--links", type=str, default=None,
                        help="Path to file with CCv2 download links (tab-separated: filename\\turl)")
    parser.add_argument("--zip-dir", type=str, default=None,
                        help="Directory containing already-downloaded CCv2 zip files")
    parser.add_argument("--mp4-dir", type=str, default=None,
                        help="Directory containing already-extracted English MP4 files")
    parser.add_argument("--max-per-class", type=int, default=5000,
                        help="Max files per class (scripted/nonscripted). Default: 5000")
    parser.add_argument("--max-parts", type=int, default=None,
                        help="Only download this many zip parts (for testing)")
    parser.add_argument("--keep-zips", action="store_true",
                        help="Don't delete zip files after extraction")
    parser.add_argument("--keep-mp4s", action="store_true",
                        help="Don't delete MP4 files after audio extraction")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MP4_TEMP.mkdir(parents=True, exist_ok=True)

    log(f"")
    log(f"{'='*60}")
    log(f"CCv2 DOWNLOADER")
    log(f"{'='*60}")
    log(f"Output dir:     {OUT_DIR}")
    log(f"Scripted WAVs:  {AUDIO_DIR}")
    log(f"Nonscripted:    {AUDIO_DIR_UNSCRIPTED}")
    log(f"Max per class:  {args.max_per_class}")

    s, n = get_wav_counts()
    log(f"Already on disk: {s} scripted, {n} nonscripted")

    # Check if already processed
    manifest_path = OUT_DIR / "manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        existing = df["filepath"].apply(os.path.exists).sum()
        if existing > 100:
            log(f"")
            log(f"Casual Conversations already processed ({existing} files).")
            log(f"  Scripted:    {(df['label_int']==1).sum()}")
            log(f"  Nonscripted: {(df['label_int']==0).sum()}")
            log(f"Delete manifest.csv to re-process.")
            return

    # Mode 1: Process already-extracted MP4s
    if args.mp4_dir:
        convert_mp4s(args.mp4_dir, args.max_per_class)
        build_ccv2_manifest()
        return

    # Mode 2: Process already-downloaded zips
    if args.zip_dir:
        zip_files = sorted(list(Path(args.zip_dir).glob("CCv2_part_*.zip")))
        log(f"Found {len(zip_files)} zip files in {args.zip_dir}")
        for zf in zip_files:
            process_one_zip(zf, args.max_per_class, args.keep_zips, args.keep_mp4s)
        build_ccv2_manifest()
        return

    # Mode 3: Download from links file
    if args.links:
        links = parse_links_file(args.links)
        if not links:
            log(f"No CCv2_part_*.zip links found in {args.links}")
            log(f"Expected format (tab-separated):")
            log(f"  CCv2_part_1.zip\\thttps://scontent.xx.fbcdn.net/...")
            return

        # Sort: CCv2_samples.zip first (for verification), then by part number
        def sort_key(name):
            if name == "CCv2_samples.zip":
                return -1  # always first
            m = re.search(r"part_(\d+)", name)
            return int(m.group(1)) if m else 0
        sorted_links = sorted(links.items(), key=lambda x: sort_key(x[0]))

        log(f"")
        log(f"Found {len(sorted_links)} CCv2_part links in {args.links}")

        if args.max_parts:
            sorted_links = sorted_links[:args.max_parts]
            log(f"Limiting to {args.max_parts} parts")

        zip_dir = OUT_DIR / "zips"
        zip_dir.mkdir(exist_ok=True)

        # ── SAMPLE RUN: Process just the first zip to verify everything works ──
        first_name, first_url = sorted_links[0]
        log(f"")
        log(f"{'='*60}")
        log(f"SAMPLE RUN — downloading {first_name} first to verify pipeline")
        log(f"{'='*60}")

        sample_zip = zip_dir / first_name
        if not sample_zip.exists():
            log(f"Downloading {first_name}...")
            try:
                subprocess.run(
                    ["wget", "-O", str(sample_zip), "-q", "--show-progress", first_url],
                    timeout=7200,
                )
            except Exception as e:
                log(f"Download error: {e}")

        if sample_zip.exists() and sample_zip.stat().st_size > 1000:
            process_one_zip(sample_zip, args.max_per_class, args.keep_zips, args.keep_mp4s)
            s, n = get_wav_counts()
            log(f"")
            log(f"{'='*60}")
            log(f"SAMPLE RESULT: {s} scripted, {n} nonscripted WAVs on disk")
            if s > 0 or n > 0:
                log(f"Pipeline WORKS. Continuing with remaining {len(sorted_links) - 1} zips...")
            else:
                log(f"WARNING: No WAVs produced! Check the output above for errors.")
                log(f"The zip might not contain English files, trying next ones anyway...")
            log(f"{'='*60}")
        else:
            log(f"Sample download failed. Continuing with other parts...")

        # ── FULL RUN: Process remaining zips ──
        for i, (fname, url) in enumerate(sorted_links[1:], start=2):
            s, n = get_wav_counts()
            if s >= args.max_per_class and n >= args.max_per_class:
                log(f"")
                log(f"TARGET REACHED: {s} scripted, {n} nonscripted. Stopping.")
                break

            log(f"")
            log(f"[{i}/{len(sorted_links)}] {fname}")
            log(f"  Progress: {s} scripted, {n} nonscripted (target: {args.max_per_class} each)")

            zip_path = zip_dir / fname
            if not zip_path.exists():
                log(f"  Downloading...", indent=0)
                try:
                    result = subprocess.run(
                        ["wget", "-O", str(zip_path), "-q", "--show-progress", url],
                        timeout=7200,
                    )
                    if result.returncode != 0:
                        log(f"  wget failed, trying curl...")
                        subprocess.run(
                            ["curl", "-L", "-o", str(zip_path), "--progress-bar", url],
                            timeout=7200,
                        )
                except subprocess.TimeoutExpired:
                    log(f"  Download timed out. Skipping.")
                    zip_path.unlink(missing_ok=True)
                    continue

            if not zip_path.exists() or zip_path.stat().st_size < 1000:
                log(f"  Download failed. Skipping.")
                zip_path.unlink(missing_ok=True)
                continue

            process_one_zip(zip_path, args.max_per_class, args.keep_zips, args.keep_mp4s)

        # Build final manifest
        build_ccv2_manifest()

        # Cleanup temp dirs
        for d in [MP4_TEMP, zip_dir]:
            if d.exists():
                for f in d.glob("*"):
                    f.unlink(missing_ok=True)
                try:
                    d.rmdir()
                except OSError:
                    pass

        return

    # No mode specified
    log(f"Specify one of: --links, --zip-dir, or --mp4-dir")
    log(f"")
    log(f"To download from Meta's CDN links:")
    log(f"  1. Save the download links to a text file (tab-separated: filename\\turl)")
    log(f"  2. Run: python download_casual_conversations.py --links ccv2_links.txt")


if __name__ == "__main__":
    main()
