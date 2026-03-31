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
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SR = 16000
OUT_DIR = Path("data/casual_conversations")
AUDIO_DIR = OUT_DIR / "audio_scripted"
AUDIO_DIR_UNSCRIPTED = OUT_DIR / "audio_nonscripted"
MP4_TEMP = OUT_DIR / "mp4_temp"

# Pattern to match English files
ENGLISH_PATTERN = re.compile(r"(\d+)_english_(scripted|nonscripted)_(\d+)\.mp4$", re.IGNORECASE)


def parse_links_file(links_path):
    """Parse the CCv2 download links file.

    Format (tab-separated):
        file_name    cdn_link
        CCv2_part_1.zip    https://scontent.xx.fbcdn.net/...
    """
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
                # Accept CCv2_part_*.zip and CCv2_frames_part_*.zip
                if (fname.startswith("CCv2_part_") or fname.startswith("CCv2_frames_part_")) and fname.endswith(".zip"):
                    links[fname] = url
    return links


def extract_english_from_zip(zip_path, mp4_out_dir):
    """Extract only English scripted/nonscripted MP4s from a zip."""
    mp4_out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List all files and filter English ones
            all_names = zf.namelist()
            english_files = [n for n in all_names if ENGLISH_PATTERN.search(n.split("/")[-1])]

            if not english_files:
                print(f"  No English files in {zip_path.name}")
                return extracted

            print(f"  Found {len(english_files)} English files in {zip_path.name}")

            for name in tqdm(english_files, desc=f"  Extracting from {zip_path.name}"):
                # Extract to flat directory (ignore zip folder structure)
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
                    print(f"    Error extracting {name}: {e}")

    except zipfile.BadZipFile:
        print(f"  ERROR: {zip_path.name} is corrupted. Re-download it.")
    except Exception as e:
        print(f"  ERROR processing {zip_path.name}: {e}")

    return extracted


def mp4_to_wav(mp4_path, wav_path):
    """Extract audio from MP4 to 16kHz mono WAV using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-i", str(mp4_path),
            "-vn",                      # no video
            "-acodec", "pcm_s16le",     # 16-bit PCM
            "-ar", str(SR),             # 16kHz
            "-ac", "1",                 # mono
            "-y",                       # overwrite
            "-loglevel", "error",
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
    """Convert English MP4s to WAV. Does NOT build manifest (that's separate)."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR_UNSCRIPTED.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(Path(mp4_dir).glob("*.mp4"))
    if not mp4_files:
        return
    print(f"  Converting {len(mp4_files)} MP4 files to WAV...")

    for mp4 in tqdm(mp4_files, desc="  MP4 -> WAV"):
        match = ENGLISH_PATTERN.match(mp4.name)
        if not match:
            continue

        script_type = match.group(2)

        # Check max per class (count existing WAVs on disk)
        if max_per_class:
            if script_type == "scripted" and len(list(AUDIO_DIR.glob("*.wav"))) >= max_per_class:
                continue
            if script_type == "nonscripted" and len(list(AUDIO_DIR_UNSCRIPTED.glob("*.wav"))) >= max_per_class:
                continue

        wav_dir = AUDIO_DIR if script_type == "scripted" else AUDIO_DIR_UNSCRIPTED
        wav_path = wav_dir / (mp4.stem + ".wav")

        if not wav_path.exists():
            if not mp4_to_wav(mp4, wav_path):
                continue

        # Check duration — remove bad files
        duration = get_duration(wav_path)
        if duration < 3.0 or duration > 120.0:
            wav_path.unlink(missing_ok=True)


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
    print(f"\nCasual Conversations manifest: {manifest_path}")
    print(f"  Total:       {len(df)}")
    print(f"  Scripted:    {scripted_count} (label=read)")
    print(f"  Nonscripted: {nonscripted_count} (label=spontaneous)")
    if "duration_sec" in df.columns and len(df) > 0:
        print(f"  Duration:    {df['duration_sec'].sum()/3600:.1f} hours")
    return df


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

    # Check if already processed
    manifest_path = OUT_DIR / "manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        existing = df["filepath"].apply(os.path.exists).sum()
        if existing > 100:
            print(f"Casual Conversations already processed ({existing} files).")
            print(f"  Scripted:    {(df['label_int']==1).sum()}")
            print(f"  Nonscripted: {(df['label_int']==0).sum()}")
            print("Delete manifest.csv to re-process.")
            return

    # Mode 1: Process already-extracted MP4s
    if args.mp4_dir:
        convert_mp4s(args.mp4_dir, args.max_per_class)
        build_ccv2_manifest()
        return

    # Mode 2: Process already-downloaded zips
    if args.zip_dir:
        zip_files = sorted(list(Path(args.zip_dir).glob("CCv2_part_*.zip")) +
                           list(Path(args.zip_dir).glob("CCv2_frames_part_*.zip")))
        print(f"Found {len(zip_files)} zip files in {args.zip_dir}")
        for zf in zip_files:
            extract_english_from_zip(zf, MP4_TEMP)
            convert_mp4s(MP4_TEMP, args.max_per_class)
            if not args.keep_mp4s:
                for f in MP4_TEMP.glob("*.mp4"):
                    f.unlink()
        build_ccv2_manifest()
        return

    # Mode 3: Download from links file
    if args.links:
        links = parse_links_file(args.links)
        if not links:
            print(f"No CCv2_part_*.zip links found in {args.links}")
            print("Expected format (tab-separated):")
            print("  CCv2_part_1.zip\\thttps://scontent.xx.fbcdn.net/...")
            return

        print(f"Found {len(links)} CCv2 part links")

        # Sort by part number for orderly downloading
        def part_num(name):
            m = re.search(r"part_(\d+)", name)
            return int(m.group(1)) if m else 0
        sorted_links = sorted(links.items(), key=lambda x: part_num(x[0]))

        if args.max_parts:
            sorted_links = sorted_links[:args.max_parts]
            print(f"Limiting to {args.max_parts} parts")

        zip_dir = OUT_DIR / "zips"
        zip_dir.mkdir(exist_ok=True)

        # Check how many English files we already have
        existing_mp4s = list(MP4_TEMP.glob("*_english_*.mp4"))
        print(f"Already extracted: {len(existing_mp4s)} English MP4s")

        for i, (fname, url) in enumerate(sorted_links):
            # Check if we have enough WAV files already
            scripted = len(list(AUDIO_DIR.glob("*.wav")))
            nonscripted = len(list(AUDIO_DIR_UNSCRIPTED.glob("*.wav")))
            if scripted >= args.max_per_class and nonscripted >= args.max_per_class:
                print(f"\nReached target: {scripted} scripted, {nonscripted} nonscripted. Stopping downloads.")
                break

            zip_path = zip_dir / fname
            print(f"\n[{i+1}/{len(sorted_links)}] {fname}")
            print(f"  Current: {scripted} scripted, {nonscripted} nonscripted")

            # Download
            if not zip_path.exists():
                print(f"  Downloading...")
                try:
                    result = subprocess.run(
                        ["wget", "-O", str(zip_path), "-q", "--show-progress", url],
                        timeout=7200,  # 2 hour timeout per file
                    )
                    if result.returncode != 0:
                        print(f"  wget failed, trying curl...")
                        subprocess.run(
                            ["curl", "-L", "-o", str(zip_path), "--progress-bar", url],
                            timeout=7200,
                        )
                except subprocess.TimeoutExpired:
                    print(f"  Download timed out. Skipping.")
                    zip_path.unlink(missing_ok=True)
                    continue

            if not zip_path.exists() or zip_path.stat().st_size < 1000:
                print(f"  Download failed. Skipping.")
                zip_path.unlink(missing_ok=True)
                continue

            # Extract English MP4s
            extract_english_from_zip(zip_path, MP4_TEMP)

            # Delete zip IMMEDIATELY to save disk space
            if not args.keep_zips:
                print(f"  Deleting {fname} to save space...")
                zip_path.unlink(missing_ok=True)

            # Convert MP4s to WAV IMMEDIATELY (don't accumulate)
            convert_mp4s(MP4_TEMP, args.max_per_class)

            # Delete MP4s IMMEDIATELY after conversion
            if not args.keep_mp4s:
                for f in MP4_TEMP.glob("*.mp4"):
                    f.unlink()
                print(f"  Cleaned up MP4 temp files")

        # Build final manifest from all WAVs on disk
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
    print("Specify one of: --links, --zip-dir, or --mp4-dir")
    print("\nTo download from Meta's CDN links:")
    print("  1. Save the download links to a text file (tab-separated: filename\\turl)")
    print("  2. Run: python download_casual_conversations.py --links ccv2_links.txt")
    print("\nExample links file format:")
    print("  CCv2_part_1.zip\\thttps://scontent.xx.fbcdn.net/...")
    print("  CCv2_part_2.zip\\thttps://scontent.xx.fbcdn.net/...")


if __name__ == "__main__":
    main()
