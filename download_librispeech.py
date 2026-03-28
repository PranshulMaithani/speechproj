"""
Download LibriSpeech clean-100 (READ speech, ~6.3GB)
====================================================
Has audio + ground truth transcripts. 100 hours of audiobook reading.

Usage:
    python download_librispeech.py

Downloads to: datasets/librispeech/
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path


URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
OUT_DIR = Path("datasets/librispeech")
TAR_PATH = OUT_DIR / "train-clean-100.tar.gz"


class DownloadProgressBar:
    def __init__(self):
        self.last_pct = -1

    def __call__(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = int(downloaded * 100 / total_size)
            mb_done = downloaded / 1e6
            mb_total = total_size / 1e6
            if pct != self.last_pct:
                bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                print(f"\r  [{bar}] {pct}%  {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)
                self.last_pct = pct


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done_marker = OUT_DIR / ".done"
    if done_marker.exists():
        print("LibriSpeech already downloaded. Delete datasets/librispeech/.done to re-download.")
        return

    # Download
    if not TAR_PATH.exists():
        print(f"Downloading LibriSpeech clean-100 (~6.3 GB)...")
        print(f"  URL: {URL}")
        print(f"  To:  {TAR_PATH}")
        urllib.request.urlretrieve(URL, str(TAR_PATH), DownloadProgressBar())
        print("\n  Download complete!")
    else:
        print(f"Tar already exists: {TAR_PATH} ({TAR_PATH.stat().st_size/1e9:.1f} GB)")

    # Extract
    print(f"\nExtracting (this takes a few minutes)...")
    with tarfile.open(str(TAR_PATH), "r:gz") as tar:
        members = tar.getmembers()
        for i, member in enumerate(members):
            tar.extract(member, str(OUT_DIR))
            if (i + 1) % 1000 == 0:
                print(f"  Extracted {i+1}/{len(members)} files...", flush=True)

    print(f"  Extracted {len(members)} files")

    # Build manifest from transcripts
    print("\nBuilding manifest...")
    import csv
    rows = []
    audio_root = OUT_DIR / "LibriSpeech" / "train-clean-100"

    for trans_file in sorted(audio_root.rglob("*.trans.txt")):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                # Find the FLAC file
                flac_path = trans_file.parent / f"{utt_id}.flac"
                if flac_path.exists():
                    rows.append({
                        "filepath": str(flac_path.resolve()),
                        "filename": flac_path.name,
                        "source": "librispeech",
                        "label": "read",
                        "label_int": 1,
                        "text": text,
                        "speaker_id": utt_id.split("-")[0],
                    })

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {len(rows)} utterances in manifest")
    print(f"  Manifest: {manifest_path}")

    done_marker.touch()

    # Optionally delete tar to save space
    print(f"\nTar file: {TAR_PATH} ({TAR_PATH.stat().st_size/1e9:.1f} GB)")
    print("  Delete it manually to save space: del datasets\\librispeech\\train-clean-100.tar.gz")
    print("\nDone!")


if __name__ == "__main__":
    main()
