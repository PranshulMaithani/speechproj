"""
Download AMI Meeting Corpus (SPONTANEOUS speech)
=================================================
Has audio + ground truth transcripts. ~100 hours of meeting conversations.

Uses HuggingFace datasets (streams + saves to disk with progress).

Usage:
    python download_ami.py

Downloads to: datasets/ami/
"""

import os
import csv
import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio
from tqdm import tqdm

OUT_DIR = Path("datasets/ami")
SR = 16000
AMI_CONFIG = "ihm"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = OUT_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    done_marker = OUT_DIR / ".done"
    if done_marker.exists():
        print("AMI already downloaded. Delete datasets/ami/.done to re-download.")
        return

    print("Downloading AMI Meeting Corpus via HuggingFace...")
    print("  This downloads + processes ~7GB. Progress bar below.\n")

    ds = load_dataset("edinburghcstr/ami", AMI_CONFIG, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))

    print(f"\nProcessing {len(ds)} AMI utterances...")

    rows = []
    skipped = 0

    for i in tqdm(range(len(ds)), desc="AMI"):
        item = ds[i]
        audio = item["audio"]["array"]
        duration = len(audio) / SR

        # Skip very short (<2s) or very long (>60s)
        if duration < 2.0 or duration > 60.0:
            skipped += 1
            continue

        fname = f"ami_{i:06d}.wav"
        fpath = audio_dir / fname

        if not fpath.exists():
            sf.write(str(fpath), audio.astype(np.float32), SR)

        rows.append({
            "filepath": str(fpath.resolve()),
            "filename": fname,
            "source": "ami",
            "label": "spontaneous",
            "label_int": 0,
            "duration_sec": round(duration, 2),
            "text": item.get("text", ""),
            "speaker_id": str(item.get("speaker_id", "")),
        })

    manifest_path = OUT_DIR / "manifest.csv"
    fieldnames = [
        "filepath",
        "filename",
        "source",
        "label",
        "label_int",
        "duration_sec",
        "text",
        "speaker_id",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    done_marker.touch()
    print(f"\n  Saved: {len(rows)} utterances ({skipped} skipped)")
    print(f"  Manifest: {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
