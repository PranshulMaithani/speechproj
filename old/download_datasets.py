"""
Dataset Downloader — GigaSpeech (read + spontaneous)
=====================================================

Uses GigaSpeech source column to separate labels:
    source == 0  (audiobook)  ->  READ
    source == 1  (podcast)    ->  SPONTANEOUS
    source == 2  (YouTube)    ->  SKIPPED (mixed)

Output folders:
    data/read/          <- 1-minute audiobook clips
    data/spontaneous/   <- 1-minute podcast clips

Usage:
    python download_datasets.py --files-per-class 10     # test
    python download_datasets.py                           # default 500
    python download_datasets.py --files-per-class 1000   # large

Requirements:
    pip install datasets soundfile librosa tqdm numpy pandas
    huggingface-cli login                                 # required
    # Accept terms at: https://huggingface.co/datasets/speechcolab/gigaspeech
"""

import io
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# Config
# ============================================================

SR           = 16000
CLIP_SEC     = 60.0
CLIP_SAMPLES = int(CLIP_SEC * SR)   # 960000
MIN_RMS      = 0.002

DATA_ROOT = Path("data")
READ_DIR  = DATA_ROOT / "read"
SPONT_DIR = DATA_ROOT / "spontaneous"

SOURCE_AUDIOBOOK = 0
SOURCE_PODCAST   = 1


# ============================================================
# Audio utilities
# ============================================================

def load_raw_bytes(raw: bytes) -> np.ndarray:
    """Decode raw audio bytes to float32 mono at SR using soundfile + librosa."""
    try:
        audio, orig_sr = sf.read(io.BytesIO(raw))
    except Exception:
        # fallback for formats soundfile can't handle
        audio, orig_sr = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
        return audio.astype(np.float32)

    audio = np.array(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SR)
    return audio


def peak_norm(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        return audio / peak * 0.95
    return audio


def is_loud_enough(audio: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(audio ** 2))) >= MIN_RMS


def save_wav(audio: np.ndarray, path: Path):
    sf.write(str(path), audio.astype(np.float32), SR)


def load_manifest(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save_manifest(rows: list, path: Path):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


# ============================================================
# Accumulator
# ============================================================

class MinuteAccumulator:
    """Accumulates short clips into 60-second files."""

    def __init__(self, out_dir: Path, label: str, label_int: int,
                 source_name: str, target_count: int, existing_rows: list):
        self.out_dir      = out_dir
        self.label        = label
        self.label_int    = label_int
        self.source_name  = source_name
        self.target_count = target_count
        self.rows         = list(existing_rows)
        self.buffer       = np.array([], dtype=np.float32)
        out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def done(self) -> bool:
        return self.count >= self.target_count

    def add(self, chunk: np.ndarray):
        if self.done:
            return
        self.buffer = np.concatenate([self.buffer, chunk])
        while len(self.buffer) >= CLIP_SAMPLES and not self.done:
            clip        = self.buffer[:CLIP_SAMPLES]
            self.buffer = self.buffer[CLIP_SAMPLES:]
            if not is_loud_enough(clip):
                continue
            clip     = peak_norm(clip)
            fname    = f"{self.label}_{self.count:05d}.wav"
            out_path = self.out_dir / fname
            save_wav(clip, out_path)
            self.rows.append({
                "filename":  fname,
                "filepath":  str(out_path.absolute()),
                "label":     self.label,
                "label_int": self.label_int,
                "split":     "train",
                "duration":  CLIP_SEC,
                "source":    self.source_name,
            })

    def flush(self, manifest_path: Path):
        save_manifest(self.rows, manifest_path)


# ============================================================
# Download
# ============================================================

def download_gigaspeech(files_per_class: int):
    print("=" * 65)
    print("GigaSpeech — read (audiobook) + spontaneous (podcast)")
    print("=" * 65)
    print(f"Target:  {files_per_class} files x {CLIP_SEC:.0f}s per class")
    print(f"Output:  {READ_DIR}  and  {SPONT_DIR}")

    from datasets import load_dataset, Audio

    READ_DIR.mkdir(parents=True, exist_ok=True)
    SPONT_DIR.mkdir(parents=True, exist_ok=True)

    read_manifest  = READ_DIR  / "manifest.json"
    spont_manifest = SPONT_DIR / "manifest.json"

    read_existing  = load_manifest(read_manifest)
    spont_existing = load_manifest(spont_manifest)

    read_acc = MinuteAccumulator(
        READ_DIR, "read", 1,
        "gigaspeech_audiobook", files_per_class, read_existing,
    )
    spont_acc = MinuteAccumulator(
        SPONT_DIR, "spontaneous", 0,
        "gigaspeech_podcast", files_per_class, spont_existing,
    )

    if read_acc.done and spont_acc.done:
        print("Both classes already complete.")
        return read_acc.rows, spont_acc.rows

    if read_existing:
        print(f"Read:        resuming from {read_acc.count} files")
    if spont_existing:
        print(f"Spontaneous: resuming from {spont_acc.count} files")

    # xs = 10h total, s = 250h — auto-select based on target
    config = "xs" if files_per_class <= 80 else "s"
    size_map = {"xs": "10h", "s": "250h"}
    print(f"\nLoading GigaSpeech '{config}' config ({size_map.get(config, '?')})...")
    print("(decode=False — raw bytes, no torchcodec)")

    dataset = load_dataset(
        "speechcolab/gigaspeech",
        config,
        split="train",
        streaming=True,
        token=True,
    ).cast_column("audio", Audio(decode=False))

    t_start    = time.perf_counter()
    errors     = 0
    skipped_yt = 0

    read_pbar = tqdm(
        total=files_per_class, initial=read_acc.count,
        desc="Read (audiobook)  ",
        unit="file", position=0, dynamic_ncols=True,
    )
    spont_pbar = tqdm(
        total=files_per_class, initial=spont_acc.count,
        desc="Spont (podcast)   ",
        unit="file", position=1, dynamic_ncols=True,
    )

    prev_read  = read_acc.count
    prev_spont = spont_acc.count

    for sample in dataset:
        if read_acc.done and spont_acc.done:
            break

        try:
            source_id = sample.get("source", -1)

            if source_id == 2:
                skipped_yt += 1
                continue

            # Decode raw bytes — no torchcodec involved
            raw = sample["audio"].get("bytes")
            if not raw:
                continue
            audio = load_raw_bytes(raw)

            elapsed = max(time.perf_counter() - t_start, 0.01)

            if source_id == SOURCE_AUDIOBOOK and not read_acc.done:
                read_acc.add(audio)
                if read_acc.count > prev_read:
                    read_pbar.update(read_acc.count - prev_read)
                    prev_read = read_acc.count
                    read_pbar.set_postfix({
                        "saved": read_acc.count,
                        "buf":   f"{len(read_acc.buffer)/SR:.0f}s",
                        "rate":  f"{read_acc.count/elapsed:.2f}/s",
                    })

            elif source_id == SOURCE_PODCAST and not spont_acc.done:
                spont_acc.add(audio)
                if spont_acc.count > prev_spont:
                    spont_pbar.update(spont_acc.count - prev_spont)
                    prev_spont = spont_acc.count
                    spont_pbar.set_postfix({
                        "saved": spont_acc.count,
                        "buf":   f"{len(spont_acc.buffer)/SR:.0f}s",
                        "rate":  f"{spont_acc.count/elapsed:.2f}/s",
                    })

            total_new = (
                (read_acc.count - len(read_existing)) +
                (spont_acc.count - len(spont_existing))
            )
            if total_new > 0 and total_new % 100 == 0:
                read_acc.flush(read_manifest)
                spont_acc.flush(spont_manifest)

        except Exception as e:
            errors += 1
            read_pbar.set_postfix({"errors": errors, "last": str(e)[:25]})
            continue

    read_pbar.close()
    spont_pbar.close()

    read_acc.flush(read_manifest)
    spont_acc.flush(spont_manifest)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone ({elapsed/60:.1f} min):")
    print(f"  Read:        {read_acc.count} files")
    print(f"  Spontaneous: {spont_acc.count} files")
    print(f"  Errors:      {errors}")
    print(f"  YT skipped:  {skipped_yt}")

    if read_acc.count < files_per_class:
        print(f"\nWARNING: Only {read_acc.count}/{files_per_class} read files.")
        print("  Increase --files-per-class to trigger 's' config (250h).")
    if spont_acc.count < files_per_class:
        print(f"\nWARNING: Only {spont_acc.count}/{files_per_class} spontaneous files.")
        print("  Increase --files-per-class to trigger 's' config (250h).")

    return read_acc.rows, spont_acc.rows


# ============================================================
# Combined manifest
# ============================================================

def build_manifest(read_rows: list, spont_rows: list,
                   allstar_manifest: str = "outputs/manifest.csv"):
    import pandas as pd

    print("\n" + "=" * 65)
    print("Building combined manifest")
    print("=" * 65)

    frames = []

    allstar_path = Path(allstar_manifest)
    if allstar_path.exists():
        adf = pd.read_csv(allstar_path)
        frames.append(adf)
        print(f"AllStar:       {len(adf)} windows")
    else:
        print(f"WARNING: AllStar not found at {allstar_path}")

    if read_rows:
        rdf = pd.DataFrame(read_rows)
        frames.append(rdf)
        print(f"GS Read:       {len(rdf)} files")

    if spont_rows:
        sdf = pd.DataFrame(spont_rows)
        frames.append(sdf)
        print(f"GS Spont:      {len(sdf)} files")

    if not frames:
        print("Nothing to combine.")
        return

    combined = pd.concat(frames, ignore_index=True)
    for col in ["filepath", "filename", "label", "label_int", "split", "duration"]:
        if col not in combined.columns:
            combined[col] = ""

    out = Path("outputs/manifest_expanded.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    print(f"\n  Total: {len(combined)} rows")
    print(combined.groupby(["label", "split"]).size().to_string())
    print(f"\n  Saved: {out}")
    print()
    print("  Update configs/config.yaml:")
    print("    paths:")
    print('      manifest_csv: "outputs/manifest_expanded.csv"')
    return combined


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download 1-minute balanced read/spontaneous from GigaSpeech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GigaSpeech source mapping:
  audiobook (0) -> READ        -> data/read/
  podcast   (1) -> SPONTANEOUS -> data/spontaneous/
  YouTube   (2) -> SKIPPED     (mixed labels)

Config auto-selection:
  files_per_class <= 80  -> 'xs' config (10h)
  files_per_class >  80  -> 's'  config (250h)

Examples:
  python download_datasets.py --files-per-class 10
  python download_datasets.py --files-per-class 500
  python download_datasets.py --files-per-class 1000
        """,
    )
    parser.add_argument("--files-per-class", type=int, default=500,
                        help="1-minute files per class (default: 500)")
    parser.add_argument("--allstar-manifest", default="outputs/manifest.csv")
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("GigaSpeech Downloader — 1-minute balanced audio files")
    print("=" * 65)
    print(f"Files per class: {args.files_per_class}")
    print(f"Clip duration:   {CLIP_SEC:.0f}s  ({CLIP_SAMPLES} samples @ {SR}Hz)")
    print(f"Read:        {READ_DIR}")
    print(f"Spontaneous: {SPONT_DIR}")
    print()

    read_rows, spont_rows = download_gigaspeech(args.files_per_class)

    if not args.skip_manifest:
        build_manifest(read_rows, spont_rows, args.allstar_manifest)

    print("\n" + "=" * 65)
    print("Done.")
    print(f"  Read:        {len(read_rows)} files  ->  {READ_DIR}")
    print(f"  Spontaneous: {len(spont_rows)} files  ->  {SPONT_DIR}")
    print(f"  Balanced:    {'YES' if len(read_rows) == len(spont_rows) else 'NO'}")
    print("=" * 65)


if __name__ == "__main__":
    main()