"""
check_and_balance_manifest.py
==============================
1. Fixes missing AllStar durations by reading actual audio files
2. Shows window counts per class
3. Tells you exactly how many GigaSpeech files to download to balance

Run:
    python check_and_balance_manifest.py
"""

import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


MANIFEST_PATH = "outputs/manifest_expanded.csv"
SR            = 16000
WINDOW_SEC    = 5.0
HOP_SEC       = 2.5


# ============================================================
# Step 1 — Load manifest
# ============================================================

print("=" * 65)
print("Loading manifest")
print("=" * 65)

df = pd.read_csv(MANIFEST_PATH)
print(f"Total rows: {len(df)}")
print(f"Splits:     {df['split'].value_counts().to_dict()}")
print()


# ============================================================
# Step 2 — Fix missing durations for AllStar rows
# ============================================================

is_gs      = df["source"].fillna("").str.startswith("gigaspeech")
nan_dur    = df["duration"].isna() | (df["duration"] == 0)
needs_fix  = ~is_gs & nan_dur

print(f"Rows with missing duration: {needs_fix.sum()}")
print(f"  AllStar missing:     {needs_fix.sum()}")
print(f"  GigaSpeech missing:  {(is_gs & nan_dur).sum()}")
print()

if needs_fix.sum() > 0:
    print("Computing AllStar durations from audio files...")
    fixed   = 0
    failed  = 0
    indices = df[needs_fix].index

    for idx in tqdm(indices, desc="Reading durations"):
        fpath = str(df.loc[idx, "filepath"])
        try:
            dur = librosa.get_duration(path=fpath)
            df.loc[idx, "duration"] = round(dur, 2)
            fixed += 1
        except Exception as e:
            # fallback: one 5-second window
            df.loc[idx, "duration"] = WINDOW_SEC
            failed += 1

    print(f"  Fixed:  {fixed}")
    print(f"  Failed (set to 5s): {failed}")
    print()

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest saved: {MANIFEST_PATH}")
    print()
else:
    print("All durations present — no fix needed.")
    print()


# ============================================================
# Step 3 — Count windows per class
# ============================================================

def estimate_windows(d):
    try:
        d = float(d)
        if np.isnan(d) or d <= 0:
            return 0
        if d <= WINDOW_SEC:
            return 1
        return max(1, int((d - WINDOW_SEC) / HOP_SEC) + 1)
    except Exception:
        return 0


print("=" * 65)
print("Window counts (train split)")
print("=" * 65)

train  = df[df["split"] == "train"].copy()
is_gs  = train["source"].fillna("").str.startswith("gigaspeech")

train["n_windows"] = train["duration"].apply(estimate_windows)

allstar = train[~is_gs]
gs      = train[is_gs]


def section_summary(name: str, subset: pd.DataFrame):
    r = subset[subset["label"] == "read"]["n_windows"].sum()
    s = subset[subset["label"] == "spontaneous"]["n_windows"].sum()
    files_r = (subset["label"] == "read").sum()
    files_s = (subset["label"] == "spontaneous").sum()
    print(f"{name}:")
    print(f"  Read:        {r:>8,} windows  ({files_r} files)")
    print(f"  Spontaneous: {s:>8,} windows  ({files_s} files)")
    print(f"  Ratio:       {max(r,s)/max(min(r,s),1):.2f}x imbalance")
    print()
    return r, s


ar, as_ = section_summary("AllStar train", allstar)
gr, gs_ = section_summary("GigaSpeech train", gs)

total_r = ar + gr
total_s = as_ + gs_

print("=" * 65)
print("Combined train totals")
print("=" * 65)
print(f"  Read:        {total_r:>8,} windows")
print(f"  Spontaneous: {total_s:>8,} windows")
print()


# ============================================================
# Step 4 — How many GigaSpeech files to add
# ============================================================

target          = min(total_r, total_s)
ws_per_gs_file  = estimate_windows(60.0)   # 1-minute GS files → ~23 windows

gs_r_needed     = max(0, target - ar)
gs_s_needed     = max(0, target - as_)

gs_r_files      = int(np.ceil(gs_r_needed  / ws_per_gs_file)) if ws_per_gs_file > 0 else 0
gs_s_files      = int(np.ceil(gs_s_needed  / ws_per_gs_file)) if ws_per_gs_file > 0 else 0
files_needed    = max(gs_r_files, gs_s_files)

minority        = "read" if total_r < total_s else "spontaneous"

print("=" * 65)
print("Balance recommendation")
print("=" * 65)
print(f"  Target (minority class): {target:,} windows")
print(f"  Windows per 60s GS file: {ws_per_gs_file}")
print()
print(f"  GS read files needed:    {gs_r_files}")
print(f"  GS spont files needed:   {gs_s_files}")
print()

if files_needed == 0:
    print("Already balanced! No extra GigaSpeech files needed.")
else:
    print(f"  Run:")
    print(f"    python download_datasets.py --files-per-class {files_needed}")
    print()
    print(f"  Then retrain with the updated manifest.")
    print()
    print("  Note: if files_per_class > 80 the downloader auto-upgrades")
    print("  to the 250h GigaSpeech 's' config which has plenty of data.")


# ============================================================
# Step 5 — Full split breakdown
# ============================================================

print()
print("=" * 65)
print("Full manifest breakdown")
print("=" * 65)
df["n_windows"] = df["duration"].apply(estimate_windows)
breakdown = df.groupby(["label", "split"])["n_windows"].agg(
    files="count", windows="sum"
).reset_index()
print(breakdown.to_string(index=False))
print()
print(f"Total windows: {df['n_windows'].sum():,}")