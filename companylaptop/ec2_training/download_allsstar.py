"""
Download ALLSSTAR corpus from HuggingFace.
===========================================
Expects the dataset uploaded to: Pransfrance/speechproj-allsstar

Usage:
    python download_allsstar.py
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "Pransfrance/speechproj-allsstar"
_SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = _SCRIPT_DIR / "data" / "allsstar"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ALLSSTAR from {REPO_ID}...")
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(OUT_DIR),
    )
    print(f"Downloaded to: {local_path}")

    # Fix manifest filepaths to point to local paths
    manifest_path = OUT_DIR / "manifest.csv"
    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)

        # Original paths are Windows absolute — remap to local
        # Original: D:\GoodProjects\Speech project\2676\ALL_CCT_ENG_QNA\file.wav
        # Local:    data/allsstar/allsstar/2676/ALL_CCT_ENG_QNA/file.wav
        def remap_path(fp):
            fp = str(fp).replace("\\", "/")
            # Extract from the 2676/ or 2677/ part onwards
            for marker in ["2676/", "2677/"]:
                idx = fp.find(marker)
                if idx >= 0:
                    return str(OUT_DIR / "allsstar" / fp[idx:])
            return fp

        df["filepath"] = df["filepath"].apply(remap_path)
        # Filter to files that exist
        exists_mask = df["filepath"].apply(os.path.exists)
        print(f"\nFiles found on disk: {exists_mask.sum()} / {len(df)}")
        df = df[exists_mask].reset_index(drop=True)
        df.to_csv(OUT_DIR / "manifest_local.csv", index=False)

        print(f"ALLSSTAR: {len(df)} files")
        print(f"  Read:        {(df['label_int']==1).sum()}")
        print(f"  Spontaneous: {(df['label_int']==0).sum()}")
        print(f"  Duration:    {df['duration_sec'].sum()/3600:.1f} hours")
        print(f"\nSaved: {OUT_DIR / 'manifest_local.csv'}")
    else:
        print(f"WARNING: manifest.csv not found in {OUT_DIR}")
        print("Make sure you uploaded it: huggingface-cli upload ... manifest_expanded.csv manifest.csv")


if __name__ == "__main__":
    main()
