"""
Upload ALLSSTAR corpus to HuggingFace for EC2 download.
========================================================
Uploads the audio folders (2676/, 2677/) and manifest.

Usage:
    python upload_allsstar.py
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "Pransfrance/speechproj-allsstar"
REPO_TYPE = "dataset"


def main():
    api = HfApi()

    # Create dataset repo if it doesn't exist
    try:
        create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
        print(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload manifest
    manifest = Path("old/outputs/manifest_expanded.csv")
    if manifest.exists():
        print(f"\nUploading manifest ({manifest.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(manifest),
            path_in_repo="manifest.csv",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print("  Done: manifest.csv")
    else:
        print(f"WARNING: {manifest} not found")

    # Upload audio folders
    for folder in ["2676", "2677"]:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"WARNING: {folder}/ not found, skipping")
            continue

        size_gb = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file()) / 1e9
        n_files = sum(1 for f in folder_path.rglob("*.wav"))
        print(f"\nUploading {folder}/ ({n_files} wav files, {size_gb:.1f} GB)...")
        print("  This may take a while...")

        api.upload_folder(
            folder_path=str(folder_path),
            path_in_repo=f"allsstar/{folder}",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"  Done: allsstar/{folder}/")

    print(f"\nAll uploads complete!")
    print(f"  https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
