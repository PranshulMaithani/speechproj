"""Upload local ALLSTAR folders (default: 2676, 2677) to a HF dataset repo
so they can be pulled onto the company laptop with the regular HF CLI.

This script runs on whichever machine has the raw ALLSTAR audio (not the
company laptop). The data is public-corpus speech, so there's no PII concern
about uploading -- but the repo is created private by default so only your
HF account can pull it. Use --public to override.

One-time setup on the source machine:
    pip install huggingface_hub
    huggingface-cli login

Upload:
    python upload_to_hf.py \\
        --local_root /path/to/allstar_root \\
        --repo_id  <your-hf-username>/allstar-2676-2677

The script preserves the 2676/ and 2677/ subfolder structure inside the repo.

Pulling on the company laptop:
    pip install huggingface_hub
    huggingface-cli login   (once)
    huggingface-cli download <your-hf-username>/allstar-2676-2677 \\
        --repo-type dataset --local-dir companylaptop/

The downloaded folders should land alongside audios2..audios6 in
companylaptop/, so the existing neural_baseline_prep.py picks them up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_root", required=True,
                    help="local folder containing the ALLSTAR subfolders")
    ap.add_argument("--repo_id", required=True,
                    help="HF dataset repo id (e.g. 'your-username/allstar-2676-2677')")
    ap.add_argument("--folders", default="2676,2677",
                    help="comma-separated subfolder names to upload")
    ap.add_argument("--public", action="store_true",
                    help="create a public repo (default is private)")
    ap.add_argument("--commit_message", default="upload ALLSTAR folders")
    ap.add_argument("--allow_patterns", default="*.wav,*.mp3,*.flac,*.m4a,*.ogg",
                    help="comma-separated glob patterns to include")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Install first:  pip install huggingface_hub", file=sys.stderr)
        return 1

    root = Path(args.local_root).resolve()
    if not root.exists():
        print(f"local_root not found: {root}", file=sys.stderr)
        return 1

    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    allow_patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]
    private = not args.public

    print(f"creating repo  {args.repo_id}  (private={private})")
    create_repo(args.repo_id, repo_type="dataset", private=private, exist_ok=True)

    api = HfApi()
    for folder in folders:
        src = root / folder
        if not src.exists():
            print(f"skipping missing folder: {src}", file=sys.stderr)
            continue
        n_files = sum(1 for _ in src.rglob("*") if _.is_file())
        print(f"uploading  {src}  -> {args.repo_id}/{folder}/   ({n_files} files)")
        api.upload_folder(
            folder_path=str(src),
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo=folder,
            commit_message=f"{args.commit_message} ({folder})",
            allow_patterns=allow_patterns,
        )

    print()
    print("Done. On the company laptop:")
    print(f"  huggingface-cli login")
    print(f"  huggingface-cli download {args.repo_id} \\")
    print(f"      --repo-type dataset --local-dir companylaptop/")
    print()
    print("Confirm 2676/ and 2677/ now sit alongside audios2..audios6 under companylaptop/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
