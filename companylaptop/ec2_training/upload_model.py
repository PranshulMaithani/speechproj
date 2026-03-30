"""
Upload trained model to HuggingFace.
====================================
Usage:
    python upload_model.py
    python upload_model.py --prefix combined_v2
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Pransfrance/speechproj-models"


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--prefix", type=str, default="combined_v2",
                        help="Remote prefix in HF repo")
    args = parser.parse_args()

    api = HfApi()
    ckpt_dir = Path(args.checkpoint_dir)

    files = [
        ("wav2vec2_best.pt", f"{args.prefix}/wav2vec2_combined_best.pt"),
        ("wav2vec2_trained_quant.onnx", f"{args.prefix}/wav2vec2_combined_quant.onnx"),
        ("wav2vec2_trained.onnx", f"{args.prefix}/wav2vec2_combined.onnx"),
        ("results.json", f"{args.prefix}/results.json"),
        ("history.json", f"{args.prefix}/history.json"),
    ]

    for local_name, remote_path in files:
        local_path = ckpt_dir / local_name
        if local_path.exists():
            size_mb = local_path.stat().st_size / 1e6
            print(f"Uploading {local_name} ({size_mb:.1f} MB) -> {remote_path}")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  Done")
        else:
            print(f"  Skipping {local_name} (not found)")

    print(f"\nAll uploads complete -> https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
