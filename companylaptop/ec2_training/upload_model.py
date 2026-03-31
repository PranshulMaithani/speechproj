"""
Upload trained model to S3 bucket.
===================================
Usage:
    python upload_model.py --bucket your-bucket-name
    python upload_model.py --bucket your-bucket-name --prefix combined_v2
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload model to S3")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--prefix", type=str, default="combined_v2",
                        help="S3 key prefix (folder in bucket)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    s3_base = f"s3://{args.bucket}/{args.prefix}"

    files = [
        "wav2vec2_best.pt",
        "wav2vec2_trained.onnx",
        "wav2vec2_trained_quant.onnx",
        "results.json",
        "history.json",
    ]

    for fname in files:
        local_path = ckpt_dir / fname
        if local_path.exists():
            s3_path = f"{s3_base}/{fname}"
            size_mb = local_path.stat().st_size / 1e6
            print(f"Uploading {fname} ({size_mb:.1f} MB) -> {s3_path}")
            result = subprocess.run(
                ["aws", "s3", "cp", str(local_path), s3_path],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"  Done")
            else:
                print(f"  ERROR: {result.stderr.strip()}")
        else:
            print(f"  Skipping {fname} (not found)")

    print(f"\nAll uploads complete -> {s3_base}/")
    print(f"\nTo download from another machine:")
    print(f"  aws s3 cp {s3_base}/wav2vec2_trained_quant.onnx .")


if __name__ == "__main__":
    main()
