"""Upload biased wav2vec2 model to Hugging Face Hub."""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_NAME = "Pransfrance/speechproj-models"

FILES = [
    "checkpoints_biased/biased_wav2vec2_quant.onnx",
    "checkpoints_biased/biased_wav2vec2_best.pt",
    "checkpoints_biased/biased_results.json",
    "checkpoints_biased/biased_history.json",
    "eval_biased.py",
]


def main():
    print(f"Uploading to: {REPO_NAME}")
    api = HfApi()

    try:
        create_repo(repo_id=REPO_NAME, repo_type="model", private=True, exist_ok=True)
        print("Repo ready")
    except Exception as e:
        print(f"Repo status: {e}")

    uploaded, skipped, failed = 0, 0, 0

    for i, file_path in enumerate(FILES, 1):
        p = Path(file_path)
        if not p.exists():
            print(f"  [{i}/{len(FILES)}] SKIP (not found): {file_path}")
            skipped += 1
            continue

        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"  [{i}/{len(FILES)}] {p.name} ({size_mb:.1f} MB)...", end=" ", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"biased/{p.name}",
                repo_id=REPO_NAME,
                repo_type="model",
            )
            print("done")
            uploaded += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nUploaded: {uploaded}, Skipped: {skipped}, Failed: {failed}")
    print(f"Models at: https://huggingface.co/{REPO_NAME}")


if __name__ == "__main__":
    main()
