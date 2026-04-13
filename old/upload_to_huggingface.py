#!/usr/bin/env python3
"""Upload all models to Hugging Face Hub."""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

MODEL_FILES = [
    # Wav2Vec2 5sec — original production model
    "checkpoints/speech_classifier_quant.onnx",
    # Wav2Vec2 5sec — expanded data (AllStar + GigaSpeech)
    "checkpoints/speech_classifier_wav2vec2_5sec_quantexpanded_data.onnx",
    # WavLM 5sec
    "checkpoints/speech_classifier_wavlm_5sec_quant.onnx",
    # Whisper Medium — use this for inference
    "checkpoints/speech_classifier_whisper_medium_quant.onnx",
    # Whisper Medium fp32 — optional, large
    # "checkpoints/speech_classifier_whisper_medium.onnx",
    # PyTorch checkpoints
    "checkpoints/wavlm_5sec_best.pt",
    "checkpoints/whisper_medium_best.pt",
    "checkpoints/wav2vec2_5secExpanded.pt",
    # Results
    "checkpoints/wavlm_5sec_results.json",
    "checkpoints/whisper_medium_results.json",
    "checkpoints/whisper_medium_history.json",
    # Config + inference script
    "configs/config.yaml",
    "predict_cpu.py",
]

def main():
    repo_name = "Pransfrance/speechproj-models"
    print(f"Uploading to: {repo_name}")

    api = HfApi()

    try:
        create_repo(repo_id=repo_name, repo_type="model", private=True, exist_ok=True)
        print("Repo created/verified")
    except Exception as e:
        print(f"Repo status: {e}")

    total    = len(MODEL_FILES)
    uploaded = 0
    skipped  = 0
    failed   = 0

    for i, file_path in enumerate(MODEL_FILES, 1):
        p = Path(file_path)
        if not p.exists():
            print(f"  [{i}/{total}] SKIPPED (not found): {file_path}")
            skipped += 1
            continue

        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"  [{i}/{total}] {p.name} ({size_mb:.1f} MB)...", end=" ", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.name,
                repo_id=repo_name,
                repo_type="model",
            )
            print("done")
            uploaded += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Done.")
    print(f"  Uploaded: {uploaded}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")
    print(f"\nModels at: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    main()