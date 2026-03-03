#!/usr/bin/env python3
"""Download pre-trained models from HuggingFace Hub to local machine."""

from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import os

def download_models(cache_dir="./models"):
    """Download all models from HuggingFace Hub."""
    
    repo_id = "Pransfrance/speechproj-models"
    
    print(f"📥 Downloading models from {repo_id}...")
    print(f"💾 Cache directory: {cache_dir}\n")
    
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # List of essential files to download
    files = [
        "speech_classifier_quant.onnx",  # Quantized ONNX (122MB) - USE THIS for CPU
        "wav2vec2_best.pt",               # PyTorch (761MB)
        "xgboost_baseline.json",          # XGBoost model
        "xgboost_scaler.pkl",             # Feature scaler
        "config.yaml",                    # Config file
    ]
    
    print("Files to download:")
    for f in files:
        print(f"  ✓ {f}")
    print()
    
    # Download each file
    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file}...", end=" ", flush=True)
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                cache_dir=cache_dir,
                force_download=False,  # Only download if not cached
            )
            size_mb = Path(path).stat().st_size / (1024**2)
            print(f"✅ ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"⚠️  {e}")
    
    print("\n" + "="*60)
    print(f"✨ Models cached in: {Path(cache_dir).absolute()}")
    print("="*60)
    print("\nUsage in your inference script:")
    print("```python")
    print("from huggingface_hub import hf_hub_download")
    print()
    print("# Download on first run, cached after that")
    print("model_path = hf_hub_download('Pransfrance/speechproj-models', 'speech_classifier_quant.onnx')")
    print("scaler_path = hf_hub_download('Pransfrance/speechproj-models', 'xgboost_scaler.pkl')")
    print("```")

if __name__ == "__main__":
    try:
        download_models(cache_dir="./models_cache")
    except ImportError:
        print("❌ huggingface_hub not installed.")
        print("   Run: pip install huggingface_hub")
