#!/usr/bin/env python3
"""Upload trained models to Hugging Face Hub."""

from pathlib import Path
from huggingface_hub import HfApi, create_repo
import sys

# Model files to upload
MODEL_FILES = [
    "checkpoints/speech_classifier_quant.onnx",  # Quantized ONNX (122MB)
    "checkpoints/speech_classifier.onnx",        # Full ONNX (361MB)
    "checkpoints/wav2vec2_best.pt",              # PyTorch checkpoint (761MB)
    "checkpoints/xgboost_baseline.json",         # XGBoost model
    "checkpoints/xgboost_scaler.pkl",            # Feature scaler
    "checkpoints/wav2vec2_results.json",         # Training results
    "checkpoints/xgboost_results.json",
    "checkpoints/wav2vec2_history.json",
    "checkpoints/feature_importance.png",        # Visualizations
    "checkpoints/confusion_matrix.png",
    "configs/config.yaml",                       # Config
    "requirements.txt",                          # Dependencies
    "predict_cpu.py",                            # CPU inference script
]

def main():
    repo_name = "Pransfrance/speechproj-models"
    
    print(f"🚀 Uploading models to Hugging Face Hub: {repo_name}")
    print(f"📦 Total files: {len(MODEL_FILES)}")
    
    try:
        # Initialize API
        api = HfApi()
        
        # Create repo if it doesn't exist
        print("\n📝 Creating repository...")
        try:
            create_repo(repo_id=repo_name, repo_type="model", private=False, exist_ok=True)
            print(f"✅ Repo created/verified: {repo_name}")
        except Exception as e:
            print(f"⚠️  Repo status: {e}")
        
        # Upload each file
        print("\n📤 Uploading files...")
        for i, file_path in enumerate(MODEL_FILES, 1):
            p = Path(file_path)
            if not p.exists():
                print(f"⚠️  Skipping {file_path} (not found)")
                continue
            
            size_mb = p.stat().st_size / (1024**2)
            print(f"  [{i}/{len(MODEL_FILES)}] {file_path} ({size_mb:.1f}MB)...", end=" ")
            
            try:
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=p.name if "/" not in file_path else file_path,
                    repo_id=repo_name,
                    repo_type="model",
                )
                print("✅")
            except Exception as e:
                print(f"❌ {e}")
        
        print(f"\n✨ Done! Models available at: https://huggingface.co/{repo_name}")
        print(f"\n💻 Download on company laptop with:")
        print(f"   from huggingface_hub import hf_hub_download")
        print(f"   model = hf_hub_download('{repo_name}', 'speech_classifier_quant.onnx')")
        
    except ImportError:
        print("❌ huggingface_hub not installed.")
        print("   Run: pip install huggingface_hub")
        sys.exit(1)

if __name__ == "__main__":
    main()
