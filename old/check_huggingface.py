#!/usr/bin/env python3
"""Check what files are available on HuggingFace Hub."""

from huggingface_hub import list_repo_files

repo_id = "Pransfrance/speechproj-models"

print(f"Files in {repo_id}:\n")

try:
    files = list_repo_files(repo_id=repo_id)
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")
    print(f"\nTotal: {len(files)} files")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure huggingface_hub is installed:")
    print("  pip install huggingface_hub")
