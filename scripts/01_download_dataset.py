"""
Script 01 — Download the Indian number plate dataset from Kaggle.

Dataset: tkm22092/indian-number-plate-images
Source:  https://www.kaggle.com/datasets/tkm22092/indian-number-plate-images

The dataset contains ~800 JPG images of Indian number plates
(various states, lighting conditions, and vehicle types).
"""

import os
import shutil
from pathlib import Path

import kagglehub

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_SLUG = "tkm22092/indian-number-plate-images"
RAW_DIR = Path("data/raw")


def download():
    print(f"Downloading dataset: {DATASET_SLUG}")
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Kaggle cache path: {path}")

    # Copy into our project's data/raw directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(path)

    # Find image files recursively
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    print(f"Found {len(images)} images")

    for img_path in images:
        dest = RAW_DIR / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)

    final_count = len(list(RAW_DIR.glob("*")))
    print(f"Copied {final_count} images to {RAW_DIR}")
    return RAW_DIR


if __name__ == "__main__":
    raw_dir = download()
    print(f"\nDataset ready at: {raw_dir.resolve()}")
    print("Next: run scripts/02_prepare_dataset.py")
