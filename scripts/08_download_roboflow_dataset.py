"""
Script 08 — Download a full-scene Indian plate dataset from Roboflow Universe.

This gives you images of actual cars/roads where the plate is a small
object inside the frame — exactly what your model needs to learn.

Requirements:
    pip install roboflow

You need a free Roboflow account and API key:
    1. Sign up at https://roboflow.com (free)
    2. Go to https://app.roboflow.com — click your profile → API Keys
    3. Copy your Private API Key and paste it below (or set env var)

The downloaded dataset is already split into train/val/test with
YOLO-format bounding box labels. We merge it into data/processed/.
"""

import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# PASTE YOUR ROBOFLOW API KEY HERE  (or set env var ROBOFLOW_API_KEY)
# ---------------------------------------------------------------------------
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "PASTE_YOUR_KEY_HERE")

# Roboflow dataset details (public, free)
WORKSPACE   = "indian-license-plates"
PROJECT     = "indian-license-plate-detection"
VERSION     = 1          # bump if a newer version exists on Roboflow

DEST_DIR    = Path("data/processed")
BACKUP_DIR  = Path("data/processed_kaggle_backup")


def download_and_merge():
    if ROBOFLOW_API_KEY == "PASTE_YOUR_KEY_HERE":
        print("ERROR: Set your Roboflow API key in this script or via:")
        print("  set ROBOFLOW_API_KEY=your_key_here   (Windows)")
        print("  export ROBOFLOW_API_KEY=your_key_here  (Linux/Mac)")
        return

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow: pip install roboflow")
        return

    print("Downloading full-scene Indian plate dataset from Roboflow...")
    rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8", location="data/roboflow_fullscene")

    rf_dir = Path("data/roboflow_fullscene")
    print(f"Downloaded to: {rf_dir}")

    # Backup existing kaggle-based dataset
    if DEST_DIR.exists():
        print(f"Backing up existing dataset to {BACKUP_DIR}")
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(DEST_DIR, BACKUP_DIR)

    # Merge Roboflow into processed/
    for split in ("train", "valid", "test"):
        src_split = "val" if split == "valid" else split
        rf_imgs   = rf_dir / split / "images"
        rf_lbls   = rf_dir / split / "labels"

        if not rf_imgs.exists():
            continue

        dest_imgs = DEST_DIR / "images" / src_split
        dest_lbls = DEST_DIR / "labels" / src_split
        dest_imgs.mkdir(parents=True, exist_ok=True)
        dest_lbls.mkdir(parents=True, exist_ok=True)

        imgs_copied = 0
        for p in rf_imgs.glob("*"):
            shutil.copy2(p, dest_imgs / p.name)
            imgs_copied += 1
        for p in rf_lbls.glob("*.txt"):
            shutil.copy2(p, dest_lbls / p.name)

        print(f"  {split}: {imgs_copied} images merged → {dest_imgs}")

    print("\nDataset merged. Run: python scripts/03_train.py")


if __name__ == "__main__":
    download_and_merge()
