"""
Script 06 — Offline dataset augmentation.

Generates N augmented copies of every training image, targeting Indian
road-specific conditions: night, IR cameras, rain, motion blur, perspective.

Augmented images are written directly into the existing train split so
YOLO picks them up on the next training run without any config changes.

Usage:
  python scripts/06_augment_dataset.py
  python scripts/06_augment_dataset.py --multiplier 5 --mode heavy
  python scripts/06_augment_dataset.py --multiplier 3 --mode night
"""

import argparse
from pathlib import Path

from utils.augment import generate_augmented_dataset

TRAIN_IMG_DIR = "data/processed/images/train"
TRAIN_LBL_DIR = "data/processed/labels/train"


def augment(multiplier: int = 3, mode: str = "standard"):
    src_count = len(list(Path(TRAIN_IMG_DIR).glob("*.jpg"))) + \
                len(list(Path(TRAIN_IMG_DIR).glob("*.png")))

    print(f"Source training images: {src_count}")
    print(f"Mode: {mode}, Multiplier: {multiplier}")
    print(f"Expected output: {src_count * (1 + multiplier)} images")

    n = generate_augmented_dataset(
        src_img_dir=TRAIN_IMG_DIR,
        src_lbl_dir=TRAIN_LBL_DIR,
        out_img_dir=TRAIN_IMG_DIR,       # Write back to same split
        out_lbl_dir=TRAIN_LBL_DIR,
        multiplier=multiplier,
        mode=mode,
    )

    new_count = len(list(Path(TRAIN_IMG_DIR).glob("*.jpg"))) + \
                len(list(Path(TRAIN_IMG_DIR).glob("*.png")))

    print(f"\nDone. Training set: {src_count} → {new_count} images (+{n} augmented)")
    print("Next: re-run python scripts/03_train.py to train with augmented data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplier", type=int, default=3,
                        help="Augmented copies per original (default: 3)")
    parser.add_argument("--mode", default="standard",
                        choices=["standard", "night", "ir", "heavy"],
                        help="Augmentation intensity preset")
    args = parser.parse_args()
    augment(args.multiplier, args.mode)
