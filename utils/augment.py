"""
utils/augment.py — Custom augmentation pipeline for Indian number plates.

Covers conditions specific to Indian roads:
  - Night / low-light (aggressive brightness drop)
  - IR camera output (plate appears very bright/white)
  - Motion blur from moving vehicles
  - Monsoon rain streaks
  - Dust and dirt haze
  - Perspective distortion from non-frontal cameras
  - JPEG compression artefacts from dashcams
  - Faded / dirty plate paint

Usage — apply to images BEFORE writing to the YOLO dataset:
    from utils.augment import augment_plate, augment_scene

    # For a plate crop (no bbox needed)
    aug_img = augment_plate(img)

    # For a full scene with bounding boxes (YOLO format)
    aug_img, aug_bboxes = augment_scene(img, bboxes, class_labels)
"""

import random
from typing import List, Tuple

import cv2
import numpy as np

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Run: pip install albumentations")


# ---------------------------------------------------------------------------
# Plate-crop augmentation (no bounding box tracking needed)
# ---------------------------------------------------------------------------

def _build_plate_pipeline(mode: str = "standard") -> "A.Compose":
    """
    Build an Albumentations pipeline for plate crops.

    mode:
      'standard' — General real-world variation.
      'night'    — Aggressive dark + noise, simulates night / tunnel.
      'ir'       — Simulates IR camera (high brightness, low colour).
      'heavy'    — All degradations at higher probability.
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for augmentation.")

    if mode == "night":
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.6, -0.2),
                contrast_limit=(-0.2, 0.3),
                p=1.0,
            ),
            A.GaussNoise(var_limit=(20, 80), p=0.8),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
            A.ImageCompression(quality_lower=50, quality_upper=75, p=0.4),
        ])

    if mode == "ir":
        return A.Compose([
            A.ToGray(p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(0.2, 0.5),
                contrast_limit=(0.3, 0.6),
                p=1.0,
            ),
            A.GaussNoise(var_limit=(5, 25), p=0.5),
        ])

    if mode == "heavy":
        return A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.5, 0.2), p=1.0),
                A.RandomGamma(gamma_limit=(40, 200), p=1.0),
            ], p=0.8),
            A.GaussNoise(var_limit=(10, 60), p=0.5),
            A.MotionBlur(blur_limit=(3, 9), p=0.4),
            A.Perspective(scale=(0.02, 0.10), p=0.5),
            A.RandomRain(
                slant_lower=-5, slant_upper=5,
                drop_length=10, drop_width=1,
                blur_value=3, p=0.2,
            ),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
            A.ImageCompression(quality_lower=40, quality_upper=80, p=0.4),
            A.Rotate(limit=10, p=0.4),
        ])

    # Standard
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.4, 0.15),
            contrast_limit=(-0.2, 0.3),
            p=0.5,
        ),
        A.GaussNoise(var_limit=(5, 30), p=0.3),
        A.MotionBlur(blur_limit=(3, 5), p=0.3),
        A.Perspective(scale=(0.02, 0.06), p=0.4),
        A.ImageCompression(quality_lower=60, quality_upper=90, p=0.3),
        A.Rotate(limit=8, p=0.3),
        A.RandomRain(p=0.1),
    ])


def augment_plate(
    img: np.ndarray,
    mode: str = "standard",
    n: int = 1,
) -> list:
    """
    Apply augmentation to a plate crop.

    Args:
        img:  BGR plate crop.
        mode: Augmentation mode — 'standard', 'night', 'ir', 'heavy'.
        n:    Number of augmented versions to return.

    Returns:
        List of n augmented BGR images.
    """
    if not HAS_ALBUMENTATIONS:
        return [img] * n

    pipeline = _build_plate_pipeline(mode)
    results = []
    for _ in range(n):
        out = pipeline(image=img)["image"]
        results.append(out)
    return results


# ---------------------------------------------------------------------------
# Full-scene augmentation (tracks bounding boxes)
# ---------------------------------------------------------------------------

def _build_scene_pipeline() -> "A.Compose":
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required.")

    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, 0.15), p=0.5),
            A.GaussNoise(var_limit=(5, 30), p=0.3),
            A.MotionBlur(blur_limit=(3, 5), p=0.3),
            A.Perspective(scale=(0.01, 0.05), p=0.4),
            A.ImageCompression(quality_lower=60, p=0.3),
            A.Rotate(limit=5, p=0.3),
            A.RandomRain(p=0.1),
            A.HorizontalFlip(p=0.0),   # Never flip — plates have text direction
            A.VerticalFlip(p=0.0),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def augment_scene(
    img: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]],
    class_labels: List[int],
) -> Tuple[np.ndarray, List, List]:
    """
    Augment a full-scene image while preserving bounding boxes.

    Args:
        img:           BGR image.
        bboxes:        List of (cx, cy, w, h) in YOLO normalised format.
        class_labels:  Corresponding class indices.

    Returns:
        (augmented_img, augmented_bboxes, augmented_class_labels)
    """
    if not HAS_ALBUMENTATIONS:
        return img, bboxes, class_labels

    pipeline = _build_scene_pipeline()
    out = pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
    return out["image"], out["bboxes"], out["class_labels"]


# ---------------------------------------------------------------------------
# Offline augmentation utility — generate extra training images
# ---------------------------------------------------------------------------

def generate_augmented_dataset(
    src_img_dir: str,
    src_lbl_dir: str,
    out_img_dir: str,
    out_lbl_dir: str,
    multiplier: int = 3,
    mode: str = "standard",
) -> int:
    """
    Generate `multiplier` augmented versions of every image in src dirs,
    writing outputs to out dirs. Preserves YOLO label format.

    Args:
        src_img_dir:  Source images directory.
        src_lbl_dir:  Source labels directory (.txt YOLO format).
        out_img_dir:  Output images directory.
        out_lbl_dir:  Output labels directory.
        multiplier:   How many augmented copies per original.
        mode:         Augmentation mode.

    Returns:
        Number of augmented images written.
    """
    from pathlib import Path
    import shutil

    src_imgs = Path(src_img_dir)
    src_lbls = Path(src_lbl_dir)
    out_imgs = Path(out_img_dir)
    out_lbls = Path(out_lbl_dir)
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    pipeline = _build_plate_pipeline(mode) if HAS_ALBUMENTATIONS else None
    count = 0

    for img_path in sorted(src_imgs.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        lbl_path = src_lbls / (img_path.stem + ".txt")

        # Copy original
        shutil.copy2(img_path, out_imgs / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, out_lbls / lbl_path.name)

        if pipeline is None:
            continue

        # Write augmented copies
        for i in range(multiplier):
            aug = pipeline(image=img)["image"]
            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(out_imgs / aug_name), aug)
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbls / (img_path.stem + f"_aug{i}.txt"))
            count += 1

    print(f"Generated {count} augmented images → {out_imgs}")
    return count
