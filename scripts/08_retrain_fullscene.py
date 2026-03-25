"""
Script 08 — Retrain with full-scene images so YOLO learns real plate locations.

This is the permanent fix for the "YOLO detects entire image" problem.

The Kaggle dataset only has plate crops. To detect plates in actual road
photos you need training images that show the whole car with a small plate
bbox — not full-image labels.

This script:
  1. Downloads a supplementary full-scene dataset (IIT-Delhi ANPR or
     Roboflow Indian plates — whichever is accessible)
  2. Merges it with your existing crop-based data
  3. Rewrites the YAML to point at the merged dataset
  4. Relaunches training

If you have your own full-scene images, place them in:
    data/fullscene/images/   (images)
    data/fullscene/labels/   (YOLO .txt annotations)

Then just run:
    python scripts/08_retrain_fullscene.py --skip-download
"""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FULLSCENE_IMG_DIR   = Path("data/fullscene/images")
FULLSCENE_LBL_DIR   = Path("data/fullscene/labels")
MERGED_DIR          = Path("data/merged")
MERGED_YAML         = Path("data/merged_plates.yaml")

# Supplementary Roboflow dataset (open-access Indian plate full-scene)
# Replace with your own dataset URL if you have one
ROBOFLOW_DATASET_URL = (
    "https://universe.roboflow.com/ds/..."
    # Sign in to Roboflow, search "indian license plate detection",
    # click Export > YOLOv8 > get download URL
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge_splits(crop_dir: Path, scene_dir: Path, out_dir: Path):
    """Merge crop-dataset and full-scene dataset into one folder structure."""
    for split in ("train", "val", "test"):
        for subdir in ("images", "labels"):
            (out_dir / subdir / split).mkdir(parents=True, exist_ok=True)

    copied = {"train": 0, "val": 0, "test": 0}

    # Copy crop-dataset
    for split in ("train", "val", "test"):
        img_src = Path("data/processed/images") / split
        lbl_src = Path("data/processed/labels") / split
        for img in img_src.glob("*"):
            shutil.copy2(img, out_dir / "images" / split / ("crop_" + img.name))
        for lbl in lbl_src.glob("*.txt"):
            shutil.copy2(lbl, out_dir / "labels" / split / ("crop_" + lbl.name))
            copied[split] += 1

    # Copy full-scene data (all goes to train, we re-split below)
    scene_imgs = list(FULLSCENE_IMG_DIR.glob("*.jpg")) + \
                 list(FULLSCENE_IMG_DIR.glob("*.png"))

    import random
    random.seed(42)
    random.shuffle(scene_imgs)
    n = len(scene_imgs)
    splits = {
        "train": scene_imgs[:int(n * 0.80)],
        "val":   scene_imgs[int(n * 0.80):int(n * 0.90)],
        "test":  scene_imgs[int(n * 0.90):],
    }

    for split, imgs in splits.items():
        for img in imgs:
            shutil.copy2(img, out_dir / "images" / split / ("scene_" + img.name))
            lbl = FULLSCENE_LBL_DIR / (img.stem + ".txt")
            dest_lbl = out_dir / "labels" / split / ("scene_" + img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dest_lbl)
            else:
                dest_lbl.write_text("")   # empty label = background image
            copied[split] += 1

    return copied


def write_yaml(merged_dir: Path):
    config = {
        "path":  str(merged_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": ["plate"],
    }
    with open(MERGED_YAML, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"YAML written: {MERGED_YAML}")


def train(weights: str = "yolov8s.pt"):
    model = YOLO(weights)
    print(f"\nRetraining on merged dataset: {MERGED_YAML}")
    results = model.train(
        data=str(MERGED_YAML),
        epochs=60,
        imgsz=640,
        batch=16,
        lr0=0.005,          # Lower LR — more careful fine-tuning
        patience=12,
        augment=True,
        degrees=10,
        perspective=0.001,
        hsv_v=0.5,
        device=0,
        project="runs/plate_det",
        name="fullscene_v1",
        save=True,
        plots=True,
    )
    best = Path("runs/plate_det/fullscene_v1/weights/best.pt")
    if best.exists():
        shutil.copy2(best, "models/best_fullscene.pt")
        print("\nBest weights → models/best_fullscene.pt")
        print("Use with:  python detect_image.py --source img.jpg --model models/best_fullscene.pt")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_download: bool = False, weights: str = "yolov8s.pt"):
    if not skip_download:
        print("=" * 60)
        print("MANUAL STEP REQUIRED")
        print("=" * 60)
        print()
        print("The Kaggle dataset contains only plate crops, not full-scene images.")
        print("You need to add full-scene images with proper plate bounding boxes.")
        print()
        print("Option A — Roboflow (recommended, free):")
        print("  1. Go to: https://universe.roboflow.com")
        print("  2. Search: 'indian license plate detection'")
        print("  3. Pick a dataset with 500+ full-scene images")
        print("  4. Export → YOLOv8 format")
        print("  5. Unzip and place:")
        print("     images → data/fullscene/images/")
        print("     labels → data/fullscene/labels/")
        print()
        print("Option B — Your own images:")
        print("  1. Take/collect 200+ road photos with plates visible")
        print("  2. Label with LabelImg or Roboflow annotate tool")
        print("  3. Export YOLOv8 format to data/fullscene/")
        print()
        print("Then re-run: python scripts/08_retrain_fullscene.py --skip-download")
        return

    if not FULLSCENE_IMG_DIR.exists() or not any(FULLSCENE_IMG_DIR.iterdir()):
        print(f"ERROR: No images found in {FULLSCENE_IMG_DIR}")
        print("Add full-scene images there first.")
        return

    n_images = len(list(FULLSCENE_IMG_DIR.glob("*")))
    print(f"Full-scene images found: {n_images}")

    print("Merging datasets...")
    copied = merge_splits(
        Path("data/processed"), FULLSCENE_IMG_DIR, MERGED_DIR
    )
    for split, n in copied.items():
        print(f"  {split}: {n} total images")

    write_yaml(MERGED_DIR)
    train(weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download instructions (data already in place)")
    parser.add_argument("--weights", default="yolov8s.pt",
                        help="Starting weights (default: yolov8s.pt)")
    args = parser.parse_args()
    main(args.skip_download, args.weights)
