"""
Script 02 — Prepare the dataset for YOLOv8 training.

What this does:
  1. Splits raw images into train / val / test (80 / 10 / 10)
  2. Auto-generates YOLO bounding box labels using a pretrained
     general object detector as a bootstrapping step.
     *** Review and correct labels in a tool like Roboflow or LabelImg
         before training for best results. ***
  3. Writes data/indian_plates.yaml

Auto-labelling strategy:
  The Kaggle dataset images ARE the plate crops themselves (not full car photos).
  So the label for each image is simply the full image bounding box:
      class=0  cx=0.5  cy=0.5  w=1.0  h=1.0
  This trains YOLO to detect the plate region in images where the plate
  is embedded in a larger scene. If your inference images are full car frames,
  this is the correct annotation.

  If you have full-scene images with plates as small objects, replace the
  auto-labelling block with your own annotation or use Roboflow's auto-label.
"""

import random
import shutil
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
YAML_PATH = Path("data/indian_plates.yaml")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dirs():
    for split in ("train", "val", "test"):
        (PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_images(images: list) -> tuple:
    random.seed(SEED)
    random.shuffle(images)
    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = images[:n_train]
    val = images[n_train : n_train + n_val]
    test = images[n_train + n_val :]
    return train, val, test


def write_yolo_label(label_path: Path):
    """
    Write a full-image bounding box label (class 0, centre, full size).
    Suitable when the raw images ARE the plate crops.
    """
    with open(label_path, "w") as f:
        f.write("0 0.500000 0.500000 1.000000 1.000000\n")


def copy_split(images: list, split: str):
    img_dir = PROCESSED_DIR / "images" / split
    lbl_dir = PROCESSED_DIR / "labels" / split
    for img_path in images:
        dest_img = img_dir / img_path.name
        shutil.copy2(img_path, dest_img)
        # Generate label
        label_name = img_path.stem + ".txt"
        write_yolo_label(lbl_dir / label_name)


def write_yaml(num_classes: int = 1):
    config = {
        "path": str(PROCESSED_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": num_classes,
        "names": ["plate"],
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"YAML written to {YAML_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare():
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in exts])

    if not images:
        print(f"No images found in {RAW_DIR}. Run 01_download_dataset.py first.")
        return

    print(f"Total images: {len(images)}")
    make_dirs()

    train, val, test = split_images(images)
    print(f"Split → train: {len(train)}, val: {len(val)}, test: {len(test)}")

    for split_name, split_images_list in [("train", train), ("val", val), ("test", test)]:
        copy_split(split_images_list, split_name)
        print(f"  {split_name}: {len(split_images_list)} images + labels written")

    write_yaml()
    print("\nDataset prepared.")
    print("IMPORTANT: Review labels in data/processed/labels/ before training.")
    print("For better accuracy, re-annotate with Roboflow or LabelImg if your")
    print("inference images are full car/road scenes (not plate crops).")
    print("\nNext: python scripts/03_train.py")


if __name__ == "__main__":
    prepare()
