"""
Script 09 — Label your own full-scene car images using LabelImg.

If you have photos of cars / roads where the plate is visible,
this script sets up LabelImg so you can draw bounding boxes.

Install:
    pip install labelImg

Usage:
    python scripts/09_label_with_labelimg.py --src data/my_car_photos/

This opens LabelImg pre-configured for YOLO format with class "plate".
Draw a tight box around each number plate, save, move to next image.
Aim for 300-500 labeled images minimum for good results.

Labelling tips:
  - Include the full plate including border frame, not just the text
  - Label plates even if partially occluded or at an angle
  - Include night images, rainy images, dusty plates
  - Include both single-line and two-line (two-wheeler) plates
  - Label every plate visible in the image, even distant/small ones
"""

import argparse
import subprocess
import sys
from pathlib import Path


CLASSES_FILE = Path("data/classes.txt")


def setup_labelimg(src_dir: str):
    src = Path(src_dir)
    if not src.exists():
        src.mkdir(parents=True)
        print(f"Created image directory: {src}")
        print(f"Place your car images in: {src.resolve()}")

    # Write classes file
    CLASSES_FILE.parent.mkdir(exist_ok=True)
    CLASSES_FILE.write_text("plate\n")
    print(f"Classes file: {CLASSES_FILE.resolve()}")

    # Create labels directory matching src
    lbl_dir = src.parent / (src.name + "_labels")
    lbl_dir.mkdir(exist_ok=True)
    print(f"Labels will be saved to: {lbl_dir.resolve()}")

    print("\nOpening LabelImg...")
    print("Keyboard shortcuts:")
    print("  W         — draw bounding box")
    print("  D / A     — next / previous image")
    print("  Ctrl+S    — save labels")
    print("  Ctrl+Z    — undo last box")

    try:
        subprocess.run([
            sys.executable, "-m", "labelImg",
            str(src),
            str(CLASSES_FILE),
            str(lbl_dir),
        ])
    except FileNotFoundError:
        print("\nLabelImg not found. Install it:")
        print("  pip install labelImg")
        print("\nOr use Roboflow Annotate (browser-based, free):")
        print("  https://app.roboflow.com")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/my_car_photos",
                        help="Directory of images to label")
    args = parser.parse_args()
    setup_labelimg(args.src)
