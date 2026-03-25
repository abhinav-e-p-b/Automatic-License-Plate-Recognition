"""
Script 07 — Prepare a full-scene dataset for training.

Use this script when your training images are full car/road photos
(not pre-cropped plate images). It uses a pretrained YOLO model to
auto-generate bounding box labels which you should then review and
correct in Roboflow or LabelImg.

Workflow:
  1. Place full-scene images in data/fullscene/
  2. Run this script — it auto-labels using a pretrained model
  3. Review / fix labels in data/fullscene/labels/
  4. Re-run scripts/02_prepare_dataset.py pointing at this new folder
     OR manually merge into data/processed/ splits

Usage:
  python scripts/07_prepare_fullscene.py --src data/fullscene/images
  python scripts/07_prepare_fullscene.py --src data/fullscene/images --model models/best.pt
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_BOOTSTRAP_MODEL = "yolov8s.pt"  # Use best.pt if already trained
CONF_THRESH = 0.35   # Lower threshold for labelling (better recall)
IOU_THRESH = 0.45
OUT_LABEL_DIR = Path("data/fullscene/labels")


def auto_label(src_dir: str, model_path: str, conf: float, iou: float):
    src = Path(src_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in src.iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"No images found in {src}")
        return

    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    print(f"Auto-labelling {len(images)} images with {model_path}")
    print(f"Confidence threshold: {conf}  (lower = more labels, more false positives)")

    labeled = 0
    unlabeled = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        results = model(img, conf=conf, iou=iou, verbose=False)
        boxes = results[0].boxes

        label_path = OUT_LABEL_DIR / (img_path.stem + ".txt")

        if len(boxes) == 0:
            # Write empty label — still needed for YOLO background training
            label_path.write_text("")
            unlabeled += 1
            continue

        lines = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text("\n".join(lines) + "\n")
        labeled += 1

    print(f"\nLabelled: {labeled} images  ({unlabeled} with no detections)")
    print(f"Labels saved to: {OUT_LABEL_DIR.resolve()}")
    print("\nIMPORTANT: Review labels before training!")
    print("Recommended: upload to Roboflow for visual review + correction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Directory of full-scene images")
    parser.add_argument("--model", default=DEFAULT_BOOTSTRAP_MODEL,
                        help="YOLO model for bootstrapping labels")
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument("--iou", type=float, default=IOU_THRESH)
    args = parser.parse_args()
    auto_label(args.src, args.model, args.conf, args.iou)
