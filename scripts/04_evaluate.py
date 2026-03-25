"""
Script 04 — Evaluate the trained model on the test split.

Outputs:
  - mAP50, mAP50-95, Precision, Recall
  - Confusion matrix image
  - Precision-Recall curve
  - F1 curve
  - Sample predictions on test images
All saved to runs/plate_det_eval/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "models/best.pt"
DATA_YAML = "data/indian_plates.yaml"
IMG_SIZE = 640
CONF_THRESH = 0.50
IOU_THRESH = 0.45
EVAL_DIR = Path("runs/plate_det_eval")


def evaluate(model_path: str = MODEL_PATH):
    model = YOLO(model_path)
    print(f"Evaluating: {model_path}")

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save_json=True,
        plots=True,
        project=str(EVAL_DIR.parent),
        name=EVAL_DIR.name,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"mAP@0.5:       {metrics.box.map50:.4f}   (target > 0.92)")
    print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")
    print(f"F1:            {2 * metrics.box.mp * metrics.box.mr / max(metrics.box.mp + metrics.box.mr, 1e-6):.4f}")
    print("=" * 50)
    print(f"\nPlots saved to: {EVAL_DIR}")

    return metrics


def quick_predict_samples(model_path: str = MODEL_PATH, n: int = 8):
    """Run inference on a few test images and save side-by-side comparison."""
    from pathlib import Path
    import random

    test_images = list(Path("data/processed/images/test").glob("*.jpg"))
    test_images += list(Path("data/processed/images/test").glob("*.png"))
    if not test_images:
        print("No test images found.")
        return

    sample = random.sample(test_images, min(n, len(test_images)))
    model = YOLO(model_path)
    results = model(sample, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)

    panels = []
    for r in results:
        annotated = r.plot()
        annotated = cv2.resize(annotated, (320, 240))
        panels.append(annotated)

    cols = 4
    rows = []
    for i in range(0, len(panels), cols):
        chunk = panels[i : i + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros((240, 320, 3), dtype=np.uint8))
        rows.append(np.hstack(chunk))
    grid = np.vstack(rows)

    out_path = EVAL_DIR / "sample_predictions.jpg"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"Sample predictions saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--samples", action="store_true",
                        help="Also generate sample prediction images")
    args = parser.parse_args()

    evaluate(args.model)
    if args.samples:
        quick_predict_samples(args.model)
