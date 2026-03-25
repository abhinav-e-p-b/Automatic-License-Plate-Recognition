"""
Script 10 — Retrain YOLOv8 with full-scene plate annotations.

Run this AFTER you have properly labeled full-scene images
(from Roboflow download or manual LabelImg labeling).

Key differences from the first training run:
  - freeze=10: freezes the first 10 backbone layers so we keep
    low-level feature detection and only retrain the detection head.
    This converges faster and prevents catastrophic forgetting.
  - warmup_epochs=3: gentle LR warmup since we're fine-tuning
  - Higher augmentation: more perspective + brightness variation
    because real road images have far more lighting/angle diversity
    than the crop-only Kaggle dataset.
  - resume: can resume an interrupted run

Usage:
    python scripts/10_retrain_fullscene.py
    python scripts/10_retrain_fullscene.py --resume runs/plate_det/v2/weights/last.pt
    python scripts/10_retrain_fullscene.py --epochs 100 --batch 8
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

DEFAULT_BASE   = "yolov8s.pt"    # start fresh from COCO
EXISTING_MODEL = "models/best.pt"  # use this if it exists (fine-tune further)
DATA_YAML      = "data/indian_plates.yaml"
PROJECT        = "runs/plate_det"
RUN_NAME       = "v2_fullscene"


def retrain(
    epochs: int  = 80,
    batch: int   = 16,
    device: int  = 0,
    resume: str  = None,
):
    # Use existing weights if available (fine-tune rather than start fresh)
    if resume:
        print(f"Resuming from: {resume}")
        model = YOLO(resume)
        results = model.train(resume=True)
        return results

    if Path(EXISTING_MODEL).exists():
        start_weights = EXISTING_MODEL
        print(f"Fine-tuning from existing model: {start_weights}")
    else:
        start_weights = DEFAULT_BASE
        print(f"Starting from COCO pretrained: {start_weights}")

    model = YOLO(start_weights)

    print(f"Training on: {DATA_YAML}")
    print(f"Epochs: {epochs}, Batch: {batch}, Device: {device}")

    results = model.train(
        data           = DATA_YAML,
        epochs         = epochs,
        imgsz          = 640,
        batch          = batch,
        lr0            = 0.005,          # Lower LR for fine-tuning
        lrf            = 0.01,
        warmup_epochs  = 3,
        patience       = 15,
        freeze         = 10,             # Freeze first 10 backbone layers
        # Augmentation — stronger than first run to handle real road conditions
        degrees        = 15.0,           # More rotation for mounted cameras
        translate      = 0.1,
        scale          = 0.6,
        shear          = 8.0,
        perspective    = 0.001,          # More perspective for angled shots
        flipud         = 0.0,
        fliplr         = 0.0,            # Never flip (plate text has direction)
        mosaic         = 1.0,
        mixup          = 0.1,
        hsv_h          = 0.02,
        hsv_s          = 0.8,
        hsv_v          = 0.5,            # Strong brightness variation (day/night)
        copy_paste     = 0.15,
        device         = device,
        project        = PROJECT,
        name           = RUN_NAME,
        save           = True,
        save_period    = 10,
        plots          = True,
        verbose        = True,
    )

    best_src = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if best_src.exists():
        Path("models").mkdir(exist_ok=True)
        shutil.copy2(best_src, "models/best.pt")
        print(f"\nBest weights → models/best.pt")
        print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch",  type=int, default=16,
                        help="Reduce to 8 if GPU runs out of memory")
    parser.add_argument("--device", default=0,
                        help="GPU index (0,1,...) or 'cpu'")
    parser.add_argument("--resume", default=None,
                        help="Path to last.pt to resume interrupted training")
    args = parser.parse_args()

    retrain(
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        resume=args.resume,
    )
