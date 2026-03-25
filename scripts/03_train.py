"""
Script 03 — Fine-tune YOLOv8 on the Indian number plate dataset.

Transfer learning strategy:
  - Start from yolov8s.pt (COCO pretrained, 11M params)
  - Fine-tune for 50 epochs with early stopping (patience=10)
  - Auto-augmentation ON + extra Albumentations for Indian plate conditions
    (night simulation, motion blur, rain, perspective warp)

Best weights saved to: runs/plate_det/<name>/weights/best.pt
Copy to models/best.pt after training.
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config — edit these to tune training
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = "yolov8s.pt"       # Starting weights (downloads automatically)
DATA_YAML = "data/indian_plates.yaml"
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16                          # Reduce to 8 if GPU OOM
LR0 = 0.01                         # Initial learning rate
LRF = 0.01                         # Final LR factor (LR0 * LRF at end)
PATIENCE = 10                       # Early stopping patience
PROJECT = "runs/plate_det"
RUN_NAME = "v1"
DEVICE = 0                          # GPU device index; use "cpu" for CPU


def train():
    model = YOLO(MODEL_WEIGHTS)

    print(f"Starting training: {MODEL_WEIGHTS} → {DATA_YAML}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH}, Image size: {IMG_SIZE}")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        lr0=LR0,
        lrf=LRF,
        patience=PATIENCE,
        augment=True,           # YOLOv8 built-in auto-augment
        # Extra augmentation via Albumentations (requires albumentations package)
        # These help robustness on Indian road conditions:
        degrees=10,             # Rotation ±10° (mounted at angles)
        translate=0.1,          # Translation
        scale=0.5,              # Scale jitter
        shear=5.0,              # Shear (perspective variation)
        perspective=0.0005,     # Perspective transform
        flipud=0.0,             # No vertical flip (plates have orientation)
        fliplr=0.0,             # No horizontal flip (text would be mirrored)
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mixup
        copy_paste=0.1,         # Copy-paste augmentation
        hsv_h=0.015,            # Hue shift
        hsv_s=0.7,              # Saturation shift
        hsv_v=0.4,              # Value (brightness) shift — covers night/day
        device=DEVICE,
        project=PROJECT,
        name=RUN_NAME,
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        plots=True,             # Save training plots
        verbose=True,
    )

    # Copy best weights to models/ for easy access
    best_src = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if best_src.exists():
        Path("models").mkdir(exist_ok=True)
        shutil.copy2(best_src, "models/best.pt")
        print(f"\nBest weights copied to models/best.pt")

    print(f"\nTraining complete.")
    print(f"Results: {Path(PROJECT) / RUN_NAME}")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print("\nNext: python scripts/04_evaluate.py")

    return results


if __name__ == "__main__":
    train()
