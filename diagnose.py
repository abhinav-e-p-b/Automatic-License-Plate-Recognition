"""
diagnose.py — Step-by-step diagnostic for "plate unreadable" failures.

Run this on any image that gives no result:
  python diagnose.py --source trial.jpg

Saves every intermediate image to outputs/diag/ and prints exactly
which stage is failing so you know what to fix.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

DIAG_DIR = Path("outputs/diag")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Fraction of image area above which a YOLO box is treated as
# "whole-image detection" (crop-trained model artefact)
WHOLE_IMAGE_THRESHOLD = 0.70


def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def save(name: str, img: np.ndarray):
    path = DIAG_DIR / name
    cv2.imwrite(str(path), img)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Stage 2 — preprocessing visual
# ---------------------------------------------------------------------------

def _diagnose_preprocessing(crop: np.ndarray, idx: int):
    from utils.preprocess import (
        upscale, to_gray, bilateral_denoise, sharpen,
        adaptive_threshold, adaptive_threshold_inv,
        otsu_threshold, otsu_threshold_inv, morph_clean,
    )

    up    = upscale(crop, 2.0)
    gray  = to_gray(up)
    den   = bilateral_denoise(gray)
    sharp = sharpen(den)
    th1   = adaptive_threshold(sharp, 31, 5)
    th2   = adaptive_threshold_inv(sharp, 31, 5)
    th3   = otsu_threshold(sharp)
    th4   = otsu_threshold_inv(sharp)
    clean = morph_clean(th2)

    stages = {
        "03_upscaled":    up,
        "04_gray":        gray,
        "05_denoised":    den,
        "06_sharp":       sharp,
        "07_adap":        th1,
        "08_adap_inv":    th2,
        "09_otsu":        th3,
        "10_otsu_inv":    th4,
        "11_clean":       clean,
    }

    print(f"  Upscaled: {crop.shape[:2]} → {up.shape[:2]}")
    white_pct = (th2 == 255).mean() * 100
    print(f"  adap_inv white %: {white_pct:.1f}%", end="  ")
    if white_pct > 90:
        print("⚠️  too white — text may be lost")
    elif white_pct < 10:
        print("⚠️  too black — text invisible")
    else:
        print("✓ good balance")

    for name, img in stages.items():
        save(f"{name}_box{idx}.jpg", img)
    print(f"  Stages saved → {DIAG_DIR}/")


# ---------------------------------------------------------------------------
# Stage 3 — OCR on all variants
# ---------------------------------------------------------------------------

def _diagnose_ocr(crop: np.ndarray, idx: int, reader):
    """
    Run EasyOCR on every preprocessing variant of `crop`.
    `reader` is a pre-created easyocr.Reader instance.
    """
    from utils.preprocess import preprocess_plate
    from utils.ocr import normalise_raw, fix_characters, validate_plate

    # preprocess_plate returns a dict {name: image}
    variants = preprocess_plate(crop)
    best = None

    for name, img in variants.items():
        # img is a numpy array — this is what EasyOCR expects
        results = reader.readtext(img, detail=1, paragraph=False)

        if not results:
            print(f"  [{name:12s}]  → no text detected")
            save(f"ocr_{name}_box{idx}.jpg", img)
            continue

        results = sorted(results, key=lambda r: r[0][0][1])   # top→bottom
        raw      = "".join(r[1] for r in results)
        avg_conf = sum(r[2] for r in results) / len(results)

        normalised = normalise_raw(raw)
        fixed      = fix_characters(normalised)
        validated  = validate_plate(fixed)

        if validated:
            status = f"✓ VALID → {validated}"
        else:
            status = f"✗ rejected  (raw='{normalised}'  fixed='{fixed}')"

        print(f"  [{name:12s}]  conf={avg_conf:.2f}  {status}")
        save(f"ocr_{name}_box{idx}.jpg", img)

        if validated and best is None:
            best = validated

    print()
    if best:
        print(f"  ✓ Best reading: {best}")
    else:
        print("  ✗ No valid plate across all variants.")
        print()
        print("  Next steps:")
        print("  1. Open outputs/diag/02_crop_0.jpg — can YOU read the plate?")
        print("     If no → image quality is too low, nothing to fix in code.")
        print("  2. Open outputs/diag/ocr_gray_box0.jpg — does it look clean?")
        print("     If the image is mostly white/black → preprocessing issue.")
        print("  3. Look at the 'fixed=' values above:")
        print("     If they LOOK like a valid plate but were rejected →")
        print("     the state code may be missing from VALID_STATES in utils/ocr.py")
        print("  4. If YOLO bbox covered the whole image → retrain with")
        print("     full-scene annotated data (see scripts/08_download_roboflow_dataset.py)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def diagnose(source: str, model_path: str = "models/best.pt", conf: float = 0.25):
    import easyocr

    img = cv2.imread(source)
    if img is None:
        print(f"ERROR: Cannot read image: {source}")
        sys.exit(1)

    h, w = img.shape[:2]
    img_area = w * h
    print(f"\nImage: {source}  ({w}×{h} px)")
    save("00_original.jpg", img)

    # Create EasyOCR reader once (slow to initialise)
    print("\nLoading EasyOCR... (first run downloads models, ~1 min)")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print("EasyOCR ready.")

    # ------------------------------------------------------------------
    # Stage 1 — YOLO
    # ------------------------------------------------------------------
    banner("STAGE 1 — YOLO Detection")
    from ultralytics import YOLO
    model   = YOLO(model_path)
    results = model(img, conf=conf, iou=0.45, verbose=False)
    boxes   = results[0].boxes

    print(f"  Detections at conf≥{conf}: {len(boxes)}")

    if len(boxes) == 0:
        print("  ⚠  YOLO found nothing.")
        print("  Trying OCR on whole image as fallback...")
        save("01_yolo_nothing.jpg", results[0].plot())
        banner("FALLBACK — whole-image OCR")
        _diagnose_ocr(img, 0, reader)
        return

    annotated = results[0].plot()
    save("01_yolo_detections.jpg", annotated)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val = float(box.conf[0])
        bw, bh   = x2 - x1, y2 - y1
        box_area = bw * bh

        print(f"\n  Box {i}: ({x1},{y1})→({x2},{y2})  size={bw}×{bh}  conf={conf_val:.3f}")

        # Detect crop-model artefact: bbox covers almost the whole image
        if box_area / img_area > WHOLE_IMAGE_THRESHOLD:
            print(f"  ⚠  WHOLE-IMAGE BOX detected ({box_area/img_area*100:.0f}% of frame).")
            print("     This means YOLO was trained on plate crops and has never seen")
            print("     a plate as a small object inside a larger photo.")
            print("     OCR will fail here — retrain with full-scene images.")
            print("     See: scripts/08_download_roboflow_dataset.py")
            print("     Running OCR anyway for reference...")

        if bw < 20 or bh < 8:
            print(f"  ⚠  Crop is very small ({bw}×{bh}px) — OCR likely to fail.")

        crop = img[y1:y2, x1:x2]
        save(f"02_crop_{i}.jpg", crop)

        banner(f"STAGE 2 — Preprocessing (box {i})")
        _diagnose_preprocessing(crop, i)

        banner(f"STAGE 3 — OCR (box {i})")
        _diagnose_ocr(crop, i, reader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose ANPR pipeline failures")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model",  default="models/best.pt")
    parser.add_argument("--conf",   type=float, default=0.25)
    args = parser.parse_args()
    diagnose(args.source, args.model, args.conf)
