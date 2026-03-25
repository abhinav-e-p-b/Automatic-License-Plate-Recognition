"""
detect_image.py — Run the full ANPR pipeline on a single image.

Usage:
  python detect_image.py --source trial.jpg
  python detect_image.py --source trial.jpg --show
  python detect_image.py --source trial.jpg --output outputs/result.jpg
  python detect_image.py --source trial.jpg --debug
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

from utils.preprocess import preprocess_plate
from utils.ocr import normalise_raw, fix_characters, validate_plate
from utils.visualise import draw_detections

DEFAULT_MODEL = "models/best.pt"
CONF_THRESH   = 0.25    # Low — catches weak detections
IOU_THRESH    = 0.45
OCR_MIN_CONF  = 0.15
CROP_PAD      = 6       # Extra pixels around each YOLO crop
# Boxes covering more than this fraction are "whole-image" (crop-model artefact)
WHOLE_IMAGE_FRACTION = 0.70


# ---------------------------------------------------------------------------
# OCR across all variants
# ---------------------------------------------------------------------------

def _ocr_variants(
    crop: np.ndarray,
    reader,
    debug: bool = False,
) -> Tuple[Optional[str], float]:
    """
    Run EasyOCR on every preprocessing variant of `crop`.
    Returns the first validated plate string + its confidence, or (None, 0).
    """
    # preprocess_plate returns dict {name: image_array}
    variants = preprocess_plate(crop)

    for name, img in variants.items():
        # img is a numpy array — pass directly to readtext
        results = reader.readtext(img, detail=1, paragraph=False)

        if not results:
            if debug:
                print(f"    [{name:12s}] → no text")
            continue

        results  = sorted(results, key=lambda r: r[0][0][1])  # top → bottom
        raw      = "".join(r[1] for r in results)
        avg_conf = sum(r[2] for r in results) / len(results)

        if avg_conf < OCR_MIN_CONF:
            if debug:
                print(f"    [{name:12s}] raw='{raw}'  conf={avg_conf:.2f}  → below min_conf")
            continue

        normalised = normalise_raw(raw)
        fixed      = fix_characters(normalised)
        plate      = validate_plate(fixed)

        if debug:
            verdict = plate if plate else f"rejected (raw='{raw}' fixed='{fixed}')"
            print(f"    [{name:12s}] conf={avg_conf:.2f}  {verdict}")

        if plate:
            return plate, avg_conf

    return None, 0.0


# ---------------------------------------------------------------------------
# Contour-based plate region finder (classical CV fallback)
# ---------------------------------------------------------------------------

def _find_plate_candidates(img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """
    Find candidate plate bounding boxes using edge detection + contour filtering.
    Returns list of (x1,y1,x2,y2) sorted by area descending.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 200)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > w * h * 0.70:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect = bw / bh
        if 1.5 <= aspect <= 6.0:           # Indian plates are 2:1 to 5:1
            candidates.append((x, y, x + bw, y + bh))

    candidates.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
    return candidates[:5]


def _pad(img, x1, y1, x2, y2, pad=CROP_PAD):
    h, w = img.shape[:2]
    return (max(0, x1-pad), max(0, y1-pad),
            min(w, x2+pad), min(h, y2+pad))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    source: str,
    model_path: str = DEFAULT_MODEL,
    conf: float     = CONF_THRESH,
    iou: float      = IOU_THRESH,
    show: bool      = False,
    output: str     = None,
    debug: bool     = False,
) -> list:

    img = cv2.imread(source)
    if img is None:
        print(f"ERROR: Cannot read '{source}'")
        sys.exit(1)

    h, w   = img.shape[:2]
    img_area = w * h
    print(f"Image : {source}  ({w}×{h})")

    import easyocr
    detector = YOLO(model_path)
    reader   = easyocr.Reader(["en"], gpu=True, verbose=False)

    t0 = time.perf_counter()

    yolo_res  = detector(img, conf=conf, iou=iou, verbose=False)
    all_boxes = yolo_res[0].boxes
    print(f"YOLO  : {len(all_boxes)} detection(s) at conf≥{conf}")

    plates   = []
    det_list = []

    # Separate "tight" boxes (real plate regions) from whole-image boxes
    tight_boxes = []
    for box in all_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2-x1)*(y2-y1) / img_area <= WHOLE_IMAGE_FRACTION:
            tight_boxes.append(box)
        else:
            print(f"  Skipping whole-image box ({(x2-x1)*(y2-y1)/img_area*100:.0f}% of frame)")
            print("  ← model was trained on plate crops, not full-scene images")
            print("    Run: python scripts/10_retrain_fullscene.py  to fix permanently")

    # ------------------------------------------------------------------
    # Path A — YOLO gave tight boxes
    # ------------------------------------------------------------------
    if tight_boxes:
        for box in tight_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])
            x1p, y1p, x2p, y2p = _pad(img, x1, y1, x2, y2)
            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            if debug:
                print(f"\n  Box ({x1},{y1})→({x2},{y2}) conf={det_conf:.3f}")
            plate, ocr_conf = _ocr_variants(crop, reader, debug)
            det_list.append((x1p, y1p, x2p, y2p, det_conf))
            plates.append({"bbox":(x1p,y1p,x2p,y2p), "det_conf":det_conf,
                            "plate":plate, "ocr_conf":round(ocr_conf,3)})

    else:
        # ------------------------------------------------------------------
        # Path B — No tight YOLO boxes: try contour finder first
        # ------------------------------------------------------------------
        print("  Trying contour-based plate finder...")
        candidates = _find_plate_candidates(img)
        print(f"  Contour candidates: {len(candidates)}")

        found_via_contour = False
        for x1, y1, x2, y2 in candidates:
            crop = img[y1:y2, x1:x2]
            if debug:
                print(f"\n  Contour ({x1},{y1})→({x2},{y2})")
            plate, ocr_conf = _ocr_variants(crop, reader, debug)
            if plate:
                det_list.append((x1, y1, x2, y2, 0.6))
                plates.append({"bbox":(x1,y1,x2,y2), "det_conf":0.6,
                                "plate":plate, "ocr_conf":round(ocr_conf,3)})
                found_via_contour = True
                break

        # ------------------------------------------------------------------
        # Path C — Last resort: OCR on whole image
        # (works when input image IS already a tight plate crop)
        # ------------------------------------------------------------------
        if not found_via_contour:
            print("  Trying OCR on whole image (last resort)...")
            if debug:
                print()
            plate, ocr_conf = _ocr_variants(img, reader, debug)
            det_list.append((0, 0, w, h, 1.0))
            plates.append({"bbox":(0,0,w,h), "det_conf":1.0,
                            "plate":plate, "ocr_conf":round(ocr_conf,3)})

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n{'─'*52}")
    for p in plates:
        x1,y1,x2,y2 = p["bbox"]
        result = p["plate"] or "[unreadable]"
        print(f"  Plate    : {result}")
        if p["plate"]:
            print(f"  OCR conf : {p['ocr_conf']:.2f}")
        print(f"  Det conf : {p['det_conf']:.3f}")
        print(f"  Bbox     : ({x1},{y1})→({x2},{y2})")

    if not any(p["plate"] for p in plates):
        print()
        print("  Still unreadable. Run the full diagnostic:")
        print(f"  python diagnose.py --source {source}")

    print(f"\n  Time : {elapsed*1000:.0f}ms")
    print(f"{'─'*52}\n")

    # ------------------------------------------------------------------
    # Annotate + save/show
    # ------------------------------------------------------------------
    plate_texts = [p["plate"] for p in plates]
    annotated   = draw_detections(img, det_list, plate_texts)

    if show:
        cv2.imshow("Indian ANPR", annotated)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output, annotated)
        print(f"Saved  : {output}")

    return plates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian ANPR — single image")
    parser.add_argument("--source",  required=True)
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--conf",    type=float, default=CONF_THRESH)
    parser.add_argument("--iou",     type=float, default=IOU_THRESH)
    parser.add_argument("--show",    action="store_true")
    parser.add_argument("--output",  default=None)
    parser.add_argument("--debug",   action="store_true",
                        help="Print per-variant OCR output")
    args = parser.parse_args()

    run(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        output=args.output,
        debug=args.debug,
    )
