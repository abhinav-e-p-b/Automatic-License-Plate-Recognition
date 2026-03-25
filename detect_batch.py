"""
detect_batch.py — Run ANPR on every image in a directory.

Outputs:
  - Annotated images saved alongside originals (or to a separate folder)
  - CSV summary: filename, plate_text, confidence, bbox
  - Console summary with success rate

Usage:
  python detect_batch.py --source data/raw/
  python detect_batch.py --source data/raw/ --output outputs/batch/ --csv outputs/results.csv
  python detect_batch.py --source data/raw/ --workers 4
"""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils.preprocess import preprocess_plate
from utils.ocr import PlateReader
from utils.visualise import draw_detections

DEFAULT_MODEL = "models/best.pt"
CONF_THRESH = 0.50
IOU_THRESH = 0.45
OCR_MIN_CONF = 0.30
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def process_single(
    img_path: Path,
    detector: YOLO,
    reader: PlateReader,
    conf: float,
    iou: float,
    output_dir: Path = None,
) -> dict:
    """Process one image. Returns a result dict."""
    img = cv2.imread(str(img_path))
    if img is None:
        return {"file": img_path.name, "plate": None, "error": "unreadable"}

    results = detector(img, conf=conf, iou=iou, verbose=False)
    boxes = results[0].boxes

    det_list = []
    plate_texts = []
    best_plate = None
    best_conf = 0.0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        det_conf = float(box.conf[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        processed = preprocess_plate(crop)
        plate_text = reader.read(processed, min_conf=OCR_MIN_CONF)

        det_list.append((x1, y1, x2, y2, det_conf))
        plate_texts.append(plate_text)

        if plate_text and det_conf > best_conf:
            best_plate = plate_text
            best_conf = det_conf
            best_bbox = (x1, y1, x2, y2)

    if output_dir and det_list:
        annotated = draw_detections(img, det_list, plate_texts)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / img_path.name), annotated)

    result = {
        "file": img_path.name,
        "plate": best_plate,
        "det_conf": round(best_conf, 3) if best_plate else None,
        "n_detections": len(boxes),
        "error": None,
    }
    if best_plate:
        result["bbox"] = best_bbox

    return result


def run_batch(
    source: str,
    model_path: str = DEFAULT_MODEL,
    conf: float = CONF_THRESH,
    iou: float = IOU_THRESH,
    output: str = None,
    save_csv: str = None,
    workers: int = 1,
) -> list:

    src = Path(source)
    images = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS])

    if not images:
        print(f"No images found in {src}")
        return []

    print(f"Batch inference on {len(images)} images")
    print(f"Model: {model_path}  conf={conf}  iou={iou}")

    detector = YOLO(model_path)
    reader = PlateReader(gpu=True)
    output_dir = Path(output) if output else None

    t0 = time.perf_counter()
    all_results = []

    if workers > 1:
        # Note: YOLO + EasyOCR are GPU-bound; threading helps mostly with I/O
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_single, p, detector, reader, conf, iou, output_dir): p
                for p in images
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                all_results.append(result)
                status = result["plate"] or result.get("error") or "no plate"
                print(f"  [{i:4d}/{len(images)}]  {result['file']:40s}  {status}")
    else:
        for i, img_path in enumerate(images, 1):
            result = process_single(img_path, detector, reader, conf, iou, output_dir)
            all_results.append(result)
            status = result["plate"] or result.get("error") or "no plate"
            print(f"  [{i:4d}/{len(images)}]  {result['file']:40s}  {status}")

    elapsed = time.perf_counter() - t0
    plates_found = [r for r in all_results if r["plate"]]
    success_rate = len(plates_found) / len(all_results) * 100

    print(f"\n{'='*55}")
    print(f"Processed:    {len(all_results)} images  in {elapsed:.1f}s")
    print(f"Plates read:  {len(plates_found)} / {len(all_results)} ({success_rate:.1f}%)")
    print(f"Avg speed:    {elapsed/len(all_results)*1000:.0f}ms per image")
    print(f"{'='*55}")

    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file", "plate", "det_conf", "n_detections", "error"])
            writer.writeheader()
            writer.writerows(
                {k: r.get(k, "") for k in ["file", "plate", "det_conf", "n_detections", "error"]}
                for r in all_results
            )
        print(f"CSV saved: {save_csv}")

    if output:
        print(f"Annotated images: {output_dir}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian ANPR — batch inference")
    parser.add_argument("--source", required=True, help="Directory of input images")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument("--iou", type=float, default=IOU_THRESH)
    parser.add_argument("--output", default=None, help="Save annotated images here")
    parser.add_argument("--csv", default=None, dest="save_csv",
                        help="Save results to CSV")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker threads (default: 1)")
    args = parser.parse_args()

    run_batch(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        output=args.output,
        save_csv=args.save_csv,
        workers=args.workers,
    )
