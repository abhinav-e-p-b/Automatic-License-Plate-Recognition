"""
detect_video.py — Run the ANPR pipeline on a video file.

Three-gate efficiency architecture:
  Gate 1 — Nth-frame sampler (skip frames)
  Gate 2 — Motion check (skip static scenes)
  Gate 3 — YOLO detection + OCR

Temporal deduplication: once a plate is read at high confidence,
it is suppressed for COOLDOWN_FRAMES frames to avoid duplicate logs.

Usage:
  python detect_video.py --source video.mp4
  python detect_video.py --source video.mp4 --output outputs/result.mp4 --nth 3
  python detect_video.py --source video.mp4 --show
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils.preprocess import preprocess_plate
from utils.ocr import PlateReader
from utils.tracker import PlateTracker
from utils.visualise import draw_detections, add_fps_overlay

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "models/best.pt"
CONF_THRESH = 0.50
IOU_THRESH = 0.45
OCR_MIN_CONF = 0.30
NTH_FRAME = 3
MOTION_THRESH = 15
COOLDOWN_FRAMES = 30   # Frames to suppress re-logging the same plate
CONFIRM_FRAMES = 3     # Frames before a plate is logged (deduplication)
MAX_LOST = 15


def process_video(
    source: str,
    model_path: str = DEFAULT_MODEL,
    conf: float = CONF_THRESH,
    iou: float = IOU_THRESH,
    nth: int = NTH_FRAME,
    motion_thresh: float = MOTION_THRESH,
    show: bool = False,
    output: str = None,
    save_csv: str = None,
) -> list:

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {source}")
    print(f"Resolution: {width}×{height}  FPS: {fps_in:.1f}  Frames: {total_frames}")

    # Video writer
    writer = None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps_in, (width, height))

    # Models
    detector = YOLO(model_path)
    reader = PlateReader(gpu=True)

    # State
    frame_id = 0
    prev_gray = None
    all_detections = []                  # [{frame, plate, conf, bbox}, ...]
    cooldown: dict = defaultdict(int)    # plate → frames until next log
    fps_timer = time.perf_counter()
    fps_display = 0.0

    # CSV writer
    csv_file = None
    csv_writer = None
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(save_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "timestamp_s", "plate", "det_conf", "x1", "y1", "x2", "y2"])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # ----------------------------------------------------------------
            # Gate 1 — Nth-frame sampler
            # ----------------------------------------------------------------
            if frame_id % nth != 0:
                if writer:
                    writer.write(frame)
                continue

            # ----------------------------------------------------------------
            # Gate 2 — Motion check
            # ----------------------------------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray).mean()
                if diff < motion_thresh:
                    prev_gray = gray
                    if writer:
                        writer.write(frame)
                    continue
            prev_gray = gray

            # ----------------------------------------------------------------
            # Gate 3 — YOLO detection
            # ----------------------------------------------------------------
            yolo_results = detector(frame, conf=conf, iou=iou, verbose=False)
            boxes = yolo_results[0].boxes

            det_list = []
            plate_texts = []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_conf = float(box.conf[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                processed = preprocess_plate(crop)
                plate_text = reader.read(processed, min_conf=OCR_MIN_CONF)

                det_list.append((x1, y1, x2, y2, det_conf))
                plate_texts.append(plate_text)

                # Log if not in cooldown
                key = plate_text or f"_box_{x1}_{y1}"
                if cooldown[key] <= 0 and plate_text:
                    timestamp = frame_id / fps_in
                    record = {
                        "frame": frame_id,
                        "timestamp_s": round(timestamp, 2),
                        "plate": plate_text,
                        "det_conf": round(det_conf, 3),
                        "bbox": (x1, y1, x2, y2),
                    }
                    all_detections.append(record)
                    print(f"  [{frame_id:06d}]  t={timestamp:.1f}s  {plate_text}  conf={det_conf:.2f}")
                    cooldown[key] = COOLDOWN_FRAMES

                    if csv_writer:
                        csv_writer.writerow([
                            frame_id, round(timestamp, 2), plate_text,
                            round(det_conf, 3), x1, y1, x2, y2
                        ])

                # Decrement all cooldowns
                for k in list(cooldown.keys()):
                    cooldown[k] = max(0, cooldown[k] - 1)

            # ----------------------------------------------------------------
            # FPS calculation
            # ----------------------------------------------------------------
            now = time.perf_counter()
            fps_display = 1.0 / max(now - fps_timer, 1e-6)
            fps_timer = now

            # ----------------------------------------------------------------
            # Annotate and write
            # ----------------------------------------------------------------
            annotated = draw_detections(frame, det_list, plate_texts)
            annotated = add_fps_overlay(annotated, fps_display)

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("Indian ANPR", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("User quit.")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_id} frames.")
    print(f"Unique plate events logged: {len(all_detections)}")
    if output:
        print(f"Output video: {output}")
    if save_csv:
        print(f"CSV log: {save_csv}")

    return all_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian ANPR — video")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument("--iou", type=float, default=IOU_THRESH)
    parser.add_argument("--nth", type=int, default=NTH_FRAME,
                        help="Process every Nth frame (default: 3)")
    parser.add_argument("--motion-thresh", type=float, default=MOTION_THRESH)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default=None, help="Save annotated video here")
    parser.add_argument("--csv", default=None, dest="save_csv",
                        help="Save detection log as CSV")
    args = parser.parse_args()

    process_video(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        nth=args.nth,
        motion_thresh=args.motion_thresh,
        show=args.show,
        output=args.output,
        save_csv=args.save_csv,
    )
