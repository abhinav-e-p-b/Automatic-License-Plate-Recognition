"""
detect_webcam.py — Real-time ANPR from webcam or IP camera stream.

Usage:
  python detect_webcam.py                         # Default webcam (index 0)
  python detect_webcam.py --source 1              # Second webcam
  python detect_webcam.py --source rtsp://...     # IP camera RTSP stream
  python detect_webcam.py --source http://...     # IP camera HTTP stream

Controls (OpenCV window):
  q — quit
  s — save current frame + annotated result to outputs/
  r — reset plate log
"""

import argparse
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils.preprocess import preprocess_plate
from utils.ocr import PlateReader
from utils.visualise import draw_detections, add_fps_overlay

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "models/best.pt"
CONF_THRESH = 0.50
IOU_THRESH = 0.45
OCR_MIN_CONF = 0.30
NTH_FRAME = 2
MOTION_THRESH = 12
COOLDOWN_FRAMES = 45
LOG_MAXLEN = 10      # Max plates shown in on-screen log


def run_webcam(
    source=0,
    model_path: str = DEFAULT_MODEL,
    conf: float = CONF_THRESH,
    iou: float = IOU_THRESH,
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {source}")

    # Prefer 720p for balance of quality and speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = YOLO(model_path)
    reader = PlateReader(gpu=True)

    frame_id = 0
    prev_gray = None
    cooldown: dict = defaultdict(int)
    plate_log: deque = deque(maxlen=LOG_MAXLEN)
    fps_timer = time.perf_counter()
    fps_display = 0.0

    print("Webcam ANPR running. Press 'q' to quit, 's' to save frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame_id += 1

        # Gate 1 — Nth frame
        if frame_id % NTH_FRAME != 0:
            cv2.imshow("Indian ANPR — Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Gate 2 — Motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray).mean()
            if diff < MOTION_THRESH:
                prev_gray = gray
                cv2.imshow("Indian ANPR — Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
        prev_gray = gray

        # Gate 3 — YOLO + OCR
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

            key = plate_text or f"_unk_{x1}_{y1}"
            if cooldown[key] <= 0 and plate_text:
                ts = datetime.now().strftime("%H:%M:%S")
                plate_log.append(f"{ts}  {plate_text}  ({det_conf:.2f})")
                print(f"  {ts}  {plate_text}  conf={det_conf:.2f}")
                cooldown[key] = COOLDOWN_FRAMES

        for k in list(cooldown.keys()):
            cooldown[k] = max(0, cooldown[k] - 1)

        # FPS
        now = time.perf_counter()
        fps_display = 0.8 * fps_display + 0.2 * (1.0 / max(now - fps_timer, 1e-6))
        fps_timer = now

        # Draw
        annotated = draw_detections(frame, det_list, plate_texts)
        annotated = add_fps_overlay(annotated, fps_display)

        # On-screen plate log (bottom-left)
        for idx, entry in enumerate(reversed(plate_log)):
            y_pos = annotated.shape[0] - 14 - idx * 22
            if y_pos < 10:
                break
            cv2.putText(
                annotated, entry,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 100), 1, cv2.LINE_AA,
            )

        cv2.imshow("Indian ANPR — Webcam", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            Path("outputs").mkdir(exist_ok=True)
            cv2.imwrite(f"outputs/frame_{ts}.jpg", frame)
            cv2.imwrite(f"outputs/annotated_{ts}.jpg", annotated)
            print(f"Saved outputs/frame_{ts}.jpg and annotated version.")
        elif key == ord("r"):
            plate_log.clear()
            print("Plate log cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. {len(plate_log)} plates in final log.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian ANPR — Webcam / IP Camera")
    parser.add_argument("--source", default=0,
                        help="Camera index (0,1,...) or RTSP/HTTP URL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument("--iou", type=float, default=IOU_THRESH)
    args = parser.parse_args()

    # Allow integer source
    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    run_webcam(source, args.model, args.conf, args.iou)
