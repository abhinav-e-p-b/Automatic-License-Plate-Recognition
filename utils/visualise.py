"""
Visualisation helpers — drawing bounding boxes and plate text onto frames.
"""

import cv2
import numpy as np

# Plate type colours (BGR)
PLATE_COLOURS = {
    "private":    (0, 200, 0),     # green
    "commercial": (0, 180, 255),   # amber
    "ev":         (180, 60, 0),    # teal (BGR)
    "unknown":    (200, 200, 200), # gray
}


def plate_type_from_text(plate: str) -> str:
    """Infer plate type from format (placeholder logic)."""
    if plate and len(plate) >= 4 and plate[2:4] == "BH":
        return "bh"
    return "unknown"


def draw_detections(
    frame: np.ndarray,
    detections: list,
    plate_texts: list = None,
    box_colour: tuple = (0, 200, 0),
    text_colour: tuple = (255, 255, 255),
    font_scale: float = 0.7,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw YOLO bounding boxes and optional plate text onto a frame.

    Args:
        frame:        BGR image.
        detections:   List of (x1, y1, x2, y2, conf) tuples.
        plate_texts:  Corresponding list of plate strings (or None per entry).
        box_colour:   BGR colour for bounding box.
        text_colour:  BGR colour for label text.
        font_scale:   OpenCV font scale.
        thickness:    Line thickness.

    Returns:
        Annotated frame (copy).
    """
    out = frame.copy()
    plate_texts = plate_texts or [None] * len(detections)

    for (x1, y1, x2, y2, conf), text in zip(detections, plate_texts):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), box_colour, thickness)

        # Label: plate text + confidence
        label = f"{text or '???'}  {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(y1 - 8, th + 8)

        # Background pill for readability
        cv2.rectangle(
            out,
            (x1, label_y - th - baseline - 4),
            (x1 + tw + 8, label_y + baseline - 2),
            box_colour,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 4, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_colour,
            thickness,
            cv2.LINE_AA,
        )

    return out


def draw_plate_result(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    plate_text: str,
    conf: float,
    colour: tuple = (0, 200, 0),
) -> np.ndarray:
    """Draw a single detection result — convenience wrapper."""
    return draw_detections(
        frame,
        [(x1, y1, x2, y2, conf)],
        [plate_text],
        box_colour=colour,
    )


def add_fps_overlay(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter in top-right corner."""
    out = frame.copy()
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = frame.shape[1] - tw - 12
    cv2.putText(
        out, text, (x, th + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
    )
    return out


def tile_images(images: list, cols: int = 4, target_h: int = 120) -> np.ndarray:
    """
    Tile a list of BGR crops into a grid for display.
    Useful for visualising a batch of plate detections.
    """
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_h / h
        resized.append(cv2.resize(img, (int(w * scale), target_h)))

    rows = []
    for i in range(0, len(resized), cols):
        row_imgs = resized[i : i + cols]
        # Pad last row if needed
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros((target_h, resized[0].shape[1], 3), dtype=np.uint8))
        rows.append(np.hstack(row_imgs))

    return np.vstack(rows) if rows else np.zeros((target_h, 100, 3), dtype=np.uint8)
