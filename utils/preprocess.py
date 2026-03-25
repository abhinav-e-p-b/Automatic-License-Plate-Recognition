"""
utils/preprocess.py — Image preprocessing for Indian number plate crops.

preprocess_plate(crop) returns a dict of named image variants.
Each caller iterates the dict, runs OCR on each variant, and picks
the first one that produces a valid plate reading.

Variants (ordered best-first):
  'gray'      — CLAHE-enhanced grayscale   ← best for clean well-lit plates
  'sharp'     — sharpened grayscale        ← slightly blurry plates
  'boosted'   — gray + edge overlay        ← faint / faded characters
  'otsu'      — Otsu binarisation          ← strong contrast fallback
  'otsu_inv'  — Otsu inverted              ← IR camera / dark-bg plates
  'adap'      — adaptive threshold         ← uneven lighting
  'adap_inv'  — adaptive inverted          ← uneven + dark-background
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual step functions — importable by diagnose.py
# ---------------------------------------------------------------------------

def upscale(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Scale image up by `scale` using bicubic interpolation."""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_CUBIC)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale. No-op if already single-channel."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bilateral_denoise(gray: np.ndarray, d: int = 11, sigma: float = 17) -> np.ndarray:
    """Bilateral filter — smooth noise, preserve character edges."""
    return cv2.bilateralFilter(gray, d, sigma, sigma)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp-mask kernel — boost character stroke edges."""
    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]])
    return cv2.filter2D(img, -1, kernel)


def adaptive_threshold(gray: np.ndarray, block_size: int = 31, c: int = 5) -> np.ndarray:
    """Adaptive Gaussian threshold — handles uneven lighting."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )


def adaptive_threshold_inv(gray: np.ndarray, block_size: int = 31, c: int = 5) -> np.ndarray:
    """Adaptive threshold inverted — dark-background plates."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c,
    )


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Otsu automatic binarisation."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def otsu_threshold_inv(gray: np.ndarray) -> np.ndarray:
    """Otsu binarisation inverted."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def morph_clean(binary: np.ndarray, k: int = 2) -> np.ndarray:
    """Morphological opening — remove tiny noise blobs."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_plate(crop: np.ndarray) -> dict:
    """
    Run the full preprocessing pipeline on a plate crop.

    Args:
        crop: BGR or grayscale image (any size — upscaled internally).

    Returns:
        dict of {variant_name: image_array}, ordered best-first.
        Pass each image to EasyOCR and stop at the first valid plate reading.
    """
    up   = upscale(crop, 2.0)
    gray = to_gray(up)

    clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    sharp   = sharpen(gray_enhanced)
    edges   = cv2.Canny(sharp, 50, 150)
    boosted = cv2.addWeighted(sharp, 0.8, edges, 0.2, 0)

    return {
        "gray":     gray_enhanced,
        "sharp":    sharp,
        "boosted":  boosted,
        "otsu":     otsu_threshold(sharp),
        "otsu_inv": otsu_threshold_inv(sharp),
        "adap":     adaptive_threshold(sharp, 31, 5),
        "adap_inv": adaptive_threshold_inv(sharp, 31, 5),
    }
