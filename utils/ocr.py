"""
utils/ocr.py — OCR module for Indian number plates.

PlateReader.read() accepts two input types:
  - numpy array  → used as-is (single image passed to EasyOCR)
  - dict         → output of preprocess_plate(); iterates all variants
                   and returns the first valid plate found

This means no other file needs to change: detect_video.py,
detect_webcam.py, detect_batch.py all call reader.read(processed)
and it works correctly whether processed is an array or a dict.
"""

import re
from typing import Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Indian state / UT codes (as of 2024)
# ---------------------------------------------------------------------------
VALID_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD",
    "DL", "DN", "GA", "GJ", "HR", "HP", "JK", "JH",
    "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML",
    "MZ", "NL", "OD", "PY", "PB", "RJ", "SK", "TN",
    "TS", "TR", "UP", "UK", "WB",
}

# Standard plate: SS DD LL DDDD  or  SS DD L DDDD
STANDARD_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")

# BH series: YY BH DDDD LL
BH_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Position-aware substitution (0-indexed, spaces stripped)
LETTER_POS   = {0, 1, 4, 5}
DIGIT_POS    = {2, 3, 6, 7, 8, 9}
LETTER_FIXES = {"0": "O", "1": "I", "l": "I", "8": "B", "5": "S"}
DIGIT_FIXES  = {"O": "0", "I": "1", "l": "1", "B": "8", "S": "5"}


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def fix_characters(raw: str) -> str:
    """Position-aware character substitution for Indian plate format."""
    chars = list(raw)
    for i, ch in enumerate(chars):
        if i in LETTER_POS and ch in LETTER_FIXES:
            chars[i] = LETTER_FIXES[ch]
        elif i in DIGIT_POS and ch in DIGIT_FIXES:
            chars[i] = DIGIT_FIXES[ch]
    return "".join(chars)


def normalise_raw(raw: str) -> str:
    """Strip spaces/hyphens/dots and uppercase."""
    return raw.upper().replace(" ", "").replace("-", "").replace(".", "")


def validate_plate(plate: str) -> Optional[str]:
    """Return plate if it matches a valid Indian format, else None."""
    if len(plate) < 8:
        return None
    if BH_PATTERN.match(plate):
        return plate
    if plate[:2] not in VALID_STATES:
        return None
    if STANDARD_PATTERN.match(plate):
        return plate
    return None


def merge_multiline(results: list) -> str:
    """Sort EasyOCR boxes top-to-bottom and join text (two-wheeler plates)."""
    return "".join(r[1] for r in sorted(results, key=lambda r: r[0][0][1]))


# ---------------------------------------------------------------------------
# Internal: run OCR on a single numpy image and return (plate, conf) or (None, 0)
# ---------------------------------------------------------------------------

def _ocr_single(
    img: np.ndarray,
    reader,          # easyocr.Reader instance
    min_conf: float,
) -> tuple:
    """
    Run EasyOCR on one image array.
    Returns (validated_plate_or_None, avg_confidence).
    """
    if not isinstance(img, np.ndarray):
        return None, 0.0

    results = reader.readtext(img, detail=1, paragraph=False)
    results = [r for r in results if r[2] >= min_conf]
    if not results:
        return None, 0.0

    raw      = merge_multiline(results)
    raw      = normalise_raw(raw)
    avg_conf = sum(r[2] for r in results) / len(results)

    if len(raw) < 6:
        return None, 0.0

    fixed = fix_characters(raw)
    plate = validate_plate(fixed)
    return plate, avg_conf


# ---------------------------------------------------------------------------
# PlateReader
# ---------------------------------------------------------------------------

class PlateReader:
    """
    Wrapper around EasyOCR with full Indian plate post-processing.

    Accepts both raw crops AND preprocess_plate() dict output:

        reader = PlateReader(gpu=False)

        # Option A — pass a raw crop (single numpy array)
        plate = reader.read(crop)

        # Option B — pass the dict from preprocess_plate()
        from utils.preprocess import preprocess_plate
        plate = reader.read(preprocess_plate(crop))
    """

    def __init__(self, gpu: bool = True, languages: list = None):
        import easyocr
        langs = languages or ["en"]
        self.reader = easyocr.Reader(langs, gpu=gpu)

    def read(
        self,
        crop_or_variants: Union[np.ndarray, dict],
        min_conf: float = 0.15,
        detail: bool    = False,
    ) -> Optional[str]:
        """
        Run OCR and return a validated plate string, or None.

        Args:
            crop_or_variants: Either a numpy image array, OR the dict
                              returned by preprocess_plate(). When a dict
                              is passed, every variant is tried in order
                              and the first valid plate is returned.
            min_conf:         Minimum per-character EasyOCR confidence.
            detail:           If True, return (plate, raw_text, conf) tuple.

        Returns:
            Validated plate string (e.g. "KL07BB1234"), or None.
        """
        # ------------------------------------------------------------------
        # Build list of images to try
        # ------------------------------------------------------------------
        if isinstance(crop_or_variants, dict):
            # preprocess_plate() output — iterate all variants
            images = list(crop_or_variants.values())
        elif isinstance(crop_or_variants, np.ndarray):
            # Single image — wrap in list
            images = [crop_or_variants]
        else:
            return None

        # ------------------------------------------------------------------
        # Try each image until we get a valid plate
        # ------------------------------------------------------------------
        best_plate = None
        best_conf  = 0.0

        for img in images:
            plate, conf = _ocr_single(img, self.reader, min_conf)
            if plate:
                best_plate = plate
                best_conf  = conf
                break       # stop at first valid reading

        if detail:
            return best_plate, best_conf
        return best_plate


# ---------------------------------------------------------------------------
# Convenience one-shot function
# ---------------------------------------------------------------------------

def read_plate(
    crop_or_variants: Union[np.ndarray, dict],
    gpu: bool       = True,
    min_conf: float = 0.15,
) -> Optional[str]:
    """
    One-shot helper. For repeated use, keep a PlateReader instance.
    """
    reader = PlateReader(gpu=gpu)
    return reader.read(crop_or_variants, min_conf=min_conf)
