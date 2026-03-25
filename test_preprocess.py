"""
tests/test_preprocess.py — Unit tests for image preprocessing functions.

Run with:
  python -m pytest tests/test_preprocess.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from utils.preprocess import (
    upscale,
    to_gray,
    bilateral_denoise,
    adaptive_threshold,
    morphological_clean,
    invert_if_dark,
    preprocess_plate,
)


def make_bgr(h: int = 60, w: int = 200) -> np.ndarray:
    """Create a synthetic plate-like BGR image with dark text on light bg."""
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    # Draw some dark horizontal bars to simulate text
    img[20:40, 20:180] = 30
    return img


def make_dark_bgr(h: int = 60, w: int = 200) -> np.ndarray:
    """Dark background (IR/night plate simulation)."""
    img = np.full((h, w, 3), 20, dtype=np.uint8)
    img[15:45, 10:190] = 240   # bright text on dark bg
    return img


# ---------------------------------------------------------------------------
# upscale
# ---------------------------------------------------------------------------

class TestUpscale:
    def test_doubles_dimensions(self):
        img = make_bgr(60, 200)
        out = upscale(img, scale=2.0)
        assert out.shape == (120, 400, 3)

    def test_custom_scale(self):
        img = make_bgr(60, 200)
        out = upscale(img, scale=1.5)
        assert out.shape == (90, 300, 3)

    def test_preserves_channels(self):
        img = make_bgr()
        out = upscale(img)
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_dtype_preserved(self):
        img = make_bgr()
        out = upscale(img)
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# to_gray
# ---------------------------------------------------------------------------

class TestToGray:
    def test_bgr_to_gray(self):
        img = make_bgr()
        gray = to_gray(img)
        assert gray.ndim == 2

    def test_already_gray_passthrough(self):
        img = make_bgr()
        gray = to_gray(img)
        result = to_gray(gray)   # should not error
        assert result.ndim == 2

    def test_output_values_in_range(self):
        img = make_bgr()
        gray = to_gray(img)
        assert gray.min() >= 0
        assert gray.max() <= 255


# ---------------------------------------------------------------------------
# bilateral_denoise
# ---------------------------------------------------------------------------

class TestBilateralDenoise:
    def test_output_shape_unchanged(self):
        gray = to_gray(make_bgr())
        out = bilateral_denoise(gray)
        assert out.shape == gray.shape

    def test_output_dtype_uint8(self):
        gray = to_gray(make_bgr())
        out = bilateral_denoise(gray)
        assert out.dtype == np.uint8

    def test_noisy_image_smoothed(self):
        """After denoising, variance in a flat region should decrease."""
        img = np.full((60, 200), 200, dtype=np.uint8)
        noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        denoised = bilateral_denoise(noisy)
        assert np.std(denoised.astype(float)) < np.std(noisy.astype(float))


# ---------------------------------------------------------------------------
# adaptive_threshold
# ---------------------------------------------------------------------------

class TestAdaptiveThreshold:
    def test_output_is_binary(self):
        gray = to_gray(make_bgr())
        denoised = bilateral_denoise(gray)
        binary = adaptive_threshold(denoised)
        unique = np.unique(binary)
        assert set(unique).issubset({0, 255})

    def test_output_shape_unchanged(self):
        gray = to_gray(make_bgr())
        denoised = bilateral_denoise(gray)
        binary = adaptive_threshold(denoised)
        assert binary.shape == gray.shape


# ---------------------------------------------------------------------------
# morphological_clean
# ---------------------------------------------------------------------------

class TestMorphologicalClean:
    def test_output_is_still_binary(self):
        gray = to_gray(make_bgr())
        binary = adaptive_threshold(bilateral_denoise(gray))
        cleaned = morphological_clean(binary)
        unique = np.unique(cleaned)
        assert set(unique).issubset({0, 255})

    def test_removes_single_pixel_noise(self):
        """A single bright pixel surrounded by black should be removed."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[25, 25] = 255   # isolated noise pixel
        cleaned = morphological_clean(img, kernel_size=3)
        # After opening, the isolated pixel should be gone
        assert cleaned[25, 25] == 0


# ---------------------------------------------------------------------------
# invert_if_dark
# ---------------------------------------------------------------------------

class TestInvertIfDark:
    def test_dark_image_gets_inverted(self):
        dark = np.full((50, 200), 10, dtype=np.uint8)
        result = invert_if_dark(dark, threshold=127.0)
        assert result.mean() > 127

    def test_light_image_not_inverted(self):
        light = np.full((50, 200), 240, dtype=np.uint8)
        result = invert_if_dark(light, threshold=127.0)
        assert result.mean() > 127  # still light

    def test_borderline_threshold(self):
        mid = np.full((50, 200), 128, dtype=np.uint8)
        result = invert_if_dark(mid, threshold=127.0)
        # 128 >= 127, should not invert
        assert result.mean() >= 127


# ---------------------------------------------------------------------------
# preprocess_plate (full pipeline)
# ---------------------------------------------------------------------------

class TestPreprocessPlate:
    def test_output_is_2d_binary(self):
        img = make_bgr()
        result = preprocess_plate(img)
        assert result.ndim == 2
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_output_larger_than_input(self):
        """Upscaling should make the output larger than input."""
        img = make_bgr(60, 200)
        result = preprocess_plate(img, scale=2.0)
        assert result.shape[0] > 60
        assert result.shape[1] > 200

    def test_dark_plate_handled(self):
        """Night plate (dark bg) should be inverted to light bg."""
        img = make_dark_bgr()
        result = preprocess_plate(img, fix_night=True)
        # After inversion, majority of background should be white (255)
        assert result.mean() > 100

    def test_accepts_grayscale_input(self):
        gray = to_gray(make_bgr())
        result = preprocess_plate(gray)
        assert result.ndim == 2

    def test_small_crop_handled(self):
        """Very small crops (5×20) should not crash."""
        small = np.full((5, 20, 3), 200, dtype=np.uint8)
        result = preprocess_plate(small)
        assert result is not None
