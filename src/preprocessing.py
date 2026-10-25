"""
preprocessing.py — Image loading, normalization, CLAHE enhancement.
"""

import cv2
import numpy as np


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Normalize loaded images to 3-channel BGR regardless of source dtype."""
    img = np.asarray(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return _ensure_bgr(img)


def to_gray_enhanced(img: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale and apply CLAHE for better feature detection
    in low-contrast satellite imagery.
    """
    gray = cv2.cvtColor(_ensure_bgr(img), cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def resize_for_matching(img: np.ndarray, scale: float) -> np.ndarray:
    """Downscale image for faster feature matching."""
    if scale >= 1.0:
        return img.copy()
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
