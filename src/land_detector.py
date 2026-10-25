"""
land_detector.py - Robust land/water/cloud separation for satellite imagery.

Produces a binary land mask, a cloud mask, and a per-pixel confidence map.
Tolerates clouds, haze, shadows, vegetation, noise, and varying colors.
"""

import cv2
import numpy as np

from config import LAND_MIN_AREA, OCEAN_V_MAX, CLOUD_V_MIN, CLOUD_S_MAX


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Normalize image input to a BGR 3-channel array."""
    img = np.asarray(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def detect_land(img: np.ndarray):
    """
    Detect land regions in a satellite or reference image.

    Args:
        img: BGR image (np.ndarray).

    Returns:
        land_mask:      uint8 binary mask (0 or 255), land = 255.
        cloud_mask:     uint8 binary mask (0 or 255), cloud = 255.
        confidence_map: float32 map [0..1], 1.0 = clear land, 0.0 = invalid.
    """
    img = _ensure_bgr(img)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v_chan = hsv[:, :, 2]
    s_chan = hsv[:, :, 1]

    # --- Ocean: dark pixels ---
    ocean = (v_chan < OCEAN_V_MAX).astype(np.uint8)

    # --- Cloud: bright + low saturation ---
    cloud = ((v_chan > CLOUD_V_MIN) & (s_chan < CLOUD_S_MAX)).astype(np.uint8)

    # --- Gradient magnitude for texture ---
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_norm = np.clip(grad_mag / (grad_mag.max() + 1e-6), 0, 1).astype(np.float32)

    # Low-texture bright regions are more likely clouds
    low_texture_bright = ((grad_norm < 0.05) & (v_chan > 170) & (s_chan < 80)).astype(np.uint8)
    cloud = np.clip(cloud + low_texture_bright, 0, 1).astype(np.uint8)

    # --- Raw land = not ocean and not cloud ---
    land_raw = ((ocean == 0) & (cloud == 0)).astype(np.uint8) * 255

    # --- Morphological cleanup ---
    land = cv2.morphologyEx(land_raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # --- Connected components: remove small noise ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(land, 8)
    cleaned = np.zeros_like(land)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= LAND_MIN_AREA:
            cleaned[labels == i] = 255

    # --- Cloud mask output ---
    cloud_mask = (cloud * 255).astype(np.uint8)

    # --- Confidence map ---
    confidence = np.ones((h, w), dtype=np.float32)
    confidence[ocean > 0] = 0.0      # Ocean = no confidence
    confidence[cloud > 0] = 0.1      # Cloud = very low
    # Haze: moderate brightness, moderate saturation
    haze = ((v_chan > 140) & (v_chan < CLOUD_V_MIN) & (s_chan < 40)).astype(bool)
    confidence[haze] = 0.5
    # Clear land = full confidence (already 1.0)

    return cleaned, cloud_mask, confidence


def extract_land_contours(land_mask: np.ndarray, min_length: int = 30):
    """
    Extract contours of land regions from a binary land mask.

    Args:
        land_mask: uint8 binary mask (0 or 255).
        min_length: Minimum contour point count to keep.

    Returns:
        list of np.ndarray: Each is (N, 2) array of contour points (x, y).
    """
    contours, _ = cv2.findContours(land_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = []
    for cnt in contours:
        if len(cnt) >= min_length:
            pts = cnt.reshape(-1, 2)
            result.append(pts)
    return result


def extract_land_edges(img: np.ndarray, land_mask: np.ndarray):
    """
    Extract Canny edges restricted to land regions.

    Args:
        img: BGR image.
        land_mask: Binary land mask.

    Returns:
        edges: uint8 edge image.
    """
    img = _ensure_bgr(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 40, 120)
    # Restrict to land boundaries (dilate mask slightly to capture edge pixels)
    dilated_mask = cv2.dilate(land_mask, np.ones((7, 7), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, dilated_mask)
    return edges

