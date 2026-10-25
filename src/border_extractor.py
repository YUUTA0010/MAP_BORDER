"""
border_extractor.py - Extract India's political boundary from the reference image.

Creates a thin binary border mask, distance transform, and per-piece
descriptors (contour points, curvature, edge orientation, shape descriptors).
"""

import json
import cv2
import numpy as np
from scipy import ndimage

from config import GEOJSON_PATH


def create_reference_border_mask(ref_img: np.ndarray, geojson_path: str = None):
    """
    Generate a thin binary India border mask from the reference image.

    If a GeoJSON is available, render it at the reference image's resolution.
    Otherwise, extract the border from the reference image's edges.

    Args:
        ref_img: BGR reference image (already resized to working size).
        geojson_path: Path to india_boundary.geojson (optional).

    Returns:
        border_mask: uint8 binary mask, border pixels = 255.
        border_points: (N, 2) array of border point coordinates (x, y).
        filled_mask: uint8 filled India region mask (for overlap checks).
    """
    h, w = ref_img.shape[:2]

    if geojson_path is None:
        geojson_path = GEOJSON_PATH

    # Try GeoJSON-based rendering first
    if geojson_path and _geojson_exists(geojson_path):
        return _border_from_geojson(ref_img, geojson_path)

    # Fallback: extract border from image edges
    return _border_from_image(ref_img)


def _geojson_exists(path):
    """Check if GeoJSON file exists."""
    import os
    return os.path.exists(path)


def _border_from_geojson(ref_img, geojson_path):
    """
    Render India's GeoJSON boundary onto the reference image coordinate space.

    The reference image is assumed to show the India region.
    We map lon/lat to pixel coordinates based on the image extent.
    """
    h, w = ref_img.shape[:2]

    with open(geojson_path, "r") as f:
        data = json.load(f)

    geom = data["features"][0]["geometry"]
    coords = geom["coordinates"]

    # Collect all rings
    all_rings = []
    if geom["type"] == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                all_rings.append(np.array(ring, dtype=np.float64))
    elif geom["type"] == "Polygon":
        for ring in coords:
            all_rings.append(np.array(ring, dtype=np.float64))

    if not all_rings:
        return _border_from_image(ref_img)

    # Find geographic extent of all rings
    all_pts = np.vstack(all_rings)
    lon_min, lat_min = all_pts[:, 0].min(), all_pts[:, 1].min()
    lon_max, lat_max = all_pts[:, 0].max(), all_pts[:, 1].max()

    # Add small margin
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    margin = 0.02
    lon_min -= lon_range * margin
    lon_max += lon_range * margin
    lat_min -= lat_range * margin
    lat_max += lat_range * margin

    def lonlat_to_pixel(lon, lat):
        x = (lon - lon_min) / (lon_max - lon_min) * (w - 1)
        y = (1.0 - (lat - lat_min) / (lat_max - lat_min)) * (h - 1)
        return x, y

    # Render border mask (thin line)
    border_mask = np.zeros((h, w), dtype=np.uint8)
    filled_mask = np.zeros((h, w), dtype=np.uint8)

    pixel_rings = []
    for ring in all_rings:
        pixel_pts = []
        for lon, lat in ring[:, :2]:
            px, py = lonlat_to_pixel(lon, lat)
            pixel_pts.append([int(round(px)), int(round(py))])
        pixel_pts = np.array(pixel_pts, dtype=np.int32)
        pixel_rings.append(pixel_pts)

        pts_draw = pixel_pts.reshape(-1, 1, 2)
        cv2.polylines(border_mask, [pts_draw], isClosed=True, color=255, thickness=1)
        cv2.fillPoly(filled_mask, [pts_draw], color=255)

    # Collect all border points
    border_ys, border_xs = np.where(border_mask > 0)
    border_points = np.column_stack([border_xs, border_ys]).astype(np.float64)

    return border_mask, border_points, filled_mask


def _border_from_image(ref_img):
    """
    Extract border from the reference image using edge detection.
    Used when no GeoJSON is available.
    """
    h, w = ref_img.shape[:2]
    if len(ref_img.shape) == 3:
        gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = ref_img.astype(np.uint8)

    # Threshold to get India silhouette
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Edge = border
    border_mask = cv2.Canny(binary, 50, 150)

    # Filled mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(filled_mask, contours, -1, 255, cv2.FILLED)

    border_ys, border_xs = np.where(border_mask > 0)
    border_points = np.column_stack([border_xs, border_ys]).astype(np.float64)

    return border_mask, border_points, filled_mask


def compute_distance_transform(border_mask: np.ndarray):
    """
    Compute distance transform from the border mask.

    Args:
        border_mask: uint8 binary mask (border = 255).

    Returns:
        dist_map: float32 distance transform (0 at border, increasing away).
    """
    inv = 255 - border_mask
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    return dist.astype(np.float32)


def extract_piece_descriptors(piece_img, piece_mask, piece_border):
    """
    Extract descriptors for one piece of the 3x3 grid.

    Args:
        piece_img:    BGR image piece.
        piece_mask:   Land mask piece.
        piece_border: Border mask piece.

    Returns:
        dict with:
            'contours':     list of (N,2) contour arrays
            'curvature':    list of curvature arrays per contour
            'orientations': edge orientation histogram (36 bins)
            'hu_moments':   Hu moments of the land mask
            'has_border':   bool, whether this piece contains border pixels
            'border_count': number of border pixels
            'land_fraction': fraction of piece that is land
    """
    h, w = piece_img.shape[:2]
    area = max(1, h * w)

    # Contours from land mask
    contours_raw, _ = cv2.findContours(
        piece_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [c.reshape(-1, 2).astype(np.float64) for c in contours_raw if len(c) >= 10]

    # Curvature per contour
    curvatures = []
    for cnt in contours:
        curv = _compute_curvature(cnt)
        curvatures.append(curv)

    # Edge orientation histogram
    gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY) if len(piece_img.shape) == 3 else piece_img
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx) * 180 / np.pi  # -180 to 180
    magnitudes = np.sqrt(gx ** 2 + gy ** 2)
    # Weighted histogram of orientations
    orient_hist, _ = np.histogram(
        angles[magnitudes > 10], bins=36, range=(-180, 180),
        weights=magnitudes[magnitudes > 10]
    )
    orient_hist = orient_hist.astype(np.float64)
    norm = orient_hist.sum()
    if norm > 0:
        orient_hist /= norm

    # Hu moments
    moments = cv2.moments(piece_mask)
    hu = cv2.HuMoments(moments).flatten()

    # Border info
    border_count = int(np.count_nonzero(piece_border))
    land_fraction = float(np.count_nonzero(piece_mask) / area)

    return {
        "contours": contours,
        "curvature": curvatures,
        "orientations": orient_hist,
        "hu_moments": hu,
        "has_border": border_count > 5,
        "border_count": border_count,
        "land_fraction": land_fraction,
    }


def _compute_curvature(contour, k=5):
    """
    Compute curvature at each point of a contour.

    Args:
        contour: (N, 2) array of contour points.
        k: neighborhood size for finite differences.

    Returns:
        curvature: (N,) array of curvature values.
    """
    n = len(contour)
    if n < 2 * k + 1:
        return np.zeros(n, dtype=np.float64)

    curvature = np.zeros(n, dtype=np.float64)
    for i in range(n):
        prev = contour[(i - k) % n]
        curr = contour[i]
        nxt = contour[(i + k) % n]

        v1 = curr - prev
        v2 = nxt - curr

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        norm1 = np.linalg.norm(v1) + 1e-8
        norm2 = np.linalg.norm(v2) + 1e-8

        curvature[i] = cross / (norm1 * norm2)

    return curvature

