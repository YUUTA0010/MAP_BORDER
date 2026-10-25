"""
piece_comparator.py - Cross-compare 9 satellite pieces against 9 reference pieces.

Builds a 9x9 similarity matrix, finds matching regions, and extracts
corresponding border points using contour descriptors + feature matching.
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from config import (
    ORB_N_FEATURES, AKAZE_THRESH, MATCH_RATIO,
    PIECE_SIMILARITY_THRESH,
)
from border_extractor import extract_piece_descriptors, compute_distance_transform


def divide_into_9(img, mask=None, border_mask=None):
    """
    Divide an image (and optional masks) into a 3x3 grid.

    Returns:
        list of 9 dicts, each with 'image', 'mask', 'border', 'bbox', 'index'.
    """
    h, w = img.shape[:2]
    ys = [0, h // 3, (2 * h) // 3, h]
    xs = [0, w // 3, (2 * w) // 3, w]

    pieces = []
    idx = 0
    for r in range(3):
        for c in range(3):
            y1, y2 = ys[r], ys[r + 1]
            x1, x2 = xs[c], xs[c + 1]

            piece = {
                "index": idx,
                "row": r,
                "col": c,
                "bbox": (y1, y2, x1, x2),
                "image": img[y1:y2, x1:x2].copy(),
            }
            if mask is not None:
                piece["mask"] = mask[y1:y2, x1:x2].copy()
            else:
                piece["mask"] = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

            if border_mask is not None:
                piece["border"] = border_mask[y1:y2, x1:x2].copy()
            else:
                piece["border"] = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

            idx += 1
            pieces.append(piece)

    return pieces


def build_similarity_matrix(sat_pieces, ref_pieces, sat_descs, ref_descs):
    """
    Build a 9x9 similarity matrix comparing all satellite pieces to all reference pieces.

    Uses multiple signals:
    1. Hu Moments shape similarity
    2. Edge orientation histogram correlation
    3. Land-body overlap ratio
    4. Chamfer distance (boundary proximity)

    Args:
        sat_pieces: list of 9 satellite piece dicts.
        ref_pieces: list of 9 reference piece dicts.
        sat_descs:  list of 9 satellite descriptor dicts.
        ref_descs:  list of 9 reference descriptor dicts.

    Returns:
        sim_matrix: (9, 9) float array, sim_matrix[i][j] = similarity(ref_i, sat_j).
    """
    sim = np.zeros((9, 9), dtype=np.float64)

    for i in range(9):
        rd = ref_descs[i]
        if not rd["has_border"] and rd["land_fraction"] < 0.01:
            # Reference piece has no border and no land - skip
            continue

        for j in range(9):
            sd = sat_descs[j]

            scores = []

            # 1. Hu Moments similarity (log-scale comparison)
            hu_sim = _hu_moment_similarity(rd["hu_moments"], sd["hu_moments"])
            scores.append(hu_sim)

            # 2. Edge orientation histogram correlation
            orient_sim = _histogram_correlation(rd["orientations"], sd["orientations"])
            scores.append(orient_sim)

            # 3. Land-body overlap ratio
            land_sim = 1.0 - abs(rd["land_fraction"] - sd["land_fraction"])
            scores.append(max(0.0, land_sim))

            # 4. Chamfer score (if both have borders/contours)
            chamfer_sim = 0.0
            if rd["has_border"] and len(sd["contours"]) > 0:
                chamfer_sim = _chamfer_similarity(
                    ref_pieces[i]["border"], sat_pieces[j]["mask"]
                )
            scores.append(chamfer_sim)

            sim[i, j] = float(np.mean(scores))

    return sim


def find_matching_pairs(sim_matrix, threshold=None):
    """
    From the 9x9 similarity matrix, find the best matching piece pairs.

    Args:
        sim_matrix: (9, 9) array.
        threshold: Minimum similarity to consider a match.

    Returns:
        list of (ref_idx, sat_idx, score) tuples, sorted by score descending.
    """
    if threshold is None:
        threshold = PIECE_SIMILARITY_THRESH

    pairs = []
    for i in range(9):
        for j in range(9):
            if sim_matrix[i, j] >= threshold:
                pairs.append((i, j, float(sim_matrix[i, j])))

    # Sort by score descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def find_corresponding_points(sat_pieces, ref_pieces, matching_pairs, ref_border_points):
    """
    For matched piece pairs, find corresponding border points using feature matching.

    Uses ORB + AKAZE feature detection on edge images, plus contour shape matching.

    Args:
        sat_pieces:       list of 9 satellite piece dicts.
        ref_pieces:       list of 9 reference piece dicts.
        matching_pairs:   list of (ref_idx, sat_idx, score).
        ref_border_points: (N, 2) array of all reference border points in working coords.

    Returns:
        ref_pts: (M, 2) array of matched reference points.
        sat_pts: (M, 2) array of matched satellite points.
        confidences: (M,) array of match confidence values.
    """
    all_ref_pts = []
    all_sat_pts = []
    all_confidences = []

    for ref_idx, sat_idx, pair_score in matching_pairs:
        ref_piece = ref_pieces[ref_idx]
        sat_piece = sat_pieces[sat_idx]

        ref_bbox = ref_piece["bbox"]
        sat_bbox = sat_piece["bbox"]

        # Get edge images for feature matching
        ref_edge = _prepare_edge_image(ref_piece["image"], ref_piece["border"])
        sat_edge = _prepare_edge_image(sat_piece["image"], sat_piece["mask"])

        # ORB matching
        orb_ref, orb_sat, orb_conf = _match_features_orb(ref_edge, sat_edge)
        # AKAZE matching
        aka_ref, aka_sat, aka_conf = _match_features_akaze(
            ref_piece["image"], sat_piece["image"]
        )

        # Contour shape matching
        cnt_ref, cnt_sat, cnt_conf = _match_contour_points(
            ref_piece["border"], sat_piece["mask"]
        )

        # Convert piece-local coordinates to full working-image coordinates
        for pts_r, pts_s, confs in [
            (orb_ref, orb_sat, orb_conf),
            (aka_ref, aka_sat, aka_conf),
            (cnt_ref, cnt_sat, cnt_conf),
        ]:
            if len(pts_r) != len(pts_s) or len(pts_r) != len(confs):
                continue
            for k in range(len(pts_r)):
                # Offset by piece bbox origin
                rp = pts_r[k].copy()
                sp = pts_s[k].copy()
                rp[0] += ref_bbox[2]  # x offset
                rp[1] += ref_bbox[0]  # y offset
                sp[0] += sat_bbox[2]
                sp[1] += sat_bbox[0]

                all_ref_pts.append(rp)
                all_sat_pts.append(sp)
                all_confidences.append(float(confs[k]) * float(pair_score))

    if not all_ref_pts:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)

    return (
        np.array(all_ref_pts, dtype=np.float64),
        np.array(all_sat_pts, dtype=np.float64),
        np.array(all_confidences, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hu_moment_similarity(hu1, hu2):
    """Compare two Hu moment vectors using log-scale matching."""
    eps = 1e-10
    log1 = np.sign(hu1) * np.log10(np.abs(hu1) + eps)
    log2 = np.sign(hu2) * np.log10(np.abs(hu2) + eps)
    diff = np.abs(log1 - log2)
    # Normalize to [0, 1] range (lower diff = higher similarity)
    score = np.exp(-np.mean(diff))
    return float(np.clip(score, 0, 1))


def _histogram_correlation(h1, h2):
    """Correlation between two normalized histograms."""
    if h1.sum() < 1e-8 or h2.sum() < 1e-8:
        return 0.0
    # Normalize
    h1n = h1 / (h1.sum() + 1e-8)
    h2n = h2 / (h2.sum() + 1e-8)
    corr = np.corrcoef(h1n, h2n)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(max(0.0, corr))


def _chamfer_similarity(border_mask, sat_land_mask):
    """
    Compute Chamfer-based similarity between reference border and satellite land edges.
    """
    if np.count_nonzero(border_mask) < 5 or np.count_nonzero(sat_land_mask) < 5:
        return 0.0

    # Get satellite land edges
    sat_edges = cv2.Canny(sat_land_mask, 50, 150)
    if np.count_nonzero(sat_edges) < 5:
        return 0.0

    # Distance transform from satellite edges
    dist = cv2.distanceTransform(255 - sat_edges, cv2.DIST_L2, 5)

    # Sample border points
    by, bx = np.where(border_mask > 0)
    if len(by) == 0:
        return 0.0

    # Clamp to bounds
    bx = np.clip(bx, 0, dist.shape[1] - 1)
    by = np.clip(by, 0, dist.shape[0] - 1)

    # Average distance of border points to nearest satellite edge
    distances = dist[by, bx]
    avg_dist = np.mean(distances)
    max_dim = max(border_mask.shape)

    # Convert to similarity (0 to 1)
    sim = np.exp(-avg_dist / (max_dim * 0.1 + 1e-6))
    return float(np.clip(sim, 0, 1))


def _prepare_edge_image(img, mask):
    """Create an edge image suitable for feature matching."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
    edges = cv2.Canny(blurred, 30, 100)

    if mask is not None and np.count_nonzero(mask) > 0:
        dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, dilated)

    return edges


def _match_features_orb(img1, img2):
    """Match ORB features between two images."""
    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        return [], [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return [], [], []

    ref_pts = []
    sat_pts = []
    confidences = []

    for m_pair in matches:
        if len(m_pair) < 2:
            continue
        m, n = m_pair
        if m.distance < MATCH_RATIO * n.distance:
            p1 = np.array(kp1[m.queryIdx].pt, dtype=np.float64)
            p2 = np.array(kp2[m.trainIdx].pt, dtype=np.float64)
            conf = 1.0 - (m.distance / (n.distance + 1e-6))
            ref_pts.append(p1)
            sat_pts.append(p2)
            confidences.append(float(conf))

    return ref_pts, sat_pts, confidences


def _match_features_akaze(img1, img2):
    """Match AKAZE features between two images. Falls back to ORB if AKAZE unavailable."""
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1
    if len(img2.shape) == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g2 = img2

    # Try AKAZE, fall back to ORB
    try:
        detector = cv2.AKAZE_create(threshold=AKAZE_THRESH)
        norm_type = cv2.NORM_HAMMING
    except AttributeError:
        try:
            detector = cv2.SIFT_create(nfeatures=ORB_N_FEATURES)
            norm_type = cv2.NORM_L2
        except AttributeError:
            detector = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
            norm_type = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(g1, None)
    kp2, des2 = detector.detectAndCompute(g2, None)

    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        return [], [], []

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return [], [], []

    ref_pts = []
    sat_pts = []
    confidences = []

    for m_pair in matches:
        if len(m_pair) < 2:
            continue
        m, n = m_pair
        if m.distance < MATCH_RATIO * n.distance:
            p1 = np.array(kp1[m.queryIdx].pt, dtype=np.float64)
            p2 = np.array(kp2[m.trainIdx].pt, dtype=np.float64)
            conf = 1.0 - (m.distance / (n.distance + 1e-6))
            ref_pts.append(p1)
            sat_pts.append(p2)
            confidences.append(float(conf))

    return ref_pts, sat_pts, confidences


def _match_contour_points(ref_border_piece, sat_mask_piece):
    """
    Match contour sample points using shape context / curvature matching.
    """
    # Extract contours from reference border
    ref_contours, _ = cv2.findContours(
        ref_border_piece.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    # Extract contours from satellite land mask
    sat_contours, _ = cv2.findContours(
        sat_mask_piece.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not ref_contours or not sat_contours:
        return [], [], []

    # Get the largest contour from each
    ref_cnt = max(ref_contours, key=len).reshape(-1, 2).astype(np.float64)
    sat_cnt = max(sat_contours, key=len).reshape(-1, 2).astype(np.float64)

    if len(ref_cnt) < 10 or len(sat_cnt) < 10:
        return [], [], []

    # Subsample contours
    n_sample = min(50, len(ref_cnt), len(sat_cnt))
    ref_idx = np.linspace(0, len(ref_cnt) - 1, n_sample, dtype=int)
    sat_idx = np.linspace(0, len(sat_cnt) - 1, n_sample, dtype=int)
    ref_sub = ref_cnt[ref_idx]
    sat_sub = sat_cnt[sat_idx]

    # Compute curvature at sample points
    ref_curv = _curvature_at_points(ref_cnt, ref_idx)
    sat_curv = _curvature_at_points(sat_cnt, sat_idx)

    # Match by curvature similarity
    curv_dist = cdist(ref_curv.reshape(-1, 1), sat_curv.reshape(-1, 1), metric="euclidean")

    ref_pts = []
    sat_pts = []
    confidences = []

    used_sat = set()
    for i in range(n_sample):
        best_j = -1
        best_d = float("inf")
        for j in range(n_sample):
            if j in used_sat:
                continue
            if curv_dist[i, j] < best_d:
                best_d = curv_dist[i, j]
                best_j = j

        if best_j >= 0 and best_d < 0.5:
            used_sat.add(best_j)
            ref_pts.append(ref_sub[i])
            sat_pts.append(sat_sub[best_j])
            conf = max(0.1, 1.0 - best_d)
            confidences.append(conf)

    return ref_pts, sat_pts, confidences


def _curvature_at_points(contour, indices, k=5):
    """Compute curvature at specified indices of a contour."""
    n = len(contour)
    curvatures = np.zeros(len(indices), dtype=np.float64)
    for idx_pos, i in enumerate(indices):
        prev = contour[(i - k) % n]
        curr = contour[i]
        nxt = contour[(i + k) % n]
        v1 = curr - prev
        v2 = nxt - curr
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        norm1 = np.linalg.norm(v1) + 1e-8
        norm2 = np.linalg.norm(v2) + 1e-8
        curvatures[idx_pos] = cross / (norm1 * norm2)
    return curvatures
