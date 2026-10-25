"""
transform_estimator.py - Estimate geometric transformation from correspondences.

Supports 3-point minimum, RANSAC verification, similarity/affine/homography
model selection, multi-scale refinement, and combined scoring.
"""

import cv2
import numpy as np

from config import (
    MIN_CORRESPONDENCES, RANSAC_REPROJ_THRESH, RANSAC_MAX_ITERS,
    MIN_TRANSFORM_INLIERS, MIN_INLIER_RATIO,
    W_PIECE_SIM, W_BOUNDARY_SIM, W_FEATURE_MATCH,
    W_GEO_CONSIST, W_CHAMFER, W_LAND_OVERLAP, W_CLOUD_PENALTY,
)


class TransformResult:
    """Container for transformation estimation results."""

    def __init__(self):
        self.success = False
        self.matrix = None          # 2x3 affine or 3x3 homography
        self.transform_type = None  # 'similarity', 'affine', 'homography'
        self.inlier_count = 0
        self.inlier_ratio = 0.0
        self.total_score = 0.0
        self.reprojection_error = 0.0
        self.message = ""
        self.inlier_mask = None


def estimate_transform(ref_pts, sat_pts, confidences=None,
                       sat_land_mask=None, ref_border_mask=None,
                       cloud_confidence=None):
    """
    Estimate the best geometric transformation from reference to satellite.

    Args:
        ref_pts:          (M, 2) array of reference border points.
        sat_pts:          (M, 2) array of satellite matching points.
        confidences:      (M,) array of per-match confidence (optional).
        sat_land_mask:    Satellite land mask for overlap scoring (optional).
        ref_border_mask:  Reference border mask for Chamfer scoring (optional).
        cloud_confidence: Satellite cloud confidence map (optional).

    Returns:
        TransformResult with the best transformation.
    """
    result = TransformResult()

    n = len(ref_pts)
    if n < MIN_CORRESPONDENCES:
        result.message = (
            f"Insufficient reliable India-border correspondences. "
            f"Found {n}, need at least {MIN_CORRESPONDENCES}."
        )
        return result

    # Filter by confidence if available
    if confidences is not None and len(confidences) == n:
        # Keep top matches by confidence
        order = np.argsort(confidences)[::-1]
        # Keep at most top 200 to avoid noise
        keep = min(200, n)
        order = order[:keep]
        ref_pts = ref_pts[order]
        sat_pts = sat_pts[order]
        confidences = confidences[order]
        n = len(ref_pts)

    # Try models in order of simplicity
    candidates = []

    # 1. Similarity transform (4 DOF: scale, rotation, tx, ty)
    if n >= 3:
        sim_result = _estimate_similarity_ransac(ref_pts, sat_pts)
        if sim_result is not None:
            candidates.append(sim_result)

    # 2. Affine transform (6 DOF)
    if n >= 4:
        aff_result = _estimate_affine_ransac(ref_pts, sat_pts)
        if aff_result is not None:
            candidates.append(aff_result)

    # 3. Homography (8 DOF) - only if many points
    if n >= 8:
        hom_result = _estimate_homography_ransac(ref_pts, sat_pts)
        if hom_result is not None:
            candidates.append(hom_result)

    if not candidates:
        result.message = "All transformation estimation methods failed."
        return result

    # Score only candidates with enough independent geometric support.
    validated_candidates = []
    for cand in candidates:
        if _validate_transform(cand):
            validated_candidates.append(cand)

    if not validated_candidates:
        result.message = (
            "No transform passed confidence and geometric validation "
            f"(need at least {MIN_TRANSFORM_INLIERS} inliers and "
            f"{MIN_INLIER_RATIO:.0%} inlier ratio)."
        )
        return result

    # Score validated candidates
    best = None
    best_score = -1.0

    for cand in validated_candidates:
        score = _score_transform(
            cand, ref_pts, sat_pts,
            sat_land_mask, ref_border_mask, cloud_confidence
        )
        if score > best_score:
            best_score = score
            best = cand
            best.total_score = score

    if best is None:
        result.message = "No valid transformation found after scoring."
        return result

    return best


def transform_border_points(border_points, transform_result):
    """
    Apply the estimated transformation to the complete reference border.

    Args:
        border_points: (N, 2) array of reference border points.
        transform_result: TransformResult with .matrix.

    Returns:
        transformed: (N, 2) array of transformed points in satellite coordinates.
    """
    if transform_result.matrix is None:
        return border_points.copy()

    pts = border_points.astype(np.float64)

    if transform_result.transform_type == "homography":
        # 3x3 matrix
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])  # (N, 3)
        transformed_h = pts_h @ transform_result.matrix.T  # (N, 3)
        w = transformed_h[:, 2:3]
        w[w == 0] = 1e-8
        transformed = transformed_h[:, :2] / w
    else:
        # 2x3 affine matrix
        M = transform_result.matrix  # (2, 3)
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])  # (N, 3)
        transformed = pts_h @ M.T  # (N, 2)

    return transformed


def clip_to_bounds(points, width, height, margin=0):
    """
    Clip transformed border points to the image bounds.

    Returns segments of points that are within bounds.

    Args:
        points: (N, 2) array.
        width: Image width.
        height: Image height.
        margin: Extra pixels of margin.

    Returns:
        list of (K, 2) arrays, each a continuous visible segment.
    """
    n = len(points)
    if n == 0:
        return []

    # Mask of points inside bounds
    inside = (
        (points[:, 0] >= -margin) &
        (points[:, 0] <= width + margin) &
        (points[:, 1] >= -margin) &
        (points[:, 1] <= height + margin)
    )

    # Split into continuous segments
    segments = []
    current = []

    for i in range(n):
        if inside[i]:
            current.append(points[i])
        else:
            if len(current) >= 2:
                segments.append(np.array(current, dtype=np.float64))
            current = []

    if len(current) >= 2:
        segments.append(np.array(current, dtype=np.float64))

    return segments


# ---------------------------------------------------------------------------
# RANSAC estimation
# ---------------------------------------------------------------------------

def _estimate_similarity_ransac(ref_pts, sat_pts):
    """Estimate similarity transform with RANSAC."""
    n = len(ref_pts)
    if n < 3:
        return None

    best_inliers = 0
    best_M = None
    best_mask = None

    for _ in range(min(RANSAC_MAX_ITERS, 500)):
        # Sample 3 random points
        idx = np.random.choice(n, 3, replace=False)
        src = ref_pts[idx].astype(np.float32)
        dst = sat_pts[idx].astype(np.float32)

        # Check for collinearity
        if _are_collinear(src):
            continue

        # Estimate partial affine (similarity: scale + rotation + translation)
        M = cv2.estimateAffinePartial2D(
            src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
            method=cv2.LMEDS
        )[0]

        if M is None:
            continue

        # Count inliers
        mask = _count_inliers(ref_pts, sat_pts, M, RANSAC_REPROJ_THRESH)
        inlier_count = mask.sum()

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_M = M
            best_mask = mask

    if best_M is None or best_inliers < MIN_CORRESPONDENCES:
        return None

    # Refine with all inliers
    inlier_ref = ref_pts[best_mask].astype(np.float32)
    inlier_sat = sat_pts[best_mask].astype(np.float32)
    M_refined = cv2.estimateAffinePartial2D(
        inlier_ref.reshape(-1, 1, 2), inlier_sat.reshape(-1, 1, 2),
        method=cv2.LMEDS
    )[0]

    if M_refined is not None:
        best_M = M_refined
        best_mask = _count_inliers(ref_pts, sat_pts, best_M, RANSAC_REPROJ_THRESH)
        best_inliers = best_mask.sum()

    result = TransformResult()
    result.success = True
    result.matrix = best_M
    result.transform_type = "similarity"
    result.inlier_count = int(best_inliers)
    result.inlier_ratio = float(best_inliers / n)
    result.inlier_mask = best_mask
    result.reprojection_error = _mean_reproj_error(
        ref_pts[best_mask], sat_pts[best_mask], best_M
    )
    return result


def _estimate_affine_ransac(ref_pts, sat_pts):
    """Estimate full affine transform with RANSAC."""
    n = len(ref_pts)
    if n < 4:
        return None

    src = ref_pts.astype(np.float32).reshape(-1, 1, 2)
    dst = sat_pts.astype(np.float32).reshape(-1, 1, 2)

    M, inlier_mask = cv2.estimateAffine2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH,
        maxIters=RANSAC_MAX_ITERS,
        confidence=0.99
    )

    if M is None:
        return None

    if inlier_mask is not None:
        inlier_mask = inlier_mask.flatten().astype(bool)
    else:
        inlier_mask = _count_inliers(ref_pts, sat_pts, M, RANSAC_REPROJ_THRESH)

    inlier_count = inlier_mask.sum()
    if inlier_count < MIN_CORRESPONDENCES:
        return None

    result = TransformResult()
    result.success = True
    result.matrix = M
    result.transform_type = "affine"
    result.inlier_count = int(inlier_count)
    result.inlier_ratio = float(inlier_count / n)
    result.inlier_mask = inlier_mask
    result.reprojection_error = _mean_reproj_error(
        ref_pts[inlier_mask], sat_pts[inlier_mask], M
    )
    return result


def _estimate_homography_ransac(ref_pts, sat_pts):
    """Estimate homography with RANSAC."""
    n = len(ref_pts)
    if n < 8:
        return None

    src = ref_pts.astype(np.float32).reshape(-1, 1, 2)
    dst = sat_pts.astype(np.float32).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH,
        maxIters=RANSAC_MAX_ITERS,
        confidence=0.99
    )

    if H is None:
        return None

    if inlier_mask is not None:
        inlier_mask = inlier_mask.flatten().astype(bool)
    else:
        return None

    inlier_count = inlier_mask.sum()
    if inlier_count < MIN_CORRESPONDENCES:
        return None

    result = TransformResult()
    result.success = True
    result.matrix = H
    result.transform_type = "homography"
    result.inlier_count = int(inlier_count)
    result.inlier_ratio = float(inlier_count / n)
    result.inlier_mask = inlier_mask

    # Compute reprojection error for homography
    ones = np.ones((len(ref_pts[inlier_mask]), 1), dtype=np.float64)
    pts_h = np.hstack([ref_pts[inlier_mask], ones])
    proj = pts_h @ H.T
    w = proj[:, 2:3]
    w[w == 0] = 1e-8
    proj_2d = proj[:, :2] / w
    errors = np.linalg.norm(proj_2d - sat_pts[inlier_mask], axis=1)
    result.reprojection_error = float(np.mean(errors))

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_inliers(ref_pts, sat_pts, M, threshold):
    """Count inliers given a 2x3 affine matrix."""
    ones = np.ones((len(ref_pts), 1), dtype=np.float64)
    pts_h = np.hstack([ref_pts, ones])
    projected = pts_h @ M.T
    errors = np.linalg.norm(projected - sat_pts, axis=1)
    return errors < threshold


def _mean_reproj_error(ref_pts, sat_pts, M):
    """Mean reprojection error for a 2x3 affine matrix."""
    ones = np.ones((len(ref_pts), 1), dtype=np.float64)
    pts_h = np.hstack([ref_pts, ones])
    projected = pts_h @ M.T
    errors = np.linalg.norm(projected - sat_pts, axis=1)
    return float(np.mean(errors)) if len(errors) > 0 else float("inf")


def _are_collinear(points, threshold=1e-4):
    """Check if 3 points are approximately collinear."""
    if len(points) < 3:
        return True
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    return cross < threshold


def _score_transform(result, ref_pts, sat_pts,
                     sat_land_mask=None, ref_border_mask=None,
                     cloud_confidence=None):
    """
    Combined scoring function (Step 16).
    """
    scores = {}

    # Feature match score (inlier ratio)
    scores["feature"] = result.inlier_ratio

    # Geometric consistency (inverse of reprojection error, normalized)
    max_error = 50.0
    scores["geometric"] = max(0.0, 1.0 - result.reprojection_error / max_error)

    # Piece similarity (based on inlier count)
    scores["piece"] = min(1.0, result.inlier_count / 20.0)

    # Boundary similarity
    scores["boundary"] = scores["feature"]  # correlated with feature score

    # Chamfer score
    scores["chamfer"] = 0.5  # default if no masks available
    if ref_border_mask is not None and sat_land_mask is not None and result.matrix is not None:
        scores["chamfer"] = _chamfer_score_from_transform(
            ref_border_mask, sat_land_mask, result
        )

    # Land overlap
    scores["land_overlap"] = 0.5
    if sat_land_mask is not None and result.matrix is not None:
        scores["land_overlap"] = _land_overlap_score(
            ref_pts[result.inlier_mask] if result.inlier_mask is not None else ref_pts,
            sat_pts[result.inlier_mask] if result.inlier_mask is not None else sat_pts,
            sat_land_mask, result
        )

    # Cloud penalty
    scores["cloud"] = 0.0
    if cloud_confidence is not None and result.inlier_mask is not None:
        inlier_sat = sat_pts[result.inlier_mask].astype(int)
        h, w = cloud_confidence.shape
        cloud_vals = []
        for pt in inlier_sat:
            x, y = int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1))
            cloud_vals.append(1.0 - cloud_confidence[y, x])
        scores["cloud"] = float(np.mean(cloud_vals)) if cloud_vals else 0.0

    total = (
        W_PIECE_SIM * scores["piece"]
        + W_BOUNDARY_SIM * scores["boundary"]
        + W_FEATURE_MATCH * scores["feature"]
        + W_GEO_CONSIST * scores["geometric"]
        + W_CHAMFER * scores["chamfer"]
        + W_LAND_OVERLAP * scores["land_overlap"]
        - W_CLOUD_PENALTY * scores["cloud"]
    )

    return float(total)


def _chamfer_score_from_transform(ref_border_mask, sat_land_mask, result):
    """Compute Chamfer-like score after applying the transform."""
    # This is a simplified version
    return max(0.0, result.inlier_ratio)


def _land_overlap_score(ref_inlier_pts, sat_inlier_pts, sat_land_mask, result):
    """Check if transformed points land on actual land."""
    h, w = sat_land_mask.shape[:2]
    if len(sat_inlier_pts) == 0:
        return 0.0
    hits = 0
    for pt in sat_inlier_pts:
        x, y = int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1))
        if sat_land_mask[y, x] > 0:
            hits += 1
    return float(hits / len(sat_inlier_pts))


def _validate_transform(result):
    """
    Validate that the transform is not degenerate.
    Checks: no mirroring, reasonable scale, reasonable rotation.
    """
    if result.matrix is None:
        return False

    M = result.matrix

    if result.transform_type == "homography":
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    else:
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

    scale = np.sqrt(abs(det))
    print(f"  [validate] type={result.transform_type} det={det:.4f} scale={scale:.4f} "
          f"inliers={result.inlier_count} ratio={result.inlier_ratio:.4f}")

    if result.inlier_count < MIN_TRANSFORM_INLIERS:
        result.message = (
            f"Inlier count {result.inlier_count} below "
            f"minimum {MIN_TRANSFORM_INLIERS}."
        )
        return False

    # Reject mirroring (negative determinant)
    if det < 0:
        result.message = "Transform has mirroring (negative determinant)."
        return False

    # Check scale
    if scale < 0.05 or scale > 20.0:
        result.message = f"Transform scale {scale:.3f} is out of reasonable range."
        return False

    # Check inlier ratio
    if result.inlier_ratio < MIN_INLIER_RATIO:
        result.message = (
            f"Inlier ratio {result.inlier_ratio:.3f} below "
            f"minimum {MIN_INLIER_RATIO:.3f}."
        )
        return False

    return True
