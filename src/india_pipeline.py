"""
india_pipeline.py - Top-level orchestrator for India border registration.

Implements the complete pipeline:
  Load -> Resize -> Land detect -> Border extract -> 9-piece divide ->
  Cross-compare -> Feature match -> RANSAC transform -> Clip -> Render -> Output
"""

import cv2
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    WORK_SIZE, OUTPUT_DIR, DEBUG_DIR, GEOJSON_PATH, REFERENCE_PATH,
    BORDER_COLOR, BORDER_THICKNESS, LINE_TYPE, FINAL_OUTPUT_NAME,
    MIN_CORRESPONDENCES,
)
from preprocessing import load_image
from land_detector import detect_land, extract_land_contours, extract_land_edges
from border_extractor import (
    create_reference_border_mask, compute_distance_transform,
    extract_piece_descriptors,
)
from piece_comparator import (
    divide_into_9, build_similarity_matrix,
    find_matching_pairs, find_corresponding_points,
)
from transform_estimator import (
    estimate_transform, transform_border_points,
)


def run(satellite_path: str, reference_path: str = None, debug: bool = False):
    """
    Run the complete India border registration pipeline.

    Args:
        satellite_path: Path to satellite image.
        reference_path: Path to reference India image (default: config).
        debug: If True, save intermediate debug images.

    Returns:
        output_path: Path to final_india_border.png (or None on failure).
        success: bool.
        message: Explanation string.
    """
    if reference_path is None:
        reference_path = REFERENCE_PATH

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    # ======================================================================
    # STEP 1: Load both images
    # ======================================================================
    print("[pipeline] Step 1: Loading images...")
    sat_original = load_image(satellite_path)
    sat_output_original = cv2.imread(satellite_path, cv2.IMREAD_UNCHANGED)
    if sat_output_original is None:
        raise FileNotFoundError(f"Cannot load image: {satellite_path}")
    ref_original = load_image(reference_path)

    sat_h, sat_w = sat_original.shape[:2]
    ref_h, ref_w = ref_original.shape[:2]
    print(f"  Satellite: {sat_w}x{sat_h}")
    print(f"  Reference: {ref_w}x{ref_h}")

    # ======================================================================
    # STEP 2: Resize both to common working dimensions
    # ======================================================================
    print("[pipeline] Step 2: Resizing to common working size...")
    sat_work, sat_scale_info = _resize_preserve_aspect(sat_original, WORK_SIZE)
    ref_work, ref_scale_info = _resize_preserve_aspect(ref_original, WORK_SIZE)

    work_h, work_w = sat_work.shape[:2]
    print(f"  Working size: {work_w}x{work_h}")

    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_sat_work.png"), sat_work)
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_ref_work.png"), ref_work)

    # ======================================================================
    # STEP 3: Land-body detection
    # ======================================================================
    print("[pipeline] Step 3: Detecting land bodies...")
    sat_land, sat_cloud, sat_confidence = detect_land(sat_work)
    ref_land, _, _ = detect_land(ref_work)

    sat_edges = extract_land_edges(sat_work, sat_land)

    print(f"  Satellite land pixels: {np.count_nonzero(sat_land)}")
    print(f"  Reference land pixels: {np.count_nonzero(ref_land)}")

    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, "02_sat_land_mask.png"), sat_land)
        cv2.imwrite(os.path.join(DEBUG_DIR, "02_ref_land_mask.png"), ref_land)
        cv2.imwrite(os.path.join(DEBUG_DIR, "02_sat_cloud_mask.png"), sat_cloud)
        cv2.imwrite(os.path.join(DEBUG_DIR, "02_sat_edges.png"), sat_edges)

    # ======================================================================
    # STEP 4: Reference India border mask
    # ======================================================================
    print("[pipeline] Step 4: Extracting reference India border...")
    ref_border_mask, ref_border_pts, ref_filled = create_reference_border_mask(
        ref_work, GEOJSON_PATH
    )
    ref_dist_map = compute_distance_transform(ref_border_mask)

    print(f"  Reference border points: {len(ref_border_pts)}")

    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, "03_ref_border_mask.png"), ref_border_mask)
        cv2.imwrite(os.path.join(DEBUG_DIR, "03_ref_filled.png"), ref_filled)
        dist_vis = (ref_dist_map / (ref_dist_map.max() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, "03_ref_distance_map.png"), dist_vis)

    if len(ref_border_pts) < 10:
        msg = "Failed: Reference border has too few points."
        print(f"[pipeline] {msg}")
        return None, False, msg

    # ======================================================================
    # STEP 5: Divide both into 9 pieces (3x3)
    # ======================================================================
    print("[pipeline] Step 5: Dividing into 3x3 grids...")
    sat_pieces = divide_into_9(sat_work, sat_land, None)
    ref_pieces = divide_into_9(ref_work, ref_land, ref_border_mask)

    if debug:
        for i, sp in enumerate(sat_pieces):
            cv2.imwrite(os.path.join(DEBUG_DIR, f"04_sat_piece_{i+1}.png"), sp["image"])
        for i, rp in enumerate(ref_pieces):
            cv2.imwrite(os.path.join(DEBUG_DIR, f"04_ref_piece_{i+1}.png"), rp["image"])

    # ======================================================================
    # STEP 6: Extract border descriptors per piece
    # ======================================================================
    print("[pipeline] Step 6: Extracting per-piece descriptors...")
    sat_descs = []
    ref_descs = []
    for i in range(9):
        sd = extract_piece_descriptors(
            sat_pieces[i]["image"], sat_pieces[i]["mask"], sat_pieces[i]["border"]
        )
        rd = extract_piece_descriptors(
            ref_pieces[i]["image"], ref_pieces[i]["mask"], ref_pieces[i]["border"]
        )
        sat_descs.append(sd)
        ref_descs.append(rd)

    # ======================================================================
    # STEP 7: Cross-compare all 9x9 piece pairs
    # ======================================================================
    print("[pipeline] Step 7: Building 9x9 similarity matrix...")
    sim_matrix = build_similarity_matrix(sat_pieces, ref_pieces, sat_descs, ref_descs)

    print("  Similarity matrix:")
    for i in range(9):
        row = " ".join(f"{sim_matrix[i, j]:.3f}" for j in range(9))
        print(f"    R{i+1}: [{row}]")

    if debug:
        _save_similarity_matrix_image(sim_matrix, DEBUG_DIR)

    # ======================================================================
    # STEP 8: Find matching piece pairs
    # ======================================================================
    print("[pipeline] Step 8: Finding matching piece pairs...")
    matching_pairs = find_matching_pairs(sim_matrix)
    print(f"  Found {len(matching_pairs)} candidate piece pairs")
    for ri, si, score in matching_pairs[:10]:
        print(f"    Ref piece {ri+1} <-> Sat piece {si+1}: score={score:.3f}")

    if not matching_pairs:
        msg = "Failed: No matching piece pairs found above threshold."
        print(f"[pipeline] {msg}")
        _save_failure(sat_output_original, msg)
        return None, False, msg

    # ======================================================================
    # STEP 9: Find corresponding border points
    # ======================================================================
    print("[pipeline] Step 9: Finding corresponding border points...")
    ref_match_pts, sat_match_pts, match_confs = find_corresponding_points(
        sat_pieces, ref_pieces, matching_pairs, ref_border_pts
    )
    print(f"  Found {len(ref_match_pts)} candidate correspondences")

    if len(ref_match_pts) < MIN_CORRESPONDENCES:
        msg = (
            f"Insufficient reliable India-border correspondences. "
            f"Found {len(ref_match_pts)}, need at least {MIN_CORRESPONDENCES}."
        )
        print(f"[pipeline] {msg}")
        _save_failure(sat_output_original, msg)
        return None, False, msg

    if debug:
        _save_matches_image(sat_work, ref_work, sat_match_pts, ref_match_pts, DEBUG_DIR)

    # ======================================================================
    # STEP 10-11: RANSAC + Estimate transformation
    # ======================================================================
    print("[pipeline] Step 10-11: Estimating transformation (RANSAC)...")
    transform_result = estimate_transform(
        ref_match_pts, sat_match_pts, match_confs,
        sat_land_mask=sat_land,
        ref_border_mask=ref_border_mask,
        cloud_confidence=sat_confidence,
    )

    if not transform_result.success:
        msg = f"Transform estimation failed: {transform_result.message}"
        print(f"[pipeline] {msg}")
        _save_failure(sat_output_original, msg)
        return None, False, msg

    print(f"  Transform type: {transform_result.transform_type}")
    print(f"  Inliers: {transform_result.inlier_count} ({transform_result.inlier_ratio:.1%})")
    print(f"  Reprojection error: {transform_result.reprojection_error:.2f}px")
    print(f"  Total score: {transform_result.total_score:.3f}")

    # ======================================================================
    # STEP 12-13: Transform ordered border contours and clip at draw time
    # ======================================================================
    print("[pipeline] Step 12-13: Transforming ordered border contours...")
    border_contours, _ = cv2.findContours(
        ref_border_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    ordered_border_paths = [
        contour.reshape(-1, 2).astype(np.float64)
        for contour in border_contours
        if len(contour) >= 2
    ]
    transformed_paths = [
        transform_border_points(path, transform_result)
        for path in ordered_border_paths
    ]
    print(f"  Transformed {len(transformed_paths)} ordered border paths")

    # ======================================================================
    # STEP 14-18: Map back to original resolution and draw border
    # ======================================================================
    print("[pipeline] Step 18: Drawing border on original satellite image...")
    if sat_output_original.ndim == 2:
        final_img = cv2.cvtColor(sat_output_original, cv2.COLOR_GRAY2BGR)
    else:
        final_img = sat_output_original.copy()
    overlay_mask = np.zeros((sat_h, sat_w), dtype=np.uint8)

    for path in transformed_paths:
        orig_pts = _working_to_original(path, sat_scale_info)
        if len(orig_pts) < 2 or not np.isfinite(orig_pts).all():
            continue
        if np.max(np.abs(orig_pts)) > 10_000_000:
            continue
        pts_int = np.rint(orig_pts).astype(np.int32)
        for index in range(len(pts_int)):
            start = tuple(pts_int[index])
            end = tuple(pts_int[(index + 1) % len(pts_int)])
            intersects, clipped_start, clipped_end = cv2.clipLine(
                (0, 0, sat_w, sat_h), start, end
            )
            if intersects:
                cv2.line(
                    overlay_mask, clipped_start, clipped_end,
                    color=255, thickness=BORDER_THICKNESS, lineType=LINE_TYPE,
                )

    visible_pixels = int(np.count_nonzero(overlay_mask))
    if visible_pixels == 0:
        msg = "Failed: No visible India border within satellite image bounds."
        print(f"[pipeline] {msg}")
        _save_failure(sat_original, msg)
        return None, False, msg

    if debug:
        debug_paths = [
            path for path in transformed_paths
            if len(path) >= 2 and np.isfinite(path).all()
        ]
        _save_transformed_border_debug(sat_work, debug_paths, DEBUG_DIR)

    if final_img.ndim == 3 and final_img.shape[2] == 4:
        final_img[overlay_mask > 0, :3] = BORDER_COLOR
    else:
        final_img[overlay_mask > 0] = BORDER_COLOR

    # ======================================================================
    # STEP 19: Save ONE final output
    # ======================================================================
    output_path = os.path.join(OUTPUT_DIR, FINAL_OUTPUT_NAME)
    cv2.imwrite(output_path, final_img)
    print(f"[pipeline] Final output saved: {output_path}")

    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, "99_final.png"), final_img)

    return output_path, True, "Success"


# ---------------------------------------------------------------------------
# Resize helpers
# ---------------------------------------------------------------------------

def _resize_preserve_aspect(img, target_size):
    """
    Resize image to fit within target_size x target_size, preserving aspect ratio.
    Pads with black to reach exact target_size x target_size.

    Returns:
        resized: Padded image of shape (target_size, target_size, C).
        scale_info: dict with 'scale', 'offset_x', 'offset_y', 'orig_h', 'orig_w'.
    """
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to target_size x target_size
    canvas = np.zeros((target_size, target_size, 3), dtype=img.dtype)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    return canvas, {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "orig_h": h,
        "orig_w": w,
        "new_w": new_w,
        "new_h": new_h,
    }


def _working_to_original(pts, scale_info):
    """
    Map points from working coordinates back to original image coordinates.
    """
    pts = pts.copy()
    pts[:, 0] = (pts[:, 0] - scale_info["offset_x"]) / scale_info["scale"]
    pts[:, 1] = (pts[:, 1] - scale_info["offset_y"]) / scale_info["scale"]
    # Clip to original bounds
    pts[:, 0] = np.clip(pts[:, 0], 0, scale_info["orig_w"] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, scale_info["orig_h"] - 1)
    return pts


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _save_failure(sat_output_original, msg):
    """Save unmodified satellite image when pipeline fails."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, FINAL_OUTPUT_NAME)
    cv2.imwrite(output_path, sat_output_original)
    # Save failure report
    report = {"status": "FAILED", "message": msg}
    with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)


def _save_similarity_matrix_image(sim_matrix, debug_dir):
    """Visualize the 9x9 similarity matrix as a heatmap."""
    norm = sim_matrix / (sim_matrix.max() + 1e-6) * 255
    vis = norm.astype(np.uint8)
    vis = cv2.resize(vis, (270, 270), interpolation=cv2.INTER_NEAREST)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(debug_dir, "05_similarity_matrix.png"), vis_color)


def _save_matches_image(sat_work, ref_work, sat_pts, ref_pts, debug_dir):
    """Save a side-by-side visualization of matched points."""
    h, w = sat_work.shape[:2]
    combined = np.hstack([ref_work, sat_work])
    n = min(len(ref_pts), len(sat_pts), 50)  # limit for clarity
    for i in range(n):
        rp = ref_pts[i].astype(int)
        sp = sat_pts[i].astype(int)
        sp_offset = sp.copy()
        sp_offset[0] += w
        color = (0, 255, 0)
        cv2.circle(combined, tuple(rp), 3, color, -1)
        cv2.circle(combined, tuple(sp_offset), 3, color, -1)
        cv2.line(combined, tuple(rp), tuple(sp_offset), color, 1)
    cv2.imwrite(os.path.join(debug_dir, "06_matches.png"), combined)


def _save_transformed_border_debug(sat_work, segments, debug_dir):
    """Save the transformed border overlaid on the working-size satellite image."""
    vis = sat_work.copy()
    for seg in segments:
        pts = seg.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=False,
                      color=BORDER_COLOR, thickness=BORDER_THICKNESS, lineType=LINE_TYPE)
    cv2.imwrite(os.path.join(debug_dir, "07_border_on_working.png"), vis)
