"""
config.py - Configuration for the India border registration system.
"""
import os
import cv2

# --- PATHS ---
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR      = os.path.join(BASE_DIR, "input")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
DEBUG_DIR      = os.path.join(BASE_DIR, "output", "debug")

REFERENCE_PATH = os.path.join(INPUT_DIR, "reference_india.png")
GEOJSON_PATH   = os.path.join(INPUT_DIR, "india_boundary.geojson")

FINAL_OUTPUT_NAME = "final_india_border.png"

# --- WORKING RESOLUTION ---
WORK_SIZE = 900   # Both images resized to this for comparison

# --- BORDER RENDERING ---
BORDER_COLOR     = (0, 0, 255)   # Pure red in BGR
BORDER_THICKNESS = 2
LINE_TYPE        = cv2.LINE_AA

# --- LAND DETECTION ---
LAND_MIN_AREA    = 500    # Minimum connected-component area to count as land
OCEAN_V_MAX      = 120    # HSV Value below this = ocean
CLOUD_V_MIN      = 185    # HSV Value above this + low saturation = cloud
CLOUD_S_MAX      = 60

# --- MATCHING ---
MIN_CORRESPONDENCES     = 3      # Minimum reliable point pairs to attempt transform
RANSAC_REPROJ_THRESH    = 8.0    # Reprojection error threshold for RANSAC inliers
RANSAC_MAX_ITERS        = 2000
MIN_TRANSFORM_INLIERS   = 12
MIN_INLIER_RATIO        = 0.10   # Reject weak registrations rather than draw a speculative border
PIECE_SIMILARITY_THRESH = 0.15   # Minimum similarity score for a piece pair to be considered

# --- FEATURE DETECTION ---
ORB_N_FEATURES  = 2000
AKAZE_THRESH    = 0.001
MATCH_RATIO     = 0.75    # Lowe's ratio test threshold

# --- SCORING WEIGHTS (Step 16) ---
W_PIECE_SIM     = 0.15
W_BOUNDARY_SIM  = 0.25
W_FEATURE_MATCH = 0.20
W_GEO_CONSIST   = 0.15
W_CHAMFER       = 0.10
W_LAND_OVERLAP  = 0.10
W_CLOUD_PENALTY = 0.05
