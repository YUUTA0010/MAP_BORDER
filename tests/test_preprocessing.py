import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from land_detector import detect_land
from preprocessing import load_image
from transform_estimator import TransformResult, _validate_transform


class PreprocessingTests(unittest.TestCase):
    def test_load_image_accepts_grayscale_and_alpha(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gray_path = os.path.join(tmpdir, "gray.png")
            gray = np.full((16, 20), 128, dtype=np.uint8)
            cv2.imwrite(gray_path, gray)

            loaded = load_image(gray_path)
            self.assertEqual(loaded.shape, (16, 20, 3))
            self.assertEqual(loaded.dtype, np.uint8)
            self.assertEqual(int(loaded[0, 0, 0]), 128)

            alpha_path = os.path.join(tmpdir, "rgba.png")
            rgba = np.zeros((12, 14, 4), dtype=np.uint8)
            rgba[..., 0] = 255
            rgba[..., 1] = 127
            rgba[..., 2] = 64
            rgba[..., 3] = 200
            cv2.imwrite(alpha_path, rgba)

            loaded_rgba = load_image(alpha_path)
            self.assertEqual(loaded_rgba.shape, (12, 14, 3))
            self.assertEqual(loaded_rgba.dtype, np.uint8)

    def test_detect_land_accepts_grayscale_input(self):
        gray = np.full((32, 40), 200, dtype=np.uint8)
        land_mask, cloud_mask, confidence = detect_land(gray)

        self.assertEqual(land_mask.shape, (32, 40))
        self.assertEqual(cloud_mask.shape, (32, 40))
        self.assertEqual(confidence.shape, (32, 40))
        self.assertEqual(land_mask.dtype, np.uint8)
        self.assertEqual(cloud_mask.dtype, np.uint8)
        self.assertEqual(confidence.dtype, np.float32)

    def test_transform_validation_rejects_weak_registration(self):
        result = TransformResult()
        result.matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result.transform_type = "affine"
        result.inlier_count = 8
        result.inlier_ratio = 0.04

        self.assertFalse(_validate_transform(result))

    def test_transform_validation_accepts_well_supported_registration(self):
        result = TransformResult()
        result.matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result.transform_type = "affine"
        result.inlier_count = 30
        result.inlier_ratio = 0.25

        self.assertTrue(_validate_transform(result))


if __name__ == "__main__":
    unittest.main()
