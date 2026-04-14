"""
Tests for image utilities
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from typing import Any, List, Tuple
from unittest.mock import patch, MagicMock
from src.utils.image_utils import (
    load_image,
    save_image,
    resize_image,
    normalize_image,
    apply_colormap,
    blend_images,
    create_thumbnail,
    calculate_histogram,
    enhance_contrast,
    remove_noise,
)


class TestImageUtils(unittest.TestCase):
    """Test cases for image utilities"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        # Create test image
        self.test_image_path = os.path.join(self.temp_dir, "test_image.png")
        self.test_image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Save test image
        try:
            import cv2

            cv2.imwrite(self.test_image_path, self.test_image_array)
        except ImportError:
            # If cv2 is not available, create empty file
            with open(self.test_image_path, "wb") as f:
                f.write(b"\x00" * 1000)

    def tearDown(self) -> None:
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("cv2.imread")
    def test_load_image(self, mock_imread: MagicMock) -> None:
        """Test loading image"""
        # Mock setup
        mock_imread.return_value = self.test_image_array

        result = load_image(self.test_image_path)

        self.assertTrue(np.array_equal(result, self.test_image_array))
        mock_imread.assert_called_once_with(self.test_image_path, cv2.IMREAD_COLOR)

    @patch("cv2.imread")
    def test_load_image_grayscale(self, mock_imread: MagicMock) -> None:
        """Test loading image in grayscale"""
        # Mock setup
        grayscale_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mock_imread.return_value = grayscale_array

        result = load_image(self.test_image_path, mode="GRAY")

        self.assertTrue(np.array_equal(result, grayscale_array))
        mock_imread.assert_called_once_with(self.test_image_path, cv2.IMREAD_GRAYSCALE)

    @patch("cv2.imread")
    def test_load_image_file_not_found(self, mock_imread: MagicMock) -> None:
        """Test loading non-existent image"""
        mock_imread.return_value = None

        with self.assertRaises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")

    @patch("cv2.imwrite")
    def test_save_image(self, mock_imwrite: MagicMock) -> None:
        """Test saving image"""
        output_path = os.path.join(self.temp_dir, "output_image.png")
        mock_imwrite.return_value = True

        result = save_image(self.test_image_array, output_path)

        self.assertTrue(result)
        mock_imwrite.assert_called_once()

    @patch("cv2.imwrite")
    def test_save_image_failure(self, mock_imwrite: MagicMock) -> None:
        """Test handling save image error"""
        output_path = os.path.join(self.temp_dir, "output_image.png")
        mock_imwrite.return_value = False

        with self.assertRaises(Exception):
            save_image(self.test_image_array, output_path)

    @patch("cv2.resize")
    def test_resize_image(self, mock_resize: MagicMock) -> None:
        """Test resizing image"""
        # Mock setup
        resized_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        mock_resize.return_value = resized_array

        target_size = (50, 50)
        result = resize_image(self.test_image_array, target_size)

        self.assertEqual(
            result.shape[:2], target_size[::-1]
        )  # OpenCV uses (height, width)
        mock_resize.assert_called_once()

    @patch("cv2.resize")
    def test_resize_image_with_interpolation(self, mock_resize: MagicMock) -> None:
        """Test resizing image with interpolation"""
        # Mock setup
        resized_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        mock_resize.return_value = resized_array

        target_size = (50, 50)
        result = resize_image(
            self.test_image_array, target_size, interpolation=cv2.INTER_CUBIC
        )

        self.assertEqual(result.shape[:2], target_size[::-1])
        mock_resize.assert_called_once()

    def test_normalize_image_minmax(self) -> None:
        """Test image normalization using minmax method"""
        # Create image with known values
        test_image = np.array([[0, 128], [255, 64]], dtype=np.uint8)

        normalized = normalize_image(test_image, method="minmax")

        self.assertEqual(normalized.min(), 0.0)
        self.assertEqual(normalized.max(), 1.0)
        self.assertEqual(normalized[0, 0], 0.0)  # Minimum value
        self.assertEqual(normalized[0, 1], 0.5)  # Middle value
        self.assertEqual(normalized[1, 0], 1.0)  # Maximum value

    def test_normalize_image_zscore(self) -> None:
        """Test image normalization using z-score method"""
        # Create image with known values
        test_image = np.array([[0, 128], [255, 64]], dtype=np.uint8)

        normalized = normalize_image(test_image, method="zscore")

        # Check that mean is close to 0 and standard deviation to 1
        self.assertAlmostEqual(normalized.mean(), 0.0, places=5)
        self.assertAlmostEqual(normalized.std(), 1.0, places=5)

    def test_normalize_image_invalid_method(self) -> None:
        """Test image normalization with invalid method"""
        test_image = np.array([[0, 128], [255, 64]], dtype=np.uint8)

        with self.assertRaises(ValueError):
            normalize_image(test_image, method="invalid_method")

    @patch("cv2.applyColorMap")
    def test_apply_colormap(self, mock_apply_colormap: MagicMock) -> None:
        """Test applying colormap"""
        # Mock setup
        colored_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mock_apply_colormap.return_value = colored_image

        grayscale_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        result = apply_colormap(grayscale_image)

        self.assertTrue(np.array_equal(result, colored_image))
        mock_apply_colormap.assert_called_once()

    def test_blend_images(self) -> None:
        """Test blending images"""
        image1 = np.full((50, 50, 3), 255, dtype=np.uint8)  # White image
        image2 = np.zeros((50, 50, 3), dtype=np.uint8)  # Black image

        # Blending in equal proportions should give gray image
        blended = blend_images(image1, image2, alpha=0.5)

        expected_color = 128  # Approximately half of 255
        self.assertTrue(np.allclose(blended, expected_color, atol=1))

    def test_blend_images_different_sizes(self) -> None:
        """Test blending images of different sizes"""
        image1 = np.full((50, 50, 3), 255, dtype=np.uint8)
        image2 = np.full((100, 100, 3), 0, dtype=np.uint8)

        with self.assertRaises(ValueError):
            blend_images(image1, image2)

    def test_blend_images_invalid_alpha(self) -> None:
        """Test blending images with invalid alpha value"""
        image1 = np.full((50, 50, 3), 255, dtype=np.uint8)
        image2 = np.full((50, 50, 3), 0, dtype=np.uint8)

        with self.assertRaises(ValueError):
            blend_images(image1, image2, alpha=1.5)

        with self.assertRaises(ValueError):
            blend_images(image1, image2, alpha=-0.5)

    @patch("cv2.resize")
    @patch("cv2.imwrite")
    def test_create_thumbnail(self, mock_imwrite: MagicMock, mock_resize: MagicMock) -> None:
        """Test creating thumbnail"""
        # Mock setup
        thumbnail_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        mock_resize.return_value = thumbnail_array
        mock_imwrite.return_value = True

        thumbnail_path = os.path.join(self.temp_dir, "thumbnail.png")

        result = create_thumbnail(self.test_image_path, size=(256, 256))

        self.assertEqual(result, thumbnail_path)
        mock_resize.assert_called_once()
        mock_imwrite.assert_called_once()

    def test_calculate_histogram(self) -> None:
        """Test calculating histogram"""
        # Create image with known values
        test_image = np.array(
            [[[0, 0, 0], [255, 255, 255]], [[128, 128, 128], [64, 64, 64]]],
            dtype=np.uint8,
        )

        hist = calculate_histogram(test_image, bins=256)

        self.assertEqual(len(hist), 3)  # Three channels
        self.assertEqual(len(hist[0]), 256)  # 256 bins

        # Check that histogram contains expected values
        self.assertEqual(hist[0][0], 1)  # One pixel with value 0
        self.assertEqual(hist[0][64], 1)  # One pixel with value 64
        self.assertEqual(hist[0][128], 1)  # One pixel with value 128
        self.assertEqual(hist[0][255], 1)  # One pixel with value 255

    def test_calculate_histogram_grayscale(self) -> None:
        """Test calculating histogram for grayscale"""
        # Create grayscale image
        test_image = np.array([[0, 255], [128, 64]], dtype=np.uint8)

        hist = calculate_histogram(test_image, bins=256)

        self.assertEqual(len(hist), 1)  # One channel
        self.assertEqual(len(hist[0]), 256)  # 256 bins

        # Check values
        self.assertEqual(hist[0][0], 1)  # One pixel with value 0
        self.assertEqual(hist[0][64], 1)  # One pixel with value 64
        self.assertEqual(hist[0][128], 1)  # One pixel with value 128
        self.assertEqual(hist[0][255], 1)  # One pixel with value 255

    @patch("cv2.equalizeHist")
    def test_enhance_contrast_histogram_equalization(self, mock_equalize_hist: MagicMock) -> None:
        """Test contrast enhancement using histogram equalization"""
        # Mock setup
        enhanced_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mock_equalize_hist.return_value = enhanced_image

        test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        result = enhance_contrast(test_image, method="histogram_eq")

        self.assertTrue(np.array_equal(result, enhanced_image))
        mock_equalize_hist.assert_called_once()

    @patch("cv2.createCLAHE")
    def test_enhance_contrast_clahe(self, mock_create_clahe: MagicMock) -> None:
        """Test contrast enhancement using CLAHE"""
        # Mock setup
        mock_clahe = MagicMock()
        enhanced_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mock_clahe.apply.return_value = enhanced_image
        mock_create_clahe.return_value = mock_clahe

        test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        result = enhance_contrast(test_image, method="clahe")

        self.assertTrue(np.array_equal(result, enhanced_image))
        mock_create_clahe.assert_called_once()
        mock_clahe.apply.assert_called_once()

    def test_enhance_contrast_invalid_method(self) -> None:
        """Test contrast enhancement with invalid method"""
        test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        with self.assertRaises(ValueError):
            enhance_contrast(test_image, method="invalid_method")

    @patch("cv2.GaussianBlur")
    def test_remove_noise_gaussian(self, mock_gaussian_blur: MagicMock) -> None:
        """Test noise removal using Gaussian method"""
        # Mock setup
        denoised_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mock_gaussian_blur.return_value = denoised_image

        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = remove_noise(test_image, method="gaussian")

        self.assertTrue(np.array_equal(result, denoised_image))
        mock_gaussian_blur.assert_called_once()

    @patch("cv2.medianBlur")
    def test_remove_noise_median(self, mock_median_blur: MagicMock) -> None:
        """Test noise removal using median filter"""
        # Mock setup
        denoised_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mock_median_blur.return_value = denoised_image

        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = remove_noise(test_image, method="median")

        self.assertTrue(np.array_equal(result, denoised_image))
        mock_median_blur.assert_called_once()

    def test_remove_noise_invalid_method(self) -> None:
        """Test noise removal with invalid method"""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            remove_noise(test_image, method="invalid_method")


if __name__ == "__main__":
    unittest.main()
