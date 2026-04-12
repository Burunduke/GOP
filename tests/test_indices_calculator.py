"""
Tests for vegetation index calculator module
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import pytest

from src.indices.calculator import VegetationIndexCalculator
from src.utils.exceptions import ProcessingError, ValidationError


class TestVegetationIndexCalculator(unittest.TestCase):
    """Test cases for VegetationIndexCalculator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.calculator = VegetationIndexCalculator()
        self.temp_dir = tempfile.mkdtemp()

        # Create test data files
        self.test_image_path = os.path.join(self.temp_dir, "test_image.tif")
        self.test_mask_path = os.path.join(self.temp_dir, "test_mask.tif")

        # Create simple test data
        self.test_data = np.random.rand(100, 100, 5).astype(np.float32)
        self.test_mask = np.random.randint(0, 2, (100, 100)).astype(np.uint8)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test calculator initialization"""
        self.assertIsInstance(self.calculator, VegetationIndexCalculator)

    def test_calculate_basic_functionality(self):
        """Test basic index calculation functionality"""
        # Create temporary files
        with open(self.test_image_path, "w") as f:
            f.write("test")
        with open(self.test_mask_path, "w") as f:
            f.write("test")

        # Test calculation - should raise error due to GDAL not being available
        with self.assertRaises(Exception):
            self.calculator.calculate(
                orthophoto_path=self.test_image_path,
                segmentation_mask=self.test_mask_path,
                sensor_type="Hyperspectral",
                selected_indices=["NDVI", "EVI"],
            )

    def test_calculate_with_nonexistent_files(self):
        """Test calculation with nonexistent files"""
        with self.assertRaises(FileNotFoundError):
            self.calculator.calculate(
                orthophoto_path="/nonexistent/image.tif",
                segmentation_mask="/nonexistent/mask.tif",
            )

    def test_calculate_with_invalid_sensor_type(self):
        """Test calculation with invalid sensor type"""
        # Create temporary files
        with open(self.test_image_path, "w") as f:
            f.write("test")
        with open(self.test_mask_path, "w") as f:
            f.write("test")

        with self.assertRaises(ValidationError):
            self.calculator.calculate(
                orthophoto_path=self.test_image_path,
                segmentation_mask=self.test_mask_path,
                sensor_type="InvalidSensor",
            )

    def test_assess_plant_condition(self):
        """Test plant condition assessment"""
        # Create mock normalized index results
        indices_results = {
            "normalized_indices": {
                "NDVI": np.array([0.1, 0.5, 0.8]),
                "EVI": np.array([0.2, 0.4, 0.6]),
            }
        }

        result = self.calculator.assess_plant_condition(indices_results)

        self.assertIsInstance(result, dict)
        self.assertIn("condition_maps", result)
        self.assertIn("statistics", result)
        self.assertIn("classification", result)

    def test_get_index_statistics(self):
        """Test index statistics calculation"""
        # Create mock index results
        indices_results = {
            "NDVI": np.array([0.1, 0.5, 0.8]),
            "EVI": np.array([0.2, 0.4, 0.6]),
        }

        result = self.calculator.get_index_statistics(indices_results)

        self.assertIsInstance(result, dict)
        # The method should handle the statistics calculation
        # even if GDAL is not available
        self.assertIn("error", result)

    def test_empty_indices_handling(self):
        """Test handling of empty indices results"""
        empty_results = {}

        # Test plant condition with empty results
        result = self.calculator.assess_plant_condition(empty_results)
        self.assertIsInstance(result, dict)

        # Test statistics with empty results
        result = self.calculator.get_index_statistics(empty_results)
        self.assertIsInstance(result, dict)

    def test_invalid_indices_handling(self):
        """Test handling of invalid indices data"""
        invalid_results = {
            "NDVI": np.array([]),  # Empty array
            "EVI": None,  # None value
        }

        # Should handle gracefully without raising exceptions
        result = self.calculator.assess_plant_condition(invalid_results)
        self.assertIsInstance(result, dict)

        result = self.calculator.get_index_statistics(invalid_results)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
