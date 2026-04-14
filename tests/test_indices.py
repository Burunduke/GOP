"""
Tests for vegetation indices calculation module
"""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from src.indices.calculator import VegetationIndexCalculator
from src.indices.definitions import IndexDefinitions


class TestIndexCalculator(unittest.TestCase):
    """Tests for vegetation index calculator"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.calculator = VegetationIndexCalculator()

        # Create test spectral data
        self.width, self.height, self.bands = 100, 100, 125
        self.spectral_data = (
            np.random.rand(self.height, self.width, self.bands) * 0.5 + 0.25
        )

        # Create test segmentation mask
        self.segmentation_mask = np.random.randint(0, 5, (self.height, self.width))

        # Temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calculate_single_index(self) -> None:
        """Test calculation of a single index"""
        # Calculate GNDVI
        result = self.calculator.calculate_from_arrays_simple(
            self.spectral_data, self.segmentation_mask, "Hyperspectral", ["GNDVI"]
        )

        self.assertIn("GNDVI", result)
        self.assertEqual(result["GNDVI"].shape, (self.height, self.width))
        self.assertFalse(np.all(np.isnan(result["GNDVI"])))

    def test_calculate_multiple_indices(self) -> None:
        """Test calculation of multiple indices"""
        indices = ["GNDVI", "NDWI", "MCARI"]
        result = self.calculator.calculate_from_arrays_simple(
            self.spectral_data, self.segmentation_mask, "Hyperspectral", indices
        )

        for index_name in indices:
            self.assertIn(index_name, result)
            self.assertEqual(result[index_name].shape, (self.height, self.width))
            self.assertFalse(np.all(np.isnan(result[index_name])))

    def test_calculate_all_indices(self) -> None:
        """Test calculation of all indices"""
        result = self.calculator.calculate_from_arrays_simple(
            self.spectral_data, self.segmentation_mask, "Hyperspectral", None  # All indices
        )

        # Check presence of all indices
        all_indices = IndexDefinitions.get_all_indices()
        for index_name in all_indices:
            self.assertIn(index_name, result)

    def test_assess_plant_condition(self) -> None:
        """Test plant condition assessment"""
        # Calculate indices using the main calculate method which returns structured results
        indices_result = self.calculator.calculate_from_arrays(
            self.spectral_data, self.segmentation_mask, "Hyperspectral", ["GNDVI", "NDWI", "MCARI"]
        )

        # Assess condition
        condition_result = self.calculator.assess_plant_condition(indices_result)

        self.assertIn("classification", condition_result)
        self.assertIn("indices", condition_result)

        classification = condition_result["classification"]
        self.assertIn("class", classification)
        self.assertIn("score", classification)
        self.assertGreaterEqual(classification["score"], 0)
        self.assertLessEqual(classification["score"], 1)

    def test_invalid_index(self) -> None:
        """Test handling of invalid index"""
        with self.assertRaises(Exception):  # Could be ValidationError or other exception
            self.calculator.calculate_from_arrays_simple(
                self.spectral_data, self.segmentation_mask, "Hyperspectral", ["INVALID_INDEX"]
            )

    def test_empty_segmentation_mask(self) -> None:
        """Test with empty segmentation mask"""
        empty_mask = np.zeros((self.height, self.width), dtype=int)

        result = self.calculator.calculate_from_arrays_simple(
            self.spectral_data, empty_mask, "Hyperspectral", ["GNDVI"]
        )

        self.assertIn("GNDVI", result)
        # Result should contain NaN values for empty areas
        self.assertTrue(np.all(np.isnan(result["GNDVI"][empty_mask == 0])))

    def test_save_and_load_results(self) -> None:
        """Test saving and loading results"""
        # Calculate indices
        result = self.calculator.calculate_from_arrays_simple(
            self.spectral_data, self.segmentation_mask, "Hyperspectral", ["GNDVI", "NDWI"]
        )

        # Save
        save_path = os.path.join(self.temp_dir, "test_results.json")
        self.calculator.save_results(result, save_path)
        self.assertTrue(os.path.exists(save_path))

        # Load
        loaded_result = self.calculator.load_results(save_path)

        # Check
        self.assertEqual(set(result.keys()), set(loaded_result.keys()))


class TestIndexDefinitions(unittest.TestCase):
    """Tests for vegetation index definitions"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.calculator = VegetationIndexCalculator()

        # Create test spectral data
        self.width, self.height, self.bands = 100, 100, 125
        self.spectral_data = (
            np.random.rand(self.height, self.width, self.bands) * 0.5 + 0.25
        )

        # Create test segmentation mask
        self.segmentation_mask = np.random.randint(0, 5, (self.height, self.width))

        # Temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_index_info(self) -> None:
        """Test getting index information"""
        info = IndexDefinitions.get_index_info("GNDVI")

        self.assertIn("name", info)
        self.assertIn("description", info)
        self.assertIn("formula", info)
        self.assertIn("required_bands", info)
        self.assertIn("category", info)

        self.assertEqual(info["name"], "GNDVI")
        self.assertIn("NIR", info["required_bands"])
        self.assertIn("Green", info["required_bands"])

    def test_get_indices_by_category(self) -> None:
        """Test getting indices by category"""
        greenness_indices = IndexDefinitions.get_indices_by_category("greenness")

        self.assertIsInstance(greenness_indices, list)
        self.assertGreater(len(greenness_indices), 0)
        self.assertIn("GNDVI", greenness_indices)

    def test_validate_index_requirements(self) -> None:
        """Test validation of index requirements"""
        # Test with sufficient bands
        available_bands = ["NIR", "Green", "Red", "Blue"]
        self.assertTrue(
            IndexDefinitions.validate_index_requirements("GNDVI", available_bands)
        )

        # Test with insufficient bands
        insufficient_bands = ["Red", "Blue"]
        self.assertFalse(
            IndexDefinitions.validate_index_requirements("GNDVI", insufficient_bands)
        )

    def test_get_all_indices(self) -> None:
        """Test getting all indices"""
        all_indices = IndexDefinitions.get_all_indices()

        self.assertIsInstance(all_indices, list)
        self.assertGreater(len(all_indices), 0)

        # Check presence of main indices
        expected_indices = ["GNDVI", "NDWI", "MCARI"]
        for index in expected_indices:
            self.assertIn(index, all_indices)

    def test_calculate_index_with_missing_bands(self) -> None:
        """Test index calculation with missing bands"""
        # Create data with insufficient number of bands
        insufficient_data = np.random.rand(
            self.height, self.width, 3
        )  # Only 3 bands

        with self.assertRaises(ValueError):
            self.calculator.calculate_from_arrays_simple(
                insufficient_data,
                self.segmentation_mask,
                "Hyperspectral",
                ["GNDVI"],  # Requires NIR and Green bands
            )

    def test_calculate_index_with_invalid_data_type(self) -> None:
        """Test index calculation with invalid data type"""
        # Create data with invalid type
        invalid_data = np.random.randint(0, 256, (self.height, self.width, self.bands))

        # Should work with type conversion
        result = self.calculator.calculate_from_arrays_simple(
            invalid_data.astype(np.uint8), self.segmentation_mask, "Hyperspectral", ["GNDVI"]
        )

        self.assertIn("GNDVI", result)

    def test_assess_plant_condition_with_empty_indices(self) -> None:
        """Test plant condition assessment with empty indices"""
        empty_indices = {}

        with self.assertRaises(ValueError):
            self.calculator.assess_plant_condition(
                empty_indices
            )

    def test_save_results_with_invalid_path(self) -> None:
        """Test saving results with invalid path"""
        result = self.calculator.calculate_from_arrays_simple(
            self.spectral_data, self.segmentation_mask, "Hyperspectral", ["GNDVI"]
        )

        invalid_path = "/invalid/path/that/cannot/be/created/results.json"

        with self.assertRaises(Exception):
            self.calculator.save_results(result, invalid_path)

    def test_load_results_with_invalid_file(self) -> None:
        """Test loading results from invalid file"""
        invalid_path = "/nonexistent/path/results.json"

        with self.assertRaises(FileNotFoundError):
            self.calculator.load_results(invalid_path)

    def test_load_results_with_invalid_json(self) -> None:
        """Test loading results from invalid JSON file"""
        invalid_json_path = os.path.join(self.temp_dir, "invalid.json")

        # Create file with invalid JSON
        with open(invalid_json_path, "w") as f:
            f.write("{ invalid json }")

        with self.assertRaises(json.JSONDecodeError):
            self.calculator.load_results(invalid_json_path)

    def test_get_index_formula(self) -> None:
        """Test getting index formula"""
        formula = IndexDefinitions.get_index_formula("GNDVI")

        self.assertIsInstance(formula, str)
        self.assertIn("NIR", formula)
        self.assertIn("Green", formula)

    def test_get_index_description(self) -> None:
        """Test getting index description"""
        description = IndexDefinitions.get_index_description("GNDVI")

        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)

    def test_get_index_info_nonexistent(self) -> None:
        """Test getting information about non-existent index"""
        with self.assertRaises(KeyError):
            IndexDefinitions.get_index_info("NONEXISTENT_INDEX")

    def test_get_indices_by_group_nonexistent(self) -> None:
        """Test getting indices by non-existent group"""
        result = IndexDefinitions.get_indices_by_group("nonexistent_group")

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_normalize_index(self) -> None:
        """Test index normalization"""
        # Create test data
        test_data = np.random.rand(50, 50) * 0.8 + 0.1

        # Add NaN values
        test_data[10:20, 10:20] = np.nan

        normalized = IndexDefinitions.normalize_values(test_data)

        # Check that result doesn't contain NaN (in valid areas)
        valid_mask = ~np.isnan(test_data)
        self.assertFalse(np.any(np.isnan(normalized[valid_mask])))

        # Check range (using nanmin/nanmax to ignore NaN values)
        self.assertGreaterEqual(np.nanmin(normalized), 0)
        self.assertLessEqual(np.nanmax(normalized), 1)

    def test_calculate_index_edge_cases(self) -> None:
        """Test index calculation in edge cases"""
        # Create data with zero values
        zero_data = np.zeros((self.height, self.width, self.bands))

        result = self.calculator.calculate_from_arrays_simple(zero_data, self.segmentation_mask, "Hyperspectral", ["GNDVI"])

        self.assertIn("GNDVI", result)

        # Create data with very large values
        large_data = np.full((self.height, self.width, self.bands), 1e6)

        result = self.calculator.calculate_from_arrays_simple(
            large_data, self.segmentation_mask, "Hyperspectral", ["GNDVI"]
        )

        self.assertIn("GNDVI", result)

    @patch("src.indices.calculator.np.savez")
    def test_save_results_numpy_format(self, mock_save: MagicMock) -> None:
        """Test saving results in numpy format"""
        result = self.calculator.calculate_from_arrays_simple(
            self.spectral_data, self.segmentation_mask, "Hyperspectral", ["GNDVI"]
        )

        save_path = os.path.join(self.temp_dir, "results.npz")
        self.calculator.save_results(result, save_path, format="numpy")

        mock_save.assert_called_once()

    def test_assess_plant_condition_different_masks(self) -> None:
        """Test plant condition assessment with different masks"""
        # Create mask with single class
        single_class_mask = np.ones((self.height, self.width), dtype=int)

        indices_result = self.calculator.calculate_from_arrays_simple(
            self.spectral_data, single_class_mask, "Hyperspectral", ["GNDVI", "NDWI"]
        )

        condition_result = self.calculator.assess_plant_condition(
            indices_result, single_class_mask
        )

        self.assertIn("classification", condition_result)
        self.assertIn("indices", condition_result)


if __name__ == "__main__":
    unittest.main()
