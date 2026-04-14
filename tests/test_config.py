"""
Tests for configuration module
"""

import unittest
import tempfile
import os
import shutil
import yaml
from typing import Dict, Any
from src.core.config import Config, get_config


class TestConfig(unittest.TestCase):
    """Test cases for configuration class"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, "test_config.yaml")

        # Create test configuration
        self.test_config: Dict[str, Any] = {
            "processing": {
                "hyperspectral": {
                    "radiometric_correction": True,
                    "noise_reduction": True,
                    "output_format": "GTiff",
                },
                "orthophoto": {"resolution": 0.01, "method": "odm"},
            },
            "segmentation": {
                "compression_ratio": 0.125,
                "preliminary_model": "deeplabv3plus",
                "refinement_model": "cascade_psp",
            },
            "indices": {
                "default_indices": ["GNDVI", "NDWI", "MCARI"],
                "normalization_method": "minmax",
            },
            "output": {"save_intermediate": False, "create_visualizations": True},
        }

        # Save test configuration
        with open(self.test_config_path, "w") as f:
            yaml.dump(self.test_config, f)

        # Reset singleton instance before each test
        Config.reset_instance()

    def tearDown(self) -> None:
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Reset singleton instance after each test
        Config.reset_instance()

    def test_config_initialization_with_path(self) -> None:
        """Test configuration initialization with file path"""
        config = Config(self.test_config_path)

        self.assertEqual(
            config.config["processing"]["hyperspectral"]["radiometric_correction"], True
        )
        self.assertEqual(config.config["segmentation"]["compression_ratio"], 0.125)
        self.assertEqual(
            config.config["indices"]["default_indices"], ["GNDVI", "NDWI", "MCARI"]
        )

    def test_config_initialization_without_path(self) -> None:
        """Test configuration initialization without path (uses default configuration)"""
        config = Config()

        # Check presence of main sections
        self.assertIn("processing", config.config)
        self.assertIn("segmentation", config.config)
        self.assertIn("indices", config.config)
        self.assertIn("output", config.config)

    def test_get_config_function(self) -> None:
        """Test get_config function with singleton pattern"""
        # First call should create instance
        config1 = get_config()
        self.assertIsInstance(config1, Config)

        # Second call should return same instance
        config2 = get_config()
        self.assertIs(config1, config2)

        # Reset and get new instance
        Config.reset_instance()
        config3 = get_config()
        self.assertIsNot(config1, config3)

    def test_get_config_with_custom_instance(self) -> None:
        """Test get_config function with custom instance"""
        custom_config = Config(self.test_config_path)
        config = get_config(custom_config)
        
        self.assertIs(config, custom_config)
        self.assertEqual(
            config.get("processing.hyperspectral.radiometric_correction"), True
        )

    def test_get_method(self) -> None:
        """Test get method"""
        config = Config(self.test_config_path)

        # Test getting existing value
        self.assertEqual(
            config.get("processing.hyperspectral.radiometric_correction"), True
        )

        # Test getting default value
        self.assertEqual(
            config.get("nonexistent.key", "default_value"), "default_value"
        )

        # Test getting section
        processing_config = config.get("processing")
        self.assertIsInstance(processing_config, dict)
        self.assertIn("hyperspectral", processing_config)

    def test_set_method(self) -> None:
        """Test set method"""
        config = Config(self.test_config_path)

        # Test setting new value
        config.set("test.new_parameter", "test_value")
        self.assertEqual(config.get("test.new_parameter"), "test_value")

        # Test modifying existing value
        config.set("processing.hyperspectral.radiometric_correction", False)
        self.assertEqual(
            config.get("processing.hyperspectral.radiometric_correction"), False
        )

    def test_save_method(self) -> None:
        """Test save method"""
        config = Config(self.test_config_path)

        # Modify configuration
        config.set("test.parameter", "test_value")

        # Save to new file
        save_path = os.path.join(self.temp_dir, "saved_config.yaml")
        config.save(save_path)

        # Check save
        self.assertTrue(os.path.exists(save_path))

        # Load and verify
        with open(save_path, "r") as f:
            saved_config = yaml.safe_load(f)

        self.assertEqual(saved_config["test"]["parameter"], "test_value")

    def test_update_method(self) -> None:
        """Test update method"""
        config = Config(self.test_config_path)

        # Update configuration
        update_dict = {
            "new_section": {"parameter1": "value1", "parameter2": "value2"},
            "processing": {"hyperspectral": {"new_parameter": "new_value"}},
        }

        config.update(update_dict)

        # Check updates
        self.assertEqual(config.get("new_section.parameter1"), "value1")
        self.assertEqual(
            config.get("processing.hyperspectral.new_parameter"), "new_value"
        )

        # Check preservation of existing values
        self.assertEqual(
            config.get("processing.hyperspectral.radiometric_correction"), True
        )

    def test_deep_update_method(self) -> None:
        """Test _deep_update method"""
        config = Config(self.test_config_path)

        base_dict = {
            "section1": {
                "subsection1": {"param1": "value1", "param2": "value2"},
                "subsection2": {"param3": "value3"},
            },
            "section2": {"param4": "value4"},
        }

        update_dict = {
            "section1": {
                "subsection1": {"param2": "updated_value2", "param5": "value5"},
                "subsection3": {"param6": "value6"},
            },
            "section3": {"param7": "value7"},
        }

        config._deep_update(base_dict, update_dict)

        # Check results
        self.assertEqual(
            base_dict["section1"]["subsection1"]["param1"], "value1"
        )  # preserved
        self.assertEqual(
            base_dict["section1"]["subsection1"]["param2"], "updated_value2"
        )  # updated
        self.assertEqual(
            base_dict["section1"]["subsection1"]["param5"], "value5"
        )  # added
        self.assertEqual(
            base_dict["section1"]["subsection2"]["param3"], "value3"
        )  # preserved
        self.assertEqual(
            base_dict["section1"]["subsection3"]["param6"], "value6"
        )  # added
        self.assertEqual(base_dict["section2"]["param4"], "value4")  # preserved
        self.assertEqual(base_dict["section3"]["param7"], "value7")  # added

    def test_invalid_config_path(self) -> None:
        """Test handling of invalid configuration path"""
        # New implementation returns default config instead of raising exception
        config = Config("/nonexistent/path/config.yaml")
        self.assertIsInstance(config, Config)
        # Should have default configuration
        self.assertIsInstance(config.config, dict)

    def test_invalid_yaml_config(self) -> None:
        """Test handling of invalid YAML file"""
        invalid_config_path = os.path.join(self.temp_dir, "invalid_config.yaml")

        # Create invalid YAML file
        with open(invalid_config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        # New implementation returns default config instead of raising exception
        config = Config(invalid_config_path)
        self.assertIsInstance(config, Config)
        # Should have default configuration
        self.assertIsInstance(config.config, dict)

    def test_config_property(self) -> None:
        """Test config property"""
        config = Config(self.test_config_path)

        # Check that property returns a dictionary
        self.assertIsInstance(config.config, dict)

        # Check that modifying returned dictionary doesn't affect original
        config_dict = config.config
        config_dict["test"] = "value"

        self.assertNotIn("test", config.config)


if __name__ == "__main__":
    unittest.main()
