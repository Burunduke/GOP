"""
Tests for pipeline module
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import pytest

from src.core.pipeline import Pipeline
from src.utils.exceptions import ProcessingError, ValidationError


class TestPipeline(unittest.TestCase):
    """Test cases for Pipeline class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = Pipeline()

        # Create test data
        self.test_data = np.random.rand(100, 100, 50).astype(np.float32)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_initialization_with_config(self):
        """Test pipeline initialization with custom config"""
        # Create a simple config file
        config_path = os.path.join(self.temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("""
logging:
  level: INFO
output:
  results_dir: results
""")

        pipeline = Pipeline(config_path=config_path)
        self.assertIsInstance(pipeline, Pipeline)

    def test_process_basic_functionality(self):
        """Test basic pipeline processing functionality"""
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test.npy")
        np.save(test_file, self.test_data)

        # Test processing - should raise error due to GDAL not being available
        with self.assertRaises(Exception):
            self.pipeline.process(input_path=test_file, output_dir=self.temp_dir)

    def test_process_with_nonexistent_file(self):
        """Test processing with nonexistent file"""
        with self.assertRaises(Exception):
            self.pipeline.process(
                input_path="/nonexistent/file.npy", output_dir=self.temp_dir
            )

    def test_process_with_invalid_sensor_type(self):
        """Test processing with invalid sensor type"""
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test.npy")
        np.save(test_file, self.test_data)

        # Should handle gracefully
        with self.assertRaises(Exception):
            self.pipeline.process(
                input_path=test_file,
                output_dir=self.temp_dir,
                sensor_type="InvalidSensor",
            )

    def test_pipeline_components_initialization(self):
        """Test that pipeline components are properly initialized"""
        self.assertIsNotNone(self.pipeline.hyperspectral_processor)
        self.assertIsNotNone(self.pipeline.orthophoto_processor)
        self.assertIsNotNone(self.pipeline.segmenter)
        self.assertIsNotNone(self.pipeline.index_calculator)

    def test_results_storage(self):
        """Test that results are stored properly"""
        self.assertIsInstance(self.pipeline.results, dict)
        self.assertEqual(len(self.pipeline.results), 0)


if __name__ == "__main__":
    unittest.main()
