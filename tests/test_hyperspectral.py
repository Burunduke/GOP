"""
Tests for hyperspectral data processing module
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from typing import Any, Dict, Tuple
from unittest.mock import patch, MagicMock
from src.processing.hyperspectral import HyperspectralProcessor


class TestHyperspectralProcessor(unittest.TestCase):
    """Test cases for HyperspectralProcessor class"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.processor = HyperspectralProcessor()
        self.temp_dir = tempfile.mkdtemp()

        # Create test hyperspectral file
        self.test_bil_path = os.path.join(self.temp_dir, "test.bil")
        self.test_hdr_path = os.path.join(self.temp_dir, "test.hdr")

        # Create simple test image
        self.width, self.height, self.bands = 100, 100, 50
        test_data = np.random.randint(
            0, 1000, (self.height, self.width, self.bands), dtype=np.uint16
        )
        test_data.tofile(self.test_bil_path)

        # Create header
        hdr_content = f"""ENVI
samples = {self.width}
lines = {self.height}
bands = {self.bands}
header offset = 0
file type = ENVI Standard
data type = 12
interleave = bil
byte order = 0
wavelength = {{400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890}}
"""
        with open(self.test_hdr_path, "w") as f:
            f.write(hdr_content)

    def tearDown(self) -> None:
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_processor_initialization(self) -> None:
        """Test processor initialization"""
        self.assertIsInstance(self.processor, HyperspectralProcessor)

    @patch("osgeo.gdal.Open")
    def test_read_hyperspectral_data(self, mock_gdal_open: MagicMock) -> None:
        """Test reading hyperspectral data"""
        # Mock setup
        mock_dataset = MagicMock()
        mock_dataset.RasterXSize = self.width
        mock_dataset.RasterYSize = self.height
        mock_dataset.RasterCount = self.bands

        # Create mock data for each channel
        band_data = np.random.rand(self.height, self.width)
        mock_band = MagicMock()
        mock_band.ReadAsArray.return_value = band_data
        mock_dataset.GetRasterBand.return_value = mock_band

        # Mock for metadata
        mock_dataset.GetMetadata.return_value = {
            "wavelength": "400, 410, 420, 430, 440, 450, 460, 470, 480, 490"
        }

        mock_gdal_open.return_value = mock_dataset

        # Call method
        dataset, image_data, wavelengths = self.processor._read_hyperspectral_data(
            self.test_bil_path
        )

        # Assertions
        self.assertEqual(image_data.shape, (self.height, self.width, self.bands))
        self.assertIsNotNone(wavelengths)
        mock_gdal_open.assert_called_once_with(self.test_bil_path)

    def test_extract_wavelengths(self) -> None:
        """Test extracting wavelengths"""
        # Create mock dataset with metadata
        mock_dataset = MagicMock()
        mock_dataset.GetMetadata.return_value = {
            "wavelength": "400, 410, 420, 430, 440, 450, 460, 470, 480, 490"
        }

        wavelengths = self.processor._extract_wavelengths(mock_dataset)

        self.assertIsNotNone(wavelengths)
        self.assertEqual(len(wavelengths), 10)
        self.assertEqual(wavelengths[0], 400)
        self.assertEqual(wavelengths[-1], 490)

    def test_extract_wavelengths_no_metadata(self) -> None:
        """Test extracting wavelengths when metadata is missing"""
        mock_dataset = MagicMock()
        mock_dataset.GetMetadata.return_value = {}

        wavelengths = self.processor._extract_wavelengths(mock_dataset)

        self.assertIsNone(wavelengths)

    def test_analyze_data_quality(self) -> None:
        """Test data quality analysis"""
        # Create test data
        test_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1

        # Add some NaN values
        test_data[10:20, 10:20, 0] = np.nan

        quality = self.processor._analyze_data_quality(test_data)

        self.assertIn("missing_values", quality)
        self.assertIn("data_range", quality)
        self.assertIn("statistics", quality)

        # Check statistics
        stats = quality["statistics"]
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)

    def test_calculate_snr(self) -> None:
        """Test signal-to-noise ratio calculation"""
        # Create test data with good SNR
        signal = np.ones((100, 100)) * 100
        noise = np.random.normal(0, 5, (100, 100))
        data = signal + noise

        snr = self.processor._calculate_snr(data)

        self.assertIsInstance(snr, float)
        self.assertGreater(snr, 0)

    def test_calculate_snr_empty_data(self) -> None:
        """Test SNR calculation for empty data"""
        empty_data = np.array([])

        with self.assertRaises(ValueError):
            self.processor._calculate_snr(empty_data)

    def test_calculate_quality_score(self) -> None:
        """Test quality score calculation"""
        data_quality: Dict[str, Any] = {
            "missing_values": {"nan_percentage": 5.0, "inf_percentage": 0.0},
            "overall_quality": {"average_snr": 20.0},
        }

        score = self.processor._calculate_quality_score(data_quality)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)



    @patch("osgeo.gdal.GetDriverByName")
    def test_convert_to_tiff(self, mock_gdal_driver: MagicMock) -> None:
        """Test conversion to TIFF"""
        # Mock setup
        mock_driver = MagicMock()
        mock_dataset = MagicMock()
        mock_driver.Create.return_value = mock_dataset
        mock_gdal_driver.return_value = mock_driver

        # Create test data
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        wavelengths = np.linspace(400, 900, self.bands)
        metadata = {"sensor_type": "Hyperspectral"}

        output_path = os.path.join(self.temp_dir, "output.tif")

        result_path = self.processor._convert_to_tiff(
            image_data, wavelengths, metadata, output_path
        )

        self.assertEqual(result_path, output_path)
        mock_driver.Create.assert_called_once()

    def test_get_band_info(self) -> None:
        """Test getting band information"""
        with patch("osgeo.gdal.Open") as mock_gdal_open:
            # Mock setup
            mock_dataset = MagicMock()
            mock_dataset.RasterXSize = self.width
            mock_dataset.RasterYSize = self.height
            mock_dataset.RasterCount = self.bands

            # Create mock data for each channel
            band_data = np.random.rand(self.height, self.width)
            mock_band = MagicMock()
            mock_band.ReadAsArray.return_value = band_data
            mock_band.GetMinimum.return_value = 0.1
            mock_band.GetMaximum.return_value = 0.9
            mock_band.GetStatistics.return_value = (0.1, 0.9, 0.5, 0.1)
            mock_dataset.GetRasterBand.return_value = mock_band

            mock_gdal_open.return_value = mock_dataset

            # Call method
            band_info = self.processor.get_band_info(self.test_bil_path)

            # Assertions
            self.assertIn("total_bands", band_info)
            self.assertIn("bands", band_info)
            self.assertEqual(band_info["total_bands"], self.bands)
            self.assertEqual(len(band_info["bands"]), self.bands)

    def test_create_rgb_composite(self) -> None:
        """Test creating RGB composite"""
        with patch("osgeo.gdal.Open") as mock_gdal_open:
            # Mock setup
            mock_dataset = MagicMock()
            mock_dataset.RasterXSize = self.width
            mock_dataset.RasterYSize = self.height
            mock_dataset.RasterCount = self.bands

            # Create mock data for each channel
            band_data = np.random.rand(self.height, self.width)
            mock_band = MagicMock()
            mock_band.ReadAsArray.return_value = band_data
            mock_dataset.GetRasterBand.return_value = mock_band

            mock_gdal_open.return_value = mock_dataset

            # Call method
            rgb_indices = (10, 20, 30)  # R, G, B channels
            output_path = os.path.join(self.temp_dir, "rgb_composite.tif")

            with patch.object(self.processor, "_convert_to_tiff") as mock_convert:
                mock_convert.return_value = output_path

                result_path = self.processor.create_rgb_composite(
                    [self.test_bil_path], rgb_indices, output_path
                )

                self.assertEqual(result_path, output_path)

    def test_process_invalid_input_path(self) -> None:
        """Test processing invalid input path"""
        with self.assertRaises(FileNotFoundError):
            self.processor.process("/nonexistent/path/file.bil", self.temp_dir)

    def test_process_invalid_output_dir(self) -> None:
        """Test processing invalid output directory"""
        with self.assertRaises(Exception):
            self.processor.process(
                self.test_bil_path, "/invalid/path/that/cannot/be/created"
            )


if __name__ == "__main__":
    unittest.main()
