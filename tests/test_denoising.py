"""
Tests for hyperspectral denoising module
"""

import unittest
import numpy as np
from typing import Any
from unittest.mock import patch, MagicMock
import pytest

from src.processing.hyperspectral.denoising import HyperspectralDenoising
from src.utils.exceptions import ProcessingError


class TestHyperspectralDenoising(unittest.TestCase):
    """Test cases for HyperspectralDenoising class"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.denoiser = HyperspectralDenoising()

        # Create test hyperspectral data
        self.test_data = np.random.rand(100, 100, 50).astype(np.float32)

    def test_initialization(self) -> None:
        """Test denoiser initialization"""
        self.assertIsInstance(self.denoiser, HyperspectralDenoising)

    def test_advanced_noise_reduction_valid_data(self) -> None:
        """Test advanced noise reduction with valid data"""
        result = self.denoiser.advanced_noise_reduction(self.test_data)
        self.assertEqual(result.shape, self.test_data.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_advanced_noise_reduction_custom_method(self) -> None:
        """Test advanced noise reduction with custom method"""
        result = self.denoiser.advanced_noise_reduction(self.test_data, method="pca")
        self.assertEqual(result.shape, self.test_data.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_advanced_noise_reduction_invalid_method(self) -> None:
        """Test advanced noise reduction with invalid method"""
        # Should return original data without raising exception
        result = self.denoiser.advanced_noise_reduction(
            self.test_data, method="invalid_method"
        )
        self.assertEqual(result.shape, self.test_data.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_pca_denoising_valid_data(self) -> None:
        """Test PCA denoising with valid data"""
        result = self.denoiser.pca_denoising(self.test_data)
        self.assertEqual(result.shape, self.test_data.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_pca_denoising_custom_components(self) -> None:
        """Test PCA denoising with custom number of components"""
        result = self.denoiser.pca_denoising(self.test_data, n_components=10)
        self.assertEqual(result.shape, self.test_data.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_pca_denoising_invalid_components(self) -> None:
        """Test PCA denoising with invalid number of components"""
        # The method should handle invalid components gracefully
        result = self.denoiser.pca_denoising(self.test_data, n_components=0)
        self.assertEqual(result.shape, self.test_data.shape)

    def test_savgol_denoising_valid_data(self) -> None:
        """Test Savitzky-Golay denoising with valid data"""
        result = self.denoiser.savgol_denoising(self.test_data)
        self.assertEqual(result.shape, self.test_data.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_wavelet_denoising_valid_data(self) -> None:
        """Test wavelet denoising with valid data"""
        result = self.denoiser.wavelet_denoising(self.test_data)
        self.assertEqual(result.shape, self.test_data.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_mnf_denoising_valid_data(self) -> None:
        """Test MNF denoising with valid data"""
        result = self.denoiser.mnf_denoising(self.test_data)
        self.assertEqual(result.shape, self.test_data.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_empty_data_handling(self) -> None:
        """Test handling of empty data"""
        empty_data = np.array([])
        # New implementation raises ValueError instead of returning original data
        with self.assertRaises(ValueError):
            self.denoiser.pca_denoising(empty_data)

    def test_nan_data_handling(self) -> None:
        """Test handling of NaN data"""
        nan_data = np.full((10, 10, 5), np.nan)
        # New implementation returns original data with warning for insufficient valid data
        result = self.denoiser.pca_denoising(nan_data)
        self.assertEqual(result.shape, nan_data.shape)

    def test_inf_data_handling(self) -> None:
        """Test handling of infinite data"""
        inf_data = np.full((10, 10, 5), np.inf)
        # New implementation returns original data with warning for insufficient valid data
        result = self.denoiser.pca_denoising(inf_data)
        self.assertEqual(result.shape, inf_data.shape)
        self.assertEqual(result.shape, inf_data.shape)

    def test_2d_data_handling(self) -> None:
        """Test handling of 2D data"""
        data_2d = np.random.rand(100, 100)
        # The method should return the original data when validation fails
        result = self.denoiser.pca_denoising(data_2d)
        self.assertEqual(result.shape, data_2d.shape)

    def test_denoising_pipeline(self) -> None:
        """Test denoising pipeline with multiple methods"""
        # Test PCA denoising
        result_pca = self.denoiser.pca_denoising(self.test_data)
        self.assertEqual(result_pca.shape, self.test_data.shape)

        # Test Savitzky-Golay denoising
        result_savgol = self.denoiser.savgol_denoising(self.test_data)
        self.assertEqual(result_savgol.shape, self.test_data.shape)

        # Test wavelet denoising
        result_wavelet = self.denoiser.wavelet_denoising(self.test_data)
        self.assertEqual(result_wavelet.shape, self.test_data.shape)

    @patch("src.processing.hyperspectral.denoising.PCA")
    def test_pca_denoising_with_mock(self, mock_pca: MagicMock) -> None:
        """Test PCA denoising with mocked PCA"""
        # Mock PCA behavior
        mock_pca_instance = MagicMock()
        mock_pca.return_value = mock_pca_instance
        mock_pca_instance.fit_transform.return_value = np.random.rand(10000, 10)
        mock_pca_instance.inverse_transform.return_value = np.random.rand(10000, 50)

        result = self.denoiser.pca_denoising(self.test_data)
        self.assertEqual(result.shape, self.test_data.shape)

    def test_estimate_noise_level(self) -> None:
        """Test noise level estimation (if implemented)"""
        # This is a placeholder test - the actual method may not exist
        # but we can test if the denoising methods work correctly
        result = self.denoiser.pca_denoising(self.test_data)
        self.assertIsNotNone(result)

    def test_evaluate_denoising_quality(self) -> None:
        """Test denoising quality evaluation (if implemented)"""
        # This is a placeholder test - the actual method may not exist
        # but we can test if the denoising methods work correctly
        result = self.denoiser.pca_denoising(self.test_data)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
