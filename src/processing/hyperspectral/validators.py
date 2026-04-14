"""
Data validation module for hyperspectral processing.

This module provides validation functions for hyperspectral data and processing parameters.
"""

import os
import numpy as np
from typing import Any, List, Optional
from src.utils.validators import (
    validate_file_path,
    validate_array,
    validate_wavelengths,
)


class HyperspectralValidator:
    """Class for validating hyperspectral data and parameters."""

    @staticmethod
    def validate_input_path(input_path: str) -> None:
        """
        Validate input file path.

        Args:
            input_path: Path to input file

        Raises:
            ValueError: If path is invalid
            FileNotFoundError: If file does not exist
        """
        validate_file_path(input_path, must_exist=True, must_be_readable=True)

    @staticmethod
    def validate_output_dir(output_dir: str) -> None:
        """
        Validate output directory.

        Args:
            output_dir: Path to output directory

        Raises:
            ValueError: If path is invalid
        """
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("output_dir must be a non-empty string")

    @staticmethod
    def validate_file_format(file_path: str, supported_formats: List[str]) -> None:
        """
        Validate file format.

        Args:
            file_path: Path to file
            supported_formats: List of supported formats

        Raises:
            ValueError: If format is not supported
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}"
            )

    @staticmethod
    def validate_image_data(image_data: np.ndarray) -> None:
        """
        Validate image data.

        Args:
            image_data: Image data

        Raises:
            ValueError: If data is invalid
        """
        if image_data is None or image_data.size == 0:
            raise ValueError("Input image data is empty or None")

        if len(image_data.shape) != 3:
            raise ValueError(f"Expected 3D array, got {len(image_data.shape)}D")

        rows, cols, bands = image_data.shape
        if rows <= 0 or cols <= 0 or bands <= 0:
            raise ValueError(f"Invalid image dimensions: {rows}x{cols}x{bands}")

    @staticmethod
    def validate_wavelengths(wavelengths: Optional[np.ndarray]) -> None:
        """
        Validate wavelengths.

        Args:
            wavelengths: Array of wavelengths

        Raises:
            ValueError: If wavelengths are invalid
        """
        if wavelengths is not None:
            if not isinstance(wavelengths, np.ndarray):
                raise ValueError("Wavelengths must be a numpy array")

            if wavelengths.size == 0:
                raise ValueError("Wavelengths array is empty")

            if np.any(np.isnan(wavelengths)) or np.any(np.isinf(wavelengths)):
                raise ValueError("Wavelengths array contains NaN or Inf values")

            if np.any(wavelengths <= 0):
                raise ValueError("Wavelengths must be positive")

    @staticmethod
    def validate_dataset(dataset: Any) -> None:
        """
        Validate GDAL dataset.

        Args:
            dataset: GDAL dataset

        Raises:
            ValueError: If dataset is invalid
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")

        if (
            hasattr(dataset, "RasterXSize")
            and hasattr(dataset, "RasterYSize")
            and hasattr(dataset, "RasterCount")
        ):
            if (
                dataset.RasterXSize <= 0
                or dataset.RasterYSize <= 0
                or dataset.RasterCount <= 0
            ):
                raise ValueError(
                    f"Invalid dataset dimensions: {dataset.RasterYSize}x{dataset.RasterXSize}, channels: {dataset.RasterCount}"
                )
        else:
            raise ValueError("Dataset does not have required attributes")

    @staticmethod
    def validate_processing_parameters(
        method: str, available_methods: List[str]
    ) -> None:
        """
        Validate processing parameters.

        Args:
            method: Processing method
            available_methods: List of available methods

        Raises:
            ValueError: If method is not available
        """
        if method not in available_methods:
            raise ValueError(
                f"Unknown method: {method}. Available methods: {available_methods}"
            )

    @staticmethod
    def validate_pca_parameters(n_components: float) -> None:
        """
        Validate PCA parameters.

        Args:
            n_components: Number of components or explained variance ratio

        Raises:
            ValueError: If parameters are invalid
        """
        if not (0 < n_components <= 1) and not isinstance(n_components, int):
            raise ValueError(
                "n_components must be in range (0, 1] or an integer"
            )

    @staticmethod
    def validate_rgb_bands(rgb_bands: tuple, max_bands: int) -> None:
        """
        Validate RGB composite parameters.

        Args:
            rgb_bands: Channel indices for RGB
            max_bands: Maximum number of channels

        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(rgb_bands, tuple) or len(rgb_bands) != 3:
            raise ValueError("rgb_bands must be a tuple of 3 elements")

        if not all(isinstance(band, int) and band > 0 for band in rgb_bands):
            raise ValueError("rgb_bands must contain positive integers")

        if max(rgb_bands) > max_bands:
            raise ValueError(
                f"Insufficient channels for RGB composite. Required: {max(rgb_bands)}, available: {max_bands}"
            )
