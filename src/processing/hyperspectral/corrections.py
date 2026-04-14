"""
Hyperspectral data corrections module.

This module provides atmospheric and radiometric correction methods
for hyperspectral data processing.
"""

import numpy as np
import logging
from typing import Any, Dict, Optional

# Constants for correction methods
DARK_CURRENT_METHOD = "dark_current"
EMPIRICAL_LINE_METHOD = "empirical_line"
FLAT_FIELD_METHOD = "flat_field"

# Numerical constants
ZERO_THRESHOLD = 1e-8
DEFAULT_PERCENTILE_LOW = 1
DEFAULT_PERCENTILE_HIGH = 99
DEFAULT_BRIGHT_THRESHOLD = 95
MIN_BRIGHT_PIXELS = 100


class HyperspectralCorrections:
    """Class for hyperspectral data corrections."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize corrections class.

        Args:
            logger: Logger for message recording
        """
        self.logger = logger or logging.getLogger(__name__)
        self.correction_methods = [DARK_CURRENT_METHOD, EMPIRICAL_LINE_METHOD, FLAT_FIELD_METHOD]

    def radiometric_correction(
        self, image_data: np.ndarray, method: str = EMPIRICAL_LINE_METHOD
    ) -> np.ndarray:
        """
        Perform radiometric correction on image data.

        Args:
            image_data: Input image data
            method: Correction method

        Returns:
            Corrected data

        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Validate input data
            if image_data is None or image_data.size == 0:
                raise ValueError("Input image data is empty")

            if method not in self.correction_methods:
                self.logger.warning(
                    f"Unknown correction method: {method}. Available methods: {self.correction_methods}"
                )
                return image_data

            if method == DARK_CURRENT_METHOD:
                return self.dark_current_correction(image_data)
            elif method == EMPIRICAL_LINE_METHOD:
                return self.empirical_line_correction(image_data)
            elif method == FLAT_FIELD_METHOD:
                return self.flat_field_correction(image_data)
            else:
                self.logger.warning(f"Unknown correction method: {method}")
                return image_data

        except ValueError as e:
            self.logger.error(f"Validation error during radiometric correction: {e}")
            raise
        except (RuntimeError, MemoryError) as e:
            self.logger.error(f"Execution error during radiometric correction: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during radiometric correction: {e}")
            raise

    def dark_current_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Perform dark current correction.

        Args:
            image_data: Input image data

        Returns:
            Corrected data

        Raises:
            Exception: If correction fails
        """
        try:
            # Calibration using dark current (1st percentile)
            dark_reference = np.percentile(image_data, DEFAULT_PERCENTILE_LOW, axis=(0, 1))
            corrected = image_data - dark_reference

            # Limit negative values
            corrected = np.maximum(corrected, 0)

            self.logger.info("Dark current correction completed")
            return corrected

        except Exception as e:
            self.logger.error(f"Error during dark current correction: {e}")
            raise

    def empirical_line_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Perform empirical line correction.

        Args:
            image_data: Input image data

        Returns:
            Corrected data

        Raises:
            Exception: If correction fails
        """
        try:
            # Calibration using dark current (1st percentile)
            dark_reference = np.percentile(image_data, DEFAULT_PERCENTILE_LOW, axis=(0, 1))
            corrected = image_data - dark_reference

            # Calibration using white reference (99th percentile)
            white_reference = np.percentile(image_data, DEFAULT_PERCENTILE_HIGH, axis=(0, 1))
            denominator = white_reference - dark_reference

            # Handle division by zero using np.where
            corrected = np.where(
                np.abs(denominator) > ZERO_THRESHOLD,
                corrected / denominator,
                0.0,  # Default value when dividing by zero
            )

            # Limit values
            corrected = np.clip(corrected, 0, 1)

            self.logger.info("Empirical line correction completed")
            return corrected

        except Exception as e:
            self.logger.error(f"Error during empirical line correction: {e}")
            raise

    def flat_field_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Perform flat field correction.

        Args:
            image_data: Input image data

        Returns:
            Corrected data

        Raises:
            Exception: If correction fails
        """
        try:
            # Create reference spectrum based on bright areas
            bright_threshold = np.percentile(image_data, DEFAULT_BRIGHT_THRESHOLD, axis=(0, 1))
            bright_mask = np.all(image_data > bright_threshold, axis=2)

            if np.sum(bright_mask) > MIN_BRIGHT_PIXELS:
                reference_spectrum = np.mean(image_data[bright_mask], axis=0)

                # Safe division with zero value handling
                corrected = np.where(
                    np.abs(reference_spectrum) > ZERO_THRESHOLD,
                    image_data / reference_spectrum,
                    image_data,  # Keep original values if divisor is close to zero
                )
            else:
                # Alternative method
                corrected = self.empirical_line_correction(image_data)

            # Limit values
            corrected = np.clip(corrected, 0, 1)

            self.logger.info("Flat field correction completed")
            return corrected

        except Exception as e:
            self.logger.error(f"Error during flat field correction: {e}")
            raise

    def atmospheric_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Perform simplified atmospheric correction.

        Args:
            image_data: Input image data

        Returns:
            Corrected data

        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Validate input data
            if image_data is None or image_data.size == 0:
                raise ValueError("Input image data is empty")

            # Simplified atmospheric correction based on statistics
            # In a real system, a more complex model should be used here

            # Estimate atmospheric haze based on dark objects
            dark_pixels = np.percentile(image_data, 2, axis=(0, 1))

            # Correction considering atmospheric effects
            corrected = image_data - dark_pixels
            corrected = np.maximum(corrected, 0)

            # Normalization with safe division
            max_values = np.percentile(corrected, 98, axis=(0, 1))
            corrected = np.where(
                np.abs(max_values) > ZERO_THRESHOLD,
                corrected / max_values,
                0.0,  # Default value when dividing by zero
            )
            corrected = np.clip(corrected, 0, 1)

            self.logger.info("Atmospheric correction completed")
            return corrected

        except ValueError as e:
            self.logger.error(f"Validation error during atmospheric correction: {e}")
            raise
        except (RuntimeError, MemoryError) as e:
            self.logger.error(f"Execution error during atmospheric correction: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during atmospheric correction: {e}")
            raise

    def calculate_correction_statistics(
        self, original_data: np.ndarray, corrected_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate correction statistics.

        Args:
            original_data: Original data
            corrected_data: Corrected data

        Returns:
            Dictionary with correction statistics

        Raises:
            ValueError: If input data is invalid
        """
        try:
            if original_data is None or corrected_data is None:
                raise ValueError("Input data cannot be None")

            if original_data.shape != corrected_data.shape:
                raise ValueError(
                    "Shapes of original and corrected data must match"
                )

            # Remove NaN and Inf values
            valid_mask = (
                ~np.isnan(original_data)
                & ~np.isinf(original_data)
                & ~np.isnan(corrected_data)
                & ~np.isinf(corrected_data)
            )

            if not np.any(valid_mask):
                return {"error": "No valid data for analysis"}

            orig_valid = original_data[valid_mask]
            corr_valid = corrected_data[valid_mask]

            statistics = {
                "original_mean": float(np.mean(orig_valid)),
                "original_std": float(np.std(orig_valid)),
                "original_min": float(np.min(orig_valid)),
                "original_max": float(np.max(orig_valid)),
                "corrected_mean": float(np.mean(corr_valid)),
                "corrected_std": float(np.std(corr_valid)),
                "corrected_min": float(np.min(corr_valid)),
                "corrected_max": float(np.max(corr_valid)),
                "mean_change": float(np.mean(corr_valid) - np.mean(orig_valid)),
                "std_change": float(np.std(corr_valid) - np.std(orig_valid)),
                "dynamic_range_change": float(
                    (np.max(corr_valid) - np.min(corr_valid))
                    - (np.max(orig_valid) - np.min(orig_valid))
                ),
            }

            return statistics

        except Exception as e:
            self.logger.error(f"Error calculating correction statistics: {e}")
            return {"error": str(e)}
