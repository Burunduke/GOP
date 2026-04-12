"""
Tests for validation utilities.
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import validators


def test_validate_array():
    """Test array validation."""
    # Valid array
    valid_array = np.array([[1, 2], [3, 4]])
    validators.validate_array(valid_array)

    # Test shape validation
    with pytest.raises(ValueError):
        validators.validate_array(valid_array, expected_shape=(3, 3))

    # Test dtype validation
    with pytest.raises(ValueError):
        validators.validate_array(valid_array, expected_dtype=np.float32)

    # Test NaN validation
    nan_array = np.array([1, np.nan, 3])
    with pytest.raises(ValueError):
        validators.validate_array(nan_array)

    # Test inf validation
    inf_array = np.array([1, np.inf, 3])
    with pytest.raises(ValueError):
        validators.validate_array(inf_array)

    # Test value range validation
    with pytest.raises(ValueError):
        validators.validate_array(valid_array, min_value=5)

    with pytest.raises(ValueError):
        validators.validate_array(valid_array, max_value=0)


def test_validate_wavelengths():
    """Test wavelength validation."""
    # Valid wavelengths
    valid_wavelengths = [400, 500, 600, 700]
    validators.validate_wavelengths(valid_wavelengths)

    # Test out of range
    with pytest.raises(ValueError):
        validators.validate_wavelengths([300, 400, 500])

    with pytest.raises(ValueError):
        validators.validate_wavelengths([400, 500, 3000])

    # Test unsorted
    with pytest.raises(ValueError):
        validators.validate_wavelengths([500, 400, 600])

    # Test NaN values
    with pytest.raises(ValueError):
        validators.validate_wavelengths([400, np.nan, 600])


def test_validate_file_path():
    """Test file path validation."""
    # Test existing file
    current_file = __file__
    validators.validate_file_path(current_file)

    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        validators.validate_file_path("/nonexistent/file.txt")

    # Test file extension
    with pytest.raises(ValueError):
        validators.validate_file_path(current_file, allowed_extensions=[".tif", ".hdr"])


def test_validate_band_names():
    """Test band name validation."""
    # Valid band names
    valid_bands = ["Red", "Green", "Blue", "NIR"]
    validators.validate_band_names(valid_bands)

    # Test required bands
    with pytest.raises(ValueError):
        validators.validate_band_names(valid_bands, required_bands=["SWIR"])

    # Test allowed bands
    with pytest.raises(ValueError):
        validators.validate_band_names(valid_bands, allowed_bands=["Red", "Green"])


def test_validate_config():
    """Test configuration validation."""
    # Valid config
    config = {"key1": "value1", "key2": "value2"}
    validators.validate_config(config, ["key1", "key2"])

    # Test missing keys
    with pytest.raises(ValueError):
        validators.validate_config(config, ["key1", "key2", "key3"])


def test_validate_positive_number():
    """Test positive number validation."""
    # Valid numbers
    validators.validate_positive_number(5)
    validators.validate_positive_number(3.14)

    # Test non-positive numbers
    with pytest.raises(ValueError):
        validators.validate_positive_number(0)

    with pytest.raises(ValueError):
        validators.validate_positive_number(-5)

    # Test non-number
    with pytest.raises(TypeError):
        validators.validate_positive_number("not a number")


if __name__ == "__main__":
    test_validate_array()
    test_validate_wavelengths()
    test_validate_file_path()
    test_validate_band_names()
    test_validate_config()
    test_validate_positive_number()
    print("All validation tests passed!")
