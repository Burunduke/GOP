"""
Validation utilities for data integrity and input validation.

This module provides comprehensive validation functions for various data types
used in the GOP project, including arrays, file paths, and wavelength data.
"""

import os
import numpy as np
from typing import Union, List, Tuple, Any
from pathlib import Path

from .exceptions import ValidationError, FileError, ConfigurationError


def validate_array(
    array: Any,
    expected_shape: tuple = None,
    expected_dtype: type = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_value: float = None,
    max_value: float = None,
) -> None:
    """
    Validate a numpy array for shape, dtype, and value constraints.

    Parameters
    ----------
    array : Any
        The array to validate
    expected_shape : tuple, optional
        Expected shape of the array
    expected_dtype : type, optional
        Expected data type of the array
    allow_nan : bool, optional
        Whether to allow NaN values (default: False)
    allow_inf : bool, optional
        Whether to allow infinite values (default: False)
    min_value : float, optional
        Minimum allowed value
    max_value : float, optional
        Maximum allowed value

    Raises
    ------
    ValidationError
        If the array fails validation
    TypeError
        If the input is not a numpy array
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(array).__name__}")

    if expected_shape is not None and array.shape != expected_shape:
        raise ValidationError(
            f"Array shape {array.shape} does not match expected shape {expected_shape}",
            details={"expected_shape": expected_shape, "actual_shape": array.shape},
        )

    if expected_dtype is not None and array.dtype != expected_dtype:
        raise ValidationError(
            f"Array dtype {array.dtype} does not match expected dtype {expected_dtype}",
            details={
                "expected_dtype": str(expected_dtype),
                "actual_dtype": str(array.dtype),
            },
        )

    if not allow_nan and np.any(np.isnan(array)):
        raise ValidationError("Array contains NaN values which are not allowed")

    if not allow_inf and np.any(np.isinf(array)):
        raise ValidationError("Array contains infinite values which are not allowed")

    if min_value is not None and np.any(array < min_value):
        raise ValidationError(
            f"Array contains values below minimum allowed value {min_value}",
            details={"min_value": min_value, "min_found": float(np.min(array))},
        )

    if max_value is not None and np.any(array > max_value):
        raise ValidationError(
            f"Array contains values above maximum allowed value {max_value}",
            details={"max_value": max_value, "max_found": float(np.max(array))},
        )


def validate_wavelengths(
    wavelengths: Union[List[float], np.ndarray],
    min_wavelength: float = 400,
    max_wavelength: float = 2500,
    require_sorted: bool = True,
) -> None:
    """
    Validate wavelength data for spectral analysis.

    Parameters
    ----------
    wavelengths : list or np.ndarray
        Array of wavelength values
    min_wavelength : float, optional
        Minimum allowed wavelength in nanometers (default: 400)
    max_wavelength : float, optional
        Maximum allowed wavelength in nanometers (default: 2500)
    require_sorted : bool, optional
        Whether wavelengths must be sorted (default: True)

    Raises
    ------
    ValidationError
        If wavelengths fail validation
    """
    if not isinstance(wavelengths, (list, np.ndarray)):
        raise TypeError(
            f"Expected list or numpy array, got {type(wavelengths).__name__}"
        )

    wavelengths_array = np.asarray(wavelengths)

    if wavelengths_array.size == 0:
        raise ValidationError("Wavelength array cannot be empty")

    if np.any(wavelengths_array < min_wavelength):
        raise ValidationError(
            f"Wavelengths below minimum {min_wavelength} nm found",
            details={
                "min_wavelength": min_wavelength,
                "min_found": float(np.min(wavelengths_array)),
            },
        )

    if np.any(wavelengths_array > max_wavelength):
        raise ValidationError(
            f"Wavelengths above maximum {max_wavelength} nm found",
            details={
                "max_wavelength": max_wavelength,
                "max_found": float(np.max(wavelengths_array)),
            },
        )

    if np.any(np.isnan(wavelengths_array)):
        raise ValidationError("Wavelength array contains NaN values")

    if require_sorted and not np.all(np.diff(wavelengths_array) > 0):
        raise ValidationError("Wavelengths must be sorted in ascending order")


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    must_be_readable: bool = True,
    must_be_writable: bool = False,
    allowed_extensions: List[str] = None,
) -> None:
    """
    Validate a file path for existence, readability, and file type.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to validate
    must_exist : bool, optional
        Whether the file must exist (default: True)
    must_be_readable : bool, optional
        Whether the file must be readable (default: True)
    must_be_writable : bool, optional
        Whether the file must be writable (default: False)
    allowed_extensions : list, optional
        List of allowed file extensions (e.g., ['.tif', '.hdr'])

    Raises
    ------
    FileError
        If the file path fails validation
    FileNotFoundError
        If the file doesn't exist when must_exist=True
    """
    file_path = Path(file_path)

    if must_exist and not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if must_exist and must_be_readable and not os.access(file_path, os.R_OK):
        raise FileError(
            f"File is not readable: {file_path}",
            file_path=str(file_path),
            operation="read",
        )

    if must_be_writable and not os.access(file_path, os.W_OK):
        raise FileError(
            f"File is not writable: {file_path}",
            file_path=str(file_path),
            operation="write",
        )

    if allowed_extensions is not None:
        file_ext = file_path.suffix.lower()
        if file_ext not in [ext.lower() for ext in allowed_extensions]:
            raise FileError(
                f"File extension {file_ext} not allowed. Allowed: {allowed_extensions}",
                file_path=str(file_path),
                details={
                    "allowed_extensions": allowed_extensions,
                    "actual_extension": file_ext,
                },
            )


def validate_band_names(
    band_names: List[str],
    required_bands: List[str] = None,
    allowed_bands: List[str] = None,
) -> None:
    """
    Validate band names for spectral data.

    Parameters
    ----------
    band_names : list of str
        List of band names to validate
    required_bands : list of str, optional
        Bands that must be present
    allowed_bands : list of str, optional
        Bands that are allowed (if provided, only these bands are permitted)

    Raises
    ------
    ValidationError
        If band names fail validation
    """
    if not isinstance(band_names, list) or not all(
        isinstance(b, str) for b in band_names
    ):
        raise TypeError("Band names must be a list of strings")

    if len(band_names) == 0:
        raise ValidationError("Band names list cannot be empty")

    if required_bands is not None:
        missing_bands = set(required_bands) - set(band_names)
        if missing_bands:
            raise ValidationError(
                f"Required bands missing: {sorted(missing_bands)}",
                details={
                    "required_bands": required_bands,
                    "missing_bands": sorted(missing_bands),
                },
            )

    if allowed_bands is not None:
        invalid_bands = set(band_names) - set(allowed_bands)
        if invalid_bands:
            raise ValidationError(
                f"Invalid bands found: {sorted(invalid_bands)}",
                details={
                    "allowed_bands": allowed_bands,
                    "invalid_bands": sorted(invalid_bands),
                },
            )


def validate_config(config: dict, required_keys: List[str]) -> None:
    """
    Validate configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary to validate
    required_keys : list of str
        Keys that must be present in the configuration

    Raises
    ------
    ConfigurationError
        If configuration fails validation
    """
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")

    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        raise ConfigurationError(
            f"Missing required configuration keys: {sorted(missing_keys)}",
            details={
                "required_keys": required_keys,
                "missing_keys": sorted(missing_keys),
            },
        )


def validate_positive_number(value: Union[int, float], name: str = "value") -> None:
    """
    Validate that a number is positive.

    Parameters
    ----------
    value : int or float
        Number to validate
    name : str, optional
        Name of the parameter for error messages

    Raises
    ------
    ValidationError
        If the value is not positive
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")

    if value <= 0:
        raise ValidationError(
            f"{name} must be positive, got {value}",
            details={"parameter_name": name, "value": value},
        )
