"""
Mathematical utility functions for safe numerical operations.

This module provides safe mathematical operations that handle edge cases like
division by zero, NaN values, and infinite values gracefully.
"""

import numpy as np
from typing import Union, Any
import warnings


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    default: float = np.nan,
    epsilon: float = 1e-12,
) -> Union[float, np.ndarray]:
    """
    Safely divide two numbers or arrays, handling division by zero and invalid inputs.

    Parameters
    ----------
    numerator : float or np.ndarray
        The numerator value(s)
    denominator : float or np.ndarray
        The denominator value(s)
    default : float, optional
        Value to return when division is invalid (default: np.nan)
    epsilon : float, optional
        Small value to avoid floating point precision issues (default: 1e-12)

    Returns
    -------
    float or np.ndarray
        The result of the division, or default value for invalid divisions

    Examples
    --------
    >>> safe_divide(10, 2)
    5.0
    >>> safe_divide(10, 0)
    nan
    >>> safe_divide(np.array([10, 20]), np.array([2, 0]))
    array([5., nan])
    """
    # Handle scalar inputs
    if np.isscalar(numerator) and np.isscalar(denominator):
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        if np.isnan(numerator) or np.isinf(numerator):
            return default
        return numerator / denominator

    # Handle mixed scalar/array inputs
    if np.isscalar(numerator):
        numerator = np.full_like(denominator, numerator)
    if np.isscalar(denominator):
        denominator = np.full_like(numerator, denominator)

    # Handle array inputs
    result = np.full_like(numerator, default, dtype=np.float64)

    # Create mask for valid divisions
    valid_mask = (
        (~np.isnan(denominator)) & (~np.isinf(denominator)) & (denominator != 0)
    )
    valid_mask &= (~np.isnan(numerator)) & (~np.isinf(numerator))

    # Perform division only on valid elements
    if np.any(valid_mask):
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return result


def safe_normalize(
    values: np.ndarray, value_range: tuple = None, default: float = np.nan
) -> np.ndarray:
    """
    Safely normalize values to [0, 1] range, handling edge cases.

    Parameters
    ----------
    values : np.ndarray
        Input values to normalize
    value_range : tuple, optional
        Custom range (min, max) for normalization. If None, uses min/max of values.
    default : float, optional
        Value to return when normalization is invalid (default: np.nan)

    Returns
    -------
    np.ndarray
        Normalized values in [0, 1] range
    """
    if values.size == 0:
        return np.array([])

    if value_range is None:
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
    else:
        min_val, max_val = value_range

    # Check if range is valid
    if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
        return np.full_like(values, default)

    return safe_divide(values - min_val, max_val - min_val, default=default)


def is_valid_number(value: Any) -> bool:
    """
    Check if a value is a valid, finite number.

    Parameters
    ----------
    value : Any
        Value to check

    Returns
    -------
    bool
        True if value is a valid finite number, False otherwise
    """
    try:
        return (
            np.isfinite(float(value))
            and not np.isnan(float(value))
            and not np.isinf(float(value))
        )
    except (ValueError, TypeError):
        return False


def safe_sqrt(value: Union[float, np.ndarray], default: float = np.nan) -> Union[float, np.ndarray]:
    """
    Safely compute square root, handling negative values and invalid inputs.

    Parameters
    ----------
    value : float or np.ndarray
        Input value(s) for square root calculation
    default : float, optional
        Value to return when square root is invalid (default: np.nan)

    Returns
    -------
    float or np.ndarray
        Square root of input values, or default value for invalid inputs
    """
    if np.isscalar(value):
        if value < 0 or np.isnan(value) or np.isinf(value):
            return default
        return np.sqrt(value)
    
    # Handle array inputs
    result = np.full_like(value, default, dtype=np.float64)
    valid_mask = (value >= 0) & (~np.isnan(value)) & (~np.isinf(value))
    
    if np.any(valid_mask):
        result[valid_mask] = np.sqrt(value[valid_mask])
    
    return result


def safe_log(value: Union[float, np.ndarray], default: float = np.nan) -> Union[float, np.ndarray]:
    """
    Safely compute natural logarithm, handling non-positive values and invalid inputs.

    Parameters
    ----------
    value : float or np.ndarray
        Input value(s) for logarithm calculation
    default : float, optional
        Value to return when logarithm is invalid (default: np.nan)

    Returns
    -------
    float or np.ndarray
        Natural logarithm of input values, or default value for invalid inputs
    """
    if np.isscalar(value):
        if value <= 0 or np.isnan(value) or np.isinf(value):
            return default
        return np.log(value)
    
    # Handle array inputs
    result = np.full_like(value, default, dtype=np.float64)
    valid_mask = (value > 0) & (~np.isnan(value)) & (~np.isinf(value))
    
    if np.any(valid_mask):
        result[valid_mask] = np.log(value[valid_mask])
    
    return result


__all__ = [
    "safe_divide",
    "safe_normalize",
    "is_valid_number",
    "safe_sqrt",
    "safe_log",
]
