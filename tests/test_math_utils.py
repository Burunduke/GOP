"""
Unit tests for math_utils module.
"""

import numpy as np
import pytest
from src.utils.math_utils import safe_divide, safe_normalize, is_valid_number


class TestSafeDivide:
    """Test cases for safe_divide function."""

    def test_divide_valid_numbers(self):
        """Test division of valid numbers."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(15, 3) == 5.0
        assert safe_divide(7.5, 2.5) == 3.0

    def test_divide_by_zero(self):
        """Test division by zero returns default value."""
        assert np.isnan(safe_divide(10, 0))
        assert safe_divide(10, 0, default=0.0) == 0.0
        assert safe_divide(10, 0, default=-999) == -999

    def test_nan_inputs(self):
        """Test handling of NaN inputs."""
        assert np.isnan(safe_divide(np.nan, 2))
        assert np.isnan(safe_divide(10, np.nan))
        assert np.isnan(safe_divide(np.nan, np.nan))

    def test_infinite_inputs(self):
        """Test handling of infinite inputs."""
        assert np.isnan(safe_divide(np.inf, 2))
        assert np.isnan(safe_divide(10, np.inf))
        assert np.isnan(safe_divide(np.inf, np.inf))
        assert np.isnan(safe_divide(-np.inf, 2))

    def test_array_operations(self):
        """Test safe_divide with numpy arrays."""
        numerator = np.array([10, 20, 30, 40])
        denominator = np.array([2, 0, 5, np.nan])
        result = safe_divide(numerator, denominator)

        expected = np.array([5.0, np.nan, 6.0, np.nan])
        np.testing.assert_array_equal(result, expected, equal_nan=True)

    def test_custom_default(self):
        """Test custom default values."""
        result = safe_divide(10, 0, default=999)
        assert result == 999

        result_array = safe_divide(np.array([10, 20]), np.array([0, 0]), default=-1)
        expected = np.array([-1, -1])
        np.testing.assert_array_equal(result_array, expected)


class TestSafeNormalize:
    """Test cases for safe_normalize function."""

    def test_normalize_valid_array(self):
        """Test normalization of valid array."""
        values = np.array([0, 50, 100])
        normalized = safe_normalize(values)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_custom_range(self):
        """Test normalization with custom range."""
        values = np.array([20, 40, 60])
        normalized = safe_normalize(values, value_range=(10, 70))
        expected = np.array([0.1667, 0.5, 0.8333])
        np.testing.assert_array_almost_equal(normalized, expected, decimal=4)

    def test_normalize_invalid_range(self):
        """Test normalization with invalid range."""
        values = np.array([10, 20, 30])
        normalized = safe_normalize(values, value_range=(30, 30))  # min == max
        assert np.all(np.isnan(normalized))

    def test_normalize_empty_array(self):
        """Test normalization of empty array."""
        values = np.array([])
        normalized = safe_normalize(values)
        assert normalized.size == 0

    def test_normalize_with_nan_values(self):
        """Test normalization with NaN values."""
        values = np.array([10, np.nan, 30])
        normalized = safe_normalize(values)
        # Should handle NaN values gracefully
        assert np.isnan(normalized[1])


class TestIsValidNumber:
    """Test cases for is_valid_number function."""

    def test_valid_numbers(self):
        """Test valid number detection."""
        assert is_valid_number(42)
        assert is_valid_number(3.14)
        assert is_valid_number(-10.5)
        assert is_valid_number(0)

    def test_invalid_values(self):
        """Test invalid value detection."""
        assert not is_valid_number(np.nan)
        assert not is_valid_number(np.inf)
        assert not is_valid_number(-np.inf)
        assert not is_valid_number("not a number")
        assert not is_valid_number(None)

    def test_string_numbers(self):
        """Test string representations of numbers."""
        assert is_valid_number("42")
        assert is_valid_number("3.14")
        assert not is_valid_number("abc")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
