"""
Custom exception hierarchy for the GOP project.

This module provides a structured exception hierarchy for consistent error handling
across the application. All custom exceptions inherit from GOPException.
"""

from typing import Optional, Dict, Any


class GOPException(Exception):
    """Base exception class for all GOP-specific exceptions."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.

        Parameters
        ----------
        message : str
            Human-readable error message
        details : dict, optional
            Additional error details for debugging
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """String representation of the exception."""
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ValidationError(GOPException):
    """Exception raised for validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize validation error.

        Parameters
        ----------
        message : str
            Validation error message
        field : str, optional
            Name of the field that failed validation
        value : Any, optional
            Value that failed validation
        details : dict, optional
            Additional validation details
        """
        if field is not None:
            details = details or {}
            details["field"] = field
            if value is not None:
                details["value"] = value

        super().__init__(f"Validation error: {message}", details)


class ProcessingError(GOPException):
    """Exception raised for data processing errors."""

    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        input_data: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize processing error.

        Parameters
        ----------
        message : str
            Processing error message
        step : str, optional
            Name of the processing step that failed
        input_data : Any, optional
            Input data that caused the error
        details : dict, optional
            Additional processing details
        """
        if step is not None:
            details = details or {}
            details["step"] = step
            if input_data is not None:
                details["input_data_type"] = type(input_data).__name__

        super().__init__(f"Processing error: {message}", details)


class ConfigurationError(GOPException):
    """Exception raised for configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Any = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize configuration error.

        Parameters
        ----------
        message : str
            Configuration error message
        config_key : str, optional
            Configuration key that caused the error
        config_value : Any, optional
            Configuration value that caused the error
        details : dict, optional
            Additional configuration details
        """
        if config_key is not None:
            details = details or {}
            details["config_key"] = config_key
            if config_value is not None:
                details["config_value"] = config_value

        super().__init__(f"Configuration error: {message}", details)


class FileError(GOPException):
    """Exception raised for file-related errors."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize file error.

        Parameters
        ----------
        message : str
            File error message
        file_path : str, optional
            Path to the file that caused the error
        operation : str, optional
            File operation that failed (read, write, etc.)
        details : dict, optional
            Additional file details
        """
        if file_path is not None:
            details = details or {}
            details["file_path"] = file_path
            if operation is not None:
                details["operation"] = operation

        super().__init__(f"File error: {message}", details)


class GDALError(GOPException):
    """Exception raised for GDAL-related errors."""

    def __init__(
        self,
        message: str,
        gdal_error_code: Optional[int] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize GDAL error.

        Parameters
        ----------
        message : str
            GDAL error message
        gdal_error_code : int, optional
            GDAL error code
        dataset_info : dict, optional
            Information about the GDAL dataset
        details : dict, optional
            Additional GDAL details
        """
        if gdal_error_code is not None:
            details = details or {}
            details["gdal_error_code"] = gdal_error_code
        if dataset_info is not None:
            details = details or {}
            details["dataset_info"] = dataset_info

        super().__init__(f"GDAL error: {message}", details)




__all__ = [
    "GOPException",
    "ValidationError",
    "ProcessingError",
    "ConfigurationError",
    "FileError",
    "GDALError",
]
