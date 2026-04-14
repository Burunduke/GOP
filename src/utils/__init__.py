"""
Utility modules for the GOP project.

This package provides various utility functions for file operations, image processing,
mathematical operations, validation, visualization, and logging.
"""

from .logger import setup_logger, get_logger
from .visualization import visualize_indices, create_comparison_plot
from .file_utils import ensure_dir, get_file_extension, validate_file_path
from .image_utils import resize_image, normalize_image, load_image
from .math_utils import safe_divide, safe_sqrt, safe_log
from .validators import validate_array, validate_file_path as validate_file_path_validator
from .exceptions import GOPException, ValidationError, FileError, ConfigurationError
from .gdal_utils import GDALDatasetManager, read_raster_band

__all__ = [
    # Logger functions
    "setup_logger",
    "get_logger",
    
    # Visualization functions
    "visualize_indices",
    "create_comparison_plot",
    
    # File utilities
    "ensure_dir",
    "get_file_extension",
    "validate_file_path",
    
    # Image utilities
    "resize_image",
    "normalize_image",
    "load_image",
    
    # Math utilities
    "safe_divide",
    "safe_sqrt",
    "safe_log",
    
    # Validation functions
    "validate_array",
    "validate_file_path_validator",
    
    # Exceptions
    "GOPException",
    "ValidationError",
    "FileError",
    "ConfigurationError",
    
    # GDAL utilities
    "GDALDatasetManager",
    "read_raster_band",
]
