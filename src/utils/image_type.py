"""
Image type detection utility for GOP project.

This module provides functionality to detect whether an image file is RGB or hyperspectral
based on its file extension and, when necessary, its raster band count.
"""

import os
from pathlib import Path
from typing import Union

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

from .logger import setup_logger

# Set up logger for this module
logger = setup_logger("image_type")

# File extension categories
RGB_EXTENSIONS = {".png", ".jpg", ".jpeg"}
HS_EXTENSIONS = {".bil", ".hdr", ".dat", ".img"}
AMBIGUOUS_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
RGB_BAND_LIMIT = 4  # 1=gray, 3=RGB, 4=RGBA — anything more is hyperspectral


def detect_image_type(path: Union[str, Path]) -> str:
    """
    Detect whether an image file is RGB or hyperspectral.
    
    Detection rules:
    1. Lowercase the file extension.
    2. If extension in {".png", ".jpg", ".jpeg"} → return "rgb".
    3. If extension in {".bil", ".hdr", ".dat", ".img"} → return "hyperspectral".
    4. If extension in {".tif", ".tiff", ".geotiff"}:
       - Open with GDAL, read RasterCount.
       - RasterCount <= 4 → "rgb", else "hyperspectral".
       - On GDAL open failure, log a warning and default to "hyperspectral".
    5. Unknown extension → log warning and default to "hyperspectral".
    
    Args:
        path: Path to the image file
        
    Returns:
        "rgb" or "hyperspectral"
        
    Examples:
        >>> detect_image_type("photo.png")
        'rgb'
        >>> detect_image_type("cube.bil")
        'hyperspectral'
        >>> detect_image_type("rgb_image.tif")  # 3-band TIFF
        'rgb'
        >>> detect_image_type("hyperspectral_cube.tif")  # 200-band TIFF
        'hyperspectral'
    """
    # Convert Path to string if needed
    if isinstance(path, Path):
        path = str(path)
    
    # Get file extension and lowercase it
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    
    # Fast path: decide by extension
    if ext in RGB_EXTENSIONS:
        return "rgb"
    elif ext in HS_EXTENSIONS:
        return "hyperspectral"
    elif ext in AMBIGUOUS_EXTENSIONS:
        # For TIFF files, we need to check the band count
        if not GDAL_AVAILABLE:
            logger.warning(f"GDAL not available, defaulting {path} to hyperspectral")
            return "hyperspectral"
            
        try:
            # Open with GDAL and read raster count
            dataset = gdal.Open(path, gdal.GA_ReadOnly)
            if dataset is None:
                logger.warning(f"Failed to open {path} with GDAL, defaulting to hyperspectral")
                return "hyperspectral"
                
            band_count = dataset.RasterCount
            dataset = None  # Close dataset
            
            if band_count <= RGB_BAND_LIMIT:
                return "rgb"
            else:
                return "hyperspectral"
        except Exception as e:
            logger.warning(f"Error reading {path} with GDAL: {e}, defaulting to hyperspectral")
            return "hyperspectral"
    else:
        # Unknown extension, default to hyperspectral for safety
        logger.warning(f"Unknown extension {ext} for file {path}, defaulting to hyperspectral")
        return "hyperspectral"


__all__ = ["detect_image_type"]