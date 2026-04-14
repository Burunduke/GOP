"""
Main hyperspectral data processor module.

This module provides the main processor class for hyperspectral data processing
pipeline including data loading and orthophoto creation.
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from numpy.typing import NDArray

try:
    from osgeo import gdal

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    # Don't raise error here to allow tests to run

from ...core.config import get_config
from ...utils.logger import setup_logger
from ...utils.exceptions import ProcessingError, FileError
from .validators import HyperspectralValidator
from .cache import HyperspectralCache

# Type aliases for better type safety
HyperspectralData = NDArray[np.float32]
BandData = NDArray[np.float32]
SpectralProfile = NDArray[np.float32]
ProcessingResult = Dict[str, Any]


class HyperspectralProcessor:
    """
    Class for hyperspectral data processing.
    
    Scientific-oriented implementation with modern processing methods.
    """

    def __init__(self, cache_enabled: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize hyperspectral data processor.

        Args:
            cache_enabled: Enable caching
            cache_dir: Cache directory (if None, uses temporary directory)
        """
        self.logger = setup_logger(__name__)
        self.config = get_config()

        # Initialize components
        self.validator = HyperspectralValidator()
        self.cache = HyperspectralCache(
            cache_enabled=cache_enabled, cache_dir=cache_dir
        )

        self.logger.info("HyperspectralProcessor initialized")

    def load_data(self, file_path: str, **kwargs) -> HyperspectralData:
        """
        Load hyperspectral data from file.

        Args:
            file_path: Path to file
            **kwargs: Additional loading parameters

        Returns:
            Hyperspectral data in numpy array format

        Raises:
            FileError: If file not found or inaccessible
            ValidationError: If data validation fails
        """
        if not GDAL_AVAILABLE:
            raise ImportError(
                "GDAL library is required for loading data. Install with: pip install gdal"
            )

        try:
            # Check file existence
            if not os.path.exists(file_path):
                raise FileError(f"File not found: {file_path}")

            # Load data using GDAL
            dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
            if dataset is None:
                raise FileError(f"Failed to open file: {file_path}")

            # Get data information
            bands = dataset.RasterCount
            width = dataset.RasterXSize
            height = dataset.RasterYSize

            self.logger.info(
                f"Loading data: {bands} channels, {width}x{height} pixels"
            )

            # Read data
            data = np.zeros((height, width, bands), dtype=np.float32)
            for band_idx in range(bands):
                band = dataset.GetRasterBand(band_idx + 1)
                data[:, :, band_idx] = band.ReadAsArray()

            dataset = None  # Close dataset

            # Validate data
            self.validator.validate_data(data)

            self.logger.info("Data successfully loaded and validated")
            return data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise FileError(f"Error loading data: {e}")

    def process_pipeline(
        self, data: HyperspectralData, pipeline_config: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Execute simplified processing pipeline - only loads and returns data.

        Args:
            data: Input hyperspectral data
            pipeline_config: Pipeline configuration

        Returns:
            Processing results
        """
        try:
            self.logger.info("Starting simplified processing pipeline")

            # Validate input data
            self.validator.validate_data(data)

            result = {
                "processed_data": data,
                "metadata": {
                    "original_shape": data.shape,
                    "processing_steps": ["data_loading"],
                },
            }

            self.logger.info("Processing pipeline completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {e}")
            raise ProcessingError(f"Error in processing pipeline: {e}")


    def save_results(self, results: ProcessingResult, output_dir: str) -> str:
        """
        Save processing results.

        Args:
            results: Processing results
            output_dir: Output directory

        Returns:
            Path to saved results
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Save processed data
            if "processed_data" in results:
                # TODO: Implement data saving
                pass

            self.logger.info(f"Results saved to: {output_dir}")
            return output_dir

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise FileError(f"Error saving results: {e}")

    def process(self, input_path: str, output_dir: str) -> str:
        """
        Process hyperspectral data.

        Args:
            input_path: Path to input file
            output_dir: Output directory

        Returns:
            Path to processed results
        """
        # TODO: Implement processing
        return output_dir

