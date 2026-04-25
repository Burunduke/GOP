"""
Main hyperspectral data processor module.

This module provides the main processor class for hyperspectral data processing
pipeline including data loading and orthophoto creation.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, List
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
                f"[{os.path.basename(file_path)}] Loading data: {bands} channels, {width}x{height} pixels"
            )

            # Read data
            data = np.zeros((height, width, bands), dtype=np.float32)
            for band_idx in range(bands):
                band = dataset.GetRasterBand(band_idx + 1)
                data[:, :, band_idx] = band.ReadAsArray()

            dataset = None  # Close dataset

            # Validate data
            self.validator.validate_data(data)

            self.logger.info(f"[{os.path.basename(file_path)}] Data successfully loaded and validated")
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

            # Save processed data as GeoTIFF
            if "processed_data" in results:
                if not GDAL_AVAILABLE:
                    raise ImportError(
                        "GDAL library is required for saving data. Install with: pip install gdal"
                    )

                data: np.ndarray = results["processed_data"]
                height, width, bands = data.shape

                tif_path = os.path.join(output_dir, "hyperspectral_processed.tif")
                driver = gdal.GetDriverByName("GTiff")
                dataset = driver.Create(tif_path, width, height, bands, gdal.GDT_Float32)
                for band_idx in range(bands):
                    band = dataset.GetRasterBand(band_idx + 1)
                    band.WriteArray(data[:, :, band_idx])
                dataset.FlushCache()
                dataset = None  # Close dataset

                self.logger.info(f"Saved GeoTIFF: {tif_path}")

            # Save metadata as JSON
            if "metadata" in results:
                metadata = results["metadata"]
                # Convert non-serializable types (e.g. tuples) to lists
                serializable_metadata = json.loads(
                    json.dumps(metadata, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
                )
                json_path = os.path.join(output_dir, "metadata.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_metadata, f, indent=2)
                self.logger.info(f"Saved metadata: {json_path}")

            self.logger.info(f"Results saved to: {output_dir}")
            return output_dir

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise FileError(f"Error saving results: {e}")

    def process(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process hyperspectral data with real preprocessing steps.

        Args:
            input_path: Path to input file
            output_dir: Output directory

        Returns:
            Dictionary with tiff_paths and metadata for orthophoto processor
        """
        try:
            self.logger.info(f"[{os.path.basename(input_path)}] Starting processing")

            # Step 1: Load data from file
            data = self.load_data(input_path)

            # Step 2: Get preprocessing configuration
            preprocessing_config = self.config.get("processing", {})

            # Step 3: Apply preprocessing steps
            processed_data, applied_steps, metadata = self._apply_preprocessing(data, input_path, preprocessing_config)

            # Step 4: Save results as per-band GeoTIFFs
            tiff_paths = self._save_band_tiffs(processed_data, input_path, output_dir, metadata)

            # Step 5: Prepare result dictionary for orthophoto processor
            result = {
                "tiff_paths": tiff_paths,
                "metadata": metadata
            }

            self.logger.info(f"[{os.path.basename(input_path)}] Processing completed. {len(tiff_paths)} GeoTIFFs created.")
            return result

        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            raise ProcessingError(f"Error during processing: {e}")

    def _apply_preprocessing(self, data: HyperspectralData, input_path: str, config: Dict[str, Any]) -> tuple:
        """
        Apply preprocessing steps to hyperspectral data.

        Args:
            data: Input hyperspectral data
            input_path: Path to input file
            config: Processing configuration

        Returns:
            Tuple of (processed_data, applied_steps, metadata)
        """
        import time
        from ...utils.gdal_utils import get_raster_metadata
        
        processed_data = data.copy()
        applied_steps = []
        
        # Get source metadata
        try:
            source_metadata = get_raster_metadata(input_path)
        except Exception as e:
            self.logger.warning(f"Could not extract metadata from source: {e}")
            source_metadata = {}
        
        # Preprocessing order: dark_current → flat_field → radiometric → atmospheric → noise → normalization
        
        # 1. Dark current subtraction
        radiometric_config = config.get("radiometric_correction", {})
        if radiometric_config.get("method") == "dark_current":
            start_time = time.time()
            dark_percentile = radiometric_config.get("dark_percentile", 1)
            dark_value = np.percentile(processed_data, dark_percentile)
            processed_data = processed_data - dark_value
            applied_steps.append("dark_current_subtraction")
            duration = time.time() - start_time
            self.logger.info(f"Applied dark current subtraction (value: {dark_value:.4f}) in {duration:.2f}s")
        
        # 2. Flat field correction
        if radiometric_config.get("method") == "flat_field":
            start_time = time.time()
            mean_value = np.mean(processed_data)
            if mean_value > 0:
                processed_data = processed_data / mean_value
            applied_steps.append("flat_field_correction")
            duration = time.time() - start_time
            self.logger.info(f"Applied flat field correction (mean: {mean_value:.4f}) in {duration:.2f}s")
        
        # 3. Radiometric correction
        if radiometric_config.get("method") == "empirical_line":
            start_time = time.time()
            gain = radiometric_config.get("gain", 1.0)
            offset = radiometric_config.get("offset", 0.0)
            processed_data = gain * processed_data + offset
            applied_steps.append("radiometric_correction")
            duration = time.time() - start_time
            self.logger.info(f"Applied radiometric correction (gain: {gain}, offset: {offset}) in {duration:.2f}s")
        
        # 4. Atmospheric correction
        atmospheric_config = config.get("atmospheric_correction", {})
        if atmospheric_config.get("enabled", False):
            start_time = time.time()
            method = atmospheric_config.get("method", "simplified")
            if method == "simplified":
                # Simple scalar correction
                atm_correction = atmospheric_config.get("correction_factor", 0.95)
                processed_data = processed_data * atm_correction
                applied_steps.append("atmospheric_correction")
                duration = time.time() - start_time
                self.logger.info(f"Applied simplified atmospheric correction (factor: {atm_correction}) in {duration:.2f}s")
            else:
                # TODO: Implement other atmospheric correction methods
                self.logger.warning(f"Atmospheric correction method '{method}' not implemented, skipping")
        
        # 5. Noise filtering
        noise_config = config.get("noise_reduction", {})
        if noise_config.get("method"):
            start_time = time.time()
            method = noise_config.get("method")
            
            # Try to use scipy if available, otherwise fall back to numpy
            try:
                from scipy.ndimage import median_filter
                scipy_available = True
            except ImportError:
                scipy_available = False
                self.logger.info("scipy not available, using numpy-based filtering")
            
            if method == "savgol" and scipy_available:
                try:
                    from scipy.signal import savgol_filter
                    # Apply Savitzky-Golay filter to each band
                    window_length = noise_config.get("savgol_window", 11)
                    polyorder = noise_config.get("savgol_polyorder", 3)
                    # Ensure window_length is odd and less than array size
                    window_length = min(window_length, processed_data.shape[0], processed_data.shape[1])
                    if window_length % 2 == 0:
                        window_length -= 1
                    if window_length >= 3:
                        for band_idx in range(processed_data.shape[2]):
                            processed_data[:, :, band_idx] = savgol_filter(
                                processed_data[:, :, band_idx], window_length, polyorder
                            )
                        applied_steps.append("noise_filtering_savgol")
                        duration = time.time() - start_time
                        self.logger.info(f"Applied Savitzky-Golay filtering in {duration:.2f}s")
                    else:
                        # Fallback to median filter
                        processed_data = median_filter(processed_data, size=3)
                        applied_steps.append("noise_filtering_median")
                        duration = time.time() - start_time
                        self.logger.info(f"Applied median filtering (fallback) in {duration:.2f}s")
                except Exception as e:
                    self.logger.warning(f"Error applying Savitzky-Golay filter: {e}, falling back to median filter")
                    processed_data = median_filter(processed_data, size=3)
                    applied_steps.append("noise_filtering_median")
                    duration = time.time() - start_time
                    self.logger.info(f"Applied median filtering (fallback) in {duration:.2f}s")
            elif method == "median" or not scipy_available:
                # Simple 3x3 median filter using scipy or numpy
                if scipy_available:
                    processed_data = median_filter(processed_data, size=3)
                    applied_steps.append("noise_filtering_median")
                    duration = time.time() - start_time
                    self.logger.info(f"Applied median filtering in {duration:.2f}s")
                else:
                    processed_data = self._numpy_median_filter(processed_data, 3)
                    applied_steps.append("noise_filtering_median")
                    duration = time.time() - start_time
                    self.logger.info(f"Applied numpy-based median filtering in {duration:.2f}s")
            else:
                # Fallback to simple mean filter
                kernel_size = 3
                processed_data = self._numpy_mean_filter(processed_data, kernel_size)
                applied_steps.append("noise_filtering_mean")
                duration = time.time() - start_time
                self.logger.info(f"Applied mean filtering in {duration:.2f}s")
        
        # 6. Normalization - check if it's enabled in spectral calibration or as a separate config
        spectral_config = config.get("spectral_calibration", {})
        if spectral_config.get("normalization", False):
            start_time = time.time()
            # Use minmax normalization by default
            # Normalize each band to [0, 1]
            for band_idx in range(processed_data.shape[2]):
                band = processed_data[:, :, band_idx]
                min_val = np.min(band)
                max_val = np.max(band)
                if max_val > min_val:
                    processed_data[:, :, band_idx] = (band - min_val) / (max_val - min_val)
            applied_steps.append("normalization_minmax")
            duration = time.time() - start_time
            self.logger.info(f"Applied min-max normalization in {duration:.2f}s")
        
        # Build metadata
        metadata = {
            "crs": source_metadata.get("projection", None),
            "transform": source_metadata.get("geotransform", None),
            "width": processed_data.shape[1],
            "height": processed_data.shape[0],
            "band_count": processed_data.shape[2],
            "dtype": str(processed_data.dtype),
            "source_files": [input_path],
            "applied_steps": applied_steps
        }
        
        return processed_data, applied_steps, metadata
    
    def _numpy_median_filter(self, data: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Simple median filter implementation using numpy.
        
        Args:
            data: Input data
            kernel_size: Size of the kernel
            
        Returns:
            Filtered data
        """
        # For simplicity, we'll use a basic approach for 3D data
        # This is a simplified implementation - in practice, you might want to use scipy
        filtered_data = data.copy()
        pad = kernel_size // 2
        
        # Apply filter to each band separately
        for band_idx in range(data.shape[2]):
            band = data[:, :, band_idx]
            padded = np.pad(band, pad, mode='edge')
            for i in range(band.shape[0]):
                for j in range(band.shape[1]):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    filtered_data[i, j, band_idx] = np.median(window)
        
        return filtered_data
    
    def _numpy_mean_filter(self, data: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Simple mean filter implementation using numpy.
        
        Args:
            data: Input data
            kernel_size: Size of the kernel
            
        Returns:
            Filtered data
        """
        # For simplicity, we'll use a basic approach for 3D data
        filtered_data = data.copy()
        pad = kernel_size // 2
        
        # Apply filter to each band separately
        for band_idx in range(data.shape[2]):
            band = data[:, :, band_idx]
            padded = np.pad(band, pad, mode='edge')
            for i in range(band.shape[0]):
                for j in range(band.shape[1]):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    filtered_data[i, j, band_idx] = np.mean(window)
        
        return filtered_data
    
    def _save_band_tiffs(self, data: HyperspectralData, input_path: str, output_dir: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Save processed data as per-band GeoTIFFs.
        
        Args:
            data: Processed hyperspectral data
            input_path: Path to input file
            output_dir: Output directory
            metadata: Metadata dictionary
            
        Returns:
            List of paths to saved GeoTIFFs
        """
        import os
        from ...utils.gdal_utils import write_raster
        
        os.makedirs(output_dir, exist_ok=True)
        tiff_paths = []
        
        # Save each band as a separate GeoTIFF
        for band_idx in range(data.shape[2]):
            band_data = data[:, :, band_idx]
            tiff_path = os.path.join(output_dir, f"band_{band_idx:03d}.tif")
            
            # Write raster with georeferencing from source
            write_raster(
                band_data,
                tiff_path,
                source_path=input_path,
                geotransform=metadata.get("transform"),
                projection=metadata.get("crs"),
                data_type=6  # gdal.GDT_Float32
            )
            
            tiff_paths.append(tiff_path)
            self.logger.info(f"[{os.path.basename(input_path)}] Saved band {band_idx} to {tiff_path}")
        
        return tiff_paths


