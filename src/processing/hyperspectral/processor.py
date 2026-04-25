"""
Main hyperspectral data processor module.

This module provides the main processor class for hyperspectral data processing
pipeline including data loading and orthophoto creation.
"""

import os
import json
import numpy as np
import time
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
from ...utils.memory_monitor import ResourceMonitor
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
        process_start_time = time.perf_counter()
        input_basename = os.path.basename(input_path)
        self.logger.info(f"[hsp] process: start input={input_basename} output={output_dir}")

        try:
            # Step 1: Load data from file
            self.logger.info("[hsp] process: Step 1/4 load_data - start")
            load_start_time = time.perf_counter()
            with ResourceMonitor("process.load_data", logger=self.logger):
                data = self.load_data(input_path)
            load_duration = time.perf_counter() - load_start_time
            self.logger.info(
                f"[hsp] process: Step 1/4 load_data - end "
                f"shape={data.shape} dtype={data.dtype} "
                f"duration={load_duration:.2f}s"
            )

            # Step 2: Get preprocessing configuration
            preprocessing_config = self.config.get("processing", {})

            # Step 3: Apply preprocessing steps
            self.logger.info("[hsp] process: Step 2/4 _apply_preprocessing - start")
            preprocess_start_time = time.perf_counter()
            with ResourceMonitor("process.apply_preprocessing", logger=self.logger, interval_s=30.0):
                data, applied_steps, metadata = self._apply_preprocessing(data, input_path, preprocessing_config)
                # Drop the original reference to free memory
                processed_data = data
                data = None  # Explicitly drop reference
            preprocess_duration = time.perf_counter() - preprocess_start_time
            self.logger.info(
                f"[hsp] process: Step 2/4 _apply_preprocessing - end "
                f"shape={processed_data.shape} dtype={processed_data.dtype} "
                f"duration={preprocess_duration:.2f}s"
            )

            # Step 4: Save results as per-band GeoTIFFs
            self.logger.info("[hsp] process: Step 3/4 _save_band_tiffs - start")
            save_start_time = time.perf_counter()
            with ResourceMonitor("process._save_band_tiffs", logger=self.logger):
                tiff_paths = self._save_band_tiffs(processed_data, input_path, output_dir, metadata)
            save_duration = time.perf_counter() - save_start_time
            self.logger.info(
                f"[hsp] process: Step 3/4 _save_band_tiffs - end "
                f"tiffs={len(tiff_paths)} duration={save_duration:.2f}s"
            )

            # Step 5: Prepare result dictionary for orthophoto processor
            result = {
                "tiff_paths": tiff_paths,
                "metadata": metadata
            }

            process_duration = time.perf_counter() - process_start_time
            self.logger.info(
                f"[hsp] process: Step 4/4 done - "
                f"tiffs={len(tiff_paths)} total_duration={process_duration:.2f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            raise ProcessingError(f"Error during processing: {e}")

    def _apply_preprocessing(self, processed_data: HyperspectralData, input_path: str, config: Dict[str, Any]) -> tuple:
        """
        Apply preprocessing steps to hyperspectral data.

        Args:
            data: Input hyperspectral data
            input_path: Path to input file
            config: Processing configuration

        Returns:
            Tuple of (processed_data, applied_steps, metadata)
        """
        from ...utils.gdal_utils import get_raster_metadata
        
        self.logger.info(
            f"[hsp] _apply_preprocessing: start "
            f"shape={data.shape} dtype={data.dtype} "
            f"size={data.nbytes / 1024 / 1024:.1f} MiB"
        )
        
        # Explicit log before data.copy() - the suspected OOM point
        copy_size_mb = data.nbytes / 1024 / 1024
        self.logger.info(
            f"[hsp] _apply_preprocessing: About to process input cube "
            f"shape={processed_data.shape} dtype={processed_data.dtype} size={copy_size_mb:.1f} MiB"
        )
        with ResourceMonitor("apply_preprocessing.copy_input", logger=self.logger):
            # In-place processing - no copy needed
            pass
        self.logger.info("[hsp] _apply_preprocessing: Processing ready")
        
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
            self.logger.info("[hsp] _apply_preprocessing: dark_current - start")
            start_time = time.perf_counter()
            dark_percentile = radiometric_config.get("dark_percentile", 1)
            dark_value = np.percentile(processed_data, dark_percentile)
            # Ensure dark_value is the same dtype as processed_data to avoid upcasting
            dark_value = np.float32(dark_value)
            processed_data -= dark_value  # In-place subtraction
            applied_steps.append("dark_current_subtraction")
            duration = time.perf_counter() - start_time
            self.logger.info(
                f"[hsp] _apply_preprocessing: dark_current - end "
                f"shape={processed_data.shape} dtype={processed_data.dtype} "
                f"duration={duration:.2f}s"
            )
        
        # 2. Flat field correction
        if radiometric_config.get("method") == "flat_field":
            self.logger.info("[hsp] _apply_preprocessing: flat_field - start")
            start_time = time.perf_counter()
            mean_value = np.mean(processed_data)
            if mean_value > 0:
                # Ensure mean_value is the same dtype as processed_data to avoid upcasting
                mean_value = np.float32(mean_value)
                processed_data /= mean_value  # In-place division
            applied_steps.append("flat_field_correction")
            duration = time.perf_counter() - start_time
            self.logger.info(
                f"[hsp] _apply_preprocessing: flat_field - end "
                f"shape={processed_data.shape} dtype={processed_data.dtype} "
                f"duration={duration:.2f}s"
            )
        
        # 3. Radiometric correction
        if radiometric_config.get("method") == "empirical_line":
            self.logger.info("[hsp] _apply_preprocessing: radiometric - start")
            start_time = time.perf_counter()
            gain = radiometric_config.get("gain", 1.0)
            offset = radiometric_config.get("offset", 0.0)
            # Ensure gain and offset are the same dtype as processed_data to avoid upcasting
            gain = np.float32(gain)
            offset = np.float32(offset)
            processed_data *= gain  # In-place multiplication
            processed_data += offset  # In-place addition
            applied_steps.append("radiometric_correction")
            duration = time.perf_counter() - start_time
            self.logger.info(
                f"[hsp] _apply_preprocessing: radiometric - end "
                f"shape={processed_data.shape} dtype={processed_data.dtype} "
                f"duration={duration:.2f}s"
            )
        
        # 4. Atmospheric correction
        atmospheric_config = config.get("atmospheric_correction", {})
        if atmospheric_config.get("enabled", False):
            self.logger.info("[hsp] _apply_preprocessing: atmospheric - start")
            start_time = time.perf_counter()
            method = atmospheric_config.get("method", "simplified")
            if method == "simplified":
                # Simple scalar correction
                atm_correction = atmospheric_config.get("correction_factor", 0.95)
                # Ensure atm_correction is the same dtype as processed_data to avoid upcasting
                atm_correction = np.float32(atm_correction)
                processed_data *= atm_correction  # In-place multiplication
                applied_steps.append("atmospheric_correction")
                duration = time.perf_counter() - start_time
                self.logger.info(
                    f"[hsp] _apply_preprocessing: atmospheric - end "
                    f"shape={processed_data.shape} dtype={processed_data.dtype} "
                    f"duration={duration:.2f}s"
                )
            else:
                # TODO: Implement other atmospheric correction methods
                self.logger.warning(f"Atmospheric correction method '{method}' not implemented, skipping")
        
        # 5. Noise filtering
        noise_config = config.get("noise_reduction", {})
        if noise_config.get("method"):
            self.logger.info("[hsp] _apply_preprocessing: noise_filtering - start")
            start_time = time.perf_counter()
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
                        duration = time.perf_counter() - start_time
                        self.logger.info(
                            f"[hsp] _apply_preprocessing: noise_filtering_savgol - end "
                            f"shape={processed_data.shape} dtype={processed_data.dtype} "
                            f"duration={duration:.2f}s"
                        )
                    else:
                        # Fallback to median filter
                        processed_data = median_filter(processed_data, size=3)
                        applied_steps.append("noise_filtering_median")
                        duration = time.perf_counter() - start_time
                        self.logger.info(
                            f"[hsp] _apply_preprocessing: noise_filtering_median (fallback) - end "
                            f"shape={processed_data.shape} dtype={processed_data.dtype} "
                            f"duration={duration:.2f}s"
                        )
                except Exception as e:
                    self.logger.warning(f"Error applying Savitzky-Golay filter: {e}, falling back to median filter")
                    processed_data = median_filter(processed_data, size=3)
                    applied_steps.append("noise_filtering_median")
                    duration = time.perf_counter() - start_time
                    self.logger.info(
                        f"[hsp] _apply_preprocessing: noise_filtering_median (fallback) - end "
                        f"shape={processed_data.shape} dtype={processed_data.dtype} "
                        f"duration={duration:.2f}s"
                    )
            elif method == "median" or not scipy_available:
                # Simple 3x3 median filter using scipy or numpy
                if scipy_available:
                    processed_data = median_filter(processed_data, size=3)
                    applied_steps.append("noise_filtering_median")
                    duration = time.perf_counter() - start_time
                    self.logger.info(
                        f"[hsp] _apply_preprocessing: noise_filtering_median (scipy) - end "
                        f"shape={processed_data.shape} dtype={processed_data.dtype} "
                        f"duration={duration:.2f}s"
                    )
                else:
                    processed_data = self._numpy_median_filter(processed_data, 3)
                    applied_steps.append("noise_filtering_median")
                    duration = time.perf_counter() - start_time
                    self.logger.info(
                        f"[hsp] _apply_preprocessing: noise_filtering_median (numpy) - end "
                        f"shape={processed_data.shape} dtype={processed_data.dtype} "
                        f"duration={duration:.2f}s"
                    )
            elif method in ("mean", "gaussian"):
                # Fallback to simple mean filter for implemented methods only
                kernel_size = 3
                processed_data = self._numpy_mean_filter(processed_data, kernel_size)
                applied_steps.append("noise_filtering_mean")
                duration = time.perf_counter() - start_time
                self.logger.info(
                    f"[hsp] _apply_preprocessing: noise_filtering_mean - end "
                    f"shape={processed_data.shape} dtype={processed_data.dtype} "
                    f"duration={duration:.2f}s"
                )
            else:
                # Method is not implemented - log warning and skip noise reduction
                self.logger.warning(f"noise_reduction.method='{method}' is not implemented yet; skipping noise reduction")
                duration = time.perf_counter() - start_time
                self.logger.info(
                    f"[hsp] _apply_preprocessing: noise_filtering - skipped "
                    f"duration={duration:.2f}s"
                )
        
        # 6. Normalization - check if it's enabled in spectral calibration or as a separate config
        spectral_config = config.get("spectral_calibration", {})
        if spectral_config.get("normalization", False):
            self.logger.info("[hsp] _apply_preprocessing: normalization - start")
            start_time = time.perf_counter()
            # Use minmax normalization by default
            # Normalize each band to [0, 1] using in-place operations
            for band_idx in range(processed_data.shape[2]):
                band_view = processed_data[:, :, band_idx]  # View, no copy
                min_val = band_view.min()
                max_val = band_view.max()
                span = max_val - min_val
                if span > 0:
                    band_view -= min_val  # In-place subtraction
                    band_view /= span     # In-place division
            applied_steps.append("normalization_minmax")
            duration = time.perf_counter() - start_time
            self.logger.info(
                f"[hsp] _apply_preprocessing: normalization - end "
                f"shape={processed_data.shape} dtype={processed_data.dtype} "
                f"duration={duration:.2f}s"
            )
        
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
        start_time = time.perf_counter()
        self.logger.info("[hsp] _numpy_median_filter: start - using scipy.ndimage.median_filter")
        
        from scipy.ndimage import median_filter
        # size=(k, k, 1) → per-band 2-D median; bands NOT mixed.
        # mode='nearest' → equivalent to np.pad(..., mode='edge').
        filtered = median_filter(data, size=(kernel_size, kernel_size, 1), mode='nearest')
        
        duration = time.perf_counter() - start_time
        self.logger.info(
            f"[hsp] _numpy_median_filter: end "
            f"shape={data.shape} dtype={data.dtype} duration={duration:.2f}s"
        )
        return filtered
    
    def _numpy_mean_filter(self, data: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Simple mean filter implementation using numpy.
        
        Args:
            data: Input data
            kernel_size: Size of the kernel
            
        Returns:
            Filtered data
        """
        start_time = time.perf_counter()
        self.logger.info("[hsp] _numpy_mean_filter: start - using scipy.ndimage.uniform_filter")
        
        from scipy.ndimage import uniform_filter
        # data shape: (H, W, B), float32. kernel_size is an odd int (3 in practice).
        # size=(k, k, 1) → average a k×k window per band; bands are NOT mixed.
        # mode='nearest' → replicate edge values, equivalent to np.pad(..., mode='edge').
        filtered = np.empty_like(data)
        uniform_filter(data, size=(kernel_size, kernel_size, 1),
                       mode='nearest', output=filtered)
        
        duration = time.perf_counter() - start_time
        self.logger.info(
            f"[hsp] _numpy_mean_filter: end "
            f"shape={data.shape} dtype={data.dtype} duration={duration:.2f}s"
        )
        return filtered
    
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
        from ...utils.gdal_utils import write_raster
        
        input_basename = os.path.basename(input_path)
        self.logger.info(f"[hsp] _save_band_tiffs: start bands={data.shape[2]}")
        start_time = time.perf_counter()
        
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
            self.logger.info(f"[{input_basename}] Saved band {band_idx} to {tiff_path}")
        
        duration = time.perf_counter() - start_time
        self.logger.info(f"[hsp] _save_band_tiffs: end duration={duration:.2f}s")
        return tiff_paths


