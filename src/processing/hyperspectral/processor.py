"""
Main hyperspectral data processor module.

This module provides the main processor class for hyperspectral data processing
pipeline including data loading and orthophoto creation.
"""

import os
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

    def __init__(self, cache_enabled: bool = False, cache_dir: Optional[str] = None):
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
        
        NOTE: This method is kept for backward compatibility only and is not used
        in the streaming processing path. For new code, use the streaming methods.

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
                # Read as float32 directly (Step 7)
                data[:, :, band_idx] = band.ReadAsArray(buf_type=gdal.GDT_Float32)

            dataset = None  # Close dataset

            # Validate data
            self.validator.validate_data(data)

            self.logger.info(f"[{os.path.basename(file_path)}] Data successfully loaded and validated")
            # Warn if this method is used since it's kept for backward compatibility only
            self.logger.warning(
                f"[{os.path.basename(file_path)}] load_data() is deprecated and kept for backward compatibility only. "
                "Use streaming methods for new code."
            )
            return data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise FileError(f"Error loading data: {e}")




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
            # Get preprocessing configuration
            preprocessing_config = self.config.get("processing", {})
            
            # Use streaming path instead of loading full cube (Step 5 + Step 7)
            self.logger.info("[hsp] process: Step 1/2 _apply_preprocessing_streaming - start")
            preprocess_start_time = time.perf_counter()
            with ResourceMonitor("process.apply_preprocessing_streaming", logger=self.logger, interval_s=30.0):
                tiff_paths, metadata = self._apply_preprocessing_streaming(input_path, output_dir, preprocessing_config)
            preprocess_duration = time.perf_counter() - preprocess_start_time
            self.logger.info(
                f"[hsp] process: Step 1/2 _apply_preprocessing_streaming - end "
                f"tiffs={len(tiff_paths)} duration={preprocess_duration:.2f}s"
            )

            # Prepare result dictionary for orthophoto processor
            result = {
                "tiff_paths": tiff_paths,
                "metadata": metadata
            }

            process_duration = time.perf_counter() - process_start_time
            self.logger.info(
                f"[hsp] process: Step 2/2 done - "
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
            processed_data: Input hyperspectral data
            input_path: Path to input file
            config: Processing configuration

        Returns:
            Tuple of (processed_data, applied_steps, metadata)
        """
        from ...utils.gdal_utils import get_raster_metadata
        
        self.logger.info(
            f"[hsp] _apply_preprocessing: start "
            f"shape={processed_data.shape} dtype={processed_data.dtype} "
            f"size={processed_data.nbytes / 1024 / 1024:.1f} MiB"
        )
        
        # Explicit log before data.copy() - the suspected OOM point
        copy_size_mb = processed_data.nbytes / 1024 / 1024
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

    def _iter_bands(self, input_path: str):
        """
        Generator that yields (band_index, band_array_2d_float32) one at a time using GDAL.
        
        Args:
            input_path: Path to input file
            
        Yields:
            tuple: (band_index, band_array_2d_float32)
        """
        if not GDAL_AVAILABLE:
            raise ImportError(
                "GDAL library is required for loading data. Install with: pip install gdal"
            )
        
        # Open dataset
        ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        if ds is None:
            raise FileError(f"Failed to open file: {input_path}")
        
        try:
            # Iterate through bands
            for i in range(ds.RasterCount):
                band = ds.GetRasterBand(i + 1)
                # Read as float32 directly (Step 7)
                arr = band.ReadAsArray(buf_type=gdal.GDT_Float32)
                yield i, arr
                # Explicitly delete the array to free memory immediately
                del arr
        finally:
            # Close dataset
            ds = None

    def _save_single_band_tiff(self, band_arr: np.ndarray, index: int, input_path: str,
                                 output_dir: str, metadata: Dict[str, Any]) -> str:
        """
        Save a single band array as GeoTIFF.
        
        Args:
            band_arr: 2D numpy array of band data
            index: Band index (0-based)
            input_path: Path to input file
            output_dir: Output directory
            metadata: Metadata dictionary
            
        Returns:
            Path to saved GeoTIFF
        """
        from ...utils.gdal_utils import write_raster
        
        os.makedirs(output_dir, exist_ok=True)
        tiff_path = os.path.join(output_dir, f"band_{index:03d}.tif")
        
        # Write raster with georeferencing from source
        write_raster(
            band_arr,
            tiff_path,
            source_path=input_path,
            geotransform=metadata.get("transform"),
            projection=metadata.get("crs"),
            data_type=6  # gdal.GDT_Float32
        )
        
        return tiff_path

    def _apply_preprocessing_streaming(self, input_path: str, output_dir: str, config: Dict[str, Any]) -> tuple:
        """
        Apply preprocessing steps to hyperspectral data in a streaming fashion.
        
        Args:
            input_path: Path to input file
            output_dir: Output directory
            config: Processing configuration
            
        Returns:
            Tuple of (tiff_paths, metadata)
        """
        from ...utils.gdal_utils import get_raster_metadata
        
        self.logger.info(f"[hsp] _apply_preprocessing_streaming: start input={input_path}")
        
        # Get source metadata
        try:
            source_metadata = get_raster_metadata(input_path)
        except Exception as e:
            self.logger.warning(f"Could not extract metadata from source: {e}")
            source_metadata = {}
        
        # Build metadata for output
        metadata = {
            "crs": source_metadata.get("projection", None),
            "transform": source_metadata.get("geotransform", None),
            "source_files": [input_path],
            "applied_steps": []
        }
        
        # Open dataset to get dimensions
        ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        if ds is None:
            raise FileError(f"Failed to open file: {input_path}")
        
        try:
            bands = ds.RasterCount
            width = ds.RasterXSize
            height = ds.RasterYSize
            
            # Update metadata with dimensions
            metadata.update({
                "width": width,
                "height": height,
                "band_count": bands,
                "dtype": "float32"
            })
            
            self.logger.info(f"[hsp] _apply_preprocessing_streaming: {bands} bands, {width}x{height} pixels")
            
            # Preprocessing configuration
            radiometric_config = config.get("radiometric_correction", {})
            atmospheric_config = config.get("atmospheric_correction", {})
            noise_config = config.get("noise_reduction", {})
            spectral_config = config.get("spectral_calibration", {})
            
            # Pre-compute global statistics if needed
            dark_value = None
            mean_value = None
            atm_correction = None
            
            # 1. Compute dark current value if needed (per-band approximation)
            if radiometric_config.get("method") == "dark_current":
                self.logger.info("[hsp] _apply_preprocessing_streaming: computing dark current")
                dark_percentile = radiometric_config.get("dark_percentile", 1)
                per_band = np.empty(bands, dtype=np.float32)
                
                # First pass: compute per-band percentiles
                for i, band_arr in self._iter_bands(input_path):
                    per_band[i] = np.percentile(band_arr, dark_percentile)
                    # Free memory immediately
                    del band_arr
                
                dark_value = float(per_band.min())
                metadata["applied_steps"].append("dark_current_subtraction")
                self.logger.info(f"[hsp] _apply_preprocessing_streaming: dark_value={dark_value}")
            
            # 2. Compute flat field mean if needed
            if radiometric_config.get("method") == "flat_field":
                self.logger.info("[hsp] _apply_preprocessing_streaming: computing flat field mean")
                sum_mean = 0.0
                
                # First pass: compute mean across all bands
                for i, band_arr in self._iter_bands(input_path):
                    sum_mean += band_arr.mean()
                    # Free memory immediately
                    del band_arr
                
                mean_value = np.float32(sum_mean / bands)
                metadata["applied_steps"].append("flat_field_correction")
                self.logger.info(f"[hsp] _apply_preprocessing_streaming: mean_value={mean_value}")
            
            # 3. Get atmospheric correction factor if needed
            if atmospheric_config.get("enabled", False):
                method = atmospheric_config.get("method", "simplified")
                if method == "simplified":
                    atm_correction = np.float32(atmospheric_config.get("correction_factor", 0.95))
                    metadata["applied_steps"].append("atmospheric_correction")
                    self.logger.info(f"[hsp] _apply_preprocessing_streaming: atm_correction={atm_correction}")
                else:
                    self.logger.warning(f"Atmospheric correction method '{method}' not implemented, skipping")
            
            # 4. Process bands one by one
            tiff_paths = []
            bands_processed = 0
            log_interval = max(1, bands // 10)  # Log every 10% or at least every band
            
            self.logger.info("[hsp] _apply_preprocessing_streaming: processing bands")
            
            for i, band_arr in self._iter_bands(input_path):
                # Log progress
                if bands_processed % log_interval == 0 or bands_processed == bands - 1:
                    self.logger.info(f"[hsp] Streaming band {bands_processed+1}/{bands}")
                
                # Apply preprocessing steps in order
                
                # 1. Dark current subtraction
                if dark_value is not None:
                    band_arr -= dark_value  # In-place subtraction
                
                # 2. Flat field correction
                if mean_value is not None and mean_value > 0:
                    band_arr /= mean_value  # In-place division
                
                # 3. Radiometric correction
                if radiometric_config.get("method") == "empirical_line":
                    gain = np.float32(radiometric_config.get("gain", 1.0))
                    offset = np.float32(radiometric_config.get("offset", 0.0))
                    band_arr *= gain  # In-place multiplication
                    band_arr += offset  # In-place addition
                    if "radiometric_correction" not in metadata["applied_steps"]:
                        metadata["applied_steps"].append("radiometric_correction")
                
                # 4. Atmospheric correction
                if atm_correction is not None:
                    band_arr *= atm_correction  # In-place multiplication
                
                # 5. Noise filtering (per-band 2-D filters only)
                if noise_config.get("method"):
                    method = noise_config.get("method")
                    
                    # Try to use scipy if available
                    try:
                        from scipy.ndimage import median_filter, uniform_filter
                        scipy_available = True
                    except ImportError:
                        scipy_available = False
                        self.logger.info("scipy not available, noise filtering skipped")
                    
                    if method == "savgol" and scipy_available:
                        try:
                            from scipy.signal import savgol_filter
                            window_length = noise_config.get("savgol_window", 11)
                            polyorder = noise_config.get("savgol_polyorder", 3)
                            # Ensure window_length is odd and less than array size
                            window_length = min(window_length, band_arr.shape[0], band_arr.shape[1])
                            if window_length % 2 == 0:
                                window_length -= 1
                            if window_length >= 3:
                                band_arr = savgol_filter(band_arr, window_length, polyorder)
                                if "noise_filtering_savgol" not in metadata["applied_steps"]:
                                    metadata["applied_steps"].append("noise_filtering_savgol")
                            else:
                                # Fallback to median filter
                                band_arr = median_filter(band_arr, size=3)
                                if "noise_filtering_median" not in metadata["applied_steps"]:
                                    metadata["applied_steps"].append("noise_filtering_median")
                        except Exception as e:
                            self.logger.warning(f"Error applying Savitzky-Golay filter: {e}, falling back to median filter")
                            band_arr = median_filter(band_arr, size=3)
                            if "noise_filtering_median" not in metadata["applied_steps"]:
                                metadata["applied_steps"].append("noise_filtering_median")
                    elif method == "median" and scipy_available:
                        band_arr = median_filter(band_arr, size=3)
                        if "noise_filtering_median" not in metadata["applied_steps"]:
                            metadata["applied_steps"].append("noise_filtering_median")
                    elif method == "mean" and scipy_available:
                        band_arr = uniform_filter(band_arr, size=3)
                        if "noise_filtering_mean" not in metadata["applied_steps"]:
                            metadata["applied_steps"].append("noise_filtering_mean")
                    elif method == "gaussian" and scipy_available:
                        from scipy.ndimage import gaussian_filter
                        sigma = noise_config.get("gaussian_sigma", 1.0)
                        band_arr = gaussian_filter(band_arr, sigma=sigma)
                        if "noise_filtering_gaussian" not in metadata["applied_steps"]:
                            metadata["applied_steps"].append("noise_filtering_gaussian")
                    elif method in ("pca", "mnf"):
                        # Method is not implemented - log warning and skip noise reduction
                        self.logger.warning(f"noise_reduction.method='{method}' is not implemented yet; skipping noise reduction")
                    else:
                        # Other methods or scipy not available - skip
                        self.logger.info(f"Skipping noise reduction method '{method}'")
                
                # 6. Normalization
                if spectral_config.get("normalization", False):
                    min_val = band_arr.min()
                    max_val = band_arr.max()
                    span = max_val - min_val
                    if span > 0:
                        band_arr -= min_val  # In-place subtraction
                        band_arr /= span     # In-place division
                    if "normalization_minmax" not in metadata["applied_steps"]:
                        metadata["applied_steps"].append("normalization_minmax")
                
                # Save band immediately
                tiff_path = self._save_single_band_tiff(band_arr, i, input_path, output_dir, metadata)
                tiff_paths.append(tiff_path)
                
                # Free memory
                del band_arr
                
                bands_processed += 1
            
            self.logger.info(f"[hsp] _apply_preprocessing_streaming: processed {bands_processed} bands")
            return tiff_paths, metadata
            
        finally:
            # Close dataset
            ds = None


