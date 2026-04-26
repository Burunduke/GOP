"""
Main pipeline for hyperspectral data processing
Science-oriented architecture without GUI dependencies
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .config import Config, get_config, create_config
from ..processing.hyperspectral import HyperspectralProcessor
from ..processing.orthophoto import OrthophotoProcessor
from ..utils.logger import setup_logger
from ..utils.image_type import detect_image_type
from ..utils.gdal_utils import get_raster_metadata
import tempfile
from osgeo import gdal
import os

# Type aliases for better type safety
PipelineResult = Dict[str, Any]
BandData = NDArray[np.float32]
ImageData = NDArray[np.uint8]
ProcessingResult = Dict[str, Union[str, BandData, ImageData, Dict[str, Any]]]

# Constants for magic numbers
CORRELATION_THRESHOLD = 0.7
Z_SCORE_THRESHOLD = 1.96  # p < 0.05
MIN_DATA_POINTS_FOR_CORRELATION = 100


class Pipeline:
    """
    Main pipeline class for hyperspectral data processing
    Science-oriented architecture for data processing and plant analysis
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_instance: Optional[Config] = None,
    ):
        """
        Initialize the pipeline

        Args:
            config_path: Path to configuration file
            config_instance: Optional configuration instance for dependency injection
        """
        # Load configuration with dependency injection support
        if config_instance is not None:
            self.config = config_instance
        elif config_path:
            self.config = create_config(config_path)
        else:
            self.config = get_config()

        # Setup logging
        self.logger = setup_logger(
            name="GOP",
            level=self.config.get("logging.level", "INFO"),
            log_file=self.config.get("logging.file"),
        )

        # Initialize components
        self.hyperspectral_processor = HyperspectralProcessor()
        self.orthophoto_processor = OrthophotoProcessor()

        # Processing results
        self.results = {}

        self.logger.info("GOP scientific pipeline initialized")

    def _prepare_rgb_inputs(self, input_path: str, work_dir: str) -> Dict[str, Any]:
        """
        Prepare RGB inputs for orthophoto processor.
        
        RGB images skip hyperspectral preprocessing because they only have 3–4 bands.
        
        Args:
            input_path: Path to input file or directory
            work_dir: Working directory for temporary files
            
        Returns:
            Dictionary with tiff_paths and metadata for orthophoto processor
        """
        self.logger.info(f"[pipeline] Preparing RGB inputs from: {input_path}")
        
        # Get list of input files
        if os.path.isdir(input_path):
            # Get all files in directory
            input_files = []
            for f in os.listdir(input_path):
                file_path = os.path.join(input_path, f)
                if os.path.isfile(file_path):
                    input_files.append(file_path)
        else:
            input_files = [input_path]
        
        # Filter to only image files
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".geotiff"}
        image_files = []
        for file_path in input_files:
            _, ext = os.path.splitext(file_path)
            if ext.lower() in image_extensions:
                image_files.append(file_path)
        
        if not image_files:
            raise ValueError("No image files found in input")
        
        self.logger.info(f"[pipeline] Found {len(image_files)} image files")
        
        # Convert non-GeoTIFF files to temporary GeoTIFFs
        tiff_paths = []
        rgb_work_dir = os.path.join(work_dir, "rgb_converted")
        os.makedirs(rgb_work_dir, exist_ok=True)
        
        for i, file_path in enumerate(image_files):
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext in {".tif", ".tiff", ".geotiff"}:
                # Already a GeoTIFF, use as-is
                tiff_paths.append(file_path)
                self.logger.info(f"[pipeline] Using existing GeoTIFF: {file_path}")
            else:
                # Convert PNG/JPG/JPEG to temporary GeoTIFF
                tiff_path = os.path.join(rgb_work_dir, f"converted_{i:03d}.tif")
                self.logger.info(f"[pipeline] Converting {ext} to GeoTIFF: {tiff_path}")
                
                # Use gdal.Translate to convert
                gdal.Translate(tiff_path, file_path, format="GTiff")
                tiff_paths.append(tiff_path)
        
        # Build metadata
        metadata = {
            "source_files": image_files,
            "applied_steps": [],  # No preprocessing steps for RGB
        }
        
        # Try to get metadata from first file if possible
        try:
            if image_files:
                first_file = image_files[0]
                file_metadata = get_raster_metadata(first_file)
                metadata.update({
                    "width": file_metadata.get("width"),
                    "height": file_metadata.get("height"),
                    "band_count": file_metadata.get("bands"),
                    "dtype": file_metadata.get("bands_metadata", {}).get("band_1", {}).get("data_type") if file_metadata.get("bands_metadata") else None,
                    "crs": file_metadata.get("projection"),
                    "transform": file_metadata.get("geotransform"),
                })
        except Exception as e:
            self.logger.warning(f"[pipeline] Could not extract metadata from source files: {e}")
        
        self.logger.info(f"[pipeline] Prepared {len(tiff_paths)} TIFF files for orthophoto processing")
        return {
            "tiff_paths": tiff_paths,
            "metadata": metadata
        }

    def process(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        sensor_type: Optional[str] = None,
        stitching_method: Optional[str] = None,
    ) -> PipelineResult:
        """
        Complete data processing cycle with scientific methodology

        Args:
            input_path: Path to input data
            output_dir: Directory for saving results
            sensor_type: Sensor type ('rgb', 'hyperspectral', or None for auto-detection)
            stitching_method: Orthophoto stitching method ('gdal', 'opencv', 'odm')

        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"Starting scientific data processing: {input_path}")

            # Setup output directory
            output_dir = output_dir or self.config.get("output.results_dir", "results")
            os.makedirs(output_dir, exist_ok=True)

            # Auto-detect sensor type if not provided
            if sensor_type is None:
                if os.path.isdir(input_path):
                    # Get first file in directory for detection
                    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
                    if files:
                        first_file_path = os.path.join(input_path, files[0])
                        sensor_type = detect_image_type(first_file_path)
                    else:
                        sensor_type = "hyperspectral"  # Default fallback
                else:
                    sensor_type = detect_image_type(input_path)

            self.logger.info(f"Processing with sensor type: {sensor_type}")

            # Branch based on sensor type
            if sensor_type == "rgb":
                # RGB images skip hyperspectral preprocessing because they only have 3–4 bands
                self.logger.info("Stage 1: RGB processing (skipping hyperspectral preprocessing)")
                processed_data = self._prepare_rgb_inputs(input_path, output_dir)
            elif sensor_type == "hyperspectral" or sensor_type is None:
                # Hyperspectral processing (existing behavior)
                self.logger.info("Stage 1: Hyperspectral data preprocessing")
                processed_data = self._preprocess_hyperspectral(input_path, output_dir)
            else:
                raise ValueError(f"Unknown sensor_type: {sensor_type}")

            # Stage 2: Orthophoto creation (same for both paths)
            self.logger.info("Stage 2: Orthophoto creation")
            orthophoto_path = self._create_orthophoto(processed_data, output_dir, stitching_method)

            # Collect results
            self.results = {
                "input_path": input_path,
                "output_dir": output_dir,
                "sensor_type": sensor_type,
                "processed_data": processed_data,
                "orthophoto_path": orthophoto_path,
                "processing_metadata": self._get_processing_metadata(),
            }

            self.logger.info("Scientific processing completed successfully")
            return self.results

        except Exception as e:
            self.logger.error(f"Error in scientific processing: {e}")
            raise

    def _preprocess_hyperspectral(
        self, input_path: str, output_dir: str
    ) -> Dict[str, Any]:
        """
        Preprocess hyperspectral data

        Args:
            input_path: Path to input data
            output_dir: Directory for saving results

        Returns:
            Dictionary with preprocessing results
        """
        return self.hyperspectral_processor.process(input_path, output_dir)

    def _create_orthophoto(
        self, processed_data: Dict[str, Any], output_dir: str, stitching_method: Optional[str] = None
    ) -> str:
        """
        Create orthophoto

        Args:
            processed_data: Preprocessing results
            output_dir: Directory for saving results
            stitching_method: Orthophoto stitching method ('gdal', 'opencv', 'odm')

        Returns:
            Path to created orthophoto
        """
        # If stitching method is provided, temporarily override the processor's method
        # Note: The config uses "processing.orthophoto.stitching_method" while project overrides
        # use "orthophoto.stitching_method" - both paths are handled by the pipeline executor
        # and this temporary override mechanism maintains backward compatibility.
        original_method = None
        if stitching_method is not None:
            original_method = self.orthophoto_processor.stitching_method
            self.orthophoto_processor.stitching_method = stitching_method
        
        try:
            return self.orthophoto_processor.create_orthophoto(processed_data, output_dir)
        finally:
            # Restore original method if we changed it
            if original_method is not None:
                self.orthophoto_processor.stitching_method = original_method


    def _get_processing_metadata(self) -> Dict[str, Any]:
        """Get processing metadata"""
        return {
            "pipeline_version": "2.0.0",
            "processing_date": datetime.now().isoformat(),
            "config_used": self.config.config,
            "scientific_methods": [
                "hyperspectral_data_loading",
                "orthophoto_creation",
            ],
        }

    def get_results(self) -> Dict[str, Any]:
        """
        Get results of the last processing

        Returns:
            Dictionary with results
        """
        return self.results.copy()

    def save_results(self, results: Optional[Dict[str, Any]], output_path: str) -> None:
        """
        Save processing results to file

        Args:
            results: Optional results to save (if None, saves pipeline results)
            output_path: Path for saving results
        """
        try:
            # Use provided results or pipeline results
            results_to_save = results if results is not None else self.results
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load processing results from file

        Args:
            input_path: Path to load results from

        Returns:
            Dictionary with loaded results
        """
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            self.logger.info(f"Results loaded from: {input_path}")
            return results
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            raise

