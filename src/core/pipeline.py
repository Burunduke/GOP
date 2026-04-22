"""
Main pipeline for hyperspectral data processing
Science-oriented architecture without GUI dependencies
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
from numpy.typing import NDArray

from .config import Config, get_config, create_config
from ..processing.hyperspectral import HyperspectralProcessor
from ..processing.orthophoto import OrthophotoProcessor
from ..utils.logger import setup_logger

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

    def process(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        sensor_type: str = "Hyperspectral",
    ) -> PipelineResult:
        """
        Complete data processing cycle with scientific methodology

        Args:
            input_path: Path to input data
            output_dir: Directory for saving results
            sensor_type: Sensor type ('RGB', 'Multispectral', 'Hyperspectral')
            segmentation_mask: Path to segmentation mask (if None, will be created)
            use_refinement: Use boundary refinement for segmentation
            compression_ratio: Compression ratio for segmentation

        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"Starting scientific data processing: {input_path}")

            # Setup output directory
            output_dir = output_dir or self.config.get("output.results_dir", "results")
            os.makedirs(output_dir, exist_ok=True)

            # Stage 1: Hyperspectral data preprocessing
            self.logger.info("Stage 1: Hyperspectral data preprocessing")
            processed_data = self._preprocess_hyperspectral(input_path, output_dir)

            # Stage 2: Orthophoto creation
            self.logger.info("Stage 2: Orthophoto creation")
            orthophoto_path = self._create_orthophoto(processed_data, output_dir)

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
        self, processed_data: Dict[str, Any], output_dir: str
    ) -> str:
        """
        Create orthophoto

        Args:
            processed_data: Preprocessing results
            output_dir: Directory for saving results

        Returns:
            Path to created orthophoto
        """
        return self.orthophoto_processor.create_orthophoto(processed_data, output_dir)


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

