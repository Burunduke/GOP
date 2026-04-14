"""
Main pipeline for hyperspectral data processing
Science-oriented architecture without GUI dependencies
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
from numpy.typing import NDArray

from .config import Config, get_config, create_config
from ..indices.calculator import VegetationIndexCalculator
from ..processing.hyperspectral import HyperspectralProcessor
from ..processing.orthophoto import OrthophotoProcessor
from ..segmentation.segmenter import ImageSegmenter
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
        self.segmenter = ImageSegmenter()
        self.index_calculator = VegetationIndexCalculator()

        # Processing results
        self.results = {}

        self.logger.info("GOP scientific pipeline initialized")

    def process(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        sensor_type: str = "Hyperspectral",
        segmentation_mask: Optional[str] = None,
        selected_indices: Optional[List[str]] = None,
        use_refinement: bool = True,
        compression_ratio: Optional[float] = None,
    ) -> PipelineResult:
        """
        Complete data processing cycle with scientific methodology

        Args:
            input_path: Path to input data
            output_dir: Directory for saving results
            sensor_type: Sensor type ('RGB', 'Multispectral', 'Hyperspectral')
            segmentation_mask: Path to segmentation mask (if None, will be created)
            selected_indices: List of indices to calculate
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

            # Stage 3: High-resolution image segmentation
            self.logger.info("Stage 3: Image segmentation")
            if segmentation_mask is None:
                segmentation_mask = self._segment_image(
                    orthophoto_path, output_dir, use_refinement, compression_ratio
                )

            # Stage 4: Vegetation indices calculation
            self.logger.info("Stage 4: Vegetation indices calculation")
            indices_results = self._calculate_indices(
                orthophoto_path,
                segmentation_mask,
                sensor_type,
                selected_indices,
                output_dir,
            )

            # Stage 5: Comprehensive plant condition assessment
            self.logger.info("Stage 5: Comprehensive plant condition assessment")
            plant_condition = self._assess_plant_condition(indices_results)

            # Stage 6: Scientific analysis and statistics
            self.logger.info("Stage 6: Scientific analysis and statistics")
            scientific_analysis = self._perform_scientific_analysis(
                indices_results, plant_condition, output_dir
            )

            # Collect results
            self.results = {
                "input_path": input_path,
                "output_dir": output_dir,
                "sensor_type": sensor_type,
                "processed_data": processed_data,
                "orthophoto_path": orthophoto_path,
                "segmentation_mask": segmentation_mask,
                "indices": indices_results,
                "plant_condition": plant_condition,
                "scientific_analysis": scientific_analysis,
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

    def _segment_image(
        self,
        orthophoto_path: str,
        output_dir: str,
        use_refinement: bool = True,
        compression_ratio: Optional[float] = None,
    ) -> str:
        """
        Image segmentation using cascade approach

        Args:
            orthophoto_path: Path to orthophoto
            output_dir: Directory for saving results
            use_refinement: Use boundary refinement
            compression_ratio: Compression ratio

        Returns:
            Path to segmentation mask
        """
        return self.segmenter.segment(
            orthophoto_path, output_dir, use_refinement, compression_ratio
        )

    def _calculate_indices(
        self,
        orthophoto_path: str,
        segmentation_mask: str,
        sensor_type: str,
        selected_indices: Optional[List[str]],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Calculate vegetation indices

        Args:
            orthophoto_path: Path to orthophoto
            segmentation_mask: Path to segmentation mask
            sensor_type: Sensor type
            selected_indices: List of indices to calculate
            output_dir: Directory for saving results

        Returns:
            Dictionary with index calculation results
        """
        if selected_indices is None:
            selected_indices = self.config.get("indices.default_indices", [])

        return self.index_calculator.calculate(
            orthophoto_path,
            segmentation_mask,
            sensor_type,
            selected_indices,
            output_dir,
        )

    def _assess_plant_condition(
        self, indices_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive plant condition assessment

        Args:
            indices_results: Index calculation results

        Returns:
            Dictionary with plant condition assessment
        """
        return self.index_calculator.assess_plant_condition(indices_results)

    def _perform_scientific_analysis(
        self,
        indices_results: Dict[str, Any],
        plant_condition: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Scientific analysis of results

        Args:
            indices_results: Index calculation results
            plant_condition: Plant condition assessment
            output_dir: Directory for saving results

        Returns:
            Dictionary with scientific analysis
        """
        try:
            analysis = {}

            # Statistical analysis of indices
            analysis["index_statistics"] = self._analyze_index_statistics(
                indices_results
            )

            # Correlation analysis
            analysis["correlation_analysis"] = self._perform_correlation_analysis(
                indices_results
            )

            # Spatial analysis
            analysis["spatial_analysis"] = self._perform_spatial_analysis(
                plant_condition
            )

            # Plant condition classification
            analysis["plant_classification"] = self._classify_plant_condition(
                plant_condition
            )

            # Save scientific report
            self._save_scientific_report(analysis, output_dir)

            return analysis

        except Exception as e:
            self.logger.error(f"Error in scientific analysis: {e}")
            return {"error": str(e)}

    def _analyze_index_statistics(
        self, indices_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Statistical analysis of vegetation indices

        Args:
            indices_results: Index calculation results

        Returns:
            Dictionary with statistics
        """
        statistics = {}
        normalized_indices = indices_results.get("normalized_indices", {})

        for index_name, index_data in normalized_indices.items():
            if isinstance(index_data, np.ndarray):
                # Calculate statistics only for masked area
                valid_data = index_data[index_data > 0]

                if len(valid_data) > 0:
                    statistics[index_name] = {
                        "mean": float(np.mean(valid_data)),
                        "std": float(np.std(valid_data)),
                        "min": float(np.min(valid_data)),
                        "max": float(np.max(valid_data)),
                        "median": float(np.median(valid_data)),
                        "q25": float(np.percentile(valid_data, 25)),
                        "q75": float(np.percentile(valid_data, 75)),
                        "skewness": float(self._calculate_skewness(valid_data)),
                        "kurtosis": float(self._calculate_kurtosis(valid_data)),
                    }

        return statistics

    def _perform_correlation_analysis(
        self, indices_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Correlation analysis of indices

        Args:
            indices_results: Index calculation results

        Returns:
            Dictionary with correlation analysis
        """
        try:
            normalized_indices = indices_results.get("normalized_indices", {})

            # Create data matrix for correlation analysis
            index_names = []
            index_vectors = []

            for index_name, index_data in normalized_indices.items():
                if isinstance(index_data, np.ndarray):
                    valid_data = index_data[index_data > 0]
                    if len(valid_data) > MIN_DATA_POINTS_FOR_CORRELATION:
                        index_names.append(index_name)
                        index_vectors.append(valid_data)

            if len(index_vectors) < 2:
                return {"error": "Insufficient data for correlation analysis"}

            # Align vectors to minimum length
            min_length = min(len(vec) for vec in index_vectors)
            aligned_vectors = [vec[:min_length] for vec in index_vectors]

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(aligned_vectors)

            # Format results
            correlation_analysis = {
                "index_names": index_names,
                "correlation_matrix": correlation_matrix.tolist(),
                "strong_correlations": self._find_strong_correlations(
                    index_names, correlation_matrix, threshold=CORRELATION_THRESHOLD
                ),
            }

            return correlation_analysis

        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            return {"error": str(e)}

    def _perform_spatial_analysis(
        self, plant_condition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Spatial analysis of plant condition

        Args:
            plant_condition: Plant condition assessment

        Returns:
            Dictionary with spatial analysis
        """
        try:
            condition_maps = plant_condition.get("condition_maps", {})
            spatial_analysis = {}

            for condition_name, condition_data in condition_maps.items():
                if isinstance(condition_data, np.ndarray):
                    spatial_analysis[condition_name] = {
                        "spatial_autocorrelation": self._calculate_morans_i(
                            condition_data
                        ),
                        "hotspot_analysis": self._perform_hotspot_analysis(
                            condition_data
                        ),
                        "fragmentation_index": self._calculate_fragmentation_index(
                            condition_data
                        ),
                    }

            return spatial_analysis

        except Exception as e:
            self.logger.error(f"Error in spatial analysis: {e}")
            return {"error": str(e)}

    def _classify_plant_condition(
        self, plant_condition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Plant condition classification

        Args:
            plant_condition: Plant condition assessment

        Returns:
            Dictionary with classification
        """
        try:
            statistics = plant_condition.get("statistics", {})
            overall_stats = statistics.get("overall", {})

            if not overall_stats:
                return {"error": "No data available for classification"}

            overall_mean = overall_stats.get("mean", 0)
            overall_std = overall_stats.get("std", 0)

            # Scientific classification based on mean value and variability
            if overall_mean > 0.8 and overall_std < 0.1:
                condition_class = "Excellent"
                condition_description = "Plants in excellent condition, high uniformity"
                confidence = 0.9
            elif overall_mean > 0.6 and overall_std < 0.2:
                condition_class = "Good"
                condition_description = "Plants in good condition, moderate uniformity"
                confidence = 0.8
            elif overall_mean > 0.4:
                condition_class = "Satisfactory"
                condition_description = "Plants in satisfactory condition, some issues present"
                confidence = 0.7
            else:
                condition_class = "Poor"
                condition_description = "Plants in poor condition, intervention required"
                confidence = 0.8

            return {
                "class": condition_class,
                "description": condition_description,
                "confidence": confidence,
                "overall_score": overall_mean,
                "variability": overall_std,
            }

        except Exception as e:
            self.logger.error(f"Error in classification: {e}")
            return {"error": str(e)}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate distribution skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate distribution kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _find_strong_correlations(
        self,
        index_names: List[str],
        correlation_matrix: np.ndarray,
        threshold: float = CORRELATION_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Find strong correlations between indices"""
        strong_correlations = []

        for i in range(len(index_names)):
            for j in range(i + 1, len(index_names)):
                corr_value = correlation_matrix[i, j]
                if abs(corr_value) > threshold:
                    strong_correlations.append(
                        {
                            "index1": index_names[i],
                            "index2": index_names[j],
                            "correlation": float(corr_value),
                            "type": "positive" if corr_value > 0 else "negative",
                        }
                    )

        return strong_correlations

    def _calculate_morans_i(self, data: np.ndarray) -> float:
        """Calculate Moran's I spatial autocorrelation index"""
        try:
            # More efficient implementation using scipy if available
            try:
                from scipy.spatial.distance import pdist, squareform
                from scipy.sparse import lil_matrix
                
                rows, cols = data.shape
                if rows < 3 or cols < 3:
                    return 0.0
                
                # Flatten the data and create coordinates
                n = rows * cols
                flat_data = data.flatten()
                
                # Create coordinate grid
                coords = np.array([(i, j) for i in range(rows) for j in range(cols)])
                
                # Create spatial weights matrix (queen contiguity)
                weights = lil_matrix((n, n))
                
                for idx, (i, j) in enumerate(coords):
                    # Check 8-neighborhood
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                neighbor_idx = ni * cols + nj
                                weights[idx, neighbor_idx] = 1
                
                weights = weights.tocsr()
                
                # Calculate Moran's I
                mean_val = np.mean(flat_data)
                deviations = flat_data - mean_val
                
                # Sum of weights
                weight_sum = weights.sum()
                if weight_sum == 0:
                    return 0.0
                
                # Calculate numerator and denominator
                numerator = np.sum(weights.multiply(np.outer(deviations, deviations)))
                denominator = np.sum(deviations ** 2)
                
                if denominator == 0:
                    return 0.0
                
                morans_i = (n / weight_sum) * (numerator / denominator)
                return float(morans_i)
                
            except ImportError:
                # Fallback to simplified implementation if scipy not available
                return self._calculate_morans_i_simple(data)
                
        except Exception:
            return 0.0

    def _calculate_morans_i_simple(self, data: np.ndarray) -> float:
        """Simplified Moran's I implementation for when scipy is not available"""
        try:
            rows, cols = data.shape
            if rows < 3 or cols < 3:
                return 0.0

            # Create weight matrix (neighborhood) - optimized version
            n = rows * cols
            mean_val = np.mean(data)

            numerator = 0
            denominator = 0
            weight_sum = 0

            # Optimized implementation with reduced loops
            for i in range(rows):
                for j in range(cols):
                    deviation_i = data[i, j] - mean_val
                    denominator += deviation_i ** 2
                    
                    # Check neighboring pixels
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                deviation_j = data[ni, nj] - mean_val
                                numerator += deviation_i * deviation_j
                                weight_sum += 1

            if weight_sum == 0 or denominator == 0:
                return 0.0

            morans_i = (n / weight_sum) * (numerator / denominator)
            return float(morans_i)

        except Exception:
            return 0.0

    def _perform_hotspot_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Hotspot analysis"""
        try:
            # Simplified hotspot analysis based on z-scores
            mean_val = np.mean(data)
            std_val = np.std(data)

            if std_val == 0:
                return {"hotspots": 0, "coldspots": 0, "neutral": data.size}

            # Pixel classification
            z_scores = (data - mean_val) / std_val

            hotspots = np.sum(z_scores > Z_SCORE_THRESHOLD)  # p < 0.05
            coldspots = np.sum(z_scores < -Z_SCORE_THRESHOLD)  # p < 0.05
            neutral = data.size - hotspots - coldspots

            return {
                "hotspots": int(hotspots),
                "coldspots": int(coldspots),
                "neutral": int(neutral),
                "hotspot_percentage": float(hotspots / data.size * 100),
                "coldspot_percentage": float(coldspots / data.size * 100),
            }

        except Exception:
            return {"hotspots": 0, "coldspots": 0, "neutral": data.size}

    def _calculate_fragmentation_index(self, data: np.ndarray) -> float:
        """Calculate fragmentation index"""
        try:
            # Data binarization
            threshold = np.mean(data)
            binary = (data > threshold).astype(np.uint8)

            # Count connected components
            from scipy import ndimage

            labeled, num_features = ndimage.label(binary)

            # Calculate fragmentation index
            if num_features == 0:
                return 0.0

            total_area = np.sum(binary)
            if total_area == 0:
                return 0.0

            fragmentation = num_features / total_area
            return float(fragmentation)

        except Exception:
            return 0.0

    def _save_scientific_report(
        self, analysis: Dict[str, Any], output_dir: str
    ) -> None:
        """Save scientific report"""
        try:
            report_path = os.path.join(output_dir, "scientific_report.json")

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Scientific report saved: {report_path}")

        except Exception as e:
            self.logger.error(f"Error saving scientific report: {e}")

    def _get_processing_metadata(self) -> Dict[str, Any]:
        """Get processing metadata"""
        return {
            "pipeline_version": "2.0.0",
            "processing_date": str(Path.cwd()),
            "config_used": self.config.config,
            "scientific_methods": [
                "radiometric_correction",
                "pca_denoising",
                "vegetation_indices",
                "spatial_analysis",
                "correlation_analysis",
                "statistical_analysis",
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

    def export_scientific_data(self, output_dir: str) -> None:
        """
        Export scientific data for further analysis

        Args:
            output_dir: Directory for export
        """
        try:
            import pandas as pd

            # Create export directory
            export_dir = os.path.join(output_dir, "scientific_export")
            os.makedirs(export_dir, exist_ok=True)

            # Export index statistics
            if "scientific_analysis" in self.results:
                analysis = self.results["scientific_analysis"]

                if "index_statistics" in analysis:
                    stats_df = pd.DataFrame(analysis["index_statistics"]).T
                    stats_path = os.path.join(export_dir, "index_statistics.csv")
                    stats_df.to_csv(stats_path)
                    self.logger.info(
                        f"Index statistics exported: {stats_path}"
                    )

                if "correlation_analysis" in analysis:
                    corr_data = analysis["correlation_analysis"]
                    if "correlation_matrix" in corr_data:
                        corr_df = pd.DataFrame(
                            corr_data["correlation_matrix"],
                            index=corr_data["index_names"],
                            columns=corr_data["index_names"],
                        )
                        corr_path = os.path.join(export_dir, "correlation_matrix.csv")
                        corr_df.to_csv(corr_path)
                        self.logger.info(
                            f"Correlation matrix exported: {corr_path}"
                        )

            self.logger.info(f"Scientific data exported to: {export_dir}")

        except Exception as e:
            self.logger.error(f"Error exporting scientific data: {e}")
