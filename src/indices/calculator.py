"""
Vegetation Index Calculator

This module provides a comprehensive calculator for vegetation indices from remote sensing data.
It supports RGB, multispectral, and hyperspectral sensors with proper band mapping and validation.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from numpy.typing import NDArray

# Type aliases for better type safety
IndexResult = Dict[str, Union[str, NDArray[np.float32], Dict[str, Any]]]
BandData = NDArray[np.float32]
IndexData = NDArray[np.float32]

try:
    from osgeo import gdal

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    # Don't raise error here to allow tests to run

from .definitions import IndexDefinitions
from ..core.config import Config, get_config, create_config
from ..utils.logger import setup_logger
from ..utils.gdal_utils import open_gdal_dataset
from ..utils.exceptions import ProcessingError, ValidationError


# Constants for band indices and thresholds
DEFAULT_COMPRESSION_RATIO = 0.125
DEFAULT_NDVI_THRESHOLD = 0.2
PLANT_CONDITION_EXCELLENT_THRESHOLD = 0.7
PLANT_CONDITION_SATISFACTORY_THRESHOLD = 0.4

# Band indices for hyperspectral sensors (approximate wavelengths)
HYPERSPECTRAL_BAND_INDICES = {
    "Blue": 10,    # ~450 nm
    "Green": 20,   # ~550 nm
    "Red": 30,     # ~650 nm
    "RedEdge": 35, # ~720 nm
    "NIR": 50,     # ~800 nm
    "SWIR1": 80,   # ~1600 nm
    "SWIR2": 100,  # ~2200 nm
}

class VegetationIndexCalculator:
    """
    Calculator for vegetation indices from remote sensing data.
    
    This class provides methods to calculate various vegetation indices from
    RGB, multispectral, and hyperspectral imagery with proper band mapping,
    validation, and result normalization.
    """

    def __init__(self, config_instance: Optional[Config] = None):
        """
        Initialize the vegetation index calculator.

        Args:
            config_instance: Optional configuration instance for dependency injection
        """
        self.config = get_config(config_instance)
        self.logger = setup_logger("VegetationIndexCalculator")
        self.definitions = IndexDefinitions()

    def calculate(
        self,
        orthophoto_path: str,
        segmentation_mask: str,
        sensor_type: str = "Hyperspectral",
        selected_indices: Optional[List[str]] = None,
        output_dir: str = "results",
    ) -> IndexResult:
        """
        Calculate vegetation indices from orthophoto and segmentation mask.

        Args:
            orthophoto_path: Path to the orthophoto
            segmentation_mask: Path to the segmentation mask
            sensor_type: Sensor type ('RGB', 'Multispectral', 'Hyperspectral')
            selected_indices: List of indices to calculate
            output_dir: Directory for saving results

        Returns:
            Dictionary with index calculation results

        Raises:
            FileNotFoundError: If input files are not found
            ValidationError: If no indices are available for the sensor type
        """
        try:
            self.logger.info(
                f"Starting vegetation index calculation for sensor: {sensor_type}"
            )

            # Validate input files
            if not os.path.exists(orthophoto_path):
                raise FileNotFoundError(f"Orthophoto not found: {orthophoto_path}")

            if not os.path.exists(segmentation_mask):
                raise FileNotFoundError(
                    f"Segmentation mask not found: {segmentation_mask}"
                )

            # Determine available indices
            if selected_indices is None:
                selected_indices = self.config.get("indices.default_indices", [])

            available_indices = self.definitions.get_available_indices(sensor_type)
            indices_to_calculate = [
                idx for idx in selected_indices if idx in available_indices
            ]

            if not indices_to_calculate:
                raise ValidationError(
                    f"No available indices for sensor: {sensor_type}",
                    details={"sensor_type": sensor_type},
                )

            self.logger.info(f"Calculating indices: {indices_to_calculate}")

            # Read data
            image_data = self._read_image_data(orthophoto_path, sensor_type)
            mask_data = self._read_mask_data(segmentation_mask)

            # Extract spectral bands
            bands = self._extract_bands(image_data, sensor_type)

            # Calculate indices
            indices_results = {}
            normalized_indices = {}

            for index_name in indices_to_calculate:
                self.logger.info(f"Calculating index: {index_name}")

                # Calculate index values
                index_values = self.definitions.calculate_index(index_name, bands)

                # Normalize values
                normalized_values = self.definitions.normalize_index(
                    index_name, index_values, mask_data
                )

                indices_results[index_name] = index_values
                normalized_indices[index_name] = normalized_values

                # Save index
                self._save_index(index_values, index_name, output_dir, orthophoto_path)
                self._save_index(
                    normalized_values,
                    f"{index_name}_normalized",
                    output_dir,
                    orthophoto_path,
                )

            # Comprehensive plant condition assessment
            plant_condition = self._calculate_plant_condition(
                normalized_indices, mask_data
            )

            # Save plant condition assessment
            self._save_plant_condition(plant_condition, output_dir, orthophoto_path)

            results = {
                "sensor_type": sensor_type,
                "calculated_indices": indices_to_calculate,
                "indices_values": indices_results,
                "normalized_indices": normalized_indices,
                "plant_condition": plant_condition,
                "output_dir": output_dir,
            }

            self.logger.info("Vegetation index calculation completed")
            return results

        except Exception as e:
            self.logger.error(f"Error calculating vegetation indices: {e}")
            raise

    def calculate_from_arrays(
        self,
        image_data: np.ndarray,
        mask_data: np.ndarray,
        sensor_type: str = "Hyperspectral",
        selected_indices: Optional[List[str]] = None,
        output_dir: str = "results",
    ) -> IndexResult:
        """
        Calculate vegetation indices directly from numpy arrays.
        
        Args:
            image_data: Image data array
            mask_data: Segmentation mask array
            sensor_type: Sensor type ('RGB', 'Multispectral', 'Hyperspectral')
            selected_indices: List of indices to calculate
            output_dir: Directory for saving results
            
        Returns:
            Dictionary with index calculation results
            
        Raises:
            ValidationError: If no indices are available for the sensor type
        """
        try:
            self.logger.info(
                f"Starting vegetation index calculation from arrays for sensor: {sensor_type}"
            )

            # Determine available indices
            if selected_indices is None:
                selected_indices = self.config.get("indices.default_indices", [])

            available_indices = self.definitions.get_available_indices(sensor_type)
            indices_to_calculate = [
                idx for idx in selected_indices if idx in available_indices
            ]

            if not indices_to_calculate:
                raise ValidationError(
                    f"No available indices for sensor: {sensor_type}",
                    details={"sensor_type": sensor_type},
                )

            self.logger.info(f"Calculating indices: {indices_to_calculate}")

            # Extract spectral bands
            bands = self._extract_bands(image_data, sensor_type)

            # Calculate indices
            indices_results = {}
            normalized_indices = {}

            for index_name in indices_to_calculate:
                self.logger.info(f"Calculating index: {index_name}")

                # Calculate index values
                index_values = self.definitions.calculate_index(index_name, bands)

                # Normalize values
                normalized_values = self.definitions.normalize_index(
                    index_name, index_values, mask_data
                )

                indices_results[index_name] = index_values
                normalized_indices[index_name] = normalized_values

                # Save index
                # Note: We don't save to file when calculating from arrays
                # as we don't have file paths

            # Comprehensive plant condition assessment
            plant_condition = self._calculate_plant_condition(
                normalized_indices, mask_data
            )

            # Note: We don't save plant condition to file when calculating from arrays
            # as we don't have file paths

            results = {
                "sensor_type": sensor_type,
                "calculated_indices": indices_to_calculate,
                "indices_values": indices_results,
                "normalized_indices": normalized_indices,
                "plant_condition": plant_condition,
                "output_dir": output_dir,
            }

            self.logger.info("Vegetation index calculation from arrays completed")
            return results

        except Exception as e:
            self.logger.error(f"Error calculating vegetation indices from arrays: {e}")
            raise

    def calculate_from_arrays_simple(
        self,
        image_data: np.ndarray,
        mask_data: np.ndarray,
        sensor_type: str = "Hyperspectral",
        selected_indices: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate vegetation indices directly from numpy arrays (simple format).
        
        Args:
            image_data: Image data array
            mask_data: Segmentation mask array
            sensor_type: Sensor type ('RGB', 'Multispectral', 'Hyperspectral')
            selected_indices: List of indices to calculate
            
        Returns:
            Dictionary with index names as keys and index values as values
            
        Raises:
            ValidationError: If no indices are available for the sensor type
        """
        try:
            self.logger.info(
                f"Starting vegetation index calculation from arrays (simple) for sensor: {sensor_type}"
            )

            # Determine available indices
            self.logger.info(f"DEBUG: selected_indices initial value: {selected_indices}")
            # Check if indices were explicitly set to None (meaning calculate all indices)
            if selected_indices is None:
                # Calculate all indices
                indices_to_calculate = self.definitions.get_all_indices()
                self.logger.info(f"DEBUG: selected_indices after None check: {selected_indices}")
            else:
                # Indices were specified, use them or get defaults from config
                if not selected_indices:  # Empty list
                    selected_indices = self.config.get("indices.default_indices", [])

                available_indices = self.definitions.get_available_indices(sensor_type)
                self.logger.info(f"DEBUG: available_indices: {available_indices}")
                self.logger.info(f"DEBUG: selected_indices truthiness: {bool(selected_indices)}")
                if selected_indices:
                    indices_to_calculate = [
                        idx for idx in selected_indices if idx in available_indices
                    ]
                    self.logger.info(f"DEBUG: Using filtered indices: {indices_to_calculate}")
                else:
                    # If no indices specified, calculate all available indices
                    # But for testing purposes, we want to try all indices that could potentially be calculated
                    indices_to_calculate = self.definitions.get_all_indices()
                    self.logger.info(f"DEBUG: All indices: {indices_to_calculate}")

            if not indices_to_calculate:
                raise ValidationError(
                    f"No available indices for sensor: {sensor_type}",
                    details={"sensor_type": sensor_type},
                )

            self.logger.info(f"Calculating indices: {indices_to_calculate}")

            # Extract spectral bands
            try:
                bands = self._extract_bands(image_data, sensor_type)
            except ValidationError as e:
                # Re-raise as ValueError for API compatibility
                raise ValueError(e.message) from e

            # Calculate indices
            results = {}

            for index_name in indices_to_calculate:
                self.logger.info(f"Calculating index: {index_name}")

                try:
                    # Calculate index values
                    index_values = self.definitions.calculate_index(index_name, bands)
                    
                    # Apply mask - set values to NaN where mask is 0
                    if mask_data is not None:
                        masked_index_values = index_values.copy()
                        masked_index_values[mask_data == 0] = np.nan
                        results[index_name] = masked_index_values
                    else:
                        results[index_name] = index_values
                except ValidationError as e:
                    self.logger.error(f"Skipping index {index_name} due to missing bands: {e}")
                    # If this is the only index and it failed, re-raise as ValueError
                    if len(indices_to_calculate) == 1:
                        raise ValueError(e.message) from e
                    continue

            self.logger.info("Vegetation index calculation from arrays (simple) completed")
            return results

        except Exception as e:
            self.logger.error(f"Error calculating vegetation indices from arrays (simple): {e}")
            raise

    def _read_image_data(self, image_path: str, sensor_type: str) -> np.ndarray:
        """
        Read image data from file.

        Args:
            image_path: Path to the image file
            sensor_type: Sensor type

        Returns:
            Image data array

        Raises:
            Exception: If image reading fails
        """
        try:
            from ..utils.gdal_utils import read_raster_bands

            # Read all channels using centralized utility
            image_data = read_raster_bands(image_path)

            self.logger.info(f"Image loaded: {image_data.shape}")
            return image_data

        except Exception as e:
            self.logger.error(f"Error reading image: {e}")
            raise

    def _read_mask_data(self, mask_path: str) -> np.ndarray:
        """
        Read mask data from file.

        Args:
            mask_path: Path to the mask file

        Returns:
            Mask data array

        Raises:
            Exception: If mask reading fails
        """
        try:
            from ..utils.gdal_utils import read_raster_band

            # Read first channel using centralized utility
            mask_data = read_raster_band(mask_path, band_number=1)

            # Binarize mask
            mask_data = (mask_data > 0).astype(np.uint8)

            self.logger.info(
                f"Mask loaded: {mask_data.shape}, region pixels: {mask_data.sum()}"
            )
            return mask_data

        except Exception as e:
            self.logger.error(f"Error reading mask: {e}")
            raise

    def _extract_bands(
        self, image_data: np.ndarray, sensor_type: str
    ) -> Dict[str, np.ndarray]:
        """
        Extract spectral bands from image data based on sensor type.

        Args:
            image_data: Image data array
            sensor_type: Sensor type ('RGB', 'Multispectral', 'Hyperspectral')

        Returns:
            Dictionary with spectral bands

        Raises:
            ValidationError: If required bands are missing
        """
        bands = {}

        if sensor_type == "RGB":
            # RGB: 3 channels (B, G, R)
            if image_data.shape[2] >= 3:
                bands["Blue"] = image_data[:, :, 0]
                bands["Green"] = image_data[:, :, 1]
                bands["Red"] = image_data[:, :, 2]

        elif sensor_type == "Multispectral":
            # Multispectral: 5 channels
            if image_data.shape[2] >= 5:
                bands["Blue"] = image_data[:, :, 0]
                bands["Green"] = image_data[:, :, 1]
                bands["Red"] = image_data[:, :, 2]
                bands["RedEdge"] = image_data[:, :, 3]
                bands["NIR"] = image_data[:, :, 4]

        elif sensor_type == "Hyperspectral":
            # Hyperspectral: select channels based on wavelength mapping
            if image_data.shape[2] >= 100:
                # Use predefined band indices for typical hyperspectral sensors
                for band_name, band_index in HYPERSPECTRAL_BAND_INDICES.items():
                    if band_index < image_data.shape[2]:
                        bands[band_name] = image_data[:, :, band_index]

        # Check for missing required bands
        missing_bands = [name for name in bands.keys() if bands[name] is None]
        if missing_bands:
            raise ValidationError(
                f"Missing bands: {missing_bands}",
                details={"missing_bands": missing_bands},
            )

        self.logger.info(f"Extracted bands: {list(bands.keys())}")
        return bands

    def _save_index(
        self,
        index_data: np.ndarray,
        index_name: str,
        output_dir: str,
        reference_path: str,
    ) -> None:
        """
        Save index data to file.

        Args:
            index_data: Index data array
            index_name: Name of the index
            output_dir: Directory for saving
            reference_path: Path to reference image

        Raises:
            Exception: If saving fails
        """
        try:
            from ..utils.gdal_utils import write_raster

            # Create directory for indices
            indices_dir = os.path.join(output_dir, "indices")
            os.makedirs(indices_dir, exist_ok=True)

            output_path = os.path.join(indices_dir, f"{index_name}.tif")

            # Save using centralized utility
            write_raster(index_data, output_path, source_path=reference_path)

            self.logger.debug(f"Index saved: {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving index {index_name}: {e}")
            raise

    def _calculate_plant_condition(
        self, normalized_indices: Dict[str, np.ndarray], mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive plant condition assessment.

        Args:
            normalized_indices: Normalized index values
            mask: Region mask

        Returns:
            Dictionary with plant condition assessment

        Raises:
            Exception: If calculation fails
        """
        try:
            # Define index groups for condition assessment
            greenness_indices = ["GNDVI", "MCARI", "MNLI", "OSAVI", "TVI"]
            stress_indices = ["SIPI2", "mARI"]
            water_indices = ["NDWI", "MSI"]

            # Calculate mean values for each group
            greenness_values = []
            for idx in greenness_indices:
                if idx in normalized_indices:
                    values = normalized_indices[idx][mask > 0]
                    greenness_values.append(values)

            stress_values = []
            for idx in stress_indices:
                if idx in normalized_indices:
                    values = normalized_indices[idx][mask > 0]
                    stress_values.append(values)

            water_values = []
            for idx in water_indices:
                if idx in normalized_indices:
                    values = normalized_indices[idx][mask > 0]
                    water_values.append(values)

            # Calculate composite condition maps
            condition_maps = {}

            if greenness_values:
                condition_maps["greenness"] = np.mean(greenness_values, axis=0)

            if stress_values:
                condition_maps["stress"] = np.mean(stress_values, axis=0)

            if water_values:
                condition_maps["water"] = np.mean(water_values, axis=0)

            # Overall assessment
            if condition_maps:
                overall_values = list(condition_maps.values())
                condition_maps["overall"] = np.mean(overall_values, axis=0)

            # Statistics
            statistics = {}
            for name, values in condition_maps.items():
                statistics[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

            result = {"condition_maps": condition_maps, "statistics": statistics}

            self.logger.info("Comprehensive plant condition assessment calculated")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating plant condition assessment: {e}")
            raise

    def _save_plant_condition(
        self, plant_condition: Dict[str, Any], output_dir: str, reference_path: str
    ) -> None:
        """
        Save plant condition assessment results.

        Args:
            plant_condition: Plant condition assessment results
            output_dir: Directory for saving
            reference_path: Path to reference image

        Raises:
            Exception: If saving fails
        """
        try:
            condition_maps = plant_condition.get("condition_maps", {})

            for name, data in condition_maps.items():
                output_path = os.path.join(
                    output_dir, "indices", f"plant_condition_{name}.tif"
                )
                self._save_index(
                    data, f"plant_condition_{name}", output_dir, reference_path
                )

            self.logger.info("Plant condition assessment saved")

        except Exception as e:
            self.logger.error(f"Error saving plant condition assessment: {e}")
            raise

    def assess_plant_condition(self, indices_results: Dict[str, Any], mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Assess plant condition based on calculated indices.

        Args:
            indices_results: Results from index calculation
            mask: Optional segmentation mask (if not provided, will be created from first index)

        Returns:
            Dictionary with plant condition assessment
        """
        try:
            # Handle both simple and structured result formats
            if "normalized_indices" in indices_results:
                # Structured format from calculate_from_arrays
                normalized_indices = indices_results["normalized_indices"]
            else:
                # Simple format from calculate_from_arrays_simple - need to normalize
                normalized_indices = {}
                # Use the first available index to create a mask if not provided
                if mask is None and indices_results:
                    first_index = list(indices_results.values())[0]
                    mask = (first_index > 0).astype(np.uint8)
                
                for index_name, index_values in indices_results.items():
                    try:
                        normalized_indices[index_name] = self.definitions.normalize_index(
                            index_name, index_values, mask
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not normalize index {index_name}: {e}")
                        # Use raw values if normalization fails
                        normalized_indices[index_name] = index_values

            if not normalized_indices:
                raise ValueError("No normalized indices available for plant condition assessment")

            # Use provided mask or create mask based on first index
            if mask is None:
                first_index = list(normalized_indices.values())[0]
                mask = (first_index > 0).astype(np.uint8)

            # Calculate comprehensive assessment
            plant_condition = self._calculate_plant_condition(normalized_indices, mask)

            # Classify condition
            overall_stats = plant_condition["statistics"].get("overall", {})
            overall_mean = overall_stats.get("mean", 0)

            if overall_mean > PLANT_CONDITION_EXCELLENT_THRESHOLD:
                condition_class = "Excellent"
                condition_color = "green"
            elif overall_mean > PLANT_CONDITION_SATISFACTORY_THRESHOLD:
                condition_class = "Satisfactory"
                condition_color = "yellow"
            else:
                condition_class = "Poor"
                condition_color = "red"

            plant_condition["classification"] = {
                "class": condition_class,
                "color": condition_color,
                "score": overall_mean,
            }
            
            # Include indices in result for backward compatibility
            plant_condition["indices"] = normalized_indices

            return plant_condition

        except ValueError:
            # Re-raise ValueError as expected by the API
            raise
        except Exception as e:
            self.logger.error(f"Error assessing plant condition: {e}")
            return {"error": str(e)}

    def get_index_statistics(
        self, index_path: str, mask_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for an index file.

        Args:
            index_path: Path to the index file
            mask_path: Path to mask file (optional)

        Returns:
            Dictionary with index statistics
        """
        try:
            from ..utils.gdal_utils import read_raster_band

            # Read index using centralized utility
            index_data = read_raster_band(index_path, band_number=1)

            # Apply mask if provided
            if mask_path and os.path.exists(mask_path):
                mask_data = read_raster_band(mask_path, band_number=1)
                index_data = index_data[mask_data > 0]

            # Calculate statistics
            statistics = {
                "count": int(np.count_nonzero(~np.isnan(index_data))),
                "mean": float(np.nanmean(index_data)),
                "std": float(np.nanstd(index_data)),
                "min": float(np.nanmin(index_data)),
                "max": float(np.nanmax(index_data)),
                "median": float(np.nanmedian(index_data)),
                "q25": float(np.nanpercentile(index_data, 25)),
                "q75": float(np.nanpercentile(index_data, 75)),
            }

            return statistics

        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {"error": str(e)}

    def save_results(self, results: Dict[str, Any], file_path: str, format: str = "json") -> None:
        """
        Save calculation results to a file.
        
        Args:
            results: Results dictionary to save
            file_path: Path to save the results
            format: Format to save in ('json' or 'numpy')
            
        Raises:
            Exception: If saving fails
        """
        try:
            if format == "numpy":
                import numpy as np
                
                # Convert numpy arrays to lists for numpy serialization
                def convert_arrays(obj):
                    import numpy as np  # Import numpy in the function scope
                    if isinstance(obj, np.ndarray):
                        return obj
                    elif isinstance(obj, dict):
                        return {key: convert_arrays(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_arrays(item) for item in obj]
                    else:
                        return obj
                
                serializable_results = convert_arrays(results)
                
                # Save as numpy file
                np.savez(file_path.replace('.json', '.npz') if file_path.endswith('.json') else file_path + '.npz', **serializable_results)
                self.logger.info(f"Results saved to {file_path.replace('.json', '.npz') if file_path.endswith('.json') else file_path + '.npz'}")
            else:
                import json
                
                # Convert numpy arrays to lists for JSON serialization
                def convert_arrays(obj):
                    import numpy as np  # Import numpy in the function scope
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_arrays(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_arrays(item) for item in obj]
                    else:
                        return obj
                
                serializable_results = convert_arrays(results)
                
                with open(file_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                    
                self.logger.info(f"Results saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

    def load_results(self, file_path: str) -> Dict[str, Any]:
        """
        Load calculation results from a JSON file.
        
        Args:
            file_path: Path to load the results from
            
        Returns:
            Dictionary with loaded results
            
        Raises:
            Exception: If loading fails
        """
        try:
            import json
            import numpy as np
            
            with open(file_path, 'r') as f:
                results = json.load(f)
                
            self.logger.info(f"Results loaded from {file_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            raise
