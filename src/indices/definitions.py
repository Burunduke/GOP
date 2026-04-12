"""
Vegetation Index Definitions

This module provides comprehensive definitions for vegetation indices used in remote sensing
analysis. It includes formulas, band requirements, and calculation functions for various
vegetation indices across different sensor types (RGB, Multispectral, Hyperspectral).

References:
- Tucker, C.J. (1979). Red and photographic infrared linear combinations for monitoring vegetation.
- Gitelson, A.A., et al. (1996). Use of a green channel in remote sensing of global vegetation.
- Huete, A.R. (1988). A soil-adjusted vegetation index (SAVI).
- Gao, B.C. (1996). NDWI - A normalized difference water index.
- Penuelas, J., et al. (1995). The Photochemical Reflectance Index (PRI).
"""

import numpy as np
from typing import Dict, Callable, List, Tuple, Union
from src.utils.math_utils import safe_divide
from src.utils.exceptions import ValidationError


class IndexDefinitions:
    """
    Comprehensive vegetation index definitions with scientific formulas and calculations.

    This class provides definitions for over 20 vegetation indices across three main categories:
    - Greenness indices: Measure chlorophyll content and vegetation density
    - Stress indices: Detect plant stress, disease, and pigment changes
    - Water indices: Assess water content and moisture stress

    All indices include proper band wavelength mappings for RGB, Multispectral, and Hyperspectral sensors.
    """

    # Band wavelength mappings for different sensor types
    BAND_WAVELENGTHS = {
        "RGB": {
            "Blue": (450, 490),  # Blue channel ~470 nm
            "Green": (520, 580),  # Green channel ~550 nm
            "Red": (620, 680),  # Red channel ~650 nm
        },
        "Multispectral": {
            "Blue": (450, 490),  # Blue band ~470 nm
            "Green": (520, 580),  # Green band ~550 nm
            "Red": (620, 680),  # Red band ~650 nm
            "RedEdge": (690, 730),  # Red edge band ~720 nm
            "NIR": (760, 900),  # Near-infrared band ~850 nm
        },
        "Hyperspectral": {
            "Blue": (450, 490),  # Blue region ~470 nm
            "Green": (520, 580),  # Green region ~550 nm
            "Red": (620, 680),  # Red region ~650 nm
            "RedEdge": (690, 730),  # Red edge region ~720 nm
            "NIR": (760, 900),  # Near-infrared region ~850 nm
            "SWIR1": (1550, 1750),  # Short-wave infrared 1 ~1650 nm
            "SWIR2": (2080, 2350),  # Short-wave infrared 2 ~2200 nm
        },
    }

    # Greenness indices - measure chlorophyll content and vegetation density
    GREENNESS_INDICES = {
        "NDVI": {
            "name": "Normalized Difference Vegetation Index",
            "formula": "(NIR - Red) / (NIR + Red)",
            "description": "Standard vegetation index for biomass estimation and vegetation monitoring",
            "required_bands": ["NIR", "Red"],
            "range": (-1, 1),
            "category": "greenness",
            "function": lambda nir, red: safe_divide(nir - red, nir + red),
            "reference": "Tucker, C.J. (1979). Remote Sensing of Environment",
        },
        "GNDVI": {
            "name": "Green Normalized Difference Vegetation Index",
            "formula": "(NIR - Green) / (NIR + Green)",
            "description": "Enhanced vegetation index using green band for better chlorophyll sensitivity",
            "required_bands": ["NIR", "Green"],
            "range": (-1, 1),
            "category": "greenness",
            "function": lambda nir, green: safe_divide(nir - green, nir + green),
            "reference": "Gitelson, A.A., et al. (1996). Remote Sensing of Environment",
        },
        "MCARI": {
            "name": "Modified Chlorophyll Absorption Ratio Index",
            "formula": "((RedEdge - Red) - 0.2 * (RedEdge - Green)) * (RedEdge / Red)",
            "description": "Modified chlorophyll absorption index for leaf chlorophyll content estimation",
            "required_bands": ["RedEdge", "Red", "Green"],
            "range": (0, 1),
            "category": "greenness",
            "function": lambda red_edge, red, green: (
                (red_edge - red) - 0.2 * (red_edge - green)
            )
            * safe_divide(red_edge, red),
            "reference": "Daughtry, C.S.T., et al. (2000). Remote Sensing of Environment",
        },
        "MNLI": {
            "name": "Modified Non-Linear Index",
            "formula": "(NIR^2 - Red) / (NIR^2 + Red)",
            "description": "Non-linear vegetation index for improved sensitivity to high biomass",
            "required_bands": ["NIR", "Red"],
            "range": (-1, 1),
            "category": "greenness",
            "function": lambda nir, red: safe_divide(nir**2 - red, nir**2 + red),
            "reference": "Gitelson, A.A. (2004). Remote Sensing of Environment",
        },
        "OSAVI": {
            "name": "Optimized Soil Adjusted Vegetation Index",
            "formula": "(NIR - Red) / (NIR + Red + 0.16)",
            "description": "Soil-adjusted vegetation index optimized for minimal soil background influence",
            "required_bands": ["NIR", "Red"],
            "range": (-1, 1),
            "category": "greenness",
            "function": lambda nir, red: (nir - red) / (nir + red + 0.16),
            "reference": "Rondeaux, G., et al. (1996). Remote Sensing of Environment",
        },
        "TVI": {
            "name": "Triangular Vegetation Index",
            "formula": "0.5 * (120 * (NIR - Green) - 200 * (Red - Green))",
            "description": "Triangular index based on the area of the triangle formed by green, red, and NIR reflectance",
            "required_bands": ["NIR", "Green", "Red"],
            "range": (0, 1),
            "category": "greenness",
            "function": lambda nir, green, red: 0.5
            * (120 * (nir - green) - 200 * (red - green)),
            "reference": "Broge, N.H., & Leblanc, E. (2001). Remote Sensing of Environment",
        },
        "EVI": {
            "name": "Enhanced Vegetation Index",
            "formula": "2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)",
            "description": "Enhanced vegetation index with atmospheric resistance and soil adjustment",
            "required_bands": ["NIR", "Red", "Blue"],
            "range": (-1, 1),
            "category": "greenness",
            "function": lambda nir, red, blue: 2.5
            * (nir - red)
            / (nir + 6 * red - 7.5 * blue + 1),
            "reference": "Huete, A.R., et al. (2002). Remote Sensing of Environment",
        },
        "SAVI": {
            "name": "Soil Adjusted Vegetation Index",
            "formula": "(NIR - Red) / (NIR + Red + L) * (1 + L)",
            "description": "Soil-adjusted vegetation index with canopy background correction (L=0.5)",
            "required_bands": ["NIR", "Red"],
            "range": (-1, 1),
            "category": "greenness",
            "function": lambda nir, red: (nir - red) / (nir + red + 0.5) * 1.5,
            "reference": "Huete, A.R. (1988). Remote Sensing of Environment",
        },
        "MSAVI": {
            "name": "Modified Soil Adjusted Vegetation Index",
            "formula": "(2 * NIR + 1 - sqrt((2 * NIR + 1)^2 - 8 * (NIR - Red))) / 2",
            "description": "Modified SAVI with self-adjusting L factor for varying vegetation density",
            "required_bands": ["NIR", "Red"],
            "range": (-1, 1),
            "category": "greenness",
            "function": lambda nir, red: (
                2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))
            )
            / 2,
            "reference": "Qi, J., et al. (1994). Remote Sensing of Environment",
        },
    }

    # Stress indices - detect plant stress, disease, and pigment changes
    STRESS_INDICES = {
        "SIPI2": {
            "name": "Structure Insensitive Pigment Index 2",
            "formula": "(NIR - Blue) / (NIR - Red)",
            "description": "Pigment index insensitive to canopy structure variations",
            "required_bands": ["NIR", "Blue", "Red"],
            "range": (0, 2),
            "category": "stress",
            "function": lambda nir, blue, red: safe_divide(nir - blue, nir - red),
            "reference": "Penuelas, J., et al. (1995). Remote Sensing of Environment",
        },
        "mARI": {
            "name": "modified Anthocyanin Reflectance Index",
            "formula": "(Green - Red) / (Green + Red)",
            "description": "Modified index for anthocyanin pigment content estimation",
            "required_bands": ["Green", "Red"],
            "range": (-1, 1),
            "category": "stress",
            "function": lambda green, red: safe_divide(green - red, green + red),
            "reference": "Gitelson, A.A., et al. (2001). Journal of Plant Physiology",
        },
        "PRI": {
            "name": "Photochemical Reflectance Index",
            "formula": "(R531 - R570) / (R531 + R570)",
            "description": "Index for photosynthetic efficiency and light use efficiency assessment",
            "required_bands": ["Green", "Red"],  # Approximation using available bands
            "range": (-1, 1),
            "category": "stress",
            "function": lambda green, red: safe_divide(
                green - red, green + red
            ),  # Approximation
            "reference": "Gamon, J.A., et al. (1992). Remote Sensing of Environment",
        },
        "CRI": {
            "name": "Carotenoid Reflectance Index",
            "formula": "(1 / Blue) - (1 / Green)",
            "description": "Index for carotenoid pigment content estimation",
            "required_bands": ["Blue", "Green"],
            "range": (0, 10),
            "category": "stress",
            "function": lambda blue, green: safe_divide(1, blue)
            - safe_divide(1, green),
            "reference": "Gitelson, A.A., et al. (2002). Remote Sensing of Environment",
        },
        "ARI": {
            "name": "Anthocyanin Reflectance Index",
            "formula": "(1 / Green) - (1 / RedEdge)",
            "description": "Index for anthocyanin pigment content in leaves",
            "required_bands": ["Green", "RedEdge"],
            "range": (0, 10),
            "category": "stress",
            "function": lambda green, red_edge: safe_divide(1, green)
            - safe_divide(1, red_edge),
            "reference": "Gitelson, A.A., et al. (2001). Journal of Plant Physiology",
        },
        "PSRI": {
            "name": "Plant Senescence Reflectance Index",
            "formula": "(Red - Blue) / RedEdge",
            "description": "Index for plant senescence and carotenoid/chlorophyll ratio",
            "required_bands": ["Red", "Blue", "RedEdge"],
            "range": (-1, 1),
            "category": "stress",
            "function": lambda red, blue, red_edge: safe_divide(red - blue, red_edge),
            "reference": "Merzlyak, M.N., et al. (1999). Journal of Plant Physiology",
        },
        "NPCI": {
            "name": "Normalized Pigment Chlorophyll Index",
            "formula": "(Red - Blue) / (Red + Blue)",
            "description": "Normalized index for pigment composition analysis",
            "required_bands": ["Red", "Blue"],
            "range": (-1, 1),
            "category": "stress",
            "function": lambda red, blue: safe_divide(red - blue, red + blue),
            "reference": "Penuelas, J., et al. (1994). Remote Sensing of Environment",
        },
    }

    # Water indices - assess water content and moisture stress
    WATER_INDICES = {
        "NDWI": {
            "name": "Normalized Difference Water Index",
            "formula": "(Green - NIR) / (Green + NIR)",
            "description": "Standard water index for vegetation water content estimation",
            "required_bands": ["Green", "NIR"],
            "range": (-1, 1),
            "category": "water",
            "function": lambda green, nir: safe_divide(green - nir, green + nir),
            "reference": "Gao, B.C. (1996). Remote Sensing of Environment",
        },
        "MSI": {
            "name": "Moisture Stress Index",
            "formula": "NIR / SWIR",
            "description": "Index for vegetation water stress and moisture content",
            "required_bands": ["NIR", "SWIR1"],
            "range": (0, 3),
            "category": "water",
            "function": lambda nir, swir: safe_divide(nir, swir),
            "reference": "Rock, B.N., et al. (1986). Remote Sensing of Environment",
        },
        "WI": {
            "name": "Water Index",
            "formula": "NIR / Green",
            "description": "Simple water index using NIR and green bands",
            "required_bands": ["NIR", "Green"],
            "range": (0, 5),
            "category": "water",
            "function": lambda nir, green: safe_divide(nir, green),
            "reference": "Peñuelas, J., et al. (1993). Remote Sensing of Environment",
        },
        "NDII": {
            "name": "Normalized Difference Infrared Index",
            "formula": "(NIR - SWIR) / (NIR + SWIR)",
            "description": "Normalized difference index for water content using SWIR",
            "required_bands": ["NIR", "SWIR1"],
            "range": (-1, 1),
            "category": "water",
            "function": lambda nir, swir: safe_divide(nir - swir, nir + swir),
            "reference": "Hardisky, M.A., et al. (1983). Remote Sensing of Environment",
        },
        "GVMI": {
            "name": "Global Vegetation Moisture Index",
            "formula": "(NIR + 0.1) - (SWIR + 0.02) / (NIR + 0.1) + (SWIR + 0.02)",
            "description": "Global vegetation moisture index for large-scale applications",
            "required_bands": ["NIR", "SWIR1"],
            "range": (-1, 1),
            "category": "water",
            "function": lambda nir, swir: ((nir + 0.1) - (swir + 0.02))
            / ((nir + 0.1) + (swir + 0.02)),
            "reference": "Ceccato, P., et al. (2002). Remote Sensing of Environment",
        },
        "NDWI2": {
            "name": "Normalized Difference Water Index 2",
            "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
            "description": "Alternative water index using SWIR2 band",
            "required_bands": ["NIR", "SWIR2"],
            "range": (-1, 1),
            "category": "water",
            "function": lambda nir, swir2: safe_divide(nir - swir2, nir + swir2),
            "reference": "McFeeters, S.K. (1996). International Journal of Remote Sensing",
        },
    }

    # Combined dictionary of all indices
    ALL_INDICES = {**GREENNESS_INDICES, **STRESS_INDICES, **WATER_INDICES}

    # Index groups for categorization
    INDEX_GROUPS = {
        "greenness": list(GREENNESS_INDICES.keys()),
        "stress": list(STRESS_INDICES.keys()),
        "water": list(WATER_INDICES.keys()),
    }

    @classmethod
    def get_index_info(cls, index_name: str) -> Dict:
        """
        Get comprehensive information about a specific vegetation index.

        Args:
            index_name: Name of the vegetation index (e.g., 'NDVI', 'GNDVI')

        Returns:
            Dictionary containing index metadata including formula, description,
            required bands, range, category, and reference
        """
        return cls.ALL_INDICES.get(index_name, {})

    @classmethod
    def get_indices_by_group(cls, group: str) -> Dict[str, Dict]:
        """
        Get all indices belonging to a specific category.

        Args:
            group: Category name ('greenness', 'stress', 'water')

        Returns:
            Dictionary of indices in the specified category
        """
        if group == "greenness":
            return cls.GREENNESS_INDICES
        elif group == "stress":
            return cls.STRESS_INDICES
        elif group == "water":
            return cls.WATER_INDICES
        else:
            return {}

    @classmethod
    def get_available_indices(cls, sensor_type: str) -> List[str]:
        """
        Get list of indices available for a specific sensor type.

        Args:
            sensor_type: Type of sensor ('RGB', 'Multispectral', 'Hyperspectral')

        Returns:
            List of index names that can be calculated with the sensor's bands
        """
        # Get available bands for the sensor type
        available_bands = list(cls.BAND_WAVELENGTHS.get(sensor_type, {}).keys())
        available_indices = []

        for index_name, index_info in cls.ALL_INDICES.items():
            required_bands = index_info.get("required_bands", [])
            if all(band in available_bands for band in required_bands):
                available_indices.append(index_name)

        return available_indices

    @classmethod
    def calculate_index(
        cls, index_name: str, bands: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate vegetation index values from spectral bands.

        Args:
            index_name: Name of the vegetation index to calculate
            bands: Dictionary containing spectral band arrays

        Returns:
            Array containing calculated index values

        Raises:
            ValueError: If index is unknown or required bands are missing
        """
        index_info = cls.get_index_info(index_name)
        if not index_info:
            raise ValidationError(
                f"Unknown vegetation index: {index_name}",
                details={"index_name": index_name},
            )

        function = index_info.get("function")
        if not function:
            raise ValidationError(
                f"Missing calculation function for index: {index_name}",
                details={"index_name": index_name},
            )

        required_bands = index_info.get("required_bands", [])
        band_values = [bands.get(band) for band in required_bands]

        if any(band is None for band in band_values):
            missing_bands = [
                band
                for band, value in zip(required_bands, band_values)
                if value is None
            ]
            raise ValidationError(
                f"Missing bands for index {index_name}: {missing_bands}",
                details={"index_name": index_name, "missing_bands": missing_bands},
            )

        return function(*band_values)

    @classmethod
    def normalize_index(
        cls, index_name: str, values: np.ndarray, mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Normalize index values to 0-1 range based on index category and expected range.

        Args:
            index_name: Name of the vegetation index
            values: Array of index values to normalize
            mask: Optional mask array for region of interest

        Returns:
            Normalized values in 0-1 range
        """
        index_info = cls.get_index_info(index_name)
        value_range = index_info.get("range", (-1, 1))
        category = index_info.get("category", "unknown")

        # Apply mask if provided
        if mask is not None:
            masked_values = values[mask > 0]
        else:
            masked_values = values

        # Category-specific normalization
        if category == "greenness":
            # Greenness indices: values typically -1 to 1, normalize to 0-1
            normalized = np.clip(values, -1, 1) * 0.5 + 0.5
        elif category == "stress":
            # Stress indices: various ranges, use adaptive normalization
            min_val, max_val = value_range
            if min_val == -1 and max_val == 1:
                normalized = np.clip(values, -1, 1) * 0.5 + 0.5
            else:
                normalized = safe_divide(values - min_val, max_val - min_val)
                normalized = np.clip(normalized, 0, 1)
        elif category == "water":
            # Water indices: specific normalization based on index type
            if index_name == "NDWI":
                normalized = np.clip(values, -1, 1) * 0.5 + 0.5
            elif index_name == "MSI":
                normalized = 1 - np.clip(values, 0, 3) / 3
            elif index_name == "WI":
                normalized = np.clip(values, 0, 5) / 5
            else:
                normalized = np.clip(values, -1, 1) * 0.5 + 0.5
        else:
            # General normalization for unknown categories
            min_val, max_val = value_range
            normalized = safe_divide(values - min_val, max_val - min_val)
            normalized = np.clip(normalized, 0, 1)

        return normalized

    @classmethod
    def get_index_formula(cls, index_name: str) -> str:
        """
        Get the mathematical formula for a vegetation index.

        Args:
            index_name: Name of the vegetation index

        Returns:
            String containing the mathematical formula
        """
        index_info = cls.get_index_info(index_name)
        return index_info.get("formula", "")

    @classmethod
    def get_index_description(cls, index_name: str) -> str:
        """
        Get the description and purpose of a vegetation index.

        Args:
            index_name: Name of the vegetation index

        Returns:
            String containing the index description
        """
        index_info = cls.get_index_info(index_name)
        return index_info.get("description", "")

    @classmethod
    def get_index_reference(cls, index_name: str) -> str:
        """
        Get the scientific reference for a vegetation index.

        Args:
            index_name: Name of the vegetation index

        Returns:
            String containing the scientific reference
        """
        index_info = cls.get_index_info(index_name)
        return index_info.get("reference", "")

    @classmethod
    def get_band_wavelengths(
        cls, sensor_type: str, band_name: str
    ) -> Tuple[float, float]:
        """
        Get wavelength range for a specific band and sensor type.

        Args:
            sensor_type: Type of sensor ('RGB', 'Multispectral', 'Hyperspectral')
            band_name: Name of the spectral band

        Returns:
            Tuple containing (min_wavelength, max_wavelength) in nanometers
        """
        sensor_bands = cls.BAND_WAVELENGTHS.get(sensor_type, {})
        return sensor_bands.get(band_name, (0, 0))

    @classmethod
    def validate_bands_for_index(
        cls, index_name: str, available_bands: List[str]
    ) -> bool:
        """
        Validate if available bands are sufficient for calculating an index.

        Args:
            index_name: Name of the vegetation index
            available_bands: List of available band names

        Returns:
            True if bands are sufficient, False otherwise
        """
        index_info = cls.get_index_info(index_name)
        if not index_info:
            return False

        required_bands = index_info.get("required_bands", [])
        return all(band in available_bands for band in required_bands)

    @classmethod
    def get_all_indices_info(cls) -> Dict[str, Dict]:
        """
        Get comprehensive information about all available vegetation indices.

        Returns:
            Dictionary containing metadata for all indices
        """
        return cls.ALL_INDICES

    @classmethod
    def get_indices_by_sensor_compatibility(cls, sensor_type: str) -> Dict[str, Dict]:
        """
        Get all indices compatible with a specific sensor type.

        Args:
            sensor_type: Type of sensor ('RGB', 'Multispectral', 'Hyperspectral')

        Returns:
            Dictionary of compatible indices with their metadata
        """
        available_indices = cls.get_available_indices(sensor_type)
        return {idx: cls.ALL_INDICES[idx] for idx in available_indices}


# Convenience functions for common operations
def calculate_multiple_indices(
    indices: List[str], bands: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Calculate multiple vegetation indices simultaneously.

    Args:
        indices: List of index names to calculate
        bands: Dictionary containing spectral band arrays

    Returns:
        Dictionary mapping index names to calculated values
    """
    results = {}
    for index_name in indices:
        try:
            results[index_name] = IndexDefinitions.calculate_index(index_name, bands)
        except (ValueError, ZeroDivisionError) as e:
            print(f"Warning: Could not calculate {index_name}: {e}")
            results[index_name] = None

    return results


def get_index_categories() -> Dict[str, List[str]]:
    """
    Get all index categories and their associated indices.

    Returns:
        Dictionary mapping category names to lists of index names
    """
    return IndexDefinitions.INDEX_GROUPS


if __name__ == "__main__":
    # Example usage and testing
    print("Available vegetation indices:")
    for category, indices in IndexDefinitions.INDEX_GROUPS.items():
        print(f"\n{category.upper()} INDICES ({len(indices)}):")
        for idx in indices:
            info = IndexDefinitions.get_index_info(idx)
            print(f"  - {idx}: {info['name']}")

    print(f"\nTotal indices: {len(IndexDefinitions.ALL_INDICES)}")

    # Test sensor compatibility
    for sensor in ["RGB", "Multispectral", "Hyperspectral"]:
        available = IndexDefinitions.get_available_indices(sensor)
        print(f"\n{sensor} sensor: {len(available)} indices available")
        print(f"  Indices: {', '.join(available)}")
