"""
GOP - Hyperspectral Processing and Plant Analysis

Package for processing hyperspectral data, creating orthophotos,
and analyzing plant condition using vegetation indices.
"""

__version__ = "2.0.0"
__author__ = "Dmitry Indykov"
__email__ = "indykovdm@example.com"

from .core import Pipeline, get_config, create_config
from .processing import HyperspectralProcessor
from .segmentation import ImageSegmenter
from .indices import VegetationIndexCalculator

__all__ = [
    "Pipeline",
    "HyperspectralProcessor",
    "ImageSegmenter",
    "VegetationIndexCalculator",
    "get_config",
    "create_config",
]
