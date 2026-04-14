"""
GOP - Hyperspectral Processing and Plant Analysis

Package for processing hyperspectral data and creating orthophotos.
"""

__version__ = "2.0.0"
__author__ = "Dmitry Indykov"
__email__ = "indykovdm@example.com"

from .core import Pipeline, get_config, create_config
from .processing import HyperspectralProcessor

__all__ = [
    "Pipeline",
    "HyperspectralProcessor",
    "get_config",
    "create_config",
]
