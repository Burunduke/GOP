"""
Vegetation indices calculation modules.

This module provides classes and functions for calculating vegetation indices
from remote sensing data including RGB, multispectral, and hyperspectral imagery.
"""

from .calculator import VegetationIndexCalculator
from .definitions import IndexDefinitions

__all__ = ["VegetationIndexCalculator", "IndexDefinitions"]
