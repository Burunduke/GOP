"""
Data processing modules for the GOP project.

This module provides processors for hyperspectral and orthophoto data processing.
"""

from .hyperspectral import HyperspectralProcessor
from .orthophoto import OrthophotoProcessor

__all__ = ["HyperspectralProcessor", "OrthophotoProcessor"]
