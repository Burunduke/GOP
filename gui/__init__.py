"""
GUI module for GOP - Hyperspectral processing and plant analysis
"""

__version__ = "1.0.0"
__author__ = "Dmitry Indykov"
__email__ = "indykovdm@example.com"
__description__ = "Web interface for GOP - Hyperspectral processing and plant analysis"

from .app import create_app

__all__ = ["create_app", "__version__"]