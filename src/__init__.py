"""
GOP - Гиперспектральная обработка и анализ растений

Пакет для обработки гиперспектральных данных, создания ортофотопланов
и анализа состояния растений с использованием вегетационных индексов.
"""

__version__ = "2.0.0"
__author__ = "Индыков Дмитрий Андреевич"
__email__ = "indykovdm@example.com"

from .core import Pipeline
from .processing import HyperspectralProcessor
from .segmentation import ImageSegmenter
from .indices import VegetationIndexCalculator

__all__ = [
    'Pipeline',
    'HyperspectralProcessor',
    'ImageSegmenter',
    'VegetationIndexCalculator'
]