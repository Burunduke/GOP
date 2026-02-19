"""
GUI модуль для GOP - Гиперспектральная обработка и анализ растений
"""

__version__ = "1.0.0"
__author__ = "Индыков Дмитрий Андреевич"
__email__ = "indykovdm@example.com"
__description__ = "Веб-интерфейс для GOP - Гиперспектральная обработка и анализ растений"

from .app import create_app

__all__ = ["create_app", "__version__"]