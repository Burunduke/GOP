"""
Сервисный слой для GUI приложения GOP
"""

from .gop_adapter import GOPAdapter
from .session_manager import SessionManager
from .cache_manager import CacheManager

__all__ = ["GOPAdapter", "SessionManager", "CacheManager"]