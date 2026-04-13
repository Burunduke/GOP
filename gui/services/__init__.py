"""
Сервисный слой для GUI приложения GOP
"""

from .gop_adapter import GOPAdapter
from .session_manager import SessionManager
from .cache_manager import CacheManager
from .project_manager import ProjectManager
from .pipeline_executor import PipelineExecutor

__all__ = ["GOPAdapter", "SessionManager", "CacheManager", "ProjectManager", "PipelineExecutor"]