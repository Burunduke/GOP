"""
Service layer for GOP GUI application
"""

from .gop_adapter import GOPAdapter
from .project_manager import ProjectManager
from .pipeline_executor import PipelineExecutor

__all__ = ["GOPAdapter", "ProjectManager", "PipelineExecutor"]