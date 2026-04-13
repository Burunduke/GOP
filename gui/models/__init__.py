"""
Модели данных для GUI приложения GOP
"""

from .project import (
    Project,
    ProjectFile,
    ProjectStatus,
    PipelineStage,
    ProcessingConfig,
    ProcessingResult,
    ProcessingHistory
)

__all__ = [
    "Project",
    "ProjectFile", 
    "ProjectStatus",
    "PipelineStage",
    "ProcessingConfig",
    "ProcessingResult",
    "ProcessingHistory"
]