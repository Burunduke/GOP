"""Data models for GOP project management."""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
import json


class ProjectStatus(str, Enum):
    """Project status enumeration."""
    NEW = "new"                  # New project, no files uploaded
    READY = "ready"              # Files uploaded, ready for processing
    RUN = "run"                  # Processing in progress (replaces PROCESSING)
    DONE = "done"                # Processing completed (replaces COMPLETED)
    ERROR = "error"              # Processing error
    CANCELLED = "cancelled"      # Processing cancelled


class PipelineStage(str, Enum):
    """Pipeline processing stages enumeration."""
    PREPROCESSING = "preprocessing"
    ORTHOPHOTO = "orthophoto"


@dataclass
class ProjectFile:
    """File associated with a project."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    original_name: str = ""
    file_path: str = ""
    file_size: int = 0
    file_type: str = ""  # "hyperspectral", "orthophoto", "auxiliary"
    upload_date: str = field(default_factory=lambda: datetime.now().isoformat())
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the file
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectFile":
        """Create from dictionary representation.
        
        Args:
            data: Dictionary containing file data
            
        Returns:
            ProjectFile instance
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingConfig:
    """Project processing configuration."""
    stages: List[str] = field(default_factory=lambda: [s.value for s in PipelineStage])
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "radiometric_correction": True,
        "atmospheric_correction": True
    })
    orthophoto: Dict[str, Any] = field(default_factory=lambda: {
        "resolution": 0.1,
        "crs": "EPSG:4326"
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingConfig":
        """Create from dictionary representation.
        
        Args:
            data: Dictionary containing configuration data
            
        Returns:
            ProcessingConfig instance
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingResult:
    """Processing stage result."""
    stage: str = ""
    status: str = "pending"  # pending, running, completed, error, skipped
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the result
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingResult":
        """Create from dictionary representation.
        
        Args:
            data: Dictionary containing result data
            
        Returns:
            ProcessingResult instance
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingHistory:
    """Processing history record."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    status: str = "running"  # running, completed, error, cancelled
    config: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the history
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingHistory":
        """Create from dictionary representation.
        
        Args:
            data: Dictionary containing history data
            
        Returns:
            ProcessingHistory instance
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Project:
    """Main project model."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: str = ProjectStatus.NEW.value
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    files: List[Dict[str, Any]] = field(default_factory=list)
    processing_config: Dict[str, Any] = field(default_factory=lambda: ProcessingConfig().to_dict())
    current_stage: Optional[str] = None
    progress: float = 0.0  # 0.0 to 100.0
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the project
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Create from dictionary representation.
        
        Args:
            data: Dictionary containing project data
            
        Returns:
            Project instance
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_json(self) -> str:
        """Convert to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Project":
        """Create from JSON string.
        
        Args:
            json_str: JSON string containing project data
            
        Returns:
            Project instance
        """
        return cls.from_dict(json.loads(json_str))
    
    def get_file_count(self) -> int:
        """Get number of files in project.
        
        Returns:
            Number of files
        """
        return len(self.files)
    
    def get_total_file_size(self) -> int:
        """Get total size of all files in project.
        
        Returns:
            Total file size in bytes
        """
        return sum(f.get("file_size", 0) for f in self.files)
    
    def get_status_display(self) -> str:
        """Get display name for status.
        
        Returns:
            Display name for the status
        """
        status_names = {
            ProjectStatus.NEW.value: "New",
            ProjectStatus.READY.value: "Ready for processing",
            ProjectStatus.RUN.value: "Run",
            ProjectStatus.DONE.value: "Done",
            ProjectStatus.ERROR.value: "Error",
            ProjectStatus.CANCELLED.value: "Cancelled",
        }
        return status_names.get(self.status, self.status)
    
    def get_status_color(self) -> str:
        """Get color for status display.
        
        Returns:
            Color name for the status
        """
        status_colors = {
            ProjectStatus.NEW.value: "secondary",
            ProjectStatus.READY.value: "info",
            ProjectStatus.RUN.value: "warning",
            ProjectStatus.DONE.value: "success",
            ProjectStatus.ERROR.value: "danger",
            ProjectStatus.CANCELLED.value: "dark",
        }
        return status_colors.get(self.status, "secondary")