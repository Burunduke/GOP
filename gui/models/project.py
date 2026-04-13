"""Модели данных для управления проектами GOP."""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import json


class ProjectStatus(str, Enum):
    """Статус проекта."""
    NEW = "new"                  # Новый проект, файлы не загружены
    READY = "ready"              # Файлы загружены, готов к обработке
    PROCESSING = "processing"    # Идёт обработка
    COMPLETED = "completed"      # Обработка завершена
    ERROR = "error"              # Ошибка обработки
    CANCELLED = "cancelled"      # Обработка отменена


class PipelineStage(str, Enum):
    """Этапы пайплайна обработки."""
    PREPROCESSING = "preprocessing"
    ORTHOPHOTO = "orthophoto"
    SEGMENTATION = "segmentation"
    INDICES = "indices"
    ASSESSMENT = "assessment"
    ANALYSIS = "analysis"


@dataclass
class ProjectFile:
    """Файл, связанный с проектом."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    original_name: str = ""
    file_path: str = ""
    file_size: int = 0
    file_type: str = ""  # "hyperspectral", "orthophoto", "auxiliary"
    upload_date: str = field(default_factory=lambda: datetime.now().isoformat())
    checksum: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProjectFile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingConfig:
    """Конфигурация обработки проекта."""
    stages: list[str] = field(default_factory=lambda: [s.value for s in PipelineStage])
    preprocessing: dict = field(default_factory=lambda: {
        "radiometric_correction": True,
        "atmospheric_correction": True,
        "denoising_method": "pca",
        "denoising_components": 10
    })
    orthophoto: dict = field(default_factory=lambda: {
        "resolution": 0.1,
        "crs": "EPSG:4326"
    })
    segmentation: dict = field(default_factory=lambda: {
        "model": "deeplabv3",
        "refinement": True,
        "min_area": 100
    })
    indices: dict = field(default_factory=lambda: {
        "selected_indices": ["NDVI", "GNDVI", "MCARI", "RENDVI"],
        "custom_indices": []
    })
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingResult:
    """Результат этапа обработки."""
    stage: str = ""
    status: str = "pending"  # pending, running, completed, error, skipped
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    output_files: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessingHistory:
    """Запись истории обработки."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    status: str = "running"  # running, completed, error, cancelled
    config: dict = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    total_duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingHistory":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Project:
    """Основная модель проекта."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: str = ProjectStatus.NEW.value
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    files: list[dict] = field(default_factory=list)
    processing_config: dict = field(default_factory=lambda: ProcessingConfig().to_dict())
    current_stage: Optional[str] = None
    progress: float = 0.0  # 0.0 to 100.0
    processing_history: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Project":
        return cls.from_dict(json.loads(json_str))
    
    def get_file_count(self) -> int:
        return len(self.files)
    
    def get_total_file_size(self) -> int:
        return sum(f.get("file_size", 0) for f in self.files)
    
    def get_status_display(self) -> str:
        """Возвращает отображаемое название статуса на русском."""
        status_names = {
            ProjectStatus.NEW.value: "Новый",
            ProjectStatus.READY.value: "Готов к обработке",
            ProjectStatus.PROCESSING.value: "Обработка",
            ProjectStatus.COMPLETED.value: "Завершён",
            ProjectStatus.ERROR.value: "Ошибка",
            ProjectStatus.CANCELLED.value: "Отменён",
        }
        return status_names.get(self.status, self.status)
    
    def get_status_color(self) -> str:
        """Возвращает цвет для статуса."""
        status_colors = {
            ProjectStatus.NEW.value: "secondary",
            ProjectStatus.READY.value: "info",
            ProjectStatus.PROCESSING.value: "warning",
            ProjectStatus.COMPLETED.value: "success",
            ProjectStatus.ERROR.value: "danger",
            ProjectStatus.CANCELLED.value: "dark",
        }
        return status_colors.get(self.status, "secondary")