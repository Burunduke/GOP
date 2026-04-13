"""Менеджер проектов GOP - управление жизненным циклом проектов."""

import json
import os
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from gui.models.project import (
    Project, ProjectFile, ProcessingConfig, 
    ProcessingResult, ProcessingHistory, ProjectStatus, PipelineStage
)

logger = logging.getLogger(__name__)


class ProjectManager:
    """Менеджер для управления проектами GOP."""
    
    def __init__(self, projects_dir: str = "projects"):
        """
        Инициализация менеджера проектов.
        
        Args:
            projects_dir: Путь к директории хранения проектов
        """
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Project] = {}
        self._load_all_projects()
    
    def _load_all_projects(self) -> None:
        """Загрузка всех проектов из файловой системы в кэш."""
        self._cache.clear()
        if not self.projects_dir.exists():
            return
        
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir():
                project_file = project_dir / "project.json"
                if project_file.exists():
                    try:
                        with open(project_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        project = Project.from_dict(data)
                        self._cache[project.id] = project
                    except (json.JSONDecodeError, Exception) as e:
                        logger.error(f"Ошибка загрузки проекта из {project_file}: {e}")
    
    def _save_project(self, project: Project) -> None:
        """Сохранение проекта в файловую систему."""
        project_dir = self.projects_dir / project.id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаём поддиректории
        (project_dir / "files").mkdir(exist_ok=True)
        (project_dir / "results").mkdir(exist_ok=True)
        
        project_file = project_dir / "project.json"
        with open(project_file, "w", encoding="utf-8") as f:
            f.write(project.to_json())
        
        self._cache[project.id] = project
    
    # === CRUD операции ===
    
    def create_project(self, name: str, description: str = "", tags: list[str] = None) -> Project:
        """
        Создание нового проекта.
        
        Args:
            name: Название проекта
            description: Описание проекта
            tags: Теги проекта
            
        Returns:
            Созданный проект
        """
        project = Project(
            name=name,
            description=description,
            tags=tags or [],
        )
        self._save_project(project)
        logger.info(f"Создан проект: {project.name} (ID: {project.id})")
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Получение проекта по ID.
        
        Args:
            project_id: ID проекта
            
        Returns:
            Проект или None если не найден
        """
        return self._cache.get(project_id)
    
    def update_project(self, project_id: str, **kwargs) -> Optional[Project]:
        """
        Обновление полей проекта.
        
        Args:
            project_id: ID проекта
            **kwargs: Поля для обновления
            
        Returns:
            Обновлённый проект или None
        """
        project = self.get_project(project_id)
        if project is None:
            logger.warning(f"Проект не найден: {project_id}")
            return None
        
        for key, value in kwargs.items():
            if hasattr(project, key):
                setattr(project, key, value)
        
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        logger.info(f"Обновлён проект: {project.name} (ID: {project.id})")
        return project
    
    def delete_project(self, project_id: str) -> bool:
        """
        Удаление проекта.
        
        Args:
            project_id: ID проекта
            
        Returns:
            True если проект удалён
        """
        project = self.get_project(project_id)
        if project is None:
            return False
        
        project_dir = self.projects_dir / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
        
        self._cache.pop(project_id, None)
        logger.info(f"Удалён проект: {project.name} (ID: {project_id})")
        return True
    
    def list_projects(
        self, 
        status: Optional[str] = None,
        sort_by: str = "updated_at",
        reverse: bool = True
    ) -> list[Project]:
        """
        Получение списка проектов с фильтрацией.
        
        Args:
            status: Фильтр по статусу
            sort_by: Поле для сортировки
            reverse: Обратная сортировка
            
        Returns:
            Список проектов
        """
        projects = list(self._cache.values())
        
        if status:
            projects = [p for p in projects if p.status == status]
        
        projects.sort(key=lambda p: getattr(p, sort_by, ""), reverse=reverse)
        return projects
    
    def search_projects(self, query: str) -> list[Project]:
        """
        Поиск проектов по имени и описанию.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Список найденных проектов
        """
        query_lower = query.lower()
        return [
            p for p in self._cache.values()
            if query_lower in p.name.lower() or query_lower in p.description.lower()
        ]
    
    # === Управление файлами ===
    
    def add_file_to_project(
        self, 
        project_id: str, 
        filename: str, 
        file_content: bytes,
        file_type: str = "hyperspectral"
    ) -> Optional[ProjectFile]:
        """
        Добавление файла к проекту.
        
        Args:
            project_id: ID проекта
            filename: Имя файла
            file_content: Содержимое файла
            file_type: Тип файла
            
        Returns:
            Объект ProjectFile или None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        project_file = ProjectFile(
            filename=filename,
            original_name=filename,
            file_type=file_type,
            file_size=len(file_content),
            checksum=hashlib.md5(file_content).hexdigest(),
        )
        
        # Сохраняем файл на диск
        files_dir = self.projects_dir / project_id / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        file_path = files_dir / f"{project_file.id}_{filename}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        project_file.file_path = str(file_path)
        
        # Обновляем проект
        project.files.append(project_file.to_dict())
        
        # Обновляем статус если были файлы
        if project.status == ProjectStatus.NEW.value and len(project.files) > 0:
            project.status = ProjectStatus.READY.value
        
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        
        logger.info(f"Добавлен файл {filename} к проекту {project.name}")
        return project_file
    
    def remove_file_from_project(self, project_id: str, file_id: str) -> bool:
        """
        Удаление файла из проекта.
        
        Args:
            project_id: ID проекта
            file_id: ID файла
            
        Returns:
            True если файл удалён
        """
        project = self.get_project(project_id)
        if project is None:
            return False
        
        file_data = None
        for f in project.files:
            if f.get("id") == file_id:
                file_data = f
                break
        
        if file_data is None:
            return False
        
        # Удаляем файл с диска
        file_path = file_data.get("file_path", "")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Удаляем из списка
        project.files = [f for f in project.files if f.get("id") != file_id]
        
        # Обновляем статус
        if len(project.files) == 0 and project.status == ProjectStatus.READY.value:
            project.status = ProjectStatus.NEW.value
        
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        
        logger.info(f"Удалён файл {file_id} из проекта {project.name}")
        return True
    
    # === Управление обработкой ===
    
    def update_processing_config(
        self, project_id: str, config: dict
    ) -> Optional[Project]:
        """
        Обновление конфигурации обработки проекта.
        
        Args:
            project_id: ID проекта
            config: Новая конфигурация
            
        Returns:
            Обновлённый проект или None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        # Мержим с существующей конфигурацией
        current_config = project.processing_config
        current_config.update(config)
        project.processing_config = current_config
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        return project
    
    def start_processing(self, project_id: str) -> Optional[ProcessingHistory]:
        """
        Начало обработки проекта.
        
        Args:
            project_id: ID проекта
            
        Returns:
            Запись истории обработки или None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        if project.status == ProjectStatus.PROCESSING.value:
            logger.warning(f"Проект {project.name} уже обрабатывается")
            return None
        
        # Создаём запись истории
        history = ProcessingHistory(
            config=project.processing_config,
        )
        
        # Создаём директорию для результатов
        results_dir = self.projects_dir / project_id / "results" / history.run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Обновляем проект
        project.status = ProjectStatus.PROCESSING.value
        project.progress = 0.0
        project.current_stage = PipelineStage.PREPROCESSING.value
        project.processing_history.append(history.to_dict())
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        
        logger.info(f"Начата обработка проекта {project.name} (run: {history.run_id})")
        return history
    
    def update_processing_progress(
        self, 
        project_id: str, 
        stage: str, 
        progress: float,
        stage_result: Optional[dict] = None
    ) -> Optional[Project]:
        """
        Обновление прогресса обработки.
        
        Args:
            project_id: ID проекта
            stage: Текущий этап
            progress: Прогресс (0-100)
            stage_result: Результат этапа
            
        Returns:
            Обновлённый проект или None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        project.current_stage = stage
        project.progress = progress
        
        # Обновляем результат этапа в последней записи истории
        if stage_result and project.processing_history:
            last_history = project.processing_history[-1]
            last_history.setdefault("results", [])
            
            # Обновляем или добавляем результат этапа
            updated = False
            for i, r in enumerate(last_history["results"]):
                if r.get("stage") == stage:
                    last_history["results"][i] = stage_result
                    updated = True
                    break
            if not updated:
                last_history["results"].append(stage_result)
        
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        return project
    
    def complete_processing(
        self, 
        project_id: str, 
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Optional[Project]:
        """
        Завершение обработки проекта.
        
        Args:
            project_id: ID проекта
            success: Успешно ли завершена обработка
            error_message: Сообщение об ошибке
            
        Returns:
            Обновлённый проект или None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        now = datetime.now().isoformat()
        
        if success:
            project.status = ProjectStatus.COMPLETED.value
            project.progress = 100.0
        else:
            project.status = ProjectStatus.ERROR.value
        
        project.current_stage = None
        
        # Обновляем последнюю запись истории
        if project.processing_history:
            last_history = project.processing_history[-1]
            last_history["end_time"] = now
            last_history["status"] = "completed" if success else "error"
            last_history["error_message"] = error_message
            
            # Вычисляем длительность
            try:
                start = datetime.fromisoformat(last_history["start_time"])
                end = datetime.fromisoformat(now)
                last_history["total_duration_seconds"] = (end - start).total_seconds()
            except (ValueError, KeyError):
                pass
        
        project.updated_at = now
        self._save_project(project)
        
        status_text = "успешно" if success else f"с ошибкой: {error_message}"
        logger.info(f"Обработка проекта {project.name} завершена {status_text}")
        return project
    
    def cancel_processing(self, project_id: str) -> Optional[Project]:
        """
        Отмена обработки проекта.
        
        Args:
            project_id: ID проекта
            
        Returns:
            Обновлённый проект или None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        if project.status != ProjectStatus.PROCESSING.value:
            return project
        
        now = datetime.now().isoformat()
        project.status = ProjectStatus.CANCELLED.value
        project.current_stage = None
        
        if project.processing_history:
            last_history = project.processing_history[-1]
            last_history["end_time"] = now
            last_history["status"] = "cancelled"
        
        project.updated_at = now
        self._save_project(project)
        
        logger.info(f"Обработка проекта {project.name} отменена")
        return project
    
    # === Статистика ===
    
    def get_statistics(self) -> dict:
        """
        Получение общей статистики по проектам.
        
        Returns:
            Словарь со статистикой
        """
        projects = list(self._cache.values())
        total = len(projects)
        
        status_counts = {}
        for status in ProjectStatus:
            status_counts[status.value] = sum(
                1 for p in projects if p.status == status.value
            )
        
        total_files = sum(p.get_file_count() for p in projects)
        total_size = sum(p.get_total_file_size() for p in projects)
        
        return {
            "total_projects": total,
            "status_counts": status_counts,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size > 0 else 0,
        }
    
    def get_recent_projects(self, limit: int = 5) -> list[Project]:
        """
        Получение последних проектов.
        
        Args:
            limit: Максимальное количество
            
        Returns:
            Список последних проектов
        """
        projects = self.list_projects(sort_by="updated_at", reverse=True)
        return projects[:limit]