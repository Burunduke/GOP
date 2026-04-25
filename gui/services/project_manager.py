"""GOP project manager - project lifecycle management."""

import json
import os
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from gui.models.project import (
    Project, ProjectFile,
    ProcessingHistory, ProjectStatus,
    PipelineStage
)
from gui.config import config

logger = logging.getLogger(__name__)


class ProjectManager:
    """Manager for GOP project management."""
    
    def __init__(self, projects_dir: str = "projects") -> None:
        """
        Initialize project manager.
        
        Args:
            projects_dir: Path to project storage directory
        """
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Project] = {}
        self._load_all_projects()
    
    def _load_all_projects(self) -> None:
        """Load all projects from filesystem into cache."""
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
                        logger.error(f"Error loading project from {project_file}: {e}")
    
    def _save_project(self, project: Project) -> None:
        """Save project to filesystem."""
        project_dir = self.projects_dir / project.id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (project_dir / "files").mkdir(exist_ok=True)
        (project_dir / "results").mkdir(exist_ok=True)
        
        project_file = project_dir / "project.json"
        with open(project_file, "w", encoding="utf-8") as f:
            f.write(project.to_json())
        
        self._cache[project.id] = project
    
    # === CRUD operations ===
    
    def create_project(self, name: str, description: str = "", tags: List[str] = None) -> Project:
        """
        Create a new project.
        
        Args:
            name: Project name
            description: Project description
            tags: Project tags
            
        Returns:
            Created project
        """
        project = Project(
            name=name,
            description=description,
            tags=tags or [],
        )
        self._save_project(project)
        logger.info(f"Created project: {project.name} (ID: {project.id})")
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get project by ID.
        
        Args:
            project_id: Project ID
            
        Returns:
            Project or None if not found
        """
        return self._cache.get(project_id)
    
    def update_project(self, project_id: str, **kwargs) -> Optional[Project]:
        """
        Update project fields.
        
        Args:
            project_id: Project ID
            **kwargs: Fields to update
            
        Returns:
            Updated project or None
        """
        project = self.get_project(project_id)
        if project is None:
            logger.warning(f"Project not found: {project_id}")
            return None
        
        for key, value in kwargs.items():
            if hasattr(project, key):
                setattr(project, key, value)
        
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        logger.info(f"Updated project: {project.name} (ID: {project.id})")
        return project
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete project.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if project deleted
        """
        project = self.get_project(project_id)
        if project is None:
            return False
        
        project_dir = self.projects_dir / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
        
        self._cache.pop(project_id, None)
        logger.info(f"Deleted project: {project.name} (ID: {project_id})")
        return True
    
    def list_projects(
        self,
        status: Optional[str] = None,
        sort_by: str = "updated_at",
        reverse: bool = True
    ) -> List[Project]:
        """
        Get list of projects with filtering.
        
        Args:
            status: Status filter
            sort_by: Field to sort by
            reverse: Reverse sort
            
        Returns:
            List of projects
        """
        projects = list(self._cache.values())
        
        if status:
            projects = [p for p in projects if p.status == status]
        
        projects.sort(key=lambda p: getattr(p, sort_by, ""), reverse=reverse)
        return projects
    
    def list_projects_dicts(self) -> List[dict]:
        """
        Get list of projects as dictionaries.
        
        Returns:
            List of project dictionaries
        """
        return [p.to_dict() for p in self.list_projects()]
    
    def create_project_safe(self, name: str, description: str = "") -> dict:
        """
        Create a new project with validation and error handling.
        
        Args:
            name: Project name
            description: Project description
            
        Returns:
            Dictionary with project data on success or error information
        """
        if not name or not name.strip():
            return {"error": "Field 'name' is required"}
        
        try:
            project = self.create_project(name=name.strip(), description=description)
            return project.to_dict()
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return {"error": str(e)}
    
    def search_projects(self, query: str) -> List[Project]:
        """
        Search projects by name and description.
        
        Args:
            query: Search query
            
        Returns:
            List of found projects
        """
        query_lower = query.lower()
        return [
            p for p in self._cache.values()
            if query_lower in p.name.lower() or query_lower in p.description.lower()
        ]
    
    # === File management ===
    
    def add_file_to_project(
        self,
        project_id: str,
        filename: str,
        file_content: Optional[bytes] = None,
        file_path: Optional[str] = None,
        file_type: str = "hyperspectral"
    ) -> Optional[ProjectFile]:
        """
        Add file to project.
        
        Args:
            project_id: Project ID
            filename: File name
            file_content: File content (bytes) - for small files
            file_path: Path to temporary file - for large files
            file_type: File type
            
        Returns:
            ProjectFile object or None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        if file_content is None and file_path is None:
            raise ValueError("Either file_content or file_path must be provided")
        
        # Calculate file size and checksum
        if file_path:
            # For large files - read from temporary file
            file_size = os.path.getsize(file_path)
            checksum = self._calculate_file_checksum(file_path)
        else:
            # For small files - use in-memory content
            file_size = len(file_content)
            checksum = hashlib.md5(file_content).hexdigest()
        
        project_file = ProjectFile(
            filename=filename,
            original_name=filename,
            file_type=file_type,
            file_size=file_size,
            checksum=checksum,
        )
        
        # Save file to disk
        files_dir = self.projects_dir / project_id / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        final_file_path = files_dir / f"{project_file.id}_{filename}"
        
        if file_path:
            # Move temporary file to final location
            shutil.move(file_path, final_file_path)
        else:
            # Write in-memory content
            with open(final_file_path, "wb") as f:
                f.write(file_content)
        
        project_file.file_path = str(final_file_path)
        
        # Update project
        project.files.append(project_file.to_dict())
        
        # Update status if there were files
        if project.status == ProjectStatus.NEW.value and len(project.files) > 0:
            project.status = ProjectStatus.READY.value
        
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        
        logger.info(f"Added file {filename} to project {project.name}")
        return project_file
    
    def add_file_by_server_path(
        self,
        project_id: str,
        source_path: str,
        file_type: str = "hyperspectral",
        copy: bool = True,
    ) -> Optional[ProjectFile]:
        """
        Add a file to the project by copying/moving it from a server-side path.

        This method avoids reading the file into memory entirely. It uses
        shutil.copy2 (or shutil.move) to transfer the file at the filesystem
        level, which consumes constant memory regardless of file size.

        This is the recommended way to add large files (> 500 MB) to a project
        instead of uploading through the browser (which encodes the file as
        base64 and loads it entirely into memory).

        Args:
            project_id: Project ID.
            source_path: Absolute or relative path to the source file on the server.
            file_type: File type label (e.g. 'hyperspectral', 'geotiff').
            copy: If True, copy the file; if False, move it.

        Returns:
            ProjectFile object, or None if the project was not found.

        Raises:
            FileNotFoundError: If source_path does not exist.
            ValueError: If source_path is not a file.
        """
        project = self.get_project(project_id)
        if project is None:
            return None

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if not source.is_file():
            raise ValueError(f"Source path is not a file: {source_path}")

        filename = source.name
        file_size = source.stat().st_size
        checksum = self._calculate_file_checksum(str(source))

        project_file = ProjectFile(
            filename=filename,
            original_name=filename,
            file_type=file_type,
            file_size=file_size,
            checksum=checksum,
        )

        # Prepare destination directory
        files_dir = self.projects_dir / project_id / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        final_file_path = files_dir / f"{project_file.id}_{filename}"

        # Copy or move at filesystem level — constant memory usage
        if copy:
            shutil.copy2(str(source), str(final_file_path))
        else:
            shutil.move(str(source), str(final_file_path))

        project_file.file_path = str(final_file_path)

        # Update project
        project.files.append(project_file.to_dict())

        if project.status == ProjectStatus.NEW.value and len(project.files) > 0:
            project.status = ProjectStatus.READY.value

        project.updated_at = datetime.now().isoformat()
        self._save_project(project)

        logger.info(
            f"Added server file {filename} ({file_size:,} bytes) "
            f"to project {project.name} via filesystem {'copy' if copy else 'move'}"
        )
        return project_file

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file using streaming."""
        app_config = config['default']
        chunk_size = app_config.STREAMING_CHUNK_SIZE
        
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def remove_file_from_project(self, project_id: str, file_id: str) -> bool:
        """
        Remove file from project.
        
        Args:
            project_id: Project ID
            file_id: File ID
            
        Returns:
            True if file was removed
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
        
        # Remove file from disk
        file_path = file_data.get("file_path", "")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove from list
        project.files = [f for f in project.files if f.get("id") != file_id]
        
        # Update status
        if len(project.files) == 0 and project.status == ProjectStatus.READY.value:
            project.status = ProjectStatus.NEW.value
        
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        
        logger.info(f"Removed file {file_id} from project {project.name}")
        return True
    
    # === Processing management ===
    
    def update_processing_config(
        self, project_id: str, config: dict
    ) -> Optional[Project]:
        """
        Update project processing configuration.
        
        Args:
            project_id: Project ID
            config: New configuration
            
        Returns:
            Updated project or None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        # Merge with existing configuration
        current_config = project.processing_config
        current_config.update(config)
        project.processing_config = current_config
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        return project
    
    def start_processing(self, project_id: str) -> Optional[ProcessingHistory]:
        """
        Start project processing.
        
        Args:
            project_id: Project ID
            
        Returns:
            Processing history record or None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        if project.status == ProjectStatus.RUN.value:
            logger.warning(f"Project {project.name} is already being processed")
            return None
        
        # Create history record
        history = ProcessingHistory(
            config=project.processing_config,
        )
        
        # Create directory for results
        results_dir = self.projects_dir / project_id / "results" / history.run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Update project
        project.status = ProjectStatus.RUN.value
        project.progress = 0.0
        project.current_stage = PipelineStage.PREPROCESSING.value
        project.processing_history.append(history.to_dict())
        project.updated_at = datetime.now().isoformat()
        self._save_project(project)
        
        logger.info(f"Started processing project {project.name} (run: {history.run_id})")
        return history
    
    def update_processing_progress(
        self, 
        project_id: str, 
        stage: str, 
        progress: float,
        stage_result: Optional[dict] = None
    ) -> Optional[Project]:
        """
        Update processing progress.
        
        Args:
            project_id: Project ID
            stage: Current stage
            progress: Progress (0-100)
            stage_result: Stage result
            
        Returns:
            Updated project or None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        project.current_stage = stage
        project.progress = progress
        
        # Update stage result in the last history record
        if stage_result and project.processing_history:
            last_history = project.processing_history[-1]
            last_history.setdefault("results", [])
            
            # Update or add stage result
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
        Complete project processing.
        
        Args:
            project_id: Project ID
            success: Whether processing completed successfully
            error_message: Error message
            
        Returns:
            Updated project or None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        now = datetime.now().isoformat()
        
        if success:
            project.status = ProjectStatus.DONE.value
            project.progress = 100.0
        else:
            project.status = ProjectStatus.ERROR.value
        
        project.current_stage = None
        
        # Update the last history record
        if project.processing_history:
            last_history = project.processing_history[-1]
            last_history["end_time"] = now
            last_history["status"] = "completed" if success else "error"
            last_history["error_message"] = error_message
            
            # Calculate duration
            try:
                start = datetime.fromisoformat(last_history["start_time"])
                end = datetime.fromisoformat(now)
                last_history["total_duration_seconds"] = (end - start).total_seconds()
            except (ValueError, KeyError):
                pass
        
        project.updated_at = now
        self._save_project(project)
        
        status_text = "successfully" if success else f"with error: {error_message}"
        logger.info(f"Processing of project {project.name} completed {status_text}")
        return project
    
    def cancel_processing(self, project_id: str) -> Optional[Project]:
        """
        Cancel project processing.
        
        Args:
            project_id: Project ID
            
        Returns:
            Updated project or None
        """
        project = self.get_project(project_id)
        if project is None:
            return None
        
        if project.status != ProjectStatus.RUN.value:
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
        
        logger.info(f"Processing of project {project.name} cancelled")
        return project
    
    # === Statistics ===
    
    def get_statistics(self) -> dict:
        """
        Get overall project statistics.
        
        Returns:
            Dictionary with statistics
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
        Get recent projects.
        
        Args:
            limit: Maximum number
            
        Returns:
            List of recent projects
        """
        projects = self.list_projects(sort_by="updated_at", reverse=True)
        return projects[:limit]