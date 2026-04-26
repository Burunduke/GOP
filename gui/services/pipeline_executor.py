"""Pipeline executor - connects projects with GOP processing."""

import logging
import threading
from datetime import datetime
from typing import Optional, Callable, Dict
from concurrent.futures import ThreadPoolExecutor, Future

from gui.models.project import PipelineStage, ProcessingResult
from gui.services.project_manager import ProjectManager

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Pipeline executor for project processing. Uses real GOP processing only."""
    
    # Mapping of stages to their weights for progress calculation
    STAGE_WEIGHTS: Dict[str, int] = {
        PipelineStage.PREPROCESSING.value: 50,
        PipelineStage.ORTHOPHOTO.value: 50,
    }
    
    STAGE_NAMES: Dict[str, str] = {
        PipelineStage.PREPROCESSING.value: "Preprocessing",
        PipelineStage.ORTHOPHOTO.value: "Orthophoto creation",
    }
    
    def __init__(self, project_manager: ProjectManager, gop_adapter=None, max_workers: int = 2) -> None:
        """
        Args:
            project_manager: Project manager
            gop_adapter: GOP adapter for real processing
            max_workers: Maximum number of concurrent workers (default: 2)
        """
        self.project_manager = project_manager
        self.gop_adapter = gop_adapter
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pipeline")
        self._running_tasks: Dict[str, Future] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._progress_callbacks: Dict[str, Callable] = {}
    
    def start_project_safe(self, project_id: str) -> dict:
        """
        Start project processing with validation and error handling.
        
        Args:
            project_id: Project ID
            
        Returns:
            Dictionary with status information
        """
        try:
            project = self.project_manager.get_project(project_id)
            if project is None:
                return {"error": "Project not found"}
            
            # Guard: already running
            if self.is_running(project_id):
                return {"error": "Project is already being processed"}
            
            started = self.execute_project(project_id)
            if not started:
                return {"error": "Could not start processing — check project status or files"}
            
            return {"status": "started", "project_id": project_id}
        except Exception as e:
            logger.error(f"Error starting processing for project {project_id}: {e}")
            return {"error": str(e)}
    
    def get_status_dict(self, project_id: str) -> dict:
        """
        Get processing status as a dictionary.
        
        Args:
            project_id: Project ID
            
        Returns:
            Dictionary with status information
        """
        try:
            project = self.project_manager.get_project(project_id)
            if project is None:
                return {"error": "Project not found"}
            
            return {
                'project_id': project_id,
                'status': project.status,
                'progress': project.progress,
                'stage': project.current_stage,
                'is_running': self.is_running(project_id),
            }
        except Exception as e:
            logger.error(f"Error getting status for project {project_id}: {e}")
            return {"error": str(e)}
    
    def execute_project(
        self,
        project_id: str,
        on_progress: Optional[Callable] = None
    ) -> bool:
        """
        Start project processing in background thread.
        
        Args:
            project_id: Project ID
            on_progress: Callback for progress updates
            
        Returns:
            True if processing started
        """
        project = self.project_manager.get_project(project_id)
        if project is None:
            logger.error(f"Project not found: {project_id}")
            return False
        
        if project_id in self._running_tasks:
            logger.warning(f"Project {project_id} is already being processed")
            return False
        
        # Start processing in ProjectManager
        history = self.project_manager.start_processing(project_id)
        if history is None:
            return False
        
        # Set up cancellation flag
        cancel_event = threading.Event()
        self._cancel_flags[project_id] = cancel_event
        
        if on_progress:
            self._progress_callbacks[project_id] = on_progress
        
        # Run in background thread
        future = self._pool.submit(self._run_pipeline, project_id, history.run_id, cancel_event)
        self._running_tasks[project_id] = future
        
        logger.info(f"Started processing project {project_id}")
        return True
    
    def cancel_project(self, project_id: str) -> bool:
        """Cancel project processing."""
        if project_id not in self._cancel_flags:
            return False
        
        # Cancel the future if it exists
        if project_id in self._running_tasks:
            future = self._running_tasks[project_id]
            future.cancel()
        
        self._cancel_flags[project_id].set()
        self.project_manager.cancel_processing(project_id)
        
        # Clean up
        self._cleanup_task(project_id)
        logger.info(f"Processing project {project_id} cancelled")
        return True
    
    def is_running(self, project_id: str) -> bool:
        """Check if project processing is running."""
        if project_id not in self._running_tasks:
            return False
        future = self._running_tasks[project_id]
        return not future.done()
    
    def get_running_projects(self) -> list[str]:
        """Get list of project IDs being processed."""
        return [pid for pid, future in self._running_tasks.items() if not future.done()]
    
    def _run_pipeline(self, project_id: str, run_id: str, cancel_event: threading.Event) -> None:
        """
        Main pipeline execution method (runs in background thread).
        """
        try:
            project = self.project_manager.get_project(project_id)
            if project is None:
                return
            
            # Get configured stages
            config = project.processing_config
            stages = config.get("stages", [s.value for s in PipelineStage])
            
            # Calculate progress increments
            total_weight = sum(self.STAGE_WEIGHTS.get(s, 10) for s in stages)
            cumulative_progress = 0.0
            
            for stage in stages:
                if cancel_event.is_set():
                    logger.info(f"Processing project {project_id} cancelled at stage {stage}")
                    return
                
                stage_weight = self.STAGE_WEIGHTS.get(stage, 10)
                stage_name = self.STAGE_NAMES.get(stage, stage)
                
                # Update progress - stage starting
                stage_result = ProcessingResult(
                    stage=stage,
                    status="running",
                    start_time=datetime.now().isoformat(),
                )
                
                self.project_manager.update_processing_progress(
                    project_id, stage, cumulative_progress,
                    stage_result=stage_result.to_dict()
                )
                
                # Execute stage
                try:
                    # Check if project has files before processing
                    project = self.project_manager.get_project(project_id)
                    if not project or not project.files:
                        raise ValueError(f"Project {project_id} has no files to process. Please add files before starting processing.")
                    
                    result_data = self._execute_stage(project_id, stage, config, cancel_event)
                    
                    if cancel_event.is_set():
                        return
                    
                    # Stage completed
                    stage_result.status = "completed"
                    stage_result.end_time = datetime.now().isoformat()
                    stage_result.metrics = result_data.get("metrics", {})
                    stage_result.output_files = result_data.get("output_files", [])
                    
                    # Calculate duration
                    try:
                        start = datetime.fromisoformat(stage_result.start_time)
                        end = datetime.fromisoformat(stage_result.end_time)
                        stage_result.duration_seconds = (end - start).total_seconds()
                    except ValueError:
                        pass
                    
                except Exception as e:
                    logger.error(f"Error at stage {stage} project {project_id}: {e}")
                    stage_result.status = "error"
                    stage_result.end_time = datetime.now().isoformat()
                    stage_result.error_message = str(e)
                    
                    self.project_manager.update_processing_progress(
                        project_id, stage, cumulative_progress,
                        stage_result=stage_result.to_dict()
                    )
                    
                    # Complete with error
                    self.project_manager.complete_processing(
                        project_id, success=False, error_message=str(e)
                    )
                    return
                
                # Update cumulative progress
                cumulative_progress += (stage_weight / total_weight) * 100
                
                self.project_manager.update_processing_progress(
                    project_id, stage, min(cumulative_progress, 99.0),
                    stage_result=stage_result.to_dict()
                )
                
                # Notify progress callback
                callback = self._progress_callbacks.get(project_id)
                if callback:
                    try:
                        callback(project_id, stage, cumulative_progress)
                    except Exception:
                        pass
            
            # All stages completed
            self.project_manager.complete_processing(project_id, success=True)
            logger.info(f"Processing of project {project_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Critical error processing project {project_id}: {e}")
            self.project_manager.complete_processing(
                project_id, success=False, error_message=str(e)
            )
        finally:
            self._cleanup_task(project_id)
    
    def _execute_stage(
        self, 
        project_id: str, 
        stage: str, 
        config: dict,
        cancel_event: threading.Event
    ) -> dict:
        """
        Execute one pipeline stage.
        
        Uses real processing through GOPAdapter.
        """
        return self._execute_real_stage(project_id, stage, config)
    
    def _execute_real_stage(self, project_id: str, stage: str, config: dict) -> dict:
        """Execute real stage through GOPAdapter."""
        # This will be fully implemented when GOP modules are available
        # For now, delegate to GOPAdapter
        if self.gop_adapter:
            # Get project to check files
            project = self.project_manager.get_project(project_id)
            if project is None:
                raise ValueError(f"Project not found: {project_id}")
            
            # Check if project has files
            if not project.files:
                raise ValueError(f"Project {project_id} has no files to process")
            
            # Get the files directory using the project manager's helper
            project = self.project_manager.get_project(project_id)
            if project is None:
                raise ValueError(f"Project not found: {project_id}")
            files_dir = self.project_manager.get_project_dir(project) / "files"
            
            # Get the current run folder from project manager
            project = self.project_manager.get_project(project_id)
            run_folder = None
            if project and project.processing_history:
                # Get the last run (current run)
                last_history = project.processing_history[-1]
                run_folder_name = last_history.get("run_folder_name")
                run_id = last_history.get("run_id")
                if run_folder_name:
                    # For new runs, we have the folder name directly
                    project_dir = self.project_manager.get_project_dir(project)
                    run_folder = str(project_dir / "results" / run_folder_name)
                elif run_id:
                    # For legacy runs, use the run_id as folder name
                    project_dir = self.project_manager.get_project_dir(project)
                    run_folder = str(project_dir / "results" / run_id)
            
            # For hyperspectral processing, we need to pass the directory containing files
            # The processor should handle multiple files in the directory
            # Note: processing_type is the stage name, not the sensor type - GOPAdapter derives that from files
            result = self.gop_adapter.process_data(
                data_path=str(files_dir),
                processing_type=stage,  # Stage name (not sensor type - GOPAdapter derives that from files)
                parameters={
                    **config.get(stage, {}),
                    "output_dir": run_folder,
                    "stitching_method": config.get("orthophoto", {}).get("stitching_method", "gdal")
                }
            )
            return result if isinstance(result, dict) else {"metrics": {}, "output_files": []}
        else:
            raise RuntimeError("GOPAdapter is not available for real processing")
    
    
    
    def _cleanup_task(self, project_id: str):
        """Clean up task resources."""
        self._running_tasks.pop(project_id, None)
        self._cancel_flags.pop(project_id, None)
        self._progress_callbacks.pop(project_id, None)
    
    def shutdown(self, wait: bool = False) -> None:
        """
        Shutdown the thread pool executor.
        
        Args:
            wait: Whether to wait for running tasks to complete
        """
        self._pool.shutdown(wait=wait)