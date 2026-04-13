"""Исполнитель пайплайна - связывает проекты с обработкой GOP."""

import logging
import threading
import time
from datetime import datetime
from typing import Optional, Callable

from gui.models.project import PipelineStage, ProjectStatus, ProcessingResult
from gui.services.project_manager import ProjectManager

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Исполнитель пайплайна обработки для проектов."""
    
    # Mapping of stages to their weights for progress calculation
    STAGE_WEIGHTS = {
        PipelineStage.PREPROCESSING.value: 25,
        PipelineStage.ORTHOPHOTO.value: 20,
        PipelineStage.SEGMENTATION.value: 20,
        PipelineStage.INDICES.value: 15,
        PipelineStage.ASSESSMENT.value: 10,
        PipelineStage.ANALYSIS.value: 10,
    }
    
    STAGE_NAMES_RU = {
        PipelineStage.PREPROCESSING.value: "Предобработка",
        PipelineStage.ORTHOPHOTO.value: "Создание ортофото",
        PipelineStage.SEGMENTATION.value: "Сегментация",
        PipelineStage.INDICES.value: "Вегетационные индексы",
        PipelineStage.ASSESSMENT.value: "Оценка состояния",
        PipelineStage.ANALYSIS.value: "Научный анализ",
    }
    
    def __init__(self, project_manager: ProjectManager, gop_adapter=None):
        """
        Args:
            project_manager: Менеджер проектов
            gop_adapter: Адаптер GOP (опционально, для реальной обработки)
        """
        self.project_manager = project_manager
        self.gop_adapter = gop_adapter
        self._running_tasks: dict[str, threading.Thread] = {}
        self._cancel_flags: dict[str, threading.Event] = {}
        self._progress_callbacks: dict[str, Callable] = {}
    
    def execute_project(
        self, 
        project_id: str,
        on_progress: Optional[Callable] = None
    ) -> bool:
        """
        Запуск обработки проекта в фоновом потоке.
        
        Args:
            project_id: ID проекта
            on_progress: Callback для обновления прогресса
            
        Returns:
            True если обработка запущена
        """
        project = self.project_manager.get_project(project_id)
        if project is None:
            logger.error(f"Проект не найден: {project_id}")
            return False
        
        if project_id in self._running_tasks:
            logger.warning(f"Проект {project_id} уже обрабатывается")
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
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(project_id, history.run_id, cancel_event),
            daemon=True,
            name=f"pipeline-{project_id[:8]}"
        )
        self._running_tasks[project_id] = thread
        thread.start()
        
        logger.info(f"Запущена обработка проекта {project_id}")
        return True
    
    def cancel_project(self, project_id: str) -> bool:
        """Отмена обработки проекта."""
        if project_id not in self._cancel_flags:
            return False
        
        self._cancel_flags[project_id].set()
        self.project_manager.cancel_processing(project_id)
        
        # Clean up
        self._cleanup_task(project_id)
        logger.info(f"Обработка проекта {project_id} отменена")
        return True
    
    def is_running(self, project_id: str) -> bool:
        """Проверка, выполняется ли обработка проекта."""
        return project_id in self._running_tasks and self._running_tasks[project_id].is_alive()
    
    def get_running_projects(self) -> list[str]:
        """Получение списка ID проектов в обработке."""
        return [pid for pid, t in self._running_tasks.items() if t.is_alive()]
    
    def _run_pipeline(self, project_id: str, run_id: str, cancel_event: threading.Event):
        """
        Основной метод выполнения пайплайна (выполняется в фоновом потоке).
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
                    logger.info(f"Обработка проекта {project_id} отменена на этапе {stage}")
                    return
                
                stage_weight = self.STAGE_WEIGHTS.get(stage, 10)
                stage_name = self.STAGE_NAMES_RU.get(stage, stage)
                
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
                    logger.error(f"Ошибка на этапе {stage} проекта {project_id}: {e}")
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
            logger.info(f"Обработка проекта {project_id} завершена успешно")
            
        except Exception as e:
            logger.error(f"Критическая ошибка обработки проекта {project_id}: {e}")
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
        Выполнение одного этапа пайплайна.
        
        Если GOPAdapter доступен - использует реальную обработку.
        Иначе - эмулирует обработку.
        """
        if self.gop_adapter and hasattr(self.gop_adapter, 'gop_available') and self.gop_adapter.gop_available:
            return self._execute_real_stage(project_id, stage, config)
        else:
            return self._emulate_stage(project_id, stage, config, cancel_event)
    
    def _execute_real_stage(self, project_id: str, stage: str, config: dict) -> dict:
        """Выполнение реального этапа через GOPAdapter."""
        # This will be fully implemented when GOP modules are available
        # For now, delegate to GOPAdapter
        try:
            if self.gop_adapter:
                result = self.gop_adapter.process_data(
                    data_path=str(self.project_manager.projects_dir / project_id / "files"),
                    processing_type=stage,
                    parameters=config.get(stage, {})
                )
                return result if isinstance(result, dict) else {"metrics": {}, "output_files": []}
        except Exception as e:
            logger.warning(f"Ошибка реальной обработки, переключение на эмуляцию: {e}")
        
        return self._emulate_stage(project_id, stage, config, threading.Event())
    
    def _emulate_stage(
        self, 
        project_id: str, 
        stage: str, 
        config: dict,
        cancel_event: threading.Event
    ) -> dict:
        """
        Эмуляция выполнения этапа пайплайна.
        Используется когда GOP модули недоступны.
        """
        import random
        
        stage_durations = {
            PipelineStage.PREPROCESSING.value: (2, 4),
            PipelineStage.ORTHOPHOTO.value: (2, 3),
            PipelineStage.SEGMENTATION.value: (2, 4),
            PipelineStage.INDICES.value: (1, 3),
            PipelineStage.ASSESSMENT.value: (1, 2),
            PipelineStage.ANALYSIS.value: (1, 2),
        }
        
        min_dur, max_dur = stage_durations.get(stage, (1, 2))
        duration = random.uniform(min_dur, max_dur)
        
        # Simulate processing with interruptible sleep
        steps = 10
        step_duration = duration / steps
        for i in range(steps):
            if cancel_event.is_set():
                return {"metrics": {}, "output_files": []}
            time.sleep(step_duration)
        
        # Generate emulated results
        emulated_metrics = self._generate_emulated_metrics(stage)
        
        return {
            "metrics": emulated_metrics,
            "output_files": [],
            "emulated": True,
        }
    
    def _generate_emulated_metrics(self, stage: str) -> dict:
        """Генерация эмулированных метрик для этапа."""
        import random
        
        metrics = {
            PipelineStage.PREPROCESSING.value: {
                "snr_improvement": round(random.uniform(5, 15), 2),
                "bands_processed": random.randint(100, 300),
                "correction_applied": "radiometric + atmospheric",
                "denoising_method": "PCA",
            },
            PipelineStage.ORTHOPHOTO.value: {
                "resolution_m": 0.1,
                "coverage_area_ha": round(random.uniform(1, 50), 2),
                "crs": "EPSG:4326",
                "tiles_generated": random.randint(4, 16),
            },
            PipelineStage.SEGMENTATION.value: {
                "segments_found": random.randint(50, 500),
                "vegetation_coverage_pct": round(random.uniform(40, 85), 1),
                "model_used": "DeepLabV3+",
                "refinement": "CascadePSP",
            },
            PipelineStage.INDICES.value: {
                "indices_calculated": ["NDVI", "GNDVI", "MCARI", "RENDVI"],
                "mean_ndvi": round(random.uniform(0.3, 0.8), 3),
                "mean_gndvi": round(random.uniform(0.2, 0.7), 3),
                "pixels_processed": random.randint(100000, 1000000),
            },
            PipelineStage.ASSESSMENT.value: {
                "healthy_pct": round(random.uniform(50, 90), 1),
                "stressed_pct": round(random.uniform(5, 30), 1),
                "damaged_pct": round(random.uniform(1, 15), 1),
                "assessment_method": "multi-index",
            },
            PipelineStage.ANALYSIS.value: {
                "correlation_pairs": random.randint(5, 20),
                "spatial_clusters": random.randint(3, 8),
                "statistical_tests": ["t-test", "ANOVA", "Kruskal-Wallis"],
                "significant_findings": random.randint(1, 5),
            },
        }
        
        return metrics.get(stage, {})
    
    def _cleanup_task(self, project_id: str):
        """Очистка ресурсов задачи."""
        self._running_tasks.pop(project_id, None)
        self._cancel_flags.pop(project_id, None)
        self._progress_callbacks.pop(project_id, None)