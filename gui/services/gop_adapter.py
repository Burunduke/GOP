"""
Adapter for integrating GUI with GOP
"""

import os
import sys
import asyncio
import concurrent.futures
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add path to GOP source code for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from src.core.pipeline import Pipeline
    from src.indices.calculator import VegetationIndexCalculator
    from src.processing.hyperspectral.processor import HyperspectralProcessor
    from src.segmentation.segmenter import ImageSegmenter
    GOP_AVAILABLE = True
except ImportError:
    GOP_AVAILABLE = False
    logging.getLogger(__name__).warning("GOP modules not found. Using emulation mode.")

logger = logging.getLogger(__name__)


class GOPAdapter:
    """Adapter for working with GOP through GUI"""
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize adapter
        
        Args:
            config_path: Path to GOP configuration file
        """
        self.config_path = config_path or 'config/config.yaml'
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        if GOP_AVAILABLE:
            try:
                self.pipeline = Pipeline(self.config_path)
                # Use components from pipeline instead of creating separate instances
                self.indices_calculator = self.pipeline.index_calculator
                self.hyperspectral_processor = self.pipeline.hyperspectral_processor
                self.segmenter = self.pipeline.segmenter
                self.gop_mode = "full"
            except Exception as e:
                logger.error(f"GOP initialization error: {e}")
                self.gop_mode = "emulation"
        else:
            self.gop_mode = "emulation"
    
    async def process_data_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous data processing through GOP
        
        Args:
            config: Processing configuration
            
        Returns:
            Processing result
        """
        try:
            if self.gop_mode == "full":
                # Start processing in separate thread
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._process_sync,
                    config
                )
                
                return {
                    'status': 'completed',
                    'result': result,
                    'error': None
                }
            else:
                # Emulate processing
                await asyncio.sleep(2)  # Simulate processing time
                return self._emulate_processing_result(config)
                
        except Exception as e:
            return {
                'status': 'error',
                'result': None,
                'error': str(e)
            }
    
    def _process_sync(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous data processing"""
        if self.gop_mode != "full":
            return self._emulate_processing_result(config)['result']
        
        try:
            result = self.pipeline.process(
                input_path=config['input_path'],
                output_dir=config['output_dir'],
                sensor_type=config.get('sensor_type', 'hyperspectral'),
                selected_indices=config.get('selected_indices', ['NDVI']),
                use_refinement=config.get('use_refinement', True)
            )
            return result
        except Exception as e:
            raise Exception(f"GOP processing error: {str(e)}")
    
    def _emulate_processing_result(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Emulate processing result"""
        import uuid
        from datetime import datetime
        
        task_id = str(uuid.uuid4())
        
        result = {
            'task_id': task_id,
            'input_path': config.get('input_path', 'unknown'),
            'output_dir': config.get('output_dir', f'data/results/{task_id}'),
            'processing_time': '00:02:15',
            'indices_calculated': config.get('selected_indices', ['NDVI']),
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'files_generated': [
                f'{index}_map.tif' for index in config.get('selected_indices', ['NDVI'])
            ],
            'statistics': {
                'total_pixels': 1000000,
                'processed_pixels': 950000,
                'ndvi_mean': 0.65,
                'ndvi_std': 0.15,
                'vegetation_coverage': 0.78
            }
        }
        
        return {
            'status': 'completed',
            'result': result,
            'error': None
        }
    
    def get_available_indices(self, sensor_type: str = 'hyperspectral') -> List[Dict[str, Any]]:
        """
        Get available vegetation indices
        
        Args:
            sensor_type: Sensor type
            
        Returns:
            List of available indices
        """
        if self.gop_mode == "full":
            try:
                from src.indices.definitions import IndexDefinitions
                indices = IndexDefinitions.get_available_indices(sensor_type)
                return [{'id': idx, 'name': idx, 'description': f'Index {idx}'} for idx in indices]
            except Exception:
                pass
        
        # Return basic indices in emulation mode
        return [
            {
                'id': 'NDVI',
                'name': 'Normalized Difference Vegetation Index',
                'description': 'Normalized Difference Vegetation Index',
                'formula': '(NIR - Red) / (NIR + Red)'
            },
            {
                'id': 'EVI',
                'name': 'Enhanced Vegetation Index',
                'description': 'Enhanced Vegetation Index',
                'formula': '2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))'
            },
            {
                'id': 'SAVI',
                'name': 'Soil Adjusted Vegetation Index',
                'description': 'Soil Adjusted Vegetation Index',
                'formula': '((NIR - Red) / (NIR + Red + L)) * (1 + L)'
            }
        ]

    def validate_input_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate input file
        
        Args:
            file_path: Path to file
            
        Returns:
            Validation result
        """
        try:
            # Check file existence
            if not os.path.exists(file_path):
                return {'valid': False, 'error': 'File does not exist'}
            
            # Check file size
            file_size = os.path.getsize(file_path)
            max_size = 10 * 1024 * 1024 * 1024  # 10GB
            if file_size > max_size:
                return {'valid': False, 'error': f'File too large (maximum {max_size / (1024**3):.1f}GB)'}
            
            # Проверка формата файла
            supported_formats = ['.bil', '.hdr', '.tif', '.tiff', '.dat']
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in supported_formats:
                return {'valid': False, 'error': f'Неподдерживаемый формат: {file_ext}'}
            
            # Дополнительная проверка в режиме полной функциональности
            if self.gop_mode == "full":
                try:
                    # Здесь можно добавить проверку через GOP валидаторы
                    pass
                except Exception as e:
                    return {'valid': False, 'error': f'Ошибка валидации GOP: {str(e)}'}
            
            return {
                'valid': True, 
                'file_size': file_size,
                'file_format': file_ext,
                'estimated_processing_time': self._estimate_processing_time(file_size)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _estimate_processing_time(self, file_size: int) -> str:
        """Оценка времени обработки файла"""
        # Простая эвристика: ~1 секунда на МБ
        seconds = file_size / (1024 * 1024)
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_processing_status(self, task_id: str) -> Dict[str, Any]:
        """
        Получение статуса обработки задачи
        
        Args:
            task_id: ID задачи
            
        Returns:
            Статус задачи
        """
        # Временная реализация - в будущем будет интеграция с Celery
        return {
            'task_id': task_id,
            'status': 'completed',
            'progress': 100,
            'message': 'Обработка завершена успешно',
            'result': {
                'output_path': f'data/results/{task_id}',
                'indices_calculated': ['NDVI', 'EVI'],
                'processing_time': '00:05:23'
            }
        }
    
    def cancel_processing(self, task_id: str) -> Dict[str, Any]:
        """
        Отмена обработки задачи
        
        Args:
            task_id: ID задачи
            
        Returns:
            Результат отмены
        """
        return {
            'task_id': task_id,
            'status': 'cancelled',
            'message': 'Задача отменена'
        }
    
    def __del__(self):
        """Очистка ресурсов при удалении"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)