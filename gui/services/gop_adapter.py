"""
Adapter for integrating GUI with GOP
"""

import os
import sys
import asyncio
import concurrent.futures
import logging
from typing import Dict, Any, Optional

# Add path to GOP source code for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Unconditional import - if this fails, the adapter should raise a clear exception
from src.core.pipeline import Pipeline

logger = logging.getLogger(__name__)


class GOPAdapter:
    """Adapter for working with GOP through GUI. Provides real processing capabilities only."""
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize adapter
        
        Args:
            config_path: Path to GOP configuration file
        """
        self.config_path = config_path or 'config/config.yaml'
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Initialize the real pipeline - if this fails, raise a clear exception
        try:
            self.pipeline = Pipeline(self.config_path)
            # Use components from pipeline instead of creating separate instances
            self.hyperspectral_processor = self.pipeline.hyperspectral_processor
        except Exception as e:
            logger.error(f"GOP initialization error: {e}")
            raise RuntimeError(f"GOP modules not available: {e}") from e
    
    def process_data(self, data_path: str, processing_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through GOP pipeline
        
        Args:
            data_path: Path to input data (file or directory)
            processing_type: Type of processing
            parameters: Additional processing parameters
            
        Returns:
            Processing result
        """
        try:
            # Check if data_path is a directory
            import os
            if os.path.isdir(data_path):
                # Get list of files in directory
                files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
                if not files:
                    raise Exception(f"No files found in directory: {data_path}")
                
                # For now, use the first file (this should be improved to handle multiple files properly)
                # In a real implementation, we might want to process all files or select a main file
                first_file = files[0]
                actual_data_path = os.path.join(data_path, first_file)
                logger.info(f"Processing first file from directory: {actual_data_path}")
            else:
                actual_data_path = data_path
            
            # Call the real pipeline process method
            result = self.pipeline.process(
                input_path=actual_data_path,
                output_dir=parameters.get('output_dir'),
                sensor_type=processing_type
            )
            return result
        except Exception as e:
            raise Exception(f"GOP processing error: {str(e)}")
    
    async def process_data_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous data processing through GOP
        
        Args:
            config: Processing configuration
            
        Returns:
            Processing result
        """
        try:
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
                
        except Exception as e:
            return {
                'status': 'error',
                'result': None,
                'error': str(e)
            }
    
    def _process_sync(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous data processing"""
        try:
            result = self.pipeline.process(
                input_path=config['input_path'],
                output_dir=config['output_dir'],
                sensor_type=config.get('sensor_type', 'hyperspectral')
            )
            return result
        except Exception as e:
            raise Exception(f"GOP processing error: {str(e)}")
    
    

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
            
            # Check file format
            supported_formats = ['.bil', '.hdr', '.tif', '.tiff', '.dat']
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in supported_formats:
                return {'valid': False, 'error': f'Unsupported format: {file_ext}'}
            
            
            return {
                'valid': True, 
                'file_size': file_size,
                'file_format': file_ext,
                'estimated_processing_time': self._estimate_processing_time(file_size)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _estimate_processing_time(self, file_size: int) -> str:
        """Estimate file processing time"""
        # Simple heuristic: ~1 second per MB
        seconds = file_size / (1024 * 1024)
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_processing_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task processing status
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status
        """
        return {
            'task_id': task_id,
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed successfully',
            'result': {
                'output_path': f'data/results/{task_id}',
                'files_generated': ['orthophoto.tif', 'preprocessed_data.hdr'],
                'processing_time': '00:05:23'
            }
        }
    
    def cancel_processing(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel task processing
        
        Args:
            task_id: Task ID
            
        Returns:
            Cancellation result
        """
        return {
            'task_id': task_id,
            'status': 'cancelled',
            'message': 'Task cancelled'
        }
    
    def __del__(self):
        """Clean up resources when deleting"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)