"""
File upload utilities for handling large files with streaming to avoid memory issues.
"""

import os
import tempfile
import hashlib
from pathlib import Path
import logging

from .memory_monitor import MemoryMonitor
from ..config import config

logger = logging.getLogger(__name__)


class FileUploadManager:
    """Manager for handling file uploads with memory-efficient streaming."""
    
    def __init__(self, temp_dir: str = "data/temp_uploads"):
        """
        Initialize file upload manager.
        
        Args:
            temp_dir: Directory for temporary file storage
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_content_to_temp_file(
        self,
        content: str,
        filename: str
    ) -> Tuple[str, int, str]:
        """
        Save uploaded base64 content to temporary file using streaming.
        
        Args:
            content: Base64 encoded file content
            filename: Original filename
            
        Returns:
            Tuple of (temp_file_path, file_size, checksum)
        """
        monitor = MemoryMonitor()
        
        # Get configuration
        app_config = config['default']
        max_memory_file_size = app_config.MAX_MEMORY_FILE_SIZE
        streaming_chunk_size = app_config.STREAMING_CHUNK_SIZE
        
        # Check file size limit
        estimated_size = len(content) * 0.75  # Base64 to binary estimate
        if estimated_size > max_memory_file_size:
            # For very large files, we'll use a different approach
            logger.warning(f"File {filename} is large ({estimated_size / (1024*1024):.1f} MB), using optimized processing")
        
        # Check memory safety before processing
        estimated_size_mb = estimated_size / (1024 * 1024)
        if not monitor.is_memory_safe_for_large_operation(estimated_size_mb):
            logger.warning(f"Insufficient memory to process file {filename} ({estimated_size_mb:.1f} MB) in one go, using chunked processing")
        
        monitor.log_memory_usage(f"before processing {filename}")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            dir=self.temp_dir,
            prefix=f"upload_",
            suffix=f"_{filename}",
            delete=False
        )
        temp_file_path = temp_file.name
        
        try:
            # Extract base64 content
            content_type, content_string = content.split(",")
            
            # Create hasher for checksum
            hasher = hashlib.md5()
            file_size = 0
            
            # Use true streaming base64 decoder with incremental processing
            import base64
            
            # For very large files, we need to implement true streaming
            # The key insight: we can't avoid loading the base64 string entirely in Python
            # but we can significantly reduce memory usage by:
            # 1. Processing in larger chunks for better efficiency
            # 2. Using file-based storage immediately
            # 3. Forcing garbage collection
            
            # Use adaptive chunk sizing based on file size
            if estimated_size <= 10 * 1024 * 1024:  # <= 10MB
                optimal_chunk_size = streaming_chunk_size * 4  # 256KB
            elif estimated_size <= 100 * 1024 * 1024:  # <= 100MB
                optimal_chunk_size = streaming_chunk_size * 8  # 512KB
            else:  # > 100MB
                optimal_chunk_size = min(streaming_chunk_size * 16, 1024 * 1024)  # Up to 1MB
            
            # Decode and process in chunks
            for i in range(0, len(content_string), optimal_chunk_size * 4):  # Base64 chunks are 4 chars
                chunk_base64 = content_string[i:i + optimal_chunk_size * 4]
                
                # Ensure proper padding for the chunk
                if len(chunk_base64) % 4 != 0:
                    chunk_base64 += '=' * (4 - len(chunk_base64) % 4)
                
                # Decode this chunk
                chunk_binary = base64.b64decode(chunk_base64)
                temp_file.write(chunk_binary)
                hasher.update(chunk_binary)
                file_size += len(chunk_binary)
                
                # Force garbage collection periodically for very large files
                if estimated_size > 100 * 1024 * 1024 and i % (optimal_chunk_size * 10) == 0:
                    import gc
                    gc.collect()
            
            checksum = hasher.hexdigest()
            
            logger.info(f"Successfully saved {filename} to temporary file ({file_size:,} bytes)")
            
            return temp_file_path, file_size, checksum
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            logger.error(f"Error processing file {filename}: {e}")
            raise e
        finally:
            temp_file.close()
            monitor.force_garbage_collection()
            monitor.log_memory_usage(f"after processing {filename}")
    
    def save_streaming_upload_to_temp_file(
        self,
        file_stream,
        filename: str
    ) -> Tuple[str, int, str]:
        """
        Save uploaded file stream to temporary file using true streaming.
        This avoids loading the entire file into memory.
        
        Args:
            file_stream: File-like object (stream)
            filename: Original filename
            
        Returns:
            Tuple of (temp_file_path, file_size, checksum)
        """
        monitor = MemoryMonitor()
        
        # Get configuration
        app_config = config['default']
        max_memory_file_size = app_config.MAX_MEMORY_FILE_SIZE
        streaming_chunk_size = app_config.STREAMING_CHUNK_SIZE
        
        monitor.log_memory_usage(f"before streaming upload {filename}")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            dir=self.temp_dir,
            prefix=f"stream_upload_",
            suffix=f"_{filename}",
            delete=False
        )
        temp_file_path = temp_file.name
        
        try:
            # Create hasher for checksum
            hasher = hashlib.md5()
            file_size = 0
            
            # Use larger chunks for better performance with large files
            optimal_chunk_size = min(streaming_chunk_size * 8, 512 * 1024)  # Up to 512KB chunks
            
            # Stream the file content directly to disk
            while True:
                chunk = file_stream.read(optimal_chunk_size)
                if not chunk:
                    break
                
                temp_file.write(chunk)
                hasher.update(chunk)
                file_size += len(chunk)
                
                # Force garbage collection periodically for very large files
                if file_size > 100 * 1024 * 1024 and file_size % (optimal_chunk_size * 50) == 0:
                    import gc
                    gc.collect()
            
            checksum = hasher.hexdigest()
            
            logger.info(f"Successfully streamed {filename} to temporary file ({file_size:,} bytes)")
            
            return temp_file_path, file_size, checksum
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            logger.error(f"Error streaming file {filename}: {e}")
            raise e
        finally:
            temp_file.close()
            monitor.force_garbage_collection()
            monitor.log_memory_usage(f"after streaming upload {filename}")
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary file.
        
        Args:
            file_path: Path to temporary file to remove
        """
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                # File might be in use, ignore
                pass
    
    def cleanup_old_temp_files(self, max_age_hours: int = 24) -> None:
        """
        Clean up temporary files older than specified hours.
        
        Args:
            max_age_hours: Maximum age of files to keep (hours)
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in self.temp_dir.glob("upload_*"):
            try:
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.unlink(file_path)
            except OSError:
                # File might be in use, skip
                continue