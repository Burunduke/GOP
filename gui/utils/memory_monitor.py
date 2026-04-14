"""
Memory monitoring utilities for detecting and preventing out-of-memory errors.
"""

import psutil
import logging
import gc
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage and prevent out-of-memory errors."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold: Memory usage threshold for warnings (0.0-1.0)
            critical_threshold: Memory usage threshold for critical actions (0.0-1.0)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
            "system_memory_used_mb": system_memory.used / 1024 / 1024,
            "system_memory_total_mb": system_memory.total / 1024 / 1024,
            "system_memory_percent": system_memory.percent / 100,
            "available_memory_mb": system_memory.available / 1024 / 1024
        }
    
    def check_memory_status(self) -> Dict[str, str]:
        """Check memory status and return status information."""
        memory_info = self.get_memory_usage()
        system_percent = memory_info["system_memory_percent"]
        
        if system_percent >= self.critical_threshold:
            status = "CRITICAL"
            action = "Immediate action required - system memory critically low"
        elif system_percent >= self.warning_threshold:
            status = "WARNING"
            action = "Memory usage high - consider cleanup"
        else:
            status = "OK"
            action = "Memory usage normal"
        
        return {
            "status": status,
            "action": action,
            "system_memory_percent": f"{system_percent:.1%}",
            "process_memory_mb": f"{memory_info['process_memory_mb']:.1f} MB",
            "available_memory_mb": f"{memory_info['available_memory_mb']:.1f} MB"
        }
    
    def force_garbage_collection(self) -> None:
        """Force garbage collection to free up memory."""
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
    
    def is_memory_safe_for_large_operation(self, estimated_size_mb: float) -> bool:
        """
        Check if there's enough memory for a large operation.
        
        Args:
            estimated_size_mb: Estimated memory requirement in MB
            
        Returns:
            True if operation should be safe
        """
        memory_info = self.get_memory_usage()
        available_mb = memory_info["available_memory_mb"]
        
        # Require at least 2x the estimated size as available memory
        required_mb = estimated_size_mb * 2
        
        if available_mb < required_mb:
            logger.warning(
                f"Insufficient memory for operation: {estimated_size_mb:.1f} MB required, "
                f"but only {available_mb:.1f} MB available"
            )
            return False
        
        return True
    
    def log_memory_usage(self, operation_name: str = "") -> None:
        """Log current memory usage for debugging."""
        memory_info = self.get_memory_usage()
        status_info = self.check_memory_status()
        
        logger.info(
            f"Memory usage {operation_name}: "
            f"Process: {memory_info['process_memory_mb']:.1f} MB, "
            f"System: {status_info['system_memory_percent']}, "
            f"Status: {status_info['status']}"
        )


def create_memory_safe_operation(max_size_mb: float = 500):
    """
    Decorator to ensure memory safety for operations.
    
    Args:
        max_size_mb: Maximum allowed memory usage for the operation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor()
            
            # Check if operation is safe
            if not monitor.is_memory_safe_for_large_operation(max_size_mb):
                raise MemoryError(
                    f"Operation requires {max_size_mb} MB but insufficient memory available"
                )
            
            # Log memory usage before operation
            monitor.log_memory_usage(f"before {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Force cleanup after operation
                monitor.force_garbage_collection()
                monitor.log_memory_usage(f"after {func.__name__}")
        
        return wrapper
    return decorator