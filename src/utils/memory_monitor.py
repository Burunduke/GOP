"""
Resource monitoring utilities for tracking memory and CPU usage during processing.

This module provides utilities for monitoring system resources during long-running
operations. It includes functions for getting resource snapshots and context
managers for automatically tracking resource usage at the start and end of operations.

The main components are:
- get_resource_snapshot(): Get current resource usage
- MemoryMonitor: Class for monitoring memory usage (moved from gui/utils)
- ResourceMonitor: Context manager for wrapping operations with resource tracking
"""

import psutil
import logging
import gc
import time
import threading
from typing import Dict, Optional, Any

from .logger import setup_logger

# Type alias for resource snapshots
ResourceSnapshot = Dict[str, float]


def get_resource_snapshot() -> ResourceSnapshot:
    """
    Get a snapshot of current system resource usage.
    
    Returns:
        Dictionary with resource usage metrics:
        - rss_mib: Process RSS memory in MiB
        - vms_mib: Process virtual memory in MiB
        - cpu_percent: Process CPU usage percentage
        - available_mib: Available system memory in MiB
        - percent: System memory usage percentage
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    return {
        "rss_mib": memory_info.rss / 1024 / 1024,
        "vms_mib": memory_info.vms / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "available_mib": system_memory.available / 1024 / 1024,
        "percent": system_memory.percent
    }


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
        self.logger = setup_logger(__name__)
    
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
        self.logger.info(f"Garbage collection freed {collected} objects")
    
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
            self.logger.warning(
                f"Insufficient memory for operation: {estimated_size_mb:.1f} MB required, "
                f"but only {available_mb:.1f} MB available"
            )
            return False
        
        return True
    
    def log_memory_usage(self, operation_name: str = "") -> None:
        """Log current memory usage for debugging."""
        memory_info = self.get_memory_usage()
        status_info = self.check_memory_status()
        
        self.logger.info(
            f"Memory usage {operation_name}: "
            f"Process: {memory_info['process_memory_mb']:.1f} MB, "
            f"System: {status_info['system_memory_percent']}, "
            f"Status: {status_info['status']}"
        )


class ResourceMonitor:
    """
    Context manager that logs resource usage on enter/exit and optionally samples periodically.
    
    This context manager is designed to be junior-friendly with explicit enter/exit methods
    and simple threading for periodic sampling.
    
    Example usage:
        with ResourceMonitor("dark_current", interval_s=5.0):
            processed_data = processed_data - dark_value
    """
    
    def __init__(self, label: str, logger: Optional[logging.Logger] = None, 
                 interval_s: Optional[float] = None):
        """
        Initialize resource monitor.
        
        Args:
            label: Label for this monitoring session
            logger: Logger to use (if None, creates one with setup_logger)
            interval_s: Interval for periodic sampling (if None, no sampling)
        """
        self.label = label
        self.interval_s = interval_s
        self.logger = logger or setup_logger("resource_monitor")
        self.start_time = 0.0
        self.start_snapshot: Optional[ResourceSnapshot] = None
        self.sampler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
    
    def __enter__(self) -> "ResourceMonitor":
        """Enter the context manager, logging resource usage at start."""
        # Get initial resource snapshot
        self.start_snapshot = get_resource_snapshot()
        self.start_time = time.perf_counter()
        
        # Log start information
        self.logger.info(
            f"[res] {self.label} start "
            f"rss={self.start_snapshot['rss_mib']:.1f}MiB "
            f"cpu={self.start_snapshot['cpu_percent']:.1f}% "
            f"avail={self.start_snapshot['available_mib']:.1f}MiB"
        )
        
        # Start periodic sampling if requested
        if self.interval_s is not None:
            self.stop_event.clear()
            self.sampler_thread = threading.Thread(
                target=self._sample_resources, 
                daemon=True,
                name=f"ResourceSampler-{self.label}"
            )
            self.sampler_thread.start()
        
        # Prime CPU percentage calculation
        psutil.Process().cpu_percent()
        
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager, stopping sampling and logging final resource usage."""
        # Stop the sampler thread if it exists
        if self.sampler_thread is not None:
            self.stop_event.set()
            self.sampler_thread.join(timeout=1.0)  # Wait up to 1 second for thread to stop
        
        # Get final resource snapshot
        end_snapshot = get_resource_snapshot()
        duration = time.perf_counter() - self.start_time
        
        # Calculate RSS delta
        if self.start_snapshot is not None:
            rss_delta = end_snapshot["rss_mib"] - self.start_snapshot["rss_mib"]
            rss_delta_str = f"{rss_delta:+.1f}MiB"
        else:
            rss_delta_str = "N/A"
        
        # Check if an exception occurred
        status = "error" if exc_type is not None else "ok"
        
        # Log end information
        self.logger.info(
            f"[res] {self.label} end "
            f"rss={end_snapshot['rss_mib']:.1f}MiB "
            f"Δrss={rss_delta_str} "
            f"cpu={end_snapshot['cpu_percent']:.1f}% "
            f"duration={duration:.2f}s "
            f"status={status}"
        )
    
    def _sample_resources(self) -> None:
        """Periodically sample and log resource usage."""
        # Prime CPU percentage calculation
        psutil.Process().cpu_percent()
        
        while not self.stop_event.wait(self.interval_s or 1.0):
            try:
                snapshot = get_resource_snapshot()
                self.logger.info(
                    f"[res] {self.label} sample "
                    f"rss={snapshot['rss_mib']:.1f}MiB "
                    f"cpu={snapshot['cpu_percent']:.1f}% "
                    f"avail={snapshot['available_mib']:.1f}MiB"
                )
            except Exception:
                # Don't let sampling errors break the main operation
                pass