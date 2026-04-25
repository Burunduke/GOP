"""
Logging module for the GOP project.

This module provides logging utilities with configurable handlers and formatters.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Type aliases for better type safety
LogLevel = Union[int, str]
LogConfig = Dict[str, Any]

# Extract formatter to constant as recommended in refactoring plan
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(
    name: str,
    level: LogLevel = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure logger with specified settings.

    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file
        console: Enable console output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create log directory
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def create_default_log_file(base_dir: str = "logs") -> str:
    """
    Create default log file name with timestamp.

    Args:
        base_dir: Base directory for logs

    Returns:
        Path to log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(base_dir, f"gop_{timestamp}.log")
    return log_file


def configure_logging_from_config(config: LogConfig) -> None:
    """
    Configure logging based on configuration dictionary.

    Args:
        config: Logging configuration dictionary
    """
    level = config.get("level", "INFO")
    log_file = config.get("file")
    console = config.get("console", True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create log directory
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


__all__ = [
    "setup_logger",
    "get_logger",
    "create_default_log_file",
    "configure_logging_from_config",
]
