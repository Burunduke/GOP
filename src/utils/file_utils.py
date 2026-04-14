"""
File utility functions for the GOP project.

This module provides file operations including directory management, file validation,
copying, moving, and backup operations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union
from glob import glob

# Type aliases for better type safety
FilePath = Union[str, Path]
FileList = List[FilePath]


def ensure_dir(directory: FilePath) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to the directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_extension(file_path: FilePath) -> str:
    """
    Get file extension without the dot.

    Args:
        file_path: Path to the file

    Returns:
        File extension in lowercase (without dot)
    """
    return os.path.splitext(file_path)[1][1:].lower()


def validate_file_path(
    file_path: FilePath, extensions: Optional[List[str]] = None
) -> bool:
    """
    Validate file existence and extension.

    Args:
        file_path: Path to the file
        extensions: List of allowed extensions

    Returns:
        True if file exists and has valid extension
    """
    if not os.path.exists(file_path):
        return False

    if extensions:
        file_ext = get_file_extension(file_path)
        # Normalize extensions to remove leading dots if present
        normalized_extensions = [ext.lstrip('.').lower() for ext in extensions]
        return file_ext in normalized_extensions

    return True


def copy_file(src: FilePath, dst: FilePath) -> None:
    """
    Copy file from source to destination.

    Args:
        src: Source file path
        dst: Destination file path
    """
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def move_file(src: FilePath, dst: FilePath) -> None:
    """
    Move file from source to destination.

    Args:
        src: Source file path
        dst: Destination file path
    """
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)


def delete_file(file_path: FilePath) -> None:
    """
    Delete file if it exists.

    Args:
        file_path: Path to the file
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def get_file_size(file_path: FilePath) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def find_files(directory: FilePath, pattern: str = "*") -> List[str]:
    """
    Find files in directory matching pattern.

    Args:
        directory: Directory to search
        pattern: File name pattern

    Returns:
        List of found files
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    return glob(os.path.join(directory, pattern))


def create_backup(file_path: FilePath, backup_suffix: str = ".bak") -> str:
    """
    Create backup copy of a file.

    Args:
        file_path: Path to the file
        backup_suffix: Suffix for backup file

    Returns:
        Path to the backup file
    """
    backup_path = str(file_path) + backup_suffix
    copy_file(file_path, backup_path)
    return backup_path


__all__ = [
    "ensure_dir",
    "get_file_extension",
    "validate_file_path",
    "copy_file",
    "move_file",
    "delete_file",
    "get_file_size",
    "find_files",
    "create_backup",
]
