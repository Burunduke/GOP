"""
Утилиты для работы с файлами
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

# Type aliases for better type safety
FilePath = Union[str, Path]
FileList = List[FilePath]


def ensure_dir(directory: FilePath) -> None:
    """
    Создание директории, если она не существует

    Args:
        directory: Путь к директории
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_extension(file_path: FilePath) -> str:
    """
    Получение расширения файла

    Args:
        file_path: Путь к файлу

    Returns:
        Расширение файла (без точки)
    """
    return os.path.splitext(file_path)[1][1:].lower()


def validate_file_path(
    file_path: FilePath, extensions: Optional[List[str]] = None
) -> bool:
    """
    Проверка существования файла и его расширения

    Args:
        file_path: Путь к файлу
        extensions: Список допустимых расширений

    Returns:
        True если файл существует и имеет допустимое расширение
    """
    if not os.path.exists(file_path):
        return False

    if extensions:
        file_ext = get_file_extension(file_path)
        return file_ext in [ext.lower() for ext in extensions]

    return True


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Копирование файла

    Args:
        src: Исходный путь
        dst: Целевой путь
    """
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Перемещение файла

    Args:
        src: Исходный путь
        dst: Целевой путь
    """
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)


def delete_file(file_path: Union[str, Path]) -> None:
    """
    Удаление файла

    Args:
        file_path: Путь к файлу
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Получение размера файла в байтах

    Args:
        file_path: Путь к файлу

    Returns:
        Размер файла в байтах
    """
    return os.path.getsize(file_path)


def find_files(directory: Union[str, Path], pattern: str = "*") -> List[str]:
    """
    Поиск файлов в директории по шаблону

    Args:
        directory: Директория поиска
        pattern: Шаблон имени файла

    Returns:
        Список найденных файлов
    """
    from glob import glob

    return glob(os.path.join(directory, pattern))


def create_backup(file_path: Union[str, Path], backup_suffix: str = ".bak") -> str:
    """
    Создание резервной копии файла

    Args:
        file_path: Путь к файлу
        backup_suffix: Суффикс для резервной копии

    Returns:
        Путь к резервной копии
    """
    backup_path = str(file_path) + backup_suffix
    copy_file(file_path, backup_path)
    return backup_path
