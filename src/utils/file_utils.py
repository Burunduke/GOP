"""
Утилиты для работы с файлами
"""

import os
import shutil
from pathlib import Path


def ensure_dir(directory):
    """
    Создание директории, если она не существует
    
    Args:
        directory (str): Путь к директории
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_extension(file_path):
    """
    Получение расширения файла
    
    Args:
        file_path (str): Путь к файлу
        
    Returns:
        str: Расширение файла (без точки)
    """
    return os.path.splitext(file_path)[1][1:].lower()


def validate_file_path(file_path, extensions=None):
    """
    Проверка существования файла и его расширения
    
    Args:
        file_path (str): Путь к файлу
        extensions (list, optional): Список допустимых расширений
        
    Returns:
        bool: True если файл существует и имеет допустимое расширение
    """
    if not os.path.exists(file_path):
        return False
    
    if extensions:
        file_ext = get_file_extension(file_path)
        return file_ext in [ext.lower() for ext in extensions]
    
    return True


def copy_file(src, dst):
    """
    Копирование файла
    
    Args:
        src (str): Исходный путь
        dst (str): Целевой путь
    """
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def move_file(src, dst):
    """
    Перемещение файла
    
    Args:
        src (str): Исходный путь
        dst (str): Целевой путь
    """
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)


def delete_file(file_path):
    """
    Удаление файла
    
    Args:
        file_path (str): Путь к файлу
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def get_file_size(file_path):
    """
    Получение размера файла в байтах
    
    Args:
        file_path (str): Путь к файлу
        
    Returns:
        int: Размер файла в байтах
    """
    return os.path.getsize(file_path)


def find_files(directory, pattern="*"):
    """
    Поиск файлов в директории по шаблону
    
    Args:
        directory (str): Директория поиска
        pattern (str): Шаблон имени файла
        
    Returns:
        list: Список найденных файлов
    """
    from glob import glob
    return glob(os.path.join(directory, pattern))


def create_backup(file_path, backup_suffix=".bak"):
    """
    Создание резервной копии файла
    
    Args:
        file_path (str): Путь к файлу
        backup_suffix (str): Суффикс для резервной копии
        
    Returns:
        str: Путь к резервной копии
    """
    backup_path = file_path + backup_suffix
    copy_file(file_path, backup_path)
    return backup_path