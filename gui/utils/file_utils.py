"""
Утилиты для работы с файлами в GUI приложении GOP
"""

import os
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def validate_file_format(file_path: str, supported_formats: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Валидация формата файла
    
    Args:
        file_path: Путь к файлу
        supported_formats: Список поддерживаемых форматов
        
    Returns:
        Результат валидации
    """
    if supported_formats is None:
        supported_formats = ['.bil', '.hdr', '.tif', '.tiff', '.dat', '.png', '.jpg', '.jpeg']
    
    result = {
        'valid': False,
        'error': None,
        'file_info': {}
    }
    
    try:
        # Проверка существования файла
        if not os.path.exists(file_path):
            result['error'] = 'Файл не существует'
            return result
        
        # Получение информации о файле
        file_path_obj = Path(file_path)
        file_size = file_path_obj.stat().st_size
        file_ext = file_path_obj.suffix.lower()
        
        result['file_info'] = {
            'name': file_path_obj.name,
            'size': file_size,
            'extension': file_ext,
            'mime_type': mimetypes.guess_type(file_path)[0],
            'created_time': datetime.fromtimestamp(file_path_obj.stat().st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(file_path_obj.stat().st_mtime).isoformat()
        }
        
        # Проверка формата
        if file_ext not in supported_formats:
            result['error'] = f'Неподдерживаемый формат: {file_ext}. Поддерживаемые: {", ".join(supported_formats)}'
            return result
        
        # Проверка размера (максимум 10GB)
        max_size = 10 * 1024 * 1024 * 1024
        if file_size > max_size:
            result['error'] = f'Файл слишком большой: {format_file_size(file_size)}. Максимум: {format_file_size(max_size)}'
            return result
        
        # Дополнительная проверка для специфических форматов
        if file_ext in ['.bil', '.hdr']:
            if not _validate_hyperspectral_file(file_path):
                result['error'] = 'Файл не является валидным гиперспектральным данным'
                return result
        
        result['valid'] = True
        
    except Exception as e:
        result['error'] = f'Ошибка при валидации файла: {str(e)}'
    
    return result


def format_file_size(size_bytes: int) -> str:
    """
    Форматирование размера файла в человекочитаемый формат
    
    Args:
        size_bytes: Размер в байтах
        
    Returns:
        Отформатированный размер
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Получение метаданных файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Метаданные файла
    """
    try:
        file_path_obj = Path(file_path)
        stat = file_path_obj.stat()
        
        metadata = {
            'name': file_path_obj.name,
            'path': str(file_path_obj.absolute()),
            'size': stat.st_size,
            'size_formatted': format_file_size(stat.st_size),
            'extension': file_path_obj.suffix.lower(),
            'mime_type': mimetypes.guess_type(file_path)[0],
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed_time': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'is_readable': os.access(file_path, os.R_OK),
            'is_writable': os.access(file_path, os.W_OK),
        }
        
        # Добавление специфичной информации для разных форматов
        if file_path_obj.suffix.lower() in ['.bil', '.hdr']:
            metadata.update(_get_hyperspectral_metadata(file_path))
        elif file_path_obj.suffix.lower() in ['.tif', '.tiff']:
            metadata.update(_get_geotiff_metadata(file_path))
        
        return metadata
        
    except Exception as e:
        return {
            'error': f'Ошибка получения метаданных: {str(e)}',
            'path': file_path
        }


def _validate_hyperspectral_file(file_path: str) -> bool:
    """
    Валидация гиперспектрального файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        True если файл валидный
    """
    try:
        # Базовая проверка - для полноценной валидации нужен GDAL/spectral
        file_path_obj = Path(file_path)
        
        # Проверка наличия HDR файла для BIL
        if file_path_obj.suffix.lower() == '.bil':
            hdr_file = file_path_obj.with_suffix('.hdr')
            if not hdr_file.exists():
                return False
        
        # Проверка минимального размера
        if file_path_obj.stat().st_size < 1024:  # Минимум 1KB
            return False
        
        return True
        
    except Exception:
        return False


def _get_hyperspectral_metadata(file_path: str) -> Dict[str, Any]:
    """
    Получение метаданных гиперспектрального файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Метаданные гиперспектрального файла
    """
    metadata = {
        'file_type': 'hyperspectral',
        'sensor_type': 'unknown',
        'bands_count': 0,
        'wavelength_range': None
    }
    
    try:
        file_path_obj = Path(file_path)
        
        # Попытка прочитать HDR файл
        if file_path_obj.suffix.lower() == '.hdr':
            hdr_content = file_path_obj.read_text()
            # Простая парсилка HDR файла
            if 'samples' in hdr_content.lower():
                # Извлечение базовой информации из HDR
                metadata['samples'] = _extract_hdr_value(hdr_content, 'samples')
                metadata['lines'] = _extract_hdr_value(hdr_content, 'lines')
                metadata['bands'] = _extract_hdr_value(hdr_content, 'bands')
                metadata['bands_count'] = metadata.get('bands', 0)
        
        elif file_path_obj.suffix.lower() == '.bil':
            # Поиск соответствующего HDR файла
            hdr_file = file_path_obj.with_suffix('.hdr')
            if hdr_file.exists():
                return _get_hyperspectral_metadata(str(hdr_file))
        
    except Exception:
        pass
    
    return metadata


def _get_geotiff_metadata(file_path: str) -> Dict[str, Any]:
    """
    Получение метаданных GeoTIFF файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Метаданные GeoTIFF файла
    """
    metadata = {
        'file_type': 'geotiff',
        'coordinate_system': 'unknown',
        'pixel_size': None,
        'bounds': None
    }
    
    try:
        # Для полноценного извлечения метаданных нужен GDAL/rasterio
        # Здесь базовая реализация
        file_path_obj = Path(file_path)
        
        # Проверка размера для определения примерного разрешения
        size_mb = file_path_obj.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            metadata['estimated_resolution'] = 'high'
        elif size_mb > 10:
            metadata['estimated_resolution'] = 'medium'
        else:
            metadata['estimated_resolution'] = 'low'
        
    except Exception:
        pass
    
    return metadata


def _extract_hdr_value(hdr_content: str, key: str) -> Optional[int]:
    """
    Извлечение числового значения из HDR файла
    
    Args:
        hdr_content: Содержимое HDR файла
        key: Ключ для поиска
        
    Returns:
        Числовое значение или None
    """
    try:
        lines = hdr_content.split('\n')
        for line in lines:
            if key.lower() in line.lower() and '=' in line:
                value_str = line.split('=')[1].strip()
                return int(value_str)
    except (ValueError, IndexError):
        pass
    return None


def create_safe_filename(filename: str) -> str:
    """
    Создание безопасного имени файла
    
    Args:
        filename: Исходное имя файла
        
    Returns:
        Безопасное имя файла
    """
    import re
    
    # Удаление недопустимых символов
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Ограничение длины
    if len(safe_name) > 255:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:255-len(ext)] + ext
    
    return safe_name


def ensure_directory_exists(directory: str) -> bool:
    """
    Убедиться что директория существует
    
    Args:
        directory: Путь к директории
        
    Returns:
        True если директория существует или создана
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_directory_size(directory: str) -> int:
    """
    Получение размера директории в байтах
    
    Args:
        directory: Путь к директории
        
    Returns:
        Размер в байтах
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception:
        pass
    return total_size


def cleanup_old_files(directory: str, max_age_days: int = 30) -> int:
    """
    Очистка старых файлов в директории
    
    Args:
        directory: Путь к директории
        max_age_days: Максимальный возраст файлов в днях
        
    Returns:
        Количество удаленных файлов
    """
    deleted_count = 0
    cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
    
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.unlink(file_path)
                        deleted_count += 1
                except Exception:
                    pass
    except Exception:
        pass
    
    return deleted_count