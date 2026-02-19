"""
Утилиты валидации для GUI приложения GOP
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime


def validate_project_data(project_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Валидация данных проекта
    
    Args:
        project_data: Данные проекта для валидации
        
    Returns:
        Результат валидации
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Валидация названия проекта
    if 'name' not in project_data or not project_data['name']:
        result['errors'].append('Название проекта обязательно')
        result['valid'] = False
    elif len(project_data['name']) < 3:
        result['errors'].append('Название проекта должно содержать минимум 3 символа')
        result['valid'] = False
    elif len(project_data['name']) > 100:
        result['errors'].append('Название проекта не должно превышать 100 символов')
        result['valid'] = False
    elif not re.match(r'^[a-zA-Zа-яА-Я0-9\s\-_]+$', project_data['name']):
        result['errors'].append('Название проекта содержит недопустимые символы')
        result['valid'] = False
    
    # Валидация описания
    if 'description' in project_data and project_data['description']:
        if len(project_data['description']) > 1000:
            result['errors'].append('Описание проекта не должно превышать 1000 символов')
            result['valid'] = False
    
    # Валидация файлов
    if 'files' in project_data and project_data['files']:
        if not isinstance(project_data['files'], list):
            result['errors'].append('Файлы должны быть представлены в виде списка')
            result['valid'] = False
        elif len(project_data['files']) > 50:
            result['warnings'].append('Количество файлов превышает рекомендуемый лимит (50)')
    
    # Валидация настроек
    if 'settings' in project_data and project_data['settings']:
        settings_validation = validate_processing_config(project_data['settings'])
        if not settings_validation['valid']:
            result['errors'].extend(settings_validation['errors'])
            result['valid'] = False
        result['warnings'].extend(settings_validation['warnings'])
    
    return result


def validate_processing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Валидация конфигурации обработки
    
    Args:
        config: Конфигурация обработки
        
    Returns:
        Результат валидации
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Валидация типа сенсора
    if 'sensor_type' in config:
        valid_sensor_types = ['hyperspectral', 'multispectral', 'rgb']
        if config['sensor_type'] not in valid_sensor_types:
            result['errors'].append(f'Недопустимый тип сенсора. Допустимые: {", ".join(valid_sensor_types)}')
            result['valid'] = False
    
    # Валидация вегетационных индексов
    if 'selected_indices' in config:
        if not isinstance(config['selected_indices'], list):
            result['errors'].append('Индексы должны быть представлены в виде списка')
            result['valid'] = False
        else:
            valid_indices = ['NDVI', 'EVI', 'SAVI', 'MSAVI', 'GNDVI', 'NDRE']
            for index in config['selected_indices']:
                if index not in valid_indices:
                    result['warnings'].append(f'Индекс {index} может не поддерживаться выбранным типом сенсора')
    
    # Валидация опций обработки
    if 'processing_options' in config:
        valid_options = ['atmospheric_correction', 'denoising', 'segmentation', 'geometric_correction']
        for option in config['processing_options']:
            if option not in valid_options:
                result['warnings'].append(f'Опция обработки {option} может не поддерживаться')
    
    # Валидация параметров качества
    if 'quality_parameters' in config:
        quality_params = config['quality_parameters']
        
        if 'cloud_threshold' in quality_params:
            threshold = quality_params['cloud_threshold']
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                result['errors'].append('Порог облачности должен быть числом от 0 до 1')
                result['valid'] = False
        
        if 'min_vegetation_coverage' in quality_params:
            coverage = quality_params['min_vegetation_coverage']
            if not isinstance(coverage, (int, float)) or not (0 <= coverage <= 1):
                result['errors'].append('Минимальная покрытость растительностью должна быть числом от 0 до 1')
                result['valid'] = False
    
    return result


def validate_file_upload_data(upload_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Валидация данных загрузки файлов
    
    Args:
        upload_data: Данные загрузки
        
    Returns:
        Результат валидации
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Валидация ID проекта
    if 'project_id' not in upload_data or not upload_data['project_id']:
        result['errors'].append('ID проекта обязателен')
        result['valid'] = False
    elif not re.match(r'^[a-f0-9\-]{36}$', upload_data['project_id']):
        result['errors'].append('Некорректный формат ID проекта')
        result['valid'] = False
    
    # Валидация файлов
    if 'files' not in upload_data or not upload_data['files']:
        result['errors'].append('Список файлов не может быть пустым')
        result['valid'] = False
    elif len(upload_data['files']) > 100:
        result['errors'].append('Превышено максимальное количество файлов (100)')
        result['valid'] = False
    
    # Валидация каждого файла
    total_size = 0
    for i, file_data in enumerate(upload_data.get('files', [])):
        if not isinstance(file_data, dict):
            result['errors'].append(f'Файл {i+1} должен быть объектом')
            result['valid'] = False
            continue
        
        if 'filename' not in file_data or not file_data['filename']:
            result['errors'].append(f'Файл {i+1} должен иметь имя')
            result['valid'] = False
        
        if 'size' in file_data:
            size = file_data['size']
            if not isinstance(size, int) or size < 0:
                result['errors'].append(f'Размер файла {i+1} должен быть положительным числом')
                result['valid'] = False
            else:
                total_size += size
                
                # Проверка размера отдельного файла (10GB)
                max_file_size = 10 * 1024 * 1024 * 1024
                if size > max_file_size:
                    result['errors'].append(f'Файл {i+1} превышает максимальный размер (10GB)')
                    result['valid'] = False
    
    # Проверка общего размера
    max_total_size = 50 * 1024 * 1024 * 1024  # 50GB
    if total_size > max_total_size:
        result['errors'].append(f'Общий размер файлов превышает лимит (50GB)')
        result['valid'] = False
    
    return result


def validate_analysis_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Валидация параметров анализа
    
    Args:
        params: Параметры анализа
        
    Returns:
        Результат валидации
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Валидация области анализа
    if 'area_of_interest' in params:
        aoi = params['area_of_interest']
        if isinstance(aoi, dict):
            if 'coordinates' in aoi:
                coords = aoi['coordinates']
                if not isinstance(coords, list) or len(coords) < 3:
                    result['errors'].append('Область интереса должна содержать минимум 3 точки')
                    result['valid'] = False
                else:
                    for coord in coords:
                        if not isinstance(coord, list) or len(coord) != 2:
                            result['errors'].append('Координаты должны быть парами [долгота, широта]')
                            result['valid'] = False
                        elif not all(isinstance(c, (int, float)) for c in coord):
                            result['errors'].append('Координаты должны быть числовыми')
                            result['valid'] = False
    
    # Валидация временного периода
    if 'time_period' in params:
        time_period = params['time_period']
        if isinstance(time_period, dict):
            if 'start_date' in time_period:
                if not _is_valid_date(time_period['start_date']):
                    result['errors'].append('Некорректный формат начальной даты')
                    result['valid'] = False
            
            if 'end_date' in time_period:
                if not _is_valid_date(time_period['end_date']):
                    result['errors'].append('Некорректный формат конечной даты')
                    result['valid'] = False
            
            # Проверка хронологии дат
            if ('start_date' in time_period and 'end_date' in time_period and
                _is_valid_date(time_period['start_date']) and _is_valid_date(time_period['end_date'])):
                
                start_date = datetime.fromisoformat(time_period['start_date'])
                end_date = datetime.fromisoformat(time_period['end_date'])
                
                if start_date >= end_date:
                    result['errors'].append('Начальная дата должна быть раньше конечной')
                    result['valid'] = False
                
                # Проверка разумного периода (не более 1 года)
                if (end_date - start_date).days > 365:
                    result['warnings'].append('Период анализа превышает 1 год, что может замедлить обработку')
    
    # Валидация пороговых значений
    if 'thresholds' in params:
        thresholds = params['thresholds']
        for key, value in thresholds.items():
            if not isinstance(value, (int, float)):
                result['errors'].append(f'Порог {key} должен быть числовым')
                result['valid'] = False
            elif not (0 <= value <= 1):
                result['warnings'].append(f'Порог {key} должен быть в диапазоне [0, 1]')
    
    return result


def validate_export_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Валидация параметров экспорта
    
    Args:
        params: Параметры экспорта
        
    Returns:
        Результат валидации
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Валидация формата экспорта
    if 'format' in params:
        valid_formats = ['geotiff', 'shapefile', 'csv', 'json', 'pdf']
        if params['format'] not in valid_formats:
            result['errors'].append(f'Неподдерживаемый формат экспорта. Допустимые: {", ".join(valid_formats)}')
            result['valid'] = False
    
    # Валидация разрешения
    if 'resolution' in params:
        resolution = params['resolution']
        if not isinstance(resolution, (int, float)) or resolution <= 0:
            result['errors'].append('Разрешение должно быть положительным числом')
            result['valid'] = False
        elif resolution > 1000:
            result['warnings'].append('Высокое разрешение может привести к большим размерам файлов')
    
    # Валидация системы координат
    if 'coordinate_system' in params:
        cs = params['coordinate_system']
        valid_cs = ['WGS84', 'UTM', 'Web Mercator']
        if cs not in valid_cs:
            result['warnings'].append(f'Система координат {cs} может не поддерживаться')
    
    return result


def _is_valid_date(date_string: str) -> bool:
    """
    Проверка корректности даты в формате ISO
    
    Args:
        date_string: Строка с датой
        
    Returns:
        True если дата корректна
    """
    try:
        datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


def sanitize_string(input_string: str, max_length: int = 255) -> str:
    """
    Очистка строки от потенциально опасных символов
    
    Args:
        input_string: Входная строка
        max_length: Максимальная длина
        
    Returns:
        Очищенная строка
    """
    if not isinstance(input_string, str):
        return ''
    
    # Удаление HTML тегов
    cleaned = re.sub(r'<[^>]+>', '', input_string)
    
    # Удаление специальных символов
    cleaned = re.sub(r'[<>"\'\&]', '', cleaned)
    
    # Ограничение длины
    cleaned = cleaned[:max_length]
    
    return cleaned.strip()


def validate_email(email: str) -> bool:
    """
    Валидация email адреса
    
    Args:
        email: Email адрес
        
    Returns:
        True если email корректен
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Валидация географических координат
    
    Args:
        lat: Широта
        lon: Долгота
        
    Returns:
        True если координаты корректны
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180