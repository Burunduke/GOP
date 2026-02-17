"""
Определения вегетационных индексов
"""

import numpy as np
from typing import Dict, Callable, List, Tuple


class IndexDefinitions:
    """
    Класс с определениями вегетационных индексов
    """
    
    # Определения индексов озеленения
    GREENNESS_INDICES = {
        'GNDVI': {
            'name': 'Green Normalized Difference Vegetation Index',
            'formula': '(NIR - Green) / (NIR + Green)',
            'description': 'Индекс для оценки содержания хлорофилла',
            'required_bands': ['NIR', 'Green'],
            'range': (-1, 1),
            'function': lambda nir, green: (nir - green) / (nir + green + 1e-8)
        },
        'MCARI': {
            'name': 'Modified Chlorophyll Absorption Ratio Index',
            'formula': '((RedEdge - Red) - 0.2 * (RedEdge - Green)) * (RedEdge / Red)',
            'description': 'Модифицированный индекс поглощения хлорофилла',
            'required_bands': ['RedEdge', 'Red', 'Green'],
            'range': (0, 1),
            'function': lambda red_edge, red, green: ((red_edge - red) - 0.2 * (red_edge - green)) * (red_edge / (red + 1e-8))
        },
        'MNLI': {
            'name': 'Modified Non-Linear Index',
            'formula': '(NIR^2 - Red) / (NIR^2 + Red)',
            'description': 'Модифицированный нелинейный индекс',
            'required_bands': ['NIR', 'Red'],
            'range': (-1, 1),
            'function': lambda nir, red: (nir**2 - red) / (nir**2 + red + 1e-8)
        },
        'OSAVI': {
            'name': 'Optimized Soil Adjusted Vegetation Index',
            'formula': '(NIR - Red) / (NIR + Red + 0.16)',
            'description': 'Оптимизированный индекс с поправкой на почву',
            'required_bands': ['NIR', 'Red'],
            'range': (-1, 1),
            'function': lambda nir, red: (nir - red) / (nir + red + 0.16)
        },
        'TVI': {
            'name': 'Triangular Vegetation Index',
            'formula': '0.5 * (120 * (NIR - Green) - 200 * (Red - Green))',
            'description': 'Треугольный вегетационный индекс',
            'required_bands': ['NIR', 'Green', 'Red'],
            'range': (0, 1),
            'function': lambda nir, green, red: 0.5 * (120 * (nir - green) - 200 * (red - green))
        }
    }
    
    # Определения индексов поглощения света
    LIGHT_ABSORPTION_INDICES = {
        'SIPI2': {
            'name': 'Structure Insensitive Pigment Index 2',
            'formula': '(NIR - Blue) / (NIR - Red)',
            'description': 'Структурно-независимый пигментный индекс',
            'required_bands': ['NIR', 'Blue', 'Red'],
            'range': (0, 2),
            'function': lambda nir, blue, red: (nir - blue) / (nir - red + 1e-8)
        },
        'mARI': {
            'name': 'modified Anthocyanin Reflectance Index',
            'formula': '(Green - Red) / (Green + Red)',
            'description': 'Модифицированный индекс отражения антоцианов',
            'required_bands': ['Green', 'Red'],
            'range': (-1, 1),
            'function': lambda green, red: (green - red) / (green + red + 1e-8)
        }
    }
    
    # Определения индексов насыщения водой
    WATER_CONTENT_INDICES = {
        'NDWI': {
            'name': 'Normalized Difference Water Index',
            'formula': '(Green - NIR) / (Green + NIR)',
            'description': 'Нормализованный разностный водный индекс',
            'required_bands': ['Green', 'NIR'],
            'range': (-1, 1),
            'function': lambda green, nir: (green - nir) / (green + nir + 1e-8)
        },
        'MSI': {
            'name': 'Moisture Stress Index',
            'formula': 'NIR / SWIR',
            'description': 'Индекс водного стресса',
            'required_bands': ['NIR', 'SWIR'],
            'range': (0, 3),
            'function': lambda nir, swir: nir / (swir + 1e-8)
        }
    }
    
    # Все индексы
    ALL_INDICES = {
        **GREENNESS_INDICES,
        **LIGHT_ABSORPTION_INDICES,
        **WATER_CONTENT_INDICES
    }
    
    # Группы индексов
    INDEX_GROUPS = {
        'greenness': list(GREENNESS_INDICES.keys()),
        'light_absorption': list(LIGHT_ABSORPTION_INDICES.keys()),
        'water_content': list(WATER_CONTENT_INDICES.keys())
    }
    
    @classmethod
    def get_index_info(cls, index_name: str) -> Dict:
        """
        Получить информацию об индексе
        
        Args:
            index_name: Название индекса
            
        Returns:
            Словарь с информацией об индексе
        """
        return cls.ALL_INDICES.get(index_name, {})
    
    @classmethod
    def get_indices_by_group(cls, group: str) -> Dict[str, Dict]:
        """
        Получить индексы по группе
        
        Args:
            group: Название группы ('greenness', 'light_absorption', 'water_content')
            
        Returns:
            Словарь с индексами группы
        """
        if group == 'greenness':
            return cls.GREENNESS_INDICES
        elif group == 'light_absorption':
            return cls.LIGHT_ABSORPTION_INDICES
        elif group == 'water_content':
            return cls.WATER_CONTENT_INDICES
        else:
            return {}
    
    @classmethod
    def get_available_indices(cls, sensor_type: str) -> List[str]:
        """
        Получить доступные индексы для типа сенсора
        
        Args:
            sensor_type: Тип сенсора ('RGB', 'Multispectral', 'Hyperspectral')
            
        Returns:
            Список доступных индексов
        """
        # Определение доступных каналов для каждого типа сенсора
        sensor_bands = {
            'RGB': ['Blue', 'Green', 'Red'],
            'Multispectral': ['Blue', 'Green', 'Red', 'RedEdge', 'NIR'],
            'Hyperspectral': ['Blue', 'Green', 'Red', 'RedEdge', 'NIR', 'SWIR']
        }
        
        available_bands = sensor_bands.get(sensor_type, [])
        available_indices = []
        
        for index_name, index_info in cls.ALL_INDICES.items():
            required_bands = index_info.get('required_bands', [])
            if all(band in available_bands for band in required_bands):
                available_indices.append(index_name)
        
        return available_indices
    
    @classmethod
    def calculate_index(cls, 
                       index_name: str, 
                       bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Рассчитать вегетационный индекс
        
        Args:
            index_name: Название индекса
            bands: Словарь с спектральными каналами
            
        Returns:
            Массив со значениями индекса
        """
        index_info = cls.get_index_info(index_name)
        if not index_info:
            raise ValueError(f"Неизвестный индекс: {index_name}")
        
        function = index_info.get('function')
        if not function:
            raise ValueError(f"Отсутствует функция расчета для индекса: {index_name}")
        
        required_bands = index_info.get('required_bands', [])
        band_values = [bands.get(band) for band in required_bands]
        
        if any(band is None for band in band_values):
            missing_bands = [band for band, value in zip(required_bands, band_values) if value is None]
            raise ValueError(f"Отсутствуют каналы для расчета индекса {index_name}: {missing_bands}")
        
        return function(*band_values)
    
    @classmethod
    def normalize_index(cls, 
                       index_name: str, 
                       values: np.ndarray, 
                       mask: np.ndarray = None) -> np.ndarray:
        """
        Нормализовать значения индекса
        
        Args:
            index_name: Название индекса
            values: Значения индекса
            mask: Маска области интереса
            
        Returns:
            Нормализованные значения
        """
        index_info = cls.get_index_info(index_name)
        value_range = index_info.get('range', (-1, 1))
        
        # Применение маски
        if mask is not None:
            masked_values = values[mask > 0]
        else:
            masked_values = values
        
        # Нормализация в зависимости от типа индекса
        if index_name in cls.GREENNESS_INDICES:
            # Индексы озеленения: значения от 0 до 1
            normalized = np.clip(values, 0, 1)
        elif index_name in cls.LIGHT_ABSORPTION_INDICES:
            # Индексы поглощения света: инвертированная нормализация
            normalized = 1 - np.clip(values, 0, 1)
        elif index_name in cls.WATER_CONTENT_INDICES:
            # Индексы насыщения водой: нормализация в зависимости от типа
            if index_name == 'NDWI':
                normalized = np.clip(values, -1, 1) * 0.5 + 0.5
            else:  # MSI
                normalized = 1 - np.clip(values, 0, 2) * 0.5
        else:
            # Общая нормализация
            min_val, max_val = value_range
            normalized = (values - min_val) / (max_val - min_val + 1e-8)
            normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    @classmethod
    def get_index_formula(cls, index_name: str) -> str:
        """
        Получить математическую формулу индекса
        
        Args:
            index_name: Название индекса
            
        Returns:
            Строка с формулой
        """
        index_info = cls.get_index_info(index_name)
        return index_info.get('formula', '')
    
    @classmethod
    def get_index_description(cls, index_name: str) -> str:
        """
        Получить описание индекса
        
        Args:
            index_name: Название индекса
            
        Returns:
            Строка с описанием
        """
        index_info = cls.get_index_info(index_name)
        return index_info.get('description', '')