"""
Конфигурационный модуль для управления настройками проекта
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Класс для управления конфигурацией проекта
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация конфигурации
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Получить путь к файлу конфигурации по умолчанию"""
        # Ищем config.yaml в корневой директории проекта
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / "config" / "config.yaml"
        return str(config_file)
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Загрузить конфигурацию из файла
        
        Returns:
            Словарь с настройками
        """
        if not os.path.exists(self.config_path):
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Получить конфигурацию по умолчанию
        
        Returns:
            Словарь с настройками по умолчанию
        """
        return {
            'processing': {
                'max_image_size': 10000,
                'compression_ratio': 0.125,
                'batch_size': 32,
                'num_workers': 4
            },
            'segmentation': {
                'model_path': 'models/deeplabv3_resnet101.pth',
                'device': 'auto',
                'confidence_threshold': 0.5
            },
            'indices': {
                'sensor_types': ['RGB', 'Multispectral', 'Hyperspectral'],
                'default_indices': [
                    'GNDVI', 'MCARI', 'MNLI', 'OSAVI', 'TVI',
                    'SIPI2', 'mARI', 'NDWI', 'MSI'
                ]
            },
            'output': {
                'results_dir': 'results',
                'save_intermediate': True,
                'output_format': 'GeoTIFF'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/gop.log'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получить значение параметра конфигурации
        
        Args:
            key: Ключ параметра (поддерживает вложенные ключи через точку)
            default: Значение по умолчанию
            
        Returns:
            Значение параметра
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Установить значение параметра конфигурации
        
        Args:
            key: Ключ параметра (поддерживает вложенные ключи через точку)
            value: Значение параметра
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Сохранить конфигурацию в файл
        
        Args:
            path: Путь для сохранения (если None, используется текущий путь)
        """
        save_path = path or self.config_path
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Обновить конфигурацию из словаря
        
        Args:
            config_dict: Словарь с новыми настройками
        """
        self._deep_update(self._config, config_dict)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Рекурсивное обновление словаря
        
        Args:
            base_dict: Базовый словарь
            update_dict: Словарь с обновлениями
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    @property
    def config(self) -> Dict[str, Any]:
        """Получить полный словарь конфигурации"""
        return self._config.copy()


# Глобальный экземпляр конфигурации
config = Config()