"""
Модуль валидации данных для гиперспектральной обработки
"""

import os
import numpy as np
from typing import Any, List, Optional


class HyperspectralValidator:
    """Класс для валидации гиперспектральных данных и параметров"""
    
    @staticmethod
    def validate_input_path(input_path: str) -> None:
        """
        Валидация пути к входному файлу
        
        Args:
            input_path: Путь к входному файлу
            
        Raises:
            ValueError: Если путь невалидный
            FileNotFoundError: Если файл не существует
        """
        if not input_path or not isinstance(input_path, str):
            raise ValueError("input_path должен быть непустой строкой")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Входной файл не найден: {input_path}")
    
    @staticmethod
    def validate_output_dir(output_dir: str) -> None:
        """
        Валидация выходной директории
        
        Args:
            output_dir: Путь к выходной директории
            
        Raises:
            ValueError: Если путь невалидный
        """
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("output_dir должен быть непустой строкой")
    
    @staticmethod
    def validate_file_format(file_path: str, supported_formats: List[str]) -> None:
        """
        Валидация формата файла
        
        Args:
            file_path: Путь к файлу
            supported_formats: Список поддерживаемых форматов
            
        Raises:
            ValueError: Если формат не поддерживается
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in supported_formats:
            raise ValueError(f"Неподдерживаемый формат файла: {file_ext}. Поддерживаемые форматы: {supported_formats}")
    
    @staticmethod
    def validate_image_data(image_data: np.ndarray) -> None:
        """
        Валидация данных изображения
        
        Args:
            image_data: Данные изображения
            
        Raises:
            ValueError: Если данные невалидные
        """
        if image_data is None or image_data.size == 0:
            raise ValueError("Входные данные изображения пусты или None")
        
        if len(image_data.shape) != 3:
            raise ValueError(f"Ожидается 3D массив, получен {len(image_data.shape)}D")
        
        rows, cols, bands = image_data.shape
        if rows <= 0 or cols <= 0 or bands <= 0:
            raise ValueError(f"Некорректные размеры изображения: {rows}x{cols}x{bands}")
    
    @staticmethod
    def validate_wavelengths(wavelengths: Optional[np.ndarray]) -> None:
        """
        Валидация длин волн
        
        Args:
            wavelengths: Массив длин волн
            
        Raises:
            ValueError: Если длины волн невалидные
        """
        if wavelengths is not None:
            if not isinstance(wavelengths, np.ndarray):
                raise ValueError("Длины волн должны быть numpy массивом")
            
            if wavelengths.size == 0:
                raise ValueError("Массив длин волн пуст")
            
            if np.any(np.isnan(wavelengths)) or np.any(np.isinf(wavelengths)):
                raise ValueError("Массив длин волн содержит NaN или Inf значения")
            
            if np.any(wavelengths <= 0):
                raise ValueError("Длины волн должны быть положительными")
    
    @staticmethod
    def validate_dataset(dataset: Any) -> None:
        """
        Валидация набора данных GDAL
        
        Args:
            dataset: Набор данных GDAL
            
        Raises:
            ValueError: Если набор данных невалидный
        """
        if dataset is None:
            raise ValueError("Набор данных не может быть None")
        
        if hasattr(dataset, 'RasterXSize') and hasattr(dataset, 'RasterYSize') and hasattr(dataset, 'RasterCount'):
            if dataset.RasterXSize <= 0 or dataset.RasterYSize <= 0 or dataset.RasterCount <= 0:
                raise ValueError(f"Некорректные размеры набора данных: {dataset.RasterYSize}x{dataset.RasterXSize}, каналов: {dataset.RasterCount}")
        else:
            raise ValueError("Набор данных не имеет необходимых атрибутов")
    
    @staticmethod
    def validate_processing_parameters(method: str, available_methods: List[str]) -> None:
        """
        Валидация параметров обработки
        
        Args:
            method: Метод обработки
            available_methods: Список доступных методов
            
        Raises:
            ValueError: Если метод недоступен
        """
        if method not in available_methods:
            raise ValueError(f"Неизвестный метод: {method}. Доступные методы: {available_methods}")
    
    @staticmethod
    def validate_pca_parameters(n_components: float) -> None:
        """
        Валидация параметров PCA
        
        Args:
            n_components: Количество компонентов или доля объясненной дисперсии
            
        Raises:
            ValueError: Если параметры невалидные
        """
        if not (0 < n_components <= 1) and not isinstance(n_components, int):
            raise ValueError("n_components должен быть в диапазоне (0, 1] или целым числом")
    
    @staticmethod
    def validate_rgb_bands(rgb_bands: tuple, max_bands: int) -> None:
        """
        Валидация параметров RGB композита
        
        Args:
            rgb_bands: Индексы каналов для RGB
            max_bands: Максимальное количество каналов
            
        Raises:
            ValueError: Если параметры невалидные
        """
        if not isinstance(rgb_bands, tuple) or len(rgb_bands) != 3:
            raise ValueError("rgb_bands должен быть кортежем из 3 элементов")
        
        if not all(isinstance(band, int) and band > 0 for band in rgb_bands):
            raise ValueError("rgb_bands должен содержать положительные целые числа")
        
        if max(rgb_bands) > max_bands:
            raise ValueError(f"Недостаточно каналов для RGB композита. Требуется: {max(rgb_bands)}, доступно: {max_bands}")