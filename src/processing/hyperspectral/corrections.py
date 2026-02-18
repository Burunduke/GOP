"""
Модуль коррекции гиперспектральных данных
"""

import numpy as np
import logging
from typing import Any, Dict, Optional


class HyperspectralCorrections:
    """Класс для коррекции гиперспектральных данных"""
    
    def __init__(self, logger: logging.Logger = None):
        """
        Инициализация класса коррекции
        
        Args:
            logger: Логгер для записи сообщений
        """
        self.logger = logger or logging.getLogger(__name__)
        self.correction_methods = ['dark_current', 'empirical_line', 'flat_field']
    
    def radiometric_correction(self, 
                              image_data: np.ndarray, 
                              method: str = 'empirical_line') -> np.ndarray:
        """
        Радиометрическая коррекция изображения
        
        Args:
            image_data: Входные данные изображения
            method: Метод коррекции
            
        Returns:
            Скорректированные данные
        """
        try:
            # Валидация входных данных
            if image_data is None or image_data.size == 0:
                raise ValueError("Входные данные изображения пусты")
            
            if method not in self.correction_methods:
                self.logger.warning(f"Неизвестный метод коррекции: {method}. Доступные методы: {self.correction_methods}")
                return image_data
            
            if method == 'dark_current':
                return self.dark_current_correction(image_data)
            elif method == 'empirical_line':
                return self.empirical_line_correction(image_data)
            elif method == 'flat_field':
                return self.flat_field_correction(image_data)
            else:
                self.logger.warning(f"Неизвестный метод коррекции: {method}")
                return image_data
                
        except ValueError as e:
            self.logger.error(f"Ошибка валидации при радиометрической коррекции: {e}")
            return image_data
        except (RuntimeError, MemoryError) as e:
            self.logger.error(f"Ошибка выполнения при радиометрической коррекции: {e}")
            return image_data
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка радиометрической коррекции: {e}")
            return image_data
    
    def dark_current_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Коррекция темнового тока
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Скорректированные данные
        """
        try:
            # Калибровка по темному току (1-й перцентиль)
            dark_reference = np.percentile(image_data, 1, axis=(0, 1))
            corrected = image_data - dark_reference
            
            # Ограничение отрицательных значений
            corrected = np.maximum(corrected, 0)
            
            self.logger.info("Коррекция темнового тока завершена")
            return corrected
            
        except Exception as e:
            self.logger.error(f"Ошибка коррекции темнового тока: {e}")
            return image_data
    
    def empirical_line_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Эмпирическая линейная коррекция
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Скорректированные данные
        """
        try:
            # Калибровка по темному току (1-й перцентиль)
            dark_reference = np.percentile(image_data, 1, axis=(0, 1))
            corrected = image_data - dark_reference
            
            # Калибровка по белому эталону (99-й перцентиль)
            white_reference = np.percentile(image_data, 99, axis=(0, 1))
            denominator = white_reference - dark_reference
            
            # Обработка деления на ноль с использованием np.where
            corrected = np.where(
                np.abs(denominator) > 1e-8,
                corrected / denominator,
                0.0  # Значение по умолчанию при делении на ноль
            )
            
            # Ограничение значений
            corrected = np.clip(corrected, 0, 1)
            
            self.logger.info("Эмпирическая линейная коррекция завершена")
            return corrected
            
        except Exception as e:
            self.logger.error(f"Ошибка эмпирической линейной коррекции: {e}")
            return image_data
    
    def flat_field_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Коррекция плоского поля
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Скорректированные данные
        """
        try:
            # Создание эталонного спектра на основе ярких областей
            bright_threshold = np.percentile(image_data, 95, axis=(0, 1))
            bright_mask = np.all(image_data > bright_threshold, axis=2)
            
            if np.sum(bright_mask) > 100:  # Минимальное количество пикселей
                reference_spectrum = np.mean(image_data[bright_mask], axis=0)
                
                # Безопасное деление с обработкой нулевых значений
                corrected = np.where(
                    np.abs(reference_spectrum) > 1e-8,
                    image_data / reference_spectrum,
                    image_data  # Оставляем исходные значения если делитель близок к нулю
                )
            else:
                # Альтернативный метод
                corrected = self.empirical_line_correction(image_data)
            
            # Ограничение значений
            corrected = np.clip(corrected, 0, 1)
            
            self.logger.info("Коррекция плоского поля завершена")
            return corrected
            
        except Exception as e:
            self.logger.error(f"Ошибка коррекции плоского поля: {e}")
            return image_data
    
    def atmospheric_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Упрощенная атмосферная коррекция
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Скорректированные данные
        """
        try:
            # Валидация входных данных
            if image_data is None or image_data.size == 0:
                raise ValueError("Входные данные изображения пусты")
            
            # Упрощенная атмосферная коррекция на основе статистики
            # В реальной системе здесь должна быть более сложная модель
            
            # Оценка атмосферной дымки на основе темных объектов
            dark_pixels = np.percentile(image_data, 2, axis=(0, 1))
            
            # Коррекция с учетом атмосферных эффектов
            corrected = image_data - dark_pixels
            corrected = np.maximum(corrected, 0)
            
            # Нормализация с безопасным делением
            max_values = np.percentile(corrected, 98, axis=(0, 1))
            corrected = np.where(
                np.abs(max_values) > 1e-8,
                corrected / max_values,
                0.0  # Значение по умолчанию при делении на ноль
            )
            corrected = np.clip(corrected, 0, 1)
            
            self.logger.info("Атмосферная коррекция завершена")
            return corrected
            
        except ValueError as e:
            self.logger.error(f"Ошибка валидации при атмосферной коррекции: {e}")
            return image_data
        except (RuntimeError, MemoryError) as e:
            self.logger.error(f"Ошибка выполнения при атмосферной коррекции: {e}")
            return image_data
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка атмосферной коррекции: {e}")
            return image_data
    
    def calculate_correction_statistics(self, 
                                      original_data: np.ndarray, 
                                      corrected_data: np.ndarray) -> Dict[str, Any]:
        """
        Расчет статистики коррекции
        
        Args:
            original_data: Исходные данные
            corrected_data: Скорректированные данные
            
        Returns:
            Словарь со статистикой коррекции
        """
        try:
            if original_data is None or corrected_data is None:
                raise ValueError("Входные данные не могут быть None")
            
            if original_data.shape != corrected_data.shape:
                raise ValueError("Формы исходных и скорректированных данных должны совпадать")
            
            # Удаление NaN и Inf значений
            valid_mask = ~np.isnan(original_data) & ~np.isinf(original_data) & \
                        ~np.isnan(corrected_data) & ~np.isinf(corrected_data)
            
            if not np.any(valid_mask):
                return {'error': 'Нет валидных данных для анализа'}
            
            orig_valid = original_data[valid_mask]
            corr_valid = corrected_data[valid_mask]
            
            statistics = {
                'original_mean': float(np.mean(orig_valid)),
                'original_std': float(np.std(orig_valid)),
                'original_min': float(np.min(orig_valid)),
                'original_max': float(np.max(orig_valid)),
                'corrected_mean': float(np.mean(corr_valid)),
                'corrected_std': float(np.std(corr_valid)),
                'corrected_min': float(np.min(corr_valid)),
                'corrected_max': float(np.max(corr_valid)),
                'mean_change': float(np.mean(corr_valid) - np.mean(orig_valid)),
                'std_change': float(np.std(corr_valid) - np.std(orig_valid)),
                'dynamic_range_change': float(
                    (np.max(corr_valid) - np.min(corr_valid)) - 
                    (np.max(orig_valid) - np.min(orig_valid))
                )
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета статистики коррекции: {e}")
            return {'error': str(e)}