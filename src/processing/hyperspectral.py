"""
Модуль обработки гиперспектральных данных
Научно-ориентированная реализация с современными методами
"""

import os
import logging
import numpy as np
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

try:
    from osgeo import gdal
    from osgeo import osr
except ImportError:
    raise ImportError("GDAL library is required. Install with: pip install gdal")

try:
    from sklearn.decomposition import PCA
    from scipy.signal import savgol_filter
except ImportError:
    raise ImportError("Scientific libraries are required. Install with: pip install scikit-learn scipy")

from ..core.config import config
from ..utils.logger import setup_logger
from .hyperspectral.validators import HyperspectralValidator
from .hyperspectral.cache import HyperspectralCache
from .hyperspectral.corrections import HyperspectralCorrections
from .hyperspectral.denoising import HyperspectralDenoising


class HyperspectralProcessor:
    """
    Класс для обработки гиперспектральных данных
    Научно-ориентированная реализация с современными методами обработки
    """
    
    def __init__(self, cache_enabled: bool = True, cache_dir: str = None):
        """
        Инициализация процессора гиперспектральных данных
        
        Args:
            cache_enabled: Включить кэширование результатов
            cache_dir: Директория для кэша (по умолчанию ~/.gop_cache)
        """
        self.logger = setup_logger('HyperspectralProcessor')
        self.supported_formats = ['.bil', '.hdr', '.dat', '.tif', '.tiff', '.img']
        
        # Инициализация специализированных модулей
        self.validator = HyperspectralValidator()
        self.cache = HyperspectralCache(cache_enabled, cache_dir)
        self.corrections = HyperspectralCorrections(self.logger)
        self.denoising = HyperspectralDenoising(self.logger)
        
        # Научные параметры обработки
        self.denoising_methods = self.denoising.denoising_methods
        self.correction_methods = self.corrections.correction_methods
        
        self.logger.info(f"Процессор гиперспектральных данных инициализирован. Кэширование: {'включено' if cache_enabled else 'выключено'}")
        
    def clear_cache(self) -> None:
        """Очистка кэша"""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Получение статистики кэша
        
        Returns:
            Словарь со статистикой кэша
        """
        return self.cache.get_stats()

    def process(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Полный цикл научной обработки гиперспектральных данных
        
        Args:
            input_path: Путь к входному файлу
            output_dir: Директория для сохранения результатов
            
        Returns:
            Словарь с результатами обработки
        """
        try:
            # Валидация входных параметров с использованием специализированного валидатора
            self.validator.validate_input_path(input_path)
            self.validator.validate_output_dir(output_dir)
            self.validator.validate_file_format(input_path, self.supported_formats)
            
            self.logger.info(f"Начало научной обработки гиперспектральных данных: {input_path}")
            
            # Создание выходной директории
            os.makedirs(output_dir, exist_ok=True)
            
            # Чтение данных
            self.logger.info("Чтение гиперспектральных данных")
            dataset, image_data, wavelengths = self._read_hyperspectral_data(input_path)
            
            # Валидация прочитанных данных
            self.validator.validate_image_data(image_data)
            self.validator.validate_wavelengths(wavelengths)
            self.validator.validate_dataset(dataset)
            
            # Предварительный анализ данных
            self.logger.info("Предварительный анализ данных")
            data_quality = self._analyze_data_quality(image_data)
            
            # Радиометрическая коррекция
            self.logger.info("Радиометрическая коррекция")
            corrected_data = self.corrections.radiometric_correction(image_data, method='empirical_line')
            
            # Атмосферная коррекция (упрощенная)
            self.logger.info("Атмосферная коррекция")
            atmospheric_corrected = self.corrections.atmospheric_correction(corrected_data)
            
            # Шумоподавление с использованием нескольких методов
            self.logger.info("Шумоподавление")
            denoised_data = self.denoising.advanced_noise_reduction(atmospheric_corrected)
            
            # Спектральная калибровка
            self.logger.info("Спектральная калибровка")
            calibrated_data = self._spectral_calibration(denoised_data, wavelengths)
            
            # Конвертация в TIFF
            self.logger.info("Конвертация в TIFF формат")
            tiff_paths = self._convert_to_tiff(calibrated_data, dataset, output_dir)
            
            # Извлечение метаданных
            metadata = self._extract_metadata(dataset, wavelengths, data_quality)
            
            # Создание научного отчета
            scientific_report = self._create_scientific_report(
                image_data, corrected_data, denoised_data, calibrated_data
            )
            
            results = {
                'input_path': input_path,
                'output_dir': output_dir,
                'tiff_paths': tiff_paths,
                'metadata': metadata,
                'data_quality': data_quality,
                'shape': calibrated_data.shape,
                'bands': len(tiff_paths),
                'wavelengths': wavelengths.tolist() if wavelengths is not None else None,
                'scientific_report': scientific_report
            }
            
            self.logger.info("Научная обработка гиперспектральных данных завершена")
            return results
            
        except (FileNotFoundError, ValueError, TypeError) as e:
            self.logger.error(f"Ошибка валидации входных данных: {e}")
            raise
        except (RuntimeError, MemoryError) as e:
            self.logger.error(f"Ошибка выполнения обработки: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка научной обработки гиперспектральных данных: {e}")
            raise
    
    def _read_hyperspectral_data(self, input_path: str) -> Tuple[Any, np.ndarray, Optional[np.ndarray]]:
        """
        Чтение гиперспектральных данных с поддержкой метаданных
        
        Args:
            input_path: Путь к входному файлу
            
        Returns:
            Кортеж (dataset, image_data, wavelengths)
        """
        try:
            # Попытка открыть с помощью GDAL
            dataset = gdal.Open(input_path)
            if dataset is None:
                raise ValueError(f"Не удалось открыть файл с помощью GDAL: {input_path}")
            
            # Валидация набора данных
            self.validator.validate_dataset(dataset)
            
            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            bands = dataset.RasterCount
            
            # Проверка на слишком большие изображения
            max_pixels = 10000 * 10000  # 100M пикселей
            if rows * cols * bands > max_pixels:
                self.logger.warning(f"Очень большое изображение: {rows}x{cols}x{bands} пикселей")
            
            self.logger.info(f"Размер изображения: {rows}x{cols}, каналов: {bands}")
            
            # Извлечение всех каналов с валидацией
            image_data = np.zeros((rows, cols, bands), dtype=np.float32)
            
            for band in range(1, bands + 1):
                try:
                    band_data = dataset.GetRasterBand(band)
                    if band_data is None:
                        raise ValueError(f"Не удалось прочитать канал {band}")
                    
                    band_array = band_data.ReadAsArray()
                    if band_array is None:
                        raise ValueError(f"Не удалось прочитать данные канала {band}")
                    
                    image_data[:, :, band-1] = band_array.astype(np.float32)
                except Exception as e:
                    raise ValueError(f"Ошибка чтения канала {band}: {e}")
            
            # Извлечение длин волн из метаданных
            wavelengths = self._extract_wavelengths(dataset)
            
            return dataset, image_data, wavelengths
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Ошибка валидации при чтении данных: {e}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Ошибка выполнения GDAL при чтении данных: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка чтения данных: {e}")
            raise
    
    def _extract_wavelengths(self, dataset: Any) -> Optional[np.ndarray]:
        """
        Извлечение длин волн из метаданных
        
        Args:
            dataset: Набор данных GDAL
            
        Returns:
            Массив длин волн или None
        """
        try:
            # Попытка извлечь длины волн из различных источников метаданных
            metadata = dataset.GetMetadata()
            
            # Проверка различных форматов метаданных
            wavelength_keys = [
                'wavelengths', 'wavelength', 'band_wavelengths',
                'spectral_wavelengths', 'center_wavelengths'
            ]
            
            for key in wavelength_keys:
                if key in metadata:
                    wavelengths_str = metadata[key]
                    wavelengths = np.array([float(w) for w in wavelengths_str.split(',')])
                    return wavelengths
            
            # Попытка извлечь из метаданных отдельных каналов
            wavelengths = []
            for band in range(1, dataset.RasterCount + 1):
                band_data = dataset.GetRasterBand(band)
                band_metadata = band_data.GetMetadata()
                
                for key in ['wavelength', 'center_wavelength', 'band_wavelength']:
                    if key in band_metadata:
                        wavelengths.append(float(band_metadata[key]))
                        break
                else:
                    # Если длина волны не найдена, используем приближение
                    wavelengths.append(400 + (band - 1) * (2500 - 400) / dataset.RasterCount)
            
            return np.array(wavelengths)
            
        except Exception as e:
            self.logger.warning(f"Не удалось извлечь длины волн: {e}")
            return None
    
    def _analyze_data_quality(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Анализ качества данных
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Словарь с оценкой качества данных
        """
        try:
            # Валидация входных данных
            if image_data is None or image_data.size == 0:
                raise ValueError("Входные данные изображения пусты или None")
            
            if len(image_data.shape) != 3:
                raise ValueError(f"Ожидается 3D массив, получен {len(image_data.shape)}D")
            
            rows, cols, bands = image_data.shape
            
            # Базовая статистика с обработкой NaN/Inf
            valid_data = image_data[~np.isnan(image_data) & ~np.isinf(image_data)]
            
            if valid_data.size == 0:
                raise ValueError("Нет валидных данных для анализа качества")
            
            data_quality = {
                'total_pixels': rows * cols,
                'total_bands': bands,
                'valid_pixels': int(valid_data.size),
                'data_range': {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data))
                }
            }
            
            # Анализ пропущенных значений
            nan_count = np.sum(np.isnan(image_data))
            inf_count = np.sum(np.isinf(image_data))
            
            # Безопасное вычисление процентов
            total_pixels = image_data.size
            data_quality['missing_values'] = {
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
                'nan_percentage': float(nan_count / total_pixels * 100) if total_pixels > 0 else 0,
                'inf_percentage': float(inf_count / total_pixels * 100) if total_pixels > 0 else 0
            }
            
            # Анализ динамического диапазона по каналам
            band_ranges = []
            for band in range(bands):
                try:
                    band_data = image_data[:, :, band]
                    valid_band_data = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]
                    
                    if len(valid_band_data) > 0:
                        band_ranges.append({
                            'band': band + 1,
                            'min': float(np.min(valid_band_data)),
                            'max': float(np.max(valid_band_data)),
                            'dynamic_range': float(np.max(valid_band_data) - np.min(valid_band_data)),
                            'snr': self._calculate_snr(valid_band_data)
                        })
                    else:
                        self.logger.warning(f"Канал {band + 1} не содержит валидных данных")
                except Exception as e:
                    self.logger.warning(f"Ошибка анализа канала {band + 1}: {e}")
            
            data_quality['band_quality'] = band_ranges
            
            # Общая оценка качества с безопасными вычислениями
            valid_snrs = [b['snr'] for b in band_ranges if b.get('snr', 0) > 0]
            avg_snr = np.mean(valid_snrs) if valid_snrs else 0
            
            data_quality['overall_quality'] = {
                'average_snr': float(avg_snr),
                'quality_score': self._calculate_quality_score(data_quality)
            }
            
            return data_quality
            
        except ValueError as e:
            self.logger.error(f"Ошибка валидации при анализе качества данных: {e}")
            return {'error': str(e)}
        except (RuntimeError, MemoryError) as e:
            self.logger.error(f"Ошибка выполнения при анализе качества данных: {e}")
            return {'error': str(e)}
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка анализа качества данных: {e}")
            return {'error': str(e)}
    
    def _calculate_snr(self, data: np.ndarray) -> float:
        """
        Расчет отношения сигнал/шум
        
        Args:
            data: Данные для анализа
            
        Returns:
            Отношение сигнал/шум
        """
        try:
            # Валидация входных данных
            if data is None or data.size == 0:
                return 0.0
            
            # Удаление NaN и бесконечных значений
            valid_data = data[~np.isnan(data) & ~np.isinf(data)]
            
            if valid_data.size == 0:
                return 0.0
            
            signal = np.mean(valid_data)
            noise = np.std(valid_data)
            
            # Обработка деления на ноль
            if noise == 0 or np.isclose(noise, 0):
                return float('inf') if signal != 0 else 0.0
            
            return signal / noise
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Ошибка расчета SNR: {e}")
            return 0.0
    
    def _calculate_quality_score(self, data_quality: Dict[str, Any]) -> float:
        """
        Расчет общей оценки качества данных
        
        Args:
            data_quality: Результаты анализа качества
            
        Returns:
            Оценка качества от 0 до 1
        """
        try:
            score = 1.0
            
            # Штраф за пропущенные значения
            missing_percentage = data_quality['missing_values']['nan_percentage'] + \
                               data_quality['missing_values']['inf_percentage']
            score -= missing_percentage / 100 * 0.3
            
            # Штраф за низкое SNR
            if 'overall_quality' in data_quality:
                avg_snr = data_quality['overall_quality']['average_snr']
                if avg_snr < 10:
                    score -= 0.3
                elif avg_snr < 20:
                    score -= 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _radiometric_correction(self, 
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
            if method == 'dark_current':
                return self._dark_current_correction(image_data)
            elif method == 'empirical_line':
                return self._empirical_line_correction(image_data)
            elif method == 'flat_field':
                return self._flat_field_correction(image_data)
            else:
                self.logger.warning(f"Неизвестный метод коррекции: {method}")
                return image_data
                
        except Exception as e:
            self.logger.error(f"Ошибка радиометрической коррекции: {e}")
            return image_data
    
    def _dark_current_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Коррекция темнового тока
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Скорректированные данные
        """
        # Калибровка по темному току (1-й перцентиль)
        dark_reference = np.percentile(image_data, 1, axis=(0, 1))
        corrected = image_data - dark_reference
        
        # Ограничение отрицательных значений
        corrected = np.maximum(corrected, 0)
        
        self.logger.info("Коррекция темнового тока завершена")
        return corrected
    
    def _empirical_line_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Эмпирическая линейная коррекция
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Скорректированные данные
        """
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
    
    def _flat_field_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Коррекция плоского поля
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Скорректированные данные
        """
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
            corrected = self._empirical_line_correction(image_data)
        
        # Ограничение значений
        corrected = np.clip(corrected, 0, 1)
        
        self.logger.info("Коррекция плоского поля завершена")
        return corrected
    
    def _atmospheric_correction(self, image_data: np.ndarray) -> np.ndarray:
        """
        Упрощенная атмосферная коррекция
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Скорректированные данные
        """
        try:
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
            
        except Exception as e:
            self.logger.error(f"Ошибка атмосферной коррекции: {e}")
            return image_data
    
    def _advanced_noise_reduction(self, 
                                image_data: np.ndarray, 
                                method: str = 'pca') -> np.ndarray:
        """
        Продвинутое шумоподавление
        
        Args:
            image_data: Входные данные изображения
            method: Метод шумоподавления
            
        Returns:
            Данные после шумоподавления
        """
        try:
            if method == 'pca':
                return self._pca_denoising(image_data)
            elif method == 'mnf':
                return self._mnf_denoising(image_data)
            elif method == 'wavelet':
                return self._wavelet_denoising(image_data)
            elif method == 'savgol':
                return self._savgol_denoising(image_data)
            else:
                self.logger.warning(f"Неизвестный метод шумоподавления: {method}")
                return image_data
                
        except Exception as e:
            self.logger.error(f"Ошибка шумоподавления: {e}")
            return image_data
    
    def _pca_denoising(self, image_data: np.ndarray, n_components: float = 0.95) -> np.ndarray:
        """
        Шумоподавление с помощью PCA
        
        Args:
            image_data: Входные данные изображения
            n_components: Количество компонентов или доля объясненной дисперсии
            
        Returns:
            Данные после шумоподавления
        """
        try:
            rows, cols, bands = image_data.shape
            
            # Изменение формы данных для PCA
            reshaped = image_data.reshape(-1, bands)
            
            # Удаление NaN и бесконечных значений
            valid_mask = ~np.isnan(reshaped).any(axis=1) & ~np.isinf(reshaped).any(axis=1)
            valid_data = reshaped[valid_mask]
            
            if len(valid_data) < 100:
                self.logger.warning("Недостаточно валидных данных для PCA")
                return image_data
            
            # Применение PCA
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(valid_data)
            
            # Обратное преобразование
            denoised = pca.inverse_transform(transformed)
            
            # Восстановление исходной формы
            denoised_image = np.zeros_like(reshaped)
            denoised_image[valid_mask] = denoised
            denoised_image = denoised_image.reshape(rows, cols, -1)
            
            self.logger.info(f"PCA шумоподавление завершено. Компонентов: {pca.n_components_}")
            return denoised_image
            
        except Exception as e:
            self.logger.error(f"Ошибка PCA шумоподавления: {e}")
            return image_data
    
    def _mnf_denoising(self, image_data: np.ndarray) -> np.ndarray:
        """
        Шумоподавление с помощью MNF (Minimum Noise Fraction)
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Данные после шумоподавления
        """
        try:
            # Упрощенная реализация MNF
            # В реальной системе здесь должна быть полная реализация MNF
            
            rows, cols, bands = image_data.shape
            reshaped = image_data.reshape(-1, bands)
            
            # Оценка ковариационных матриц
            valid_mask = ~np.isnan(reshaped).any(axis=1) & ~np.isinf(reshaped).any(axis=1)
            valid_data = reshaped[valid_mask]
            
            if len(valid_data) < 100:
                return image_data
            
            # Расчет ковариационных матриц
            cov_signal = np.cov(valid_data.T)
            cov_noise = np.eye(bands) * np.var(valid_data) * 0.1  # Упрощенная оценка шума
            
            # MNF преобразование
            try:
                eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(cov_noise, cov_signal))
                
                # Сортировка по собственным значениям
                idx = eigenvalues.argsort()[::-1]
                eigenvectors = eigenvectors[:, idx]
                
                # Преобразование данных
                transformed = valid_data @ eigenvectors
                
                # Обратное преобразование с использованием только главных компонент
                n_components = min(int(bands * 0.8), len(eigenvalues))
                reconstructed = transformed[:, :n_components] @ eigenvectors[:, :n_components].T
                
                # Восстановление исходной формы
                denoised_image = np.zeros_like(reshaped)
                denoised_image[valid_mask] = reconstructed
                denoised_image = denoised_image.reshape(rows, cols, -1)
                
                self.logger.info(f"MNF шумоподавление завершено. Компонентов: {n_components}")
                return denoised_image
                
            except np.linalg.LinAlgError:
                self.logger.warning("Ошибка в MNF преобразовании, используем PCA")
                return self._pca_denoising(image_data)
                
        except Exception as e:
            self.logger.error(f"Ошибка MNF шумоподавления: {e}")
            return image_data
    
    def _wavelet_denoising(self, image_data: np.ndarray) -> np.ndarray:
        """
        Вейвлет-шумоподавление
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Данные после шумоподавления
        """
        try:
            import pywt
            
            rows, cols, bands = image_data.shape
            denoised_image = np.zeros_like(image_data)
            
            for band in range(bands):
                band_data = image_data[:, :, band]
                
                # Вейвлет-преобразование
                coeffs = pywt.wavedec2(band_data, 'db4', level=2)
                
                # Пороговая обработка коэффициентов
                threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(band_data.size))
                coeffs_thresh = list(coeffs)
                coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode='soft') 
                                   for detail in coeffs_thresh[1:]]
                
                # Обратное вейвлет-преобразование
                denoised_image[:, :, band] = pywt.waverec2(coeffs_thresh, 'db4')
            
            self.logger.info("Вейвлет-шумоподавление завершено")
            return denoised_image
            
        except ImportError:
            self.logger.warning("PyWavelets не установлен, используем PCA")
            return self._pca_denoising(image_data)
        except Exception as e:
            self.logger.error(f"Ошибка вейвлет-шумоподавления: {e}")
            return image_data
    
    def _savgol_denoising(self, image_data: np.ndarray) -> np.ndarray:
        """
        Шумоподавление с помощью фильтра Савицкого-Голея с векторизованными вычислениями
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Данные после шумоподавления
        """
        try:
            # Валидация входных данных
            if image_data is None or image_data.size == 0:
                return image_data
            
            if len(image_data.shape) != 3:
                self.logger.warning(f"Ожидается 3D массив, получен {len(image_data.shape)}D")
                return image_data
            
            rows, cols, bands = image_data.shape
            denoised_image = np.zeros_like(image_data)
            
            # Определение оптимальных размеров окон
            row_window = min(11, cols)
            col_window = min(11, rows)
            
            # Убедимся, что размеры окон нечетные
            if row_window % 2 == 0:
                row_window -= 1
            if col_window % 2 == 0:
                col_window -= 1
            
            # Минимальные размеры для фильтра
            if row_window < 3 or col_window < 3:
                self.logger.warning("Изображение слишком маленькое для фильтра Савицкого-Голея")
                return image_data
            
            try:
                # Векторизованная обработка по каналам
                for band in range(bands):
                    band_data = image_data[:, :, band]
                    
                    # Проверка на валидные данные
                    if np.all(np.isnan(band_data)) or np.all(np.isinf(band_data)):
                        denoised_image[:, :, band] = band_data
                        continue
                    
                    # Векторизованное применение фильтра по строкам
                    filtered_rows = np.zeros_like(band_data)
                    
                    # Обработка строк с проверкой на NaN/Inf
                    valid_row_mask = ~np.all(np.isnan(band_data) | np.isinf(band_data), axis=1)
                    
                    if np.any(valid_row_mask):
                        # Применяем фильтр к валидным строкам
                        for i in np.where(valid_row_mask)[0]:
                            row_data = band_data[i, :]
                            if np.any(np.isnan(row_data)) or np.any(np.isinf(row_data)):
                                # Интерполяция для восстановления пропущенных значений
                                valid_mask = ~np.isnan(row_data) & ~np.isinf(row_data)
                                if np.sum(valid_mask) >= row_window:
                                    row_data = row_data.copy()
                                    row_data[~valid_mask] = np.interp(
                                        np.where(~valid_mask)[0],
                                        np.where(valid_mask)[0],
                                        row_data[valid_mask]
                                    )
                                else:
                                    filtered_rows[i, :] = row_data
                                    continue
                            
                            try:
                                filtered_rows[i, :] = savgol_filter(row_data,
                                                                   window_length=row_window,
                                                                   polyorder=min(3, row_window - 1))
                            except Exception as e:
                                self.logger.warning(f"Ошибка фильтрации строки {i}: {e}")
                                filtered_rows[i, :] = row_data
                    
                    # Векторизованное применение фильтра по столбцам
                    filtered_both = np.zeros_like(filtered_rows)
                    
                    # Обработка столбцов с проверкой на NaN/Inf
                    valid_col_mask = ~np.all(np.isnan(filtered_rows) | np.isinf(filtered_rows), axis=0)
                    
                    if np.any(valid_col_mask):
                        # Применяем фильтр к валидным столбцам
                        for j in np.where(valid_col_mask)[0]:
                            col_data = filtered_rows[:, j]
                            if np.any(np.isnan(col_data)) or np.any(np.isinf(col_data)):
                                # Интерполяция для восстановления пропущенных значений
                                valid_mask = ~np.isnan(col_data) & ~np.isinf(col_data)
                                if np.sum(valid_mask) >= col_window:
                                    col_data = col_data.copy()
                                    col_data[~valid_mask] = np.interp(
                                        np.where(~valid_mask)[0],
                                        np.where(valid_mask)[0],
                                        col_data[valid_mask]
                                    )
                                else:
                                    filtered_both[:, j] = col_data
                                    continue
                            
                            try:
                                filtered_both[:, j] = savgol_filter(col_data,
                                                                   window_length=col_window,
                                                                   polyorder=min(3, col_window - 1))
                            except Exception as e:
                                self.logger.warning(f"Ошибка фильтрации столбца {j}: {e}")
                                filtered_both[:, j] = col_data
                    
                    denoised_image[:, :, band] = filtered_both
                
                self.logger.info(f"Шумоподавление Савицкого-Голея завершено. Обработано {bands} каналов")
                return denoised_image
                
            except Exception as e:
                self.logger.error(f"Ошибка в векторизованном шумоподавлении: {e}")
                # Откат к исходному методу при ошибке
                return self._fallback_savgol_denoising(image_data)
            
        except Exception as e:
            self.logger.error(f"Ошибка шумоподавления Савицкого-Голея: {e}")
            return image_data
    
    def _fallback_savgol_denoising(self, image_data: np.ndarray) -> np.ndarray:
        """
        Запасной метод шумоподавления Савицкого-Голея с циклами (используется при ошибках)
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Данные после шумоподавления
        """
        try:
            rows, cols, bands = image_data.shape
            denoised_image = np.zeros_like(image_data)
            
            row_window = min(11, cols)
            col_window = min(11, rows)
            
            if row_window % 2 == 0:
                row_window -= 1
            if col_window % 2 == 0:
                col_window -= 1
            
            for band in range(bands):
                band_data = image_data[:, :, band]
                
                # Применение фильтра Савицкого-Голея к каждой строке и столбцу
                filtered_rows = np.zeros_like(band_data)
                for i in range(rows):
                    try:
                        filtered_rows[i, :] = savgol_filter(band_data[i, :],
                                                           window_length=row_window,
                                                           polyorder=min(3, row_window - 1))
                    except Exception:
                        filtered_rows[i, :] = band_data[i, :]
                
                filtered_both = np.zeros_like(band_data)
                for j in range(cols):
                    try:
                        filtered_both[:, j] = savgol_filter(filtered_rows[:, j],
                                                           window_length=col_window,
                                                           polyorder=min(3, col_window - 1))
                    except Exception:
                        filtered_both[:, j] = filtered_rows[:, j]
                
                denoised_image[:, :, band] = filtered_both
            
            return denoised_image
            
        except Exception as e:
            self.logger.error(f"Ошибка в запасном методе шумоподавления: {e}")
            return image_data
    
    def _spectral_calibration(self, 
                            image_data: np.ndarray, 
                            wavelengths: Optional[np.ndarray]) -> np.ndarray:
        """
        Спектральная калибровка данных
        
        Args:
            image_data: Входные данные изображения
            wavelengths: Длины волн
            
        Returns:
            Откалиброванные данные
        """
        try:
            if wavelengths is None:
                self.logger.warning("Длины волн не доступны, пропускаем спектральную калибровку")
                return image_data
            
            # Спектральная ресемплинг (если необходимо)
            calibrated_data = self._spectral_resampling(image_data, wavelengths)
            
            # Спектральная сглаживающая фильтрация
            calibrated_data = self._spectral_smoothing(calibrated_data)
            
            self.logger.info("Спектральная калибровка завершена")
            return calibrated_data
            
        except Exception as e:
            self.logger.error(f"Ошибка спектральной калибровки: {e}")
            return image_data
    
    def _spectral_resampling(self,
                           image_data: np.ndarray,
                           wavelengths: np.ndarray) -> np.ndarray:
        """
        Спектральный ресемплинг с кэшированием
        
        Args:
            image_data: Входные данные изображения
            wavelengths: Длины волн
            
        Returns:
            Ресемплированные данные
        """
        # Используем кэш для получения или вычисления результата
        return self.cache.get_or_compute(
            (image_data, wavelengths),
            'spectral_resampling',
            self._compute_spectral_resampling
        )
    
    def _compute_spectral_resampling(self, inputs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Вычисление спектрального ресемплинга
        
        Args:
            inputs: Кортеж (image_data, wavelengths)
            
        Returns:
            Ресемплированные данные
        """
        image_data, wavelengths = inputs
        
        # Определение стандартных спектральных диапазонов
        standard_ranges = {
            'blue': (450, 495),
            'green': (495, 570),
            'red': (620, 750),
            'red_edge': (700, 780),
            'nir': (750, 1400),
            'swir1': (1400, 1800),
            'swir2': (2000, 2500)
        }
        
        # Создание новых каналов на основе стандартных диапазонов
        new_bands = []
        band_names = []
        
        for range_name, (min_wl, max_wl) in standard_ranges.items():
            # Поиск каналов в диапазоне
            mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
            
            if np.sum(mask) > 0:
                # Усреднение каналов в диапазоне с векторизацией
                range_data = np.mean(image_data[:, :, mask], axis=2)
                new_bands.append(range_data)
                band_names.append(range_name)
        
        if new_bands:
            calibrated_data = np.stack(new_bands, axis=2)
            self.logger.info(f"Спектральный ресемплинг: {image_data.shape[2]} -> {len(new_bands)} каналов")
            return calibrated_data
        else:
            return image_data
    
    def _spectral_smoothing(self, image_data: np.ndarray) -> np.ndarray:
        """
        Спектральная сглаживающая фильтрация с векторизованными вычислениями
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Сглаженные данные
        """
        try:
            # Валидация входных данных
            if image_data is None or image_data.size == 0:
                return image_data
            
            if len(image_data.shape) != 3:
                self.logger.warning(f"Ожидается 3D массив, получен {len(image_data.shape)}D")
                return image_data
            
            if image_data.shape[2] < 3:
                return image_data
            
            rows, cols, bands = image_data.shape
            
            # Векторизованное применение фильтра Савицкого-Голея
            # Изменение формы для применения фильтра ко всем пикселям одновременно
            reshaped_data = image_data.reshape(-1, bands)
            
            # Проверка на минимальное количество каналов для фильтра
            window_length = min(5, bands)
            if window_length < 3:
                self.logger.warning("Недостаточно каналов для сглаживания Савицкого-Голея")
                return image_data
            
            # Убедимся, что window_length нечетное
            if window_length % 2 == 0:
                window_length -= 1
            
            try:
                # Векторизованное применение фильтра
                smoothed_reshaped = np.zeros_like(reshaped_data)
                
                # Применяем фильтр к каждой строке (спектральному профилю)
                for i in range(reshaped_data.shape[0]):
                    spectrum = reshaped_data[i, :]
                    
                    # Проверка на NaN и Inf значения
                    if np.any(np.isnan(spectrum)) or np.any(np.isinf(spectrum)):
                        # Заменяем NaN/Inf на соседние значения или нули
                        valid_mask = ~np.isnan(spectrum) & ~np.isinf(spectrum)
                        if np.sum(valid_mask) >= window_length:
                            # Интерполяция для восстановления пропущенных значений
                            spectrum = spectrum.copy()
                            spectrum[~valid_mask] = np.interp(
                                np.where(~valid_mask)[0],
                                np.where(valid_mask)[0],
                                spectrum[valid_mask]
                            )
                        else:
                            smoothed_reshaped[i, :] = spectrum
                            continue
                    
                    # Применение фильтра Савицкого-Голея
                    try:
                        smoothed_spectrum = savgol_filter(spectrum,
                                                        window_length=window_length,
                                                        polyorder=min(2, window_length - 1))
                        smoothed_reshaped[i, :] = smoothed_spectrum
                    except Exception as e:
                        self.logger.warning(f"Ошибка применения фильтра к пикселю {i}: {e}")
                        smoothed_reshaped[i, :] = spectrum
                
                # Восстановление исходной формы
                smoothed_data = smoothed_reshaped.reshape(rows, cols, bands)
                
                self.logger.info(f"Спектральное сглаживание завершено. Обработано {rows*cols} пикселей")
                return smoothed_data
                
            except Exception as e:
                self.logger.error(f"Ошибка в векторизованном сглаживании: {e}")
                # Откат к исходному методу при ошибке
                return self._fallback_spectral_smoothing(image_data)
            
        except Exception as e:
            self.logger.error(f"Ошибка спектрального сглаживания: {e}")
            return image_data
    
    def _fallback_spectral_smoothing(self, image_data: np.ndarray) -> np.ndarray:
        """
        Запасной метод спектрального сглаживания с циклами (используется при ошибках)
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Сглаженные данные
        """
        try:
            rows, cols, bands = image_data.shape
            smoothed_data = np.zeros_like(image_data)
            
            window_length = min(5, bands)
            if window_length % 2 == 0:
                window_length -= 1
            
            for i in range(rows):
                for j in range(cols):
                    # Применение сглаживания к спектральному профилю
                    spectrum = image_data[i, j, :]
                    
                    # Проверка на валидные данные
                    if np.any(np.isnan(spectrum)) or np.any(np.isinf(spectrum)):
                        smoothed_data[i, j, :] = spectrum
                        continue
                    
                    try:
                        smoothed_spectrum = savgol_filter(spectrum,
                                                        window_length=window_length,
                                                        polyorder=min(2, window_length - 1))
                        smoothed_data[i, j, :] = smoothed_spectrum
                    except Exception:
                        smoothed_data[i, j, :] = spectrum
            
            return smoothed_data
            
        except Exception as e:
            self.logger.error(f"Ошибка в запасном методе сглаживания: {e}")
            return image_data
    
    def _convert_to_tiff(self,
                        image_data: np.ndarray,
                        reference_dataset: Any,
                        output_dir: str) -> List[str]:
        """
        Конвертация данных в TIFF формат
        
        Args:
            image_data: Данные изображения
            reference_dataset: Ссылочный набор данных для геопривязки
            output_dir: Директория для сохранения
            
        Returns:
            Список путей к созданным TIFF файлам
        """
        try:
            # Валидация входных данных
            if image_data is None or image_data.size == 0:
                raise ValueError("Входные данные изображения пусты")
            
            if reference_dataset is None:
                raise ValueError("Ссылочный набор данных не может быть None")
            
            if not output_dir or not isinstance(output_dir, str):
                raise ValueError("output_dir должен быть непустой строкой")
            
            rows, cols, bands = image_data.shape
            
            if rows <= 0 or cols <= 0 or bands <= 0:
                raise ValueError(f"Некорректные размеры изображения: {rows}x{cols}x{bands}")
            
            tiff_paths = []
            
            # Создание выходной директории
            tiff_dir = os.path.join(output_dir, 'tiff_bands')
            os.makedirs(tiff_dir, exist_ok=True)
            
            # Копирование геопривязки
            geo_transform = reference_dataset.GetGeoTransform()
            projection = reference_dataset.GetProjection()
            
            # Сохранение каждого канала в отдельный TIFF файл
            for band in range(bands):
                try:
                    output_path = os.path.join(tiff_dir, f'band_{band+1:03d}.tif')
                    
                    # Создание выходного файла
                    driver = gdal.GetDriverByName('GTiff')
                    if driver is None:
                        raise RuntimeError("Не удалось получить драйвер GTiff")
                    
                    output = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
                    if output is None:
                        raise RuntimeError(f"Не удалось создать TIFF файл: {output_path}")
                    
                    # Копирование геопривязки
                    if geo_transform:
                        output.SetGeoTransform(geo_transform)
                    if projection:
                        output.SetProjection(projection)
                    
                    # Запись канала
                    output_band = output.GetRasterBand(1)
                    if output_band is None:
                        raise RuntimeError(f"Не удалось получить канал для записи: {output_path}")
                    
                    output_band.WriteArray(image_data[:, :, band])
                    
                    # Очистка
                    output = None
                    
                    tiff_paths.append(output_path)
                    
                except Exception as e:
                    self.logger.error(f"Ошибка сохранения канала {band + 1}: {e}")
                    # Продолжаем с другими каналами
                    continue
            
            if not tiff_paths:
                raise RuntimeError("Не удалось сохранить ни одного TIFF файла")
            
            self.logger.info(f"Создано {len(tiff_paths)} TIFF файлов")
            return tiff_paths
            
        except ValueError as e:
            self.logger.error(f"Ошибка валидации при конвертации в TIFF: {e}")
            raise
        except (RuntimeError, OSError) as e:
            self.logger.error(f"Ошибка выполнения при конвертации в TIFF: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка конвертации в TIFF: {e}")
            raise
    
    def _extract_metadata(self, 
                         dataset: Any, 
                         wavelengths: Optional[np.ndarray],
                         data_quality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлечение метаданных из набора данных
        
        Args:
            dataset: Набор данных GDAL
            wavelengths: Длины волн
            data_quality: Оценка качества данных
            
        Returns:
            Словарь с метаданными
        """
        try:
            metadata = {
                'width': dataset.RasterXSize,
                'height': dataset.RasterYSize,
                'bands': dataset.RasterCount,
                'driver': dataset.GetDriver().ShortName,
                'data_quality': data_quality
            }
            
            # Геопривязка
            geo_transform = dataset.GetGeoTransform()
            if geo_transform:
                metadata['geo_transform'] = geo_transform
                metadata['pixel_size'] = abs(geo_transform[1])
            
            # Проекция
            projection = dataset.GetProjection()
            if projection:
                metadata['projection'] = projection
                
                # Извлечение информации о системе координат
                srs = osr.SpatialReference()
                srs.ImportFromWkt(projection)
                metadata['coordinate_system'] = srs.GetName()
                metadata['epsg_code'] = srs.GetAuthorityCode(None)
            
            # Спектральная информация
            if wavelengths is not None:
                metadata['wavelengths'] = {
                    'min': float(np.min(wavelengths)),
                    'max': float(np.max(wavelengths)),
                    'count': len(wavelengths),
                    'range': float(np.max(wavelengths) - np.min(wavelengths))
                }
            
            # Метаданные
            dataset_metadata = dataset.GetMetadata()
            if dataset_metadata:
                metadata['dataset_metadata'] = dataset_metadata
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения метаданных: {e}")
            return {}
    
    def _create_scientific_report(self, 
                                original_data: np.ndarray,
                                corrected_data: np.ndarray,
                                denoised_data: np.ndarray,
                                final_data: np.ndarray) -> Dict[str, Any]:
        """
        Создание научного отчета об обработке
        
        Args:
            original_data: Исходные данные
            corrected_data: Скорректированные данные
            denoised_data: Данные после шумоподавления
            final_data: Финальные данные
            
        Returns:
            Научный отчет
        """
        try:
            report = {
                'processing_steps': [
                    'radiometric_correction',
                    'atmospheric_correction', 
                    'noise_reduction',
                    'spectral_calibration'
                ],
                'data_improvement': self._calculate_data_improvement(
                    original_data, corrected_data, denoised_data, final_data
                ),
                'spectral_analysis': self._analyze_spectral_changes(
                    original_data, final_data
                )
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Ошибка создания научного отчета: {e}")
            return {'error': str(e)}
    
    def _calculate_data_improvement(self, 
                                  original_data: np.ndarray,
                                  corrected_data: np.ndarray,
                                  denoised_data: np.ndarray,
                                  final_data: np.ndarray) -> Dict[str, Any]:
        """
        Расчет улучшения качества данных
        
        Args:
            original_data: Исходные данные
            corrected_data: Скорректированные данные
            denoised_data: Данные после шумоподавления
            final_data: Финальные данные
            
        Returns:
            Метрики улучшения
        """
        try:
            improvement = {}
            
            # Улучшение отношения сигнал/шум
            original_snr = np.mean([self._calculate_snr(original_data[:, :, i]) 
                                   for i in range(original_data.shape[2])])
            final_snr = np.mean([self._calculate_snr(final_data[:, :, i]) 
                                for i in range(final_data.shape[2])])
            
            improvement['snr_improvement'] = {
                'original': float(original_snr),
                'final': float(final_snr),
                'improvement_factor': float(final_snr / original_snr) if original_snr > 0 else 0
            }
            
            # Уменьшение шума
            original_noise = np.mean([np.std(original_data[:, :, i]) 
                                     for i in range(original_data.shape[2])])
            final_noise = np.mean([np.std(final_data[:, :, i]) 
                                  for i in range(final_data.shape[2])])
            
            improvement['noise_reduction'] = {
                'original': float(original_noise),
                'final': float(final_noise),
                'reduction_percentage': float((1 - final_noise / original_noise) * 100) if original_noise > 0 else 0
            }
            
            return improvement
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета улучшения данных: {e}")
            return {}
    
    def _analyze_spectral_changes(self, 
                                original_data: np.ndarray,
                                final_data: np.ndarray) -> Dict[str, Any]:
        """
        Анализ спектральных изменений
        
        Args:
            original_data: Исходные данные
            final_data: Финальные данные
            
        Returns:
            Анализ спектральных изменений
        """
        try:
            analysis = {}
            
            # Средние спектральные профили
            original_mean_spectrum = np.mean(original_data.reshape(-1, original_data.shape[2]), axis=0)
            final_mean_spectrum = np.mean(final_data.reshape(-1, final_data.shape[2]), axis=0)
            
            analysis['spectral_correlation'] = float(np.corrcoef(original_mean_spectrum, final_mean_spectrum)[0, 1])
            
            # Спектральное расстояние
            spectral_distance = np.linalg.norm(original_mean_spectrum - final_mean_spectrum)
            analysis['spectral_distance'] = float(spectral_distance)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа спектральных изменений: {e}")
            return {}
    
    def get_band_info(self, input_path: str) -> Dict[str, Any]:
        """
        Получить информацию о спектральных каналах
        
        Args:
            input_path: Путь к файлу
            
        Returns:
            Словарь с информацией о каналах
        """
        try:
            # Валидация входных параметров
            if not input_path or not isinstance(input_path, str):
                raise ValueError("input_path должен быть непустой строкой")
            
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Файл не найден: {input_path}")
            
            dataset = gdal.Open(input_path)
            if dataset is None:
                raise ValueError(f"Не удалось открыть файл с помощью GDAL: {input_path}")
            
            if dataset.RasterCount <= 0:
                raise ValueError(f"Файл не содержит спектральных каналов: {input_path}")
            
            band_info = {
                'total_bands': dataset.RasterCount,
                'bands': []
            }
            
            for band in range(1, dataset.RasterCount + 1):
                try:
                    band_data = dataset.GetRasterBand(band)
                    if band_data is None:
                        self.logger.warning(f"Не удалось прочитать канал {band}")
                        continue
                    
                    stats = band_data.GetStatistics(True, True)
                    if stats is None or len(stats) < 4:
                        self.logger.warning(f"Не удалось получить статистику для канала {band}")
                        stats = [0, 0, 0, 0]  # Значения по умолчанию
                    
                    band_info['bands'].append({
                        'band_number': band,
                        'min': stats[0],
                        'max': stats[1],
                        'mean': stats[2],
                        'stddev': stats[3],
                        'no_data_value': band_data.GetNoDataValue()
                    })
                except Exception as e:
                    self.logger.warning(f"Ошибка обработки канала {band}: {e}")
                    continue
            
            return band_info
            
        except (FileNotFoundError, ValueError) as e:
            self.logger.error(f"Ошибка валидации при получении информации о каналах: {e}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Ошибка выполнения GDAL при получении информации о каналах: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка получения информации о каналах: {e}")
            raise
    
    def create_rgb_composite(self,
                           tiff_paths: List[str],
                           rgb_bands: Tuple[int, int, int] = (30, 20, 10),
                           output_path: str = None) -> str:
        """
        Создание RGB композита из гиперспектральных данных
        
        Args:
            tiff_paths: Список путей к TIFF файлам
            rgb_bands: Индексы каналов для RGB (R, G, B)
            output_path: Путь для сохранения результата
            
        Returns:
            Путь к созданному RGB композиту
        """
        try:
            # Валидация входных параметров
            if not tiff_paths or not isinstance(tiff_paths, list):
                raise ValueError("tiff_paths должен быть непустым списком")
            
            if len(tiff_paths) < max(rgb_bands):
                raise ValueError(f"Недостаточно каналов для RGB композита. Требуется: {max(rgb_bands)}, доступно: {len(tiff_paths)}")
            
            if not all(isinstance(band, int) and band > 0 for band in rgb_bands):
                raise ValueError("rgb_bands должен содержать положительные целые числа")
            
            # Проверка существования файлов
            for i, path in enumerate(tiff_paths):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"TIFF файл не найден: {path}")
            
            # Чтение RGB каналов с обработкой ошибок
            try:
                r_dataset = gdal.Open(tiff_paths[rgb_bands[0]-1])
                g_dataset = gdal.Open(tiff_paths[rgb_bands[1]-1])
                b_dataset = gdal.Open(tiff_paths[rgb_bands[2]-1])
                
                if r_dataset is None or g_dataset is None or b_dataset is None:
                    raise RuntimeError("Не удалось открыть один из RGB каналов")
                
                r_band = r_dataset.ReadAsArray()
                g_band = g_dataset.ReadAsArray()
                b_band = b_dataset.ReadAsArray()
                
                if r_band is None or g_band is None or b_band is None:
                    raise RuntimeError("Не удалось прочитать данные одного из RGB каналов")
                    
            except Exception as e:
                raise RuntimeError(f"Ошибка чтения RGB каналов: {e}")
            
            # Нормализация с процентильным растяжением и безопасным делением
            def normalize_band(band, lower_percent=2, upper_percent=98):
                try:
                    if band is None or band.size == 0:
                        return np.zeros_like(band) if band is not None else np.array([])
                    
                    band_min, band_max = np.percentile(band, [lower_percent, upper_percent])
                    
                    # Безопасное деление
                    if band_max > band_min and not np.isclose(band_max - band_min, 0):
                        normalized = (band - band_min) / (band_max - band_min)
                        return np.clip(normalized, 0, 1)
                    else:
                        self.logger.warning("Нулевой динамический диапазон в канале, возвращаем нули")
                        return np.zeros_like(band)
                        
                except Exception as e:
                    self.logger.warning(f"Ошибка нормализации канала: {e}")
                    return np.zeros_like(band) if band is not None else np.array([])
            
            r_norm = normalize_band(r_band)
            g_norm = normalize_band(g_band)
            b_norm = normalize_band(b_band)
            
            # Проверка размеров каналов
            if not (r_norm.shape == g_norm.shape == b_norm.shape):
                raise RuntimeError(f"Несовместимые размеры каналов: R{r_norm.shape}, G{g_norm.shape}, B{b_norm.shape}")
            
            # Создание RGB композита
            rgb_composite = np.stack([r_norm, g_norm, b_norm], axis=2)
            
            # Определение пути сохранения
            if output_path is None:
                output_path = os.path.join(os.path.dirname(tiff_paths[0]), 'rgb_composite.tif')
            
            # Создание выходной директории
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Сохранение с обработкой ошибок
            rows, cols = rgb_composite.shape[:2]
            driver = gdal.GetDriverByName('GTiff')
            if driver is None:
                raise RuntimeError("Не удалось получить драйвер GTiff")
            
            output = driver.Create(output_path, cols, rows, 3, gdal.GDT_Float32)
            if output is None:
                raise RuntimeError(f"Не удалось создать выходной файл: {output_path}")
            
            try:
                # Копирование геопривязки из первого канала
                reference_dataset = gdal.Open(tiff_paths[0])
                if reference_dataset is not None:
                    geo_transform = reference_dataset.GetGeoTransform()
                    projection = reference_dataset.GetProjection()
                    
                    if geo_transform:
                        output.SetGeoTransform(geo_transform)
                    if projection:
                        output.SetProjection(projection)
                
                # Запись каналов
                for i, band_data in enumerate([r_norm, g_norm, b_norm]):
                    output_band = output.GetRasterBand(i + 1)
                    if output_band is None:
                        raise RuntimeError(f"Не удалось получить канал {i+1} для записи")
                    output_band.WriteArray(band_data)
                
            finally:
                # Очистка ресурсов
                output = None
                if 'reference_dataset' in locals() and reference_dataset is not None:
                    reference_dataset = None
            
            self.logger.info(f"RGB композит сохранен: {output_path}")
            return output_path
            
        except (FileNotFoundError, ValueError) as e:
            self.logger.error(f"Ошибка валидации при создании RGB композита: {e}")
            raise
        except (RuntimeError, OSError) as e:
            self.logger.error(f"Ошибка выполнения при создании RGB композита: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка создания RGB композита: {e}")
            raise