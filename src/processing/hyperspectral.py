"""
Модуль обработки гиперспектральных данных
Научно-ориентированная реализация с современными методами
"""

import os
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

try:
    from osgeo import gdal
    from osgeo import osr
except ImportError:
    raise ImportError("GDAL library is required. Install with: pip install gdal")

try:
    from sklearn.decomposition import PCA
    from scipy import ndimage
    from scipy.signal import savgol_filter
except ImportError:
    raise ImportError("Scientific libraries are required. Install with: pip install scikit-learn scipy")

from ..core.config import config
from ..utils.logger import setup_logger


class HyperspectralProcessor:
    """
    Класс для обработки гиперспектральных данных
    Научно-ориентированная реализация с современными методами обработки
    """
    
    def __init__(self):
        """Инициализация процессора гиперспектральных данных"""
        self.logger = setup_logger('HyperspectralProcessor')
        self.supported_formats = ['.bil', '.hdr', '.dat', '.tif', '.tiff', '.img']
        
        # Научные параметры обработки
        self.denoising_methods = ['pca', 'mnf', 'wavelet', 'savgol']
        self.correction_methods = ['dark_current', 'empirical_line', 'flat_field']
        
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
            self.logger.info(f"Начало научной обработки гиперспектральных данных: {input_path}")
            
            # Проверка входного файла
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Входной файл не найден: {input_path}")
            
            # Чтение данных
            self.logger.info("Чтение гиперспектральных данных")
            dataset, image_data, wavelengths = self._read_hyperspectral_data(input_path)
            
            # Предварительный анализ данных
            self.logger.info("Предварительный анализ данных")
            data_quality = self._analyze_data_quality(image_data)
            
            # Радиометрическая коррекция
            self.logger.info("Радиометрическая коррекция")
            corrected_data = self._radiometric_correction(image_data, method='empirical_line')
            
            # Атмосферная коррекция (упрощенная)
            self.logger.info("Атмосферная коррекция")
            atmospheric_corrected = self._atmospheric_correction(corrected_data)
            
            # Шумоподавление с использованием нескольких методов
            self.logger.info("Шумоподавление")
            denoised_data = self._advanced_noise_reduction(atmospheric_corrected)
            
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
            
        except Exception as e:
            self.logger.error(f"Ошибка научной обработки гиперспектральных данных: {e}")
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
                raise ValueError(f"Не удалось открыть файл: {input_path}")
            
            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            bands = dataset.RasterCount
            
            self.logger.info(f"Размер изображения: {rows}x{cols}, каналов: {bands}")
            
            # Извлечение всех каналов
            image_data = np.zeros((rows, cols, bands), dtype=np.float32)
            
            for band in range(1, bands + 1):
                band_data = dataset.GetRasterBand(band)
                image_data[:, :, band-1] = band_data.ReadAsArray().astype(np.float32)
            
            # Извлечение длин волн из метаданных
            wavelengths = self._extract_wavelengths(dataset)
            
            return dataset, image_data, wavelengths
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения данных: {e}")
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
            rows, cols, bands = image_data.shape
            
            # Базовая статистика
            data_quality = {
                'total_pixels': rows * cols,
                'total_bands': bands,
                'data_range': {
                    'min': float(np.min(image_data)),
                    'max': float(np.max(image_data)),
                    'mean': float(np.mean(image_data)),
                    'std': float(np.std(image_data))
                }
            }
            
            # Анализ пропущенных значений
            nan_count = np.sum(np.isnan(image_data))
            inf_count = np.sum(np.isinf(image_data))
            
            data_quality['missing_values'] = {
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
                'nan_percentage': float(nan_count / image_data.size * 100),
                'inf_percentage': float(inf_count / image_data.size * 100)
            }
            
            # Анализ динамического диапазона по каналам
            band_ranges = []
            for band in range(bands):
                band_data = image_data[:, :, band]
                valid_data = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]
                
                if len(valid_data) > 0:
                    band_ranges.append({
                        'band': band + 1,
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'dynamic_range': float(np.max(valid_data) - np.min(valid_data)),
                        'snr': self._calculate_snr(valid_data)
                    })
            
            data_quality['band_quality'] = band_ranges
            
            # Общая оценка качества
            avg_snr = np.mean([b['snr'] for b in band_ranges if b['snr'] > 0])
            data_quality['overall_quality'] = {
                'average_snr': float(avg_snr) if avg_snr > 0 else 0,
                'quality_score': self._calculate_quality_score(data_quality)
            }
            
            return data_quality
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа качества данных: {e}")
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
            signal = np.mean(data)
            noise = np.std(data)
            
            if noise == 0:
                return float('inf')
            
            return signal / noise
            
        except Exception:
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
        corrected = corrected / (white_reference - dark_reference + 1e-8)
        
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
            corrected = image_data / (reference_spectrum + 1e-8)
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
            
            # Нормализация
            max_values = np.percentile(corrected, 98, axis=(0, 1))
            corrected = corrected / (max_values + 1e-8)
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
        Шумоподавление с помощью фильтра Савицкого-Голея
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Данные после шумоподавления
        """
        try:
            rows, cols, bands = image_data.shape
            denoised_image = np.zeros_like(image_data)
            
            for band in range(bands):
                band_data = image_data[:, :, band]
                
                # Применение фильтра Савицкого-Голея к каждой строке и столбцу
                filtered_rows = np.zeros_like(band_data)
                for i in range(rows):
                    filtered_rows[i, :] = savgol_filter(band_data[i, :], 
                                                       window_length=min(11, cols), 
                                                       polyorder=3)
                
                filtered_both = np.zeros_like(band_data)
                for j in range(cols):
                    filtered_both[:, j] = savgol_filter(filtered_rows[:, j], 
                                                       window_length=min(11, rows), 
                                                       polyorder=3)
                
                denoised_image[:, :, band] = filtered_both
            
            self.logger.info("Шумоподавление Савицкого-Голея завершено")
            return denoised_image
            
        except Exception as e:
            self.logger.error(f"Ошибка шумоподавления Савицкого-Голея: {e}")
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
        Спектральный ресемплинг
        
        Args:
            image_data: Входные данные изображения
            wavelengths: Длины волн
            
        Returns:
            Ресемплированные данные
        """
        try:
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
                    # Усреднение каналов в диапазоне
                    range_data = np.mean(image_data[:, :, mask], axis=2)
                    new_bands.append(range_data)
                    band_names.append(range_name)
            
            if new_bands:
                calibrated_data = np.stack(new_bands, axis=2)
                self.logger.info(f"Спектральный ресемплинг: {image_data.shape[2]} -> {len(new_bands)} каналов")
                return calibrated_data
            else:
                return image_data
                
        except Exception as e:
            self.logger.error(f"Ошибка спектрального ресемплинга: {e}")
            return image_data
    
    def _spectral_smoothing(self, image_data: np.ndarray) -> np.ndarray:
        """
        Спектральная сглаживающая фильтрация
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Сглаженные данные
        """
        try:
            if image_data.shape[2] < 3:
                return image_data
            
            rows, cols, bands = image_data.shape
            smoothed_data = np.zeros_like(image_data)
            
            for i in range(rows):
                for j in range(cols):
                    # Применение сглаживания к спектральному профилю
                    spectrum = image_data[i, j, :]
                    smoothed_spectrum = savgol_filter(spectrum, 
                                                    window_length=min(5, bands), 
                                                    polyorder=2)
                    smoothed_data[i, j, :] = smoothed_spectrum
            
            return smoothed_data
            
        except Exception as e:
            self.logger.error(f"Ошибка спектрального сглаживания: {e}")
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
            rows, cols, bands = image_data.shape
            tiff_paths = []
            
            # Создание выходной директории
            tiff_dir = os.path.join(output_dir, 'tiff_bands')
            os.makedirs(tiff_dir, exist_ok=True)
            
            # Копирование геопривязки
            geo_transform = reference_dataset.GetGeoTransform()
            projection = reference_dataset.GetProjection()
            
            # Сохранение каждого канала в отдельный TIFF файл
            for band in range(bands):
                output_path = os.path.join(tiff_dir, f'band_{band+1:03d}.tif')
                
                # Создание выходного файла
                driver = gdal.GetDriverByName('GTiff')
                output = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
                
                # Копирование геопривязки
                if geo_transform:
                    output.SetGeoTransform(geo_transform)
                if projection:
                    output.SetProjection(projection)
                
                # Запись канала
                output_band = output.GetRasterBand(1)
                output_band.WriteArray(image_data[:, :, band])
                
                # Очистка
                output = None
                
                tiff_paths.append(output_path)
            
            self.logger.info(f"Создано {len(tiff_paths)} TIFF файлов")
            return tiff_paths
            
        except Exception as e:
            self.logger.error(f"Ошибка конвертации в TIFF: {e}")
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
            dataset = gdal.Open(input_path)
            if dataset is None:
                raise ValueError(f"Не удалось открыть файл: {input_path}")
            
            band_info = {
                'total_bands': dataset.RasterCount,
                'bands': []
            }
            
            for band in range(1, dataset.RasterCount + 1):
                band_data = dataset.GetRasterBand(band)
                stats = band_data.GetStatistics(True, True)
                
                band_info['bands'].append({
                    'band_number': band,
                    'min': stats[0],
                    'max': stats[1],
                    'mean': stats[2],
                    'stddev': stats[3],
                    'no_data_value': band_data.GetNoDataValue()
                })
            
            return band_info
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о каналах: {e}")
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
            if len(tiff_paths) < max(rgb_bands):
                raise ValueError(f"Недостаточно каналов для RGB композита. Требуется: {max(rgb_bands)}")
            
            # Чтение RGB каналов
            r_band = gdal.Open(tiff_paths[rgb_bands[0]-1]).ReadAsArray()
            g_band = gdal.Open(tiff_paths[rgb_bands[1]-1]).ReadAsArray()
            b_band = gdal.Open(tiff_paths[rgb_bands[2]-1]).ReadAsArray()
            
            # Нормализация с процентильным растяжением
            def normalize_band(band, lower_percent=2, upper_percent=98):
                band_min, band_max = np.percentile(band, [lower_percent, upper_percent])
                if band_max > band_min:
                    normalized = (band - band_min) / (band_max - band_min)
                    return np.clip(normalized, 0, 1)
                return band
            
            r_norm = normalize_band(r_band)
            g_norm = normalize_band(g_band)
            b_norm = normalize_band(b_band)
            
            # Создание RGB композита
            rgb_composite = np.stack([r_norm, g_norm, b_norm], axis=2)
            
            # Сохранение
            if output_path is None:
                output_path = os.path.join(os.path.dirname(tiff_paths[0]), 'rgb_composite.tif')
            
            rows, cols = rgb_composite.shape[:2]
            driver = gdal.GetDriverByName('GTiff')
            output = driver.Create(output_path, cols, rows, 3, gdal.GDT_Float32)
            
            # Копирование геопривязки из первого канала
            reference_dataset = gdal.Open(tiff_paths[0])
            output.SetGeoTransform(reference_dataset.GetGeoTransform())
            output.SetProjection(reference_dataset.GetProjection())
            
            # Запись каналов
            for i, band_data in enumerate([r_norm, g_norm, b_norm]):
                output_band = output.GetRasterBand(i + 1)
                output_band.WriteArray(band_data)
            
            output = None
            
            self.logger.info(f"RGB композит сохранен: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка создания RGB композита: {e}")
            raise