"""
Модуль шумоподавления для гиперспектральных данных
"""

import numpy as np
import logging
from typing import Any, Dict, Optional

try:
    from sklearn.decomposition import PCA
    from scipy.signal import savgol_filter
except ImportError:
    raise ImportError("Scientific libraries are required. Install with: pip install scikit-learn scipy")


class HyperspectralDenoising:
    """Класс для шумоподавления гиперспектральных данных"""
    
    def __init__(self, logger: logging.Logger = None):
        """
        Инициализация класса шумоподавления
        
        Args:
            logger: Логгер для записи сообщений
        """
        self.logger = logger or logging.getLogger(__name__)
        self.denoising_methods = ['pca', 'mnf', 'wavelet', 'savgol']
    
    def advanced_noise_reduction(self, 
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
            # Валидация входных данных
            if image_data is None or image_data.size == 0:
                raise ValueError("Входные данные изображения пусты")
            
            if method not in self.denoising_methods:
                self.logger.warning(f"Неизвестный метод шумоподавления: {method}. Доступные методы: {self.denoising_methods}")
                return image_data
            
            if method == 'pca':
                return self.pca_denoising(image_data)
            elif method == 'mnf':
                return self.mnf_denoising(image_data)
            elif method == 'wavelet':
                return self.wavelet_denoising(image_data)
            elif method == 'savgol':
                return self.savgol_denoising(image_data)
            else:
                self.logger.warning(f"Неизвестный метод шумоподавления: {method}")
                return image_data
                
        except ValueError as e:
            self.logger.error(f"Ошибка валидации при шумоподавлении: {e}")
            return image_data
        except (RuntimeError, MemoryError) as e:
            self.logger.error(f"Ошибка выполнения при шумоподавлении: {e}")
            return image_data
        except ImportError as e:
            self.logger.error(f"Ошибка импорта модуля для шумоподавления: {e}")
            return image_data
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка шумоподавления: {e}")
            return image_data
    
    def pca_denoising(self, image_data: np.ndarray, n_components: float = 0.95) -> np.ndarray:
        """
        Шумоподавление с помощью PCA
        
        Args:
            image_data: Входные данные изображения
            n_components: Количество компонентов или доля объясненной дисперсии
            
        Returns:
            Данные после шумоподавления
        """
        try:
            # Валидация входных данных
            if image_data is None or image_data.size == 0:
                raise ValueError("Входные данные изображения пусты")
            
            if len(image_data.shape) != 3:
                raise ValueError(f"Ожидается 3D массив, получен {len(image_data.shape)}D")
            
            if not (0 < n_components <= 1) and not isinstance(n_components, int):
                raise ValueError("n_components должен быть в диапазоне (0, 1] или целым числом")
            
            rows, cols, bands = image_data.shape
            
            # Изменение формы данных для PCA
            reshaped = image_data.reshape(-1, bands)
            
            # Удаление NaN и бесконечных значений
            valid_mask = ~np.isnan(reshaped).any(axis=1) & ~np.isinf(reshaped).any(axis=1)
            valid_data = reshaped[valid_mask]
            
            if len(valid_data) < 100:
                self.logger.warning("Недостаточно валидных данных для PCA")
                return image_data
            
            # Применение PCA с обработкой ошибок
            try:
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
                
            except ValueError as e:
                self.logger.error(f"Ошибка в PCA алгоритме: {e}")
                return image_data
            
        except ValueError as e:
            self.logger.error(f"Ошибка валидации в PCA шумоподавлении: {e}")
            return image_data
        except (RuntimeError, MemoryError) as e:
            self.logger.error(f"Ошибка выполнения в PCA шумоподавлении: {e}")
            return image_data
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка PCA шумоподавления: {e}")
            return image_data
    
    def mnf_denoising(self, image_data: np.ndarray) -> np.ndarray:
        """
        Шумоподавление с помощью MNF (Minimum Noise Fraction)
        
        Args:
            image_data: Входные данные изображения
            
        Returns:
            Данные после шумоподавления
        """
        try:
            # Валидация входных данных
            if image_data is None or image_data.size == 0:
                raise ValueError("Входные данные изображения пусты")
            
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
                return self.pca_denoising(image_data)
                
        except Exception as e:
            self.logger.error(f"Ошибка MNF шумоподавления: {e}")
            return image_data
    
    def wavelet_denoising(self, image_data: np.ndarray) -> np.ndarray:
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
                
                # Проверка на валидные данные
                if np.all(np.isnan(band_data)) or np.all(np.isinf(band_data)):
                    denoised_image[:, :, band] = band_data
                    continue
                
                try:
                    # Вейвлет-преобразование
                    coeffs = pywt.wavedec2(band_data, 'db4', level=2)
                    
                    # Пороговая обработка коэффициентов
                    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(band_data.size))
                    coeffs_thresh = list(coeffs)
                    coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode='soft') 
                                       for detail in coeffs_thresh[1:]]
                    
                    # Обратное вейвлет-преобразование
                    denoised_image[:, :, band] = pywt.waverec2(coeffs_thresh, 'db4')
                except Exception as e:
                    self.logger.warning(f"Ошибка вейвлет-обработки канала {band}: {e}")
                    denoised_image[:, :, band] = band_data
            
            self.logger.info("Вейвлет-шумоподавление завершено")
            return denoised_image
            
        except ImportError:
            self.logger.warning("PyWavelets не установлен, используем PCA")
            return self.pca_denoising(image_data)
        except Exception as e:
            self.logger.error(f"Ошибка вейвлет-шумоподавления: {e}")
            return image_data
    
    def savgol_denoising(self, image_data: np.ndarray) -> np.ndarray:
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
    
    def calculate_denoising_statistics(self, 
                                     original_data: np.ndarray, 
                                     denoised_data: np.ndarray) -> Dict[str, Any]:
        """
        Расчет статистики шумоподавления
        
        Args:
            original_data: Исходные данные
            denoised_data: Данные после шумоподавления
            
        Returns:
            Словарь со статистикой шумоподавления
        """
        try:
            if original_data is None or denoised_data is None:
                raise ValueError("Входные данные не могут быть None")
            
            if original_data.shape != denoised_data.shape:
                raise ValueError("Формы исходных и обработанных данных должны совпадать")
            
            # Удаление NaN и Inf значений
            valid_mask = ~np.isnan(original_data) & ~np.isinf(original_data) & \
                        ~np.isnan(denoised_data) & ~np.isinf(denoised_data)
            
            if not np.any(valid_mask):
                return {'error': 'Нет валидных данных для анализа'}
            
            orig_valid = original_data[valid_mask]
            denoised_valid = denoised_data[valid_mask]
            
            # Расчет SNR до и после
            def calculate_snr(data):
                signal = np.mean(data)
                noise = np.std(data)
                if noise == 0 or np.isclose(noise, 0):
                    return float('inf') if signal != 0 else 0.0
                return signal / noise
            
            original_snr = calculate_snr(orig_valid)
            denoised_snr = calculate_snr(denoised_valid)
            
            statistics = {
                'original_snr': float(original_snr),
                'denoised_snr': float(denoised_snr),
                'snr_improvement': float(denoised_snr - original_snr),
                'snr_improvement_factor': float(denoised_snr / original_snr) if original_snr > 0 else 0,
                'original_noise': float(np.std(orig_valid)),
                'denoised_noise': float(np.std(denoised_valid)),
                'noise_reduction': float(np.std(orig_valid) - np.std(denoised_valid)),
                'noise_reduction_percentage': float((1 - np.std(denoised_valid) / np.std(orig_valid)) * 100) if np.std(orig_valid) > 0 else 0
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета статистики шумоподавления: {e}")
            return {'error': str(e)}