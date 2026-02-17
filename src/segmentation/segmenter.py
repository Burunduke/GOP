"""
Основной класс сегментации изображений
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from osgeo import gdal
except ImportError:
    raise ImportError("GDAL library is required. Install with: pip install gdal")

from ..core.config import config
from ..utils.logger import setup_logger


class ImageSegmenter:
    """
    Основной класс для сегментации изображений сверхвысокого разрешения
    """
    
    def __init__(self):
        """Инициализация сегментатора"""
        self.logger = setup_logger('ImageSegmenter')
        
        # Инициализация моделей (заглушки для совместимости)
        self.deeplab_segmenter = None
        self.cascade_segmenter = None
        
    def segment(self, 
                image_path: str, 
                output_dir: str = 'results',
                use_refinement: bool = True,
                compression_ratio: float = None) -> str:
        """
        Сегментация изображения с использованием каскадного подхода
        
        Args:
            image_path: Путь к изображению
            output_dir: Директория для сохранения результатов
            use_refinement: Использовать уточнение границ
            compression_ratio: Коэффициент сжатия для предварительной сегментации
            
        Returns:
            Путь к финальной маске сегментации
        """
        try:
            self.logger.info(f"Начало сегментации изображения: {image_path}")
            
            # Проверка входного файла
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Изображение не найдено: {image_path}")
            
            # Настройка параметров
            if compression_ratio is None:
                compression_ratio = config.get('processing.compression_ratio', 0.125)
            
            # Создание выходной директории
            os.makedirs(output_dir, exist_ok=True)
            
            # Этап 1: Предварительная сегментация с помощью DeepLabV3+
            self.logger.info("Этап 1: Предварительная сегментация DeepLabV3+")
            coarse_mask_path = self._preliminary_segmentation(
                image_path, output_dir, compression_ratio
            )
            
            if not use_refinement:
                self.logger.info("Уточнение границ отключено")
                return coarse_mask_path
            
            # Этап 2: Уточнение границ с помощью CascadePSP
            self.logger.info("Этап 2: Уточнение границ CascadePSP")
            refined_mask_path = self._refine_segmentation(
                image_path, coarse_mask_path, output_dir
            )
            
            # Этап 3: Выбор лучшей маски
            self.logger.info("Этап 3: Выбор оптимальной маски")
            final_mask_path = self._select_best_mask(
                image_path, coarse_mask_path, refined_mask_path, output_dir
            )
            
            self.logger.info(f"Сегментация завершена: {final_mask_path}")
            return final_mask_path
            
        except Exception as e:
            self.logger.error(f"Ошибка сегментации: {e}")
            raise
    
    def _preliminary_segmentation(self, 
                                 image_path: str, 
                                 output_dir: str, 
                                 compression_ratio: float) -> str:
        """
        Предварительная сегментация с помощью DeepLabV3+
        
        Args:
            image_path: Путь к изображению
            output_dir: Директория для сохранения
            compression_ratio: Коэффициент сжатия
            
        Returns:
            Путь к маске предварительной сегментации
        """
        try:
            # Чтение и сжатие изображения
            image_data, original_shape = self._read_and_compress_image(
                image_path, compression_ratio
            )
            
            # Упрощенная сегментация (заглушка)
            mask_data = self._simple_segmentation(image_data)
            
            # Масштабирование маски к исходному размеру
            mask_resized = self._resize_mask(mask_data, original_shape)
            
            # Сохранение маски
            output_path = os.path.join(output_dir, 'coarse_segmentation.tif')
            self._save_mask(mask_resized, output_path, image_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка предварительной сегментации: {e}")
            raise
    
    def _refine_segmentation(self, 
                           image_path: str, 
                           coarse_mask_path: str, 
                           output_dir: str) -> str:
        """
        Уточнение границ с помощью CascadePSP
        
        Args:
            image_path: Путь к исходному изображению
            coarse_mask_path: Путь к грубой маске
            output_dir: Директория для сохранения
            
        Returns:
            Путь к уточненной маске
        """
        try:
            # Чтение исходного изображения и маски
            image_data = self._read_image(image_path)
            mask_data = self._read_mask(coarse_mask_path)
            
            # Упрощенное уточнение (заглушка)
            refined_mask = self._simple_refinement(image_data, mask_data)
            
            # Сохранение уточненной маски
            output_path = os.path.join(output_dir, 'refined_segmentation.tif')
            self._save_mask(refined_mask, output_path, image_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка уточнения сегментации: {e}")
            raise
    
    def _select_best_mask(self, 
                         image_path: str,
                         coarse_mask_path: str,
                         refined_mask_path: str,
                         output_dir: str) -> str:
        """
        Выбор лучшей маски сегментации
        
        Args:
            image_path: Путь к исходному изображению
            coarse_mask_path: Путь к грубой маске
            refined_mask_path: Путь к уточненной маске
            output_dir: Директория для сохранения
            
        Returns:
            Путь к лучшей маске
        """
        try:
            # Оценка качества масок
            coarse_quality = self._evaluate_mask_quality(coarse_mask_path)
            refined_quality = self._evaluate_mask_quality(refined_mask_path)
            
            self.logger.info(f"Качество грубой маски: {coarse_quality:.3f}")
            self.logger.info(f"Качество уточненной маски: {refined_quality:.3f}")
            
            # Выбор лучшей маски
            if refined_quality > coarse_quality:
                best_mask_path = refined_mask_path
                self.logger.info("Выбрана уточненная маска")
            else:
                best_mask_path = coarse_mask_path
                self.logger.info("Выбрана грубая маска")
            
            # Копирование лучшей маски как финальной
            final_mask_path = os.path.join(output_dir, 'final_segmentation.tif')
            self._copy_file(best_mask_path, final_mask_path)
            
            return final_mask_path
            
        except Exception as e:
            self.logger.error(f"Ошибка выбора лучшей маски: {e}")
            # В случае ошибки возвращаем грубую маску
            return coarse_mask_path
    
    def _read_and_compress_image(self, 
                                image_path: str, 
                                compression_ratio: float) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Чтение и сжатие изображения
        
        Args:
            image_path: Путь к изображению
            compression_ratio: Коэффициент сжатия
            
        Returns:
            Кортеж (сжатое изображение, исходный размер)
        """
        try:
            dataset = gdal.Open(image_path)
            if dataset is None:
                raise ValueError(f"Не удалось открыть изображение: {image_path}")
            
            original_shape = (dataset.RasterYSize, dataset.RasterXSize)
            
            # Чтение данных
            bands = dataset.RasterCount
            image_data = np.zeros((original_shape[0], original_shape[1], bands), dtype=np.float32)
            
            for band in range(1, bands + 1):
                band_data = dataset.GetRasterBand(band)
                image_data[:, :, band-1] = band_data.ReadAsArray().astype(np.float32)
            
            # Сжатие изображения
            if compression_ratio < 1.0:
                new_height = int(original_shape[0] * compression_ratio)
                new_width = int(original_shape[1] * compression_ratio)
                
                # Простое сжатие (в реальном приложении можно использовать более сложные методы)
                compressed_image = np.zeros((new_height, new_width, bands), dtype=np.float32)
                
                for band in range(bands):
                    # Использование билинейной интерполяции
                    from skimage.transform import resize
                    compressed_image[:, :, band] = resize(
                        image_data[:, :, band], 
                        (new_height, new_width), 
                        preserve_range=True,
                        anti_aliasing=True
                    )
                
                return compressed_image, original_shape
            else:
                return image_data, original_shape
                
        except Exception as e:
            self.logger.error(f"Ошибка чтения и сжатия изображения: {e}")
            raise
    
    def _read_image(self, image_path: str) -> np.ndarray:
        """
        Чтение изображения
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Массив данных изображения
        """
        dataset = gdal.Open(image_path)
        if dataset is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")
        
        rows, cols = dataset.RasterYSize, dataset.RasterXSize
        bands = dataset.RasterCount
        
        image_data = np.zeros((rows, cols, bands), dtype=np.float32)
        
        for band in range(1, bands + 1):
            band_data = dataset.GetRasterBand(band)
            image_data[:, :, band-1] = band_data.ReadAsArray().astype(np.float32)
        
        return image_data
    
    def _read_mask(self, mask_path: str) -> np.ndarray:
        """
        Чтение маски
        
        Args:
            mask_path: Путь к маске
            
        Returns:
            Массив данных маски
        """
        dataset = gdal.Open(mask_path)
        if dataset is None:
            raise ValueError(f"Не удалось открыть маску: {mask_path}")
        
        mask_data = dataset.GetRasterBand(1).ReadAsArray()
        return mask_data.astype(np.uint8)
    
    def _resize_mask(self, mask_data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Масштабирование маски к целевому размеру
        
        Args:
            mask_data: Исходная маска
            target_shape: Целевой размер (height, width)
            
        Returns:
            Масштабированная маска
        """
        try:
            from skimage.transform import resize
            
            resized_mask = resize(
                mask_data, 
                target_shape, 
                preserve_range=True,
                anti_aliasing=False,
                order=0  # Ближайший сосед для масок
            )
            
            return resized_mask.astype(np.uint8)
            
        except ImportError:
            # Альтернативный метод без skimage
            import cv2
            resized_mask = cv2.resize(
                mask_data, 
                (target_shape[1], target_shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
            return resized_mask
    
    def _save_mask(self, 
                  mask_data: np.ndarray, 
                  output_path: str, 
                  reference_path: str) -> None:
        """
        Сохранение маски
        
        Args:
            mask_data: Данные маски
            output_path: Путь для сохранения
            reference_path: Путь к референсному изображению
        """
        try:
            # Получение геопривязки из референсного изображения
            reference_dataset = gdal.Open(reference_path)
            geo_transform = reference_dataset.GetGeoTransform()
            projection = reference_dataset.GetProjection()
            
            # Создание выходного файла
            driver = gdal.GetDriverByName('GTiff')
            rows, cols = mask_data.shape
            output = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte)
            
            # Копирование геопривязки
            if geo_transform:
                output.SetGeoTransform(geo_transform)
            if projection:
                output.SetProjection(projection)
            
            # Запись данных
            output_band = output.GetRasterBand(1)
            output_band.WriteArray(mask_data)
            
            # Очистка
            output = None
            reference_dataset = None
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения маски: {e}")
            raise
    
    def _evaluate_mask_quality(self, mask_path: str) -> float:
        """
        Оценка качества маски
        
        Args:
            mask_path: Путь к маске
            
        Returns:
            Оценка качества (0-1)
        """
        try:
            mask_data = self._read_mask(mask_path)
            
            # Простые метрики качества
            # 1. Отношение площади сегментированной области к общей площади
            area_ratio = np.sum(mask_data > 0) / mask_data.size
            
            # 2. Компактность (отношение площади к периметру в квадрате)
            from skimage.measure import regionprops
            regions = regionprops(mask_data)
            
            if regions:
                compactness = regions[0].area / (regions[0].perimeter ** 2 + 1e-8)
            else:
                compactness = 0
            
            # Комбинированная оценка
            quality = 0.7 * area_ratio + 0.3 * compactness
            
            return np.clip(quality, 0, 1)
            
        except Exception as e:
            self.logger.warning(f"Ошибка оценки качества маски: {e}")
            return 0.5  # Средняя оценка по умолчанию
    
    def _copy_file(self, src: str, dst: str) -> None:
        """
        Копирование файла
        
        Args:
            src: Исходный путь
            dst: Целевой путь
        """
        import shutil
        shutil.copy2(src, dst)
    
    def segment_batch(self, 
                     image_paths: list, 
                     output_dir: str = 'results',
                     **kwargs) -> list:
        """
        Пакетная сегментация изображений
        
        Args:
            image_paths: Список путей к изображениям
            output_dir: Директория для сохранения результатов
            **kwargs: Дополнительные параметры
            
        Returns:
            Список путей к результатам
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                self.logger.info(f"Обработка изображения {i+1}/{len(image_paths)}: {image_path}")
                
                # Создание индивидуальной директории для каждого изображения
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                image_output_dir = os.path.join(output_dir, image_name)
                
                # Сегментация
                mask_path = self.segment(image_path, image_output_dir, **kwargs)
                results.append(mask_path)
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки изображения {image_path}: {e}")
                results.append(None)
        
        return results
    
    def _simple_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        """
        Упрощенная сегментация на основе порогового значения
        """
        try:
            # Использование NDVI-подобного подхода для сегментации растительности
            if image_data.shape[2] >= 3:
                # Предполагаем, что каналы: Red, Green, NIR
                red = image_data[:, :, 0]
                nir = image_data[:, :, 2] if image_data.shape[2] >= 3 else image_data[:, :, 1]
                
                # Расчет NDVI-подобного индекса
                ndvi_like = (nir - red) / (nir + red + 1e-8)
                
                # Пороговая сегментация
                mask = (ndvi_like > 0.2).astype(np.uint8)
                return mask
            else:
                # Для RGB изображений используем зеленый канал
                green = image_data[:, :, 1]
                mask = (green > np.mean(green)).astype(np.uint8)
                return mask
                
        except Exception as e:
            self.logger.warning(f"Ошибка упрощенной сегментации: {e}")
            # Возвращаем маску по умолчанию
            return np.ones(image_data.shape[:2], dtype=np.uint8)
    
    def _simple_refinement(self, image_data: np.ndarray, mask_data: np.ndarray) -> np.ndarray:
        """
        Упрощенное уточнение границ сегментации
        """
        try:
            import cv2
            
            # Морфологические операции для уточнения границ
            kernel = np.ones((3, 3), np.uint8)
            
            # Закрытие для заполнения мелких отверстий
            closed = cv2.morphologyEx(mask_data, cv2.MORPH_CLOSE, kernel)
            
            # Открытие для удаления мелких объектов
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            # Гауссово размытие для сглаживания границ
            smoothed = cv2.GaussianBlur(opened.astype(np.float32), (5, 5), 0)
            
            # Бинаризация обратно
            refined = (smoothed > 0.5).astype(np.uint8)
            
            return refined
            
        except ImportError:
            # Если OpenCV недоступен, возвращаем исходную маску
            self.logger.warning("OpenCV не доступен, уточнение не выполнено")
            return mask_data
        except Exception as e:
            self.logger.warning(f"Ошибка упрощенного уточнения: {e}")
            return mask_data