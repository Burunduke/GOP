"""
Калькулятор вегетационных индексов
"""

import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    from osgeo import gdal
except ImportError:
    raise ImportError("GDAL library is required. Install with: pip install gdal")

from .definitions import IndexDefinitions
from ..core.config import config
from ..utils.logger import setup_logger


class VegetationIndexCalculator:
    """
    Класс для расчета вегетационных индексов
    """
    
    def __init__(self):
        """Инициализация калькулятора индексов"""
        self.logger = setup_logger('VegetationIndexCalculator')
        self.definitions = IndexDefinitions()
        
    def calculate(self, 
                 orthophoto_path: str,
                 segmentation_mask: str,
                 sensor_type: str = 'Hyperspectral',
                 selected_indices: Optional[List[str]] = None,
                 output_dir: str = 'results') -> Dict[str, Any]:
        """
        Расчет вегетационных индексов
        
        Args:
            orthophoto_path: Путь к ортофотоплану
            segmentation_mask: Путь к маске сегментации
            sensor_type: Тип сенсора ('RGB', 'Multispectral', 'Hyperspectral')
            selected_indices: Список индексов для расчета
            output_dir: Директория для сохранения результатов
            
        Returns:
            Словарь с результатами расчета индексов
        """
        try:
            self.logger.info(f"Начало расчета вегетационных индексов для сенсора: {sensor_type}")
            
            # Проверка входных файлов
            if not os.path.exists(orthophoto_path):
                raise FileNotFoundError(f"Ортофотоплан не найден: {orthophoto_path}")
            
            if not os.path.exists(segmentation_mask):
                raise FileNotFoundError(f"Маска сегментации не найдена: {segmentation_mask}")
            
            # Определение доступных индексов
            if selected_indices is None:
                selected_indices = config.get('indices.default_indices', [])
            
            available_indices = self.definitions.get_available_indices(sensor_type)
            indices_to_calculate = [idx for idx in selected_indices if idx in available_indices]
            
            if not indices_to_calculate:
                raise ValueError(f"Нет доступных индексов для сенсора: {sensor_type}")
            
            self.logger.info(f"Расчет индексов: {indices_to_calculate}")
            
            # Чтение данных
            image_data = self._read_image_data(orthophoto_path, sensor_type)
            mask_data = self._read_mask_data(segmentation_mask)
            
            # Извлечение спектральных каналов
            bands = self._extract_bands(image_data, sensor_type)
            
            # Расчет индексов
            indices_results = {}
            normalized_indices = {}
            
            for index_name in indices_to_calculate:
                self.logger.info(f"Расчет индекса: {index_name}")
                
                # Расчет значений индекса
                index_values = self.definitions.calculate_index(index_name, bands)
                
                # Нормализация значений
                normalized_values = self.definitions.normalize_index(index_name, index_values, mask_data)
                
                indices_results[index_name] = index_values
                normalized_indices[index_name] = normalized_values
                
                # Сохранение индекса
                self._save_index(index_values, index_name, output_dir, orthophoto_path)
                self._save_index(normalized_values, f"{index_name}_normalized", output_dir, orthophoto_path)
            
            # Комплексная оценка состояния растений
            plant_condition = self._calculate_plant_condition(normalized_indices, mask_data)
            
            # Сохранение комплексной оценки
            self._save_plant_condition(plant_condition, output_dir, orthophoto_path)
            
            results = {
                'sensor_type': sensor_type,
                'calculated_indices': indices_to_calculate,
                'indices_values': indices_results,
                'normalized_indices': normalized_indices,
                'plant_condition': plant_condition,
                'output_dir': output_dir
            }
            
            self.logger.info("Расчет вегетационных индексов завершен")
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета вегетационных индексов: {e}")
            raise
    
    def _read_image_data(self, image_path: str, sensor_type: str) -> np.ndarray:
        """
        Чтение данных изображения
        
        Args:
            image_path: Путь к изображению
            sensor_type: Тип сенсора
            
        Returns:
            Массив данных изображения
        """
        try:
            dataset = gdal.Open(image_path)
            if dataset is None:
                raise ValueError(f"Не удалось открыть изображение: {image_path}")
            
            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            bands = dataset.RasterCount
            
            # Чтение всех каналов
            image_data = np.zeros((rows, cols, bands), dtype=np.float32)
            
            for band in range(1, bands + 1):
                band_data = dataset.GetRasterBand(band)
                image_data[:, :, band-1] = band_data.ReadAsArray().astype(np.float32)
            
            self.logger.info(f"Изображение загружено: {rows}x{cols}x{bands}")
            return image_data
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения изображения: {e}")
            raise
    
    def _read_mask_data(self, mask_path: str) -> np.ndarray:
        """
        Чтение данных маски
        
        Args:
            mask_path: Путь к маске
            
        Returns:
            Массив данных маски
        """
        try:
            dataset = gdal.Open(mask_path)
            if dataset is None:
                raise ValueError(f"Не удалось открыть маску: {mask_path}")
            
            mask_data = dataset.GetRasterBand(1).ReadAsArray()
            
            # Бинаризация маски
            mask_data = (mask_data > 0).astype(np.uint8)
            
            self.logger.info(f"Маска загружена: {mask_data.shape}, пикселей области: {mask_data.sum()}")
            return mask_data
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения маски: {e}")
            raise
    
    def _extract_bands(self, image_data: np.ndarray, sensor_type: str) -> Dict[str, np.ndarray]:
        """
        Извлечение спектральных каналов
        
        Args:
            image_data: Данные изображения
            sensor_type: Тип сенсора
            
        Returns:
            Словарь с спектральными каналами
        """
        bands = {}
        
        if sensor_type == 'RGB':
            # RGB: 3 канала (B, G, R)
            if image_data.shape[2] >= 3:
                bands['Blue'] = image_data[:, :, 0]
                bands['Green'] = image_data[:, :, 1]
                bands['Red'] = image_data[:, :, 2]
        
        elif sensor_type == 'Multispectral':
            # Мультиспектральный: 5 каналов
            if image_data.shape[2] >= 5:
                bands['Blue'] = image_data[:, :, 0]
                bands['Green'] = image_data[:, :, 1]
                bands['Red'] = image_data[:, :, 2]
                bands['RedEdge'] = image_data[:, :, 3]
                bands['NIR'] = image_data[:, :, 4]
        
        elif sensor_type == 'Hyperspectral':
            # Гиперспектральный: выбор каналов по длинам волн
            if image_data.shape[2] >= 100:
                # Приблизительные индексы каналов для типичного гиперспектрального сенсора
                bands['Blue'] = image_data[:, :, 10]    # ~450 нм
                bands['Green'] = image_data[:, :, 20]   # ~550 нм
                bands['Red'] = image_data[:, :, 30]     # ~650 нм
                bands['RedEdge'] = image_data[:, :, 35] # ~720 нм
                bands['NIR'] = image_data[:, :, 50]     # ~800 нм
                bands['SWIR'] = image_data[:, :, 80]    # ~1600 нм
        
        # Проверка наличия необходимых каналов
        missing_bands = [name for name in bands.keys() if bands[name] is None]
        if missing_bands:
            raise ValueError(f"Отсутствуют каналы: {missing_bands}")
        
        self.logger.info(f"Извлечено каналов: {list(bands.keys())}")
        return bands
    
    def _save_index(self, 
                   index_data: np.ndarray, 
                   index_name: str, 
                   output_dir: str, 
                   reference_path: str) -> None:
        """
        Сохранение индекса в файл
        
        Args:
            index_data: Данные индекса
            index_name: Название индекса
            output_dir: Директория для сохранения
            reference_path: Путь к референсному изображению
        """
        try:
            # Создание директории для индексов
            indices_dir = os.path.join(output_dir, 'indices')
            os.makedirs(indices_dir, exist_ok=True)
            
            output_path = os.path.join(indices_dir, f"{index_name}.tif")
            
            # Получение геопривязки из референсного изображения
            reference_dataset = gdal.Open(reference_path)
            geo_transform = reference_dataset.GetGeoTransform()
            projection = reference_dataset.GetProjection()
            
            # Создание выходного файла
            driver = gdal.GetDriverByName('GTiff')
            rows, cols = index_data.shape
            output = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
            
            # Копирование геопривязки
            if geo_transform:
                output.SetGeoTransform(geo_transform)
            if projection:
                output.SetProjection(projection)
            
            # Запись данных
            output_band = output.GetRasterBand(1)
            output_band.WriteArray(index_data)
            
            # Очистка
            output = None
            reference_dataset = None
            
            self.logger.debug(f"Индекс сохранен: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения индекса {index_name}: {e}")
            raise
    
    def _calculate_plant_condition(self, 
                                 normalized_indices: Dict[str, np.ndarray], 
                                 mask: np.ndarray) -> Dict[str, Any]:
        """
        Расчет комплексной оценки состояния растений
        
        Args:
            normalized_indices: Нормализованные индексы
            mask: Маска области
            
        Returns:
            Словарь с оценкой состояния растений
        """
        try:
            # Усреднение индексов по группам
            greenness_indices = ['GNDVI', 'MCARI', 'MNLI', 'OSAVI', 'TVI']
            stress_indices = ['SIPI2', 'mARI']
            water_indices = ['NDWI', 'MSI']
            
            # Расчет средних значений по группам
            greenness_values = []
            for idx in greenness_indices:
                if idx in normalized_indices:
                    values = normalized_indices[idx][mask > 0]
                    greenness_values.append(values)
            
            stress_values = []
            for idx in stress_indices:
                if idx in normalized_indices:
                    values = normalized_indices[idx][mask > 0]
                    stress_values.append(values)
            
            water_values = []
            for idx in water_indices:
                if idx in normalized_indices:
                    values = normalized_indices[idx][mask > 0]
                    water_values.append(values)
            
            # Расчет комплексных оценок
            condition_maps = {}
            
            if greenness_values:
                condition_maps['greenness'] = np.mean(greenness_values, axis=0)
            
            if stress_values:
                condition_maps['stress'] = np.mean(stress_values, axis=0)
            
            if water_values:
                condition_maps['water'] = np.mean(water_values, axis=0)
            
            # Общая оценка
            if condition_maps:
                overall_values = list(condition_maps.values())
                condition_maps['overall'] = np.mean(overall_values, axis=0)
            
            # Статистика
            statistics = {}
            for name, values in condition_maps.items():
                statistics[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            result = {
                'condition_maps': condition_maps,
                'statistics': statistics
            }
            
            self.logger.info("Комплексная оценка состояния растений рассчитана")
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета комплексной оценки: {e}")
            raise
    
    def _save_plant_condition(self, 
                            plant_condition: Dict[str, Any], 
                            output_dir: str, 
                            reference_path: str) -> None:
        """
        Сохранение комплексной оценки состояния растений
        
        Args:
            plant_condition: Результаты оценки
            output_dir: Директория для сохранения
            reference_path: Путь к референсному изображению
        """
        try:
            condition_maps = plant_condition.get('condition_maps', {})
            
            for name, data in condition_maps.items():
                output_path = os.path.join(output_dir, 'indices', f"plant_condition_{name}.tif")
                self._save_index(data, f"plant_condition_{name}", output_dir, reference_path)
            
            self.logger.info("Комплексная оценка состояния растений сохранена")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения комплексной оценки: {e}")
            raise
    
    def assess_plant_condition(self, indices_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оценка состояния растений на основе индексов
        
        Args:
            indices_results: Результаты расчета индексов
            
        Returns:
            Словарь с оценкой состояния
        """
        try:
            normalized_indices = indices_results.get('normalized_indices', {})
            
            if not normalized_indices:
                return {'error': 'Отсутствуют нормализованные индексы'}
            
            # Создание маски на основе первого индекса
            first_index = list(normalized_indices.values())[0]
            mask = (first_index > 0).astype(np.uint8)
            
            # Расчет комплексной оценки
            plant_condition = self._calculate_plant_condition(normalized_indices, mask)
            
            # Классификация состояния
            overall_stats = plant_condition['statistics'].get('overall', {})
            overall_mean = overall_stats.get('mean', 0)
            
            if overall_mean > 0.7:
                condition_class = 'Отличное'
                condition_color = 'green'
            elif overall_mean > 0.4:
                condition_class = 'Удовлетворительное'
                condition_color = 'yellow'
            else:
                condition_class = 'Плохое'
                condition_color = 'red'
            
            plant_condition['classification'] = {
                'class': condition_class,
                'color': condition_color,
                'score': overall_mean
            }
            
            return plant_condition
            
        except Exception as e:
            self.logger.error(f"Ошибка оценки состояния растений: {e}")
            return {'error': str(e)}
    
    def get_index_statistics(self, 
                           index_path: str, 
                           mask_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Получить статистику по индексу
        
        Args:
            index_path: Путь к файлу индекса
            mask_path: Путь к маске (опционально)
            
        Returns:
            Словарь со статистикой
        """
        try:
            # Чтение индекса
            dataset = gdal.Open(index_path)
            if dataset is None:
                raise ValueError(f"Не удалось открыть индекс: {index_path}")
            
            index_data = dataset.GetRasterBand(1).ReadAsArray()
            
            # Применение маски
            if mask_path and os.path.exists(mask_path):
                mask_dataset = gdal.Open(mask_path)
                mask_data = mask_dataset.GetRasterBand(1).ReadAsArray()
                index_data = index_data[mask_data > 0]
            
            # Расчет статистики
            statistics = {
                'count': int(np.count_nonzero(~np.isnan(index_data))),
                'mean': float(np.nanmean(index_data)),
                'std': float(np.nanstd(index_data)),
                'min': float(np.nanmin(index_data)),
                'max': float(np.nanmax(index_data)),
                'median': float(np.nanmedian(index_data)),
                'q25': float(np.nanpercentile(index_data, 25)),
                'q75': float(np.nanpercentile(index_data, 75))
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета статистики: {e}")
            return {'error': str(e)}