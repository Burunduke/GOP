"""
Калькулятор вегетационных индексов
"""

import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from numpy.typing import NDArray

# Type aliases for better type safety
IndexResult = Dict[str, Union[str, NDArray[np.float32], Dict[str, Any]]]
BandData = NDArray[np.float32]
IndexData = NDArray[np.float32]

try:
    from osgeo import gdal

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    # Don't raise error here to allow tests to run

from .definitions import IndexDefinitions
from ..core.config import Config, config, get_config, create_config
from ..utils.logger import setup_logger
from ..utils.gdal_utils import open_gdal_dataset
from ..utils.exceptions import ProcessingError, ValidationError


class VegetationIndexCalculator:
    """
    Класс для расчета вегетационных индексов
    """

    def __init__(self, config_instance: Optional[Config] = None):
        """
        Инициализация калькулятора индексов

        Args:
            config_instance: Опциональный экземпляр конфигурации для dependency injection
        """
        self.config = get_config(config_instance)
        self.logger = setup_logger("VegetationIndexCalculator")
        self.definitions = IndexDefinitions()

    def calculate(
        self,
        orthophoto_path: str,
        segmentation_mask: str,
        sensor_type: str = "Hyperspectral",
        selected_indices: Optional[List[str]] = None,
        output_dir: str = "results",
    ) -> IndexResult:
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
            self.logger.info(
                f"Начало расчета вегетационных индексов для сенсора: {sensor_type}"
            )

            # Проверка входных файлов
            if not os.path.exists(orthophoto_path):
                raise FileNotFoundError(f"Ортофотоплан не найден: {orthophoto_path}")

            if not os.path.exists(segmentation_mask):
                raise FileNotFoundError(
                    f"Маска сегментации не найдена: {segmentation_mask}"
                )

            # Определение доступных индексов
            if selected_indices is None:
                selected_indices = self.config.get("indices.default_indices", [])

            available_indices = self.definitions.get_available_indices(sensor_type)
            indices_to_calculate = [
                idx for idx in selected_indices if idx in available_indices
            ]

            if not indices_to_calculate:
                raise ValidationError(
                    f"Нет доступных индексов для сенсора: {sensor_type}",
                    details={"sensor_type": sensor_type},
                )

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
                normalized_values = self.definitions.normalize_index(
                    index_name, index_values, mask_data
                )

                indices_results[index_name] = index_values
                normalized_indices[index_name] = normalized_values

                # Сохранение индекса
                self._save_index(index_values, index_name, output_dir, orthophoto_path)
                self._save_index(
                    normalized_values,
                    f"{index_name}_normalized",
                    output_dir,
                    orthophoto_path,
                )

            # Комплексная оценка состояния растений
            plant_condition = self._calculate_plant_condition(
                normalized_indices, mask_data
            )

            # Сохранение комплексной оценки
            self._save_plant_condition(plant_condition, output_dir, orthophoto_path)

            results = {
                "sensor_type": sensor_type,
                "calculated_indices": indices_to_calculate,
                "indices_values": indices_results,
                "normalized_indices": normalized_indices,
                "plant_condition": plant_condition,
                "output_dir": output_dir,
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
            from ..utils.gdal_utils import read_raster_bands

            # Чтение всех каналов с использованием централизованной утилиты
            image_data = read_raster_bands(image_path)

            self.logger.info(f"Изображение загружено: {image_data.shape}")
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
            from ..utils.gdal_utils import read_raster_band

            # Чтение первого канала с использованием централизованной утилиты
            mask_data = read_raster_band(mask_path, band_number=1)

            # Бинаризация маски
            mask_data = (mask_data > 0).astype(np.uint8)

            self.logger.info(
                f"Маска загружена: {mask_data.shape}, пикселей области: {mask_data.sum()}"
            )
            return mask_data

        except Exception as e:
            self.logger.error(f"Ошибка чтения маски: {e}")
            raise

    def _extract_bands(
        self, image_data: np.ndarray, sensor_type: str
    ) -> Dict[str, np.ndarray]:
        """
        Извлечение спектральных каналов

        Args:
            image_data: Данные изображения
            sensor_type: Тип сенсора

        Returns:
            Словарь с спектральными каналами
        """
        bands = {}

        if sensor_type == "RGB":
            # RGB: 3 канала (B, G, R)
            if image_data.shape[2] >= 3:
                bands["Blue"] = image_data[:, :, 0]
                bands["Green"] = image_data[:, :, 1]
                bands["Red"] = image_data[:, :, 2]

        elif sensor_type == "Multispectral":
            # Мультиспектральный: 5 каналов
            if image_data.shape[2] >= 5:
                bands["Blue"] = image_data[:, :, 0]
                bands["Green"] = image_data[:, :, 1]
                bands["Red"] = image_data[:, :, 2]
                bands["RedEdge"] = image_data[:, :, 3]
                bands["NIR"] = image_data[:, :, 4]

        elif sensor_type == "Hyperspectral":
            # Гиперспектральный: выбор каналов по длинам волн
            if image_data.shape[2] >= 100:
                # Приблизительные индексы каналов для типичного гиперспектрального сенсора
                bands["Blue"] = image_data[:, :, 10]  # ~450 нм
                bands["Green"] = image_data[:, :, 20]  # ~550 нм
                bands["Red"] = image_data[:, :, 30]  # ~650 нм
                bands["RedEdge"] = image_data[:, :, 35]  # ~720 нм
                bands["NIR"] = image_data[:, :, 50]  # ~800 нм
                bands["SWIR"] = image_data[:, :, 80]  # ~1600 нм

        # Проверка наличия необходимых каналов
        missing_bands = [name for name in bands.keys() if bands[name] is None]
        if missing_bands:
            raise ValidationError(
                f"Отсутствуют каналы: {missing_bands}",
                details={"missing_bands": missing_bands},
            )

        self.logger.info(f"Извлечено каналов: {list(bands.keys())}")
        return bands

    def _save_index(
        self,
        index_data: np.ndarray,
        index_name: str,
        output_dir: str,
        reference_path: str,
    ) -> None:
        """
        Сохранение индекса в файл

        Args:
            index_data: Данные индекса
            index_name: Название индекса
            output_dir: Директория для сохранения
            reference_path: Путь к референсному изображению
        """
        try:
            from ..utils.gdal_utils import write_raster

            # Создание директории для индексов
            indices_dir = os.path.join(output_dir, "indices")
            os.makedirs(indices_dir, exist_ok=True)

            output_path = os.path.join(indices_dir, f"{index_name}.tif")

            # Сохранение с использованием централизованной утилиты
            write_raster(index_data, output_path, source_path=reference_path)

            self.logger.debug(f"Индекс сохранен: {output_path}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения индекса {index_name}: {e}")
            raise

    def _calculate_plant_condition(
        self, normalized_indices: Dict[str, np.ndarray], mask: np.ndarray
    ) -> Dict[str, Any]:
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
            greenness_indices = ["GNDVI", "MCARI", "MNLI", "OSAVI", "TVI"]
            stress_indices = ["SIPI2", "mARI"]
            water_indices = ["NDWI", "MSI"]

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
                condition_maps["greenness"] = np.mean(greenness_values, axis=0)

            if stress_values:
                condition_maps["stress"] = np.mean(stress_values, axis=0)

            if water_values:
                condition_maps["water"] = np.mean(water_values, axis=0)

            # Общая оценка
            if condition_maps:
                overall_values = list(condition_maps.values())
                condition_maps["overall"] = np.mean(overall_values, axis=0)

            # Статистика
            statistics = {}
            for name, values in condition_maps.items():
                statistics[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

            result = {"condition_maps": condition_maps, "statistics": statistics}

            self.logger.info("Комплексная оценка состояния растений рассчитана")
            return result

        except Exception as e:
            self.logger.error(f"Ошибка расчета комплексной оценки: {e}")
            raise

    def _save_plant_condition(
        self, plant_condition: Dict[str, Any], output_dir: str, reference_path: str
    ) -> None:
        """
        Сохранение комплексной оценки состояния растений

        Args:
            plant_condition: Результаты оценки
            output_dir: Директория для сохранения
            reference_path: Путь к референсному изображению
        """
        try:
            condition_maps = plant_condition.get("condition_maps", {})

            for name, data in condition_maps.items():
                output_path = os.path.join(
                    output_dir, "indices", f"plant_condition_{name}.tif"
                )
                self._save_index(
                    data, f"plant_condition_{name}", output_dir, reference_path
                )

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
            normalized_indices = indices_results.get("normalized_indices", {})

            if not normalized_indices:
                return {"error": "Отсутствуют нормализованные индексы"}

            # Создание маски на основе первого индекса
            first_index = list(normalized_indices.values())[0]
            mask = (first_index > 0).astype(np.uint8)

            # Расчет комплексной оценки
            plant_condition = self._calculate_plant_condition(normalized_indices, mask)

            # Классификация состояния
            overall_stats = plant_condition["statistics"].get("overall", {})
            overall_mean = overall_stats.get("mean", 0)

            if overall_mean > 0.7:
                condition_class = "Отличное"
                condition_color = "green"
            elif overall_mean > 0.4:
                condition_class = "Удовлетворительное"
                condition_color = "yellow"
            else:
                condition_class = "Плохое"
                condition_color = "red"

            plant_condition["classification"] = {
                "class": condition_class,
                "color": condition_color,
                "score": overall_mean,
            }

            return plant_condition

        except Exception as e:
            self.logger.error(f"Ошибка оценки состояния растений: {e}")
            return {"error": str(e)}

    def get_index_statistics(
        self, index_path: str, mask_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получить статистику по индексу

        Args:
            index_path: Путь к файлу индекса
            mask_path: Путь к маске (опционально)

        Returns:
            Словарь со статистикой
        """
        try:
            from ..utils.gdal_utils import read_raster_band

            # Чтение индекса с использованием централизованной утилиты
            index_data = read_raster_band(index_path, band_number=1)

            # Применение маски
            if mask_path and os.path.exists(mask_path):
                mask_data = read_raster_band(mask_path, band_number=1)
                index_data = index_data[mask_data > 0]

            # Расчет статистики
            statistics = {
                "count": int(np.count_nonzero(~np.isnan(index_data))),
                "mean": float(np.nanmean(index_data)),
                "std": float(np.nanstd(index_data)),
                "min": float(np.nanmin(index_data)),
                "max": float(np.nanmax(index_data)),
                "median": float(np.nanmedian(index_data)),
                "q25": float(np.nanpercentile(index_data, 25)),
                "q75": float(np.nanpercentile(index_data, 75)),
            }

            return statistics

        except Exception as e:
            self.logger.error(f"Ошибка расчета статистики: {e}")
            return {"error": str(e)}
