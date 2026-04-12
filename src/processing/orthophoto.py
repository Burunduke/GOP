"""
Модуль создания ортофотопланов
"""

import os
import logging
import subprocess
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..core.config import Config, config, get_config, create_config
from ..utils.logger import setup_logger
from ..utils.validators import validate_file_path, validate_config
from ..utils.exceptions import ProcessingError, ValidationError, FileError


class OrthophotoProcessor:
    """
    Класс для создания ортофотопланов с использованием OpenDroneMap
    """

    def __init__(self, config_instance: Optional[Config] = None):
        """
        Инициализация процессора ортофотопланов

        Args:
            config_instance: Опциональный экземпляр конфигурации для dependency injection
        """
        self.config = get_config(config_instance)
        self.logger = setup_logger("OrthophotoProcessor")
        self.odm_path = self._find_odm_path()

    def _find_odm_path(self) -> Optional[str]:
        """
        Найти путь к OpenDroneMap

        Returns:
            Путь к OpenDroneMap или None если не найден
        """
        # Проверка常见 путей установки ODM
        possible_paths = [
            "/opt/opendronemap",
            "/usr/local/opendronemap",
            os.path.expanduser("~/OpenDroneMap"),
            "./OpenDroneMap",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                run_script = os.path.join(path, "run.sh")
                if os.path.exists(run_script):
                    self.logger.info(f"OpenDroneMap найден: {path}")
                    return path

        self.logger.warning(
            "OpenDroneMap не найден. Используется альтернативный метод."
        )
        return None

    def create_orthophoto(self, processed_data: Dict[str, Any], output_dir: str) -> str:
        """
        Создание ортофотоплана

        Args:
            processed_data: Результаты предварительной обработки
            output_dir: Директория для сохранения результатов

        Returns:
            Путь к созданному ортофотоплану
        """
        try:
            # Validate input parameters
            if not processed_data or not isinstance(processed_data, dict):
                raise ValidationError(
                    "processed_data must be a non-empty dictionary",
                    details={"processed_data_type": type(processed_data)},
                )

            validate_file_path(output_dir, must_exist=False, must_be_writable=True)

            # Validate required keys in processed_data
            required_keys = ["tiff_paths", "metadata"]
            for key in required_keys:
                if key not in processed_data:
                    raise ValidationError(
                        f"Missing required key in processed_data: {key}",
                        details={
                            "required_key": key,
                            "available_keys": list(processed_data.keys()),
                        },
                    )

            # Validate TIFF paths
            tiff_paths = processed_data.get("tiff_paths", [])
            if not isinstance(tiff_paths, list) or len(tiff_paths) == 0:
                raise ValidationError(
                    "tiff_paths must be a non-empty list",
                    details={
                        "tiff_paths_type": type(tiff_paths),
                        "tiff_paths_length": (
                            len(tiff_paths) if isinstance(tiff_paths, list) else None
                        ),
                    },
                )

            for tiff_path in tiff_paths:
                validate_file_path(
                    tiff_path,
                    must_exist=True,
                    must_be_readable=True,
                    allowed_extensions=[".tif", ".tiff"],
                )

            self.logger.info("Начало создания ортофотоплана")

            tiff_paths = processed_data.get("tiff_paths", [])
            if not tiff_paths:
                raise ValidationError(
                    "Отсутствуют TIFF файлы для создания ортофотоплана",
                    details={"tiff_paths": tiff_paths},
                )

            # Создание ортофотоплана
            if self.odm_path:
                orthophoto_path = self._create_with_odm(tiff_paths, output_dir, processed_data)
            else:
                orthophoto_path = self._create_with_gdal(tiff_paths, output_dir)

            self.logger.info(f"Ортофотоплан создан: {orthophoto_path}")
            return orthophoto_path

        except Exception as e:
            self.logger.error(f"Ошибка создания ортофотоплана: {e}")
            raise

    def _create_with_odm(self, tiff_paths: List[str], output_dir: str, processed_data: Dict[str, Any] = None) -> str:
        """
        Создание ортофотоплана с помощью OpenDroneMap

        Args:
            tiff_paths: Список путей к TIFF файлам
            output_dir: Директория для сохранения результатов

        Returns:
            Путь к созданному ортофотоплану
        """
        try:
            # Создание временной директории для ODM
            with tempfile.TemporaryDirectory() as temp_dir:
                project_dir = os.path.join(temp_dir, "project")
                os.makedirs(project_dir, exist_ok=True)

                # Копирование файлов в структуру ODM
                images_dir = os.path.join(project_dir, "images")
                os.makedirs(images_dir, exist_ok=True)

                for tiff_path in tiff_paths:
                    dest_path = os.path.join(images_dir, os.path.basename(tiff_path))
                    self._copy_file(tiff_path, dest_path)

                # Создание GPS файла если необходимо
                gps_file = self._create_gps_file(processed_data, project_dir)

                # Формирование команды ODM
                cmd = [
                    os.path.join(self.odm_path, "run.sh"),
                    "--project-path",
                    project_dir,
                    "--orthophoto-resolution",
                    str(self.config.get("processing.orthophoto_resolution", 0.05)),
                    "--dem-resolution",
                    str(self.config.get("processing.dem_resolution", 0.1)),
                    "--feature-quality",
                    self.config.get("processing.feature_quality", "high"),
                    "--matcher-neighbors",
                    str(self.config.get("processing.matcher_neighbors", 8)),
                    "--use-exif",
                    "false",
                ]

                if gps_file:
                    cmd.extend(["--gps-file", gps_file])

                # Запуск ODM
                self.logger.info("Запуск OpenDroneMap...")
                result = subprocess.run(
                    cmd,
                    cwd=self.odm_path,
                    capture_output=True,
                    text=True,
                    timeout=self.config.get(
                        "processing.odm_timeout", 3600
                    ),  # 1 час по умолчанию
                )

                if result.returncode != 0:
                    self.logger.error(f"ODM завершился с ошибкой: {result.stderr}")
                    raise RuntimeError(f"OpenDroneMap error: {result.stderr}")

                # Копирование результатов
                odm_results_dir = os.path.join(
                    project_dir, "odm_orthophoto", "odm_orthophoto.tif"
                )
                if os.path.exists(odm_results_dir):
                    output_path = os.path.join(output_dir, "orthophoto.tif")
                    self._copy_file(odm_results_dir, output_path)
                    return output_path
                else:
                    raise FileNotFoundError("Результаты ODM не найдены")

        except subprocess.TimeoutExpired:
            self.logger.error("Превышено время выполнения OpenDroneMap")
            raise RuntimeError("OpenDroneMap timeout")
        except Exception as e:
            self.logger.error(f"Ошибка при работе с OpenDroneMap: {e}")
            raise

    def _create_with_gdal(self, tiff_paths: List[str], output_dir: str) -> str:
        """
        Создание ортофотоплана с помощью GDAL (альтернативный метод)

        Args:
            tiff_paths: Список путей к TIFF файлам
            output_dir: Директория для сохранения результатов

        Returns:
            Путь к созданному ортофотоплану
        """
        try:
            self.logger.info("Создание ортофотоплана с помощью GDAL")

            # Использование gdal_merge.py для создания мозаики
            output_path = os.path.join(output_dir, "orthophoto.tif")

            cmd = [
                "gdal_merge.py",
                "-o",
                output_path,
                "-of",
                "GTiff",
                "-co",
                "COMPRESS=LZW",
                "-co",
                "TILED=YES",
            ]
            cmd.extend(tiff_paths)

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"GDAL merge error: {result.stderr}")
                raise RuntimeError(f"GDAL merge error: {result.stderr}")

            return output_path

        except Exception as e:
            self.logger.error(f"Ошибка создания ортофотоплана с помощью GDAL: {e}")
            raise

    def _create_gps_file(
        self, processed_data: Dict[str, Any], project_dir: str
    ) -> Optional[str]:
        """
        Создание GPS файла для OpenDroneMap

        Args:
            processed_data: Данные обработки
            project_dir: Директория проекта

        Returns:
            Путь к GPS файлу или None
        """
        try:
            # Проверка наличия GPS данных в метаданных
            metadata = processed_data.get("metadata", {})
            if not metadata:
                return None

            gps_file = os.path.join(project_dir, "gps.txt")

            # Здесь должна быть логика извлечения GPS данных из метаданных
            # и сохранения в формате, понятном ODM

            return gps_file if os.path.exists(gps_file) else None

        except Exception as e:
            self.logger.warning(f"Ошибка создания GPS файла: {e}")
            return None

    def _copy_file(self, src: str, dst: str) -> None:
        """
        Копирование файла

        Args:
            src: Исходный путь
            dst: Целевой путь
        """
        import shutil

        shutil.copy2(src, dst)

    def validate_orthophoto(self, orthophoto_path: str) -> Dict[str, Any]:
        """
        Валидация созданного ортофотоплана

        Args:
            orthophoto_path: Путь к ортофотоплану

        Returns:
            Словарь с результатами валидации
        """
        try:
            from ..utils.gdal_utils import get_raster_metadata

            metadata = get_raster_metadata(orthophoto_path)

            validation_results = {
                "valid": True,
                "width": metadata["width"],
                "height": metadata["height"],
                "bands": metadata["band_count"],
                "has_georeference": metadata["has_georeference"],
                "has_projection": metadata["has_projection"],
            }

            # Проверка на пустые области
            if metadata["band_stats"] and metadata["band_stats"].get(1):
                stats = metadata["band_stats"][1]
                if stats["max"] <= stats["min"]:  # max <= min
                    validation_results["valid"] = False
                    validation_results["error"] = (
                        "Изображение содержит только пустые значения"
                    )

            return validation_results

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def optimize_orthophoto(self, orthophoto_path: str, output_path: str = None) -> str:
        """
        Оптимизация ортофотоплана (сжатие, пирамиды)

        Args:
            orthophoto_path: Путь к исходному ортофотоплану
            output_path: Путь для сохранения оптимизированного файла

        Returns:
            Путь к оптимизированному ортофотоплану
        """
        try:
            if output_path is None:
                base_path = os.path.splitext(orthophoto_path)[0]
                output_path = f"{base_path}_optimized.tif"

            # Команда GDAL для оптимизации
            cmd = [
                "gdal_translate",
                orthophoto_path,
                output_path,
                "-co",
                "COMPRESS=LZW",
                "-co",
                "TILED=YES",
                "-co",
                "BIGTIFF=IF_NEEDED",
                "-co",
                "PREDICTOR=2",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"Ошибка оптимизации: {result.stderr}")
                raise RuntimeError(f"Optimization error: {result.stderr}")

            # Создание пирамид
            pyramid_cmd = [
                "gdaladdo",
                "-r",
                "average",
                output_path,
                "2",
                "4",
                "8",
                "16",
            ]

            subprocess.run(pyramid_cmd, capture_output=True, text=True)

            self.logger.info(f"Ортофотоплан оптимизирован: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации ортофотоплана: {e}")
            raise
