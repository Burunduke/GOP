"""
Модуль создания ортофотопланов
"""

import os
import logging
import subprocess
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..core.config import config
from ..utils.logger import setup_logger


class OrthophotoProcessor:
    """
    Класс для создания ортофотопланов с использованием OpenDroneMap
    """
    
    def __init__(self):
        """Инициализация процессора ортофотопланов"""
        self.logger = setup_logger('OrthophotoProcessor')
        self.odm_path = self._find_odm_path()
        
    def _find_odm_path(self) -> Optional[str]:
        """
        Найти путь к OpenDroneMap
        
        Returns:
            Путь к OpenDroneMap или None если не найден
        """
        # Проверка常见 путей установки ODM
        possible_paths = [
            '/opt/opendronemap',
            '/usr/local/opendronemap',
            os.path.expanduser('~/OpenDroneMap'),
            './OpenDroneMap'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                run_script = os.path.join(path, 'run.sh')
                if os.path.exists(run_script):
                    self.logger.info(f"OpenDroneMap найден: {path}")
                    return path
        
        self.logger.warning("OpenDroneMap не найден. Используется альтернативный метод.")
        return None
    
    def create_orthophoto(self, 
                         processed_data: Dict[str, Any], 
                         output_dir: str) -> str:
        """
        Создание ортофотоплана
        
        Args:
            processed_data: Результаты предварительной обработки
            output_dir: Директория для сохранения результатов
            
        Returns:
            Путь к созданному ортофотоплану
        """
        try:
            self.logger.info("Начало создания ортофотоплана")
            
            tiff_paths = processed_data.get('tiff_paths', [])
            if not tiff_paths:
                raise ValueError("Отсутствуют TIFF файлы для создания ортофотоплана")
            
            # Создание ортофотоплана
            if self.odm_path:
                orthophoto_path = self._create_with_odm(tiff_paths, output_dir)
            else:
                orthophoto_path = self._create_with_gdal(tiff_paths, output_dir)
            
            self.logger.info(f"Ортофотоплан создан: {orthophoto_path}")
            return orthophoto_path
            
        except Exception as e:
            self.logger.error(f"Ошибка создания ортофотоплана: {e}")
            raise
    
    def _create_with_odm(self, tiff_paths: List[str], output_dir: str) -> str:
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
                project_dir = os.path.join(temp_dir, 'project')
                os.makedirs(project_dir, exist_ok=True)
                
                # Копирование файлов в структуру ODM
                images_dir = os.path.join(project_dir, 'images')
                os.makedirs(images_dir, exist_ok=True)
                
                for tiff_path in tiff_paths:
                    dest_path = os.path.join(images_dir, os.path.basename(tiff_path))
                    self._copy_file(tiff_path, dest_path)
                
                # Создание GPS файла если необходимо
                gps_file = self._create_gps_file(processed_data, project_dir)
                
                # Формирование команды ODM
                cmd = [
                    os.path.join(self.odm_path, 'run.sh'),
                    '--project-path', project_dir,
                    '--orthophoto-resolution', str(config.get('processing.orthophoto_resolution', 0.05)),
                    '--dem-resolution', str(config.get('processing.dem_resolution', 0.1)),
                    '--feature-quality', config.get('processing.feature_quality', 'high'),
                    '--matcher-neighbors', str(config.get('processing.matcher_neighbors', 8)),
                    '--use-exif', 'false'
                ]
                
                if gps_file:
                    cmd.extend(['--gps-file', gps_file])
                
                # Запуск ODM
                self.logger.info("Запуск OpenDroneMap...")
                result = subprocess.run(
                    cmd,
                    cwd=self.odm_path,
                    capture_output=True,
                    text=True,
                    timeout=config.get('processing.odm_timeout', 3600)  # 1 час по умолчанию
                )
                
                if result.returncode != 0:
                    self.logger.error(f"ODM завершился с ошибкой: {result.stderr}")
                    raise RuntimeError(f"OpenDroneMap error: {result.stderr}")
                
                # Копирование результатов
                odm_results_dir = os.path.join(project_dir, 'odm_orthophoto', 'odm_orthophoto.tif')
                if os.path.exists(odm_results_dir):
                    output_path = os.path.join(output_dir, 'orthophoto.tif')
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
            output_path = os.path.join(output_dir, 'orthophoto.tif')
            
            cmd = [
                'gdal_merge.py',
                '-o', output_path,
                '-of', 'GTiff',
                '-co', 'COMPRESS=LZW',
                '-co', 'TILED=YES'
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
    
    def _create_gps_file(self, processed_data: Dict[str, Any], project_dir: str) -> Optional[str]:
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
            metadata = processed_data.get('metadata', {})
            if not metadata:
                return None
            
            gps_file = os.path.join(project_dir, 'gps.txt')
            
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
            from osgeo import gdal
            
            dataset = gdal.Open(orthophoto_path)
            if dataset is None:
                raise ValueError(f"Не удалось открыть ортофотоплан: {orthophoto_path}")
            
            validation_results = {
                'valid': True,
                'width': dataset.RasterXSize,
                'height': dataset.RasterYSize,
                'bands': dataset.RasterCount,
                'has_georeference': False,
                'has_projection': False
            }
            
            # Проверка геопривязки
            geo_transform = dataset.GetGeoTransform()
            if geo_transform and not (geo_transform[0] == 0 and geo_transform[3] == 0):
                validation_results['has_georeference'] = True
            
            # Проверка проекции
            projection = dataset.GetProjection()
            if projection:
                validation_results['has_projection'] = True
            
            # Проверка на пустые области
            band = dataset.GetRasterBand(1)
            stats = band.GetStatistics(False, True)
            if stats[1] <= stats[0]:  # max <= min
                validation_results['valid'] = False
                validation_results['error'] = 'Изображение содержит только пустые значения'
            
            return validation_results
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def optimize_orthophoto(self, 
                          orthophoto_path: str, 
                          output_path: str = None) -> str:
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
                'gdal_translate',
                orthophoto_path,
                output_path,
                '-co', 'COMPRESS=LZW',
                '-co', 'TILED=YES',
                '-co', 'BIGTIFF=IF_NEEDED',
                '-co', 'PREDICTOR=2'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Ошибка оптимизации: {result.stderr}")
                raise RuntimeError(f"Optimization error: {result.stderr}")
            
            # Создание пирамид
            pyramid_cmd = [
                'gdaladdo',
                '-r', 'average',
                output_path,
                '2', '4', '8', '16'
            ]
            
            subprocess.run(pyramid_cmd, capture_output=True, text=True)
            
            self.logger.info(f"Ортофотоплан оптимизирован: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации ортофотоплана: {e}")
            raise