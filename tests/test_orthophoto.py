"""
Тесты для модуля обработки ортофотопланов
"""

import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock, mock_open
from src.processing.orthophoto import OrthophotoProcessor


class TestOrthophotoProcessor(unittest.TestCase):
    """Тесты процессора ортофотопланов"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.processor = OrthophotoProcessor()
        self.temp_dir = tempfile.mkdtemp()
        
        # Создание тестовых TIFF файлов
        self.test_tiff_paths = []
        for i in range(3):
            tiff_path = os.path.join(self.temp_dir, f'test_{i}.tif')
            self.test_tiff_paths.append(tiff_path)
            
            # Создание простого тестового файла
            with open(tiff_path, 'w') as f:
                f.write(f"test tiff file {i}")
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_processor_initialization(self):
        """Тест инициализации процессора"""
        self.assertIsInstance(self.processor, OrthophotoProcessor)
    
    @patch('shutil.which')
    def test_find_odm_path_found(self, mock_which):
        """Тест поиска пути ODM при наличии"""
        mock_which.return_value = '/usr/local/bin/odm'
        
        odm_path = self.processor._find_odm_path()
        
        self.assertEqual(odm_path, '/usr/local/bin/odm')
        mock_which.assert_called_once_with('odm')
    
    @patch('shutil.which')
    def test_find_odm_path_not_found(self, mock_which):
        """Тест поиска пути ODM при отсутствии"""
        mock_which.return_value = None
        
        odm_path = self.processor._find_odm_path()
        
        self.assertIsNone(odm_path)
        mock_which.assert_called_once_with('odm')
    
    @patch.object(OrthophotoProcessor, '_find_odm_path')
    @patch.object(OrthophotoProcessor, '_create_with_odm')
    @patch.object(OrthophotoProcessor, '_create_with_gdal')
    def test_create_orthophoto_with_odm(self, mock_gdal, mock_odm, mock_find_odm):
        """Тест создания ортофотоплана с использованием ODM"""
        # Настройка моков
        mock_find_odm.return_value = '/usr/local/bin/odm'
        mock_odm.return_value = os.path.join(self.temp_dir, 'orthophoto_odm.tif')
        
        # Создание тестовых данных
        processed_data = {
            'processed_data': [[1, 2, 3], [4, 5, 6]],
            'metadata': {'sensor_type': 'Hyperspectral'}
        }
        
        result_path = self.processor.create_orthophoto(
            self.test_tiff_paths, self.temp_dir, processed_data
        )
        
        self.assertEqual(result_path, os.path.join(self.temp_dir, 'orthophoto_odm.tif'))
        mock_odm.assert_called_once()
        mock_gdal.assert_not_called()
    
    @patch.object(OrthophotoProcessor, '_find_odm_path')
    @patch.object(OrthophotoProcessor, '_create_with_odm')
    @patch.object(OrthophotoProcessor, '_create_with_gdal')
    def test_create_orthophoto_with_gdal(self, mock_gdal, mock_odm, mock_find_odm):
        """Тест создания ортофотоплана с использованием GDAL"""
        # Настройка моков
        mock_find_odm.return_value = None  # ODM не найден
        mock_gdal.return_value = os.path.join(self.temp_dir, 'orthophoto_gdal.tif')
        
        # Создание тестовых данных
        processed_data = {
            'processed_data': [[1, 2, 3], [4, 5, 6]],
            'metadata': {'sensor_type': 'Hyperspectral'}
        }
        
        result_path = self.processor.create_orthophoto(
            self.test_tiff_paths, self.temp_dir, processed_data
        )
        
        self.assertEqual(result_path, os.path.join(self.temp_dir, 'orthophoto_gdal.tif'))
        mock_gdal.assert_called_once()
        mock_odm.assert_not_called()
    
    @patch('subprocess.run')
    @patch('tempfile.TemporaryDirectory')
    @patch.object(OrthophotoProcessor, '_create_gps_file')
    def test_create_with_odm(self, mock_gps, mock_temp_dir, mock_subprocess):
        """Тест создания ортофотоплана через ODM"""
        # Настройка моков
        mock_temp_dir.return_value.__enter__.return_value = self.temp_dir
        mock_gps.return_value = os.path.join(self.temp_dir, 'gps.txt')
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        expected_output = os.path.join(self.temp_dir, 'odm_orthophoto', 'odm_georeferenced_model.geo.tif')
        
        result_path = self.processor._create_with_odm(self.test_tiff_paths, self.temp_dir)
        
        self.assertEqual(result_path, expected_output)
        mock_subprocess.assert_called_once()
        mock_gps.assert_called_once()
    
    @patch('subprocess.run')
    @patch.object(OrthophotoProcessor, '_create_gps_file')
    def test_create_with_odm_failure(self, mock_gps, mock_subprocess):
        """Тест обработки ошибки при создании ортофотоплана через ODM"""
        # Настройка моков
        mock_gps.return_value = None  # GPS файл не создан
        
        with self.assertRaises(Exception):
            self.processor._create_with_odm(self.test_tiff_paths, self.temp_dir)
        
        mock_gps.assert_called_once()
        mock_subprocess.assert_not_called()
    
    @patch('subprocess.run')
    def test_create_with_odm_subprocess_error(self, mock_subprocess):
        """Тест обработки ошибки subprocess при создании через ODM"""
        # Настройка моков
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ODM error"
        mock_subprocess.return_value = mock_result
        
        with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
            mock_temp_dir.return_value.__enter__.return_value = self.temp_dir
            
            with patch.object(self.processor, '_create_gps_file') as mock_gps:
                mock_gps.return_value = os.path.join(self.temp_dir, 'gps.txt')
                
                with self.assertRaises(Exception):
                    self.processor._create_with_odm(self.test_tiff_paths, self.temp_dir)
    
    @patch('src.processing.orthophoto.gdal.Warp')
    @patch('src.processing.orthophoto.gdal.Open')
    def test_create_with_gdal(self, mock_gdal_open, mock_gdal_warp):
        """Тест создания ортофотоплана через GDAL"""
        # Настройка моков
        mock_dataset = MagicMock()
        mock_gdal_open.return_value = mock_dataset
        
        mock_output_dataset = MagicMock()
        mock_gdal_warp.return_value = mock_output_dataset
        
        expected_output = os.path.join(self.temp_dir, 'merged_orthophoto.tif')
        
        result_path = self.processor._create_with_gdal(self.test_tiff_paths, self.temp_dir)
        
        self.assertEqual(result_path, expected_output)
        mock_gdal_warp.assert_called_once()
    
    @patch('src.processing.orthophoto.gdal.Open')
    def test_create_with_gdal_open_error(self, mock_gdal_open):
        """Тест обработки ошибки открытия файлов в GDAL"""
        mock_gdal_open.side_effect = Exception("GDAL error")
        
        with self.assertRaises(Exception):
            self.processor._create_with_gdal(self.test_tiff_paths, self.temp_dir)
    
    def test_create_gps_file_with_valid_data(self):
        """Тест создания GPS файла с валидными данными"""
        processed_data = {
            'gps_data': [
                {'lat': 55.7558, 'lon': 37.6173, 'alt': 150.0},
                {'lat': 55.7560, 'lon': 37.6175, 'alt': 151.0}
            ]
        }
        
        gps_path = self.processor._create_gps_file(processed_data, self.temp_dir)
        
        self.assertIsNotNone(gps_path)
        self.assertTrue(os.path.exists(gps_path))
        
        # Проверка содержимого файла
        with open(gps_path, 'r') as f:
            content = f.read()
            self.assertIn('55.7558', content)
            self.assertIn('37.6173', content)
            self.assertIn('150.0', content)
    
    def test_create_gps_file_without_gps_data(self):
        """Тест создания GPS файла без GPS данных"""
        processed_data = {
            'processed_data': [[1, 2, 3], [4, 5, 6]],
            'metadata': {'sensor_type': 'Hyperspectral'}
        }
        
        gps_path = self.processor._create_gps_file(processed_data, self.temp_dir)
        
        self.assertIsNone(gps_path)
    
    def test_copy_file(self):
        """Тест копирования файла"""
        src_path = self.test_tiff_paths[0]
        dst_path = os.path.join(self.temp_dir, 'copied_file.tif')
        
        self.processor._copy_file(src_path, dst_path)
        
        self.assertTrue(os.path.exists(dst_path))
        
        # Проверка содержимого
        with open(src_path, 'r') as src, open(dst_path, 'r') as dst:
            self.assertEqual(src.read(), dst.read())
    
    @patch('src.processing.orthophoto.gdal.Open')
    def test_validate_orthophoto(self, mock_gdal_open):
        """Тест валидации ортофотоплана"""
        # Настройка мока
        mock_dataset = MagicMock()
        mock_dataset.RasterXSize = 1000
        mock_dataset.RasterYSize = 1000
        mock_dataset.RasterCount = 3
        mock_dataset.GetGeoTransform.return_value = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
        mock_dataset.GetProjection.return_value = 'PROJCS["WGS 84"]'
        
        # Мок для статистики каналов
        mock_band = MagicMock()
        mock_band.GetStatistics.return_value = (0.0, 255.0, 127.5, 50.0)
        mock_dataset.GetRasterBand.return_value = mock_band
        
        mock_gdal_open.return_value = mock_dataset
        
        orthophoto_path = self.test_tiff_paths[0]
        validation_result = self.processor.validate_orthophoto(orthophoto_path)
        
        self.assertIn('valid', validation_result)
        self.assertIn('dimensions', validation_result)
        self.assertIn('bands', validation_result)
        self.assertIn('coordinate_system', validation_result)
        self.assertIn('statistics', validation_result)
        
        self.assertTrue(validation_result['valid'])
        self.assertEqual(validation_result['dimensions']['width'], 1000)
        self.assertEqual(validation_result['dimensions']['height'], 1000)
        self.assertEqual(validation_result['bands'], 3)
    
    @patch('src.processing.orthophoto.gdal.Open')
    def test_validate_orthophoto_invalid_file(self, mock_gdal_open):
        """Тест валидации некорректного ортофотоплана"""
        mock_gdal_open.return_value = None
        
        orthophoto_path = self.test_tiff_paths[0]
        validation_result = self.processor.validate_orthophoto(orthophoto_path)
        
        self.assertFalse(validation_result['valid'])
        self.assertIn('error', validation_result)
    
    @patch('src.processing.orthophoto.gdal.Translate')
    @patch('src.processing.orthophoto.gdal.Open')
    def test_optimize_orthophoto(self, mock_gdal_open, mock_gdal_translate):
        """Тест оптимизации ортофотоплана"""
        # Настройка моков
        mock_dataset = MagicMock()
        mock_gdal_open.return_value = mock_dataset
        
        mock_output_dataset = MagicMock()
        mock_gdal_translate.return_value = mock_output_dataset
        
        orthophoto_path = self.test_tiff_paths[0]
        output_path = os.path.join(self.temp_dir, 'optimized.tif')
        
        result_path = self.processor.optimize_orthophoto(
            orthophoto_path, output_path, 
            target_size=(500, 500), 
            compression='LZW'
        )
        
        self.assertEqual(result_path, output_path)
        mock_gdal_translate.assert_called_once()
    
    @patch('src.processing.orthophoto.gdal.Open')
    def test_optimize_orthophoto_open_error(self, mock_gdal_open):
        """Тест обработки ошибки при оптимизации ортофотоплана"""
        mock_gdal_open.return_value = None
        
        orthophoto_path = self.test_tiff_paths[0]
        output_path = os.path.join(self.temp_dir, 'optimized.tif')
        
        with self.assertRaises(Exception):
            self.processor.optimize_orthophoto(orthophoto_path, output_path)
    
    def test_create_orthophoto_empty_tiff_paths(self):
        """Тест создания ортофотоплана с пустым списком файлов"""
        processed_data = {
            'processed_data': [[1, 2, 3], [4, 5, 6]],
            'metadata': {'sensor_type': 'Hyperspectral'}
        }
        
        with self.assertRaises(ValueError):
            self.processor.create_orthophoto([], self.temp_dir, processed_data)
    
    def test_create_orthophoto_invalid_output_dir(self):
        """Тест создания ортофотоплана с неверной выходной директорией"""
        processed_data = {
            'processed_data': [[1, 2, 3], [4, 5, 6]],
            'metadata': {'sensor_type': 'Hyperspectral'}
        }
        
        with self.assertRaises(Exception):
            self.processor.create_orthophoto(
                self.test_tiff_paths, '/invalid/path/that/cannot/be/created', processed_data
            )


if __name__ == '__main__':
    unittest.main()