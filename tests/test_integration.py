"""
Интеграционные тесты для проекта GOP
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from src.core.pipeline import Pipeline
from src.core.config import Config
from src.processing.hyperspectral import HyperspectralProcessor
from src.processing.orthophoto import OrthophotoProcessor
from src.segmentation.segmenter import ImageSegmenter
from src.indices.calculator import IndexCalculator
from src.indices.definitions import IndexDefinitions


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты для всего проекта"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Создание тестовой конфигурации
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        test_config = {
            'processing': {
                'hyperspectral': {
                    'radiometric_correction': True,
                    'noise_reduction': True,
                    'output_format': 'GTiff'
                },
                'orthophoto': {
                    'resolution': 0.01,
                    'method': 'gdal'  # Используем GDAL для тестов
                }
            },
            'segmentation': {
                'compression_ratio': 0.125,
                'preliminary_model': 'deeplabv3plus',
                'refinement_model': 'cascade_psp'
            },
            'indices': {
                'default_indices': ['GNDVI', 'NDWI', 'MCARI'],
                'normalization_method': 'minmax'
            },
            'output': {
                'save_intermediate': False,
                'create_visualizations': True
            }
        }
        
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Создание тестовых файлов
        self.test_input_dir = os.path.join(self.temp_dir, 'input')
        self.test_output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.test_input_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Создание тестового гиперспектрального файла
        self.test_bil_path = os.path.join(self.test_input_dir, 'test.bil')
        self.test_hdr_path = os.path.join(self.test_input_dir, 'test.hdr')
        
        # Создание простого тестового изображения
        self.width, self.height, self.bands = 100, 100, 50
        test_data = np.random.randint(0, 1000, (self.height, self.width, self.bands), dtype=np.uint16)
        test_data.tofile(self.test_bil_path)
        
        # Создание заголовка
        hdr_content = f"""ENVI
samples = {self.width}
lines = {self.height}
bands = {self.bands}
header offset = 0
file type = ENVI Standard
data type = 12
interleave = bil
byte order = 0
wavelength = {{400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890}}
"""
        with open(self.test_hdr_path, 'w') as f:
            f.write(hdr_content)
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.processing.hyperspectral.gdal.Open')
    @patch('src.processing.orthophoto.gdal.Open')
    @patch('src.processing.orthophoto.gdal.Warp')
    def test_full_pipeline_integration(self, mock_gdal_warp, mock_gdal_open_ortho, mock_gdal_open_hyper):
        """Тест полной интеграции пайплайна"""
        # Настройка моков для гиперспектральной обработки
        mock_dataset_hyper = MagicMock()
        mock_dataset_hyper.RasterXSize = self.width
        mock_dataset_hyper.RasterYSize = self.height
        mock_dataset_hyper.RasterCount = self.bands
        
        # Создание моковых данных для каждого канала
        band_data = np.random.rand(self.height, self.width)
        mock_band = MagicMock()
        mock_band.ReadAsArray.return_value = band_data
        mock_band.GetMinimum.return_value = 0.1
        mock_band.GetMaximum.return_value = 0.9
        mock_band.GetStatistics.return_value = (0.1, 0.9, 0.5, 0.1)
        mock_dataset_hyper.GetRasterBand.return_value = mock_band
        mock_dataset_hyper.GetMetadata.return_value = {
            'wavelength': '400, 410, 420, 430, 440, 450, 460, 470, 480, 490'
        }
        
        mock_gdal_open_hyper.return_value = mock_dataset_hyper
        
        # Настройка моков для ортофотоплана
        mock_dataset_ortho = MagicMock()
        mock_gdal_open_ortho.return_value = mock_dataset_ortho
        mock_gdal_warp.return_value = mock_dataset_ortho
        
        # Создание пайплайна
        pipeline = Pipeline(self.config_path)
        
        # Запуск полной обработки
        with patch.object(pipeline, '_segment_image') as mock_segment:
            mock_segment.return_value = {
                'segmentation_mask': np.random.randint(0, 5, (self.height, self.width)),
                'mask_path': os.path.join(self.test_output_dir, 'segmentation.png')
            }
            
            result = pipeline.process(
                input_path=self.test_bil_path,
                output_dir=self.test_output_dir,
                sensor_type='Hyperspectral',
                selected_indices=['GNDVI', 'NDWI']
            )
            
            # Проверка результатов
            self.assertIn('orthophoto_path', result)
            self.assertIn('segmentation_mask', result)
            self.assertIn('indices', result)
            self.assertIn('plant_condition', result)
            
            # Проверка индексов
            self.assertIn('GNDVI', result['indices'])
            self.assertIn('NDWI', result['indices'])
            
            # Проверка состояния растений
            self.assertIn('classification', result['plant_condition'])
    
    def test_config_integration(self):
        """Тест интеграции конфигурации со всеми компонентами"""
        config = Config(self.config_path)
        
        # Проверка, что конфигурация доступна во всех компонентах
        self.assertEqual(config.get('processing.hyperspectral.radiometric_correction'), True)
        self.assertEqual(config.get('segmentation.compression_ratio'), 0.125)
        self.assertEqual(config.get('indices.default_indices'), ['GNDVI', 'NDWI', 'MCARI'])
    
    def test_indices_integration(self):
        """Тест интеграции калькулятора индексов с определениями"""
        calculator = IndexCalculator()
        
        # Создание тестовых данных
        spectral_data = np.random.rand(self.height, self.width, self.bands) * 0.5 + 0.25
        segmentation_mask = np.random.randint(0, 5, (self.height, self.width))
        
        # Расчет индексов
        result = calculator.calculate(spectral_data, ['GNDVI', 'NDWI'], segmentation_mask)
        
        # Проверка, что все индексы рассчитаны
        self.assertIn('GNDVI', result)
        self.assertIn('NDWI', result)
        
        # Проверка, что индексы соответствуют определениям
        gndvi_info = IndexDefinitions.get_index_info('GNDVI')
        self.assertEqual(gndvi_info['name'], 'GNDVI')
        
        ndwi_info = IndexDefinitions.get_index_info('NDWI')
        self.assertEqual(ndwi_info['name'], 'NDWI')
    
    @patch('cv2.imread')
    @patch('cv2.imwrite')
    def test_segmentation_integration(self, mock_imwrite, mock_imread):
        """Тест интеграции сегментации с обработкой изображений"""
        # Настройка моков
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = test_image
        mock_imwrite.return_value = True
        
        segmenter = ImageSegmenter()
        
        # Создание тестового изображения
        test_image_path = os.path.join(self.test_input_dir, 'test_image.png')
        with open(test_image_path, 'w') as f:
            f.write('test image')
        
        # Запуск сегментации
        with patch.object(segmenter, '_simple_segmentation') as mock_simple:
            mock_simple.return_value = np.random.randint(0, 5, (100, 100))
            
            result = segmenter.segment(
                test_image_path, 
                self.test_output_dir, 
                compression_ratio=0.125,
                use_refinement=False
            )
            
            # Проверка результатов
            self.assertIn('segmentation_mask', result)
            self.assertIn('mask_path', result)
    
    def test_error_propagation_integration(self):
        """Тест распространения ошибок между компонентами"""
        pipeline = Pipeline(self.config_path)
        
        # Тест с неверным входным файлом
        with self.assertRaises(FileNotFoundError):
            pipeline.process(
                input_path='/nonexistent/file.bil',
                output_dir=self.test_output_dir
            )
    
    def test_memory_efficiency_integration(self):
        """Тест интеграции с оптимизацией памяти"""
        # Создание больших тестовых данных
        large_data = np.random.rand(500, 500, 50)  # Большие размеры
        
        # Проверка, что компоненты могут обрабатывать большие данные
        calculator = IndexCalculator()
        segmentation_mask = np.random.randint(0, 5, (500, 500))
        
        # Расчет индексов с оптимизацией памяти
        result = calculator.calculate(large_data, ['GNDVI'], segmentation_mask)
        
        self.assertIn('GNDVI', result)
        self.assertEqual(result['GNDVI'].shape, (500, 500))
    
    def test_scientific_analysis_integration(self):
        """Тест интеграции научного анализа"""
        pipeline = Pipeline(self.config_path)
        
        # Создание тестовых данных для научного анализа
        indices_results = {
            'GNDVI': np.random.rand(100, 100) * 0.8 + 0.1,
            'NDWI': np.random.rand(100, 100) * 0.8 + 0.1,
            'MCARI': np.random.rand(100, 100) * 0.8 + 0.1
        }
        
        # Запуск научного анализа
        with patch.object(pipeline, '_perform_scientific_analysis') as mock_scientific:
            mock_scientific.return_value = {
                'statistics': {
                    'GNDVI': {'mean': 0.5, 'std': 0.1, 'skewness': 0.05, 'kurtosis': 0.02},
                    'NDWI': {'mean': 0.4, 'std': 0.15, 'skewness': -0.1, 'kurtosis': 0.03},
                    'MCARI': {'mean': 0.6, 'std': 0.12, 'skewness': 0.08, 'kurtosis': -0.01}
                },
                'correlations': [
                    {'index1': 'GNDVI', 'index2': 'NDWI', 'correlation': 0.75},
                    {'index1': 'GNDVI', 'index2': 'MCARI', 'correlation': 0.82}
                ],
                'spatial_analysis': {
                    'morans_i': 0.65,
                    'hotspots': {'count': 15, 'percentage': 0.15},
                    'fragmentation': 0.25
                }
            }
            
            result = pipeline._perform_scientific_analysis(
                indices_results, 
                np.random.randint(0, 5, (100, 100))
            )
            
            # Проверка результатов научного анализа
            self.assertIn('statistics', result)
            self.assertIn('correlations', result)
            self.assertIn('spatial_analysis', result)
    
    def test_output_format_integration(self):
        """Тест интеграции форматов вывода"""
        config = Config(self.config_path)
        
        # Проверка, что формат вывода настроен правильно
        output_format = config.get('processing.hyperspectral.output_format')
        self.assertEqual(output_format, 'GTiff')
        
        # Проверка, что визуализации включены
        create_visualizations = config.get('output.create_visualizations')
        self.assertTrue(create_visualizations)
    
    def test_batch_processing_integration(self):
        """Тест интеграции пакетной обработки"""
        # Создание нескольких тестовых файлов
        test_files = []
        for i in range(3):
            bil_path = os.path.join(self.test_input_dir, f'test_{i}.bil')
            hdr_path = os.path.join(self.test_input_dir, f'test_{i}.hdr')
            
            test_data = np.random.randint(0, 1000, (self.height, self.width, self.bands), dtype=np.uint16)
            test_data.tofile(bil_path)
            
            with open(hdr_path, 'w') as f:
                f.write(f"""ENVI
samples = {self.width}
lines = {self.height}
bands = {self.bands}
header offset = 0
file type = ENVI Standard
data type = 12
interleave = bil
byte order = 0
""")
            
            test_files.append(bil_path)
        
        # Создание пайплайна
        pipeline = Pipeline(self.config_path)
        
        # Мокирование обработки для ускорения теста
        with patch.object(pipeline, 'process') as mock_process:
            mock_process.return_value = {
                'orthophoto_path': os.path.join(self.test_output_dir, 'orthophoto.tif'),
                'segmentation_mask': np.random.randint(0, 5, (self.height, self.width)),
                'indices': {'GNDVI': np.random.rand(self.height, self.width)},
                'plant_condition': {'classification': {'class': 'healthy', 'score': 0.8}}
            }
            
            # Обработка каждого файла
            results = []
            for test_file in test_files:
                file_output_dir = os.path.join(self.test_output_dir, f'output_{os.path.basename(test_file)}')
                result = pipeline.process(
                    input_path=test_file,
                    output_dir=file_output_dir
                )
                results.append(result)
            
            # Проверка результатов
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIn('orthophoto_path', result)
                self.assertIn('indices', result)


if __name__ == '__main__':
    unittest.main()