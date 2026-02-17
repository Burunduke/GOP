"""
Тесты для основного пайплайна обработки
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
import json
from unittest.mock import patch, MagicMock
from src.core.pipeline import Pipeline
from src.core.config import Config


class TestPipeline(unittest.TestCase):
    """Тесты основного пайплайна обработки"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Создание тестовой конфигурации
        test_config = {
            'processing': {
                'hyperspectral': {
                    'radiometric_correction': True,
                    'noise_reduction': True,
                    'output_format': 'GTiff'
                },
                'orthophoto': {
                    'resolution': 0.01,
                    'method': 'odm'
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
        
        # Сохранение тестовой конфигурации
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Создание пайплайна
        self.pipeline = Pipeline(self.config_path)
        
        # Создание тестовых файлов
        self.test_input_dir = os.path.join(self.temp_dir, 'input')
        self.test_output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.test_input_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Создание тестового гиперспектрального файла
        self.test_bil_path = os.path.join(self.test_input_dir, 'test.bil')
        self.test_hdr_path = os.path.join(self.test_input_dir, 'test.hdr')
        
        # Создание простого тестового изображения
        test_data = np.random.randint(0, 256, (100, 100, 10), dtype=np.uint16)
        test_data.tofile(self.test_bil_path)
        
        # Создание заголовка
        hdr_content = """ENVI
samples = 100
lines = 100
bands = 10
header offset = 0
file type = ENVI Standard
data type = 12
interleave = bil
byte order = 0
"""
        with open(self.test_hdr_path, 'w') as f:
            f.write(hdr_content)
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Тест инициализации пайплайна"""
        self.assertIsInstance(self.pipeline.config, Config)
        self.assertIsNotNone(self.pipeline.hyperspectral_processor)
        self.assertIsNotNone(self.pipeline.orthophoto_processor)
        self.assertIsNotNone(self.pipeline.index_calculator)
        self.assertIsNotNone(self.pipeline.segmenter)
    
    def test_pipeline_initialization_with_default_config(self):
        """Тест инициализации пайплайна с конфигурацией по умолчанию"""
        pipeline = Pipeline()
        self.assertIsInstance(pipeline.config, Config)
    
    @patch('src.processing.hyperspectral.HyperspectralProcessor.process')
    @patch('src.processing.orthophoto.OrthophotoProcessor.process')
    @patch('src.segmentation.segmenter.ImageSegmenter.segment')
    @patch('src.indices.calculator.IndexCalculator.calculate')
    @patch('src.indices.calculator.IndexCalculator.assess_plant_condition')
    def test_process_full_pipeline(self, mock_assess, mock_calculate, 
                                  mock_segment, mock_orthophoto, mock_hyperspectral):
        """Тест полного цикла обработки"""
        # Настройка моков
        mock_hyperspectral.return_value = {
            'processed_data': np.random.rand(100, 100, 10),
            'metadata': {'sensor_type': 'Hyperspectral'}
        }
        
        mock_orthophoto.return_value = {
            'orthophoto_path': os.path.join(self.test_output_dir, 'orthophoto.tif'),
            'georeferenced_data': np.random.rand(100, 100, 3)
        }
        
        mock_segment.return_value = {
            'segmentation_mask': np.random.randint(0, 5, (100, 100)),
            'segmentation_path': os.path.join(self.test_output_dir, 'segmentation.png')
        }
        
        mock_calculate.return_value = {
            'GNDVI': np.random.rand(100, 100),
            'NDWI': np.random.rand(100, 100),
            'MCARI': np.random.rand(100, 100)
        }
        
        mock_assess.return_value = {
            'classification': {'class': 'healthy', 'score': 0.85},
            'indices': {'GNDVI': 0.75, 'NDWI': 0.65, 'MCARI': 0.70}
        }
        
        # Запуск пайплайна
        result = self.pipeline.process(
            input_path=self.test_bil_path,
            output_dir=self.test_output_dir,
            sensor_type='Hyperspectral',
            selected_indices=['GNDVI', 'NDWI', 'MCARI']
        )
        
        # Проверка результатов
        self.assertIn('orthophoto_path', result)
        self.assertIn('segmentation_mask', result)
        self.assertIn('indices', result)
        self.assertIn('plant_condition', result)
        
        # Проверка вызовов моков
        mock_hyperspectral.assert_called_once()
        mock_orthophoto.assert_called_once()
        mock_segment.assert_called_once()
        mock_calculate.assert_called_once()
        mock_assess.assert_called_once()
    
    def test_process_invalid_input_path(self):
        """Тест обработки неверного пути к входному файлу"""
        with self.assertRaises(FileNotFoundError):
            self.pipeline.process(
                input_path='nonexistent_file.bil',
                output_dir=self.test_output_dir
            )
    
    def test_process_invalid_output_dir(self):
        """Тест обработки неверной выходной директории"""
        with self.assertRaises(Exception):
            self.pipeline.process(
                input_path=self.test_bil_path,
                output_dir='/invalid/path/that/cannot/be/created'
            )
    
    def test_process_with_custom_indices(self):
        """Тест обработки с пользовательскими индексами"""
        with patch.object(self.pipeline.index_calculator, 'calculate') as mock_calculate:
            mock_calculate.return_value = {'GNDVI': np.random.rand(100, 100)}
            
            self.pipeline.process(
                input_path=self.test_bil_path,
                output_dir=self.test_output_dir,
                selected_indices=['GNDVI']
            )
            
            # Проверка вызова с правильными индексами
            mock_calculate.assert_called_once()
            args, kwargs = mock_calculate.call_args
            self.assertEqual(args[1], ['GNDVI'])
    
    def test_save_results(self):
        """Тест сохранения результатов"""
        test_results = {
            'orthophoto_path': '/path/to/orthophoto.tif',
            'segmentation_mask': np.random.randint(0, 5, (100, 100)),
            'indices': {'GNDVI': np.random.rand(100, 100)},
            'plant_condition': {
                'classification': {'class': 'healthy', 'score': 0.85}
            }
        }
        
        save_path = os.path.join(self.temp_dir, 'results.json')
        self.pipeline.save_results(test_results, save_path)
        
        self.assertTrue(os.path.exists(save_path))
    
    def test_load_results(self):
        """Тест загрузки результатов"""
        # Создание тестового файла результатов
        test_results = {
            'orthophoto_path': '/path/to/orthophoto.tif',
            'indices': {'GNDVI': [0.1, 0.2, 0.3]},
            'plant_condition': {
                'classification': {'class': 'healthy', 'score': 0.85}
            }
        }
        
        save_path = os.path.join(self.temp_dir, 'results.json')
        self.pipeline.save_results(test_results, save_path)
        
        # Загрузка результатов
        loaded_results = self.pipeline.load_results(save_path)
        
        self.assertEqual(loaded_results['orthophoto_path'], test_results['orthophoto_path'])
        self.assertEqual(loaded_results['indices'], test_results['indices'])
        self.assertEqual(loaded_results['plant_condition'], test_results['plant_condition'])
    
    def test_get_processing_status(self):
        """Тест получения статуса обработки"""
        status = self.pipeline.get_processing_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('current_step', status)
        self.assertIn('total_steps', status)
        self.assertIn('progress_percentage', status)
    
    def test_cancel_processing(self):
        """Тест отмены обработки"""
        # Запуск обработки в фоновом режиме
        import threading
        
        def process_in_background():
            try:
                self.pipeline.process(
                    input_path=self.test_bil_path,
                    output_dir=self.test_output_dir
                )
            except Exception:
                pass  # Игнорируем исключения при отмене
        
        thread = threading.Thread(target=process_in_background)
        thread.start()
        
        # Небольшая задержка для начала обработки
        import time
        time.sleep(0.1)
        
        # Отмена обработки
        self.pipeline.cancel_processing()
        
        # Ожидание завершения потока
        thread.join(timeout=5)
        
        # Проверка статуса
        status = self.pipeline.get_processing_status()
        self.assertEqual(status['status'], 'cancelled')
    
    def test_process_with_invalid_sensor_type(self):
        """Тест обработки с неверным типом сенсора"""
        with self.assertRaises(ValueError):
            self.pipeline.process(
                input_path=self.test_bil_path,
                output_dir=self.test_output_dir,
                sensor_type='InvalidSensor'
            )
    
    def test_process_with_empty_indices_list(self):
        """Тест обработки с пустым списком индексов"""
        with patch.object(self.pipeline.index_calculator, 'calculate') as mock_calculate:
            mock_calculate.return_value = {}
            
            self.pipeline.process(
                input_path=self.test_bil_path,
                output_dir=self.test_output_dir,
                selected_indices=[]
            )
            
            # Проверка, что calculate был вызван с пустым списком
            args, kwargs = mock_calculate.call_args
            self.assertEqual(args[1], [])
    
    def test_load_results_invalid_json(self):
        """Тест загрузки результатов из невалидного JSON файла"""
        invalid_json_path = os.path.join(self.temp_dir, 'invalid.json')
        
        # Создание файла с невалидным JSON
        with open(invalid_json_path, 'w') as f:
            f.write('{ invalid json }')
        
        with self.assertRaises(json.JSONDecodeError):
            self.pipeline.load_results(invalid_json_path)
    
    def test_save_results_invalid_path(self):
        """Тест сохранения результатов в неверный путь"""
        test_results = {
            'orthophoto_path': '/path/to/orthophoto.tif',
            'indices': {'GNDVI': [0.1, 0.2, 0.3]}
        }
        
        invalid_path = '/invalid/path/that/cannot/be/created/results.json'
        
        with self.assertRaises(Exception):
            self.pipeline.save_results(test_results, invalid_path)
    
    def test_get_results_before_processing(self):
        """Тест получения результатов до обработки"""
        results = self.pipeline.get_results()
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 0)
    
    def test_export_scientific_data_without_processing(self):
        """Тест экспорта научных данных без обработки"""
        with self.assertRaises(ValueError):
            self.pipeline.export_scientific_data(self.test_output_dir)
    
    @patch('src.core.pipeline.Pipeline.process')
    def test_export_scientific_data_after_processing(self, mock_process):
        """Тест экспорта научных данных после обработки"""
        # Настройка мока для обработки
        mock_process.return_value = {
            'orthophoto_path': os.path.join(self.test_output_dir, 'orthophoto.tif'),
            'segmentation_mask': np.random.randint(0, 5, (100, 100)),
            'indices': {'GNDVI': np.random.rand(100, 100)},
            'plant_condition': {
                'classification': {'class': 'healthy', 'score': 0.85}
            }
        }
        
        # Запуск обработки
        self.pipeline.process(
            input_path=self.test_bil_path,
            output_dir=self.test_output_dir
        )
        
        # Экспорт данных
        self.pipeline.export_scientific_data(self.test_output_dir)
        
        # Проверка создания файлов
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'scientific_data.json')))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'processing_metadata.json')))
    
    def test_calculate_skewness_empty_data(self):
        """Тест расчета асимметрии для пустых данных"""
        empty_data = np.array([])
        
        with self.assertRaises(ValueError):
            self.pipeline._calculate_skewness(empty_data)
    
    def test_calculate_kurtosis_empty_data(self):
        """Тест расчета эксцесса для пустых данных"""
        empty_data = np.array([])
        
        with self.assertRaises(ValueError):
            self.pipeline._calculate_kurtosis(empty_data)
    
    def test_find_strong_correlations_empty_data(self):
        """Тест поиска сильных корреляций в пустых данных"""
        empty_indices = {}
        
        correlations = self.pipeline._find_strong_correlations(empty_indices, threshold=0.5)
        
        self.assertIsInstance(correlations, list)
        self.assertEqual(len(correlations), 0)
    
    def test_calculate_morans_i_empty_data(self):
        """Тест расчета индекса Морана для пустых данных"""
        empty_data = np.array([]).reshape(0, 0)
        
        with self.assertRaises(ValueError):
            self.pipeline._calculate_morans_i(empty_data)
    
    def test_perform_hotspot_analysis_empty_data(self):
        """Тест анализа горячих точек для пустых данных"""
        empty_data = np.array([]).reshape(0, 0)
        
        with self.assertRaises(ValueError):
            self.pipeline._perform_hotspot_analysis(empty_data)
    
    def test_calculate_fragmentation_index_empty_data(self):
        """Тест расчета индекса фрагментации для пустых данных"""
        empty_data = np.array([]).reshape(0, 0)
        
        with self.assertRaises(ValueError):
            self.pipeline._calculate_fragmentation_index(empty_data)
    
    def test_save_scientific_report_invalid_path(self):
        """Тест сохранения научного отчета в неверный путь"""
        analysis = {
            'indices': {'GNDVI': {'mean': 0.5, 'std': 0.1}},
            'correlations': []
        }
        
        invalid_path = '/invalid/path/that/cannot/be/created/report.json'
        
        with self.assertRaises(Exception):
            self.pipeline._save_scientific_report(analysis, invalid_path)
    
    def test_process_with_memory_efficiency(self):
        """Тест обработки с оптимизацией памяти"""
        with patch.object(self.pipeline, '_preprocess_hyperspectral') as mock_preprocess:
            mock_preprocess.return_value = {
                'processed_data': np.random.rand(100, 100, 10),
                'metadata': {'sensor_type': 'Hyperspectral'}
            }
            
            with patch.object(self.pipeline, '_create_orthophoto') as mock_orthophoto:
                mock_orthophoto.return_value = os.path.join(self.test_output_dir, 'orthophoto.tif')
                
                with patch.object(self.pipeline, '_segment_image') as mock_segment:
                    mock_segment.return_value = {
                        'segmentation_mask': np.random.randint(0, 5, (100, 100))
                    }
                    
                    with patch.object(self.pipeline, '_calculate_indices') as mock_calculate:
                        mock_calculate.return_value = {'GNDVI': np.random.rand(100, 100)}
                        
                        with patch.object(self.pipeline, '_assess_plant_condition') as mock_assess:
                            mock_assess.return_value = {
                                'classification': {'class': 'healthy', 'score': 0.85}
                            }
                            
                            # Запуск с оптимизацией памяти
                            result = self.pipeline.process(
                                input_path=self.test_bil_path,
                                output_dir=self.test_output_dir,
                                memory_efficient=True
                            )
                            
                            self.assertIn('orthophoto_path', result)
                            self.assertIn('segmentation_mask', result)
                            self.assertIn('indices', result)
                            self.assertIn('plant_condition', result)


if __name__ == '__main__':
    unittest.main()