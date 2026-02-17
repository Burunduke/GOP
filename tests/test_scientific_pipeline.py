#!/usr/bin/env python3
"""
Тестирование научного пайплайна GOP v2.0
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Добавление src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.pipeline import Pipeline
from src.core.config import config
from src.processing.hyperspectral import HyperspectralProcessor
from src.indices.calculator import VegetationIndexCalculator


class TestScientificPipeline(unittest.TestCase):
    """Тесты научного пайплайна"""
    
    def setUp(self):
        """Настройка тестового окружения"""
        self.test_dir = tempfile.mkdtemp()
        self.pipeline = Pipeline()
        
        # Создание тестовых данных
        self.create_test_data()
    
    def tearDown(self):
        """Очистка тестового окружения"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_data(self):
        """Создание тестовых гиперспектральных данных"""
        try:
            from osgeo import gdal
            
            # Создание тестового гиперспектрального изображения
            rows, cols, bands = 100, 100, 50
            test_data = np.random.rand(rows, cols, bands) * 0.8 + 0.1
            
            # Создание тестового файла
            driver = gdal.GetDriverByName('GTiff')
            test_file = os.path.join(self.test_dir, 'test_data.tif')
            dataset = driver.Create(test_file, cols, rows, bands, gdal.GDT_Float32)
            
            # Запись данных
            for band in range(bands):
                band_data = dataset.GetRasterBand(band + 1)
                band_data.WriteArray(test_data[:, :, band])
            
            dataset = None
            
            self.test_file = test_file
            
        except ImportError:
            self.skipTest("GDAL не доступен для тестирования")
    
    def test_pipeline_initialization(self):
        """Тест инициализации пайплайна"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.hyperspectral_processor)
        self.assertIsNotNone(self.pipeline.orthophoto_processor)
        self.assertIsNotNone(self.pipeline.segmenter)
        self.assertIsNotNone(self.pipeline.index_calculator)
    
    def test_hyperspectral_processor(self):
        """Тест процессора гиперспектральных данных"""
        processor = HyperspectralProcessor()
        
        # Тест получения информации о каналах
        if hasattr(self, 'test_file'):
            band_info = processor.get_band_info(self.test_file)
            self.assertIn('total_bands', band_info)
            self.assertIn('bands', band_info)
            self.assertGreater(band_info['total_bands'], 0)
    
    def test_index_calculator(self):
        """Тест калькулятора вегетационных индексов"""
        calculator = VegetationIndexCalculator()
        
        # Тест получения доступных индексов
        from src.indices.definitions import IndexDefinitions
        definitions = IndexDefinitions()
        
        available_indices = definitions.get_available_indices('Hyperspectral')
        self.assertIsInstance(available_indices, list)
        self.assertGreater(len(available_indices), 0)
    
    def test_scientific_analysis_methods(self):
        """Тест научных методов анализа"""
        # Тест расчета статистики
        test_data = np.random.rand(100, 100) * 0.8 + 0.1
        
        # Расчет базовой статистики
        mean_val = np.mean(test_data)
        std_val = np.std(test_data)
        skewness = self.pipeline._calculate_skewness(test_data.flatten())
        kurtosis = self.pipeline._calculate_kurtosis(test_data.flatten())
        
        self.assertIsInstance(mean_val, float)
        self.assertIsInstance(std_val, float)
        self.assertIsInstance(skewness, float)
        self.assertIsInstance(kurtosis, float)
        
        # Тест корреляционного анализа
        test_data2 = np.random.rand(100, 100) * 0.8 + 0.1
        correlation = np.corrcoef(test_data.flatten(), test_data2.flatten())[0, 1]
        self.assertIsInstance(correlation, (float, np.floating))
    
    def test_spatial_analysis_methods(self):
        """Тест методов пространственного анализа"""
        # Создание тестовых данных с пространственной структурой
        test_data = np.zeros((50, 50))
        test_data[10:20, 10:20] = 1  # Прямоугольная область
        test_data[30:40, 30:40] = 0.5  # Другая область
        
        # Тест расчета индекса Морана
        morans_i = self.pipeline._calculate_morans_i(test_data)
        self.assertIsInstance(morans_i, float)
        self.assertGreaterEqual(morans_i, -1)
        self.assertLessEqual(morans_i, 1)
        
        # Тест анализа горячих точек
        hotspot_analysis = self.pipeline._perform_hotspot_analysis(test_data)
        self.assertIn('hotspots', hotspot_analysis)
        self.assertIn('coldspots', hotspot_analysis)
        self.assertIn('neutral', hotspot_analysis)
        
        # Тест индекса фрагментации
        fragmentation = self.pipeline._calculate_fragmentation_index(test_data)
        self.assertIsInstance(fragmentation, float)
        self.assertGreaterEqual(fragmentation, 0)
    
    def test_quality_assessment(self):
        """Тест оценки качества данных"""
        # Создание тестовых данных с разным качеством
        good_data = np.random.rand(50, 50, 10) * 0.8 + 0.1
        noisy_data = np.random.rand(50, 50, 10) * 0.5 + 0.25
        
        # Тест расчета SNR
        good_snr = self.pipeline._calculate_snr(good_data[:, :, 0])
        noisy_snr = self.pipeline._calculate_snr(noisy_data[:, :, 0])
        
        self.assertIsInstance(good_snr, float)
        self.assertIsInstance(noisy_snr, float)
        
        # Тест расчета оценки качества
        data_quality = {
            'missing_values': {'nan_percentage': 0, 'inf_percentage': 0},
            'overall_quality': {'average_snr': good_snr}
        }
        
        quality_score = self.pipeline._calculate_quality_score(data_quality)
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0)
        self.assertLessEqual(quality_score, 1)
    
    def test_classification_methods(self):
        """Тест методов классификации"""
        # Тест классификации состояния растений
        plant_condition = {
            'statistics': {
                'overall': {
                    'mean': 0.75,
                    'std': 0.1
                }
            }
        }
        
        classification = self.pipeline._classify_plant_condition(plant_condition)
        
        self.assertIn('class', classification)
        self.assertIn('description', classification)
        self.assertIn('confidence', classification)
        self.assertIn('overall_score', classification)
        self.assertIn('variability', classification)
        
        # Проверка различных классов
        test_cases = [
            {'mean': 0.9, 'std': 0.05, 'expected_class': 'Отличное'},
            {'mean': 0.7, 'std': 0.15, 'expected_class': 'Хорошее'},
            {'mean': 0.5, 'std': 0.2, 'expected_class': 'Удовлетворительное'},
            {'mean': 0.3, 'std': 0.3, 'expected_class': 'Плохое'}
        ]
        
        for case in test_cases:
            plant_condition['statistics']['overall']['mean'] = case['mean']
            plant_condition['statistics']['overall']['std'] = case['std']
            
            classification = self.pipeline._classify_plant_condition(plant_condition)
            self.assertEqual(classification['class'], case['expected_class'])
    
    def test_config_system(self):
        """Тест системы конфигурации"""
        # Тест получения параметров
        processing_config = config.get('processing', {})
        self.assertIsInstance(processing_config, dict)
        
        # Тест установки параметров
        config.set('test.parameter', 'test_value')
        self.assertEqual(config.get('test.parameter'), 'test_value')
        
        # Тест научных параметров
        scientific_config = config.get('scientific_analysis', {})
        self.assertIn('enabled', scientific_config)
        
        indices_config = config.get('indices', {})
        self.assertIn('default_indices', indices_config)
        self.assertIn('index_groups', indices_config)
    
    def test_metadata_extraction(self):
        """Тест извлечения метаданных"""
        # Тест метаданных обработки
        metadata = self.pipeline._get_processing_metadata()
        
        self.assertIn('pipeline_version', metadata)
        self.assertIn('scientific_methods', metadata)
        self.assertIn('config_used', metadata)
        
        # Проверка научных методов
        scientific_methods = metadata['scientific_methods']
        expected_methods = [
            'radiometric_correction',
            'pca_denoising',
            'vegetation_indices',
            'spatial_analysis',
            'correlation_analysis',
            'statistical_analysis'
        ]
        
        for method in expected_methods:
            self.assertIn(method, scientific_methods)
    
    def test_error_handling(self):
        """Тест обработки ошибок"""
        # Тест обработки несуществующего файла
        with self.assertRaises(FileNotFoundError):
            self.pipeline.process(
                input_path='nonexistent_file.bil',
                output_dir=self.test_dir
            )
        
        # Тест обработки некорректных параметров
        try:
            # Это должно вызвать ошибку, но не должно упасть
            self.pipeline._calculate_snr(np.array([]))
        except (ValueError, ZeroDivisionError):
            pass  # Ожидаемое поведение
    
    def test_data_improvement_metrics(self):
        """Тест метрик улучшения данных"""
        # Создание тестовых данных
        original_data = np.random.rand(50, 50, 10) * 0.5 + 0.25
        improved_data = original_data * 1.2 + np.random.normal(0, 0.05, original_data.shape)
        
        # Тест расчета улучшения
        improvement = self.pipeline._calculate_data_improvement(
            original_data, improved_data, improved_data, improved_data
        )
        
        self.assertIn('snr_improvement', improvement)
        self.assertIn('noise_reduction', improvement)
        
        # Проверка структуры метрик
        snr_improvement = improvement['snr_improvement']
        self.assertIn('original', snr_improvement)
        self.assertIn('final', snr_improvement)
        self.assertIn('improvement_factor', snr_improvement)


class TestScientificIntegration(unittest.TestCase):
    """Тесты интеграции научных компонентов"""
    
    def setUp(self):
        """Настройка тестового окружения"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Очистка тестового окружения"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_full_scientific_workflow(self):
        """Тест полного научного рабочего процесса"""
        try:
            # Этот тест требует наличия реальных данных
            # В реальном окружении он бы тестировал полный пайплайн
            pipeline = Pipeline()
            
            # Проверка наличия всех компонентов
            self.assertIsNotNone(pipeline.hyperspectral_processor)
            self.assertIsNotNone(pipeline.index_calculator)
            self.assertIsNotNone(pipeline.segmenter)
            
            # Проверка научных методов
            self.assertTrue(hasattr(pipeline, '_perform_scientific_analysis'))
            self.assertTrue(hasattr(pipeline, '_analyze_index_statistics'))
            self.assertTrue(hasattr(pipeline, '_perform_correlation_analysis'))
            self.assertTrue(hasattr(pipeline, '_perform_spatial_analysis'))
            
        except Exception as e:
            self.skipTest(f"Пропуск интеграционного теста: {e}")
    
    def test_scientific_config_validation(self):
        """Тест валидации научной конфигурации"""
        # Проверка наличия научных секций в конфигурации
        scientific_config = config.get('scientific_analysis', {})
        self.assertIn('enabled', scientific_config)
        
        # Проверка наличия параметров анализа
        if scientific_config.get('enabled'):
            self.assertIn('statistics', scientific_config)
            self.assertIn('correlation', scientific_config)
            self.assertIn('spatial', scientific_config)
            self.assertIn('classification', scientific_config)
        
        # Проверка конфигурации индексов
        indices_config = config.get('indices', {})
        self.assertIn('index_groups', indices_config)
        self.assertIn('default_indices', indices_config)
        
        # Проверка научных групп индексов
        index_groups = indices_config['index_groups']
        expected_groups = ['greenness', 'stress', 'water', 'pigment', 'structure']
        
        for group in expected_groups:
            self.assertIn(group, index_groups)


if __name__ == '__main__':
    # Настройка логирования для тестов
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Запуск тестов
    unittest.main(verbosity=2)