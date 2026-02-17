"""
Тесты для модуля расчета вегетационных индексов
"""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from src.indices.calculator import IndexCalculator
from src.indices.definitions import IndexDefinitions


class TestIndexCalculator(unittest.TestCase):
    """Тесты калькулятора вегетационных индексов"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.calculator = IndexCalculator()
        
        # Создание тестовых спектральных данных
        self.width, self.height, self.bands = 100, 100, 125
        self.spectral_data = np.random.rand(self.height, self.width, self.bands) * 0.5 + 0.25
        
        # Создание тестовой маски сегментации
        self.segmentation_mask = np.random.randint(0, 5, (self.height, self.width))
        
        # Временная директория для тестов
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Очистка после тестов"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_calculate_single_index(self):
        """Тест расчета одного индекса"""
        # Расчет GNDVI
        result = self.calculator.calculate(
            self.spectral_data, 
            ['GNDVI'], 
            self.segmentation_mask
        )
        
        self.assertIn('GNDVI', result)
        self.assertEqual(result['GNDVI'].shape, (self.height, self.width))
        self.assertFalse(np.all(np.isnan(result['GNDVI'])))
    
    def test_calculate_multiple_indices(self):
        """Тест расчета нескольких индексов"""
        indices = ['GNDVI', 'NDWI', 'MCARI']
        result = self.calculator.calculate(
            self.spectral_data, 
            indices, 
            self.segmentation_mask
        )
        
        for index_name in indices:
            self.assertIn(index_name, result)
            self.assertEqual(result[index_name].shape, (self.height, self.width))
            self.assertFalse(np.all(np.isnan(result[index_name])))
    
    def test_calculate_all_indices(self):
        """Тест расчета всех индексов"""
        result = self.calculator.calculate(
            self.spectral_data, 
            None,  # Все индексы
            self.segmentation_mask
        )
        
        # Проверка наличия всех индексов
        all_indices = IndexDefinitions.get_all_indices()
        for index_name in all_indices:
            self.assertIn(index_name, result)
    
    def test_assess_plant_condition(self):
        """Тест оценки состояния растений"""
        # Расчет индексов
        indices_result = self.calculator.calculate(
            self.spectral_data, 
            ['GNDVI', 'NDWI', 'MCARI'], 
            self.segmentation_mask
        )
        
        # Оценка состояния
        condition_result = self.calculator.assess_plant_condition(
            indices_result, 
            self.segmentation_mask
        )
        
        self.assertIn('classification', condition_result)
        self.assertIn('indices', condition_result)
        
        classification = condition_result['classification']
        self.assertIn('class', classification)
        self.assertIn('score', classification)
        self.assertGreaterEqual(classification['score'], 0)
        self.assertLessEqual(classification['score'], 1)
    
    def test_invalid_index(self):
        """Тест обработки неверного индекса"""
        with self.assertRaises(ValueError):
            self.calculator.calculate(
                self.spectral_data, 
                ['INVALID_INDEX'], 
                self.segmentation_mask
            )
    
    def test_empty_segmentation_mask(self):
        """Тест с пустой маской сегментации"""
        empty_mask = np.zeros((self.height, self.width), dtype=int)
        
        result = self.calculator.calculate(
            self.spectral_data, 
            ['GNDVI'], 
            empty_mask
        )
        
        self.assertIn('GNDVI', result)
        # Результат должен содержать NaN значения для пустых областей
        self.assertTrue(np.all(np.isnan(result['GNDVI'][empty_mask == 0])))
    
    def test_save_and_load_results(self):
        """Тест сохранения и загрузки результатов"""
        # Расчет индексов
        result = self.calculator.calculate(
            self.spectral_data, 
            ['GNDVI', 'NDWI'], 
            self.segmentation_mask
        )
        
        # Сохранение
        save_path = os.path.join(self.temp_dir, 'test_results.json')
        self.calculator.save_results(result, save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # Загрузка
        loaded_result = self.calculator.load_results(save_path)
        
        # Проверка
        self.assertEqual(set(result.keys()), set(loaded_result.keys()))


class TestIndexDefinitions(unittest.TestCase):
    """Тесты определений вегетационных индексов"""
    
    def test_get_index_info(self):
        """Тест получения информации об индексе"""
        info = IndexDefinitions.get_index_info('GNDVI')
        
        self.assertIn('name', info)
        self.assertIn('description', info)
        self.assertIn('formula', info)
        self.assertIn('required_bands', info)
        self.assertIn('category', info)
        
        self.assertEqual(info['name'], 'GNDVI')
        self.assertIn('NIR', info['required_bands'])
        self.assertIn('Green', info['required_bands'])
    
    def test_get_indices_by_category(self):
        """Тест получения индексов по категории"""
        greenness_indices = IndexDefinitions.get_indices_by_category('greenness')
        
        self.assertIsInstance(greenness_indices, list)
        self.assertGreater(len(greenness_indices), 0)
        self.assertIn('GNDVI', greenness_indices)
    
    def test_validate_index_requirements(self):
        """Тест валидации требований индекса"""
        # Тест с достаточными каналами
        available_bands = ['NIR', 'Green', 'Red', 'Blue']
        self.assertTrue(
            IndexDefinitions.validate_index_requirements('GNDVI', available_bands)
        )
        
        # Тест с недостаточными каналами
        insufficient_bands = ['Red', 'Blue']
        self.assertFalse(
            IndexDefinitions.validate_index_requirements('GNDVI', insufficient_bands)
        )
    
    def test_get_all_indices(self):
        """Тест получения всех индексов"""
        all_indices = IndexDefinitions.get_all_indices()
        
        self.assertIsInstance(all_indices, list)
        self.assertGreater(len(all_indices), 0)
        
        # Проверка наличия основных индексов
        expected_indices = ['GNDVI', 'NDWI', 'MCARI']
        for index in expected_indices:
            self.assertIn(index, all_indices)
    
    def test_calculate_index_with_missing_bands(self):
        """Тест расчета индекса с отсутствующими каналами"""
        # Создание данных с недостаточным количеством каналов
        insufficient_data = np.random.rand(self.height, self.width, 3)  # Только 3 канала
        
        with self.assertRaises(ValueError):
            self.calculator.calculate(
                insufficient_data,
                ['GNDVI'],  # Требует NIR и Green каналы
                self.segmentation_mask
            )
    
    def test_calculate_index_with_invalid_data_type(self):
        """Тест расчета индекса с неверным типом данных"""
        # Создание данных с неверным типом
        invalid_data = np.random.randint(0, 256, (self.height, self.width, self.bands))
        
        # Должно работать, но с преобразованием типа
        result = self.calculator.calculate(
            invalid_data.astype(np.uint8),
            ['GNDVI'],
            self.segmentation_mask
        )
        
        self.assertIn('GNDVI', result)
    
    def test_assess_plant_condition_with_empty_indices(self):
        """Тест оценки состояния растений с пустыми индексами"""
        empty_indices = {}
        
        with self.assertRaises(ValueError):
            self.calculator.assess_plant_condition(empty_indices, self.segmentation_mask)
    
    def test_save_results_with_invalid_path(self):
        """Тест сохранения результатов с неверным путем"""
        result = self.calculator.calculate(
            self.spectral_data,
            ['GNDVI'],
            self.segmentation_mask
        )
        
        invalid_path = '/invalid/path/that/cannot/be/created/results.json'
        
        with self.assertRaises(Exception):
            self.calculator.save_results(result, invalid_path)
    
    def test_load_results_with_invalid_file(self):
        """Тест загрузки результатов из неверного файла"""
        invalid_path = '/nonexistent/path/results.json'
        
        with self.assertRaises(FileNotFoundError):
            self.calculator.load_results(invalid_path)
    
    def test_load_results_with_invalid_json(self):
        """Тест загрузки результатов из невалидного JSON файла"""
        invalid_json_path = os.path.join(self.temp_dir, 'invalid.json')
        
        # Создание файла с невалидным JSON
        with open(invalid_json_path, 'w') as f:
            f.write('{ invalid json }')
        
        with self.assertRaises(json.JSONDecodeError):
            self.calculator.load_results(invalid_json_path)
    
    def test_get_index_formula(self):
        """Тест получения формулы индекса"""
        formula = IndexDefinitions.get_index_formula('GNDVI')
        
        self.assertIsInstance(formula, str)
        self.assertIn('NIR', formula)
        self.assertIn('Green', formula)
    
    def test_get_index_description(self):
        """Тест получения описания индекса"""
        description = IndexDefinitions.get_index_description('GNDVI')
        
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)
    
    def test_get_index_info_nonexistent(self):
        """Тест получения информации о несуществующем индексе"""
        with self.assertRaises(KeyError):
            IndexDefinitions.get_index_info('NONEXISTENT_INDEX')
    
    def test_get_indices_by_group_nonexistent(self):
        """Тест получения индексов по несуществующей группе"""
        result = IndexDefinitions.get_indices_by_group('nonexistent_group')
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)
    
    def test_normalize_index(self):
        """Тест нормализации индекса"""
        # Создание тестовых данных
        test_data = np.random.rand(50, 50) * 0.8 + 0.1
        
        # Добавление NaN значений
        test_data[10:20, 10:20] = np.nan
        
        normalized = IndexDefinitions.normalize_index(test_data)
        
        # Проверка, что результат не содержит NaN (в допустимых областях)
        valid_mask = ~np.isnan(test_data)
        self.assertFalse(np.any(np.isnan(normalized[valid_mask])))
        
        # Проверка диапазона
        self.assertGreaterEqual(normalized.min(), 0)
        self.assertLessEqual(normalized.max(), 1)
    
    def test_calculate_index_edge_cases(self):
        """Тест расчета индекса в граничных случаях"""
        # Создание данных с нулевыми значениями
        zero_data = np.zeros((self.height, self.width, self.bands))
        
        result = self.calculator.calculate(
            zero_data,
            ['GNDVI'],
            self.segmentation_mask
        )
        
        self.assertIn('GNDVI', result)
        
        # Создание данных с очень большими значениями
        large_data = np.full((self.height, self.width, self.bands), 1e6)
        
        result = self.calculator.calculate(
            large_data,
            ['GNDVI'],
            self.segmentation_mask
        )
        
        self.assertIn('GNDVI', result)
    
    @patch('src.indices.calculator.np.savez')
    def test_save_results_numpy_format(self, mock_save):
        """Тест сохранения результатов в формате numpy"""
        result = self.calculator.calculate(
            self.spectral_data,
            ['GNDVI'],
            self.segmentation_mask
        )
        
        save_path = os.path.join(self.temp_dir, 'results.npz')
        self.calculator.save_results(result, save_path, format='numpy')
        
        mock_save.assert_called_once()
    
    def test_assess_plant_condition_different_masks(self):
        """Тест оценки состояния растений с разными масками"""
        # Создание маски с одним классом
        single_class_mask = np.ones((self.height, self.width), dtype=int)
        
        indices_result = self.calculator.calculate(
            self.spectral_data,
            ['GNDVI', 'NDWI'],
            single_class_mask
        )
        
        condition_result = self.calculator.assess_plant_condition(
            indices_result,
            single_class_mask
        )
        
        self.assertIn('classification', condition_result)
        self.assertIn('indices', condition_result)


if __name__ == '__main__':
    unittest.main()