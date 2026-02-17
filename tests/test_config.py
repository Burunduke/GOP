"""
Тесты для модуля конфигурации
"""

import unittest
import tempfile
import os
import yaml
from src.core.config import Config


class TestConfig(unittest.TestCase):
    """Тесты класса конфигурации"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Создание тестовой конфигурации
        self.test_config = {
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
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Очистка после тестов"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_initialization_with_path(self):
        """Тест инициализации конфигурации с путем к файлу"""
        config = Config(self.test_config_path)
        
        self.assertEqual(config.config['processing']['hyperspectral']['radiometric_correction'], True)
        self.assertEqual(config.config['segmentation']['compression_ratio'], 0.125)
        self.assertEqual(config.config['indices']['default_indices'], ['GNDVI', 'NDWI', 'MCARI'])
    
    def test_config_initialization_without_path(self):
        """Тест инициализации конфигурации без пути (используется конфигурация по умолчанию)"""
        config = Config()
        
        # Проверка наличия основных секций
        self.assertIn('processing', config.config)
        self.assertIn('segmentation', config.config)
        self.assertIn('indices', config.config)
        self.assertIn('output', config.config)
    
    def test_get_method(self):
        """Тест метода get"""
        config = Config(self.test_config_path)
        
        # Тест получения существующего значения
        self.assertEqual(config.get('processing.hyperspectral.radiometric_correction'), True)
        
        # Тест получения значения по умолчанию
        self.assertEqual(config.get('nonexistent.key', 'default_value'), 'default_value')
        
        # Тест получения секции
        processing_config = config.get('processing')
        self.assertIsInstance(processing_config, dict)
        self.assertIn('hyperspectral', processing_config)
    
    def test_set_method(self):
        """Тест метода set"""
        config = Config(self.test_config_path)
        
        # Тест установки нового значения
        config.set('test.new_parameter', 'test_value')
        self.assertEqual(config.get('test.new_parameter'), 'test_value')
        
        # Тест изменения существующего значения
        config.set('processing.hyperspectral.radiometric_correction', False)
        self.assertEqual(config.get('processing.hyperspectral.radiometric_correction'), False)
    
    def test_save_method(self):
        """Тест метода save"""
        config = Config(self.test_config_path)
        
        # Изменение конфигурации
        config.set('test.parameter', 'test_value')
        
        # Сохранение в новый файл
        save_path = os.path.join(self.temp_dir, 'saved_config.yaml')
        config.save(save_path)
        
        # Проверка сохранения
        self.assertTrue(os.path.exists(save_path))
        
        # Загрузка и проверка
        with open(save_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        self.assertEqual(saved_config['test']['parameter'], 'test_value')
    
    def test_update_method(self):
        """Тест метода update"""
        config = Config(self.test_config_path)
        
        # Обновление конфигурации
        update_dict = {
            'new_section': {
                'parameter1': 'value1',
                'parameter2': 'value2'
            },
            'processing': {
                'hyperspectral': {
                    'new_parameter': 'new_value'
                }
            }
        }
        
        config.update(update_dict)
        
        # Проверка обновлений
        self.assertEqual(config.get('new_section.parameter1'), 'value1')
        self.assertEqual(config.get('processing.hyperspectral.new_parameter'), 'new_value')
        
        # Проверка сохранения существующих значений
        self.assertEqual(config.get('processing.hyperspectral.radiometric_correction'), True)
    
    def test_deep_update_method(self):
        """Тест метода _deep_update"""
        config = Config(self.test_config_path)
        
        base_dict = {
            'section1': {
                'subsection1': {
                    'param1': 'value1',
                    'param2': 'value2'
                },
                'subsection2': {
                    'param3': 'value3'
                }
            },
            'section2': {
                'param4': 'value4'
            }
        }
        
        update_dict = {
            'section1': {
                'subsection1': {
                    'param2': 'updated_value2',
                    'param5': 'value5'
                },
                'subsection3': {
                    'param6': 'value6'
                }
            },
            'section3': {
                'param7': 'value7'
            }
        }
        
        config._deep_update(base_dict, update_dict)
        
        # Проверка результатов
        self.assertEqual(base_dict['section1']['subsection1']['param1'], 'value1')  # сохранено
        self.assertEqual(base_dict['section1']['subsection1']['param2'], 'updated_value2')  # обновлено
        self.assertEqual(base_dict['section1']['subsection1']['param5'], 'value5')  # добавлено
        self.assertEqual(base_dict['section1']['subsection2']['param3'], 'value3')  # сохранено
        self.assertEqual(base_dict['section1']['subsection3']['param6'], 'value6')  # добавлено
        self.assertEqual(base_dict['section2']['param4'], 'value4')  # сохранено
        self.assertEqual(base_dict['section3']['param7'], 'value7')  # добавлено
    
    def test_invalid_config_path(self):
        """Тест обработки неверного пути к конфигурации"""
        with self.assertRaises(FileNotFoundError):
            Config('/nonexistent/path/config.yaml')
    
    def test_invalid_yaml_config(self):
        """Тест обработки некорректного YAML файла"""
        invalid_config_path = os.path.join(self.temp_dir, 'invalid_config.yaml')
        
        # Создание некорректного YAML файла
        with open(invalid_config_path, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        with self.assertRaises(yaml.YAMLError):
            Config(invalid_config_path)
    
    def test_config_property(self):
        """Тест свойства config"""
        config = Config(self.test_config_path)
        
        # Проверка, что свойство возвращает словарь
        self.assertIsInstance(config.config, dict)
        
        # Проверка, что изменение возвращенного словаря не влияет на оригинал
        config_dict = config.config
        config_dict['test'] = 'value'
        
        self.assertNotIn('test', config.config)


if __name__ == '__main__':
    unittest.main()