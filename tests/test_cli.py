"""
Тесты для CLI интерфейса
"""

import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from cli import cli


class TestCLI(unittest.TestCase):
    """Тесты CLI интерфейса"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Создание тестового файла
        self.test_file_path = os.path.join(self.temp_dir, 'test.bil')
        with open(self.test_file_path, 'w') as f:
            f.write('test hyperspectral data')
        
        # Создание тестовой конфигурации
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            f.write("""
processing:
  hyperspectral:
    radiometric_correction: true
    noise_reduction: true
  orthophoto:
    resolution: 0.01
segmentation:
  compression_ratio: 0.125
indices:
  default_indices: ['GNDVI', 'NDWI', 'MCARI']
""")
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """Тест вывода справки"""
        result = self.runner.invoke(cli, ['--help'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('GOP - Гиперспектральная обработка и анализ растений', result.output)
    
    def test_cli_version(self):
        """Тест вывода версии"""
        result = self.runner.invoke(cli, ['--version'])
        
        # Проверка, что команда выполнена (даже если версия не определена)
        self.assertIn(result.exit_code, [0, 1])
    
    @patch('src.core.pipeline.Pipeline')
    def test_process_command_basic(self, mock_pipeline_class):
        """Тест базовой команды обработки"""
        # Настройка мока
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_results = {
            'orthophoto_path': os.path.join(self.temp_dir, 'orthophoto.tif'),
            'segmentation_mask': os.path.join(self.temp_dir, 'mask.png'),
            'plant_condition': {
                'classification': {'class': 'Хорошее', 'score': 0.75}
            }
        }
        mock_pipeline.process.return_value = mock_results
        
        output_dir = os.path.join(self.temp_dir, 'output')
        
        result = self.runner.invoke(cli, [
            'process', self.test_file_path, output_dir
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Результаты обработки:', result.output)
        self.assertIn('Ортофотоплан:', result.output)
        self.assertIn('Маска сегментации:', result.output)
        
        # Проверка вызовов
        mock_pipeline_class.assert_called_once_with(None)
        mock_pipeline.process.assert_called_once()
    
    @patch('src.core.pipeline.Pipeline')
    def test_process_command_with_options(self, mock_pipeline_class):
        """Тест команды обработки с опциями"""
        # Настройка мока
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_results = {
            'orthophoto_path': os.path.join(self.temp_dir, 'orthophoto.tif'),
            'segmentation_mask': os.path.join(self.temp_dir, 'mask.png'),
            'plant_condition': {
                'classification': {'class': 'Хорошее', 'score': 0.75}
            }
        }
        mock_pipeline.process.return_value = mock_results
        
        output_dir = os.path.join(self.temp_dir, 'output')
        save_results = os.path.join(self.temp_dir, 'results.json')
        
        result = self.runner.invoke(cli, [
            'process', self.test_file_path, output_dir,
            '--sensor-type', 'Multispectral',
            '--indices', 'GNDVI,NDWI',
            '--compression-ratio', '0.25',
            '--no-refinement',
            '--save-results', save_results
        ])
        
        self.assertEqual(result.exit_code, 0)
        
        # Проверка вызовов
        mock_pipeline_class.assert_called_once_with(None)
        mock_pipeline.process.assert_called_once()
        
        # Проверка аргументов вызова
        args, kwargs = mock_pipeline.process.call_args
        self.assertEqual(kwargs['sensor_type'], 'Multispectral')
        self.assertEqual(kwargs['selected_indices'], ['GNDVI', 'NDWI'])
        self.assertEqual(kwargs['compression_ratio'], 0.25)
        self.assertEqual(kwargs['use_refinement'], False)
    
    @patch('src.core.pipeline.Pipeline')
    def test_process_command_with_config(self, mock_pipeline_class):
        """Тест команды обработки с конфигурацией"""
        # Настройка мока
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_results = {
            'orthophoto_path': os.path.join(self.temp_dir, 'orthophoto.tif'),
            'segmentation_mask': os.path.join(self.temp_dir, 'mask.png')
        }
        mock_pipeline.process.return_value = mock_results
        
        output_dir = os.path.join(self.temp_dir, 'output')
        
        result = self.runner.invoke(cli, [
            '--config', self.config_path,
            'process', self.test_file_path, output_dir
        ])
        
        self.assertEqual(result.exit_code, 0)
        
        # Проверка вызовов
        mock_pipeline_class.assert_called_once_with(self.config_path)
    
    @patch('src.core.pipeline.Pipeline')
    def test_process_command_error(self, mock_pipeline_class):
        """Тест обработки ошибки в команде обработки"""
        # Настройка мока для вызова ошибки
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.process.side_effect = Exception("Processing error")
        
        output_dir = os.path.join(self.temp_dir, 'output')
        
        result = self.runner.invoke(cli, [
            'process', self.test_file_path, output_dir
        ])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn('Ошибка:', result.output)
    
    def test_process_command_invalid_file(self):
        """Тест команды обработки с неверным файлом"""
        output_dir = os.path.join(self.temp_dir, 'output')
        
        result = self.runner.invoke(cli, [
            'process', '/nonexistent/file.bil', output_dir
        ])
        
        self.assertEqual(result.exit_code, 2)  # Click error code
    
    @patch('src.core.pipeline.Pipeline')
    @patch('glob.glob')
    def test_batch_command(self, mock_glob, mock_pipeline_class):
        """Тест пакетной обработки"""
        # Настройка моков
        mock_glob.return_value = [self.test_file_path]
        
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_results = {
            'orthophoto_path': os.path.join(self.temp_dir, 'orthophoto.tif'),
            'segmentation_mask': os.path.join(self.temp_dir, 'mask.png')
        }
        mock_pipeline.process.return_value = mock_results
        
        output_dir = os.path.join(self.temp_dir, 'output')
        
        result = self.runner.invoke(cli, [
            'batch', self.temp_dir, output_dir
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Найдено файлов: 1', result.output)
        self.assertIn('Успешно: 1', result.output)
        self.assertIn('С ошибками: 0', result.output)
    
    @patch('src.core.pipeline.Pipeline')
    @patch('glob.glob')
    def test_batch_command_with_errors(self, mock_glob, mock_pipeline_class):
        """Тест пакетной обработки с ошибками"""
        # Настройка моков
        mock_glob.return_value = [self.test_file_path]
        
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.process.side_effect = Exception("Processing error")
        
        output_dir = os.path.join(self.temp_dir, 'output')
        
        result = self.runner.invoke(cli, [
            'batch', self.temp_dir, output_dir
        ])
        
        self.assertEqual(result.exit_code, 0)  # Batch продолжает работу при ошибках
        self.assertIn('Успешно: 0', result.output)
        self.assertIn('С ошибками: 1', result.output)
    
    @patch('glob.glob')
    def test_batch_command_no_files(self, mock_glob):
        """Тест пакетной обработки без файлов"""
        mock_glob.return_value = []
        
        output_dir = os.path.join(self.temp_dir, 'output')
        
        result = self.runner.invoke(cli, [
            'batch', self.temp_dir, output_dir
        ])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn('Файлы не найдены', result.output)
    
    @patch('src.indices.definitions.IndexDefinitions')
    def test_list_indices_command(self, mock_index_definitions):
        """Тест команды списка индексов"""
        # Настройка мока
        mock_index_definitions.INDEX_GROUPS = {
            'greenness': ['GNDVI', 'NDVI'],
            'water': ['NDWI'],
            'stress': ['MCARI']
        }
        
        mock_index_definitions.get_index_info.side_effect = [
            {'description': 'Green NDVI', 'formula': '(NIR - Green) / (NIR + Green)', 'required_bands': ['NIR', 'Green']},
            {'description': 'Normalized Difference Vegetation Index', 'formula': '(NIR - Red) / (NIR + Red)', 'required_bands': ['NIR', 'Red']},
            {'description': 'Normalized Difference Water Index', 'formula': '(NIR - SWIR) / (NIR + SWIR)', 'required_bands': ['NIR', 'SWIR']},
            {'description': 'Modified Chlorophyll Absorption Ratio Index', 'formula': '...', 'required_bands': ['Red', 'Green', 'NIR']}
        ]
        
        result = self.runner.invoke(cli, ['list-indices'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Доступные вегетационные индексы:', result.output)
        self.assertIn('GREENNESS:', result.output)
        self.assertIn('WATER:', result.output)
        self.assertIn('STRESS:', result.output)
    
    @patch('src.core.config.config')
    def test_show_config_command(self, mock_config):
        """Тест команды показа конфигурации"""
        # Настройка мока
        mock_config.config = {
            'processing': {
                'hyperspectral': {
                    'radiometric_correction': True,
                    'noise_reduction': True
                }
            },
            'segmentation': {
                'compression_ratio': 0.125
            }
        }
        
        result = self.runner.invoke(cli, ['show-config'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Текущая конфигурация:', result.output)
        self.assertIn('processing:', result.output)
        self.assertIn('segmentation:', result.output)
    
    @patch('src.processing.hyperspectral.HyperspectralProcessor')
    def test_info_command(self, mock_processor_class):
        """Тест команды информации о файле"""
        # Настройка мока
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        mock_band_info = {
            'total_bands': 50,
            'bands': [
                {'band_number': 1, 'min': 0.1, 'max': 0.9, 'mean': 0.5, 'stddev': 0.1},
                {'band_number': 2, 'min': 0.2, 'max': 0.8, 'mean': 0.5, 'stddev': 0.1},
                {'band_number': 3, 'min': 0.15, 'max': 0.85, 'mean': 0.5, 'stddev': 0.1},
                {'band_number': 4, 'min': 0.05, 'max': 0.95, 'mean': 0.5, 'stddev': 0.1},
                {'band_number': 5, 'min': 0.1, 'max': 0.9, 'mean': 0.5, 'stddev': 0.1}
            ]
        }
        mock_processor.get_band_info.return_value = mock_band_info
        
        result = self.runner.invoke(cli, ['info', self.test_file_path])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Информация о файле:', result.output)
        self.assertIn('Всего каналов: 50', result.output)
        self.assertIn('Канал 1:', result.output)
        self.assertIn('Минимум:', result.output)
        self.assertIn('Максимум:', result.output)
        self.assertIn('Среднее:', result.output)
        self.assertIn('СКО:', result.output)
    
    @patch('src.processing.hyperspectral.HyperspectralProcessor')
    def test_info_command_error(self, mock_processor_class):
        """Тест команды информации с ошибкой"""
        # Настройка мока для вызова ошибки
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        mock_processor.get_band_info.side_effect = Exception("File read error")
        
        result = self.runner.invoke(cli, ['info', self.test_file_path])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn('Ошибка:', result.output)
    
    @patch('src.processing.hyperspectral.HyperspectralProcessor')
    def test_create_rgb_command(self, mock_processor_class):
        """Тест команды создания RGB композита"""
        # Настройка мока
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        rgb_path = os.path.join(self.temp_dir, 'rgb_composite.tif')
        mock_processor.create_rgb_composite.return_value = rgb_path
        
        result = self.runner.invoke(cli, [
            'create-rgb', self.test_file_path, rgb_path,
            '--rgb-bands', '10,20,30'
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('RGB композит создан:', result.output)
        
        # Проверка вызовов
        mock_processor.create_rgb_composite.assert_called_once()
        args, kwargs = mock_processor.create_rgb_composite.call_args
        self.assertEqual(args[1], (10, 20, 30))  # RGB bands
    
    @patch('src.processing.hyperspectral.HyperspectralProcessor')
    def test_create_rgb_command_error(self, mock_processor_class):
        """Тест команды создания RGB композита с ошибкой"""
        # Настройка мока для вызова ошибки
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        mock_processor.create_rgb_composite.side_effect = Exception("RGB creation error")
        
        rgb_path = os.path.join(self.temp_dir, 'rgb_composite.tif')
        
        result = self.runner.invoke(cli, [
            'create-rgb', self.test_file_path, rgb_path
        ])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn('Ошибка:', result.output)
    
    def test_verbose_mode(self):
        """Тест подробного режима вывода"""
        result = self.runner.invoke(cli, ['--verbose', '--help'])
        
        self.assertEqual(result.exit_code, 0)
    
    def test_quiet_mode(self):
        """Тест тихого режима вывода"""
        result = self.runner.invoke(cli, ['--quiet', '--help'])
        
        self.assertEqual(result.exit_code, 0)
    
    def test_invalid_command(self):
        """Тест неверной команды"""
        result = self.runner.invoke(cli, ['invalid-command'])
        
        self.assertEqual(result.exit_code, 2)
        self.assertIn('No such command', result.output)


if __name__ == '__main__':
    unittest.main()