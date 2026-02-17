"""
Тесты для утилит визуализации
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from src.utils.visualization import (
    visualize_indices, create_comparison_plot, plot_index_histogram,
    create_plant_condition_chart, create_processing_workflow_chart
)


class TestVisualization(unittest.TestCase):
    """Тесты утилит визуализации"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Создание тестовых данных индексов
        self.indices_dict = {
            'GNDVI': np.random.rand(100, 100) * 0.8 + 0.1,
            'NDWI': np.random.rand(100, 100) * 0.8 + 0.1,
            'MCARI': np.random.rand(100, 100) * 0.8 + 0.1
        }
        
        # Создание тестового изображения
        self.original_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Создание тестовой маски сегментации
        self.segmentation_mask = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
        
        # Создание тестовых данных состояния растений
        self.plant_condition_data = {
            'classification': {
                'class': 'Хорошее',
                'score': 0.75,
                'description': 'Растения в хорошем состоянии'
            },
            'indices': {
                'GNDVI': 0.75,
                'NDWI': 0.65,
                'MCARI': 0.70
            },
            'statistics': {
                'overall': {
                    'mean': 0.70,
                    'std': 0.05
                }
            }
        }
        
        # Создание тестовых шагов рабочего процесса
        self.workflow_steps = [
            'Загрузка данных',
            'Предварительная обработка',
            'Сегментация',
            'Расчет индексов',
            'Анализ состояния',
            'Сохранение результатов'
        ]
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_visualize_indices(self, mock_subplots, mock_show, mock_savefig):
        """Тест визуализации индексов"""
        # Настройка моков
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(len(self.indices_dict) + 1)]  # +1 для colorbar
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        output_path = os.path.join(self.temp_dir, 'indices_visualization.png')
        
        result = visualize_indices(self.indices_dict, output_path, figsize=(15, 10))
        
        # Проверка вызовов
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()
        
        # Проверка возвращаемого значения
        self.assertEqual(result, output_path)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_visualize_indices_no_output(self, mock_subplots, mock_show, mock_savefig):
        """Тест визуализации индексов без сохранения"""
        # Настройка моков
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(len(self.indices_dict) + 1)]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        result = visualize_indices(self.indices_dict, output_path=None)
        
        # Проверка вызовов
        mock_subplots.assert_called_once()
        mock_savefig.assert_not_called()
        mock_show.assert_called_once()
        
        # Проверка возвращаемого значения
        self.assertIsNone(result)
    
    def test_visualize_indices_empty_dict(self):
        """Тест визуализации пустого словаря индексов"""
        with self.assertRaises(ValueError):
            visualize_indices({})
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_create_comparison_plot(self, mock_subplots, mock_show, mock_savefig):
        """Тест создания сравнительного графика"""
        # Настройка моков
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(4)]  # Оригинал, маска, 2 индекса
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        output_path = os.path.join(self.temp_dir, 'comparison_plot.png')
        
        result = create_comparison_plot(
            self.original_image, self.segmentation_mask, 
            self.indices_dict, output_path
        )
        
        # Проверка вызовов
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()
        
        # Проверка возвращаемого значения
        self.assertEqual(result, output_path)
    
    def test_create_comparison_plot_invalid_image(self):
        """Тест создания сравнительного графика с неверным изображением"""
        invalid_image = np.array([])  # Пустой массив
        
        with self.assertRaises(ValueError):
            create_comparison_plot(
                invalid_image, self.segmentation_mask, 
                self.indices_dict, self.temp_dir
            )
    
    def test_create_comparison_plot_invalid_mask(self):
        """Тест создания сравнительного графика с неверной маской"""
        invalid_mask = np.array([])  # Пустой массив
        
        with self.assertRaises(ValueError):
            create_comparison_plot(
                self.original_image, invalid_mask, 
                self.indices_dict, self.temp_dir
            )
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_index_histogram(self, mock_subplots, mock_show, mock_savefig):
        """Тест построения гистограммы индекса"""
        # Настройка моков
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        index_data = self.indices_dict['GNDVI']
        index_name = 'GNDVI'
        output_path = os.path.join(self.temp_dir, 'histogram.png')
        
        result = plot_index_histogram(index_data, index_name, output_path, bins=50)
        
        # Проверка вызовов
        mock_subplots.assert_called_once()
        mock_ax.hist.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_ylabel.assert_called_once()
        mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()
        
        # Проверка возвращаемого значения
        self.assertEqual(result, output_path)
    
    def test_plot_index_histogram_empty_data(self):
        """Тест построения гистограммы с пустыми данными"""
        empty_data = np.array([])
        
        with self.assertRaises(ValueError):
            plot_index_histogram(empty_data, 'GNDVI')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_create_plant_condition_chart(self, mock_subplots, mock_show, mock_savefig):
        """Тест создания графика состояния растений"""
        # Настройка моков
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(2)]  # Классификация и индексы
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        output_path = os.path.join(self.temp_dir, 'plant_condition.png')
        
        result = create_plant_condition_chart(self.plant_condition_data, output_path)
        
        # Проверка вызовов
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()
        
        # Проверка возвращаемого значения
        self.assertEqual(result, output_path)
    
    def test_create_plant_condition_chart_missing_data(self):
        """Тест создания графика состояния растений с неполными данными"""
        incomplete_data = {
            'classification': {
                'class': 'Хорошее',
                'score': 0.75
            }
            # Отсутствуют индексы и статистика
        }
        
        with self.assertRaises(ValueError):
            create_plant_condition_chart(incomplete_data)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_create_processing_workflow_chart(self, mock_subplots, mock_show, mock_savefig):
        """Тест создания графика рабочего процесса"""
        # Настройка моков
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        output_path = os.path.join(self.temp_dir, 'workflow.png')
        
        result = create_processing_workflow_chart(self.workflow_steps, output_path)
        
        # Проверка вызовов
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()
        
        # Проверка возвращаемого значения
        self.assertEqual(result, output_path)
    
    def test_create_processing_workflow_chart_empty_steps(self):
        """Тест создания графика рабочего процесса с пустыми шагами"""
        with self.assertRaises(ValueError):
            create_processing_workflow_chart([])
    
    @patch('matplotlib.pyplot.savefig')
    def test_visualization_save_error(self, mock_savefig):
        """Тест обработки ошибки сохранения визуализации"""
        # Настройка мока для вызова ошибки
        mock_savefig.side_effect = Exception("Save error")
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = [MagicMock() for _ in range(len(self.indices_dict) + 1)]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            output_path = os.path.join(self.temp_dir, 'test.png')
            
            with self.assertRaises(Exception):
                visualize_indices(self.indices_dict, output_path)
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualization_with_nan_values(self, mock_subplots):
        """Тест визуализации с NaN значениями"""
        # Создание данных с NaN значениями
        indices_with_nan = {
            'GNDVI': np.random.rand(100, 100) * 0.8 + 0.1,
            'NDWI': np.random.rand(100, 100) * 0.8 + 0.1
        }
        indices_with_nan['GNDVI'][10:20, 10:20] = np.nan
        
        # Настройка моков
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(len(indices_with_nan) + 1)]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Не должно вызывать ошибок
        try:
            visualize_indices(indices_with_nan)
        except Exception as e:
            self.fail(f"Visualization with NaN values raised an exception: {e}")
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualization_with_inf_values(self, mock_subplots):
        """Тест визуализации с бесконечными значениями"""
        # Создание данных с бесконечными значениями
        indices_with_inf = {
            'GNDVI': np.random.rand(100, 100) * 0.8 + 0.1,
            'NDWI': np.random.rand(100, 100) * 0.8 + 0.1
        }
        indices_with_inf['GNDVI'][10:20, 10:20] = np.inf
        
        # Настройка моков
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(len(indices_with_inf) + 1)]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Не должно вызывать ошибок
        try:
            visualize_indices(indices_with_inf)
        except Exception as e:
            self.fail(f"Visualization with inf values raised an exception: {e}")
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualization_custom_figsize(self, mock_subplots):
        """Тест визуализации с пользовательским размером фигуры"""
        # Настройка моков
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(len(self.indices_dict) + 1)]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        custom_figsize = (20, 15)
        
        visualize_indices(self.indices_dict, figsize=custom_figsize)
        
        # Проверка вызова с правильным размером
        mock_subplots.assert_called_once()
        args, kwargs = mock_subplots.call_args
        self.assertEqual(kwargs.get('figsize'), custom_figsize)


if __name__ == '__main__':
    unittest.main()