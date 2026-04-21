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
    create_processing_workflow_chart,
)


class TestVisualization(unittest.TestCase):
    """Тесты утилит визуализации"""

    def setUp(self):
        """Подготовка тестовых данных"""
        self.temp_dir = tempfile.mkdtemp()

        # Создание тестового изображения
        self.original_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Создание тестовых шагов рабочего процесса
        self.workflow_steps = [
            "Загрузка данных",
            "Предварительная обработка",
            "Создание ортофото",
            "Сохранение результатов",
        ]

    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)



    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_create_processing_workflow_chart(
        self, mock_subplots, mock_show, mock_savefig
    ):
        """Тест создания графика рабочего процесса"""
        # Настройка моков
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        output_path = os.path.join(self.temp_dir, "workflow.png")

        result = create_processing_workflow_chart(self.workflow_steps, output_path)

        # Проверка вызовов
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches="tight")
        mock_show.assert_called_once()

        # Проверка возвращаемого значения
        self.assertEqual(result, output_path)

    def test_create_processing_workflow_chart_empty_steps(self):
        """Тест создания графика рабочего процесса с пустыми шагами"""
        with self.assertRaises(ValueError):
            create_processing_workflow_chart([])


if __name__ == "__main__":
    unittest.main()
