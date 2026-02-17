"""
Тесты для модуля сегментации изображений
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from src.segmentation.segmenter import ImageSegmenter


class TestImageSegmenter(unittest.TestCase):
    """Тесты сегментатора изображений"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.segmenter = ImageSegmenter()
        self.temp_dir = tempfile.mkdtemp()
        
        # Создание тестового изображения
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.png')
        self.test_mask_path = os.path.join(self.temp_dir, 'test_mask.png')
        
        # Создание простого тестового изображения
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self._save_test_image(test_image, self.test_image_path)
        
        # Создание тестовой маски
        test_mask = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
        self._save_test_image(test_mask, self.test_mask_path)
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _save_test_image(self, image_array, path):
        """Вспомогательный метод для сохранения тестового изображения"""
        try:
            import cv2
            cv2.imwrite(path, image_array)
        except ImportError:
            # Если cv2 недоступен, создаем пустой файл
            with open(path, 'wb') as f:
                f.write(b'\x00' * 1000)  # Пустой файл для тестов
    
    def test_segmenter_initialization(self):
        """Тест инициализации сегментатора"""
        self.assertIsInstance(self.segmenter, ImageSegmenter)
    
    @patch.object(ImageSegmenter, '_preliminary_segmentation')
    @patch.object(ImageSegmenter, '_refine_segmentation')
    @patch.object(ImageSegmenter, '_select_best_mask')
    @patch.object(ImageSegmenter, '_read_and_compress_image')
    def test_segment_with_refinement(self, mock_read, mock_select, mock_refine, mock_prelim):
        """Тест сегментации с уточнением"""
        # Настройка моков
        mock_read.return_value = np.random.rand(50, 50, 3)
        mock_prelim.return_value = np.random.randint(0, 5, (50, 50))
        mock_refine.return_value = np.random.randint(0, 5, (50, 50))
        mock_select.return_value = np.random.randint(0, 5, (50, 50))
        
        result = self.segmenter.segment(
            self.test_image_path, 
            self.temp_dir, 
            compression_ratio=0.125,
            use_refinement=True
        )
        
        self.assertIn('segmentation_mask', result)
        self.assertIn('mask_path', result)
        mock_prelim.assert_called_once()
        mock_refine.assert_called_once()
        mock_select.assert_called_once()
    
    @patch.object(ImageSegmenter, '_preliminary_segmentation')
    @patch.object(ImageSegmenter, '_read_and_compress_image')
    def test_segment_without_refinement(self, mock_read, mock_prelim):
        """Тест сегментации без уточнения"""
        # Настройка моков
        mock_read.return_value = np.random.rand(50, 50, 3)
        mock_prelim.return_value = np.random.randint(0, 5, (50, 50))
        
        result = self.segmenter.segment(
            self.test_image_path, 
            self.temp_dir, 
            compression_ratio=0.125,
            use_refinement=False
        )
        
        self.assertIn('segmentation_mask', result)
        self.assertIn('mask_path', result)
        mock_prelim.assert_called_once()
    
    @patch('torch.hub.load')
    def test_preliminary_segmentation(self, mock_torch_hub):
        """Тест предварительной сегментации"""
        # Настройка мока
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_torch_hub.return_value = mock_model
        
        # Создание тестовых данных
        image_data = np.random.rand(100, 100, 3)
        
        with patch.object(self.segmenter, '_simple_segmentation') as mock_simple:
            mock_simple.return_value = np.random.randint(0, 5, (100, 100))
            
            result = self.segmenter._preliminary_segmentation(
                image_data, 'deeplabv3plus', self.temp_dir
            )
            
            self.assertEqual(result.shape, (100, 100))
            mock_simple.assert_called_once_with(image_data)
    
    @patch('torch.hub.load')
    def test_refine_segmentation(self, mock_torch_hub):
        """Тест уточнения сегментации"""
        # Настройка мока
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_torch_hub.return_value = mock_model
        
        # Создание тестовых данных
        image_data = np.random.rand(100, 100, 3)
        initial_mask = np.random.randint(0, 5, (100, 100))
        
        with patch.object(self.segmenter, '_simple_refinement') as mock_simple:
            mock_simple.return_value = np.random.randint(0, 5, (100, 100))
            
            result = self.segmenter._refine_segmentation(
                image_data, initial_mask, 'cascade_psp', self.temp_dir
            )
            
            self.assertEqual(result.shape, (100, 100))
            mock_simple.assert_called_once_with(image_data, initial_mask)
    
    def test_select_best_mask(self):
        """Тест выбора лучшей маски"""
        # Создание тестовых масок
        mask1 = np.random.randint(0, 5, (100, 100))
        mask2 = np.random.randint(0, 5, (100, 100))
        
        masks = [mask1, mask2]
        
        with patch.object(self.segmenter, '_evaluate_mask_quality') as mock_evaluate:
            mock_evaluate.side_effect = [0.7, 0.9]  # Вторая маска лучше
            
            result = self.segmenter._select_best_mask(masks, self.temp_dir)
            
            self.assertTrue(np.array_equal(result, mask2))
            self.assertEqual(mock_evaluate.call_count, 2)
    
    def test_select_best_mask_empty_list(self):
        """Тест выбора лучшей маски из пустого списка"""
        with self.assertRaises(ValueError):
            self.segmenter._select_best_mask([], self.temp_dir)
    
    @patch('cv2.imread')
    def test_read_and_compress_image(self, mock_imread):
        """Тест чтения и сжатия изображения"""
        # Настройка мока
        original_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        mock_imread.return_value = original_image
        
        result = self.segmenter._read_and_compress_image(
            self.test_image_path, compression_ratio=0.125
        )
        
        # Проверка, что изображение было сжато
        self.assertLess(result.shape[0], original_image.shape[0])
        self.assertLess(result.shape[1], original_image.shape[1])
        mock_imread.assert_called_once_with(self.test_image_path)
    
    @patch('cv2.imread')
    def test_read_image(self, mock_imread):
        """Тест чтения изображения"""
        # Настройка мока
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = test_image
        
        result = self.segmenter._read_image(self.test_image_path)
        
        self.assertTrue(np.array_equal(result, test_image))
        mock_imread.assert_called_once_with(self.test_image_path)
    
    @patch('cv2.imread')
    def test_read_image_file_not_found(self, mock_imread):
        """Тест чтения несуществующего изображения"""
        mock_imread.return_value = None
        
        with self.assertRaises(FileNotFoundError):
            self.segmenter._read_image('/nonexistent/path/image.png')
    
    @patch('cv2.imread')
    def test_read_mask(self, mock_imread):
        """Тест чтения маски"""
        # Настройка мока
        test_mask = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
        mock_imread.return_value = test_mask
        
        result = self.segmenter._read_mask(self.test_mask_path)
        
        self.assertTrue(np.array_equal(result, test_mask))
        mock_imread.assert_called_once_with(self.test_mask_path, cv2.IMREAD_GRAYSCALE)
    
    def test_resize_mask(self):
        """Тест изменения размера маски"""
        # Создание тестовой маски
        mask_data = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
        target_shape = (50, 50)
        
        with patch('cv2.resize') as mock_resize:
            mock_resize.return_value = np.random.randint(0, 5, target_shape, dtype=np.uint8)
            
            result = self.segmenter._resize_mask(mask_data, target_shape)
            
            self.assertEqual(result.shape, target_shape)
            mock_resize.assert_called_once()
    
    @patch('cv2.imwrite')
    def test_save_mask(self, mock_imwrite):
        """Тест сохранения маски"""
        # Создание тестовой маски
        mask_data = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
        output_path = os.path.join(self.temp_dir, 'output_mask.png')
        
        mock_imwrite.return_value = True
        
        result = self.segmenter._save_mask(mask_data, output_path)
        
        self.assertEqual(result, output_path)
        mock_imwrite.assert_called_once()
    
    @patch('cv2.imwrite')
    def test_save_mask_failure(self, mock_imwrite):
        """Тест обработки ошибки при сохранении маски"""
        # Создание тестовой маски
        mask_data = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
        output_path = os.path.join(self.temp_dir, 'output_mask.png')
        
        mock_imwrite.return_value = False
        
        with self.assertRaises(Exception):
            self.segmenter._save_mask(mask_data, output_path)
    
    def test_evaluate_mask_quality(self):
        """Тест оценки качества маски"""
        # Создание тестовой маски
        mask_data = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
        
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = mask_data
            
            quality_score = self.segmenter._evaluate_mask_quality(self.test_mask_path)
            
            self.assertIsInstance(quality_score, float)
            self.assertGreaterEqual(quality_score, 0)
            self.assertLessEqual(quality_score, 1)
    
    def test_copy_file(self):
        """Тест копирования файла"""
        src_path = self.test_image_path
        dst_path = os.path.join(self.temp_dir, 'copied_image.png')
        
        self.segmenter._copy_file(src_path, dst_path)
        
        self.assertTrue(os.path.exists(dst_path))
    
    @patch.object(ImageSegmenter, 'segment')
    def test_segment_batch(self, mock_segment):
        """Тест пакетной сегментации"""
        # Настройка мока
        mock_segment.return_value = {
            'segmentation_mask': np.random.randint(0, 5, (50, 50)),
            'mask_path': os.path.join(self.temp_dir, 'mask.png')
        }
        
        # Создание тестовых файлов
        test_files = [self.test_image_path]
        
        results = self.segmenter.segment_batch(
            test_files, self.temp_dir, compression_ratio=0.125
        )
        
        self.assertEqual(len(results), 1)
        self.assertIn(self.test_image_path, results)
        mock_segment.assert_called_once()
    
    def test_simple_segmentation(self):
        """Тест простой сегментации"""
        # Создание тестовых данных с четкими границами
        image_data = np.zeros((100, 100, 3), dtype=np.uint8)
        image_data[25:75, 25:75] = 255  # Белый квадрат в центре
        
        result = self.segmenter._simple_segmentation(image_data)
        
        self.assertEqual(result.shape, (100, 100))
        self.assertGreater(np.unique(result).size, 1)  # Должно быть более одного класса
    
    def test_simple_refinement(self):
        """Тест простого уточнения"""
        # Создание тестовых данных
        image_data = np.random.rand(100, 100, 3)
        mask_data = np.random.randint(0, 5, (100, 100))
        
        result = self.segmenter._simple_refinement(image_data, mask_data)
        
        self.assertEqual(result.shape, mask_data.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < 5))
    
    def test_segment_invalid_image_path(self):
        """Тест сегментации с неверным путем к изображению"""
        with self.assertRaises(FileNotFoundError):
            self.segmenter.segment('/nonexistent/path/image.png', self.temp_dir)
    
    def test_segment_invalid_output_dir(self):
        """Тест сегментации с неверной выходной директорией"""
        with self.assertRaises(Exception):
            self.segmenter.segment(
                self.test_image_path, '/invalid/path/that/cannot/be/created'
            )
    
    def test_segment_invalid_compression_ratio(self):
        """Тест сегментации с неверным коэффициентом сжатия"""
        with self.assertRaises(ValueError):
            self.segmenter.segment(
                self.test_image_path, self.temp_dir, compression_ratio=2.0
            )
        
        with self.assertRaises(ValueError):
            self.segmenter.segment(
                self.test_image_path, self.temp_dir, compression_ratio=-0.1
            )


if __name__ == '__main__':
    unittest.main()