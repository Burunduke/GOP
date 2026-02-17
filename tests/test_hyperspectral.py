"""
Тесты для модуля обработки гиперспектральных данных
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from src.processing.hyperspectral import HyperspectralProcessor


class TestHyperspectralProcessor(unittest.TestCase):
    """Тесты процессора гиперспектральных данных"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.processor = HyperspectralProcessor()
        self.temp_dir = tempfile.mkdtemp()
        
        # Создание тестового гиперспектрального файла
        self.test_bil_path = os.path.join(self.temp_dir, 'test.bil')
        self.test_hdr_path = os.path.join(self.temp_dir, 'test.hdr')
        
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
    
    def test_processor_initialization(self):
        """Тест инициализации процессора"""
        self.assertIsInstance(self.processor, HyperspectralProcessor)
    
    @patch('src.processing.hyperspectral.gdal.Open')
    def test_read_hyperspectral_data(self, mock_gdal_open):
        """Тест чтения гиперспектральных данных"""
        # Настройка мока
        mock_dataset = MagicMock()
        mock_dataset.RasterXSize = self.width
        mock_dataset.RasterYSize = self.height
        mock_dataset.RasterCount = self.bands
        
        # Создание моковых данных для каждого канала
        band_data = np.random.rand(self.height, self.width)
        mock_band = MagicMock()
        mock_band.ReadAsArray.return_value = band_data
        mock_dataset.GetRasterBand.return_value = mock_band
        
        # Мок для метаданных
        mock_dataset.GetMetadata.return_value = {
            'wavelength': '400, 410, 420, 430, 440, 450, 460, 470, 480, 490'
        }
        
        mock_gdal_open.return_value = mock_dataset
        
        # Вызов метода
        dataset, image_data, wavelengths = self.processor._read_hyperspectral_data(self.test_bil_path)
        
        # Проверки
        self.assertEqual(image_data.shape, (self.height, self.width, self.bands))
        self.assertIsNotNone(wavelengths)
        mock_gdal_open.assert_called_once_with(self.test_bil_path)
    
    def test_extract_wavelengths(self):
        """Тест извлечения длин волн"""
        # Создание мока датасета с метаданными
        mock_dataset = MagicMock()
        mock_dataset.GetMetadata.return_value = {
            'wavelength': '400, 410, 420, 430, 440, 450, 460, 470, 480, 490'
        }
        
        wavelengths = self.processor._extract_wavelengths(mock_dataset)
        
        self.assertIsNotNone(wavelengths)
        self.assertEqual(len(wavelengths), 10)
        self.assertEqual(wavelengths[0], 400)
        self.assertEqual(wavelengths[-1], 490)
    
    def test_extract_wavelengths_no_metadata(self):
        """Тест извлечения длин волн при отсутствии метаданных"""
        mock_dataset = MagicMock()
        mock_dataset.GetMetadata.return_value = {}
        
        wavelengths = self.processor._extract_wavelengths(mock_dataset)
        
        self.assertIsNone(wavelengths)
    
    def test_analyze_data_quality(self):
        """Тест анализа качества данных"""
        # Создание тестовых данных
        test_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        # Добавление некоторых NaN значений
        test_data[10:20, 10:20, 0] = np.nan
        
        quality = self.processor._analyze_data_quality(test_data)
        
        self.assertIn('missing_values', quality)
        self.assertIn('data_range', quality)
        self.assertIn('statistics', quality)
        
        # Проверка статистики
        stats = quality['statistics']
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
    
    def test_calculate_snr(self):
        """Тест расчета отношения сигнал/шум"""
        # Создание тестовых данных с хорошим SNR
        signal = np.ones((100, 100)) * 100
        noise = np.random.normal(0, 5, (100, 100))
        data = signal + noise
        
        snr = self.processor._calculate_snr(data)
        
        self.assertIsInstance(snr, float)
        self.assertGreater(snr, 0)
    
    def test_calculate_snr_empty_data(self):
        """Тест расчета SNR для пустых данных"""
        empty_data = np.array([])
        
        with self.assertRaises(ValueError):
            self.processor._calculate_snr(empty_data)
    
    def test_calculate_quality_score(self):
        """Тест расчета оценки качества"""
        data_quality = {
            'missing_values': {
                'nan_percentage': 5.0,
                'inf_percentage': 0.0
            },
            'overall_quality': {
                'average_snr': 20.0
            }
        }
        
        score = self.processor._calculate_quality_score(data_quality)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_radiometric_correction(self):
        """Тест радиометрической коррекции"""
        # Создание тестовых данных
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        wavelengths = np.linspace(400, 900, self.bands)
        
        corrected_data = self.processor._radiometric_correction(
            image_data, wavelengths, method='dark_current'
        )
        
        self.assertEqual(corrected_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(corrected_data)))
    
    def test_dark_current_correction(self):
        """Тест коррекции темнового тока"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        corrected_data = self.processor._dark_current_correction(image_data)
        
        self.assertEqual(corrected_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(corrected_data)))
    
    def test_empirical_line_correction(self):
        """Тест эмпирической коррекции"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        corrected_data = self.processor._empirical_line_correction(image_data)
        
        self.assertEqual(corrected_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(corrected_data)))
    
    def test_flat_field_correction(self):
        """Тест коррекции плоского поля"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        corrected_data = self.processor._flat_field_correction(image_data)
        
        self.assertEqual(corrected_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(corrected_data)))
    
    def test_atmospheric_correction(self):
        """Тест атмосферной коррекции"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        wavelengths = np.linspace(400, 900, self.bands)
        
        corrected_data = self.processor._atmospheric_correction(image_data, wavelengths)
        
        self.assertEqual(corrected_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(corrected_data)))
    
    def test_advanced_noise_reduction(self):
        """Тест продвинутого шумоподавления"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        # Добавление шума
        noise = np.random.normal(0, 0.05, image_data.shape)
        noisy_data = image_data + noise
        
        denoised_data = self.processor._advanced_noise_reduction(
            noisy_data, method='pca', n_components=0.95
        )
        
        self.assertEqual(denoised_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(denoised_data)))
    
    def test_pca_denoising(self):
        """Тест PCA шумоподавления"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        denoised_data = self.processor._pca_denoising(image_data, n_components=0.95)
        
        self.assertEqual(denoised_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(denoised_data)))
    
    def test_mnf_denoising(self):
        """Тест MNF шумоподавления"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        denoised_data = self.processor._mnf_denoising(image_data)
        
        self.assertEqual(denoised_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(denoised_data)))
    
    def test_wavelet_denoising(self):
        """Тест вейвлет-шумоподавления"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        denoised_data = self.processor._wavelet_denoising(image_data)
        
        self.assertEqual(denoised_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(denoised_data)))
    
    def test_savgol_denoising(self):
        """Тест Savgol шумоподавления"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        denoised_data = self.processor._savgol_denoising(image_data)
        
        self.assertEqual(denoised_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(denoised_data)))
    
    def test_spectral_calibration(self):
        """Тест спектральной калибровки"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        wavelengths = np.linspace(400, 900, self.bands)
        
        calibrated_data = self.processor._spectral_calibration(
            image_data, wavelengths, reference_wavelengths=np.linspace(450, 850, 25)
        )
        
        self.assertEqual(calibrated_data.shape[0], self.height)
        self.assertEqual(calibrated_data.shape[1], self.width)
        self.assertEqual(calibrated_data.shape[2], 25)  # Новое количество каналов
        self.assertFalse(np.all(np.isnan(calibrated_data)))
    
    def test_spectral_resampling(self):
        """Тест спектрального передискретизации"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        wavelengths = np.linspace(400, 900, self.bands)
        
        resampled_data = self.processor._spectral_resampling(
            image_data, wavelengths, np.linspace(450, 850, 25)
        )
        
        self.assertEqual(resampled_data.shape[0], self.height)
        self.assertEqual(resampled_data.shape[1], self.width)
        self.assertEqual(resampled_data.shape[2], 25)  # Новое количество каналов
        self.assertFalse(np.all(np.isnan(resampled_data)))
    
    def test_spectral_smoothing(self):
        """Тест спектрального сглаживания"""
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        
        smoothed_data = self.processor._spectral_smoothing(image_data)
        
        self.assertEqual(smoothed_data.shape, image_data.shape)
        self.assertFalse(np.all(np.isnan(smoothed_data)))
    
    @patch('src.processing.hyperspectral.gdal.GetDriverByName')
    def test_convert_to_tiff(self, mock_gdal_driver):
        """Тест конвертации в TIFF"""
        # Настройка мока
        mock_driver = MagicMock()
        mock_dataset = MagicMock()
        mock_driver.Create.return_value = mock_dataset
        mock_gdal_driver.return_value = mock_driver
        
        # Создание тестовых данных
        image_data = np.random.rand(self.height, self.width, self.bands) * 0.8 + 0.1
        wavelengths = np.linspace(400, 900, self.bands)
        metadata = {'sensor_type': 'Hyperspectral'}
        
        output_path = os.path.join(self.temp_dir, 'output.tif')
        
        result_path = self.processor._convert_to_tiff(
            image_data, wavelengths, metadata, output_path
        )
        
        self.assertEqual(result_path, output_path)
        mock_driver.Create.assert_called_once()
    
    def test_get_band_info(self):
        """Тест получения информации о каналах"""
        with patch('src.processing.hyperspectral.gdal.Open') as mock_gdal_open:
            # Настройка мока
            mock_dataset = MagicMock()
            mock_dataset.RasterXSize = self.width
            mock_dataset.RasterYSize = self.height
            mock_dataset.RasterCount = self.bands
            
            # Создание моковых данных для каждого канала
            band_data = np.random.rand(self.height, self.width)
            mock_band = MagicMock()
            mock_band.ReadAsArray.return_value = band_data
            mock_band.GetMinimum.return_value = 0.1
            mock_band.GetMaximum.return_value = 0.9
            mock_band.GetStatistics.return_value = (0.1, 0.9, 0.5, 0.1)
            mock_dataset.GetRasterBand.return_value = mock_band
            
            mock_gdal_open.return_value = mock_dataset
            
            # Вызов метода
            band_info = self.processor.get_band_info(self.test_bil_path)
            
            # Проверки
            self.assertIn('total_bands', band_info)
            self.assertIn('bands', band_info)
            self.assertEqual(band_info['total_bands'], self.bands)
            self.assertEqual(len(band_info['bands']), self.bands)
    
    def test_create_rgb_composite(self):
        """Тест создания RGB композита"""
        with patch('src.processing.hyperspectral.gdal.Open') as mock_gdal_open:
            # Настройка мока
            mock_dataset = MagicMock()
            mock_dataset.RasterXSize = self.width
            mock_dataset.RasterYSize = self.height
            mock_dataset.RasterCount = self.bands
            
            # Создание моковых данных для каждого канала
            band_data = np.random.rand(self.height, self.width)
            mock_band = MagicMock()
            mock_band.ReadAsArray.return_value = band_data
            mock_dataset.GetRasterBand.return_value = mock_band
            
            mock_gdal_open.return_value = mock_dataset
            
            # Вызов метода
            rgb_indices = (10, 20, 30)  # R, G, B каналы
            output_path = os.path.join(self.temp_dir, 'rgb_composite.tif')
            
            with patch.object(self.processor, '_convert_to_tiff') as mock_convert:
                mock_convert.return_value = output_path
                
                result_path = self.processor.create_rgb_composite(
                    [self.test_bil_path], rgb_indices, output_path
                )
                
                self.assertEqual(result_path, output_path)
    
    def test_process_invalid_input_path(self):
        """Тест обработки неверного пути к входному файлу"""
        with self.assertRaises(FileNotFoundError):
            self.processor.process('/nonexistent/path/file.bil', self.temp_dir)
    
    def test_process_invalid_output_dir(self):
        """Тест обработки неверной выходной директории"""
        with self.assertRaises(Exception):
            self.processor.process(
                self.test_bil_path, '/invalid/path/that/cannot/be/created'
            )


if __name__ == '__main__':
    unittest.main()