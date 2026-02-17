#!/usr/bin/env python3
"""
Пример обработки реальных гиперспектральных данных
с использованием научной библиотеки GOP v2.0

Этот пример демонстрирует:
- Загрузку и предобработку реальных гиперспектральных данных
- Применение различных методов коррекции
- Спектральный анализ и визуализацию
- Расчет специализированных индексов
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Добавление src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.pipeline import Pipeline
from src.processing.hyperspectral import HyperspectralProcessor
from src.indices.calculator import VegetationIndexCalculator
from src.utils.logger import setup_logger


def main():
    """Основная функция примера обработки реальных данных"""
    
    # Настройка логирования
    logger = setup_logger('GOP_RealData', level=logging.INFO)
    logger.info("Начало обработки реальных гиперспектральных данных")
    
    try:
        # Путь к входным данным (замените на свой путь)
        input_path = "data/real_hyperspectral_data.bil"
        output_dir = "results/hyperspectral_real_data"
        
        # Проверка наличия входных данных
        if not os.path.exists(input_path):
            logger.error(f"Входной файл не найден: {input_path}")
            logger.info("Пожалуйста, укажите корректный путь к гиперспектральным данным")
            create_sample_data(input_path)
            logger.info(f"Созданы примеры данных: {input_path}")
        
        # Создание выходной директории
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Инициализация процессора
        logger.info("Инициализация процессора гиперспектральных данных")
        processor = HyperspectralProcessor()
        
        # Шаг 1: Анализ исходных данных
        logger.info("Шаг 1: Анализ исходных данных")
        dataset, image_data, wavelengths = processor._read_hyperspectral_data(input_path)
        data_quality = processor._analyze_data_quality(image_data)
        
        print("\n" + "="*60)
        print("АНАЛИЗ ИСХОДНЫХ ДАННЫХ")
        print("="*60)
        print(f"Размер изображения: {image_data.shape}")
        print(f"Спектральный диапазон: {np.min(wavelengths):.1f} - {np.max(wavelengths):.1f} нм")
        print(f"Количество каналов: {len(wavelengths)}")
        
        if 'overall_quality' in data_quality:
            quality = data_quality['overall_quality']
            print(f"Общая оценка качества: {quality['quality_score']:.3f}")
            print(f"Среднее SNR: {quality['average_snr']:.2f}")
        
        # Шаг 2: Сравнение методов коррекции
        logger.info("Шаг 2: Сравнение методов коррекции")
        correction_methods = ['dark_current', 'empirical_line', 'flat_field']
        corrected_data = {}
        
        for method in correction_methods:
            logger.info(f"Применение метода коррекции: {method}")
            corrected_data[method] = processor._radiometric_correction(image_data, method=method)
        
        # Шаг 3: Сравнение методов шумоподавления
        logger.info("Шаг 3: Сравнение методов шумоподавления")
        denoising_methods = ['pca', 'mnf', 'savgol']
        denoised_data = {}
        
        base_data = corrected_data['empirical_line']  # Используем лучший метод коррекции
        
        for method in denoising_methods:
            logger.info(f"Применение метода шумоподавления: {method}")
            denoised_data[method] = processor._advanced_noise_reduction(base_data, method=method)
        
        # Шаг 4: Спектральный анализ
        logger.info("Шаг 4: Спектральный анализ")
        perform_spectral_analysis(image_data, base_data, denoised_data, wavelengths, output_dir)
        
        # Шаг 5: Расчет специализированных индексов
        logger.info("Шаг 5: Расчет специализированных индексов")
        calculate_specialized_indices(denoised_data['pca'], wavelengths, output_dir)
        
        # Шаг 6: Создание RGB композитов
        logger.info("Шаг 6: Создание RGB композитов")
        create_rgb_composites(denoised_data, wavelengths, output_dir)
        
        # Шаг 7: Полная обработка через пайплайн
        logger.info("Шаг 7: Полная обработка через пайплайн")
        pipeline = Pipeline()
        pipeline_results = pipeline.process(
            input_path=input_path,
            output_dir=os.path.join(output_dir, 'pipeline_results'),
            sensor_type='Hyperspectral',
            selected_indices=['GNDVI', 'MCARI', 'MNLI', 'OSAVI', 'TVI', 'SIPI2', 'mARI', 'NDWI', 'MSI'],
            use_refinement=True,
            compression_ratio=0.125
        )
        
        # Шаг 8: Сравнение результатов
        logger.info("Шаг 8: Сравнение результатов обработки")
        compare_processing_results(pipeline_results, output_dir)
        
        print("\n" + "="*60)
        print("ОБРАБОТКА РЕАЛЬНЫХ ДАННЫХ ЗАВЕРШЕНА")
        print("="*60)
        print(f"Результаты сохранены в: {output_dir}")
        print(f"Визуализации: {output_dir}/plots/")
        print(f"Результаты пайплайна: {output_dir}/pipeline_results/")
        
    except Exception as e:
        logger.error(f"Ошибка в обработке реальных данных: {e}")
        print(f"Ошибка: {e}")
        return 1
    
    return 0


def create_sample_data(output_path: str):
    """Создание примера гиперспектральных данных для демонстрации"""
    try:
        # Создание директории
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Параметры данных
        height, width, bands = 100, 100, 150
        wavelengths = np.linspace(400, 1000, bands)
        
        # Создание синтетических гиперспектральных данных
        image_data = np.zeros((height, width, bands), dtype=np.float32)
        
        # Добавление различных спектральных сигнатур
        for i in range(height):
            for j in range(width):
                # Базовый спектр почвы
                soil_spectrum = 0.1 + 0.05 * np.sin((wavelengths - 400) / 100)
                
                # Добавление растительности в некоторых областях
                if (i - 50)**2 + (j - 50)**2 < 400:  # Центральная область
                    vegetation_spectrum = create_vegetation_spectrum(wavelengths)
                    image_data[i, j, :] = 0.7 * vegetation_spectrum + 0.3 * soil_spectrum
                else:
                    image_data[i, j, :] = soil_spectrum
                
                # Добавление шума
                image_data[i, j, :] += np.random.normal(0, 0.02, bands)
        
        # Сохранение данных
        try:
            from osgeo import gdal
            
            # Создание HDR файла
            hdr_path = output_path.replace('.bil', '.hdr')
            with open(hdr_path, 'w') as f:
                f.write(f"ENVI\n")
                f.write(f"description = {{Sample hyperspectral data}}\n")
                f.write(f"samples = {width}\n")
                f.write(f"lines = {height}\n")
                f.write(f"bands = {bands}\n")
                f.write(f"data type = 4\n")  # Float32
                f.write(f"interleave = bsq\n")
                f.write(f"byte order = 0\n")
                f.write(f"wavelength = {{")
                f.write(", ".join([f"{w:.1f}" for w in wavelengths]))
                f.write("}}\n")
            
            # Создание BIL файла
            driver = gdal.GetDriverByName('ENVI')
            dataset = driver.Create(output_path, width, height, bands, gdal.GDT_Float32)
            
            for band in range(bands):
                band_data = dataset.GetRasterBand(band + 1)
                band_data.WriteArray(image_data[:, :, band])
            
            dataset = None
            
        except ImportError:
            # Альтернативный метод сохранения
            np.save(output_path.replace('.bil', '.npy'), image_data)
            np.save(output_path.replace('.bil', '_wavelengths.npy'), wavelengths)
        
        print(f"Созданы примеры данных: {output_path}")
        
    except Exception as e:
        print(f"Ошибка создания примера данных: {e}")


def create_vegetation_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Создание спектральной сигнатуры растительности"""
    spectrum = np.zeros_like(wavelengths)
    
    # Зеленый пик (550 нм)
    green_peak = np.exp(-((wavelengths - 550) / 50) ** 2)
    
    # Красный край (700-750 нм)
    red_edge = 1 / (1 + np.exp(-(wavelengths - 725) / 15))
    
    # NIR плато (750-900 нм)
    nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.8, 0)
    
    # Комбинация компонентов
    spectrum = 0.3 * green_peak + 0.5 * red_edge + nir_plateau
    
    return spectrum


def perform_spectral_analysis(original_data: np.ndarray, 
                            corrected_data: np.ndarray, 
                            denoised_data: dict,
                            wavelengths: np.ndarray,
                            output_dir: str):
    """Выполнение спектрального анализа данных"""
    try:
        # Выбор репрезентативных пикселей
        center_y, center_x = original_data.shape[0] // 2, original_data.shape[1] // 2
        
        # Извлечение спектров из разных областей
        vegetation_pixel = corrected_data[center_y, center_x, :]
        soil_pixel = corrected_data[10, 10, :]
        
        # Создание графика спектров
        plt.figure(figsize=(12, 8))
        
        # Исходные данные
        plt.subplot(2, 2, 1)
        plt.plot(wavelengths, original_data[center_y, center_x, :], 'b-', alpha=0.7, label='Исходные')
        plt.plot(wavelengths, corrected_data[center_y, center_x, :], 'g-', label='Скорректированные')
        plt.xlabel('Длина волны (нм)')
        plt.ylabel('Отражательная способность')
        plt.title('Спектр растительности')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Сравнение методов шумоподавления
        plt.subplot(2, 2, 2)
        plt.plot(wavelengths, corrected_data[center_y, center_x, :], 'k-', linewidth=2, label='После коррекции')
        for method, data in denoised_data.items():
            plt.plot(wavelengths, data[center_y, center_x, :], '--', alpha=0.7, label=method)
        plt.xlabel('Длина волны (нм)')
        plt.ylabel('Отражательная способность')
        plt.title('Сравнение методов шумоподавления')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Спектральные профили разных объектов
        plt.subplot(2, 2, 3)
        plt.plot(wavelengths, vegetation_pixel, 'g-', label='Растительность')
        plt.plot(wavelengths, soil_pixel, 'brown', label='Почва')
        plt.xlabel('Длина волны (нм)')
        plt.ylabel('Отражательная способность')
        plt.title('Спектральные профили разных объектов')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Спектральные индексы
        plt.subplot(2, 2, 4)
        calculate_and_plot_spectral_indices(corrected_data, wavelengths, plt)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'plots', 'spectral_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Спектральный анализ сохранен: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка спектрального анализа: {e}")


def calculate_and_plot_spectral_indices(data: np.ndarray, 
                                      wavelengths: np.ndarray, 
                                      plt_module):
    """Расчет и визуализация спектральных индексов"""
    try:
        # Поиск индексов каналов
        blue_idx = np.argmin(np.abs(wavelengths - 450))
        green_idx = np.argmin(np.abs(wavelengths - 550))
        red_idx = np.argmin(np.abs(wavelengths - 650))
        nir_idx = np.argmin(np.abs(wavelengths - 800))
        
        # Извлечение каналов
        blue = data[:, :, blue_idx]
        green = data[:, :, green_idx]
        red = data[:, :, red_idx]
        nir = data[:, :, nir_idx]
        
        # Расчет индексов
        ndvi = (nir - red) / (nir + red + 1e-8)
        gndvi = (nir - green) / (nir + green + 1e-8)
        
        # Визуализация распределения индексов
        plt_module.hist(ndvi.flatten(), bins=50, alpha=0.5, label='NDVI')
        plt_module.hist(gndvi.flatten(), bins=50, alpha=0.5, label='GNDVI')
        plt_module.xlabel('Значение индекса')
        plt_module.ylabel('Частота')
        plt_module.title('Распределение вегетационных индексов')
        plt_module.legend()
        plt_module.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"Ошибка расчета спектральных индексов: {e}")


def calculate_specialized_indices(data: np.ndarray, 
                                wavelengths: np.ndarray, 
                                output_dir: str):
    """Расчет специализированных гиперспектральных индексов"""
    try:
        # Поиск индексов каналов для специфических индексов
        red_edge_idx = np.argmin(np.abs(wavelengths - 720))
        nir_idx = np.argmin(np.abs(wavelengths - 800))
        swir_idx = np.argmin(np.abs(wavelengths - 1600)) if np.max(wavelengths) > 1500 else None
        
        # Извлечение каналов
        red_edge = data[:, :, red_edge_idx]
        nir = data[:, :, nir_idx]
        
        # Расчет специализированных индексов
        indices = {}
        
        # Red Edge NDVI
        indices['RENDVI'] = (nir - red_edge) / (nir + red_edge + 1e-8)
        
        # Normalized Difference Red Edge
        indices['NDRE'] = (nir - red_edge) / (nir + red_edge + 1e-8)
        
        # Если доступен SWIR канал
        if swir_idx is not None:
            swir = data[:, :, swir_idx]
            indices['NDWI'] = (nir - swir) / (nir + swir + 1e-8)
        
        # Визуализация результатов
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, index_data) in enumerate(indices.items()):
            if i >= 4:
                break
                
            im = axes[i].imshow(index_data, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[i].set_title(f'{name}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Скрытие пустых subplot'ов
        for i in range(len(indices), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'plots', 'specialized_indices.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Специализированные индексы рассчитаны и сохранены: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка расчета специализированных индексов: {e}")


def create_rgb_composites(denoised_data: dict, 
                         wavelengths: np.ndarray, 
                         output_dir: str):
    """Создание различных RGB композитов из гиперспектральных данных"""
    try:
        # Поиск индексов каналов для RGB композитов
        blue_idx = np.argmin(np.abs(wavelengths - 450))
        green_idx = np.argmin(np.abs(wavelengths - 550))
        red_idx = np.argmin(np.abs(wavelengths - 650))
        nir_idx = np.argmin(np.abs(wavelengths - 800))
        
        # Использование данных после PCA шумоподавления
        data = denoised_data['pca']
        
        # Извлечение каналов
        blue = data[:, :, blue_idx]
        green = data[:, :, green_idx]
        red = data[:, :, red_idx]
        nir = data[:, :, nir_idx]
        
        # Создание композитов
        composites = {
            'True_Color': np.stack([red, green, blue], axis=2),
            'False_Color_NIR': np.stack([nir, red, green], axis=2),
            'NIR_RG': np.stack([nir, red, green], axis=2),
            'Red_Edge': np.stack([nir, red, data[:, :, np.argmin(np.abs(wavelengths - 720))]], axis=2)
        }
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, composite) in enumerate(composites.items()):
            # Нормализация для визуализации
            normalized = np.zeros_like(composite)
            for j in range(3):
                band = composite[:, :, j]
                band_min, band_max = np.percentile(band, [2, 98])
                if band_max > band_min:
                    normalized[:, :, j] = (band - band_min) / (band_max - band_min)
            
            axes[i].imshow(np.clip(normalized, 0, 1))
            axes[i].set_title(name.replace('_', ' '))
            axes[i].axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'plots', 'rgb_composites.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"RGB композиты созданы: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка создания RGB композитов: {e}")


def compare_processing_results(pipeline_results: dict, output_dir: str):
    """Сравнение результатов обработки через пайплайн"""
    try:
        # Извлечение результатов
        indices_results = pipeline_results.get('indices', {})
        plant_condition = pipeline_results.get('plant_condition', {})
        scientific_analysis = pipeline_results.get('scientific_analysis', {})
        
        # Создание сводного отчета
        report_path = os.path.join(output_dir, 'processing_comparison.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ОБРАБОТКИ ГИПЕРСПЕКТРАЛЬНЫХ ДАННЫХ\n")
            f.write("="*60 + "\n\n")
            
            # Информация о данных
            processed_data = pipeline_results.get('processed_data', {})
            f.write("1. ИНФОРМАЦИЯ О ДАННЫХ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Размер изображения: {processed_data.get('shape', 'N/A')}\n")
            f.write(f"Количество каналов: {processed_data.get('bands', 'N/A')}\n")
            
            wavelengths = processed_data.get('wavelengths')
            if wavelengths:
                f.write(f"Спектральный диапазон: {min(wavelengths):.1f} - {max(wavelengths):.1f} нм\n")
            
            f.write(f"Тип сенсора: {pipeline_results.get('sensor_type', 'N/A')}\n\n")
            
            # Качество данных
            data_quality = processed_data.get('data_quality', {})
            if 'overall_quality' in data_quality:
                quality = data_quality['overall_quality']
                f.write("2. КАЧЕСТВО ДАННЫХ\n")
                f.write("-" * 30 + "\n")
                f.write(f"Общая оценка качества: {quality.get('quality_score', 0):.3f}\n")
                f.write(f"Среднее SNR: {quality.get('average_snr', 0):.2f}\n\n")
            
            # Рассчитанные индексы
            calculated_indices = indices_results.get('calculated_indices', [])
            f.write("3. РАССЧИТАННЫЕ ИНДЕКСЫ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Количество индексов: {len(calculated_indices)}\n")
            f.write(f"Индексы: {', '.join(calculated_indices)}\n\n")
            
            # Состояние растений
            if 'classification' in plant_condition:
                classification = plant_condition['classification']
                f.write("4. СОСТОЯНИЕ РАСТЕНИЙ\n")
                f.write("-" * 30 + "\n")
                f.write(f"Класс: {classification.get('class', 'N/A')}\n")
                f.write(f"Описание: {classification.get('description', 'N/A')}\n")
                f.write(f"Количественная оценка: {classification.get('overall_score', 0):.3f}\n")
                f.write(f"Уверенность: {classification.get('confidence', 0):.2f}\n\n")
            
            # Научный анализ
            if 'index_statistics' in scientific_analysis:
                f.write("5. СТАТИСТИЧЕСКИЙ АНАЛИЗ\n")
                f.write("-" * 30 + "\n")
                stats = scientific_analysis['index_statistics']
                for index_name, index_stats in list(stats.items())[:5]:  # Первые 5 индексов
                    f.write(f"{index_name}:\n")
                    f.write(f"  Среднее: {index_stats.get('mean', 0):.3f}\n")
                    f.write(f"  СКО: {index_stats.get('std', 0):.3f}\n")
                    f.write(f"  Минимум: {index_stats.get('min', 0):.3f}\n")
                    f.write(f"  Максимум: {index_stats.get('max', 0):.3f}\n")
                f.write("\n")
            
            # Корреляционный анализ
            if 'correlation_analysis' in scientific_analysis:
                corr_analysis = scientific_analysis['correlation_analysis']
                if 'strong_correlations' in corr_analysis:
                    f.write("6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ\n")
                    f.write("-" * 30 + "\n")
                    strong_corr = corr_analysis['strong_correlations']
                    f.write(f"Найдено сильных корреляций: {len(strong_corr)}\n")
                    for corr in strong_corr[:5]:  # Первые 5 корреляций
                        f.write(f"  {corr['index1']} - {corr['index2']}: {corr['correlation']:.3f}\n")
                    f.write("\n")
            
            # Выводы
            f.write("7. ВЫВОДЫ\n")
            f.write("-" * 30 + "\n")
            f.write("Обработка гиперспектральных данных выполнена успешно.\n")
            f.write("Применены современные методы коррекции и шумоподавления.\n")
            f.write("Рассчитан комплекс вегетационных индексов для анализа состояния растений.\n")
            f.write("Выполнен статистический и корреляционный анализ результатов.\n")
        
        print(f"Сравнительный анализ сохранен: {report_path}")
        
    except Exception as e:
        print(f"Ошибка сравнения результатов: {e}")


if __name__ == '__main__':
    sys.exit(main())