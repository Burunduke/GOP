#!/usr/bin/env python3
"""
Примеры различных сценариев использования проекта GOP
Демонстрация практического применения в различных областях

Этот пример показывает:
- Мониторинг сельскохозяйственных культур
- Анализ лесных массивов
- Оценку состояния водных ресурсов
- Экологический мониторинг
- Урбанистический анализ
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
from src.segmentation.segmenter import ImageSegmenter
from src.utils.logger import setup_logger


def main():
    """Основная функция примеров сценариев использования"""
    
    # Настройка логирования
    logger = setup_logger('GOP_Scenarios', level=logging.INFO)
    logger.info("Начало демонстрации сценариев использования GOP")
    
    try:
        # Создание выходной директории
        output_dir = "results/use_case_scenarios"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Сценарий 1: Мониторинг сельскохозяйственных культур
        logger.info("Сценарий 1: Мониторинг сельскохозяйственных культур")
        agriculture_monitoring_scenario(output_dir)
        
        # Сценарий 2: Анализ лесных массивов
        logger.info("Сценарий 2: Анализ лесных массивов")
        forest_analysis_scenario(output_dir)
        
        # Сценарий 3: Оценка состояния водных ресурсов
        logger.info("Сценарий 3: Оценка состояния водных ресурсов")
        water_resources_scenario(output_dir)
        
        # Сценарий 4: Экологический мониторинг
        logger.info("Сценарий 4: Экологический мониторинг")
        environmental_monitoring_scenario(output_dir)
        
        # Сценарий 5: Урбанистический анализ
        logger.info("Сценарий 5: Урбанистический анализ")
        urban_analysis_scenario(output_dir)
        
        # Сценарий 6: Комплексный мониторинг экосистем
        logger.info("Сценарий 6: Комплексный мониторинг экосистем")
        ecosystem_monitoring_scenario(output_dir)
        
        print("\n" + "="*60)
        print("СЦЕНАРИИ ИСПОЛЬЗОВАНИЯ ЗАВЕРШЕНЫ")
        print("="*60)
        print(f"Результаты сохранены в: {output_dir}")
        
    except Exception as e:
        logger.error(f"Ошибка в сценариях использования: {e}")
        print(f"Ошибка: {e}")
        return 1
    
    return 0


def agriculture_monitoring_scenario(output_dir: str):
    """Сценарий: Мониторинг сельскохозяйственных культур"""
    try:
        print("\n--- Сценарий 1: Мониторинг сельскохозяйственных культур ---")
        
        # Создание данных для сельскохозяйственного сценария
        data_path = os.path.join(output_dir, 'agriculture_data.bil')
        create_agriculture_data(data_path)
        
        # Инициализация пайплайна
        pipeline = Pipeline()
        
        # Обработка данных
        results = pipeline.process(
            input_path=data_path,
            output_dir=os.path.join(output_dir, 'agriculture_results'),
            sensor_type='Multispectral',
            selected_indices=['NDVI', 'GNDVI', 'OSAVI', 'TVI', 'NDWI'],
            use_refinement=True,
            compression_ratio=0.125
        )
        
        # Анализ результатов для сельского хозяйства
        analyze_agriculture_results(results, output_dir)
        
        print("Мониторинг сельскохозяйственных культур завершен")
        
    except Exception as e:
        print(f"Ошибка в сценарии сельского хозяйства: {e}")


def forest_analysis_scenario(output_dir: str):
    """Сценарий: Анализ лесных массивов"""
    try:
        print("\n--- Сценарий 2: Анализ лесных массивов ---")
        
        # Создание данных для лесного сценария
        data_path = os.path.join(output_dir, 'forest_data.bil')
        create_forest_data(data_path)
        
        # Инициализация процессора
        processor = HyperspectralProcessor()
        calculator = VegetationIndexCalculator()
        
        # Обработка данных
        processed_data = processor.process(data_path, os.path.join(output_dir, 'forest_processed'))
        
        # Расчет лесных индексов
        forest_indices = calculate_forest_indices(processed_data, output_dir)
        
        # Анализ состояния леса
        analyze_forest_health(forest_indices, output_dir)
        
        print("Анализ лесных массивов завершен")
        
    except Exception as e:
        print(f"Ошибка в сценарии лесного анализа: {e}")


def water_resources_scenario(output_dir: str):
    """Сценарий: Оценка состояния водных ресурсов"""
    try:
        print("\n--- Сценарий 3: Оценка состояния водных ресурсов ---")
        
        # Создание данных для водного сценария
        data_path = os.path.join(output_dir, 'water_data.bil')
        create_water_data(data_path)
        
        # Инициализация компонентов
        processor = HyperspectralProcessor()
        calculator = VegetationIndexCalculator()
        
        # Обработка данных
        processed_data = processor.process(data_path, os.path.join(output_dir, 'water_processed'))
        
        # Расчет водных индексов
        water_indices = calculate_water_indices(processed_data, output_dir)
        
        # Анализ качества воды
        analyze_water_quality(water_indices, output_dir)
        
        print("Оценка состояния водных ресурсов завершена")
        
    except Exception as e:
        print(f"Ошибка в сценарии водных ресурсов: {e}")


def environmental_monitoring_scenario(output_dir: str):
    """Сценарий: Экологический мониторинг"""
    try:
        print("\n--- Сценарий 4: Экологический мониторинг ---")
        
        # Создание данных для экологического сценария
        data_path = os.path.join(output_dir, 'environmental_data.bil')
        create_environmental_data(data_path)
        
        # Инициализация пайплайна
        pipeline = Pipeline()
        
        # Обработка данных
        results = pipeline.process(
            input_path=data_path,
            output_dir=os.path.join(output_dir, 'environmental_results'),
            sensor_type='Hyperspectral',
            selected_indices=['NDVI', 'GNDVI', 'MCARI', 'SIPI2', 'mARI', 'NDWI', 'MSI'],
            use_refinement=True,
            compression_ratio=0.125
        )
        
        # Экологический анализ
        perform_environmental_analysis(results, output_dir)
        
        print("Экологический мониторинг завершен")
        
    except Exception as e:
        print(f"Ошибка в сценарии экологического мониторинга: {e}")


def urban_analysis_scenario(output_dir: str):
    """Сценарий: Урбанистический анализ"""
    try:
        print("\n--- Сценарий 5: Урбанистический анализ ---")
        
        # Создание данных для урбанистического сценария
        data_path = os.path.join(output_dir, 'urban_data.bil')
        create_urban_data(data_path)
        
        # Инициализация компонентов
        processor = HyperspectralProcessor()
        calculator = VegetationIndexCalculator()
        segmenter = ImageSegmenter()
        
        # Обработка данных
        processed_data = processor.process(data_path, os.path.join(output_dir, 'urban_processed'))
        
        # Сегментация городских территорий
        orthophoto_path = processed_data['tiff_paths'][0] if processed_data['tiff_paths'] else data_path
        segmentation_mask = segmenter.segment(
            orthophoto_path, 
            output_dir=os.path.join(output_dir, 'urban_segmentation'),
            use_refinement=True
        )
        
        # Расчет урбанистических индексов
        urban_indices = calculate_urban_indices(processed_data, segmentation_mask, output_dir)
        
        # Анализ урбанизации
        analyze_urban_development(urban_indices, output_dir)
        
        print("Урбанистический анализ завершен")
        
    except Exception as e:
        print(f"Ошибка в сценарии урбанистического анализа: {e}")


def ecosystem_monitoring_scenario(output_dir: str):
    """Сценарий: Комплексный мониторинг экосистем"""
    try:
        print("\n--- Сценарий 6: Комплексный мониторинг экосистем ---")
        
        # Создание комплексных данных экосистемы
        data_path = os.path.join(output_dir, 'ecosystem_data.bil')
        create_ecosystem_data(data_path)
        
        # Инициализация пайплайна
        pipeline = Pipeline()
        
        # Комплексная обработка
        results = pipeline.process(
            input_path=data_path,
            output_dir=os.path.join(output_dir, 'ecosystem_results'),
            sensor_type='Hyperspectral',
            selected_indices=['NDVI', 'GNDVI', 'MCARI', 'MNLI', 'OSAVI', 'TVI', 'SIPI2', 'mARI', 'NDWI', 'MSI'],
            use_refinement=True,
            compression_ratio=0.125
        )
        
        # Комплексный анализ экосистемы
        perform_ecosystem_analysis(results, output_dir)
        
        # Создание карты биоразнообразия
        create_biodiversity_map(results, output_dir)
        
        print("Комплексный мониторинг экосистем завершен")
        
    except Exception as e:
        print(f"Ошибка в сценарии экосистемного мониторинга: {e}")


# Функции создания данных для различных сценариев

def create_agriculture_data(output_path: str):
    """Создание данных для сельскохозяйственного сценария"""
    try:
        height, width, bands = 150, 150, 10
        wavelengths = np.linspace(450, 900, bands)
        
        image_data = np.zeros((height, width, bands), dtype=np.float32)
        
        # Создание полей с разными культурами
        for i in range(height):
            for j in range(width):
                if i < 50:  # Пшеница
                    spectrum = create_crop_spectrum(wavelengths, crop_type='wheat', health='healthy')
                elif i < 100:  # Кукуруза
                    if j < 75:  # Здоровая кукуруза
                        spectrum = create_crop_spectrum(wavelengths, crop_type='corn', health='healthy')
                    else:  # Стрессовая кукуруза
                        spectrum = create_crop_spectrum(wavelengths, crop_type='corn', health='stressed')
                else:  # Соя
                    spectrum = create_crop_spectrum(wavelengths, crop_type='soybean', health='moderate')
                
                image_data[i, j, :] = spectrum + np.random.normal(0, 0.01, bands)
        
        save_hyperspectral_data(image_data, wavelengths, output_path)
        print(f"Созданы сельскохозяйственные данные: {output_path}")
        
    except Exception as e:
        print(f"Ошибка создания сельскохозяйственных данных: {e}")


def create_forest_data(output_path: str):
    """Создание данных для лесного сценария"""
    try:
        height, width, bands = 200, 200, 50
        wavelengths = np.linspace(400, 1000, bands)
        
        image_data = np.zeros((height, width, bands), dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                # Разные типы леса
                if (i - 100)**2 + (j - 100)**2 < 2500:  # Хвойный лес в центре
                    spectrum = create_forest_spectrum(wavelengths, forest_type='coniferous', health='healthy')
                elif (i - 50)**2 + (j - 150)**2 < 1600:  # Лиственный лес
                    spectrum = create_forest_spectrum(wavelengths, forest_type='deciduous', health='healthy')
                elif (i - 150)**2 + (j - 50)**2 < 1600:  # Поврежденный лес
                    spectrum = create_forest_spectrum(wavelengths, forest_type='mixed', health='damaged')
                else:  # Подлесок
                    spectrum = create_forest_spectrum(wavelengths, forest_type='undergrowth', health='moderate')
                
                image_data[i, j, :] = spectrum + np.random.normal(0, 0.015, bands)
        
        save_hyperspectral_data(image_data, wavelengths, output_path)
        print(f"Созданы лесные данные: {output_path}")
        
    except Exception as e:
        print(f"Ошибка создания лесных данных: {e}")


def create_water_data(output_path: str):
    """Создание данных для водного сценария"""
    try:
        height, width, bands = 120, 120, 30
        wavelengths = np.linspace(400, 900, bands)
        
        image_data = np.zeros((height, width, bands), dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                # Водные объекты и прибрежные зоны
                if (i - 60)**2 + (j - 60)**2 < 900:  # Озеро в центре
                    spectrum = create_water_spectrum(wavelengths, water_type='clear')
                elif (i - 60)**2 + (j - 60)**2 < 1600:  # Прибрежная зона
                    spectrum = create_water_spectrum(wavelengths, water_type='turbid')
                elif i > 100:  # Река
                    spectrum = create_water_spectrum(wavelengths, water_type='river')
                else:  # Суша
                    spectrum = create_vegetation_spectrum(wavelengths, health='moderate')
                
                image_data[i, j, :] = spectrum + np.random.normal(0, 0.008, bands)
        
        save_hyperspectral_data(image_data, wavelengths, output_path)
        print(f"Созданы водные данные: {output_path}")
        
    except Exception as e:
        print(f"Ошибка создания водных данных: {e}")


def create_environmental_data(output_path: str):
    """Создание данных для экологического сценария"""
    try:
        height, width, bands = 180, 180, 80
        wavelengths = np.linspace(400, 1100, bands)
        
        image_data = np.zeros((height, width, bands), dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                # Разные экологические зоны
                if i < 60:  # Заповедная зона
                    spectrum = create_pristine_spectrum(wavelengths)
                elif i < 120:  # Зона умеренного воздействия
                    if j < 90:
                        spectrum = create_moderate_impact_spectrum(wavelengths)
                    else:
                        spectrum = create_vegetation_spectrum(wavelengths, health='healthy')
                else:  # Зона сильного антропогенного воздействия
                    spectrum = create_impacted_spectrum(wavelengths)
                
                image_data[i, j, :] = spectrum + np.random.normal(0, 0.02, bands)
        
        save_hyperspectral_data(image_data, wavelengths, output_path)
        print(f"Созданы экологические данные: {output_path}")
        
    except Exception as e:
        print(f"Ошибка создания экологических данных: {e}")


def create_urban_data(output_path: str):
    """Создание данных для урбанистического сценария"""
    try:
        height, width, bands = 160, 160, 15
        wavelengths = np.linspace(450, 850, bands)
        
        image_data = np.zeros((height, width, bands), dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                # Городская структура
                if (i - 80)**2 + (j - 80)**2 < 400:  # Центр города
                    spectrum = create_urban_spectrum(wavelengths, urban_type='dense')
                elif (i - 80)**2 + (j - 80)**2 < 1600:  # Жилая зона
                    spectrum = create_urban_spectrum(wavelengths, urban_type='residential')
                elif i < 40 or i > 120 or j < 40 or j > 120:  # Промышленная зона
                    spectrum = create_urban_spectrum(wavelengths, urban_type='industrial')
                else:  # Парки и зеленые зоны
                    spectrum = create_vegetation_spectrum(wavelengths, health='moderate')
                
                image_data[i, j, :] = spectrum + np.random.normal(0, 0.012, bands)
        
        save_hyperspectral_data(image_data, wavelengths, output_path)
        print(f"Созданы урбанистические данные: {output_path}")
        
    except Exception as e:
        print(f"Ошибка создания урбанистических данных: {e}")


def create_ecosystem_data(output_path: str):
    """Создание данных для экосистемного сценария"""
    try:
        height, width, bands = 200, 200, 100
        wavelengths = np.linspace(400, 1200, bands)
        
        image_data = np.zeros((height, width, bands), dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                # Комплексная экосистема
                if i < 50:  # Водно-болотные угодья
                    spectrum = create_wetland_spectrum(wavelengths)
                elif i < 100:  # Лесная зона
                    if j < 100:
                        spectrum = create_forest_spectrum(wavelengths, forest_type='mixed', health='healthy')
                    else:
                        spectrum = create_forest_spectrum(wavelengths, forest_type='deciduous', health='moderate')
                elif i < 150:  # Саванна/луг
                    spectrum = create_grassland_spectrum(wavelengths)
                else:  # Пустынная зона
                    spectrum = create_desert_spectrum(wavelengths)
                
                image_data[i, j, :] = spectrum + np.random.normal(0, 0.018, bands)
        
        save_hyperspectral_data(image_data, wavelengths, output_path)
        print(f"Созданы экосистемные данные: {output_path}")
        
    except Exception as e:
        print(f"Ошибка создания экосистемных данных: {e}")


# Функции создания спектральных сигнатур

def create_crop_spectrum(wavelengths: np.ndarray, crop_type: str = 'wheat', health: str = 'healthy') -> np.ndarray:
    """Создание спектра сельскохозяйственной культуры"""
    spectrum = np.zeros_like(wavelengths)
    
    if crop_type == 'wheat':
        if health == 'healthy':
            green_peak = 0.35 * np.exp(-((wavelengths - 550) / 35) ** 2)
            red_edge = 0.65 / (1 + np.exp(-(wavelengths - 720) / 12))
            nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.75, 0)
        elif health == 'stressed':
            green_peak = 0.2 * np.exp(-((wavelengths - 550) / 45) ** 2)
            red_edge = 0.4 / (1 + np.exp(-(wavelengths - 720) / 18))
            nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.45, 0)
        else:  # moderate
            green_peak = 0.28 * np.exp(-((wavelengths - 550) / 40) ** 2)
            red_edge = 0.55 / (1 + np.exp(-(wavelengths - 720) / 15))
            nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.6, 0)
    
    elif crop_type == 'corn':
        if health == 'healthy':
            green_peak = 0.4 * np.exp(-((wavelengths - 550) / 30) ** 2)
            red_edge = 0.7 / (1 + np.exp(-(wavelengths - 720) / 10))
            nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.8, 0)
        else:  # stressed
            green_peak = 0.15 * np.exp(-((wavelengths - 550) / 50) ** 2)
            red_edge = 0.3 / (1 + np.exp(-(wavelengths - 720) / 20))
            nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.35, 0)
    
    else:  # soybean
        green_peak = 0.32 * np.exp(-((wavelengths - 550) / 38) ** 2)
        red_edge = 0.6 / (1 + np.exp(-(wavelengths - 720) / 14))
        nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.65, 0)
    
    spectrum = 0.08 + green_peak + red_edge + nir_plateau
    return spectrum


def create_forest_spectrum(wavelengths: np.ndarray, forest_type: str = 'mixed', health: str = 'healthy') -> np.ndarray:
    """Создание спектра леса"""
    spectrum = np.zeros_like(wavelengths)
    
    if forest_type == 'coniferous':
        if health == 'healthy':
            green_peak = 0.25 * np.exp(-((wavelengths - 540) / 40) ** 2)
            red_edge = 0.6 / (1 + np.exp(-(wavelengths - 710) / 15))
            nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 1000), 0.7, 0)
        else:  # damaged
            green_peak = 0.15 * np.exp(-((wavelengths - 540) / 50) ** 2)
            red_edge = 0.35 / (1 + np.exp(-(wavelengths - 710) / 20))
            nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 1000), 0.4, 0)
    
    elif forest_type == 'deciduous':
        green_peak = 0.35 * np.exp(-((wavelengths - 550) / 35) ** 2)
        red_edge = 0.65 / (1 + np.exp(-(wavelengths - 720) / 12))
        nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 1000), 0.75, 0)
    
    else:  # mixed or undergrowth
        green_peak = 0.3 * np.exp(-((wavelengths - 545) / 38) ** 2)
        red_edge = 0.6 / (1 + np.exp(-(wavelengths - 715) / 14))
        nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 1000), 0.65, 0)
    
    spectrum = 0.05 + green_peak + red_edge + nir_plateau
    return spectrum


def create_water_spectrum(wavelengths: np.ndarray, water_type: str = 'clear') -> np.ndarray:
    """Создание спектра воды"""
    spectrum = np.zeros_like(wavelengths)
    
    if water_type == 'clear':
        # Чистая вода - низкое отражение в NIR, высокое в синем
        blue_reflectance = 0.15 * np.exp(-((wavelengths - 450) / 100) ** 2)
        green_reflectance = 0.08 * np.exp(-((wavelengths - 550) / 120) ** 2)
        red_reflectance = 0.02 * np.exp(-((wavelengths - 650) / 80) ** 2)
        nir_absorption = np.where(wavelengths > 700, -0.1, 0)
        
    elif water_type == 'turbid':
        # Мутная вода - более высокое отражение во всех диапазонах
        blue_reflectance = 0.12 * np.exp(-((wavelengths - 450) / 90) ** 2)
        green_reflectance = 0.15 * np.exp(-((wavelengths - 550) / 100) ** 2)
        red_reflectance = 0.08 * np.exp(-((wavelengths - 650) / 70) ** 2)
        nir_absorption = np.where(wavelengths > 700, -0.05, 0)
    
    else:  # river
        blue_reflectance = 0.1 * np.exp(-((wavelengths - 450) / 95) ** 2)
        green_reflectance = 0.12 * np.exp(-((wavelengths - 550) / 110) ** 2)
        red_reflectance = 0.05 * np.exp(-((wavelengths - 650) / 75) ** 2)
        nir_absorption = np.where(wavelengths > 700, -0.08, 0)
    
    spectrum = 0.02 + blue_reflectance + green_reflectance + red_reflectance + nir_absorption
    return np.maximum(spectrum, 0)  # Убедимся, что значения неотрицательные


def create_vegetation_spectrum(wavelengths: np.ndarray, health: str = 'healthy') -> np.ndarray:
    """Создание спектра растительности"""
    spectrum = np.zeros_like(wavelengths)
    
    if health == 'healthy':
        green_peak = 0.4 * np.exp(-((wavelengths - 550) / 40) ** 2)
        red_edge = 0.7 / (1 + np.exp(-(wavelengths - 720) / 15))
        nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.8, 0)
    elif health == 'stressed':
        green_peak = 0.2 * np.exp(-((wavelengths - 550) / 50) ** 2)
        red_edge = 0.4 / (1 + np.exp(-(wavelengths - 720) / 20))
        nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.5, 0)
    else:  # moderate
        green_peak = 0.3 * np.exp(-((wavelengths - 550) / 45) ** 2)
        red_edge = 0.55 / (1 + np.exp(-(wavelengths - 720) / 18))
        nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.65, 0)
    
    spectrum = 0.1 + green_peak + red_edge + nir_plateau
    return spectrum


def create_pristine_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Создание спектра нетронутой экосистемы"""
    # Комбинация здоровой растительности и чистой почвы
    vegetation = create_vegetation_spectrum(wavelengths, 'healthy')
    soil = 0.15 + 0.03 * np.sin((wavelengths - 400) / 200)
    
    return 0.7 * vegetation + 0.3 * soil


def create_moderate_impact_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Создание спектра зоны умеренного воздействия"""
    vegetation = create_vegetation_spectrum(wavelengths, 'moderate')
    soil = 0.18 + 0.04 * np.sin((wavelengths - 400) / 180)
    
    return 0.6 * vegetation + 0.4 * soil


def create_impacted_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Создание спектра зоны сильного воздействия"""
    vegetation = create_vegetation_spectrum(wavelengths, 'stressed')
    urban = 0.2 + 0.02 * np.random.random(len(wavelengths))
    
    return 0.4 * vegetation + 0.6 * urban


def create_urban_spectrum(wavelengths: np.ndarray, urban_type: str = 'residential') -> np.ndarray:
    """Создание спектра городской территории"""
    spectrum = np.zeros_like(wavelengths)
    
    if urban_type == 'dense':
        # Плотная городская застройка
        spectrum = 0.25 + 0.05 * np.sin((wavelengths - 400) / 150) + 0.02 * np.random.random(len(wavelengths))
    elif urban_type == 'residential':
        # Жилая зона с зелеными насаждениями
        urban = 0.2 + 0.03 * np.sin((wavelengths - 400) / 160)
        vegetation = create_vegetation_spectrum(wavelengths, 'moderate')
        spectrum = 0.6 * urban + 0.4 * vegetation
    else:  # industrial
        # Промышленная зона
        spectrum = 0.3 + 0.04 * np.sin((wavelengths - 400) / 140) + 0.03 * np.random.random(len(wavelengths))
    
    return spectrum


def create_wetland_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Создание спектра водно-болотных угодий"""
    water = create_water_spectrum(wavelengths, 'turbid')
    vegetation = create_vegetation_spectrum(wavelengths, 'moderate')
    
    return 0.6 * water + 0.4 * vegetation


def create_grassland_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Создание спектра луга/саванны"""
    return create_vegetation_spectrum(wavelengths, 'moderate')


def create_desert_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Создание спектра пустыни"""
    return 0.25 + 0.08 * np.sin((wavelengths - 400) / 120) + 0.02 * np.random.random(len(wavelengths))


# Вспомогательные функции

def save_hyperspectral_data(image_data: np.ndarray, wavelengths: np.ndarray, output_path: str):
    """Сохранение гиперспектральных данных"""
    try:
        from osgeo import gdal
        
        # Создание HDR файла
        hdr_path = output_path.replace('.bil', '.hdr')
        with open(hdr_path, 'w') as f:
            f.write(f"ENVI\n")
            f.write(f"description = {{Scenario data}}\n")
            f.write(f"samples = {image_data.shape[1]}\n")
            f.write(f"lines = {image_data.shape[0]}\n")
            f.write(f"bands = {image_data.shape[2]}\n")
            f.write(f"data type = 4\n")
            f.write(f"interleave = bsq\n")
            f.write(f"byte order = 0\n")
            f.write(f"wavelength = {{")
            f.write(", ".join([f"{w:.1f}" for w in wavelengths]))
            f.write("}}\n")
        
        # Создание BIL файла
        driver = gdal.GetDriverByName('ENVI')
        dataset = driver.Create(output_path, image_data.shape[1], image_data.shape[0], 
                              image_data.shape[2], gdal.GDT_Float32)
        
        for band in range(image_data.shape[2]):
            band_data = dataset.GetRasterBand(band + 1)
            band_data.WriteArray(image_data[:, :, band])
        
        dataset = None
        
    except ImportError:
        # Альтернативный метод сохранения
        np.save(output_path.replace('.bil', '.npy'), image_data)
        np.save(output_path.replace('.bil', '_wavelengths.npy'), wavelengths)


# Функции анализа для различных сценариев

def analyze_agriculture_results(results: dict, output_dir: str):
    """Анализ результатов сельского хозяйства"""
    try:
        indices_results = results.get('indices', {})
        normalized_indices = indices_results.get('normalized_indices', {})
        
        if not normalized_indices:
            print("Нет индексов для анализа")
            return
        
        # Анализ урожайности на основе NDVI
        ndvi = normalized_indices.get('NDVI')
        if ndvi is not None:
            yield_estimate = estimate_crop_yield(ndvi)
            
            # Визуализация
            try:
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Карта NDVI
                im1 = axes[0, 0].imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1)
                axes[0, 0].set_title('NDVI - Состояние культур')
                axes[0, 0].axis('off')
                plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
                
                # Карта урожайности
                im2 = axes[0, 1].imshow(yield_estimate, cmap='YlOrRd', vmin=0, vmax=1)
                axes[0, 1].set_title('Прогноз урожайности')
                axes[0, 1].axis('off')
                plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
                
                # Распределение NDVI
                valid_ndvi = ndvi[ndvi > 0]
                axes[1, 0].hist(valid_ndvi, bins=50, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Распределение NDVI')
                axes[1, 0].set_xlabel('NDVI')
                axes[1, 0].set_ylabel('Частота')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Статистика по полям
                field_stats = calculate_field_statistics(ndvi)
                field_names = list(field_stats.keys())
                mean_ndvi = [field_stats[name]['mean'] for name in field_names]
                
                axes[1, 1].bar(field_names, mean_ndvi)
                axes[1, 1].set_title('Средний NDVI по полям')
                axes[1, 1].set_ylabel('Средний NDVI')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(output_dir, 'plots', 'agriculture_analysis.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Анализ сельского хозяйства сохранен: {plot_path}")
                
            except ImportError:
                print("Matplotlib не доступен для визуализации")
        
        # Сохранение отчета
        save_agriculture_report(results, yield_estimate if 'yield_estimate' in locals() else None, output_dir)
        
    except Exception as e:
        print(f"Ошибка анализа сельского хозяйства: {e}")


def estimate_crop_yield(ndvi: np.ndarray) -> np.ndarray:
    """Оценка урожайности на основе NDVI"""
    # Упрощенная модель оценки урожайности
    # В реальности здесь должна быть более сложная модель
    yield_estimate = np.zeros_like(ndvi)
    
    # Нормализация NDVI к диапазону урожайности
    valid_mask = ndvi > 0
    yield_estimate[valid_mask] = np.clip(ndvi[valid_mask] * 1.2, 0, 1)
    
    return yield_estimate


def calculate_field_statistics(ndvi: np.ndarray) -> dict:
    """Расчет статистики по полям"""
    height, width = ndvi.shape
    
    # Разделение на условные поля
    field_stats = {}
    
    # Поле 1: верхняя треть
    field1 = ndvi[:height//3, :]
    valid_field1 = field1[field1 > 0]
    if len(valid_field1) > 0:
        field_stats['Поле 1'] = {
            'mean': np.mean(valid_field1),
            'std': np.std(valid_field1),
            'area': len(valid_field1)
        }
    
    # Поле 2: средняя треть
    field2 = ndvi[height//3:2*height//3, :]
    valid_field2 = field2[field2 > 0]
    if len(valid_field2) > 0:
        field_stats['Поле 2'] = {
            'mean': np.mean(valid_field2),
            'std': np.std(valid_field2),
            'area': len(valid_field2)
        }
    
    # Поле 3: нижняя треть
    field3 = ndvi[2*height//3:, :]
    valid_field3 = field3[field3 > 0]
    if len(valid_field3) > 0:
        field_stats['Поле 3'] = {
            'mean': np.mean(valid_field3),
            'std': np.std(valid_field3),
            'area': len(valid_field3)
        }
    
    return field_stats


def save_agriculture_report(results: dict, yield_estimate: np.ndarray, output_dir: str):
    """Сохранение сельскохозяйственного отчета"""
    try:
        report_path = os.path.join(output_dir, 'agriculture_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("СЕЛЬСКОХОЗЯЙСТВЕННЫЙ ОТЧЕТ\n")
            f.write("Мониторинг сельскохозяйственных культур\n")
            f.write("="*50 + "\n\n")
            
            # Информация о данных
            processed_data = results.get('processed_data', {})
            f.write("1. ИНФОРМАЦИЯ О ДАННЫХ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Размер изображения: {processed_data.get('shape', 'N/A')}\n")
            f.write(f"Количество каналов: {processed_data.get('bands', 'N/A')}\n\n")
            
            # Результаты индексов
            indices_results = results.get('indices', {})
            calculated_indices = indices_results.get('calculated_indices', [])
            f.write("2. РАССЧИТАННЫЕ ИНДЕКСЫ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Индексы: {', '.join(calculated_indices)}\n\n")
            
            # Состояние культур
            plant_condition = results.get('plant_condition', {})
            if 'classification' in plant_condition:
                classification = plant_condition['classification']
                f.write("3. СОСТОЯНИЕ КУЛЬТУР\n")
                f.write("-" * 30 + "\n")
                f.write(f"Общий класс: {classification.get('class', 'N/A')}\n")
                f.write(f"Описание: {classification.get('description', 'N/A')}\n")
                f.write(f"Оценка: {classification.get('score', 0):.3f}\n\n")
            
            # Рекомендации
            f.write("4. РЕКОМЕНДАЦИИ\n")
            f.write("-" * 30 + "\n")
            
            score = classification.get('score', 0) if 'classification' in plant_condition else 0
            if score > 0.7:
                f.write("Культуры в отличном состоянии. Продолжайте текущую агротехнику.\n")
            elif score > 0.4:
                f.write("Культуры в удовлетворительном состоянии. Рекомендуется дополнительный анализ.\n")
            else:
                f.write("Культуры в стрессовом состоянии. Требуется вмешательство.\n")
            
            f.write("\nРекомендуется провести дополнительный полевой анализ для валидации результатов.\n")
        
        print(f"Сельскохозяйственный отчет сохранен: {report_path}")
        
    except Exception as e:
        print(f"Ошибка сохранения сельскохозяйственного отчета: {e}")


def calculate_forest_indices(processed_data: dict, output_dir: str) -> dict:
    """Расчет лесных индексов"""
    try:
        # Здесь должна быть реализация расчета специфических лесных индексов
        # Например, индекс листовой площади (LAI), индекс влажности листвы и т.д.
        
        forest_indices = {}
        
        # Упрощенный пример - использование стандартных индексов
        tiff_paths = processed_data.get('tiff_paths', [])
        if tiff_paths:
            # Загрузка данных и расчет индексов
            # В реальном коде здесь была бы полная реализация
            pass
        
        return forest_indices
        
    except Exception as e:
        print(f"Ошибка расчета лесных индексов: {e}")
        return {}


def analyze_forest_health(forest_indices: dict, output_dir: str):
    """Анализ состояния леса"""
    try:
        # Анализ состояния леса на основе индексов
        print("Анализ состояния леса...")
        
        # Визуализация результатов
        # Сохранение отчета
        
    except Exception as e:
        print(f"Ошибка анализа состояния леса: {e}")


def calculate_water_indices(processed_data: dict, output_dir: str) -> dict:
    """Расчет водных индексов"""
    try:
        water_indices = {}
        
        # Расчет специфических водных индексов
        # NDWI, MNDWI, WRI и т.д.
        
        return water_indices
        
    except Exception as e:
        print(f"Ошибка расчета водных индексов: {e}")
        return {}


def analyze_water_quality(water_indices: dict, output_dir: str):
    """Анализ качества воды"""
    try:
        print("Анализ качества воды...")
        
    except Exception as e:
        print(f"Ошибка анализа качества воды: {e}")


def perform_environmental_analysis(results: dict, output_dir: str):
    """Экологический анализ"""
    try:
        print("Выполнение экологического анализа...")
        
    except Exception as e:
        print(f"Ошибка экологического анализа: {e}")


def calculate_urban_indices(processed_data: dict, segmentation_mask: str, output_dir: str) -> dict:
    """Расчет урбанистических индексов"""
    try:
        urban_indices = {}
        
        # Расчет индексов урбанизации
        # NDBI, NDBaI и т.д.
        
        return urban_indices
        
    except Exception as e:
        print(f"Ошибка расчета урбанистических индексов: {e}")
        return {}


def analyze_urban_development(urban_indices: dict, output_dir: str):
    """Анализ урбанизации"""
    try:
        print("Анализ урбанизации...")
        
    except Exception as e:
        print(f"Ошибка анализа урбанизации: {e}")


def perform_ecosystem_analysis(results: dict, output_dir: str):
    """Комплексный анализ экосистемы"""
    try:
        print("Выполнение комплексного анализа экосистемы...")
        
    except Exception as e:
        print(f"Ошибка комплексного анализа экосистемы: {e}")


def create_biodiversity_map(results: dict, output_dir: str):
    """Создание карты биоразнообразия"""
    try:
        print("Создание карты биоразнообразия...")
        
    except Exception as e:
        print(f"Ошибка создания карты биоразнообразия: {e}")


if __name__ == '__main__':
    sys.exit(main())