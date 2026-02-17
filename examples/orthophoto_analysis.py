#!/usr/bin/env python3
"""
Пример анализа ортофотопланов и вегетационных индексов
с использованием научной библиотеки GOP v2.0

Этот пример демонстрирует:
- Загрузку и анализ ортофотопланов
- Расчет различных вегетационных индексов
- Пространственный анализ индексов
- Визуализацию результатов
- Классификацию состояния растений
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Добавление src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.pipeline import Pipeline
from src.processing.orthophoto import OrthophotoProcessor
from src.indices.calculator import VegetationIndexCalculator
from src.segmentation.segmenter import ImageSegmenter
from src.utils.logger import setup_logger


def main():
    """Основная функция анализа ортофотопланов"""
    
    # Настройка логирования
    logger = setup_logger('GOP_Orthophoto', level=logging.INFO)
    logger.info("Начало анализа ортофотопланов и вегетационных индексов")
    
    try:
        # Путь к входным данным
        orthophoto_path = "data/sample_orthophoto.tif"
        output_dir = "results/orthophoto_analysis"
        
        # Проверка наличия входных данных
        if not os.path.exists(orthophoto_path):
            logger.error(f"Ортофотоплан не найден: {orthophoto_path}")
            logger.info("Создание примера ортофотоплана")
            create_sample_orthophoto(orthophoto_path)
        
        # Создание выходной директории
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'indices'), exist_ok=True)
        
        # Шаг 1: Валидация ортофотоплана
        logger.info("Шаг 1: Валидация ортофотоплана")
        orthophoto_processor = OrthophotoProcessor()
        validation_results = orthophoto_processor.validate_orthophoto(orthophoto_path)
        
        print("\n" + "="*60)
        print("ВАЛИДАЦИЯ ОРТОФОТОПЛАНА")
        print("="*60)
        print(f"Размер: {validation_results.get('width', 'N/A')} x {validation_results.get('height', 'N/A')}")
        print(f"Каналов: {validation_results.get('bands', 'N/A')}")
        print(f"Геопривязка: {'Да' if validation_results.get('has_georeference') else 'Нет'}")
        print(f"Проекция: {'Да' if validation_results.get('has_projection') else 'Нет'}")
        print(f"Валидность: {'Да' if validation_results.get('valid') else 'Нет'}")
        
        if not validation_results.get('valid'):
            logger.warning("Ортофотоплан не прошел валидацию")
        
        # Шаг 2: Сегментация изображения
        logger.info("Шаг 2: Сегментация изображения")
        segmenter = ImageSegmenter()
        segmentation_mask = segmenter.segment(
            orthophoto_path, 
            output_dir=output_dir,
            use_refinement=True,
            compression_ratio=0.125
        )
        
        # Шаг 3: Расчет вегетационных индексов
        logger.info("Шаг 3: Расчет вегетационных индексов")
        index_calculator = VegetationIndexCalculator()
        
        # Определение типа сенсора на основе количества каналов
        sensor_type = determine_sensor_type(orthophoto_path)
        logger.info(f"Определен тип сенсора: {sensor_type}")
        
        # Выбор индексов в зависимости от типа сенсора
        selected_indices = get_indices_for_sensor(sensor_type)
        
        indices_results = index_calculator.calculate(
            orthophoto_path=orthophoto_path,
            segmentation_mask=segmentation_mask,
            sensor_type=sensor_type,
            selected_indices=selected_indices,
            output_dir=output_dir
        )
        
        # Шаг 4: Анализ рассчитанных индексов
        logger.info("Шаг 4: Анализ рассчитанных индексов")
        analyze_calculated_indices(indices_results, output_dir)
        
        # Шаг 5: Пространственный анализ
        logger.info("Шаг 5: Пространственный анализ индексов")
        perform_spatial_analysis(indices_results, segmentation_mask, output_dir)
        
        # Шаг 6: Временной анализ (если есть несколько изображений)
        logger.info("Шаг 6: Временной анализ")
        perform_temporal_analysis(output_dir)
        
        # Шаг 7: Классификация состояния растений
        logger.info("Шаг 7: Классификация состояния растений")
        plant_condition = index_calculator.assess_plant_condition(indices_results)
        classify_plant_health(plant_condition, output_dir)
        
        # Шаг 8: Создание комплексных визуализаций
        logger.info("Шаг 8: Создание комплексных визуализаций")
        create_comprehensive_visualizations(
            orthophoto_path, 
            indices_results, 
            segmentation_mask, 
            output_dir
        )
        
        # Шаг 9: Генерация отчета
        logger.info("Шаг 9: Генерация отчета")
        generate_analysis_report(
            validation_results, 
            indices_results, 
            plant_condition, 
            output_dir
        )
        
        print("\n" + "="*60)
        print("АНАЛИЗ ОРТОФОТОПЛАНА ЗАВЕРШЕН")
        print("="*60)
        print(f"Результаты сохранены в: {output_dir}")
        print(f"Индексы: {output_dir}/indices/")
        print(f"Визуализации: {output_dir}/plots/")
        print(f"Отчет: {output_dir}/analysis_report.txt")
        
    except Exception as e:
        logger.error(f"Ошибка в анализе ортофотоплана: {e}")
        print(f"Ошибка: {e}")
        return 1
    
    return 0


def create_sample_orthophoto(output_path: str):
    """Создание примера ортофотоплана для демонстрации"""
    try:
        # Создание директории
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Параметры изображения
        height, width, bands = 200, 200, 4  # RGB + NIR
        
        # Создание синтетического ортофотоплана
        image_data = np.zeros((height, width, bands), dtype=np.float32)
        
        # Создание различных типов поверхностей
        for i in range(height):
            for j in range(width):
                # Базовые значения для почвы
                soil_base = np.array([0.2, 0.15, 0.1, 0.3])
                
                # Добавление растительности в некоторых областях
                if (i - 100)**2 + (j - 100)**2 < 2500:  # Центральная круглая область
                    vegetation = np.array([0.1, 0.3, 0.1, 0.8])
                    image_data[i, j, :] = 0.7 * vegetation + 0.3 * soil_base
                elif (i - 50)**2 + (j - 150)**2 < 400:  # Маленькая область
                    vegetation = np.array([0.05, 0.25, 0.05, 0.7])
                    image_data[i, j, :] = 0.8 * vegetation + 0.2 * soil_base
                else:
                    image_data[i, j, :] = soil_base
                
                # Добавление шума
                image_data[i, j, :] += np.random.normal(0, 0.02, bands)
        
        # Нормализация значений
        image_data = np.clip(image_data, 0, 1)
        
        # Сохранение с помощью GDAL
        try:
            from osgeo import gdal
            
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(output_path, width, height, bands, gdal.GDT_Float32)
            
            # Установка геопривязки (пример)
            dataset.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
            
            # Запись каналов
            for band in range(bands):
                band_data = dataset.GetRasterBand(band + 1)
                band_data.WriteArray(image_data[:, :, band])
            
            dataset = None
            
        except ImportError:
            # Альтернативный метод сохранения
            np.save(output_path.replace('.tif', '.npy'), image_data)
        
        print(f"Создан пример ортофотоплана: {output_path}")
        
    except Exception as e:
        print(f"Ошибка создания примера ортофотоплана: {e}")


def determine_sensor_type(orthophoto_path: str) -> str:
    """Определение типа сенсора на основе количества каналов"""
    try:
        from osgeo import gdal
        
        dataset = gdal.Open(orthophoto_path)
        if dataset is None:
            return 'RGB'  # По умолчанию
        
        bands = dataset.RasterCount
        
        if bands == 3:
            return 'RGB'
        elif bands == 4:
            return 'Multispectral'  # RGB + NIR
        elif bands >= 10:
            return 'Hyperspectral'
        else:
            return 'RGB'
            
    except Exception:
        return 'RGB'


def get_indices_for_sensor(sensor_type: str) -> list:
    """Получение списка индексов для типа сенсора"""
    if sensor_type == 'RGB':
        return ['ExG', 'ExGR', 'CIVE', 'VEG']
    elif sensor_type == 'Multispectral':
        return ['NDVI', 'GNDVI', 'NDWI', 'MSI', 'OSAVI', 'TVI']
    elif sensor_type == 'Hyperspectral':
        return ['NDVI', 'GNDVI', 'MCARI', 'MNLI', 'OSAVI', 'TVI', 'SIPI2', 'mARI', 'NDWI', 'MSI']
    else:
        return ['NDVI']


def analyze_calculated_indices(indices_results: dict, output_dir: str):
    """Анализ рассчитанных индексов"""
    try:
        normalized_indices = indices_results.get('normalized_indices', {})
        
        if not normalized_indices:
            print("Нет рассчитанных индексов для анализа")
            return
        
        # Создание таблицы статистики
        print("\n" + "="*60)
        print("СТАТИСТИКА ВЕГЕТАЦИОННЫХ ИНДЕКСОВ")
        print("="*60)
        print(f"{'Индекс':<10} {'Среднее':<10} {'СКО':<10} {'Минимум':<10} {'Максимум':<10}")
        print("-" * 60)
        
        for index_name, index_data in normalized_indices.items():
            if isinstance(index_data, np.ndarray):
                valid_data = index_data[index_data > 0]
                if len(valid_data) > 0:
                    print(f"{index_name:<10} {np.mean(valid_data):<10.3f} "
                          f"{np.std(valid_data):<10.3f} {np.min(valid_data):<10.3f} "
                          f"{np.max(valid_data):<10.3f}")
        
        # Визуализация распределения индексов
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (index_name, index_data) in enumerate(normalized_indices.items()):
            if i >= 6:
                break
                
            if isinstance(index_data, np.ndarray):
                valid_data = index_data[index_data > 0]
                if len(valid_data) > 0:
                    axes[i].hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{index_name}')
                    axes[i].set_xlabel('Значение')
                    axes[i].set_ylabel('Частота')
                    axes[i].grid(True, alpha=0.3)
        
        # Скрытие пустых subplot'ов
        for i in range(len(normalized_indices), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'plots', 'index_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Распределения индексов сохранены: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка анализа индексов: {e}")


def perform_spatial_analysis(indices_results: dict, segmentation_mask: str, output_dir: str):
    """Выполнение пространственного анализа индексов"""
    try:
        normalized_indices = indices_results.get('normalized_indices', {})
        
        if not normalized_indices:
            return
        
        # Чтение маски сегментации
        try:
            from osgeo import gdal
            mask_dataset = gdal.Open(segmentation_mask)
            mask_data = mask_dataset.GetRasterBand(1).ReadAsArray()
            mask_data = (mask_data > 0).astype(np.uint8)
        except:
            # Если маска не доступна, создаем простую маску
            first_index = list(normalized_indices.values())[0]
            mask_data = (first_index > 0).astype(np.uint8)
        
        # Пространственный анализ для каждого индекса
        spatial_results = {}
        
        for index_name, index_data in normalized_indices.items():
            if isinstance(index_data, np.ndarray):
                # Применение маски
                masked_data = index_data * mask_data
                
                # Расчет пространственных метрик
                spatial_results[index_name] = calculate_spatial_metrics(masked_data)
        
        # Визуализация пространственных метрик
        visualize_spatial_metrics(spatial_results, output_dir)
        
        # Сохранение результатов
        save_spatial_results(spatial_results, output_dir)
        
        print("Пространственный анализ завершен")
        
    except Exception as e:
        print(f"Ошибка пространственного анализа: {e}")


def calculate_spatial_metrics(data: np.ndarray) -> dict:
    """Расчет пространственных метрик для индекса"""
    try:
        # Фильтрация валидных данных
        valid_data = data[data > 0]
        
        if len(valid_data) == 0:
            return {}
        
        # Базовые статистики
        metrics = {
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'cv': float(np.std(valid_data) / np.mean(valid_data)) if np.mean(valid_data) > 0 else 0,
            'area_ratio': float(len(valid_data) / data.size)
        }
        
        # Пространственная автокорреляция (упрощенная)
        metrics['spatial_autocorrelation'] = calculate_spatial_autocorrelation(data)
        
        # Анализ горячих точек
        hotspot_analysis = perform_hotspot_analysis(valid_data)
        metrics.update(hotspot_analysis)
        
        return metrics
        
    except Exception as e:
        print(f"Ошибка расчета пространственных метрик: {e}")
        return {}


def calculate_spatial_autocorrelation(data: np.ndarray) -> float:
    """Расчет пространственной автокорреляции (упрощенный индекс Морана)"""
    try:
        rows, cols = data.shape
        if rows < 3 or cols < 3:
            return 0.0
        
        # Упрощенный расчет локальной автокорреляции
        valid_mask = data > 0
        
        # Расчет корреляции с соседними пикселями
        correlations = []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if valid_mask[i, j]:
                    center = data[i, j]
                    neighbors = [
                        data[i-1, j], data[i+1, j], 
                        data[i, j-1], data[i, j+1]
                    ]
                    valid_neighbors = [n for n in neighbors if n > 0]
                    
                    if len(valid_neighbors) >= 2:
                        neighbor_mean = np.mean(valid_neighbors)
                        if center > 0 and neighbor_mean > 0:
                            correlations.append(abs(center - neighbor_mean))
        
        if correlations:
            # Обратная величина среднего различия (чем меньше, тем выше автокорреляция)
            return 1.0 / (1.0 + np.mean(correlations))
        
        return 0.0
        
    except Exception:
        return 0.0


def perform_hotspot_analysis(data: np.ndarray) -> dict:
    """Анализ горячих точек"""
    try:
        if len(data) < 10:
            return {'hotspot_percentage': 0, 'coldspot_percentage': 0}
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return {'hotspot_percentage': 0, 'coldspot_percentage': 0}
        
        # Классификация на основе z-оценок
        z_scores = (data - mean_val) / std_val
        
        hotspots = np.sum(z_scores > 1.96)  # p < 0.05
        coldspots = np.sum(z_scores < -1.96)  # p < 0.05
        
        return {
            'hotspot_percentage': float(hotspots / len(data) * 100),
            'coldspot_percentage': float(coldspots / len(data) * 100)
        }
        
    except Exception:
        return {'hotspot_percentage': 0, 'coldspot_percentage': 0}


def visualize_spatial_metrics(spatial_results: dict, output_dir: str):
    """Визуализация пространственных метрик"""
    try:
        if not spatial_results:
            return
        
        # Подготовка данных для визуализации
        metrics_names = ['mean', 'std', 'cv', 'spatial_autocorrelation', 'hotspot_percentage']
        metrics_data = {name: [] for name in metrics_names}
        index_names = []
        
        for index_name, metrics in spatial_results.items():
            index_names.append(index_name)
            for metric_name in metrics_names:
                metrics_data[metric_name].append(metrics.get(metric_name, 0))
        
        # Создание визуализации
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric_name in enumerate(metrics_names):
            if i >= 6:
                break
                
            values = metrics_data[metric_name]
            if values:
                axes[i].bar(index_names, values)
                axes[i].set_title(f'{metric_name.replace("_", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Скрытие пустых subplot'ов
        for i in range(len(metrics_names), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'plots', 'spatial_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Пространственные метрики визуализированы: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка визуализации пространственных метрик: {e}")


def save_spatial_results(spatial_results: dict, output_dir: str):
    """Сохранение результатов пространственного анализа"""
    try:
        import json
        
        results_path = os.path.join(output_dir, 'spatial_analysis_results.json')
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(spatial_results, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты пространственного анализа сохранены: {results_path}")
        
    except Exception as e:
        print(f"Ошибка сохранения пространственных результатов: {e}")


def perform_temporal_analysis(output_dir: str):
    """Выполнение временного анализа (имитация)"""
    try:
        # Создание примера временных данных
        dates = ['2023-06-01', '2023-06-15', '2023-07-01', '2023-07-15', '2023-08-01']
        ndvi_values = [0.3, 0.4, 0.6, 0.7, 0.65]  # Пример изменения NDVI во времени
        
        # Визуализация временных изменений
        plt.figure(figsize=(10, 6))
        plt.plot(dates, ndvi_values, 'o-', linewidth=2, markersize=8)
        plt.title('Временные изменения NDVI')
        plt.xlabel('Дата')
        plt.ylabel('NDVI')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'plots', 'temporal_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Временной анализ сохранен: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка временного анализа: {e}")


def classify_plant_health(plant_condition: dict, output_dir: str):
    """Классификация состояния растений"""
    try:
        print("\n" + "="*60)
        print("КЛАССИФИКАЦИЯ СОСТОЯНИЯ РАСТЕНИЙ")
        print("="*60)
        
        if 'classification' in plant_condition:
            classification = plant_condition['classification']
            print(f"Класс состояния: {classification.get('class', 'N/A')}")
            print(f"Описание: {classification.get('description', 'N/A')}")
            print(f"Количественная оценка: {classification.get('score', 0):.3f}")
            print(f"Уверенность: {classification.get('confidence', 0):.2f}")
        
        # Визуализация состояния растений
        condition_maps = plant_condition.get('condition_maps', {})
        
        if condition_maps:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, (name, data) in enumerate(condition_maps.items()):
                if i >= 4:
                    break
                    
                if isinstance(data, np.ndarray):
                    im = axes[i].imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
                    axes[i].set_title(f'{name.replace("_", " ").title()}')
                    axes[i].axis('off')
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            # Скрытие пустых subplot'ов
            for i in range(len(condition_maps), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'plots', 'plant_condition.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Карты состояния растений сохранены: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка классификации состояния растений: {e}")


def create_comprehensive_visualizations(orthophoto_path: str, 
                                      indices_results: dict, 
                                      segmentation_mask: str, 
                                      output_dir: str):
    """Создание комплексных визуализаций"""
    try:
        from osgeo import gdal
        
        # Чтение ортофотоплана
        dataset = gdal.Open(orthophoto_path)
        if dataset is None:
            return
        
        orthophoto_data = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount))
        for band in range(dataset.RasterCount):
            orthophoto_data[:, :, band] = dataset.GetRasterBand(band + 1).ReadAsArray()
        
        # Чтение маски сегментации
        mask_dataset = gdal.Open(segmentation_mask)
        mask_data = mask_dataset.GetRasterBand(1).ReadAsArray()
        mask_data = (mask_data > 0).astype(np.uint8)
        
        # Создание комплексной визуализации
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Исходное изображение (RGB)
        if orthophoto_data.shape[2] >= 3:
            rgb = orthophoto_data[:, :, :3]
            rgb_normalized = np.zeros_like(rgb)
            for i in range(3):
                band = rgb[:, :, i]
                band_min, band_max = np.percentile(band, [2, 98])
                if band_max > band_min:
                    rgb_normalized[:, :, i] = (band - band_min) / (band_max - band_min)
            
            axes[0, 0].imshow(np.clip(rgb_normalized, 0, 1))
            axes[0, 0].set_title('Исходное изображение (RGB)')
            axes[0, 0].axis('off')
        
        # Маска сегментации
        axes[0, 1].imshow(mask_data, cmap='gray')
        axes[0, 1].set_title('Маска сегментации')
        axes[0, 1].axis('off')
        
        # Наложение маски на изображение
        if orthophoto_data.shape[2] >= 3:
            overlay = rgb_normalized.copy()
            overlay[mask_data == 0] *= 0.5  # Затемнение областей вне маски
            axes[0, 2].imshow(np.clip(overlay, 0, 1))
            axes[0, 2].set_title('Наложение маски')
            axes[0, 2].axis('off')
        
        # Вегетационные индексы
        normalized_indices = indices_results.get('normalized_indices', {})
        index_names = list(normalized_indices.keys())[:3]
        
        for i, index_name in enumerate(index_names):
            if i >= 3:
                break
                
            index_data = normalized_indices[index_name]
            if isinstance(index_data, np.ndarray):
                im = axes[1, i].imshow(index_data, cmap='RdYlGn', vmin=0, vmax=1)
                axes[1, i].set_title(f'{index_name}')
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'plots', 'comprehensive_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Комплексная визуализация сохранена: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка создания комплексных визуализаций: {e}")


def generate_analysis_report(validation_results: dict, 
                           indices_results: dict, 
                           plant_condition: dict, 
                           output_dir: str):
    """Генерация отчета анализа"""
    try:
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ АНАЛИЗА ОРТОФОТОПЛАНА\n")
            f.write("Анализ ортофотоплана и вегетационных индексов\n")
            f.write("="*60 + "\n\n")
            
            # Информация об ортофотоплане
            f.write("1. ИНФОРМАЦИЯ ОБ ОРТОФОТОПЛАНЕ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Размер: {validation_results.get('width', 'N/A')} x {validation_results.get('height', 'N/A')}\n")
            f.write(f"Каналов: {validation_results.get('bands', 'N/A')}\n")
            f.write(f"Геопривязка: {'Да' if validation_results.get('has_georeference') else 'Нет'}\n")
            f.write(f"Проекция: {'Да' if validation_results.get('has_projection') else 'Нет'}\n")
            f.write(f"Валидность: {'Да' if validation_results.get('valid') else 'Нет'}\n\n")
            
            # Результаты расчета индексов
            f.write("2. РЕЗУЛЬТАТЫ РАСЧЕТА ИНДЕКСОВ\n")
            f.write("-" * 30 + "\n")
            calculated_indices = indices_results.get('calculated_indices', [])
            f.write(f"Рассчитано индексов: {len(calculated_indices)}\n")
            f.write(f"Индексы: {', '.join(calculated_indices)}\n\n")
            
            # Статистика индексов
            normalized_indices = indices_results.get('normalized_indices', {})
            if normalized_indices:
                f.write("Статистика индексов:\n")
                for index_name, index_data in normalized_indices.items():
                    if isinstance(index_data, np.ndarray):
                        valid_data = index_data[index_data > 0]
                        if len(valid_data) > 0:
                            f.write(f"  {index_name}:\n")
                            f.write(f"    Среднее: {np.mean(valid_data):.3f}\n")
                            f.write(f"    СКО: {np.std(valid_data):.3f}\n")
                            f.write(f"    Минимум: {np.min(valid_data):.3f}\n")
                            f.write(f"    Максимум: {np.max(valid_data):.3f}\n")
                f.write("\n")
            
            # Состояние растений
            f.write("3. СОСТОЯНИЕ РАСТЕНИЙ\n")
            f.write("-" * 30 + "\n")
            if 'classification' in plant_condition:
                classification = plant_condition['classification']
                f.write(f"Класс состояния: {classification.get('class', 'N/A')}\n")
                f.write(f"Описание: {classification.get('description', 'N/A')}\n")
                f.write(f"Количественная оценка: {classification.get('score', 0):.3f}\n")
                f.write(f"Уверенность: {classification.get('confidence', 0):.2f}\n\n")
            
            # Выводы
            f.write("4. ВЫВОДЫ\n")
            f.write("-" * 30 + "\n")
            f.write("Анализ ортофотоплана выполнен успешно.\n")
            f.write("Рассчитан комплекс вегетационных индексов для оценки состояния растительности.\n")
            f.write("Выполнена сегментация изображения для выделения областей интереса.\n")
            f.write("Проведен пространственный анализ индексов.\n")
            
            if 'classification' in plant_condition:
                classification = plant_condition['classification']
                f.write(f"Общее состояние растительности: {classification.get('class', 'N/A')}\n")
                
                score = classification.get('score', 0)
                if score > 0.7:
                    f.write("Растения находятся в отличном состоянии.\n")
                elif score > 0.4:
                    f.write("Растения в удовлетворительном состоянии.\n")
                else:
                    f.write("Растения в плохом состоянии, требуется внимание.\n")
            
            f.write("\nРекомендуется провести дополнительный анализ для принятия управленческих решений.\n")
        
        print(f"Отчет анализа сохранен: {report_path}")
        
    except Exception as e:
        print(f"Ошибка генерации отчета: {e}")


if __name__ == '__main__':
    sys.exit(main())