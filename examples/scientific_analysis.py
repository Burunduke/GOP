#!/usr/bin/env python3
"""
Пример научного анализа гиперспектральных данных
с использованием библиотеки GOP v2.0
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
from src.indices.calculator import VegetationIndexCalculator
from src.processing.hyperspectral import HyperspectralProcessor
from src.utils.logger import setup_logger


def main():
    """Основная функция научного анализа"""
    
    # Настройка логирования
    logger = setup_logger('GOP_Scientific', level=logging.INFO)
    logger.info("Начало научного анализа гиперспектральных данных")
    
    try:
        # Путь к данным
        input_path = "data/research_field.bil"
        output_dir = "results/scientific_analysis"
        
        # Проверка наличия данных
        if not os.path.exists(input_path):
            logger.error(f"Входной файл не найден: {input_path}")
            return
        
        # Создание выходной директории
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Инициализация компонентов
        logger.info("Инициализация научных компонентов")
        pipeline = Pipeline()
        processor = HyperspectralProcessor()
        calculator = VegetationIndexCalculator()
        
        # Шаг 1: Анализ качества данных
        logger.info("Шаг 1: Анализ качества исходных данных")
        dataset, image_data, wavelengths = processor._read_hyperspectral_data(input_path)
        data_quality = processor._analyze_data_quality(image_data)
        
        print("\n" + "="*60)
        print("АНАЛИЗ КАЧЕСТВА ДАННЫХ")
        print("="*60)
        
        print(f"Размер изображения: {image_data.shape}")
        print(f"Всего пикселей: {data_quality['total_pixels']:,}")
        print(f"Всего каналов: {data_quality['total_bands']}")
        
        if 'overall_quality' in data_quality:
            quality = data_quality['overall_quality']
            print(f"Общая оценка качества: {quality['quality_score']:.3f}")
            print(f"Среднее SNR: {quality['average_snr']:.2f}")
        
        # Шаг 2: Обработка данных
        logger.info("Шаг 2: Научная обработка данных")
        results = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            sensor_type='Hyperspectral',
            selected_indices=['GNDVI', 'MCARI', 'MNLI', 'OSAVI', 'TVI', 
                            'SIPI2', 'mARI', 'NDWI', 'MSI'],
            use_refinement=True,
            compression_ratio=0.125
        )
        
        # Шаг 3: Детальный научный анализ
        logger.info("Шаг 3: Детальный научный анализ")
        scientific_analysis = results.get('scientific_analysis', {})
        
        print("\n" + "="*60)
        print("НАУЧНЫЙ АНАЛИЗ")
        print("="*60)
        
        # Статистический анализ
        if 'index_statistics' in scientific_analysis:
            stats = scientific_analysis['index_statistics']
            print(f"\nСТАТИСТИЧЕСКИЙ АНАЛИЗ ИНДЕКСОВ:")
            print(f"{'Индекс':<10} {'Среднее':<10} {'СКО':<10} {'Асимметрия':<12} {'Эксцесс':<10}")
            print("-" * 60)
            
            for index_name, index_stats in stats.items():
                print(f"{index_name:<10} {index_stats['mean']:<10.3f} "
                      f"{index_stats['std']:<10.3f} {index_stats['skewness']:<12.3f} "
                      f"{index_stats['kurtosis']:<10.3f}")
        
        # Корреляционный анализ
        if 'correlation_analysis' in scientific_analysis:
            corr_analysis = scientific_analysis['correlation_analysis']
            
            print(f"\nКОРРЕЛЯЦИОННЫЙ АНАЛИЗ:")
            if 'strong_correlations' in corr_analysis:
                strong_corr = corr_analysis['strong_correlations']
                print(f"Найдено сильных корреляций (|r| > 0.7): {len(strong_corr)}")
                
                for corr in strong_corr:
                    corr_type = "положительная" if corr['correlation'] > 0 else "отрицательная"
                    print(f"  {corr['index1']} - {corr['index2']}: "
                          f"{corr['correlation']:.3f} ({corr_type})")
            
            # Визуализация корреляционной матрицы
            if 'correlation_matrix' in corr_analysis:
                plot_correlation_matrix(corr_analysis, output_dir)
        
        # Пространственный анализ
        if 'spatial_analysis' in scientific_analysis:
            spatial = scientific_analysis['spatial_analysis']
            
            print(f"\nПРОСТРАНСТВЕННЫЙ АНАЛИЗ:")
            for condition_name, spatial_data in spatial.items():
                if isinstance(spatial_data, dict):
                    print(f"\n{condition_name.upper()}:")
                    print(f"  Индекс Морана: {spatial_data.get('spatial_autocorrelation', 0):.3f}")
                    
                    if 'hotspot_analysis' in spatial_data:
                        hotspot = spatial_data['hotspot_analysis']
                        print(f"  Горячие точки: {hotspot.get('hotspots', 0)} "
                              f"({hotspot.get('hotspot_percentage', 0):.1f}%)")
                        print(f"  Холодные точки: {hotspot.get('coldspots', 0)} "
                              f"({hotspot.get('coldspot_percentage', 0):.1f}%)")
                    
                    print(f"  Индекс фрагментации: {spatial_data.get('fragmentation_index', 0):.3f}")
        
        # Шаг 4: Классификация состояния растений
        logger.info("Шаг 4: Классификация состояния растений")
        plant_condition = results.get('plant_condition', {})
        
        if 'classification' in plant_condition:
            classification = plant_condition['classification']
            
            print(f"\nКЛАССИФИКАЦИЯ СОСТОЯНИЯ РАСТЕНИЙ:")
            print(f"Класс: {classification['class']}")
            print(f"Описание: {classification['description']}")
            print(f"Количественная оценка: {classification['overall_score']:.3f}")
            print(f"Уверенность классификации: {classification['confidence']:.2f}")
            print(f"Вариабельность: {classification['variability']:.3f}")
        
        # Шаг 5: Спектральный анализ
        logger.info("Шаг 5: Спектральный анализ")
        if 'scientific_report' in results['processed_data']:
            report = results['processed_data']['scientific_report']
            
            if 'spectral_analysis' in report:
                spectral = report['spectral_analysis']
                print(f"\nСПЕКТРАЛЬНЫЙ АНАЛИЗ:")
                print(f"Спектральная корреляция (оригинал-финал): {spectral.get('spectral_correlation', 0):.3f}")
                print(f"Спектральное расстояние: {spectral.get('spectral_distance', 0):.3f}")
        
        # Шаг 6: Создание научных визуализаций
        logger.info("Шаг 6: Создание научных визуализаций")
        create_scientific_plots(results, output_dir)
        
        # Шаг 7: Генерация научного отчета
        logger.info("Шаг 7: Генерация научного отчета")
        generate_scientific_report(results, data_quality, output_dir)
        
        # Шаг 8: Экспорт данных для дальнейшего анализа
        logger.info("Шаг 8: Экспорт научных данных")
        pipeline.export_scientific_data(output_dir)
        
        print("\n" + "="*60)
        print("НАУЧНЫЙ АНАЛИЗ ЗАВЕРШЕН")
        print("="*60)
        print(f"Результаты сохранены в: {output_dir}")
        print(f"Научные данные экспортированы: {output_dir}/scientific_export/")
        print(f"Визуализации: {output_dir}/plots/")
        print(f"Отчет: {output_dir}/scientific_report.txt")
        
    except Exception as e:
        logger.error(f"Ошибка в научном анализе: {e}")
        print(f"Ошибка: {e}")
        return 1
    
    return 0


def plot_correlation_matrix(corr_analysis, output_dir):
    """Создание визуализации корреляционной матрицы"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if 'correlation_matrix' not in corr_analysis or 'index_names' not in corr_analysis:
            return
        
        matrix = np.array(corr_analysis['correlation_matrix'])
        names = corr_analysis['index_names']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, 
                   xticklabels=names, 
                   yticklabels=names,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        
        plt.title('Корреляционная матрица вегетационных индексов', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'plots', 'correlation_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Корреляционная матрица сохранена: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка при создании корреляционной матрицы: {e}")


def create_scientific_plots(results, output_dir):
    """Создание научных визуализаций"""
    try:
        import matplotlib.pyplot as plt
        
        indices_results = results.get('indices', {})
        normalized_indices = indices_results.get('normalized_indices', {})
        
        if not normalized_indices:
            return
        
        # График распределения индексов
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (index_name, index_data) in enumerate(normalized_indices.items()):
            if i >= 9:
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
        for i in range(len(normalized_indices), 9):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'plots', 'index_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Распределения индексов сохранены: {plot_path}")
        
        # График сравнения индексов
        if len(normalized_indices) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Выбор пар индексов для сравнения
            index_pairs = [
                ('GNDVI', 'MCARI'),
                ('NDWI', 'MSI'),
                ('SIPI2', 'mARI'),
                ('OSAVI', 'TVI')
            ]
            
            for i, (idx1, idx2) in enumerate(index_pairs):
                if idx1 in normalized_indices and idx2 in normalized_indices:
                    data1 = normalized_indices[idx1].flatten()
                    data2 = normalized_indices[idx2].flatten()
                    
                    # Фильтрация валидных данных
                    valid_mask = (data1 > 0) & (data2 > 0)
                    data1_valid = data1[valid_mask]
                    data2_valid = data2[valid_mask]
                    
                    if len(data1_valid) > 100:
                        axes[i//2, i%2].scatter(data1_valid[:1000], data2_valid[:1000], 
                                             alpha=0.5, s=1)
                        axes[i//2, i%2].set_xlabel(idx1)
                        axes[i//2, i%2].set_ylabel(idx2)
                        axes[i//2, i%2].set_title(f'{idx1} vs {idx2}')
                        axes[i//2, i%2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'plots', 'index_comparisons.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Сравнения индексов сохранены: {plot_path}")
        
    except Exception as e:
        print(f"Ошибка при создании визуализаций: {e}")


def generate_scientific_report(results, data_quality, output_dir):
    """Генерация текстового научного отчета"""
    try:
        report_path = os.path.join(output_dir, 'scientific_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("НАУЧНЫЙ ОТЧЕТ\n")
            f.write("Анализ гиперспектральных данных и состояния растений\n")
            f.write("="*60 + "\n\n")
            
            # Информация о данных
            f.write("1. ИНФОРМАЦИЯ О ДАННЫХ\n")
            f.write("-" * 30 + "\n")
            processed_data = results['processed_data']
            f.write(f"Размер изображения: {processed_data['shape']}\n")
            f.write(f"Количество каналов: {processed_data['bands']}\n")
            
            if 'wavelengths' in processed_data and processed_data['wavelengths']:
                wavelengths = processed_data['wavelengths']
                f.write(f"Спектральный диапазон: {min(wavelengths):.1f} - {max(wavelengths):.1f} нм\n")
            
            f.write(f"Тип сенсора: {results['sensor_type']}\n\n")
            
            # Качество данных
            f.write("2. КАЧЕСТВО ДАННЫХ\n")
            f.write("-" * 30 + "\n")
            if 'overall_quality' in data_quality:
                quality = data_quality['overall_quality']
                f.write(f"Общая оценка качества: {quality['quality_score']:.3f}\n")
                f.write(f"Среднее SNR: {quality['average_snr']:.2f}\n")
            
            missing = data_quality.get('missing_values', {})
            f.write(f"Пропущенные значения: {missing.get('nan_percentage', 0):.2f}%\n")
            f.write(f"Бесконечные значения: {missing.get('inf_percentage', 0):.2f}%\n\n")
            
            # Результаты обработки
            f.write("3. РЕЗУЛЬТАТЫ ОБРАБОТКИ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Ортофотоплан: {results['orthophoto_path']}\n")
            f.write(f"Маска сегментации: {results['segmentation_mask']}\n")
            
            indices_results = results.get('indices', {})
            f.write(f"Рассчитано индексов: {len(indices_results.get('calculated_indices', []))}\n\n")
            
            # Состояние растений
            f.write("4. СОСТОЯНИЕ РАСТЕНИЙ\n")
            f.write("-" * 30 + "\n")
            plant_condition = results.get('plant_condition', {})
            if 'classification' in plant_condition:
                classification = plant_condition['classification']
                f.write(f"Класс состояния: {classification['class']}\n")
                f.write(f"Описание: {classification['description']}\n")
                f.write(f"Количественная оценка: {classification['overall_score']:.3f}\n")
                f.write(f"Уверенность: {classification['confidence']:.2f}\n\n")
            
            # Научный анализ
            f.write("5. НАУЧНЫЙ АНАЛИЗ\n")
            f.write("-" * 30 + "\n")
            scientific_analysis = results.get('scientific_analysis', {})
            
            if 'index_statistics' in scientific_analysis:
                f.write("Статистика индексов:\n")
                stats = scientific_analysis['index_statistics']
                for index_name, index_stats in stats.items():
                    f.write(f"  {index_name}: среднее={index_stats['mean']:.3f}, "
                           f"СКО={index_stats['std']:.3f}\n")
                f.write("\n")
            
            if 'correlation_analysis' in scientific_analysis:
                corr_analysis = scientific_analysis['correlation_analysis']
                if 'strong_correlations' in corr_analysis:
                    f.write("Сильные корреляции:\n")
                    for corr in corr_analysis['strong_correlations']:
                        f.write(f"  {corr['index1']} - {corr['index2']}: "
                               f"{corr['correlation']:.3f}\n")
                f.write("\n")
            
            # Выводы
            f.write("6. ВЫВОДЫ\n")
            f.write("-" * 30 + "\n")
            
            if 'classification' in plant_condition:
                classification = plant_condition['classification']
                f.write(f"Состояние растительности на исследуемом участке: {classification['class']}\n")
                f.write(f"Количественная оценка: {classification['overall_score']:.3f}\n")
                
                if classification['overall_score'] > 0.7:
                    f.write("Растения находятся в отличном состоянии, высокий уровень фотосинтетической активности.\n")
                elif classification['overall_score'] > 0.4:
                    f.write("Растения в удовлетворительном состоянии, наблюдаются умеренные стрессовые факторы.\n")
                else:
                    f.write("Растения в плохом состоянии, требуется вмешательство и дополнительный анализ.\n")
            
            f.write("\nНаучный анализ завершен. Рекомендуется провести дополнительный полевой анализ для валидации результатов.\n")
        
        print(f"Научный отчет сохранен: {report_path}")
        
    except Exception as e:
        print(f"Ошибка при генерации отчета: {e}")


if __name__ == '__main__':
    sys.exit(main())