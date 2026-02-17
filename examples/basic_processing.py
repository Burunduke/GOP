#!/usr/bin/env python3
"""
Базовый пример обработки гиперспектральных данных
с использованием научной библиотеки GOP v2.0
"""

import os
import sys
import logging
from pathlib import Path

# Добавление src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.pipeline import Pipeline
from src.core.config import config
from src.utils.logger import setup_logger


def main():
    """Основная функция базового примера"""
    
    # Настройка логирования
    logger = setup_logger('GOP_Example', level=logging.INFO)
    logger.info("Начало базового примера обработки гиперспектральных данных")
    
    try:
        # Путь к входным данным (замените на свой путь)
        input_path = "data/sample_field.bil"
        output_dir = "results/basic_processing"
        
        # Проверка наличия входных данных
        if not os.path.exists(input_path):
            logger.error(f"Входной файл не найден: {input_path}")
            logger.info("Пожалуйста, укажите корректный путь к гиперспектральным данным")
            return
        
        # Создание выходной директории
        os.makedirs(output_dir, exist_ok=True)
        
        # Инициализация пайплайна
        logger.info("Инициализация научного пайплайна")
        pipeline = Pipeline()
        
        # Обработка данных с научным анализом
        logger.info(f"Обработка файла: {input_path}")
        results = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            sensor_type='Hyperspectral',
            selected_indices=['GNDVI', 'MCARI', 'NDWI', 'MSI', 'SIPI2'],
            use_refinement=True,
            compression_ratio=0.125
        )
        
        # Вывод результатов
        logger.info("Обработка завершена успешно")
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
        print("="*60)
        
        print(f"Входной файл: {results['input_path']}")
        print(f"Ортофотоплан: {results['orthophoto_path']}")
        print(f"Маска сегментации: {results['segmentation_mask']}")
        print(f"Тип сенсора: {results['sensor_type']}")
        print(f"Размер данных: {results['processed_data']['shape']}")
        print(f"Количество каналов: {results['processed_data']['bands']}")
        
        # Анализ состояния растений
        plant_condition = results.get('plant_condition', {})
        if 'classification' in plant_condition:
            classification = plant_condition['classification']
            print(f"\nСОСТОЯНИЕ РАСТЕНИЙ:")
            print(f"  Класс: {classification['class']}")
            print(f"  Описание: {classification['description']}")
            print(f"  Оценка: {classification['overall_score']:.3f}")
            print(f"  Уверенность: {classification['confidence']:.2f}")
        
        # Научный анализ
        scientific_analysis = results.get('scientific_analysis', {})
        if scientific_analysis:
            print(f"\nНАУЧНЫЙ АНАЛИЗ:")
            
            # Статистика индексов
            if 'index_statistics' in scientific_analysis:
                stats = scientific_analysis['index_statistics']
                print(f"  Рассчитано индексов: {len(stats)}")
                for index_name, index_stats in list(stats.items())[:3]:
                    print(f"    {index_name}: среднее={index_stats['mean']:.3f}, СКО={index_stats['std']:.3f}")
            
            # Корреляционный анализ
            if 'correlation_analysis' in scientific_analysis:
                corr_analysis = scientific_analysis['correlation_analysis']
                if 'strong_correlations' in corr_analysis:
                    strong_corr = corr_analysis['strong_correlations']
                    print(f"  Сильных корреляций: {len(strong_corr)}")
                    for corr in strong_corr[:3]:
                        print(f"    {corr['index1']} - {corr['index2']}: {corr['correlation']:.3f}")
            
            # Пространственный анализ
            if 'spatial_analysis' in scientific_analysis:
                spatial = scientific_analysis['spatial_analysis']
                if 'overall' in spatial:
                    overall_spatial = spatial['overall']
                    print(f"  Пространственная автокорреляция: {overall_spatial.get('spatial_autocorrelation', 0):.3f}")
        
        # Сохранение результатов
        results_file = os.path.join(output_dir, 'processing_results.json')
        pipeline.save_results(results_file)
        print(f"\nРезультаты сохранены: {results_file}")
        
        # Экспорт научных данных
        pipeline.export_scientific_data(output_dir)
        print(f"Научные данные экспортированы: {output_dir}/scientific_export/")
        
        # Качество данных
        data_quality = results['processed_data'].get('data_quality', {})
        if data_quality and 'overall_quality' in data_quality:
            quality = data_quality['overall_quality']
            print(f"\nКАЧЕСТВО ДАННЫХ:")
            print(f"  Общая оценка: {quality.get('quality_score', 0):.3f}")
            print(f"  Среднее SNR: {quality.get('average_snr', 0):.2f}")
        
        print("\n" + "="*60)
        print("Базовый пример завершен успешно!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Ошибка в базовом примере: {e}")
        print(f"Ошибка: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())