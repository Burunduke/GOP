"""
Основной пайплайн обработки гиперспектральных данных
Научно-ориентированная архитектура без GUI зависимостей
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np

from .config import config
from ..processing.hyperspectral import HyperspectralProcessor
from ..processing.orthophoto import OrthophotoProcessor
from ..segmentation.segmenter import ImageSegmenter
from ..indices.calculator import VegetationIndexCalculator
from ..utils.logger import setup_logger


class Pipeline:
    """
    Основной класс пайплайна для обработки гиперспектральных данных
    Научно-ориентированная архитектура для обработки данных и анализа растений
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация пайплайна
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        # Загрузка конфигурации
        if config_path:
            config.config_path = config_path
            config._config = config._load_config()
        
        # Настройка логирования
        self.logger = setup_logger(
            name='GOP',
            level=config.get('logging.level', 'INFO'),
            log_file=config.get('logging.file')
        )
        
        # Инициализация компонентов
        self.hyperspectral_processor = HyperspectralProcessor()
        self.orthophoto_processor = OrthophotoProcessor()
        self.segmenter = ImageSegmenter()
        self.index_calculator = VegetationIndexCalculator()
        
        # Результаты обработки
        self.results = {}
        
        self.logger.info("Научный пайплайн GOP инициализирован")
    
    def process(self, 
                input_path: str,
                output_dir: str = None,
                sensor_type: str = 'Hyperspectral',
                segmentation_mask: Optional[str] = None,
                selected_indices: Optional[List[str]] = None,
                use_refinement: bool = True,
                compression_ratio: float = None) -> Dict[str, Any]:
        """
        Полный цикл обработки данных с научной методикой
        
        Args:
            input_path: Путь к входным данным
            output_dir: Директория для сохранения результатов
            sensor_type: Тип сенсора ('RGB', 'Multispectral', 'Hyperspectral')
            segmentation_mask: Путь к маске сегментации (если None, будет создана)
            selected_indices: Список индексов для расчета
            use_refinement: Использовать уточнение границ сегментации
            compression_ratio: Коэффициент сжатия для сегментации
            
        Returns:
            Словарь с результатами обработки
        """
        try:
            self.logger.info(f"Начало научной обработки данных: {input_path}")
            
            # Настройка выходной директории
            output_dir = output_dir or config.get('output.results_dir', 'results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Этап 1: Предварительная обработка гиперспектральных данных
            self.logger.info("Этап 1: Предварительная обработка гиперспектральных данных")
            processed_data = self._preprocess_hyperspectral(input_path, output_dir)
            
            # Этап 2: Создание ортофотоплана
            self.logger.info("Этап 2: Создание ортофотоплана")
            orthophoto_path = self._create_orthophoto(processed_data, output_dir)
            
            # Этап 3: Сегментация изображений сверхвысокого разрешения
            self.logger.info("Этап 3: Сегментация изображений")
            if segmentation_mask is None:
                segmentation_mask = self._segment_image(
                    orthophoto_path, output_dir, use_refinement, compression_ratio
                )
            
            # Этап 4: Расчет вегетационных индексов
            self.logger.info("Этап 4: Расчет вегетационных индексов")
            indices_results = self._calculate_indices(
                orthophoto_path, segmentation_mask, sensor_type, 
                selected_indices, output_dir
            )
            
            # Этап 5: Комплексная оценка состояния растений
            self.logger.info("Этап 5: Комплексная оценка состояния растений")
            plant_condition = self._assess_plant_condition(indices_results)
            
            # Этап 6: Научный анализ и статистика
            self.logger.info("Этап 6: Научный анализ и статистика")
            scientific_analysis = self._perform_scientific_analysis(
                indices_results, plant_condition, output_dir
            )
            
            # Сбор результатов
            self.results = {
                'input_path': input_path,
                'output_dir': output_dir,
                'sensor_type': sensor_type,
                'processed_data': processed_data,
                'orthophoto_path': orthophoto_path,
                'segmentation_mask': segmentation_mask,
                'indices': indices_results,
                'plant_condition': plant_condition,
                'scientific_analysis': scientific_analysis,
                'processing_metadata': self._get_processing_metadata()
            }
            
            self.logger.info("Научная обработка завершена успешно")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Ошибка в процессе научной обработки: {e}")
            raise
    
    def _preprocess_hyperspectral(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Предварительная обработка гиперспектральных данных
        
        Args:
            input_path: Путь к входным данным
            output_dir: Директория для сохранения результатов
            
        Returns:
            Словарь с результатами предобработки
        """
        return self.hyperspectral_processor.process(input_path, output_dir)
    
    def _create_orthophoto(self, processed_data: Dict[str, Any], output_dir: str) -> str:
        """
        Создание ортофотоплана
        
        Args:
            processed_data: Результаты предобработки
            output_dir: Директория для сохранения результатов
            
        Returns:
            Путь к созданному ортофотоплану
        """
        return self.orthophoto_processor.create_orthophoto(processed_data, output_dir)
    
    def _segment_image(self, 
                      orthophoto_path: str, 
                      output_dir: str,
                      use_refinement: bool = True,
                      compression_ratio: float = None) -> str:
        """
        Сегментация изображения с использованием каскадного подхода
        
        Args:
            orthophoto_path: Путь к ортофотоплану
            output_dir: Директория для сохранения результатов
            use_refinement: Использовать уточнение границ
            compression_ratio: Коэффициент сжатия
            
        Returns:
            Путь к маске сегментации
        """
        return self.segmenter.segment(
            orthophoto_path, output_dir, use_refinement, compression_ratio
        )
    
    def _calculate_indices(self, 
                          orthophoto_path: str,
                          segmentation_mask: str,
                          sensor_type: str,
                          selected_indices: Optional[List[str]],
                          output_dir: str) -> Dict[str, Any]:
        """
        Расчет вегетационных индексов
        
        Args:
            orthophoto_path: Путь к ортофотоплану
            segmentation_mask: Путь к маске сегментации
            sensor_type: Тип сенсора
            selected_indices: Список индексов для расчета
            output_dir: Директория для сохранения результатов
            
        Returns:
            Словарь с результатами расчета индексов
        """
        if selected_indices is None:
            selected_indices = config.get('indices.default_indices', [])
        
        return self.index_calculator.calculate(
            orthophoto_path, segmentation_mask, sensor_type, 
            selected_indices, output_dir
        )
    
    def _assess_plant_condition(self, indices_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Комплексная оценка состояния растений
        
        Args:
            indices_results: Результаты расчета индексов
            
        Returns:
            Словарь с оценкой состояния растений
        """
        return self.index_calculator.assess_plant_condition(indices_results)
    
    def _perform_scientific_analysis(self, 
                                   indices_results: Dict[str, Any],
                                   plant_condition: Dict[str, Any],
                                   output_dir: str) -> Dict[str, Any]:
        """
        Научный анализ результатов
        
        Args:
            indices_results: Результаты расчета индексов
            plant_condition: Оценка состояния растений
            output_dir: Директория для сохранения результатов
            
        Returns:
            Словарь с научным анализом
        """
        try:
            analysis = {}
            
            # Статистический анализ индексов
            analysis['index_statistics'] = self._analyze_index_statistics(indices_results)
            
            # Корреляционный анализ
            analysis['correlation_analysis'] = self._perform_correlation_analysis(indices_results)
            
            # Пространственный анализ
            analysis['spatial_analysis'] = self._perform_spatial_analysis(plant_condition)
            
            # Классификация состояния растений
            analysis['plant_classification'] = self._classify_plant_condition(plant_condition)
            
            # Сохранение научного отчета
            self._save_scientific_report(analysis, output_dir)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Ошибка научного анализа: {e}")
            return {'error': str(e)}
    
    def _analyze_index_statistics(self, indices_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Статистический анализ вегетационных индексов
        
        Args:
            indices_results: Результаты расчета индексов
            
        Returns:
            Словарь со статистикой
        """
        statistics = {}
        normalized_indices = indices_results.get('normalized_indices', {})
        
        for index_name, index_data in normalized_indices.items():
            if isinstance(index_data, np.ndarray):
                # Расчет статистики только для области маски
                valid_data = index_data[index_data > 0]
                
                if len(valid_data) > 0:
                    statistics[index_name] = {
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'median': float(np.median(valid_data)),
                        'q25': float(np.percentile(valid_data, 25)),
                        'q75': float(np.percentile(valid_data, 75)),
                        'skewness': float(self._calculate_skewness(valid_data)),
                        'kurtosis': float(self._calculate_kurtosis(valid_data))
                    }
        
        return statistics
    
    def _perform_correlation_analysis(self, indices_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Корреляционный анализ индексов
        
        Args:
            indices_results: Результаты расчета индексов
            
        Returns:
            Словарь с корреляционным анализом
        """
        try:
            normalized_indices = indices_results.get('normalized_indices', {})
            
            # Создание матрицы данных для корреляционного анализа
            index_names = []
            index_vectors = []
            
            for index_name, index_data in normalized_indices.items():
                if isinstance(index_data, np.ndarray):
                    valid_data = index_data[index_data > 0]
                    if len(valid_data) > 100:  # Минимальное количество точек для анализа
                        index_names.append(index_name)
                        index_vectors.append(valid_data)
            
            if len(index_vectors) < 2:
                return {'error': 'Недостаточно данных для корреляционного анализа'}
            
            # Выравнивание векторов по минимальной длине
            min_length = min(len(vec) for vec in index_vectors)
            aligned_vectors = [vec[:min_length] for vec in index_vectors]
            
            # Расчет корреляционной матрицы
            correlation_matrix = np.corrcoef(aligned_vectors)
            
            # Формирование результатов
            correlation_analysis = {
                'index_names': index_names,
                'correlation_matrix': correlation_matrix.tolist(),
                'strong_correlations': self._find_strong_correlations(
                    index_names, correlation_matrix, threshold=0.7
                )
            }
            
            return correlation_analysis
            
        except Exception as e:
            self.logger.error(f"Ошибка корреляционного анализа: {e}")
            return {'error': str(e)}
    
    def _perform_spatial_analysis(self, plant_condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Пространственный анализ состояния растений
        
        Args:
            plant_condition: Оценка состояния растений
            
        Returns:
            Словарь с пространственным анализом
        """
        try:
            condition_maps = plant_condition.get('condition_maps', {})
            spatial_analysis = {}
            
            for condition_name, condition_data in condition_maps.items():
                if isinstance(condition_data, np.ndarray):
                    spatial_analysis[condition_name] = {
                        'spatial_autocorrelation': self._calculate_morans_i(condition_data),
                        'hotspot_analysis': self._perform_hotspot_analysis(condition_data),
                        'fragmentation_index': self._calculate_fragmentation_index(condition_data)
                    }
            
            return spatial_analysis
            
        except Exception as e:
            self.logger.error(f"Ошибка пространственного анализа: {e}")
            return {'error': str(e)}
    
    def _classify_plant_condition(self, plant_condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Классификация состояния растений
        
        Args:
            plant_condition: Оценка состояния растений
            
        Returns:
            Словарь с классификацией
        """
        try:
            statistics = plant_condition.get('statistics', {})
            overall_stats = statistics.get('overall', {})
            
            if not overall_stats:
                return {'error': 'Отсутствуют данные для классификации'}
            
            overall_mean = overall_stats.get('mean', 0)
            overall_std = overall_stats.get('std', 0)
            
            # Научная классификация на основе среднего значения и вариабельности
            if overall_mean > 0.8 and overall_std < 0.1:
                condition_class = 'Отличное'
                condition_description = 'Растения в отличном состоянии, высокая однородность'
                confidence = 0.9
            elif overall_mean > 0.6 and overall_std < 0.2:
                condition_class = 'Хорошее'
                condition_description = 'Растения в хорошем состоянии, умеренная однородность'
                confidence = 0.8
            elif overall_mean > 0.4:
                condition_class = 'Удовлетворительное'
                condition_description = 'Растения в удовлетворительном состоянии, есть проблемы'
                confidence = 0.7
            else:
                condition_class = 'Плохое'
                condition_description = 'Растения в плохом состоянии, требуется вмешательство'
                confidence = 0.8
            
            return {
                'class': condition_class,
                'description': condition_description,
                'confidence': confidence,
                'overall_score': overall_mean,
                'variability': overall_std
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка классификации: {e}")
            return {'error': str(e)}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Расчет асимметрии распределения"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Расчет эксцесса распределения"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _find_strong_correlations(self, 
                                index_names: List[str], 
                                correlation_matrix: np.ndarray,
                                threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Поиск сильных корреляций между индексами"""
        strong_correlations = []
        
        for i in range(len(index_names)):
            for j in range(i + 1, len(index_names)):
                corr_value = correlation_matrix[i, j]
                if abs(corr_value) > threshold:
                    strong_correlations.append({
                        'index1': index_names[i],
                        'index2': index_names[j],
                        'correlation': float(corr_value),
                        'type': 'positive' if corr_value > 0 else 'negative'
                    })
        
        return strong_correlations
    
    def _calculate_morans_i(self, data: np.ndarray) -> float:
        """Расчет индекса пространственной автокорреляции Морана"""
        try:
            # Упрощенная реализация индекса Морана
            rows, cols = data.shape
            if rows < 3 or cols < 3:
                return 0.0
            
            # Создание весовой матрицы (соседство)
            weights = np.zeros((rows, cols, rows, cols))
            
            for i in range(rows):
                for j in range(cols):
                    # Проверка соседних пикселей
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                weights[i, j, ni, nj] = 1
            
            # Расчет индекса Морана
            n = rows * cols
            mean_val = np.mean(data)
            
            numerator = 0
            denominator = 0
            weight_sum = 0
            
            for i in range(rows):
                for j in range(cols):
                    for ni in range(rows):
                        for nj in range(cols):
                            if weights[i, j, ni, nj] > 0:
                                numerator += weights[i, j, ni, nj] * (data[i, j] - mean_val) * (data[ni, nj] - mean_val)
                                weight_sum += weights[i, j, ni, nj]
                    
                    denominator += (data[i, j] - mean_val) ** 2
            
            if weight_sum == 0 or denominator == 0:
                return 0.0
            
            morans_i = (n / weight_sum) * (numerator / denominator)
            return float(morans_i)
            
        except Exception:
            return 0.0
    
    def _perform_hotspot_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Анализ горячих точек"""
        try:
            # Упрощенный анализ горячих точек на основе z-оценок
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0:
                return {'hotspots': 0, 'coldspots': 0, 'neutral': data.size}
            
            # Классификация пикселей
            z_scores = (data - mean_val) / std_val
            
            hotspots = np.sum(z_scores > 1.96)  # p < 0.05
            coldspots = np.sum(z_scores < -1.96)  # p < 0.05
            neutral = data.size - hotspots - coldspots
            
            return {
                'hotspots': int(hotspots),
                'coldspots': int(coldspots),
                'neutral': int(neutral),
                'hotspot_percentage': float(hotspots / data.size * 100),
                'coldspot_percentage': float(coldspots / data.size * 100)
            }
            
        except Exception:
            return {'hotspots': 0, 'coldspots': 0, 'neutral': data.size}
    
    def _calculate_fragmentation_index(self, data: np.ndarray) -> float:
        """Расчет индекса фрагментации"""
        try:
            # Бинаризация данных
            threshold = np.mean(data)
            binary = (data > threshold).astype(np.uint8)
            
            # Подсчет связанных компонентов
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)
            
            # Расчет индекса фрагментации
            if num_features == 0:
                return 0.0
            
            total_area = np.sum(binary)
            if total_area == 0:
                return 0.0
            
            fragmentation = num_features / total_area
            return float(fragmentation)
            
        except Exception:
            return 0.0
    
    def _save_scientific_report(self, analysis: Dict[str, Any], output_dir: str) -> None:
        """Сохранение научного отчета"""
        try:
            report_path = os.path.join(output_dir, 'scientific_report.json')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Научный отчет сохранен: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения научного отчета: {e}")
    
    def _get_processing_metadata(self) -> Dict[str, Any]:
        """Получение метаданных обработки"""
        return {
            'pipeline_version': '2.0.0',
            'processing_date': str(Path.cwd()),
            'config_used': config.config,
            'scientific_methods': [
                'radiometric_correction',
                'pca_denoising',
                'vegetation_indices',
                'spatial_analysis',
                'correlation_analysis',
                'statistical_analysis'
            ]
        }
    
    def get_results(self) -> Dict[str, Any]:
        """
        Получить результаты последней обработки
        
        Returns:
            Словарь с результатами
        """
        return self.results.copy()
    
    def save_results(self, output_path: str) -> None:
        """
        Сохранить результаты обработки в файл
        
        Args:
            output_path: Путь для сохранения результатов
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Результаты сохранены в: {output_path}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
    
    def export_scientific_data(self, output_dir: str) -> None:
        """
        Экспорт научных данных для дальнейшего анализа
        
        Args:
            output_dir: Директория для экспорта
        """
        try:
            import pandas as pd
            
            # Создание директории для экспорта
            export_dir = os.path.join(output_dir, 'scientific_export')
            os.makedirs(export_dir, exist_ok=True)
            
            # Экспорт статистики индексов
            if 'scientific_analysis' in self.results:
                analysis = self.results['scientific_analysis']
                
                if 'index_statistics' in analysis:
                    stats_df = pd.DataFrame(analysis['index_statistics']).T
                    stats_path = os.path.join(export_dir, 'index_statistics.csv')
                    stats_df.to_csv(stats_path)
                    self.logger.info(f"Статистика индексов экспортирована: {stats_path}")
                
                if 'correlation_analysis' in analysis:
                    corr_data = analysis['correlation_analysis']
                    if 'correlation_matrix' in corr_data:
                        corr_df = pd.DataFrame(
                            corr_data['correlation_matrix'],
                            index=corr_data['index_names'],
                            columns=corr_data['index_names']
                        )
                        corr_path = os.path.join(export_dir, 'correlation_matrix.csv')
                        corr_df.to_csv(corr_path)
                        self.logger.info(f"Корреляционная матрица экспортирована: {corr_path}")
            
            self.logger.info(f"Научные данные экспортированы в: {export_dir}")
            
        except Exception as e:
            self.logger.error(f"Ошибка экспорта научных данных: {e}")