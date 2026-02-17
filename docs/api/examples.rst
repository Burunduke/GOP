Примеры использования
=====================

В этом разделе представлены практические примеры использования библиотеки GOP для решения различных задач обработки гиперспектральных данных.

.. toctree::
   :maxdepth: 2

   basic_processing
   scientific_analysis
   batch_processing
   custom_indices
   visualization_examples

Базовая обработка гиперспектральных данных
-------------------------------------------

Пример простой обработки гиперспектральных данных с использованием стандартного пайплайна:

.. code-block:: python

   """
   Базовая обработка гиперспектральных данных
   """
   
   from src.core.pipeline import Pipeline
   from src.utils.logger import setup_logger
   
   # Настройка логирования
   logger = setup_logger('BasicProcessing', level='INFO')
   
   def basic_processing_example():
       """Пример базовой обработки"""
       
       # Создание пайплайна
       pipeline = Pipeline()
       
       # Обработка данных
       try:
           results = pipeline.process(
               input_path='data/hyperspectral.bil',
               output_dir='results/basic_processing/',
               sensor_type='Hyperspectral',
               selected_indices=['GNDVI', 'NDWI', 'MCARI']
           )
           
           # Анализ результатов
           logger.info("Обработка завершена успешно")
           logger.info(f"Ортофотоплан: {results['orthophoto_path']}")
           logger.info(f"Маска сегментации: {results['segmentation_mask']}")
           
           # Оценка состояния растений
           plant_condition = results['plant_condition']['classification']
           logger.info(f"Состояние растений: {plant_condition['class']}")
           logger.info(f"Оценка: {plant_condition['score']:.3f}")
           
           return results
           
       except Exception as e:
           logger.error(f"Ошибка обработки: {e}")
           return None
   
   if __name__ == '__main__':
       results = basic_processing_example()

Научный анализ данных
---------------------

Пример комплексного научного анализа с расчетом статистики и корреляций:

.. code-block:: python

   """
   Научный анализ гиперспектральных данных
   """
   
   import numpy as np
   import pandas as pd
   from src.core.pipeline import Pipeline
   from src.utils.visualization import visualize_indices, create_comparison_plot
   from src.utils.logger import setup_logger
   
   def scientific_analysis_example():
       """Пример научного анализа"""
       
       logger = setup_logger('ScientificAnalysis', level='INFO')
       
       # Создание пайплайна с научными настройками
       pipeline = Pipeline()
       
       # Обработка с полным набором индексов
       results = pipeline.process(
           input_path='data/research_area.bil',
           output_dir='results/scientific_analysis/',
           sensor_type='Hyperspectral',
           selected_indices=[
               'GNDVI', 'MCARI', 'MNLI', 'OSAVI', 'TVI',
               'SIPI2', 'mARI', 'NDWI', 'MSI'
           ]
       )
       
       # Научный анализ результатов
       scientific_data = results['scientific_analysis']
       
       # Анализ статистики индексов
       index_stats = scientific_data.get('index_statistics', {})
       
       # Создание таблицы статистики
       stats_df = pd.DataFrame(index_stats).T
       stats_df.to_csv('results/scientific_analysis/index_statistics.csv')
       
       logger.info("Статистика индексов сохранена")
       
       # Корреляционный анализ
       correlation_data = scientific_data.get('correlation_analysis', {})
       
       if 'correlation_matrix' in correlation_data:
           corr_df = pd.DataFrame(
               correlation_data['correlation_matrix'],
               index=correlation_data['index_names'],
               columns=correlation_data['index_names']
           )
           corr_df.to_csv('results/scientific_analysis/correlation_matrix.csv')
           
           # Поиск сильных корреляций
           strong_correlations = correlation_data.get('strong_correlations', [])
           for corr in strong_correlations:
               logger.info(f"Сильная корреляция: {corr['index1']} - {corr['index2']} = {corr['correlation']:.3f}")
       
       # Пространственный анализ
       spatial_data = scientific_data.get('spatial_analysis', {})
       
       # Визуализация результатов
       indices = results['indices']['normalized_indices']
       
       # Визуализация индексов
       fig = visualize_indices(
           indices_dict=indices,
           output_path='results/scientific_analysis/indices_visualization.png'
       )
       
       # Создание сравнительного графика
       from src.utils.image_utils import load_image
       
       original_image = load_image(results['orthophoto_path'])
       segmentation_mask = load_image(results['segmentation_mask'], mode='L')
       
       fig = create_comparison_plot(
           original_image=original_image,
           segmentation_mask=segmentation_mask,
           indices_dict=indices,
           output_path='results/scientific_analysis/comparison_plot.png'
       )
       
       # Классификация состояния растений
       plant_classification = scientific_data.get('plant_classification', {})
       logger.info(f"Классификация состояния: {plant_classification.get('class', 'Неизвестно')}")
       logger.info(f"Уверенность: {plant_classification.get('confidence', 0):.3f}")
       
       return results
   
   if __name__ == '__main__':
       results = scientific_analysis_example()

Пакетная обработка данных
--------------------------

Пример пакетной обработки множества файлов с автоматическим созданием отчетов:

.. code-block:: python

   """
   Пакетная обработка гиперспектральных данных
   """
   
   import os
   import glob
   import pandas as pd
   from datetime import datetime
   from src.core.pipeline import Pipeline
   from src.utils.logger import setup_logger
   from src.utils.file_utils import ensure_dir
   
   def batch_processing_example(input_dir, output_dir):
       """Пример пакетной обработки"""
       
       logger = setup_logger('BatchProcessing', level='INFO')
       
       # Создание выходной директории
       ensure_dir(output_dir)
       
       # Поиск файлов для обработки
       search_pattern = os.path.join(input_dir, '*.bil')
       files = glob.glob(search_pattern)
       
       if not files:
           logger.error(f"Файлы не найдены по шаблону: {search_pattern}")
           return
       
       logger.info(f"Найдено файлов для обработки: {len(files)}")
       
       # Создание пайплайна
       pipeline = Pipeline()
       
       # Список для результатов
       processing_results = []
       
       # Обработка файлов
       for i, file_path in enumerate(files, 1):
           try:
               logger.info(f"Обработка файла {i}/{len(files)}: {os.path.basename(file_path)}")
               
               # Создание индивидуальной выходной директории
               file_name = os.path.splitext(os.path.basename(file_path))[0]
               file_output_dir = os.path.join(output_dir, file_name)
               
               # Обработка
               results = pipeline.process(
                   input_path=file_path,
                   output_dir=file_output_dir,
                   sensor_type='Hyperspectral',
                   selected_indices=['GNDVI', 'NDWI', 'MCARI']
               )
               
               # Сбор результатов
               file_result = {
                   'filename': os.path.basename(file_path),
                   'status': 'success',
                   'orthophoto_path': results['orthophoto_path'],
                   'segmentation_mask': results['segmentation_mask'],
                   'processing_time': datetime.now().isoformat()
               }
               
               # Оценка состояния растений
               plant_condition = results['plant_condition']['classification']
               file_result['plant_condition'] = plant_condition['class']
               file_result['condition_score'] = plant_condition['score']
               
               # Статистика индексов
               indices = results['indices']['indices_values']
               for index_name, values in indices.items():
                   valid_values = values[values > 0]
                   if len(valid_values) > 0:
                       file_result[f'{index_name}_mean'] = valid_values.mean()
                       file_result[f'{index_name}_std'] = valid_values.std()
               
               processing_results.append(file_result)
               logger.info(f"Файл обработан успешно: {file_name}")
               
           except Exception as e:
               logger.error(f"Ошибка обработки файла {file_path}: {e}")
               
               # Добавление информации об ошибке
               file_result = {
                   'filename': os.path.basename(file_path),
                   'status': 'error',
                   'error_message': str(e),
                   'processing_time': datetime.now().isoformat()
               }
               processing_results.append(file_result)
       
       # Создание сводного отчета
       results_df = pd.DataFrame(processing_results)
       results_df.to_csv(os.path.join(output_dir, 'batch_processing_report.csv'), index=False)
       
       # Статистика обработки
       success_count = len([r for r in processing_results if r['status'] == 'success'])
       error_count = len([r for r in processing_results if r['status'] == 'error'])
       
       logger.info(f"Пакетная обработка завершена:")
       logger.info(f"Успешно: {success_count}")
       logger.info(f"С ошибками: {error_count}")
       logger.info(f"Всего: {len(processing_results)}")
       
       # Создание сводной статистики
       if success_count > 0:
           success_df = results_df[results_df['status'] == 'success']
           
           # Статистика по состоянию растений
           condition_counts = success_df['plant_condition'].value_counts()
           logger.info("Распределение состояния растений:")
           for condition, count in condition_counts.items():
               logger.info(f"  {condition}: {count}")
           
           # Средние значения индексов
           index_columns = [col for col in success_df.columns if col.endswith('_mean')]
           if index_columns:
               index_means = success_df[index_columns].mean()
               logger.info("Средние значения индексов по всем файлам:")
               for index, mean_val in index_means.items():
                   logger.info(f"  {index}: {mean_val:.3f}")
       
       return processing_results
   
   if __name__ == '__main__':
       results = batch_processing_example(
           input_dir='data/batch_input/',
           output_dir='results/batch_processing/'
       )

Создание кастомных вегетационных индексов
-----------------------------------------

Пример создания и использования собственных вегетационных индексов:

.. code-block:: python

   """
   Создание и использование кастомных вегетационных индексов
   """
   
   import numpy as np
   from src.indices.definitions import IndexDefinitions
   from src.indices.calculator import VegetationIndexCalculator
   from src.utils.logger import setup_logger
   
   def add_custom_indices():
       """Добавление кастомных индексов"""
       
       logger = setup_logger('CustomIndices', level='INFO')
       
       # Определение кастомных индексов
       custom_indices = {
           'CustomNDVI': {
               'name': 'Custom Normalized Difference Vegetation Index',
               'formula': '(NIR - Red) / (NIR + Red)',
               'description': 'Модифицированный NDVI с улучшенной чувствительностью',
               'required_bands': ['NIR', 'Red'],
               'range': (-1, 1),
               'function': lambda nir, red: (nir - red) / (nir + red + 1e-8)
           },
           'EnhancedGVI': {
               'name': 'Enhanced Green Vegetation Index',
               'formula': '(2*NIR - Green - Red) / (2*NIR + Green + Red)',
               'description': 'Улучшенный зеленый вегетационный индекс',
               'required_bands': ['NIR', 'Green', 'Red'],
               'range': (-1, 1),
               'function': lambda nir, green, red: (2*nir - green - red) / (2*nir + green + red + 1e-8)
           },
           'StressIndex': {
               'name': 'Plant Stress Index',
               'formula': '(Red - Green) / (Red + Green)',
               'description': 'Индекс стресса растений на основе красного и зеленого каналов',
               'required_bands': ['Red', 'Green'],
               'range': (-1, 1),
               'function': lambda red, green: (red - green) / (red + green + 1e-8)
           }
       }
       
       # Добавление в существующие определения
       IndexDefinitions.ALL_INDICES.update(custom_indices)
       
       # Создание новой группы индексов
       IndexDefinitions.INDEX_GROUPS['custom'] = list(custom_indices.keys())
       
       logger.info(f"Добавлено {len(custom_indices)} кастомных индексов")
       
       return custom_indices
   
   def calculate_custom_indices_example():
       """Пример расчета кастомных индексов"""
       
       logger = setup_logger('CustomIndicesCalculation', level='INFO')
       
       # Добавление кастомных индексов
       custom_indices = add_custom_indices()
       
       # Создание тестовых данных
       bands = {
           'NIR': np.random.rand(100, 100) * 0.8 + 0.2,
           'Red': np.random.rand(100, 100) * 0.6 + 0.1,
           'Green': np.random.rand(100, 100) * 0.7 + 0.2
       }
       
       # Расчет кастомных индексов
       custom_results = {}
       
       for index_name in custom_indices.keys():
           try:
               # Расчет значений индекса
               index_values = IndexDefinitions.calculate_index(index_name, bands)
               
               # Нормализация
               mask = np.ones((100, 100), dtype=np.uint8)
               normalized_values = IndexDefinitions.normalize_index(index_name, index_values, mask)
               
               custom_results[index_name] = {
                   'values': index_values,
                   'normalized': normalized_values,
                   'mean': index_values.mean(),
                   'std': index_values.std()
               }
               
               logger.info(f"Рассчитан индекс {index_name}: среднее = {index_values.mean():.3f}")
               
           except Exception as e:
               logger.error(f"Ошибка расчета индекса {index_name}: {e}")
       
       return custom_results
   
   def integrate_custom_indices_pipeline():
       """Интеграция кастомных индексов в пайплайн"""
       
       logger = setup_logger('CustomPipeline', level='INFO')
       
       # Добавление кастомных индексов
       add_custom_indices()
       
       # Создание пайплайна
       from src.core.pipeline import Pipeline
       pipeline = Pipeline()
       
       # Обработка с использованием кастомных индексов
       results = pipeline.process(
           input_path='data/hyperspectral.bil',
           output_dir='results/custom_indices/',
           sensor_type='Hyperspectral',
           selected_indices=['GNDVI', 'CustomNDVI', 'EnhancedGVI', 'StressIndex']
       )
       
       # Сравнение стандартных и кастомных индексов
       indices = results['indices']['indices_values']
       
       logger.info("Сравнение индексов:")
       for index_name, values in indices.items():
           valid_values = values[values > 0]
           if len(valid_values) > 0:
               logger.info(f"{index_name}: среднее = {valid_values.mean():.3f}, СКО = {valid_values.std():.3f}")
       
       return results
   
   if __name__ == '__main__':
       # Расчет кастомных индексов
       custom_results = calculate_custom_indices_example()
       
       # Интеграция в пайплайн
       pipeline_results = integrate_custom_indices_pipeline()

Продвинутая визуализация
------------------------

Пример создания сложных визуализаций и научных графиков:

.. code-block:: python

   """
   Продвинутая визуализация результатов
   """
   
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from src.core.pipeline import Pipeline
   from src.utils.visualization import visualize_indices, create_comparison_plot
   from src.utils.image_utils import load_image, normalize_image
   from src.utils.logger import setup_logger
   from src.utils.file_utils import ensure_dir
   
   def create_advanced_visualizations(results, output_dir):
       """Создание продвинутых визуализаций"""
       
       logger = setup_logger('AdvancedVisualization', level='INFO')
       ensure_dir(output_dir)
       
       # Загрузка данных
       original_image = load_image(results['orthophoto_path'])
       segmentation_mask = load_image(results['segmentation_mask'], mode='L')
       indices = results['indices']['normalized_indices']
       
       # 1. Создание многопанельной визуализации
       create_multi_panel_visualization(original_image, segmentation_mask, indices, output_dir)
       
       # 2. Создание корреляционной матрицы
       create_correlation_heatmap(indices, output_dir)
       
       # 3. Создание профилей спектральных индексов
       create_index_profiles(indices, segmentation_mask, output_dir)
       
       # 4. Создание 3D визуализации
       create_3d_visualization(indices, output_dir)
       
       # 5. Создание временных рядов (если есть данные)
       create_time_series_visualization(results, output_dir)
       
       logger.info("Продвинутые визуализации созданы")
   
   def create_multi_panel_visualization(original_image, segmentation_mask, indices, output_dir):
       """Создание многопанельной визуализации"""
       
       fig, axes = plt.subplots(3, 3, figsize=(20, 15))
       
       # Исходное изображение
       axes[0, 0].imshow(original_image)
       axes[0, 0].set_title('Исходное изображение')
       axes[0, 0].axis('off')
       
       # Маска сегментации
       axes[0, 1].imshow(segmentation_mask, cmap='tab20')
       axes[0, 1].set_title('Маска сегментации')
       axes[0, 1].axis('off')
       
       # Вегетационные индексы
       index_names = list(indices.keys())[:6]
       for i, index_name in enumerate(index_names):
           row = (i + 2) // 3
           col = (i + 2) % 3
           
           im = axes[row, col].imshow(indices[index_name], cmap='RdYlGn', vmin=0, vmax=1)
           axes[row, col].set_title(index_name)
           axes[row, col].axis('off')
           plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
       
       # Скрытие лишних осей
       for i in range(len(index_names) + 2, 9):
           row = i // 3
           col = i % 3
           fig.delaxes(axes[row, col])
       
       plt.tight_layout()
       plt.savefig(f'{output_dir}/multi_panel_visualization.png', dpi=300, bbox_inches='tight')
       plt.close()
   
   def create_correlation_heatmap(indices, output_dir):
       """Создание тепловой карты корреляций"""
       
       # Создание матрицы данных
       index_names = list(indices.keys())
       correlation_matrix = np.zeros((len(index_names), len(index_names)))
       
       for i, name1 in enumerate(index_names):
           for j, name2 in enumerate(index_names):
               data1 = indices[name1].flatten()
               data2 = indices[name2].flatten()
               
               # Удаление NaN значений
               valid_mask = ~np.isnan(data1) & ~np.isnan(data2)
               if np.sum(valid_mask) > 0:
                   correlation = np.corrcoef(data1[valid_mask], data2[valid_mask])[0, 1]
                   correlation_matrix[i, j] = correlation
               else:
                   correlation_matrix[i, j] = 0
       
       # Создание тепловой карты
       plt.figure(figsize=(10, 8))
       sns.heatmap(
           correlation_matrix,
           xticklabels=index_names,
           yticklabels=index_names,
           annot=True,
           cmap='coolwarm',
           center=0,
           square=True
       )
       plt.title('Корреляционная матрица вегетационных индексов')
       plt.tight_layout()
       plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
       plt.close()
   
   def create_index_profiles(indices, segmentation_mask, output_dir):
       """Создание профилей спектральных индексов"""
       
       fig, axes = plt.subplots(2, 2, figsize=(15, 10))
       
       # Выбор областей для анализа
       mask_binary = (segmentation_mask > 0).astype(np.uint8)
       
       # Профили по строкам и столбцам
       for i, (index_name, index_data) in enumerate(list(indices.items())[:4]):
           row = i // 2
           col = i % 2
           
           # Профиль по центральной строке
           center_row = index_data.shape[0] // 2
           row_profile = index_data[center_row, :]
           
           # Профиль по центральному столбцу
           center_col = index_data.shape[1] // 2
           col_profile = index_data[:, center_col]
           
           x_axis = np.arange(len(row_profile))
           
           axes[row, col].plot(x_axis, row_profile, label='Строка', alpha=0.7)
           axes[row, col].plot(x_axis, col_profile, label='Столбец', alpha=0.7)
           axes[row, col].set_title(f'Профиль индекса {index_name}')
           axes[row, col].set_xlabel('Позиция')
           axes[row, col].set_ylabel('Значение индекса')
           axes[row, col].legend()
           axes[row, col].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.savefig(f'{output_dir}/index_profiles.png', dpi=300, bbox_inches='tight')
       plt.close()
   
   def create_3d_visualization(indices, output_dir):
       """Создание 3D визуализации"""
       
       from mpl_toolkits.mplot3d import Axes3D
       
       # Выбор трех индексов для 3D визуализации
       index_names = list(indices.keys())[:3]
       
       if len(index_names) >= 3:
           # Создание 3D облака точек
           index1 = indices[index_names[0]].flatten()
           index2 = indices[index_names[1]].flatten()
           index3 = indices[index_names[2]].flatten()
           
           # Удаление NaN значений
           valid_mask = ~np.isnan(index1) & ~np.isnan(index2) & ~np.isnan(index3)
           
           # Сэмплирование для ускорения
           valid_indices = np.where(valid_mask)[0]
           if len(valid_indices) > 10000:
               sampled_indices = np.random.choice(valid_indices, 10000, replace=False)
           else:
               sampled_indices = valid_indices
           
           fig = plt.figure(figsize=(12, 9))
           ax = fig.add_subplot(111, projection='3d')
           
           scatter = ax.scatter(
               index1[sampled_indices],
               index2[sampled_indices],
               index3[sampled_indices],
               c=index3[sampled_indices],
               cmap='viridis',
               alpha=0.6,
               s=1
           )
           
           ax.set_xlabel(index_names[0])
           ax.set_ylabel(index_names[1])
           ax.set_zlabel(index_names[2])
           ax.set_title('3D распределение вегетационных индексов')
           
           plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
           plt.savefig(f'{output_dir}/3d_visualization.png', dpi=300, bbox_inches='tight')
           plt.close()
   
   def create_time_series_visualization(results, output_dir):
       """Создание визуализации временных рядов"""
       
       # Этот метод может быть расширен для визуализации данных
       # из нескольких временных точек
       
       fig, ax = plt.subplots(figsize=(12, 6))
       
       # Пример: визуализация изменения индексов во времени
       # (заглушка для демонстрации)
       time_points = np.arange(10)
       gndvi_values = np.random.normal(0.6, 0.1, 10)
       ndwi_values = np.random.normal(0.4, 0.15, 10)
       
       ax.plot(time_points, gndvi_values, 'o-', label='GNDVI', linewidth=2, markersize=6)
       ax.plot(time_points, ndwi_values, 's-', label='NDWI', linewidth=2, markersize=6)
       
       ax.set_xlabel('Время (дни)')
       ax.set_ylabel('Значение индекса')
       ax.set_title('Динамика вегетационных индексов')
       ax.legend()
       ax.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.savefig(f'{output_dir}/time_series.png', dpi=300, bbox_inches='tight')
       plt.close()
   
   def advanced_visualization_example():
       """Пример продвинутой визуализации"""
       
       logger = setup_logger('AdvancedVisualizationExample', level='INFO')
       
       # Обработка данных
       pipeline = Pipeline()
       results = pipeline.process(
           input_path='data/hyperspectral.bil',
           output_dir='results/advanced_vis/',
           sensor_type='Hyperspectral',
           selected_indices=['GNDVI', 'NDWI', 'MCARI', 'OSAVI', 'SIPI2']
       )
       
       # Создание продвинутых визуализаций
       create_advanced_visualizations(results, 'results/advanced_visualizations/')
       
       logger.info("Пример продвинутой визуализации завершен")
   
   if __name__ == '__main__':
       advanced_visualization_example()

Эти примеры демонстрируют различные сценарии использования библиотеки GOP:
от базовой обработки до сложного научного анализа и визуализации.