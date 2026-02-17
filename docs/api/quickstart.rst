Быстрое начало работы
=====================

Этот раздел поможет вам быстро начать использовать библиотеку GOP для обработки гиперспектральных данных.

Установка
---------

Установите библиотеку с помощью pip:

.. code-block:: bash

   pip install gop-hyperspectral

Или установите из исходного кода:

.. code-block:: bash

   git clone https://github.com/indykovdm/GOP.git
   cd GOP
   pip install -e .

Базовое использование
--------------------

Простая обработка гиперспектральных данных:

.. code-block:: python

   from src.core.pipeline import Pipeline

   # Создание пайплайна
   pipeline = Pipeline()

   # Обработка данных
   results = pipeline.process(
       input_path='data/hyperspectral.bil',
       output_dir='results/',
       sensor_type='Hyperspectral'
   )

   # Получение результатов
   print(f"Ортофотоплан: {results['orthophoto_path']}")
   print(f"Маска сегментации: {results['segmentation_mask']}")
   
   # Оценка состояния растений
   plant_condition = results['plant_condition']['classification']
   print(f"Состояние растений: {plant_condition['class']}")
   print(f"Оценка: {plant_condition['score']:.3f}")

Расчет конкретных вегетационных индексов
----------------------------------------

Выбор конкретных индексов для расчета:

.. code-block:: python

   from src.core.pipeline import Pipeline

   pipeline = Pipeline()

   # Расчет только выбранных индексов
   results = pipeline.process(
       input_path='data/hyperspectral.bil',
       output_dir='results/',
       sensor_type='Hyperspectral',
       selected_indices=['GNDVI', 'NDWI', 'MCARI']
   )

   # Доступ к значениям индексов
   indices = results['indices']['indices_values']
   print(f"GNDVI среднее: {indices['GNDVI'].mean():.3f}")
   print(f"NDWI среднее: {indices['NDWI'].mean():.3f}")

Работа с разными типами сенсоров
--------------------------------

RGB данные:

.. code-block:: python

   results = pipeline.process(
       input_path='data/rgb_image.tif',
       output_dir='results/',
       sensor_type='RGB',
       selected_indices=['mARI']  # Только индексы, доступные для RGB
   )

Мультиспектральные данные:

.. code-block:: python

   results = pipeline.process(
       input_path='data/multispectral.tif',
       output_dir='results/',
       sensor_type='Multispectral',
       selected_indices=['GNDVI', 'NDWI', 'MCARI', 'OSAVI']
   )

Гиперспектральные данные:

.. code-block:: python

   results = pipeline.process(
       input_path='data/hyperspectral.bil',
       output_dir='results/',
       sensor_type='Hyperspectral',
       selected_indices=['GNDVI', 'NDWI', 'MCARI', 'OSAVI', 'SIPI2', 'MSI']
   )

Использование CLI интерфейса
-----------------------------

Базовая обработка через командную строку:

.. code-block:: bash

   gop process input.bil output/ --sensor-type Hyperspectral

Выбор конкретных индексов:

.. code-block:: bash

   gop process input.bil output/ --indices GNDVI,NDWI,MCARI

Пакетная обработка:

.. code-block:: bash

   gop batch input_dir/ output/ --pattern "*.bil" --sensor-type Hyperspectral

Получение информации о файле:

.. code-block:: bash

   gop info input.bil

Просмотр доступных индексов:

.. code-block:: bash

   gop list-indices

Продвинутые примеры
-------------------

Настройка параметров обработки:

.. code-block:: python

   from src.core.pipeline import Pipeline
   from src.core.config import config

   # Настройка конфигурации
   config.set('processing.compression_ratio', 0.1)
   config.set('segmentation.confidence_threshold', 0.7)
   config.set('indices.default_indices', ['GNDVI', 'NDWI', 'MCARI'])

   pipeline = Pipeline()

   results = pipeline.process(
       input_path='data/hyperspectral.bil',
       output_dir='results/',
       sensor_type='Hyperspectral',
       use_refinement=True,
       compression_ratio=0.1
   )

Сохранение результатов в JSON:

.. code-block:: python

   # Сохранение полных результатов
   pipeline.save_results('results/processing_results.json')

   # Экспорт научных данных
   pipeline.export_scientific_data('results/')

Визуализация результатов:

.. code-block:: python

   from src.utils.visualization import visualize_indices, create_comparison_plot

   # Визуализация индексов
   indices = results['indices']['normalized_indices']
   fig = visualize_indices(indices, 'results/indices_visualization.png')

   # Создание сравнительного графика
   from src.utils.image_utils import load_image
   
   original_image = load_image(results['orthophoto_path'])
   segmentation_mask = load_image(results['segmentation_mask'])
   
   fig = create_comparison_plot(
       original_image, 
       segmentation_mask, 
       indices,
       'results/comparison_plot.png'
   )

Анализ качества данных:

.. code-block:: python

   from src.processing.hyperspectral import HyperspectralProcessor

   processor = HyperspectralProcessor()

   # Получение информации о каналах
   band_info = processor.get_band_info('data/hyperspectral.bil')
   print(f"Всего каналов: {band_info['total_bands']}")

   # Анализ качества данных
   for band in band_info['bands'][:5]:
       print(f"Канал {band['band_number']}: SNR = {band.get('snr', 'N/A'):.2f}")

Следующие шаги
--------------

После освоения базовых функций рекомендуется:

1. Изучить :doc:`api/modules` для подробного описания всех модулей
2. Посмотреть :doc:`examples` для более сложных сценариев использования
3. Ознакомиться с доступными вегетационными индексами в :class:`~src.indices.definitions.IndexDefinitions`
4. Изучить возможности настройки в :class:`~src.core.config.Config`

Устранение неполадок
--------------------

**Проблема**: ImportError при импорте модулей

**Решение**: Убедитесь, что все зависимости установлены:

.. code-block:: bash

   pip install -r requirements.txt

**Проблема**: Ошибка при открытии гиперспектрального файла

**Решение**: Проверьте поддерживаемые форматы и наличие HDR файла:

.. code-block:: python

   from src.processing.hyperspectral import HyperspectralProcessor
   
   processor = HyperspectralProcessor()
   if not any(input_path.endswith(ext) for ext in processor.supported_formats):
       print("Неподдерживаемый формат файла")

**Проблема**: Недостаточно памяти для обработки больших файлов

**Решение**: Увеличьте коэффициент сжатия или обрабатывайте файл частями:

.. code-block:: python

   results = pipeline.process(
       input_path='data/large_file.bil',
       output_dir='results/',
       compression_ratio=0.05  # Увеличить сжатие
   )