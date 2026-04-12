# Документация по конфигурации GOP

## Обзор

Этот документ предоставляет полную документацию по системе конфигурации GOP (Геопространственная обработка ортофотопланов). Приложение использует иерархический подход к конфигурации с файлами YAML и переменными окружения.

## Файлы конфигурации

### Основной файл конфигурации
- **Файл**: [`config/config.yaml`](config/config.yaml)
- **Формат**: YAML
- **Назначение**: Основная конфигурация приложения с параметрами научной обработки

### Конфигурация окружения
- **Файл**: [`.env.example`](.env.example) (шаблон) → `.env` (фактический)
- **Формат**: Пары ключ-значение
- **Назначение**: Настройки и секреты для конкретного окружения

## Структура конфигурации

### Конфигурация обработки
```yaml
processing:
  max_image_size: 15000           # Максимальный размер изображения в пикселях
  compression_ratio: 0.125        # Коэффициент сжатия для промежуточных файлов
  batch_size: 32                  # Размер пакета обработки
  num_workers: 4                  # Количество параллельных воркеров
  orthophoto_resolution: 0.05     # Разрешение ортофотоплана в метрах
  dem_resolution: 0.1             # Разрешение DEM в метрах
  feature_quality: "high"         # Качество извлечения признаков
  matcher_neighbors: 8            # Количество соседей для сопоставления признаков
  odm_timeout: 7200               # Таймаут OpenDroneMap в секундах (2 часа)
```

#### Радиометрическая коррекция
```yaml
radiometric_correction:
  method: "empirical_line"        # dark_current, empirical_line, flat_field
  dark_percentile: 1              # Процентиль для обнаружения темных пикселей
  bright_percentile: 99           # Процентиль для обнаружения ярких пикселей
```

#### Атмосферная коррекция
```yaml
atmospheric_correction:
  enabled: true
  method: "simplified"            # simplified, empirical_line, modtran
```

#### Снижение шума
```yaml
noise_reduction:
  method: "pca"                   # pca, mnf, wavelet, savgol
  n_components: 0.95              # Коэффициент компонентов PCA
  wavelet_type: "db4"             # Тип вейвлета для вейвлет-шумоподавления
  wavelet_levels: 2               # Уровни декомпозиции вейвлета
  savgol_window: 11               # Размер окна Савицкого-Голея
  savgol_polyorder: 3             # Порядок полинома Савицкого-Голея
```

### Конфигурация сегментации
```yaml
segmentation:
  model_path: "models/deeplabv3_resnet101.pth"
  device: "auto"                  # auto, cpu, cuda
  confidence_threshold: 0.5       # Минимальная уверенность для сегментации
```

#### Cascade PSP (CascadePSP)
```yaml
cascade_psp:
  enabled: true
  l_parameter: 500                # Параметр L для CascadePSP
  refinement_threshold: 0.7       # Порог уточнения
```

### Вегетационные индексы
```yaml
indices:
  sensor_types: ["RGB", "Multispectral", "Hyperspectral"]
  
  # Научная классификация индексов
  index_groups:
    greenness: ["GNDVI", "MCARI", "MNLI", "OSAVI", "TVI", "NDVI"]
    stress: ["SIPI2", "mARI", "PRI", "CRI"]
    water: ["NDWI", "MSI", "WI", "NDII"]
    pigment: ["CARI", "PSRI", "SIPI"]
    structure: ["MSR", "MSAVI", "TVI"]
  
  default_indices: ["GNDVI", "MCARI", "MNLI", "OSAVI", "TVI", "SIPI2", "mARI", "NDWI", "MSI"]
```

### Научный анализ
```yaml
scientific_analysis:
  enabled: true
  
  statistics:
    confidence_level: 0.95        # Уровень статистической уверенности
    outlier_detection: true       # Включить обнаружение выбросов
    outlier_method: "iqr"         # iqr, zscore, isolation_forest
  
  correlation:
    method: "pearson"             # pearson, spearman, kendall
    threshold: 0.7                # Порог корреляции
    significance_test: true       # Статистическое тестирование значимости
  
  spatial:
    morans_i: true                # Пространственная автокорреляция Морана
    hotspot_analysis: true        # Анализ горячих точек
    fragmentation_index: true     # Индекс фрагментации ландшафта
    spatial_autocorrelation: true # Общая пространственная автокорреляция
```

### Конфигурация вывода
```yaml
output:
  results_dir: "results"          # Выходная директория для результатов
  save_intermediate: true         # Сохранять промежуточные файлы обработки
  output_format: "GeoTIFF"        # Формат выходного файла
  
  scientific_reports:
    enabled: true
    format: "json"                # json, csv, excel
    include_statistics: true      # Включать статистический анализ
    include_correlations: true    # Включать матрицы корреляций
    include_spatial_analysis: true # Включать пространственный анализ
```

### Конфигурация производительности
```yaml
performance:
  memory:
    max_memory_usage: "8GB"       # Максимальное использование памяти
    chunk_size: 1024              # Размер порции обработки
    memory_mapping: true          # Использовать memory mapping для больших файлов
  
  parallel:
    enabled: true                 # Включить параллельную обработку
    max_workers: 4                # Максимум параллельных воркеров
    chunk_processing: true        # Обрабатывать данные порциями
  
  cache:
    enabled: true                 # Включить кэширование
    cache_dir: "cache"            # Директория кэша
    max_cache_size: "1GB"         # Максимальный размер кэша
    max_memory_entries: 100       # Максимум записей в памяти
    ttl: 3600                     # Время жизни в секундах (1 час)
    cleanup_interval: 86400       # Интервал очистки кэша (24 часа)
    compression: true             # Включить сжатие кэша
    stats_enabled: true           # Включить статистику кэша
```

### Конфигурация логирования
```yaml
logging:
  level: "INFO"                   # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "logs/gop.log"           # Путь к файлу лога
  max_size: "10MB"               # Максимальный размер файла лога
  backup_count: 5                 # Количество резервных файлов
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  scientific_logging:
    enabled: true
    log_processing_steps: true    # Логировать детальные шаги обработки
    log_quality_metrics: true     # Логировать метрики качества
    log_performance_metrics: true # Логировать метрики производительности
```

### Переменные окружения (.env)

```bash
# Режим отладки (True/False)
DEBUG=False

# Секретный ключ для управления сессиями
SECRET_KEY=your-secure-secret-key-here

# Конфигурация сервера
HOST=0.0.0.0
PORT=8050

# Конфигурация базы данных
DATABASE_URL=postgresql://username:password@localhost/gop_db

# Конфигурация кэша
REDIS_URL=redis://localhost:6379/0

# Настройки загрузки файлов
MAX_UPLOAD_SIZE=100MB
UPLOAD_FOLDER=./uploads

# Настройки обработки
CACHE_ENABLED=True
CACHE_DIR=./cache

# Уровень логирования
LOG_LEVEL=INFO

# Внешние сервисы
ODM_PATH=/opt/opendronemap

# Настройки безопасности
CSRF_ENABLED=True
SESSION_TIMEOUT=3600
```

## Порядок загрузки конфигурации

1. **Значения по умолчанию** - Жестко закодированы в [`src/core/config.py`](src/core/config.py)
2. **YAML конфигурация** - [`config/config.yaml`](config/config.yaml)
3. **Переменные окружения** - Файл `.env`
4. **Аргументы командной строки** - Переопределения времени выполнения

## Валидация и контроль качества

```yaml
validation:
  enabled: true
  
  data_validation:
    check_missing_values: true    # Проверять на отсутствующие данные
    check_outliers: true          # Проверять на статистические выбросы
    check_spectral_consistency: true # Проверять спектральную согласованность
    min_snr: 10                   # Минимальное отношение сигнал-шум
  
  result_validation:
    check_georeference: true      # Валидировать геопривязку
    check_projection: true        # Валидировать систему координат
    check_data_range: true        # Валидировать диапазоны значений данных
    check_nodata_values: true     # Проверять на значения no-data
```

## Интеграция внешних инструментов

```yaml
external_tools:
  opendronemap:
    enabled: true
    auto_detect: true
    fallback_to_gdal: true
  
  gdal:
    config_options:
      GDAL_CACHEMAX: "512"        # Размер кэша GDAL в MB
      GDAL_DATA: "/usr/share/gdal" # Директория данных GDAL
      CPL_DEBUG: "OFF"            # Режим отладки GDAL
```

## Экспериментальные функции

```yaml
experimental:
  enabled: false                  # Включить экспериментальные функции
  
  machine_learning:
    enabled: false
    auto_classification: false    # Автоматическая классификация
    anomaly_detection: false      # Обнаружение аномалий
  
  cloud_processing:
    enabled: false
    provider: "aws"               # aws, gcp, azure
    auto_scaling: false           # Возможность автоскейлинга
```

## Лучшие практики

### Оптимизация производительности
1. Настройте `batch_size` и `num_workers` в зависимости от доступной памяти
2. Используйте `memory_mapping: true` для больших наборов данных
3. Включите кэширование для повторных операций
4. Установите соответствующий `chunk_size` для сред с ограниченной памятью

### Контроль качества
1. Включите все проверки валидации для продакшен использования
2. Установите соответствующий `confidence_threshold` для сегментации
3. Настройте обнаружение выбросов на основе характеристик данных

### Научная точность
1. Выбирайте соответствующие методы коррекции для вашего типа сенсора
2. Валидируйте настройки спектральной калибровки
3. Настройте параметры статистического анализа для ваших исследовательских нужд

## Устранение неполадок

### Распространенные проблемы

1. **Ошибки памяти**: Уменьшите `max_memory_usage` или `batch_size`
2. **Медленная обработка**: Увеличьте `num_workers` или включите кэширование
3. **Сбои сегментации**: Настройте `confidence_threshold`
4. **Ошибки валидации**: Проверьте качество и формат входных данных

### Валидация конфигурации

Система конфигурации включает встроенную валидацию. Недействительные настройки будут генерировать предупреждения или ошибки во время запуска.

## Информация о версии

- **Версия конфигурации**: 2.0.0
- **Последнее обновление**: 2026-04-12
- **Совместимость**: GOP v2.0.0+

Для подробной документации API см. [Справочник API](../docs/api/index.rst).