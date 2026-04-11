# Руководство пользователя GOP

Это руководство поможет вам начать работу с GOP для обработки гиперспектральных данных и анализа растительности.

## 🚀 Быстрый старт

### Предварительные требования

- Установленный GOP (см. [установку](INSTALLATION.md))
- Примеры данных для тестирования

### Первый запуск

```bash
# Проверка установки
python -c "import src.core.pipeline; print('GOP успешно установлен')"

# Запуск примера
python examples/basic_processing.py
```

## 📊 Форматы данных

### Поддерживаемые форматы

#### Гиперспектральные данные
- **BIL/HDR** - Band Interleaved by Line с заголовком ENVI
- **TIFF** - Многоканальные TIFF файлы
- **DAT** - Сырые бинарные данные

#### Мультиспектральные данные
- **GeoTIFF** - Геопривязанные TIFF файлы
- **JPEG/PNG** - RGB изображения

### Структура данных

```
project/
├── hyperspectral/
│   ├── image.bil
│   └── image.hdr
├── multispectral/
│   └── image.tif
└── results/
    └── (результаты обработки)
```

## 🔧 Обработка данных

### Базовый пайплайн

```python
from src.core.pipeline import Pipeline

# Создание пайплайна
pipeline = Pipeline()

# Обработка данных
results = pipeline.process(
    input_path="data/hyperspectral/image.bil",
    output_dir="results",
    sensor_type="Hyperspectral",
    indices=["NDVI", "GNDVI", "NDWI"]
)
```

### Расширенная конфигурация

```python
config = {
    "preprocessing": {
        "radiometric_correction": True,
        "atmospheric_correction": True,
        "denoising": "wavelet"
    },
    "orthophoto": {
        "method": "opendronemap",
        "resolution": 0.1
    },
    "segmentation": {
        "model": "deeplabv3+",
        "refinement": True
    }
}

results = pipeline.process(
    input_path="data/hyperspectral/image.bil",
    output_dir="results",
    config=config
)
```

## 🌱 Вегетационные индексы

GOP поддерживает более 20 вегетационных индексов:

### Индексы озеленения
- **NDVI** - Нормализованный разностный вегетационный индекс
- **GNDVI** - Зеленый нормализованный разностный индекс
- **OSAVI** - Оптимизированный почвенный индекс

### Индексы водного режима
- **NDWI** - Нормализованный разностный водный индекс
- **MSI** - Индекс влажности почвы

### Индексы стресса
- **PRI** - Фотохимический индекс отражения
- **CRI** - Индекс каротиноидов

## 📈 Анализ результатов

### Визуализация

```python
from src.utils.visualization import plot_vegetation_indices

# Визуализация индексов
plot_vegetation_indices(
    results['indices'],
    output_path="results/indices_plot.png"
)
```

### Статистический анализ

```python
from src.utils.analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(results['indices'])
stats = analyzer.calculate_statistics()

print("Статистика индексов:")
for index, values in stats.items():
    print(f"{index}: mean={values['mean']:.3f}, std={values['std']:.3f}")
```

## 🌐 Веб-интерфейс

### Запуск веб-интерфейса

Для запуска веб-интерфейса используйте команду:

```bash
python main.py
```

Или напрямую:

```bash
python gui.py
```

После запуска откройте браузер и перейдите по адресу `http://localhost:8050`

### Основные возможности веб-интерфейса

- **Загрузка данных**: Поддержка различных форматов гиперспектральных данных
- **Обработка**: Автоматическая обработка и коррекция данных
- **Расчет индексов**: Интерактивный выбор вегетационных индексов
- **Визуализация**: Графическое представление результатов
- **Экспорт**: Сохранение результатов в различных форматах

```bash
# Показать справку
python cli.py --help

# Обработка с настройками
python cli.py process \
    --input data.hdr \
    --output results/ \
    --indices ndvi,gndvi,ndwi \
    --config config.yaml
```

## 🎯 Примеры использования

### Пример 1: Базовая обработка

```python
# examples/basic_processing.py
from src.core.pipeline import Pipeline

pipeline = Pipeline()

results = pipeline.process(
    input_path="data/examples/hyperspectral/sample.bil",
    output_dir="results/basic",
    sensor_type="Hyperspectral"
)

print("Результаты обработки:")
for key, value in results.items():
    print(f"{key}: {value}")
```

### Пример 2: Пакетная обработка

```python
# examples/batch_processing.py
import os
from src.core.pipeline import Pipeline

pipeline = Pipeline()

input_dir = "data/hyperspectral/batch"
output_base = "results/batch"

for filename in os.listdir(input_dir):
    if filename.endswith('.bil'):
        input_path = os.path.join(input_dir, filename)
        output_dir = os.path.join(output_base, filename.replace('.bil', ''))
        
        print(f"Обработка: {filename}")
        results = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            sensor_type="Hyperspectral"
        )
        print(f"Завершено: {filename}")
```

## 🛠️ Устранение проблем

### Распространенные проблемы

#### Ошибки загрузки данных

```python
from src.utils.file_utils import validate_hyperspectral_file

try:
    validate_hyperspectral_file("data/image.bil")
    print("Файл валиден")
except Exception as e:
    print(f"Ошибка: {e}")
```

#### Недостаточно памяти

```python
# Использование обработки по частям
results = pipeline.process(
    input_path="large_image.bil",
    output_dir="results",
    chunk_size=1024  # Обработка по частям 1024x1024
)
```

#### Медленная обработка

```python
# Использование GPU ускорения
results = pipeline.process(
    input_path="data/image.bil",
    output_dir="results",
    use_gpu=True
)
```

### Логирование

```python
import logging
from src.utils.logger import setup_logger

logger = setup_logger("gop_user", level=logging.INFO)

logger.info("Начало обработки")
results = pipeline.process(...)
logger.info("Обработка завершена")
```

## 🔗 Дополнительные ресурсы

- **[Установка и настройка](INSTALLATION.md)** - Полное руководство по установке
- **[GUI руководство](GUI_GUIDE.md)** - Использование веб-интерфейса
- **[API документация](docs/api/_build/html/index.html)** - Полная API документация
- **[Примеры данных](data/README.md)** - Примеры данных для тестирования
- **[Архитектура](ARCHITECTURE.md)** - Понимание системы

## ❓ Часто задаваемые вопросы

**Q: Какой формат данных лучше использовать?**
A: BIL/HDR для гиперспектральных данных, GeoTIFF для мультиспектральных.

**Q: Можно ли обрабатывать данные без GPU?**
A: Да, но обработка больших данных будет медленнее.

**Q: Как настроить параметры обработки?**
A: Используйте конфигурационный файл или передавайте параметры в метод process().

**Q: Где найти примеры данных?**
A: В директории `data/examples/` или скачайте с официального сайта.

---

*Для получения дополнительной помощи обратитесь к [документации разработчика](DEVELOPER.md) или создайте issue на GitHub.*