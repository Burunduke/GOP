# Примеры данных для GOP

Эта директория содержит примеры данных для тестирования и демонстрации возможностей системы GOP.

## Структура директории

```
data/
├── README.md                 # Этот файл
├── examples/                 # Примеры данных
│   ├── hyperspectral/        # Гиперспектральные данные
│   │   ├── sample.bil        # Пример гиперспектрального изображения
│   │   └── sample.hdr        # Заголовок гиперспектрального изображения
│   ├── multispectral/        # Мультиспектральные данные
│   │   └── sample.tif        # Пример мультиспектрального изображения
│   ├── rgb/                  # RGB изображения
│   │   └── sample.jpg        # Пример RGB изображения
│   └── results/              # Примеры результатов обработки
│       ├── orthophoto.tif    # Пример ортофотоплана
│       ├── segmentation.png  # Пример маски сегментации
│       └── indices/          # Примеры вегетационных индексов
│           ├── GNDVI.tif     # Индекс GNDVI
│           ├── NDWI.tif      # Индекс NDWI
│           └── MCARI.tif     # Индекс MCARI
└── test/                     # Тестовые наборы данных
    ├── small/                # Маленькие тестовые данные
    └── large/                # Большие тестовые данные
```

## Использование примеров данных

### Базовый пример

```python
from src.core.pipeline import Pipeline

# Создание пайплайна
pipeline = Pipeline()

# Обработка примера гиперспектральных данных
results = pipeline.process(
    input_path="data/examples/hyperspectral/sample.bil",
    output_dir="data/examples/results",
    sensor_type="Hyperspectral"
)

print(f"Ортофотоплан: {results['orthophoto_path']}")
print(f"Маска сегментации: {results['segmentation_mask']}")
```

### Командная строка

```bash
# Обработка примера данных
python main.py data/examples/hyperspectral/sample.bil data/examples/results/

# Пакетная обработка
python main.py --batch data/examples/hyperspectral/ data/examples/results/ --pattern "*.bil"
```

## Форматы данных

### Гиперспектральные данные

- **Формат**: BIL (Band Interleaved by Line)
- **Заголовок**: HDR
- **Спектральные каналы**: 125 каналов в диапазоне 400-1000 нм
- **Разрешение**: 2048x2048 пикселей

### Мультиспектральные данные

- **Формат**: GeoTIFF
- **Спектральные каналы**: 4-8 каналов
- **Разрешение**: Переменное

### RGB данные

- **Формат**: JPEG, PNG
- **Цветовые каналы**: 3 (RGB)
- **Разрешение**: Переменное

## Генерация синтетических данных

Для тестирования можно использовать встроенные функции генерации синтетических данных:

```python
from src.utils.data_generator import generate_synthetic_hyperspectral

# Генерация синтетических гиперспектральных данных
data, header = generate_synthetic_hyperspectral(
    width=512,
    height=512,
    bands=125,
    noise_level=0.1
)

# Сохранение
data.save("data/test/small/synthetic.bil")
header.save("data/test/small/synthetic.hdr")
```

## Метаданные

Каждый набор данных сопровождается метаданными:

- Дата съемки
- Тип сенсора
- Спектральные характеристики
- Географическая привязка
- Условия съемки

## Лицензия

Примеры данных предоставляются для тестовых и образовательных целей.