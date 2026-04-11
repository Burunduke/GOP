# GOP - Гиперспектральная обработка и анализ растений

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/indykovdm/GOP/actions)
[![Coverage](https://img.shields.io/badge/coverage-85%25-yellow.svg)](https://github.com/indykovdm/GOP/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Scientific](https://img.shields.io/badge/purpose-scientific-orange.svg)](https://github.com/indykovdm/GOP)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](docs/api/_build/html/index.html)

**Версия 2.0.0 - Чистая научная архитектура без GUI**

Научная библиотека для обработки гиперспектральных данных и анализа состояния растений с использованием вегетационных индексов. Разработана на основе современных научных методов и алгоритмов обработки данных дистанционного зондирования.

## 🚀 Быстрый старт

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/indykovdm/GOP.git
cd GOP

# Установка зависимостей
pip install -r requirements.txt

# Проверка установки
python -c "import src.core.pipeline; print('GOP успешно установлен')"
```

### Первый запуск

```bash
# Запуск примера обработки
python examples/basic_processing.py
```

## 📚 Документация

### Для пользователей
- **[Установка и настройка](docs/INSTALLATION.md)** - Полное руководство по установке
- **[Руководство пользователя](docs/USER_GUIDE.md)** - Как использовать GOP
- **[GUI руководство](docs/GUI_GUIDE.md)** - Веб-интерфейс для GOP

### Для разработчиков
- **[Руководство разработчика](docs/DEVELOPER.md)** - API документация и разработка
- **[Тестирование](docs/TESTING.md)** - Тестирование и CI/CD
- **[Архитектура](docs/ARCHITECTURE.md)** - Системная архитектура

### Научные материалы
- **[Магистерская диссертация](docs/research/MASTER_THESIS.md)** - Полный текст диссертации
- **[Бакалаврская работа](docs/research/BACHELOR_THESIS.md)** - Исходная бакалаврская работа
- **[Технические заметки](docs/research/TECHNICAL_NOTES.md)** - Технические заметки и исследования

## 🌟 Основные возможности

### 📊 Обработка гиперспектральных данных
- Чтение и обработка данных в форматах BIL/HDR, TIFF
- Радиометрическая и атмосферная коррекция
- Продвинутое шумоподавление (PCA, MNF, вейвлеты)
- Спектральная калибровка и ресемплинг

### 🗺️ Создание ортофотопланов
- Интеграция с OpenDroneMap
- Альтернативные методы на основе GDAL
- Геопривязка и проекция данных
- Оптимизация и сжатие результатов

### 🌱 Вегетационные индексы
- **Индексы озеленения**: GNDVI, MCARI, MNLI, OSAVI, TVI, NDVI
- **Индексы стресса**: SIPI2, mARI, PRI, CRI
- **Индексы водного режима**: NDWI, MSI, WI, NDII

### 🔬 Анализ растительности
- Сегментация изображений
- Статистический анализ
- Визуализация результатов
- Экспорт данных

## 📁 Структура проекта

```
GOP/
├── src/                    # Исходный код
│   ├── core/              # Основные классы и пайплайны
│   ├── processing/        # Обработка данных
│   ├── indices/           # Вегетационные индексы
│   ├── segmentation/      # Сегментация изображений
│   └── utils/             # Вспомогательные утилиты
├── examples/              # Примеры использования
├── tests/                 # Тесты
├── docs/                  # Документация
└── data/                  # Примеры данных
```

## 💻 Использование

### Веб-интерфейс

Запустите веб-интерфейс с помощью команды:

```bash
python main.py
```

Или напрямую:

```bash
python gui.py
```

После запуска откройте браузер и перейдите по адресу `http://localhost:8050`

### Программный интерфейс

```python
from src.core.pipeline import Pipeline
from src.indices.calculator import IndexCalculator

# Создание пайплайна
pipeline = Pipeline()

# Обработка данных
result = pipeline.process(
    input_path="data.hdr",
    output_dir="results/",
    indices=["ndvi", "gndvi"]
)

# Расчет индексов
calculator = IndexCalculator()
indices = calculator.calculate("orthophoto.tif", ["ndvi", "ndwi"])
```

## 🔧 Зависимости

Основные зависимости:
- `numpy` - Научные вычисления
- `scipy` - Научные алгоритмы
- `opencv-python` - Обработка изображений
- `gdal` - Геопространственные данные
- `rasterio` - Работа с растровыми данными
- `plotly` - Визуализация

Полный список зависимостей в [`requirements.txt`](requirements.txt).

## 🤝 Вклад в проект

Мы приветствуем вклад в проект! Пожалуйста, ознакомьтесь с:
- [Руководством для контрибьюторов](docs/DEVELOPER.md#вклад-в-проект)
- [Кодексом поведения](CODE_OF_CONDUCT.md)
- [Шаблоном issue](.github/ISSUE_TEMPLATE.md)

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. Подробнее в файле [LICENSE](LICENSE).

## 📞 Контакты

- **Автор**: Дмитрий Индыков
- **Email**: indykovdm@gmail.com
- **GitHub**: [indykovdm](https://github.com/indykovdm)

## 🙏 Благодарности

Проект разработан в рамках научных исследований в Санкт-Петербургском государственном университете.

---

*Последнее обновление: 2024*