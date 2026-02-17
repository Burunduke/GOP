# Тесты проекта GOP

Этот каталог содержит все тесты для проекта GOP (Гиперспектральная обработка и анализ растений).

## Структура тестов

### Основные тестовые файлы

- **`test_config.py`** - Тесты для модуля конфигурации (`src/core/config.py`)
- **`test_hyperspectral.py`** - Тесты для модуля обработки гиперспектральных данных (`src/processing/hyperspectral.py`)
- **`test_orthophoto.py`** - Тесты для модуля обработки ортофотопланов (`src/processing/orthophoto.py`)
- **`test_segmentation.py`** - Тесты для модуля сегментации изображений (`src/segmentation/segmenter.py`)
- **`test_indices.py`** - Тесты для модуля расчета вегетационных индексов (`src/indices/`)
- **`test_pipeline.py`** - Тесты для основного пайплайна обработки (`src/core/pipeline.py`)
- **`test_scientific_pipeline.py`** - Тесты для научного пайплайна
- **`test_integration.py`** - Интеграционные тесты для всего проекта

### Тесты утилит

- **`test_file_utils.py`** - Тесты для файловых утилит (`src/utils/file_utils.py`)
- **`test_image_utils.py`** - Тесты для утилит работы с изображениями (`src/utils/image_utils.py`)
- **`test_logger.py`** - Тесты для утилит логгирования (`src/utils/logger.py`)
- **`test_visualization.py`** - Тесты для утилит визуализации (`src/utils/visualization.py`)

### Тесты CLI

- **`test_cli.py`** - Тесты для командной строки интерфейса (`cli.py`)

### Вспомогательные файлы

- **`__init__.py`** - Инициализация модуля тестов
- **`run_tests.py`** - Скрипт для запуска всех тестов

## Запуск тестов

### Запуск всех тестов

```bash
# Используя встроенный скрипт
python tests/run_tests.py

# Используя pytest
pytest tests/

# Используя unittest
python -m unittest discover tests/
```

### Запуск конкретного тестового модуля

```bash
# Используя встроенный скрипт
python tests/run_tests.py --module test_config

# Используя pytest
pytest tests/test_config.py

# Используя unittest
python -m unittest tests.test_config
```

### Запуск с покрытием кода

```bash
# Используя pytest с покрытием
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Просмотр отчета о покрытии
open htmlcov/index.html
```

### Запуск с определенными маркерами

```bash
# Только быстрые тесты
pytest tests/ -m "not slow"

# Только интеграционные тесты
pytest tests/ -m integration

# Только модульные тесты
pytest tests/ -m unit
```

## Конфигурация тестов

### pytest.ini

Файл `pytest.ini` содержит конфигурацию для pytest:

- Пути к тестам
- Шаблоны имен файлов
- Опции покрытия кода
- Маркеры тестов
- Фильтры предупреждений

### Маркеры тестов

- `@pytest.mark.slow` - Медленные тесты
- `@pytest.mark.integration` - Интеграционные тесты
- `@pytest.mark.unit` - Модульные тесты
- `@pytest.mark.gpu` - Тесты, требующие GPU
- `@pytest.mark.network` - Тесты, требующие сетевой доступ

## Написание тестов

### Структура тестового класса

```python
import unittest
from unittest.mock import patch, MagicMock

class TestModuleName(unittest.TestCase):
    """Тесты для модуля module_name"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        # Создание тестовых данных
        pass
    
    def tearDown(self):
        """Очистка после тестов"""
        # Очистка ресурсов
        pass
    
    def test_method_name(self):
        """Тест конкретного метода"""
        # Логика теста
        self.assertEqual(result, expected)
```

### Использование моков

```python
@patch('module.ClassName')
def test_method_with_mock(self, mock_class):
    """Тест метода с использованием мока"""
    # Настройка мока
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    mock_instance.method.return_value = expected_value
    
    # Вызов тестируемого метода
    result = tested_method()
    
    # Проверки
    self.assertEqual(result, expected_value)
    mock_class.assert_called_once()
```

### Тестирование исключений

```python
def test_method_raises_exception(self):
    """Тест метода, который должен вызывать исключение"""
    with self.assertRaises(ValueError):
        tested_method(invalid_input)
```

### Параметризованные тесты

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_parameterized(self, input, expected):
    """Параметризованный тест"""
    result = tested_method(input)
    self.assertEqual(result, expected)
```

## Покрытие кода

### Цели покрытия

- Минимальное покрытие: 70%
- Целевое покрытие: 85%
- Идеальное покрытие: 95%

### Отчеты о покрытии

- HTML отчет: `htmlcov/index.html`
- XML отчет: `coverage.xml`
- Терминальный отчет: выводится при запуске

### Исключения из покрытия

Некоторые части кода могут быть исключены из покрытия:

```python
# pragma: no cover
def debug_function():
    """Функция отладки, не требующая покрытия"""
    pass
```

## CI/CD интеграция

### GitHub Actions

Тесты автоматически запускаются в GitHub Actions при:

- Push в ветки `main` и `develop`
- Создании Pull Request

### Воркфлоу

1. **Тестирование** (`.github/workflows/test.yml`)
   - Запуск тестов на разных версиях Python и ОС
   - Проверка кода с помощью линтеров
   - Интеграционные тесты

2. **Качество кода** (`.github/workflows/quality.yml`)
   - Проверка форматирования кода
   - Анализ сложности кода
   - Проверка безопасности

3. **Покрытие** (`.github/workflows/coverage.yml`)
   - Генерация отчетов о покрытии
   - Сравнение покрытия с базовой веткой
   - Комментарии в PR с информацией о покрытии

4. **Релиз** (`.github/workflows/release.yml`)
   - Сборка пакета
   - Публикация на PyPI
   - Создание Docker образа

## Устранение неполадок

### Распространенные проблемы

1. **Импортные ошибки**
   ```bash
   # Убедитесь, что src в Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Отсутствующие зависимости**
   ```bash
   # Установите зависимости для разработки
   pip install -r requirements-dev.txt
   ```

3. **Проблемы с GDAL**
   ```bash
   # Установите GDAL системные зависимости
   # Ubuntu/Debian:
   sudo apt-get install gdal-bin libgdal-dev
   # macOS:
   brew install gdal
   ```

### Отладка тестов

```bash
# Запуск с отладочным выводом
pytest tests/ -v -s

# Запуск конкретного теста с отладкой
pytest tests/test_module.py::TestClass::test_method -v -s

# Остановка при первом неудачном тесте
pytest tests/ -x

# Запуск только неудачных тестов
pytest tests/ --lf
```

## Лучшие практики

1. **Изолированные тесты** - Каждый тест должен быть независимым
2. **Описательные имена** - Имена тестов должны четко описывать, что тестируется
3. **Мокирование внешних зависимостей** - Используйте моки для файловой системы, сети и т.д.
4. **Тестирование граничных случаев** - Проверяйте нулевые значения, пустые массивы, исключения
5. **Регулярный запуск** - Запускайте тесты перед коммитом изменений
6. **Поддержание покрытия** - Старайтесь не снижать процент покрытия кода

## Добавление новых тестов

При добавлении нового модуля:

1. Создайте файл `test_module_name.py` в каталоге `tests/`
2. Следуйте существующей структуре тестов
3. Добавьте тесты для всех публичных методов
4. Включите тесты для обработки ошибок
5. Обновите этот README при необходимости