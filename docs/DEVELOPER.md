# Руководство разработчика GOP

Этот документ содержит информацию для разработчиков, которые хотят внести вклад в проект GOP или использовать его API.

## 🚀 Быстрый старт

### Настройка среды разработки

```bash
# Клонировать репозиторий
git clone https://github.com/indykovdm/GOP.git
cd GOP

# Установить зависимости разработки
pip install -r requirements-dev.txt

# Установить в режиме разработки
pip install -e .
```

### Запуск тестов

```bash
# Запустить все тесты
pytest tests/

# Запустить с покрытием кода
pytest tests/ --cov=src --cov-report=html

# Запустить определенный модуль
pytest tests/test_config.py
```

## 📚 Обзор API

### Основные модули

- **`src.core`** - Основные классы и конвейеры
- **`src.processing`** - Обработка гиперспектральных данных и ортофотопланов
- **`src.utils`** - Вспомогательные функции

### Ключевые классы

#### Pipeline
Основной конвейер обработки данных, который координирует все модули.

```python
from src.core.pipeline import Pipeline

# Создать экземпляр конвейера
pipeline = Pipeline()

# Обработать данные
results = pipeline.process(
    input_path="data/sample.bil",
    output_dir="results",
    sensor_type='Hyperspectral'
)
```

#### HyperspectralProcessor
Процессор для загрузки гиперспектральных данных.

```python
from src.processing.hyperspectral import HyperspectralProcessor

processor = HyperspectralProcessor()
data = processor.load_data("data/sample.bil")
```

#### Улучшенные утилитарные модули

##### Математические утилиты
```python
from src.utils.math_utils import safe_divide, safe_normalize

# Безопасные математические операции с обработкой ошибок
result = safe_divide(numerator, denominator, default=0.0)
normalized = safe_normalize(values, value_range=(0, 100))
```

##### Фреймворк валидации
```python
from src.utils.validators import validate_data_range, validate_file_path

# Комплексная валидация данных
validate_data_range(data, min_value=0, max_value=10000)
validate_file_path("data/sample.bil")
```

##### Иерархия исключений
```python
from src.utils.exceptions import GOPError, ValidationError, ProcessingError

# Структурированная обработка ошибок
try:
    validate_data_range(data, min_value=0, max_value=10000)
except ValidationError as e:
    logger.error(f"Валидация данных не удалась: {e}")
```

## 🎯 Рекомендации по разработке

### Аннотации типов (Обязательно)

Весь новый код должен включать полные аннотации типов:

```python
from typing import Dict, Optional, Union
import numpy as np

def process_data(
    data: np.ndarray,
    config: Optional[Dict] = None
) -> Dict[str, Union[str, np.ndarray]]:
    """Обработать данные с типобезопасностью.
    
    Args:
        data: Входной массив данных
        config: Опциональная конфигурация
        
    Returns:
        Словарь с результатами обработки
        
    Raises:
        ValidationError: Если входные данные недействительны
        ProcessingError: Если обработка не удалась
    """
    # Реализация
    pass
```

### Обработка ошибок

Используйте иерархическую систему исключений:

```python
from src.utils.exceptions import GOPError, ValidationError

def validate_input_data(data: np.ndarray) -> None:
    """Валидировать входные данные с правильной обработкой ошибок."""
    if data is None:
        raise ValidationError("Входные данные не могут быть None")
    
    if data.size == 0:
        raise ValidationError("Входные данные не могут быть пустыми")
    
    if np.any(np.isnan(data)):
        raise ValidationError("Входные данные содержат значения NaN")
```

### Соображения производительности

#### Эффективность памяти
```python
import numpy as np
from typing import Generator

def process_large_dataset(
    data: np.ndarray,
    chunk_size: int = 1024
) -> Generator[np.ndarray, None, None]:
    """Обрабатывать большие наборы данных порциями для эффективности памяти."""
    for i in range(0, data.shape[0], chunk_size):
        chunk = data[i:i + chunk_size]
        yield process_chunk(chunk)
```

#### Кэширование
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def calculate_expensive_operation(
    data_hash: str,
    parameters: tuple
) -> np.ndarray:
    """Кэшировать дорогостоящие операции для производительности."""
    pass
```

### Рекомендации по тестированию

#### Модульные тесты
```python
import pytest
from src.utils.math_utils import safe_divide

class TestSafeDivide:
    """Тестовые случаи для функции safe_divide."""
    
    def test_divide_valid_numbers(self):
        """Тест деления действительных чисел."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(15, 3) == 5.0
    
    def test_divide_by_zero(self):
        """Тест деления на ноль возвращает значение по умолчанию."""
        import numpy as np
        assert np.isnan(safe_divide(10, 0))
        assert safe_divide(10, 0, default=0.0) == 0.0
```

#### Интеграционные тесты
```python
import pytest
from src.core.pipeline import Pipeline

class TestPipelineIntegration:
    """Интеграционные тесты для конвейера обработки."""
    
    def test_complete_processing_workflow(self, sample_data):
        """Тест полного рабочего процесса обработки."""
        pipeline = Pipeline()
        results = pipeline.process(
            input_path=sample_data,
            output_dir="/tmp/results"
        )
        
        assert 'input_path' in results
        assert 'output_dir' in results
        assert 'processed_data' in results
```

## 🔧 Стандарты качества кода

### Форматирование кода

Используйте Black для автоматического форматирования кода:
```bash
black src/ tests/ examples/
```

### Проверка типов

Используйте MyPy для статической проверки типов:
```bash
mypy src/
```

### Линтинг

Используйте Flake8 для качества кода:
```bash
flake8 src/ tests/
```

## 📊 Разработка производительности

### Бенчмаркинг

Создавайте бенчмарки производительности для критических операций:
```python
import pytest
import numpy as np

@pytest.mark.benchmark
def test_data_loading_performance(benchmark):
    """Бенчмарк загрузки данных."""
    def load_data():
        processor = HyperspectralProcessor()
        return processor.load_data("data/sample.bil")
    
    benchmark(load_data)
```

### Профилирование памяти

Используйте профилирование памяти для выявления проблем с памятью:
```python
from memory_profiler import profile

@profile
def process_large_dataset(data):
    """Профилировать использование памяти при обработке данных."""
    # Код обработки
    pass
```

## 🔍 Отладка и устранение неполадок

### Логирование

Используйте структурированное логирование для отладки:
```python
import logging
from src.utils.logger import setup_logger

logger = setup_logger('module_name', level=logging.DEBUG)

def process_data(data):
    logger.debug("Начало обработки данных")
    try:
        # Код обработки
        logger.info("Обработка данных успешно завершена")
    except Exception as e:
        logger.error(f"Обработка данных не удалась: {e}", exc_info=True)
        raise
```

### Отслеживание ошибок

Реализуйте комплексное отслеживание ошибок:
```python
from src.utils.exceptions import ProcessingError

def safe_operation():
    """Операция с комплексным отслеживанием ошибок."""
    try:
        # Код операции
        pass
    except ValueError as e:
        raise ProcessingError(f"Ошибка значения в операции: {e}") from e
    except IOError as e:
        raise ProcessingError(f"Ошибка ввода-вывода в операции: {e}") from e
```

## 🧪 Стратегия тестирования

### Категории тестов

#### Модульные тесты
- Тестировать отдельные функции и методы
- Использовать моки для внешних зависимостей
- Покрывать граничные случаи и условия ошибок

#### Интеграционные тесты
- Тестировать взаимодействие модулей
- Валидировать поток данных между компонентами
- Тестировать с образцами реальных данных

### Управление тестовыми данными

#### Синтетические данные
```python
import numpy as np

@pytest.fixture
def synthetic_hyperspectral_data():
    """Генерировать синтетические гиперспектральные данные для тестирования."""
    return np.random.random((100, 100, 10))
```

## 📚 Стандарты документации

### Документация кода

Используйте Google-style docstrings:
```python
def process_hyperspectral_data(
    file_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Обработать гиперспектральные данные из файла.
    
    Args:
        file_path: Путь к файлу с гиперспектральными данными
        config: Конфигурация обработки
        
    Returns:
        Словарь с результатами обработки
        
    Raises:
        ValidationError: Если входные данные недействительны
        ProcessingError: Если обработка не удалась
        
    Examples:
        >>> config = {"processing": {"max_image_size": 15000}}
        >>> results = process_hyperspectral_data("data.bil", config)
        >>> 'processed_data' in results
        True
    """
    # Реализация
    pass
```

### Документация API

Генерируйте документацию API с помощью Sphinx:
```bash
cd docs/api
make html
```

## 🔄 Непрерывная интеграция

### GitHub Actions

Настройте CI/CD пайплайн для автоматического тестирования:
```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
```

## 🚀 Рекомендации по развертыванию

### Управление версиями

Используйте семантическое версионирование:
- **MAJOR** версия для несовместимых изменений API
- **MINOR** версия для новой функциональности
- **PATCH** версия для исправления ошибок

### Управление зависимостями

Поддерживайте зависимости в актуальном состоянии:
```bash
# Обновить зависимости
pip install --upgrade -r requirements.txt
```

## 🤝 Участие в разработке

### Процесс ревью кода

1. **Форкнуть репозиторий**
2. **Создать ветку фичи**
3. **Реализовать изменения с тестами**
4. **Запустить все тесты и проверки**
5. **Отправить pull request**

### Чеклист для pull request

- [ ] Код следует руководствам по стилю
- [ ] Аннотации типов полные
- [ ] Тесты включены и проходят
- [ ] Документация обновлена
- [ ] Учтено влияние на производительность

---

**Руководство разработчика GOP v2.0.0** - Создание научного ПО с качеством и производительностью.