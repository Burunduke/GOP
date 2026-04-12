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
- **`src.indices`** - Расчет вегетационных индексов
- **`src.segmentation`** - Сегментация изображений
- **`src.utils`** - Вспомогательные функции

### Ключевые классы

#### Pipeline
Основной конвейер обработки данных, который координирует все модули.

```python
from src.core.pipeline import Pipeline

# Создать экземпляр конвейера
pipeline = Pipeline()

# Обработать данные с научным анализом
results = pipeline.process(
    input_path="data/sample.bil",
    output_dir="results",
    sensor_type='Hyperspectral',
    selected_indices=['GNDVI', 'MCARI', 'NDWI']
)
```

#### HyperspectralProcessor
Продвинутый процессор для гиперспектральных данных с научными коррекциями.

```python
from src.processing.hyperspectral import HyperspectralProcessor

processor = HyperspectralProcessor()
processed_data = processor.process_data(
    data=hyperspectral_data,
    config=processing_config
)
```

#### Улучшенные утилитарные модули (Рефакторинг)

##### Математические утилиты
```python
from src.utils.math_utils import safe_divide, safe_normalize

# Безопасные математические операции с обработкой ошибок
result = safe_divide(numerator, denominator, default=0.0)
normalized = safe_normalize(values, value_range=(0, 100))
```

##### Фреймворк валидации
```python
from src.utils.validators import validate_data_range, validate_spectral_data

# Комплексная валидация данных
validate_data_range(data, min_value=0, max_value=10000)
validate_spectral_data(spectral_data, expected_bands=224)
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
from typing import List, Dict, Optional, Union
import numpy as np

def calculate_indices(
    data: np.ndarray,
    indices: List[str],
    sensor_type: str = "Hyperspectral"
) -> Dict[str, np.ndarray]:
    """Рассчитать вегетационные индексы с типобезопасностью.
    
    Args:
        data: Входной массив спектральных данных
        indices: Список названий индексов для расчета
        sensor_type: Тип данных сенсора
        
    Returns:
        Словарь, сопоставляющий названия индексов с рассчитанными значениями
        
    Raises:
        ValidationError: Если входные данные недействительны
        ProcessingError: Если расчет не удался
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
from src.processing.hyperspectral.cache import ProcessingCache

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
        
        assert 'orthophoto_path' in results
        assert 'segmentation_mask' in results
        assert 'vegetation_indices' in results
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

### Сканирование безопасности

Используйте Safety для безопасности зависимостей:
```bash
safety check
```

## 📊 Разработка производительности

### Бенчмаркинг

Создавайте бенчмарки производительности для критических операций:
```python
import pytest
import numpy as np

@pytest.mark.benchmark
def test_index_calculation_performance(benchmark):
    """Бенчмарк расчета вегетационных индексов."""
    data = np.random.random((1000, 1000, 10))
    
    def calculate():
        return calculate_indices(data, ['NDVI', 'GNDVI'])
    
    benchmark(calculate)
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

#### Тесты производительности
- Бенчмаркировать критические операции
- Мониторить использование памяти
- Тестировать масштабируемость

### Управление тестовыми данными

#### Синтетические данные
```python
import numpy as np

@pytest.fixture
def synthetic_hyperspectral_data():
    """Генерировать синтетические гиперспектральные данные для тестирования."""
    return np.random.random((100, 100, 10))
```

#### Образцы реальных данных
```python
@pytest.fixture
def real_hyperspectral_sample():
    """Загрузить образец реальных гиперспектральных данных для тестирования."""
    return load_sample_data('data/samples/sample_001.bil')
```

## 📚 Стандарты документации

### Документация кода

Используйте Google-style docstrings:
```python
def calculate_vegetation_index(
    red_band: np.ndarray,
    nir_band: np.ndarray
) -> np.ndarray:
    """Рассчитать вегетационный индекс из красного и ближнего инфракрасного каналов.
    
    Args:
        red_band: Данные красного спектрального канала
        nir_band: Данные ближнего инфракрасного спектрального канала
        
    Returns:
        Рассчитанный массив вегетационного индекса
        
    Raises:
        ValidationError: Если входные каналы недействительны
        ProcessingError: Если расчет не удался
        
    Examples:
        >>> red = np.array([0.1, 0.2, 0.3])
        >>> nir = np.array([0.4, 0.5, 0.6])
        >>> calculate_vegetation_index(red, nir)
        array([0.6, 0.428..., 0.333...])
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
      - name: Upload coverage
        uses: codecov/codecov-action@v2
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

# Проверить на уязвимости безопасности
safety check
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