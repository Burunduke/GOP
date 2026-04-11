# Тестирование и CI/CD для проекта GOP

Этот документ описывает систему тестирования и непрерывной интеграции/развертывания (CI/CD) для проекта GOP.

## 🧪 Обзор тестирования

### Типы тестов

- **Модульные тесты** - Тестирование отдельных функций и классов
- **Интеграционные тесты** - Тестирование взаимодействия между компонентами

### Покрытие кода

- **Целевое покрытие**: 70%
- **Инструменты**: pytest-cov, coverage.py
- **Отчеты**: HTML, XML, терминальный вывод

## 🚀 Запуск тестов

### Быстрый старт

```bash
# Установка зависимостей для разработки
pip install -r requirements-dev.txt

# Запуск всех тестов
pytest tests/

# Запуск с покрытием кода
pytest tests/ --cov=src --cov-report=html
```

### Конкретные тесты

```bash
# Конкретный модуль
pytest tests/test_config.py

# С определенными маркерами
pytest tests/ -m "not slow"  # Только быстрые тесты
pytest tests/ -m integration # Только интеграционные тесты

# Отладка
pytest tests/ -v -s          # Подробный вывод
pytest tests/ -x             # Остановка при первой ошибке
```

## 📝 Написание тестов

### Структура теста

```python
import unittest
from unittest.mock import patch, MagicMock

class TestModuleName(unittest.TestCase):
    """Тесты для модуля module_name"""

    def setUp(self):
        """Подготовка тестовых данных"""
        pass

    def test_method_name(self):
        """Тест конкретного метода"""
        result = tested_method()
        self.assertEqual(result, expected)
```

### Использование моков

```python
@patch('module.ClassName')
def test_method_with_mock(self, mock_class):
    """Тест метода с использованием мока"""
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    
    result = tested_method()
    
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

## 🔄 CI/CD интеграция

### GitHub Actions

Тесты автоматически запускаются при:
- Push в ветки `main` и `develop`
- Создании Pull Request

### Воркфлоу

1. **Тестирование** - Запуск тестов на разных версиях Python
2. **Качество кода** - Проверка форматирования и линтинга
3. **Покрытие** - Генерация отчетов о покрытии
4. **Безопасность** - Проверка зависимостей

## 🐛 Устранение проблем

### Распространенные проблемы

```bash
# Импортные ошибки
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Отсутствующие зависимости
pip install -r requirements-dev.txt

# Проблемы с GDAL
sudo apt-get install gdal-bin libgdal-dev  # Ubuntu/Debian
brew install gdal                          # macOS
```

## 📊 Лучшие практики

1. **Изолированные тесты** - Каждый тест должен быть независимым
2. **Описательные имена** - Имена тестов должны четко описывать, что тестируется
3. **Мокирование внешних зависимостей** - Используйте моки для файловой системы, сети
4. **Тестирование граничных случаев** - Проверяйте нулевые значения, исключения
5. **Регулярный запуск** - Запускайте тесты перед коммитом изменений

## 🔗 Полезные ссылки

- **[Руководство разработчика](DEVELOPER.md)** - Подробное руководство по разработке
- **[Архитектура](ARCHITECTURE.md)** - Системная архитектура
- **[API документация](api/_build/html/index.html)** - Полная API документация

---

*Для получения дополнительной помощи обратитесь к [руководству разработчика](DEVELOPER.md) или создайте issue на GitHub.*