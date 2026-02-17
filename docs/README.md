# Документация GOP API

Этот каталог содержит документацию API для проекта GOP (Гиперспектральная обработка и анализ растений).

## Структура документации

```
docs/
├── README.md                 # Этот файл
├── Makefile                  # Makefile для удобной генерации
├── generate_docs.py          # Скрипт для автоматической генерации
└── api/                      # Директория с исходниками Sphinx
    ├── conf.py               # Конфигурация Sphinx
    ├── index.rst             # Главный индексный файл
    ├── introduction.rst      # Введение в проект
    ├── quickstart.rst        # Быстрое начало
    ├── examples.rst          # Примеры использования
    ├── contributing.rst      # Руководство для контрибьюторов
    ├── api/                  # Автоматически сгенерированные RST файлы
    │   ├── modules.rst       # Общий модуль
    │   ├── core.rst          # Модуль core
    │   ├── processing.rst    # Модуль processing
    │   ├── indices.rst       # Модуль indices
    │   ├── segmentation.rst  # Модуль segmentation
    │   └── utils.rst         # Модуль utils
    └── _build/               # Сгенерированная HTML документация
        └── html/
            └── index.html    # Главная страница документации
```

## Генерация документации

### Способ 1: Использование Makefile (рекомендуется)

```bash
# Перейти в директорию docs
cd docs

# Проверить зависимости
make check-deps

# Установить зависимости (если необходимо)
make install-deps

# Полная генерация документации
make all

# Только сгенерировать RST файлы
make docs

# Только собрать HTML документацию
make html

# Очистить сгенерированные файлы
make clean

# Открыть документацию в браузере (только для macOS)
make open
```

### Способ 2: Использование Python скрипта

```bash
# Перейти в директорию docs
cd docs

# Проверить зависимости
python generate_docs.py --check

# Очистить и сгенерировать документацию
python generate_docs.py --clean --build

# Только сгенерировать RST файлы
python generate_docs.py

# Только собрать HTML документацию
python generate_docs.py --build
```

### Способ 3: Использование Sphinx напрямую

```bash
# Перейти в директорию api
cd docs/api

# Сгенерировать RST файлы
sphinx-apidoc -f -o api ../../src

# Собрать HTML документацию
make html
```

## Просмотр документации

После генерации HTML документации откройте файл `docs/api/_build/html/index.html` в вашем браузере.

## Автоматическая генерация

Для интеграции в CI/CD процессы можно использовать следующий скрипт:

```bash
#!/bin/bash
# Установка зависимостей
pip install sphinx sphinx_rtd_theme sphinx_autodoc_typehints

# Генерация документации
cd docs
python generate_docs.py --clean --build

# Проверка успешной генерации
if [ -f "api/_build/html/index.html" ]; then
    echo "Документация успешно сгенерирована"
    exit 0
else
    echo "Ошибка генерации документации"
    exit 1
fi
```

## Требования

- Python 3.8+
- Sphinx
- sphinx_rtd_theme
- sphinx_autodoc_typehints

## Кастомизация

### Изменение темы

Для изменения темы документации отредактируйте файл `docs/api/conf.py`:

```python
html_theme = 'sphinx_rtd_theme'  # или другая тема
```

### Добавление расширений

Для добавления новых расширений Sphinx отредактируйте `extensions` в `docs/api/conf.py`:

```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    # Добавьте новые расширения здесь
]
```

### Настройка интерсфинкса

Для добавления ссылок на документацию других проектов отредактируйте `intersphinx_mapping` в `docs/api/conf.py`:

```python
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}
```

## Устранение проблем

### Проблема: ModuleNotFoundError

Если возникают ошибки импорта модулей, убедитесь что:

1. Путь к исходному коду правильно указан в `conf.py`
2. Все зависимости установлены
3. Виртуальное окружение активировано

### Проблема: Ошибки в RST файлах

Проверьте синтаксис RST файлов, особенно:

- Правильность форматирования заголовков
- Корректность ссылок
- Правильное использование директив Sphinx

### Проблема: Отсутствие стилей

Если стили не применяются, проверьте:

- Наличие директории `_static`
- Правильность настройки `html_static_path`
- Установку темы документации

## Вклад в документацию

Для внесения изменений в документацию:

1. Отредактируйте соответствующие RST файлы
2. Обновите docstrings в исходном коде при необходимости
3. Перегенерируйте документацию
4. Проверьте результат в браузере
5. Создайте pull request с изменениями

## Лицензия

Документация распространяется под той же лицензией, что и основной проект.