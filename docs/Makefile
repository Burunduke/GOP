# Makefile для генерации документации GOP API

.PHONY: help clean docs html check-deps install-deps

# Цель по умолчанию
help:
	@echo "Доступные команды:"
	@echo "  help          - Показать эту справку"
	@echo "  check-deps    - Проверить зависимости"
	@echo "  install-deps  - Установить зависимости"
	@echo "  clean         - Очистить сгенерированные файлы"
	@echo "  docs          - Сгенерировать RST файлы"
	@echo "  html          - Сгенерировать HTML документацию"
	@echo "  all           - Полная генерация документации"

# Проверка зависимостей
check-deps:
	@echo "Проверка зависимостей..."
	@python generate_docs.py --check

# Установка зависимостей
install-deps:
	@echo "Установка зависимостей для документации..."
	@pip install sphinx sphinx_rtd_theme sphinx_autodoc_typehints

# Очистка
clean:
	@echo "Очистка документации..."
	@python generate_docs.py --clean

# Генерация RST файлов
docs:
	@echo "Генерация RST файлов..."
	@python generate_docs.py

# Сборка HTML документации
html:
	@echo "Сборка HTML документации..."
	@cd api && make html

# Полная генерация документации
all: clean docs html
	@echo "Документация успешно сгенерирована!"
	@echo "Откройте docs/api/_build/html/index.html для просмотра"

# Быстрая пересборка (только HTML)
rebuild:
	@echo "Пересборка HTML документации..."
	@cd api && make html

# Открытие документации в браузере (только для macOS)
open:
	@open api/_build/html/index.html