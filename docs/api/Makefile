# Makefile для Sphinx документации
# Вы можете использовать эти команды для сборки документации

# Настройки
SPHINXOPTS    =
SPHINXBUILD   = source ../../venv/bin/sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Помощь
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Очистка
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf api/

.PHONY: clean

# Генерация HTML документации
html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo
	@echo "Сборка завершена. HTML документация доступна в $(BUILDDIR)/html/"
	@echo "Откройте $(BUILDDIR)/html/index.html для просмотра"

.PHONY: html

# Генерация документации в формате PDF (через LaTeX)
latexpdf:
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Запуск LaTeX для генерации PDF..."
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf
	@echo "PDF документация доступна в $(BUILDDIR)/latex/GOP.pdf"

.PHONY: latexpdf

# Генерация документации в формате EPUB
epub:
	$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS) $(O)
	@echo
	@echo "Сборка завершена. EPUB документация доступна в $(BUILDDIR)/epub/"

.PHONY: epub

# Генерация документации для чтения на GitHub
github:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/github" $(SPHINXOPTS) $(O)
	@echo
	@echo "GitHub Pages документация доступна в $(BUILDDIR)/github/"

.PHONY: github

# Автоматическая генерация API документации
apidoc:
	mkdir -p api
	source ../../venv/bin/activate && sphinx-apidoc -o api/ ../../src/
	@echo "API документация сгенерирована в директории api/"

.PHONY: apidoc

# Полная пересборка документации с генерацией API
rebuild: clean apidoc html
	@echo "Полная пересборка документации завершена"

.PHONY: rebuild

# Проверка ссылок в документации
linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo
	@echo "Проверка ссылок завершена. Результаты в $(BUILDDIR)/linkcheck/"

.PHONY: linkcheck

# Проверка орфографии (требуется sphinxcontrib-spelling)
spelling:
	$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(SPHINXOPTS) $(O)
	@echo
	@echo "Проверка орфографии завершена. Результаты в $(BUILDDIR)/spelling/"

.PHONY: spelling

# Генерация документации с покрытием кода
coverage:
	$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo
	@echo "Покрытие документацией доступно в $(BUILDDIR)/coverage/python.txt"

.PHONY: coverage

# Генерация документации в формате XML
xml:
	$(SPHINXBUILD) -b xml "$(SOURCEDIR)" "$(BUILDDIR)/xml" $(SPHINXOPTS) $(O)
	@echo
	@echo "XML документация доступна в $(BUILDDIR)/xml/"

.PHONY: xml

# Быстрая сборка (без генерации API)
quick:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo
	@echo "Быстрая сборка завершена. HTML документация доступна в $(BUILDDIR)/html/"

.PHONY: quick

# Сборка с отладочной информацией
debug:
	$(SPHINXBUILD) -b html -v "$(SOURCEDIR)" "$(BUILDDIR)/debug" $(SPHINXOPTS) $(O)
	@echo
	@echo "Отладочная сборка завершена. Доступна в $(BUILDDIR)/debug/"

.PHONY: debug

# Сборка для разработки (с горячей перезагрузкой)
dev:
	@echo "Запуск сервера разработки с горячей перезагрузкой..."
	@echo "Установите sphinx-autobuild для этой команды: pip install sphinx-autobuild"
	source ../../venv/bin/activate && sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

.PHONY: dev

# Установка зависимостей для документации
install-deps:
	source ../../venv/bin/activate && pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints sphinx-autosummary-accessors sphinx-autobuild
	@echo "Зависимости для документации установлены"

.PHONY: install-deps

# Публикация документации на GitHub Pages
publish-gh: github
	@echo "Копирование документации в корень для GitHub Pages..."
	cp -r $(BUILDDIR)/github/* ../
	@echo "Документация скопирована в директорию docs/"
	@echo "Теперь можно закоммитить и запушить изменения"

.PHONY: publish-gh

# Показать статистику документации
stats:
	@echo "Статистика документации:"
	@echo "======================"
	@find "$(SOURCEDIR)" -name "*.rst" | wc -l | xargs echo "RST файлов:"
	@find "$(SOURCEDIR)" -name "*.md" | wc -l | xargs echo "Markdown файлов:"
	@find "$(SOURCEDIR)" -name "*.py" | wc -l | xargs echo "Python файлов:"
	@echo "======================"

.PHONY: stats

# Проверка качества документации
check: linkcheck spelling coverage
	@echo "Все проверки документации завершены"

.PHONY: check

# Полный цикл разработки документации
all: clean apidoc html linkcheck
	@echo "Полный цикл разработки документации завершен"

.PHONY: all

# Цель по умолчанию
.DEFAULT_GOAL := html