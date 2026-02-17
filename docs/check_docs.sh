#!/bin/bash

# Скрипт для проверки документации перед коммитом

set -e

echo "Проверка документации..."

# Проверка зависимостей
if ! python -c "import sphinx" 2>/dev/null; then
    echo "❌ Sphinx не установлен. Установите зависимости:"
    echo "pip install sphinx sphinx_rtd_theme sphinx_autodoc_typehints"
    exit 1
fi

# Проверка синтаксиса RST файлов
echo "Проверка синтаксиса RST файлов..."
find docs -name "*.rst" -exec rst2html.py {} /dev/null \; 2>/dev/null || {
    echo "❌ Обнаружены ошибки синтаксиса в RST файлах"
    exit 1
}

# Проверка наличия docstrings в измененных файлах
echo "Проверка наличия docstrings..."
changed_files=$(git diff --cached --name-only --diff-filter=ACM | grep -E "^src/.*\.py$" || true)

if [ -n "$changed_files" ]; then
    for file in $changed_files; do
        # Проверка наличия docstring в классах и функциях
        if grep -q "def " "$file" && ! grep -q '"""' "$file"; then
            echo "❌ В файле $file отсутствуют docstrings"
            exit 1
        fi
    done
fi

echo "✅ Проверка документации пройдена"