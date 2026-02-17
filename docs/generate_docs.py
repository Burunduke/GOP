#!/usr/bin/env python3
"""
Скрипт для автоматической генерации документации API проекта GOP.

Использование:
    python docs/generate_docs.py [--clean] [--build]

Опции:
    --clean    Очистить существующую документацию перед генерацией
    --build    Сгенерировать HTML документацию после создания RST файлов
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def clean_docs():
    """Очистить существующую документацию."""
    print("Очистка существующей документации...")
    
    # Очистка RST файлов
    api_dir = Path("api/api")
    if api_dir.exists():
        shutil.rmtree(api_dir)
        print(f"Удалена директория: {api_dir}")
    
    # Очистка HTML сборки
    build_dir = Path("api/_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"Удалена директория: {build_dir}")


def generate_rst():
    """Сгенерировать RST файлы из исходного кода."""
    print("Генерация RST файлов из исходного кода...")
    
    cmd = [
        sys.executable, "-m", "sphinx.ext.apidoc",
        "-f",  # force overwrite
        "-o", "api/api",  # output directory
        "../src"  # source directory
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("RST файлы успешно сгенерированы")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при генерации RST файлов: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def build_html():
    """Собрать HTML документацию."""
    print("Сборка HTML документации...")
    
    cmd = [
        sys.executable, "-m", "sphinx",
        "-b", "html",
        "api",
        "api/_build/html"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("HTML документация успешно собрана")
        print(f"Документация доступна по адресу: docs/api/_build/html/index.html")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при сборке HTML документации: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def check_dependencies():
    """Проверить наличие необходимых зависимостей."""
    print("Проверка зависимостей...")
    
    required_packages = [
        'sphinx',
        'sphinx_rtd_theme',
        'sphinx_autodoc_typehints'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (отсутствует)")
    
    if missing_packages:
        print(f"\nОтсутствующие пакеты: {', '.join(missing_packages)}")
        print("Установите их с помощью команды:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("Все зависимости установлены")
    return True


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Генератор документации GOP API")
    parser.add_argument("--clean", action="store_true", help="Очистить существующую документацию")
    parser.add_argument("--build", action="store_true", help="Собрать HTML документацию")
    parser.add_argument("--check", action="store_true", help="Проверить зависимости")
    
    args = parser.parse_args()
    
    # Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)
    
    if args.check:
        sys.exit(0)
    
    # Очистка документации
    if args.clean:
        clean_docs()
    
    # Генерация RST файлов
    if not generate_rst():
        sys.exit(1)
    
    # Сборка HTML документации
    if args.build:
        if not build_html():
            sys.exit(1)
    
    print("\nГенерация документации завершена!")
    
    if not args.build:
        print("\nДля сборки HTML документации выполните:")
        print("python docs/generate_docs.py --build")
        print("или перейдите в директорию docs/api и выполните 'make html'")


if __name__ == "__main__":
    main()