#!/usr/bin/env python3
"""
Главная точка входа в приложение GOP - Гиперспектральная обработка и анализ растений
"""

import sys
import os
from pathlib import Path

# Добавление src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from gui.app.app import main as gui_main
except ImportError as e:
    print(f"Ошибка импорта GUI модулей: {e}")
    print("Убедитесь, что все зависимости установлены:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Запуск GUI приложения"""
    print("Запуск GOP GUI приложения...")
    gui_main()


if __name__ == '__main__':
    main()