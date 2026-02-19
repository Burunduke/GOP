#!/usr/bin/env python3
"""
Точка входа для запуска GUI приложения GOP
"""

import sys
import os
from pathlib import Path

# Добавляем текущую директорию в Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Импортируем GUI приложение
from gui.app.app import main

if __name__ == '__main__':
    main()