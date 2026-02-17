#!/usr/bin/env python3
"""
Скрипт для запуска всех тестов проекта
"""

import unittest
import sys
import os
from pathlib import Path

# Добавление src в Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def discover_and_run_tests():
    """Обнаружение и запуск всех тестов"""
    # Директория с тестами
    test_dir = Path(__file__).parent
    
    # Обнаружение тестов
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Запуск тестов
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Возврат кода завершения
    return 0 if result.wasSuccessful() else 1

def run_specific_test(test_module):
    """Запуск конкретного тестового модуля"""
    try:
        suite = unittest.TestLoader().loadTestsFromName(test_module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1
    except Exception as e:
        print(f"Ошибка при запуске теста {test_module}: {e}")
        return 1

def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Запуск тестов проекта GOP')
    parser.add_argument('--module', '-m', 
                       help='Запустить конкретный тестовый модуль (например: test_indices)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    parser.add_argument('--list', '-l', action='store_true',
                       help='Показать доступные тестовые модули')
    
    args = parser.parse_args()
    
    # Показать доступные тесты
    if args.list:
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob('test_*.py'))
        print("Доступные тестовые модули:")
        for test_file in test_files:
            module_name = test_file.stem
            print(f"  {module_name}")
        return 0
    
    # Установка уровня детализации
    if args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Запуск конкретного модуля
    if args.module:
        return run_specific_test(args.module)
    
    # Запуск всех тестов
    print("Запуск всех тестов проекта GOP...")
    print("=" * 50)
    
    return discover_and_run_tests()

if __name__ == '__main__':
    sys.exit(main())