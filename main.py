#!/usr/bin/env python3
"""
Главная точка входа в приложение GOP - Гиперспектральная обработка и анализ растений
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Добавление src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.core.pipeline import Pipeline
    from src.core.config import config
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    print("Убедитесь, что все зависимости установлены:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def create_parser():
    """Создание парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='GOP - Гиперспектральная обработка и анализ растений',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Базовая обработка
  python main.py input.bil output/ --sensor-type Hyperspectral
  
  # С выбором индексов
  python main.py input.bil output/ --indices GNDVI,NDWI,MCARI
  
  # Пакетная обработка
  python main.py --batch input_dir/ output/ --pattern "*.bil"
  
  # Запуск GUI
  python main.py --gui
        """
    )
    
    # Основные аргументы
    parser.add_argument('input_path', nargs='?', 
                       help='Путь к входному файлу гиперспектральных данных')
    parser.add_argument('output_dir', nargs='?',
                       help='Директория для сохранения результатов')
    
    # Тип сенсора
    parser.add_argument('--sensor-type', 
                       choices=['RGB', 'Multispectral', 'Hyperspectral'],
                       default='Hyperspectral',
                       help='Тип сенсора (по умолчанию: Hyperspectral)')
    
    # Вегетационные индексы
    parser.add_argument('--indices',
                       help='Список индексов через запятую (например: GNDVI,NDWI,MCARI)')
    
    # Параметры обработки
    parser.add_argument('--compression-ratio', type=float, default=0.125,
                       help='Коэффициент сжатия для сегментации (по умолчанию: 0.125)')
    parser.add_argument('--no-refinement', action='store_true',
                       help='Отключить уточнение границ сегментации')
    parser.add_argument('--config', 
                       help='Путь к файлу конфигурации')
    
    # Режимы работы
    parser.add_argument('--gui', action='store_true',
                       help='Запустить графический интерфейс')
    parser.add_argument('--batch', action='store_true',
                       help='Пакетная обработка файлов')
    parser.add_argument('--pattern', default='*.bil',
                       help='Шаблон файлов для пакетной обработки (по умолчанию: *.bil)')
    
    # Вывод
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Тихий режим')
    parser.add_argument('--save-results', 
                       help='Сохранить результаты в JSON файл')
    
    # Информационные
    parser.add_argument('--version', action='version', version='%(prog)s 2.0.0')
    parser.add_argument('--list-indices', action='store_true',
                       help='Показать доступные вегетационные индексы')
    parser.add_argument('--show-config', action='store_true',
                       help='Показать текущую конфигурацию')
    
    return parser


def setup_logging(args):
    """Настройка логирования"""
    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logger = setup_logger('GOP', level=level)
    return logger


def list_indices():
    """Показать доступные вегетационные индексы"""
    from src.indices.definitions import IndexDefinitions
    
    print("Доступные вегетационные индексы:")
    print("=" * 50)
    
    for group_name, group_indices in IndexDefinitions.INDEX_GROUPS.items():
        print(f"\n{group_name.upper()}:")
        for index_name in group_indices:
            index_info = IndexDefinitions.get_index_info(index_name)
            print(f"  {index_name}: {index_info.get('description', '')}")
            print(f"    Формула: {index_info.get('formula', '')}")
            print(f"    Требуемые каналы: {', '.join(index_info.get('required_bands', []))}")


def show_config():
    """Показать текущую конфигурацию"""
    import yaml
    
    print("Текущая конфигурация:")
    print("=" * 50)
    print(yaml.dump(config.config, default_flow_style=False, indent=2))


def run_gui():
    """Запустить графический интерфейс"""
    try:
        from src.gui.main_window import MainWindow
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        
        return app.exec()
        
    except ImportError as e:
        print(f"Ошибка запуска GUI: {e}")
        print("Установите зависимости для GUI:")
        print("pip install PyQt6 PyQt6-tools")
        return 1


def process_single_file(args, logger):
    """Обработка одного файла"""
    try:
        # Проверка аргументов
        if not args.input_path or not args.output_dir:
            print("Ошибка: Укажите путь к входному файлу и выходную директорию")
            return 1
        
        if not os.path.exists(args.input_path):
            print(f"Ошибка: Файл не найден: {args.input_path}")
            return 1
        
        # Парсинг индексов
        selected_indices = None
        if args.indices:
            selected_indices = [idx.strip() for idx in args.indices.split(',')]
        
        # Создание пайплайна
        pipeline = Pipeline(args.config)
        
        # Обработка
        logger.info(f"Начало обработки файла: {args.input_path}")
        
        results = pipeline.process(
            input_path=args.input_path,
            output_dir=args.output_dir,
            sensor_type=args.sensor_type,
            selected_indices=selected_indices
        )
        
        # Вывод результатов
        print("\nРезультаты обработки:")
        print("=" * 50)
        print(f"Ортофотоплан: {results['orthophoto_path']}")
        print(f"Маска сегментации: {results['segmentation_mask']}")
        
        plant_condition = results.get('plant_condition', {})
        if 'classification' in plant_condition:
            classification = plant_condition['classification']
            print(f"Состояние растений: {classification['class']} (оценка: {classification['score']:.3f})")
        
        # Сохранение результатов
        if args.save_results:
            pipeline.save_results(args.save_results)
            print(f"Результаты сохранены: {args.save_results}")
        
        logger.info("Обработка завершена успешно")
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        print(f"Ошибка: {e}")
        return 1


def process_batch(args, logger):
    """Пакетная обработка файлов"""
    try:
        if not args.input_path or not args.output_dir:
            print("Ошибка: Укажите входную директорию и выходную директорию")
            return 1
        
        import glob
        
        # Поиск файлов
        search_pattern = os.path.join(args.input_path, args.pattern)
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"Файлы не найдены по шаблону: {search_pattern}")
            return 1
        
        print(f"Найдено файлов: {len(files)}")
        
        # Парсинг индексов
        selected_indices = None
        if args.indices:
            selected_indices = [idx.strip() for idx in args.indices.split(',')]
        
        # Создание пайплайна
        pipeline = Pipeline(args.config)
        
        # Обработка файлов
        success_count = 0
        error_count = 0
        
        for i, file_path in enumerate(files, 1):
            try:
                logger.info(f"Обработка файла {i}/{len(files)}: {file_path}")
                
                # Создание индивидуальной выходной директории
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                file_output_dir = os.path.join(args.output_dir, file_name)
                
                # Обработка
                results = pipeline.process(
                    input_path=file_path,
                    output_dir=file_output_dir,
                    sensor_type=args.sensor_type,
                    selected_indices=selected_indices
                )
                
                success_count += 1
                print(f"✓ Обработан: {file_path}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Ошибка обработки {file_path}: {e}")
                print(f"✗ Ошибка: {file_path} - {e}")
        
        # Вывод статистики
        print(f"\nОбработка завершена:")
        print(f"Успешно: {success_count}")
        print(f"С ошибками: {error_count}")
        print(f"Всего: {len(files)}")
        
        return 0 if error_count == 0 else 1
        
    except Exception as e:
        logger.error(f"Ошибка пакетной обработки: {e}")
        print(f"Ошибка: {e}")
        return 1


def main():
    """Главная функция"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging(args)
    
    # Информационные команды
    if args.list_indices:
        list_indices()
        return 0
    
    if args.show_config:
        show_config()
        return 0
    
    # Запуск GUI
    if args.gui:
        return run_gui()
    
    # Обработка данных
    if args.batch:
        return process_batch(args, logger)
    else:
        return process_single_file(args, logger)


if __name__ == '__main__':
    sys.exit(main())