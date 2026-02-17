#!/usr/bin/env python3
"""
CLI интерфейс для GOP - Гиперспектральная обработка и анализ растений
"""

import os
import sys
import logging
from pathlib import Path

# Добавление src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import click
from src.core.pipeline import Pipeline
from src.core.config import config
from src.utils.logger import setup_logger


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Подробный вывод')
@click.option('--quiet', '-q', is_flag=True, help='Тихий режим')
@click.option('--config', 'config_path', help='Путь к файлу конфигурации')
@click.pass_context
def cli(ctx, verbose, quiet, config_path):
    """
    GOP - Гиперспектральная обработка и анализ растений
    
    Научная библиотека для обработки гиперспектральных данных и анализа состояния растений
    с использованием вегетационных индексов.
    """
    # Настройка логирования
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logger = setup_logger('GOP', level=level)
    
    # Сохранение контекста
    ctx.ensure_object(dict)
    ctx.obj['logger'] = logger
    ctx.obj['config_path'] = config_path


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--sensor-type', 
              type=click.Choice(['RGB', 'Multispectral', 'Hyperspectral']),
              default='Hyperspectral',
              help='Тип сенсора (по умолчанию: Hyperspectral)')
@click.option('--indices',
              help='Список индексов через запятую (например: GNDVI,NDWI,MCARI)')
@click.option('--compression-ratio', type=float, default=0.125,
              help='Коэффициент сжатия для сегментации (по умолчанию: 0.125)')
@click.option('--no-refinement', is_flag=True,
              help='Отключить уточнение границ сегментации')
@click.option('--save-results', help='Сохранить результаты в JSON файл')
@click.pass_context
def process(ctx, input_path, output_dir, sensor_type, indices, compression_ratio, 
           no_refinement, save_results):
    """
    Обработка гиперспектральных данных
    
    INPUT_PATH: Путь к входному файлу гиперспектральных данных
    OUTPUT_DIR: Директория для сохранения результатов
    """
    logger = ctx.obj['logger']
    config_path = ctx.obj['config_path']
    
    try:
        logger.info(f"Начало обработки файла: {input_path}")
        
        # Парсинг индексов
        selected_indices = None
        if indices:
            selected_indices = [idx.strip() for idx in indices.split(',')]
        
        # Создание пайплайна
        pipeline = Pipeline(config_path)
        
        # Обработка
        results = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            sensor_type=sensor_type,
            selected_indices=selected_indices,
            use_refinement=not no_refinement,
            compression_ratio=compression_ratio
        )
        
        # Вывод результатов
        click.echo("\nРезультаты обработки:")
        click.echo("=" * 50)
        click.echo(f"Ортофотоплан: {results['orthophoto_path']}")
        click.echo(f"Маска сегментации: {results['segmentation_mask']}")
        
        plant_condition = results.get('plant_condition', {})
        if 'classification' in plant_condition:
            classification = plant_condition['classification']
            click.echo(f"Состояние растений: {classification['class']} (оценка: {classification['score']:.3f})")
        
        # Сохранение результатов
        if save_results:
            pipeline.save_results(save_results)
            click.echo(f"Результаты сохранены: {save_results}")
        
        logger.info("Обработка завершена успешно")
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        click.echo(f"Ошибка: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--pattern', default='*.bil',
              help='Шаблон файлов для пакетной обработки (по умолчанию: *.bil)')
@click.option('--sensor-type', 
              type=click.Choice(['RGB', 'Multispectral', 'Hyperspectral']),
              default='Hyperspectral',
              help='Тип сенсора (по умолчанию: Hyperspectral)')
@click.option('--indices',
              help='Список индексов через запятую (например: GNDVI,NDWI,MCARI)')
@click.pass_context
def batch(ctx, input_dir, output_dir, pattern, sensor_type, indices):
    """
    Пакетная обработка файлов
    
    INPUT_DIR: Директория с входными файлами
    OUTPUT_DIR: Директория для сохранения результатов
    """
    logger = ctx.obj['logger']
    config_path = ctx.obj['config_path']
    
    try:
        import glob
        
        # Поиск файлов
        search_pattern = os.path.join(input_dir, pattern)
        files = glob.glob(search_pattern)
        
        if not files:
            click.echo(f"Файлы не найдены по шаблону: {search_pattern}")
            sys.exit(1)
        
        click.echo(f"Найдено файлов: {len(files)}")
        
        # Парсинг индексов
        selected_indices = None
        if indices:
            selected_indices = [idx.strip() for idx in indices.split(',')]
        
        # Создание пайплайна
        pipeline = Pipeline(config_path)
        
        # Обработка файлов
        success_count = 0
        error_count = 0
        
        for i, file_path in enumerate(files, 1):
            try:
                logger.info(f"Обработка файла {i}/{len(files)}: {file_path}")
                
                # Создание индивидуальной выходной директории
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                file_output_dir = os.path.join(output_dir, file_name)
                
                # Обработка
                results = pipeline.process(
                    input_path=file_path,
                    output_dir=file_output_dir,
                    sensor_type=sensor_type,
                    selected_indices=selected_indices
                )
                
                success_count += 1
                click.echo(f"✓ Обработан: {file_path}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Ошибка обработки {file_path}: {e}")
                click.echo(f"✗ Ошибка: {file_path} - {e}")
        
        # Вывод статистики
        click.echo(f"\nОбработка завершена:")
        click.echo(f"Успешно: {success_count}")
        click.echo(f"С ошибками: {error_count}")
        click.echo(f"Всего: {len(files)}")
        
    except Exception as e:
        logger.error(f"Ошибка пакетной обработки: {e}")
        click.echo(f"Ошибка: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_indices():
    """Показать доступные вегетационные индексы"""
    from src.indices.definitions import IndexDefinitions
    
    click.echo("Доступные вегетационные индексы:")
    click.echo("=" * 50)
    
    for group_name, group_indices in IndexDefinitions.INDEX_GROUPS.items():
        click.echo(f"\n{group_name.upper()}:")
        for index_name in group_indices:
            index_info = IndexDefinitions.get_index_info(index_name)
            click.echo(f"  {index_name}: {index_info.get('description', '')}")
            click.echo(f"    Формула: {index_info.get('formula', '')}")
            click.echo(f"    Требуемые каналы: {', '.join(index_info.get('required_bands', []))}")


@cli.command()
def show_config():
    """Показать текущую конфигурацию"""
    import yaml
    
    click.echo("Текущая конфигурация:")
    click.echo("=" * 50)
    click.echo(yaml.dump(config.config, default_flow_style=False, indent=2))


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
def info(input_path):
    """
    Показать информацию о файле данных
    
    INPUT_PATH: Путь к файлу гиперспектральных данных
    """
    try:
        from src.processing.hyperspectral import HyperspectralProcessor
        
        processor = HyperspectralProcessor()
        band_info = processor.get_band_info(input_path)
        
        click.echo(f"Информация о файле: {input_path}")
        click.echo("=" * 50)
        click.echo(f"Всего каналов: {band_info['total_bands']}")
        
        for band in band_info['bands'][:5]:  # Показываем первые 5 каналов
            click.echo(f"\nКанал {band['band_number']}:")
            click.echo(f"  Минимум: {band['min']:.3f}")
            click.echo(f"  Максимум: {band['max']:.3f}")
            click.echo(f"  Среднее: {band['mean']:.3f}")
            click.echo(f"  СКО: {band['stddev']:.3f}")
        
        if len(band_info['bands']) > 5:
            click.echo(f"\n... и еще {len(band_info['bands']) - 5} каналов")
            
    except Exception as e:
        click.echo(f"Ошибка: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('orthophoto_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--rgb-bands', default='30,20,10',
              help='Индексы каналов для RGB (R,G,B) через запятую')
def create_rgb(orthophoto_path, output_path, rgb_bands):
    """
    Создать RGB композит из гиперспектральных данных
    
    ORTHOPHOTO_PATH: Путь к ортофотоплану
    OUTPUT_PATH: Путь для сохранения RGB композита
    """
    try:
        from src.processing.hyperspectral import HyperspectralProcessor
        
        # Парсинг индексов каналов
        rgb_indices = tuple(int(x.strip()) for x in rgb_bands.split(','))
        
        processor = HyperspectralProcessor()
        rgb_path = processor.create_rgb_composite(
            [orthophoto_path], rgb_indices, output_path
        )
        
        click.echo(f"RGB композит создан: {rgb_path}")
        
    except Exception as e:
        click.echo(f"Ошибка: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()