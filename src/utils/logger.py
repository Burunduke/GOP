"""
Модуль логирования для проекта GOP
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name, level=logging.INFO, log_file=None, console=True):
    """
    Настройка логгера
    
    Args:
        name (str): Имя логгера
        level (int): Уровень логирования
        log_file (str, optional): Путь к файлу логов
        console (bool): Вывод в консоль
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Очистка существующих обработчиков
    logger.handlers.clear()
    
    # Форматирование
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Обработчик для консоли
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Обработчик для файла
    if log_file:
        # Создание директории для логов
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """
    Получение существующего логгера
    
    Args:
        name (str): Имя логгера
        
    Returns:
        logging.Logger: Логгер
    """
    return logging.getLogger(name)


def create_default_log_file(base_dir="logs"):
    """
    Создание имени файла лога по умолчанию
    
    Args:
        base_dir (str): Базовая директория для логов
        
    Returns:
        str: Путь к файлу лога
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(base_dir, f"gop_{timestamp}.log")
    return log_file