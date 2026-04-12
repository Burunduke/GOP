"""
Модуль логирования для проекта GOP
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Type aliases for better type safety
LogLevel = Union[int, str]
LogConfig = Dict[str, Any]


def setup_logger(
    name: str,
    level: LogLevel = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Настройка логгера

    Args:
        name: Имя логгера
        level: Уровень логирования
        log_file: Путь к файлу логов
        console: Вывод в консоль

    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Очистка существующих обработчиков
    logger.handlers.clear()

    # Форматирование
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Получение существующего логгера

    Args:
        name: Имя логгера

    Returns:
        Логгер
    """
    return logging.getLogger(name)


def create_default_log_file(base_dir: str = "logs") -> str:
    """
    Создание имени файла лога по умолчанию

    Args:
        base_dir: Базовая директория для логов

    Returns:
        Путь к файлу лога
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(base_dir, f"gop_{timestamp}.log")
    return log_file


def configure_logging_from_config(config: LogConfig) -> None:
    """
    Настройка логирования на основе конфигурации

    Args:
        config: Конфигурация логирования
    """
    level = config.get("level", "INFO")
    log_file = config.get("file")
    console = config.get("console", True)

    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Очистка существующих обработчиков
    root_logger.handlers.clear()

    # Форматирование
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Обработчик для консоли
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Обработчик для файла
    if log_file:
        # Создание директории для логов
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
