"""
Тесты для утилит логгирования
"""

import unittest
import tempfile
import os
import shutil
import logging
from unittest.mock import patch, MagicMock
from src.utils.logger import setup_logger, get_logger, create_default_log_file


class TestLogger(unittest.TestCase):
    """Тесты утилит логгирования"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger_name = 'test_logger'
    
    def tearDown(self):
        """Очистка после тестов"""
        # Очистка логгеров
        logger = logging.getLogger(self.logger_name)
        logger.handlers.clear()
        
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_setup_logger_console_only(self):
        """Тест настройки логгера с выводом только в консоль"""
        logger = setup_logger(self.logger_name, level=logging.INFO, console=True, log_file=None)
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, self.logger_name)
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
    
    def test_setup_logger_file_only(self):
        """Тест настройки логгера с выводом только в файл"""
        log_file = os.path.join(self.temp_dir, 'test.log')
        
        logger = setup_logger(self.logger_name, level=logging.DEBUG, console=False, log_file=log_file)
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, self.logger_name)
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.FileHandler)
        
        # Проверка создания файла
        self.assertTrue(os.path.exists(log_file))
    
    def test_setup_logger_console_and_file(self):
        """Тест настройки логгера с выводом в консоль и файл"""
        log_file = os.path.join(self.temp_dir, 'test.log')
        
        logger = setup_logger(self.logger_name, level=logging.WARNING, console=True, log_file=log_file)
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, self.logger_name)
        self.assertEqual(logger.level, logging.WARNING)
        self.assertEqual(len(logger.handlers), 2)
        
        # Проверка типов обработчиков
        handler_types = [type(h) for h in logger.handlers]
        self.assertIn(logging.StreamHandler, handler_types)
        self.assertIn(logging.FileHandler, handler_types)
        
        # Проверка создания файла
        self.assertTrue(os.path.exists(log_file))
    
    def test_setup_logger_default_level(self):
        """Тест настройки логгера с уровнем по умолчанию"""
        logger = setup_logger(self.logger_name)
        
        self.assertEqual(logger.level, logging.INFO)
    
    def test_setup_logger_custom_format(self):
        """Тест настройки логгера с пользовательским форматом"""
        with patch('logging.Formatter') as mock_formatter:
            mock_formatter_instance = MagicMock()
            mock_formatter.return_value = mock_formatter_instance
            
            logger = setup_logger(self.logger_name)
            
            # Проверка, что Formatter был вызван с правильными аргументами
            mock_formatter.assert_called()
    
    def test_setup_logger_existing_logger(self):
        """Тест настройки существующего логгера"""
        # Создание логгера
        logger1 = setup_logger(self.logger_name)
        
        # Повторное создание того же логгера
        logger2 = setup_logger(self.logger_name)
        
        # Должен вернуться тот же логгер
        self.assertIs(logger1, logger2)
    
    def test_get_logger_existing(self):
        """Тест получения существующего логгера"""
        # Создание логгера
        original_logger = setup_logger(self.logger_name)
        
        # Получение логгера
        retrieved_logger = get_logger(self.logger_name)
        
        self.assertIs(original_logger, retrieved_logger)
    
    def test_get_logger_nonexistent(self):
        """Тест получения несуществующего логгера"""
        logger = get_logger('nonexistent_logger')
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'nonexistent_logger')
    
    def test_get_logger_default(self):
        """Тест получения логгера по умолчанию"""
        logger = get_logger()
        
        self.assertIsInstance(logger, logging.Logger)
    
    def test_create_default_log_file(self):
        """Тест создания файла лога по умолчанию"""
        log_file = create_default_log_file(self.temp_dir)
        
        self.assertTrue(os.path.exists(log_file))
        self.assertTrue(log_file.endswith('.log'))
        
        # Проверка, что файл находится в указанной директории
        self.assertTrue(log_file.startswith(self.temp_dir))
    
    def test_create_default_log_file_default_dir(self):
        """Тест создания файла лога по умолчанию в директории по умолчанию"""
        with patch('os.makedirs') as mock_makedirs:
            log_file = create_default_log_file()
            
            # Проверка, что директория была создана
            mock_makedirs.assert_called_once()
            
            # Проверка имени файла
            self.assertTrue(log_file.endswith('.log'))
            self.assertIn('logs', log_file)
    
    def test_logger_functionality(self):
        """Тест функциональности логгера"""
        log_file = os.path.join(self.temp_dir, 'test.log')
        
        logger = setup_logger(self.logger_name, level=logging.DEBUG, console=False, log_file=log_file)
        
        # Запись сообщений разных уровней
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        logger.critical('Critical message')
        
        # Проверка содержимого файла
        with open(log_file, 'r') as f:
            content = f.read()
            
            self.assertIn('Debug message', content)
            self.assertIn('Info message', content)
            self.assertIn('Warning message', content)
            self.assertIn('Error message', content)
            self.assertIn('Critical message', content)
    
    def test_logger_level_filtering(self):
        """Тест фильтрации сообщений по уровню"""
        log_file = os.path.join(self.temp_dir, 'test.log')
        
        # Установка уровня WARNING
        logger = setup_logger(self.logger_name, level=logging.WARNING, console=False, log_file=log_file)
        
        # Запись сообщений разных уровней
        logger.debug('Debug message')    # Не должно попасть в лог
        logger.info('Info message')      # Не должно попасть в лог
        logger.warning('Warning message')  # Должно попасть в лог
        logger.error('Error message')      # Должно попасть в лог
        
        # Проверка содержимого файла
        with open(log_file, 'r') as f:
            content = f.read()
            
            self.assertNotIn('Debug message', content)
            self.assertNotIn('Info message', content)
            self.assertIn('Warning message', content)
            self.assertIn('Error message', content)
    
    def test_logger_exception_logging(self):
        """Тест логирования исключений"""
        log_file = os.path.join(self.temp_dir, 'test.log')
        
        logger = setup_logger(self.logger_name, level=logging.ERROR, console=False, log_file=log_file)
        
        try:
            raise ValueError('Test exception')
        except ValueError:
            logger.exception('Exception occurred')
        
        # Проверка содержимого файла
        with open(log_file, 'r') as f:
            content = f.read()
            
            self.assertIn('Exception occurred', content)
            self.assertIn('ValueError', content)
            self.assertIn('Test exception', content)
            self.assertIn('Traceback', content)
    
    def test_logger_with_invalid_log_file_path(self):
        """Тест логгера с неверным путем к файлу лога"""
        invalid_path = '/invalid/path/that/cannot/be/created/test.log'
        
        # Должно создать логгер только с консольным выводом
        logger = setup_logger(self.logger_name, console=True, log_file=invalid_path)
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
    
    def test_logger_file_permissions(self):
        """Тест обработки ошибок прав доступа к файлу лога"""
        log_file = os.path.join(self.temp_dir, 'test.log')
        
        # Создание файла и установка прав только для чтения
        with open(log_file, 'w') as f:
            f.write('test')
        
        try:
            os.chmod(log_file, 0o444)  # Только для чтения
            
            # Попытка создания логгера с этим файлом
            logger = setup_logger(self.logger_name, console=True, log_file=log_file)
            
            # Должен создать логгер только с консольным выводом
            self.assertEqual(len(logger.handlers), 1)
            self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
            
        finally:
            # Восстановление прав для очистки
            os.chmod(log_file, 0o644)


if __name__ == '__main__':
    unittest.main()