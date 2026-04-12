"""
Конфигурация GUI приложения GOP
"""

import os
import secrets
from pathlib import Path


class GUIConfig:
    """Конфигурация GUI приложения"""
    
    # Основные настройки
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    
    # Warn if using default generated key
    if not os.getenv('SECRET_KEY'):
        import warnings
        warnings.warn(
            "Using auto-generated SECRET_KEY. For production, set SECRET_KEY environment variable.",
            UserWarning
        )
    
    # Настройки сервера
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8050))
    
    # Настройки базы данных
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///gop_gui.db')
    
    # Настройки Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Настройки файловой системы
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'data/uploads')
    PROJECTS_FOLDER = os.getenv('PROJECTS_FOLDER', 'data/projects')
    CACHE_FOLDER = os.getenv('CACHE_FOLDER', 'data/cache')
    
    # Ограничения
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 10 * 1024 * 1024 * 1024))  # 10GB
    MAX_UPLOAD_FILES = int(os.getenv('MAX_UPLOAD_FILES', 100))
    
    # Настройки обработки
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    
    # Настройки GOP интеграции
    GOP_CONFIG_PATH = os.getenv('GOP_CONFIG_PATH', 'config/config.yaml')
    
    @classmethod
    def init_app(cls, app):
        """Инициализация конфигурации приложения"""
        # Создание необходимых директорий
        for folder in [cls.UPLOAD_FOLDER, cls.PROJECTS_FOLDER, cls.CACHE_FOLDER]:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        # Настройка приложения
        app.config['SECRET_KEY'] = cls.SECRET_KEY
        app.config['MAX_CONTENT_LENGTH'] = cls.MAX_FILE_SIZE


class DevelopmentConfig(GUIConfig):
    """Конфигурация для разработки"""
    DEBUG = True
    TESTING = True


class ProductionConfig(GUIConfig):
    """Конфигурация для продакшена"""
    DEBUG = False
    TESTING = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}