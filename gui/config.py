"""
GUI configuration for GOP application
"""

import os
import secrets
from pathlib import Path
from typing import Dict, Type
from flask import Flask


class GUIConfig:
    """GUI configuration for GOP application"""
    
    # Basic settings
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY: str = os.getenv('SECRET_KEY', secrets.token_hex(32))
    
    # Warn if using default generated key
    if not os.getenv('SECRET_KEY'):
        import warnings
        warnings.warn(
            "Using auto-generated SECRET_KEY. For production, set SECRET_KEY environment variable.",
            UserWarning
        )
    
    # Server settings
    HOST: str = os.getenv('HOST', '127.0.0.1')
    PORT: int = int(os.getenv('PORT', 8050))
    
    # Database settings
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///gop_gui.db')
    
    # Redis settings
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # File system settings
    UPLOAD_FOLDER: str = os.getenv('UPLOAD_FOLDER', 'data/uploads')
    PROJECTS_FOLDER: str = os.getenv('PROJECTS_FOLDER', 'data/projects')
    CACHE_FOLDER: str = os.getenv('CACHE_FOLDER', 'data/cache')
    
    # Limits
    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', 10 * 1024 * 1024 * 1024))  # 10GB
    MAX_UPLOAD_FILES: int = int(os.getenv('MAX_UPLOAD_FILES', 100))
    MAX_MEMORY_FILE_SIZE: int = int(os.getenv('MAX_MEMORY_FILE_SIZE', 100 * 1024 * 1024))  # 100MB
    STREAMING_CHUNK_SIZE: int = int(os.getenv('STREAMING_CHUNK_SIZE', 8192))  # 8KB
    
    # Processing settings
    CELERY_BROKER_URL: str = REDIS_URL
    CELERY_RESULT_BACKEND: str = REDIS_URL
    
    # GOP integration settings
    GOP_CONFIG_PATH: str = os.getenv('GOP_CONFIG_PATH', 'config/config.yaml')
    
    @classmethod
    def init_app(cls, app: Flask) -> None:
        """Initialize application configuration
        
        Args:
            app: Flask application instance
        """
        # Create necessary directories
        for folder in [cls.UPLOAD_FOLDER, cls.PROJECTS_FOLDER, cls.CACHE_FOLDER]:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        # Configure application
        app.config['SECRET_KEY'] = cls.SECRET_KEY
        app.config['MAX_CONTENT_LENGTH'] = cls.MAX_FILE_SIZE


class DevelopmentConfig(GUIConfig):
    """Configuration for development environment"""
    DEBUG: bool = True
    TESTING: bool = True


class ProductionConfig(GUIConfig):
    """Configuration for production environment"""
    DEBUG: bool = False
    TESTING: bool = False


config: Dict[str, Type[GUIConfig]] = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}