# Детальная структура проекта GUI для GOP

## 1. Конфигурационные файлы

### 1.1 Основные конфигурации

#### [`pyproject.toml`](pyproject.toml) - Конфигурация проекта GUI
```toml
[project]
name = "gop-gui"
version = "1.0.0"
description = "Веб-интерфейс для GOP - Гиперспектральная обработка и анализ растений"
dependencies = [
    "dash>=2.14.0",
    "dash-bootstrap-components>=1.5.0",
    "flask>=2.3.0",
    "celery>=5.3.0",
    "redis>=4.5.0",
    "sqlalchemy>=2.0.0",
    "plotly>=5.15.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "gdal>=3.7.0",
    "rasterio>=1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]
```

#### [`requirements.txt`](requirements.txt) - Зависимости для установки
```
dash==2.14.1
dash-bootstrap-components==1.5.0
flask==2.3.3
celery==5.3.4
redis==4.6.0
sqlalchemy==2.0.23
plotly==5.17.0
numpy==1.25.2
pandas==2.1.1
gdal==3.7.2
rasterio==1.3.8
```

### 1.2 Конфигурация приложения

#### [`config/gui_config.py`](config/gui_config.py)
```python
"""
Конфигурация GUI приложения GOP
"""

import os
from pathlib import Path

class GUIConfig:
    """Конфигурация GUI приложения"""
    
    # Основные настройки
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'gop-gui-secret-key-2024')
    
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
```

## 2. Основные модули приложения

### 2.1 Точка входа приложения

#### [`app.py`](app.py)
```python
#!/usr/bin/env python3
"""
Главная точка входа GUI приложения GOP
"""

import os
import logging
from pathlib import Path

from src.core.app_factory import create_app
from config.gui_config import config

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/gui.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Главная функция"""
    # Настройка логирования
    setup_logging()
    
    # Определение конфигурации
    config_name = os.getenv('FLASK_ENV', 'default')
    app_config = config[config_name]
    
    # Создание приложения
    app = create_app(app_config)
    
    # Запуск приложения
    app.run(
        host=app_config.HOST,
        port=app_config.PORT,
        debug=app_config.DEBUG,
        threaded=True
    )

if __name__ == '__main__':
    main()
```

### 2.2 Фабрика приложения

#### [`src/core/app_factory.py`](src/core/app_factory.py)
```python
"""
Фабрика для создания приложения GOP GUI
"""

import dash
from flask import Flask
from celery import Celery

from config.gui_config import GUIConfig
from src.core.session_manager import SessionManager
from src.core.cache_manager import CacheManager
from src.api.routes import api_blueprint
from src.tasks.celery_app import make_celery

def create_app(config_class=GUIConfig):
    """Создание и конфигурация приложения"""
    
    # Создание Flask приложения
    server = Flask(__name__)
    config_class.init_app(server)
    
    # Создание Dash приложения
    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
            '/static/css/main.css'
        ],
        external_scripts=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'
        ],
        assets_folder='static'
    )
    
    # Инициализация компонентов
    _init_components(app, config_class)
    
    return app

def _init_components(app, config_class):
    """Инициализация компонентов приложения"""
    
    # Менеджеры
    app.server.session_manager = SessionManager()
    app.server.cache_manager = CacheManager(config_class.REDIS_URL)
    
    # Celery
    app.server.celery = make_celery(app.server)
    
    # API маршруты
    app.server.register_blueprint(api_blueprint, url_prefix='/api')
    
    # Настройка макета
    from src.layouts.main_layout import create_main_layout
    app.layout = create_main_layout()
    
    # Настройка колбэков
    from src.callbacks import register_all_callbacks
    register_all_callbacks(app)
```

## 3. API слой

### 3.1 Основные маршруты API

#### [`src/api/routes.py`](src/api/routes.py)
```python
"""
Основные маршруты API
"""

from flask import Blueprint, jsonify, request
from src.api.projects import projects_api
from src.api.processing import processing_api
from src.api.analysis import analysis_api
from src.api.visualization import visualization_api

api_blueprint = Blueprint('api', __name__)

# Регистрация под-маршрутов
api_blueprint.register_blueprint(projects_api, url_prefix='/projects')
api_blueprint.register_blueprint(processing_api, url_prefix='/process')
api_blueprint.register_blueprint(analysis_api, url_prefix='/analysis')
api_blueprint.register_blueprint(visualization_api, url_prefix='/visualization')

@api_blueprint.route('/health')
def health_check():
    """Проверка здоровья API"""
    return jsonify({'status': 'healthy', 'service': 'gop-gui-api'})

@api_blueprint.route('/config')
def get_config():
    """Получение конфигурации"""
    return jsonify({
        'max_file_size': current_app.config['MAX_CONTENT_LENGTH'],
        'supported_formats': ['.bil', '.hdr', '.tif', '.tiff', '.dat'],
        'version': '1.0.0'
    })
```

### 3.2 API проектов

#### [`src/api/projects.py`](src/api/projects.py)
```python
"""
API для управления проектами
"""

from flask import Blueprint, request, jsonify, current_app
from src.adapters.pipeline_adapter import PipelineAdapter

projects_api = Blueprint('projects', __name__)

@projects_api.route('/', methods=['GET'])
def list_projects():
    """Получение списка проектов"""
    session_id = request.headers.get('X-Session-ID')
    session = current_app.session_manager.get_session(session_id)
    
    if not session:
        return jsonify({'error': 'Invalid session'}), 401
    
    projects = session.get('projects', [])
    return jsonify({'projects': projects})

@projects_api.route('/', methods=['POST'])
def create_project():
    """Создание нового проекта"""
    session_id = request.headers.get('X-Session-ID')
    session = current_app.session_manager.get_session(session_id)
    
    if not session:
        return jsonify({'error': 'Invalid session'}), 401
    
    project_data = request.json
    project_id = str(uuid.uuid4())
    
    project = {
        'id': project_id,
        'name': project_data.get('name', 'Новый проект'),
        'created_at': datetime.now().isoformat(),
        'status': 'created',
        'files': []
    }
    
    # Добавление проекта в сессию
    if 'projects' not in session:
        session['projects'] = []
    session['projects'].append(project)
    
    return jsonify(project), 201

@projects_api.route('/<project_id>/files', methods=['POST'])
def upload_files(project_id):
    """Загрузка файлов в проект"""
    session_id = request.headers.get('X-Session-ID')
    session = current_app.session_manager.get_session(session_id)
    
    if not session:
        return jsonify({'error': 'Invalid session'}), 401
    
    # Поиск проекта
    project = next((p for p in session['projects'] if p['id'] == project_id), None)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    
    # Обработка загруженных файлов
    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
        
        # Сохранение файла
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], project_id, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        
        uploaded_files.append({
            'name': filename,
            'path': file_path,
            'size': os.path.getsize(file_path),
            'uploaded_at': datetime.now().isoformat()
        })
    
    project['files'].extend(uploaded_files)
    return jsonify({'uploaded_files': uploaded_files})
```

## 4. Интерфейсный слой

### 4.1 Главный макет

#### [`src/layouts/main_layout.py`](src/layouts/main_layout.py)
```python
"""
Главный макет интерфейса
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
from src.components.navigation import create_navigation
from src.components.sidebar import create_sidebar
from src.components.content import create_content_area

def create_main_layout():
    """Создание главного макета"""
    return html.Div([
        # Хранилище данных
        dcc.Store(id='session-store', storage_type='session'),
        dcc.Store(id='project-store'),
        dcc.Store(id='processing-store'),
        
        # Навигационная панель
        create_navigation(),
        
        # Основной контейнер
        html.Div([
            # Боковая панель
            create_sidebar(),
            
            # Основное содержимое
            create_content_area()
        ], className="main-container"),
        
        # Модальные окна
        _create_modals(),
        
        # Уведомления
        dbc.Toast(id="notification-toast", is_open=False, duration=4000),
    ])

def _create_modals():
    """Создание модальных окон"""
    return html.Div([
        # Модальное окно создания проекта
        dbc.Modal([
            dbc.ModalHeader("Создание проекта"),
            dbc.ModalBody([
                dbc.Input(id="project-name-input", placeholder="Название проекта"),
                dbc.Input(id="project-description-input", placeholder="Описание", type="text")
            ]),
            dbc.ModalFooter([
                dbc.Button("Создать", id="create-project-btn", color="primary"),
                dbc.Button("Отмена", id="cancel-create-project", color="secondary")
            ])
        ], id="create-project-modal"),
        
        # Модальное окно загрузки файлов
        dbc.Modal([
            dbc.ModalHeader("Загрузка файлов"),
            dbc.ModalBody([
                dcc.Upload(
                    id='file-upload',
                    children=html.Div([
                        'Перетащите файлы или ',
                        html.A('выберите файлы')
                    ]),
                    multiple=True,
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    }
                ),
                html.Div(id='upload-file-list')
            ]),
            dbc.ModalFooter([
                dbc.Button("Загрузить", id="upload-files-btn", color="primary"),
                dbc.Button("Отмена", id="cancel-upload", color="secondary")
            ])
        ], id="upload-files-modal")
    ])
```

### 4.2 Компоненты интерфейса

#### [`src/components/data_upload.py`](src/components/data_upload.py)
```python
"""
Компонент загрузки данных
"""

import dash_bootstrap_components as dbc
from dash import html, dcc

def create_data_upload_component():
    """Создание компонента загрузки данных"""
    return dbc.Card([
        dbc.CardHeader("Загрузка гиперспектральных данных"),
        dbc.CardBody([
            html.P("Поддерживаемые форматы: BIL/HDR, TIFF, DAT"),
            
            # Виджет загрузки
            dcc.Upload(
                id='data-upload',
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt me-2"),
                    "Перетащите файлы или выберите"
                ]),
                multiple=True,
                style=upload_style()
            ),
            
            # Список загруженных файлов
            html.Div(id='uploaded-files-list', className="mt-3"),
            
            # Информация о файлах
            html.Div(id='file-info', className="mt-2")
        ])
    ])

def upload_style():
    """Стили для виджета загрузки"""
    return {
        'width': '100%',
        'height': '100px',
        'lineHeight': '100px',
        'borderWidth': '2px',
        'borderStyle': 'dashed',
        'borderRadius': '10px',
        'textAlign': 'center',
        'margin': '10px 0',
        'cursor': 'pointer',
        'backgroundColor': '#f8f9fa'
    }
```

## 5. Слой данных

### 5.1 Менеджер сессий

#### [`src/core/session_manager.py`](src/core/session_manager.py)
```python
"""
Менеджер сессий пользователей
"""

import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class UserSession(Base):
    """Модель сессии пользователя"""
    __tablename__ = 'user_sessions'
    
    session_id = Column(String(36), primary_key=True)
    user_id = Column(String(36))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    session_data = Column(JSON)

class SessionManager:
    """Менеджер для управления сессиями"""
    
    def __init__(self, database_url='sqlite:///gop_gui.db'):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
    
    def create_session(self, user_id=None, expires_hours=24):
        """Создание новой сессии"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        session_data = {
            'user_id': user_id,
            'created_at': datetime.utcnow().isoformat(),
            'projects': [],
            'current_project': None,
            'preferences': {}
        }
        
        db_session = self.Session()
        user_session = UserSession(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            session_data=session_data
        )
        
        db_session.add(user_session)
        db_session.commit()
        db_session.close()
        
        return session_id
    
    def get_session(self, session_id):
        """Получение данных сессии"""
        db_session = self.Session()
        user_session = db_session.query(UserSession).filter_by(session_id=session_id).first()
        
        if not user_session or user_session.expires_at < datetime.utcnow():
            db_session.close()
            return None
        
        db_session.close()
        return user_session.session_data
    
    def update_session(self, session_id, session_data):
        """Обновление данных сессии"""
        db_session = self.Session()
        user_session = db_session.query(UserSession).filter_by(session_id=session_id).first()
        
        if user_session:
            user_session.session_data = session_data
            db_session.commit()
        
        db_session.close()
    
    def delete_session(self, session_id):
        """Удаление сессии"""
        db_session = self.Session()
        user_session = db_session.query(UserSession).filter_by(session_id=session_id).first()
        
        if user_session:
            db_session.delete(user_session)
            db_session.commit()
        
        db_session.close()
```

### 5.2 Менеджер кэша

#### [`src/core/cache_manager.py`](src/core/cache_manager.py)
```python
"""
Менеджер кэширования данных
"""

import json
import pickle
from datetime import datetime, timedelta
import redis

class CacheManager:
    """Менеджер для кэширования данных"""
    
    def __init__(self, redis_url='redis://localhost:6379/0'):
        self.redis_client = redis.from_url(redis_url)
        self.local_cache = {}
        self.default_ttl = 3600  # 1 час по умолчанию
    
    def get(self, key, use_local_cache=True):
        """Получение данных из кэша"""
        # Попытка получить из локального кэша
        if use_local_cache and key in self.local_cache:
            cached_item = self.local_cache[key]
            if self._is_valid(cached_item):
                return cached_item['data']
            else:
                del self.local_cache[key]
        
        # Попытка получить из Redis
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                cached_item = pickle.loads(cached_data)
                if self._is_valid(cached_item):
                    # Сохранение в локальный кэш
                    self.local_cache[key] = cached_item
                    return cached_item['data']
                else:
                    self.redis_client.delete(key)
        except (pickle.PickleError, redis.RedisError):
            pass
        
        return None
    
    def set(self, key, data, ttl=None):
        """Сохранение данных в кэш"""
        if ttl is None:
            ttl = self.default_ttl
        
        cache_item = {
            'data': data,
            'created_at': datetime.now().isoformat(),
            'ttl': ttl
        }
        
        # Сохранение в локальный кэш
        self.local_cache[key] = cache_item
        
        # Сохранение в Redis
        try:
            serialized_data = pickle.dumps(cache_item)
            self.redis_client.setex(key, ttl, serialized_data)
        except (pickle.PickleError, redis.RedisError):
            pass
    
    def delete(self, key):
        """Удаление данных из кэша"""
        if key in self.local_cache:
            del self.local_cache[key]
        
        try:
            self.redis_client.delete(key)
        except redis.RedisError:
            pass
    
    def _is_valid(self, cache_item):
        """Проверка валидности кэшированных данных"""
        created_at = datetime.fromisoformat(cache_item['created_at'])
        expires_at = created_at + timedelta(seconds=cache_item['ttl'])
        return datetime.now() < expires_at
    
    def clear(self):
        """Очистка кэша"""
        self.local_cache.clear()
        try:
            self.redis_client.flushdb()
        except redis.RedisError:
            pass
```

## 6. Интеграционные адаптеры

### 6.1 Адаптер для Pipeline

#### [`src/adapters/pipeline_adapter.py`](src/adapters/pipeline_adapter.py)
```python
"""
Адаптер для интеграции с GOP Pipeline
"""

import asyncio
import concurrent.futures
from typing import Dict, Any
from src.core.pipeline import Pipeline

class PipelineAdapter:
    """Адаптер для работы с GOP Pipeline"""
    
    def __init__(self, config_path=None):
        self.pipeline = Pipeline(config_path)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    async def process_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Асинхронная обработка данных через GOP Pipeline"""
        try:
            # Запуск обработки в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._process_sync,
                config
            )
            
            return {
                'status': 'completed',
                'result': result,
                'error': None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'result': None,
                'error': str(e)
            }
    
    def _process_sync(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Синхронная обработка данных"""
        return self.pipeline.process(
            input_path=config['input_path'],
            output_dir=config['output_dir'],
            sensor_type=config.get('sensor_type', 'Hyperspectral'),
            selected_indices=config.get('selected_indices'),
            use_refinement=config.get('use_refinement', True)
        )
    
    def get_available_indices(self, sensor_type: str) -> list:
        """Получение доступных индексов для типа сенсора"""
        from src.indices.definitions import IndexDefinitions
        return IndexDefinitions.get_available_indices(sensor_type)
    
    def validate_input(self, file_path: str) -> Dict[str, Any]:
        """Валидация входного файла"""
        try:
            # Проверка существования файла
            if not os.path.exists(file_path):
                return {'valid': False, 'error': 'Файл не существует'}
            
            # Проверка размера файла
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024 * 1024:  # 10GB
                return {'valid': False, 'error': 'Файл слишком большой'}
            
            # Проверка формата файла
            supported_formats = ['.bil', '.hdr', '.tif', '.tiff', '.dat']
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in supported_formats:
                return {'valid': False, 'error': f'Неподдерживаемый формат: {file_ext}'}
            
            return {'valid': True, 'file_size': file_size}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
```

Эта детальная структура проекта обеспечивает надежную основу для создания GUI приложения с четким разделением ответственности между компонентами и эффективной интеграцией с существующими модулями GOP.