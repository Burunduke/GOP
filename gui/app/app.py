"""
Основное Dash приложение для GUI GOP
"""

import os
import logging
from pathlib import Path

import dash
from flask import Flask
import dash_bootstrap_components as dbc

from ..config import config
from ..api.routes import api_blueprint
from ..components.layout import create_main_layout
from ..components.callbacks import register_callbacks


def setup_logging():
    """Настройка логирования"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'gui.log'),
            logging.StreamHandler()
        ]
    )


def create_app(config_name='default'):
    """Создание и конфигурация Dash приложения"""
    
    # Настройка логирования
    setup_logging()
    
    # Получение конфигурации
    app_config = config[config_name]
    
    # Создание Flask сервера
    server = Flask(__name__)
    server.config['SECRET_KEY'] = app_config.SECRET_KEY
    server.config['MAX_CONTENT_LENGTH'] = app_config.MAX_FILE_SIZE
    
    # Инициализация конфигурации
    app_config.init_app(server)
    
    # Создание Dash приложения
    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
            '/static/css/main.css'
        ],
        external_scripts=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'
        ],
        assets_folder='gui/static',
        url_base_pathname='/',
        suppress_callback_exceptions=True
    )
    
    # Регистрация API маршрутов
    server.register_blueprint(api_blueprint, url_prefix='/api')
    
    # Настройка основного layout
    app.layout = create_main_layout()
    
    # Инициализация сервисов
    from gui.services.project_manager import ProjectManager
    from gui.services.pipeline_executor import PipelineExecutor
    from gui.services.gop_adapter import GOPAdapter

    project_manager = ProjectManager(projects_dir=app_config.PROJECTS_FOLDER)
    gop_adapter = GOPAdapter()
    pipeline_executor = PipelineExecutor(project_manager=project_manager, gop_adapter=gop_adapter)

    # Регистрация колбэков с сервисами
    register_callbacks(app, project_manager=project_manager, pipeline_executor=pipeline_executor)
    
    # Настройка статических файлов
    _setup_static_files(server)
    
    return app


def _setup_static_files(server):
    """Настройка статических файлов"""
    @server.route('/static/<path:filename>')
    def serve_static(filename):
        from flask import send_from_directory
        static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
        return send_from_directory(static_dir, filename)
    
    @server.route('/docs/<path:filename>')
    def serve_docs(filename):
        from flask import send_from_directory
        docs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'docs')
        return send_from_directory(docs_dir, filename)


def main():
    """Главная функция для запуска GUI приложения"""
    import sys
    
    # Определение конфигурации
    config_name = os.getenv('FLASK_ENV', 'development')
    if len(sys.argv) > 1 and sys.argv[1] == '--production':
        config_name = 'production'
    
    # Создание приложения
    app = create_app(config_name)
    app_config = config[config_name]
    
    print(f"Запуск GOP GUI в режиме '{config_name}'")
    print(f"Адрес: http://{app_config.HOST}:{app_config.PORT}")
    
    # Запуск приложения
    app.run(
        host=app_config.HOST,
        port=app_config.PORT,
        debug=app_config.DEBUG,
        threaded=True
    )


if __name__ == '__main__':
    main()