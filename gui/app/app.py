"""
Main Dash application for GOP GUI
"""

import os
import logging
import sys
from pathlib import Path
from typing import Optional

import dash
from flask import Flask, redirect, send_from_directory
import dash_bootstrap_components as dbc

from ..config import config
from ..api.routes import api_blueprint
from ..components.layout import create_main_layout
from ..components.callbacks import register_callbacks

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the application"""
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


def create_app(config_name: str = 'default') -> dash.Dash:
    """Create and configure Dash application
    
    Args:
        config_name: Configuration name ('development', 'production', 'default')
        
    Returns:
        Configured Dash application instance
    """
    # Configure logging
    setup_logging()
    
    # Get configuration
    app_config = config[config_name]
    
    # Create Flask server
    server = Flask(__name__)
    server.config['SECRET_KEY'] = app_config.SECRET_KEY
    server.config['MAX_CONTENT_LENGTH'] = app_config.MAX_FILE_SIZE
    
    # Initialize configuration
    app_config.init_app(server)
    
    # Create Dash application
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
    
    # Register API routes
    server.register_blueprint(api_blueprint, url_prefix='/api')
    
    # Configure main layout
    app.layout = create_main_layout()
    
    # Initialize services
    from gui.services.project_manager import ProjectManager
    from gui.services.pipeline_executor import PipelineExecutor
    from gui.services.gop_adapter import GOPAdapter

    project_manager = ProjectManager(projects_dir=app_config.PROJECTS_FOLDER)
    gop_adapter = GOPAdapter()
    pipeline_executor = PipelineExecutor(project_manager=project_manager, gop_adapter=gop_adapter)

    # Register callbacks with services
    register_callbacks(app, project_manager=project_manager, pipeline_executor=pipeline_executor)
    
    # Configure static files
    _setup_static_files(server)
    
    logger.info(f"GUI application created with config: {config_name}")
    return app


def _setup_static_files(server: Flask) -> None:
    """Configure static file serving
    
    Args:
        server: Flask server instance
    """
    @server.route('/')
    def serve_root():
        return redirect('/dashboard')
    
    @server.route('/static/<path:filename>')
    def serve_static(filename: str):
        static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
        return send_from_directory(static_dir, filename)
    
    @server.route('/docs/<path:filename>')
    def serve_docs(filename: str):
        docs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'docs')
        return send_from_directory(docs_dir, filename)


def main() -> None:
    """Main function to launch GUI application"""
    # Determine configuration
    config_name = os.getenv('FLASK_ENV', 'development')
    if len(sys.argv) > 1 and sys.argv[1] == '--production':
        config_name = 'production'
    
    # Create application
    app = create_app(config_name)
    app_config = config[config_name]
    
    logger.info(f"Starting GOP GUI in '{config_name}' mode")
    logger.info(f"Address: http://{app_config.HOST}:{app_config.PORT}")
    
    # Run application
    app.run(
        host=app_config.HOST,
        port=app_config.PORT,
        debug=app_config.DEBUG,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )


if __name__ == '__main__':
    main()