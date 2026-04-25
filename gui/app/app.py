"""
Main Dash application for GOP GUI
"""

import os
import logging
import sys
import traceback
from pathlib import Path

import dash
from flask import Flask, redirect, send_from_directory
import dash_bootstrap_components as dbc

from ..config import config
from ..components.layout import create_main_layout
from ..components.callbacks import register_callbacks

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the application"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Set Werkzeug log level based on debug flag
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.INFO if debug else logging.ERROR)
    
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
    # Get configuration
    app_config = config[config_name]
    
    # Configure logging with debug flag
    setup_logging(app_config.DEBUG)
    
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
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css',
            '/static/css/main.css'
        ],
        external_scripts=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'
        ],
        assets_folder='gui/static',
        url_base_pathname='/',
        suppress_callback_exceptions=True
    )
    
    # Configure main layout
    app.layout = create_main_layout()
    
    # Initialize services
    from gui.services.project_manager import ProjectManager
    from gui.services.pipeline_executor import PipelineExecutor
    from gui.services.gop_adapter import GOPAdapter

    project_manager = ProjectManager(projects_dir=app_config.PROJECTS_FOLDER)
    gop_adapter = GOPAdapter()
    pipeline_executor = PipelineExecutor(project_manager=project_manager, gop_adapter=gop_adapter, max_workers=app_config.PIPELINE_MAX_WORKERS)

    # Attach services to Flask server so routes can access them via current_app
    server.project_manager = project_manager
    server.pipeline_executor = pipeline_executor

    # Register callbacks with services
    register_callbacks(app, project_manager=project_manager, pipeline_executor=pipeline_executor)
    
    # Configure static files
    _setup_static_files(server)
    
    # Add error handlers for better logging
    _setup_error_handlers(server)
    
    logger.info(f"GUI application created with config: {config_name}")
    return app


def _setup_static_files(server: Flask) -> None:
    """Configure static file serving
    
    Args:
        server: Flask server instance
    """
    @server.route('/')
    def serve_root():
        # Redirect to dashboard, but handle any browser cache issues
        return redirect('/dashboard')
    
    @server.route('/docs/api')
    def redirect_docs_api():
        # Redirect docs/api to dashboard to prevent automatic browser opening
        return redirect('/dashboard')
    
    @server.route('/static/<path:filename>')
    def serve_static(filename: str):
        static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
        return send_from_directory(static_dir, filename)
    
    @server.route('/docs/<path:filename>')
    def serve_docs(filename: str):
        docs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'docs')
        return send_from_directory(docs_dir, filename)
    
    # Test route for error handling (remove in production)
    @server.route('/test-error')
    def test_error():
        raise Exception("Test error for logging verification")


def _setup_error_handlers(server: Flask) -> None:
    """Configure error handlers for the Flask application
    
    Args:
        server: Flask server instance
    """
    
    @server.errorhandler(404)
    def not_found_error(error):
        logger.warning(f"404 error: {error}")
        return "Page not found", 404
    
    @server.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 error: {error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "Internal server error", 500
    
    @server.errorhandler(Exception)
    def unhandled_exception(error):
        logger.error(f"Unhandled exception: {error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "Internal server error", 500


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
    logger.info("Application will start without automatically opening browser")
    
    # Set environment variable to prevent browser opening
    os.environ['DASH_OPEN_BROWSER'] = 'False'
    
    try:
        # Run application using Flask server directly to prevent browser opening
        app.server.run(
            host=app_config.HOST,
            port=app_config.PORT,
            debug=app_config.DEBUG,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent double initialization
        )
    finally:
        # Shutdown pipeline executor to clean up threads
        if hasattr(app.server, 'pipeline_executor'):
            app.server.pipeline_executor.shutdown(wait=True)


if __name__ == '__main__':
    main()