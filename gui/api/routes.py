"""
Main API routes for GOP GUI application
"""

import uuid
import os
from datetime import datetime
from typing import Dict, Any
import shutil
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

# Import and initialize logger
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

api_blueprint = Blueprint('api', __name__)


@api_blueprint.route('/health')
def health_check() -> Dict[str, Any]:
    """API health check endpoint
    
    Returns:
        Health status information
    """
    return jsonify({
        'status': 'healthy',
        'service': 'gop-gui-api',
        'timestamp': datetime.now().isoformat()
    })


@api_blueprint.route('/config')
def get_config() -> Dict[str, Any]:
    """Get application configuration
    
    Returns:
        Configuration information
    """
    return jsonify({
        'max_file_size': current_app.config.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024 * 1024),
        'supported_formats': ['.bil', '.hdr', '.tif', '.tiff', '.dat', '.png', '.jpg', '.jpeg', '.geotiff'],
        'version': '2.0.0'
    })
