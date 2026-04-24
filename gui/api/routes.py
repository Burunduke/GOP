"""
Main API routes for GOP GUI application
"""

import os
from datetime import datetime
from typing import Dict, Any
from flask import Blueprint, jsonify, request, current_app

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


# ---------------------------------------------------------------------------
# Projects endpoints
# ---------------------------------------------------------------------------

@api_blueprint.route('/projects', methods=['GET'])
def list_projects():
    """Get list of all projects.

    Returns:
        200 JSON: {"projects": [...], "total": N}
    """
    try:
        project_manager = current_app.project_manager
        projects_dicts = project_manager.list_projects_dicts()
        return jsonify({
            'projects': projects_dicts,
            'total': len(projects_dicts)
        }), 200
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        return jsonify({'error': str(e)}), 500


@api_blueprint.route('/projects/<project_id>', methods=['GET'])
def get_project(project_id: str):
    """Get a single project by ID.

    Returns:
        200 JSON: project dict
        404 JSON: {"error": "Project not found"}
    """
    try:
        project_manager = current_app.project_manager
        project = project_manager.get_project(project_id)
        if project is None:
            return jsonify({'error': 'Project not found'}), 404
        return jsonify(project.to_dict()), 200
    except Exception as e:
        logger.error(f"Error getting project {project_id}: {e}")
        return jsonify({'error': str(e)}), 500


@api_blueprint.route('/projects', methods=['POST'])
def create_project():
    """Create a new project.

    Request JSON body: {"name": "...", "description": "..."}  (description optional)

    Returns:
        201 JSON: created project dict
        400 JSON: {"error": "..."}
    """
    try:
        data = request.get_json(silent=True) or {}
        name = data.get('name', '')
        description = data.get('description', '')
        project_manager = current_app.project_manager
        result = project_manager.create_project_safe(name=name, description=description)
        
        if "error" in result:
            return jsonify({'error': result["error"]}), 400
        
        return jsonify(result), 201
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Processing endpoints
# ---------------------------------------------------------------------------

@api_blueprint.route('/process/<project_id>', methods=['POST'])
def start_processing(project_id: str):
    """Start processing for a project.

    Returns:
        200 JSON: {"status": "started", "project_id": project_id}
        404 JSON: {"error": "Project not found"}
        400 JSON: {"error": "..."}  — project already running or invalid state
    """
    try:
        pipeline_executor = current_app.pipeline_executor
        result = pipeline_executor.start_project_safe(project_id)
        
        if "error" in result:
            # Determine appropriate status code based on error type
            if "not found" in result["error"].lower():
                return jsonify({'error': result["error"]}), 404
            else:
                return jsonify({'error': result["error"]}), 400
        
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error starting processing for project {project_id}: {e}")
        return jsonify({'error': str(e)}), 500


@api_blueprint.route('/process/<project_id>/status', methods=['GET'])
def get_processing_status(project_id: str):
    """Get processing status for a project.

    Returns:
        200 JSON: {"project_id": ..., "status": ..., "progress": ..., "stage": ...}
        404 JSON: {"error": "Project not found"}
    """
    try:
        pipeline_executor = current_app.pipeline_executor
        result = pipeline_executor.get_status_dict(project_id)
        
        if "error" in result and "not found" in result["error"].lower():
            return jsonify({'error': result["error"]}), 404
        
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error getting status for project {project_id}: {e}")
        return jsonify({'error': str(e)}), 500
