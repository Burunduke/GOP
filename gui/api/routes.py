"""
Main API routes for GOP GUI application
"""

import uuid
import os
from datetime import datetime
from typing import Dict, Any, List
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

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
        'supported_formats': ['.bil', '.hdr', '.tif', '.tiff', '.dat'],
        'version': '1.0.0'
    })


@api_blueprint.route('/projects', methods=['GET'])
def list_projects() -> Dict[str, Any]:
    """Get list of projects
    
    Returns:
        List of demo projects (temporary implementation)
    """
    # Temporary implementation - will use sessions in the future
    projects = [
        {
            'id': 'demo-project-1',
            'name': 'Demo Project: Wheat Field Analysis',
            'created_at': '2024-01-15T10:30:00',
            'status': 'completed',
            'files_count': 3
        },
        {
            'id': 'demo-project-2',
            'name': 'Demo Project: Index Testing',
            'created_at': '2024-01-16T14:20:00',
            'status': 'processing',
            'files_count': 2
        }
    ]
    return jsonify({'projects': projects})


@api_blueprint.route('/projects', methods=['POST'])
def create_project() -> Dict[str, Any]:
    """Create a new project
    
    Returns:
        Created project data
    """
    project_data = request.json
    project_id = str(uuid.uuid4())
    
    project = {
        'id': project_id,
        'name': project_data.get('name', 'New Project'),
        'description': project_data.get('description', ''),
        'created_at': datetime.now().isoformat(),
        'status': 'created',
        'files': []
    }
    
    return jsonify(project), 201


@api_blueprint.route('/projects/<project_id>/files', methods=['POST'])
def upload_files(project_id: str) -> Dict[str, Any]:
    """Upload files to a project
    
    Args:
        project_id: Project identifier
        
    Returns:
        Uploaded files information
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config.get('UPLOAD_FOLDER', 'data/uploads'), project_id, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        
        uploaded_files.append({
            'name': filename,
            'path': file_path,
            'size': os.path.getsize(file_path),
            'uploaded_at': datetime.now().isoformat()
        })
    
    return jsonify({'uploaded_files': uploaded_files})


@api_blueprint.route('/process', methods=['POST'])
def start_processing() -> Dict[str, Any]:
    """Start data processing
    
    Returns:
        Processing task information
    """
    processing_config = request.json
    task_id = str(uuid.uuid4())
    
    # Temporary implementation - will use Celery in the future
    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'Task queued for processing'
    }), 202


@api_blueprint.route('/process/<task_id>', methods=['GET'])
def get_processing_status(task_id: str) -> Dict[str, Any]:
    """Get processing status
    
    Args:
        task_id: Processing task identifier
        
    Returns:
        Processing status information
    """
    # Temporary implementation
    return jsonify({
        'task_id': task_id,
        'status': 'completed',
        'progress': 100,
        'message': 'Processing completed successfully',
        'result': {
            'output_path': f'data/results/{task_id}',
            'indices_calculated': ['NDVI', 'EVI', 'SAVI'],
            'processing_time': '00:05:23'
        }
    })


@api_blueprint.route('/indices', methods=['GET'])
def get_available_indices() -> Dict[str, Any]:
    """Get available vegetation indices
    
    Returns:
        List of available indices
    """
    indices = [
        {
            'id': 'NDVI',
            'name': 'Normalized Difference Vegetation Index',
            'description': 'Normalized Difference Vegetation Index',
            'formula': '(NIR - Red) / (NIR + Red)'
        },
        {
            'id': 'EVI',
            'name': 'Enhanced Vegetation Index',
            'description': 'Enhanced Vegetation Index',
            'formula': '2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))'
        },
        {
            'id': 'SAVI',
            'name': 'Soil Adjusted Vegetation Index',
            'description': 'Soil Adjusted Vegetation Index',
            'formula': '((NIR - Red) / (NIR + Red + L)) * (1 + L)'
        }
    ]
    return jsonify({'indices': indices})