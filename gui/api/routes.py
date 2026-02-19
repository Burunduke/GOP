"""
Основные маршруты API для GUI приложения GOP
"""

import uuid
import os
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

api_blueprint = Blueprint('api', __name__)


@api_blueprint.route('/health')
def health_check():
    """Проверка здоровья API"""
    return jsonify({
        'status': 'healthy', 
        'service': 'gop-gui-api',
        'timestamp': datetime.now().isoformat()
    })


@api_blueprint.route('/config')
def get_config():
    """Получение конфигурации"""
    return jsonify({
        'max_file_size': current_app.config.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024 * 1024),
        'supported_formats': ['.bil', '.hdr', '.tif', '.tiff', '.dat'],
        'version': '1.0.0'
    })


@api_blueprint.route('/projects', methods=['GET'])
def list_projects():
    """Получение списка проектов"""
    # Временная реализация - в будущем будет использовать сессии
    projects = [
        {
            'id': 'demo-project-1',
            'name': 'Демо проект: Анализ поля пшеницы',
            'created_at': '2024-01-15T10:30:00',
            'status': 'completed',
            'files_count': 3
        },
        {
            'id': 'demo-project-2',
            'name': 'Демо проект: Тестирование индексов',
            'created_at': '2024-01-16T14:20:00',
            'status': 'processing',
            'files_count': 2
        }
    ]
    return jsonify({'projects': projects})


@api_blueprint.route('/projects', methods=['POST'])
def create_project():
    """Создание нового проекта"""
    project_data = request.json
    project_id = str(uuid.uuid4())
    
    project = {
        'id': project_id,
        'name': project_data.get('name', 'Новый проект'),
        'description': project_data.get('description', ''),
        'created_at': datetime.now().isoformat(),
        'status': 'created',
        'files': []
    }
    
    return jsonify(project), 201


@api_blueprint.route('/projects/<project_id>/files', methods=['POST'])
def upload_files(project_id):
    """Загрузка файлов в проект"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
        
        # Сохранение файла
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
def start_processing():
    """Запуск обработки данных"""
    processing_config = request.json
    task_id = str(uuid.uuid4())
    
    # Временная реализация - в будущем будет использовать Celery
    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'Задача поставлена в очередь на обработку'
    }), 202


@api_blueprint.route('/process/<task_id>', methods=['GET'])
def get_processing_status(task_id):
    """Получение статуса обработки"""
    # Временная реализация
    return jsonify({
        'task_id': task_id,
        'status': 'completed',
        'progress': 100,
        'message': 'Обработка завершена успешно',
        'result': {
            'output_path': f'data/results/{task_id}',
            'indices_calculated': ['NDVI', 'EVI', 'SAVI'],
            'processing_time': '00:05:23'
        }
    })


@api_blueprint.route('/indices', methods=['GET'])
def get_available_indices():
    """Получение списка доступных вегетационных индексов"""
    indices = [
        {
            'id': 'NDVI',
            'name': 'Normalized Difference Vegetation Index',
            'description': 'Нормализованный вегетационный индекс разницы',
            'formula': '(NIR - Red) / (NIR + Red)'
        },
        {
            'id': 'EVI',
            'name': 'Enhanced Vegetation Index',
            'description': 'Улучшенный вегетационный индекс',
            'formula': '2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))'
        },
        {
            'id': 'SAVI',
            'name': 'Soil Adjusted Vegetation Index',
            'description': 'Вегетационный индекс с поправкой на почву',
            'formula': '((NIR - Red) / (NIR + Red + L)) * (1 + L)'
        }
    ]
    return jsonify({'indices': indices})