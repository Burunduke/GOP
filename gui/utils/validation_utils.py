"""
Validation utilities for GOP GUI application
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime


def validate_project_data(project_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate project data
    
    Args:
        project_data: Project data to validate
        
    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate project name
    if 'name' not in project_data or not project_data['name']:
        result['errors'].append('Project name is required')
        result['valid'] = False
    elif len(project_data['name']) < 3:
        result['errors'].append('Project name must contain at least 3 characters')
        result['valid'] = False
    elif len(project_data['name']) > 100:
        result['errors'].append('Project name must not exceed 100 characters')
        result['valid'] = False
    elif not re.match(r'^[a-zA-Zа-яА-Я0-9\s\-_]+$', project_data['name']):
        result['errors'].append('Project name contains invalid characters')
        result['valid'] = False
    
    # Validate description
    if 'description' in project_data and project_data['description']:
        if len(project_data['description']) > 1000:
            result['errors'].append('Project description must not exceed 1000 characters')
            result['valid'] = False
    
    # Validate files
    if 'files' in project_data and project_data['files']:
        if not isinstance(project_data['files'], list):
            result['errors'].append('Files must be provided as a list')
            result['valid'] = False
        elif len(project_data['files']) > 50:
            result['warnings'].append('Number of files exceeds recommended limit (50)')
    
    # Validate settings
    if 'settings' in project_data and project_data['settings']:
        settings_validation = validate_processing_config(project_data['settings'])
        if not settings_validation['valid']:
            result['errors'].extend(settings_validation['errors'])
            result['valid'] = False
        result['warnings'].extend(settings_validation['warnings'])
    
    return result


def validate_processing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate processing configuration
    
    Args:
        config: Processing configuration
        
    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate sensor type
    if 'sensor_type' in config:
        valid_sensor_types = ['hyperspectral', 'multispectral', 'rgb']
        if config['sensor_type'] not in valid_sensor_types:
            result['errors'].append(f'Invalid sensor type. Valid: {", ".join(valid_sensor_types)}')
            result['valid'] = False
    
    # Validate vegetation indices
    if 'selected_indices' in config:
        if not isinstance(config['selected_indices'], list):
            result['errors'].append('Indices must be provided as a list')
            result['valid'] = False
        else:
            valid_indices = ['NDVI', 'EVI', 'SAVI', 'MSAVI', 'GNDVI', 'NDRE']
            for index in config['selected_indices']:
                if index not in valid_indices:
                    result['warnings'].append(f'Index {index} may not be supported by the selected sensor type')
    
    # Validate processing options
    if 'processing_options' in config:
        valid_options = ['atmospheric_correction', 'geometric_correction']
        for option in config['processing_options']:
            if option not in valid_options:
                result['warnings'].append(f'Processing option {option} may not be supported')
    
    # Validate quality parameters
    if 'quality_parameters' in config:
        quality_params = config['quality_parameters']
        
        if 'cloud_threshold' in quality_params:
            threshold = quality_params['cloud_threshold']
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                result['errors'].append('Cloud threshold must be a number between 0 and 1')
                result['valid'] = False
        
        if 'min_vegetation_coverage' in quality_params:
            coverage = quality_params['min_vegetation_coverage']
            if not isinstance(coverage, (int, float)) or not (0 <= coverage <= 1):
                result['errors'].append('Minimum vegetation coverage must be a number between 0 and 1')
                result['valid'] = False
    
    return result


def validate_file_upload_data(upload_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate file upload data
    
    Args:
        upload_data: Upload data
        
    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate project ID
    if 'project_id' not in upload_data or not upload_data['project_id']:
        result['errors'].append('Project ID is required')
        result['valid'] = False
    elif not re.match(r'^[a-f0-9\-]{36}$', upload_data['project_id']):
        result['errors'].append('Invalid project ID format')
        result['valid'] = False
    
    # Validate files
    if 'files' not in upload_data or not upload_data['files']:
        result['errors'].append('File list cannot be empty')
        result['valid'] = False
    elif len(upload_data['files']) > 100:
        result['errors'].append('Maximum number of files exceeded (100)')
        result['valid'] = False
    
    # Validate each file
    total_size = 0
    for i, file_data in enumerate(upload_data.get('files', [])):
        if not isinstance(file_data, dict):
            result['errors'].append(f'File {i+1} must be an object')
            result['valid'] = False
            continue
        
        if 'filename' not in file_data or not file_data['filename']:
            result['errors'].append(f'File {i+1} must have a name')
            result['valid'] = False
        
        if 'size' in file_data:
            size = file_data['size']
            if not isinstance(size, int) or size < 0:
                result['errors'].append(f'File {i+1} size must be a positive number')
                result['valid'] = False
            else:
                total_size += size
                
                # Check individual file size (10GB)
                max_file_size = 10 * 1024 * 1024 * 1024
                if size > max_file_size:
                    result['errors'].append(f'File {i+1} exceeds maximum size (10GB)')
                    result['valid'] = False
    
    # Check total size
    max_total_size = 50 * 1024 * 1024 * 1024  # 50GB
    if total_size > max_total_size:
        result['errors'].append(f'Total file size exceeds limit (50GB)')
        result['valid'] = False
    
    return result


def validate_analysis_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate analysis parameters
    
    Args:
        params: Analysis parameters
        
    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate analysis area
    if 'area_of_interest' in params:
        aoi = params['area_of_interest']
        if isinstance(aoi, dict):
            if 'coordinates' in aoi:
                coords = aoi['coordinates']
                if not isinstance(coords, list) or len(coords) < 3:
                    result['errors'].append('Area of interest must contain at least 3 points')
                    result['valid'] = False
                else:
                    for coord in coords:
                        if not isinstance(coord, list) or len(coord) != 2:
                            result['errors'].append('Coordinates must be pairs [longitude, latitude]')
                            result['valid'] = False
                        elif not all(isinstance(c, (int, float)) for c in coord):
                            result['errors'].append('Coordinates must be numeric')
                            result['valid'] = False
    
    # Validate time period
    if 'time_period' in params:
        time_period = params['time_period']
        if isinstance(time_period, dict):
            if 'start_date' in time_period:
                if not _is_valid_date(time_period['start_date']):
                    result['errors'].append('Invalid start date format')
                    result['valid'] = False
            
            if 'end_date' in time_period:
                if not _is_valid_date(time_period['end_date']):
                    result['errors'].append('Invalid end date format')
                    result['valid'] = False
            
            # Check date chronology
            if ('start_date' in time_period and 'end_date' in time_period and
                _is_valid_date(time_period['start_date']) and _is_valid_date(time_period['end_date'])):
                
                start_date = datetime.fromisoformat(time_period['start_date'])
                end_date = datetime.fromisoformat(time_period['end_date'])
                
                if start_date >= end_date:
                    result['errors'].append('Start date must be earlier than end date')
                    result['valid'] = False
                
                # Check reasonable period (no more than 1 year)
                if (end_date - start_date).days > 365:
                    result['warnings'].append('Analysis period exceeds 1 year, which may slow down processing')
    
    # Validate threshold values
    if 'thresholds' in params:
        thresholds = params['thresholds']
        for key, value in thresholds.items():
            if not isinstance(value, (int, float)):
                result['errors'].append(f'Threshold {key} must be numeric')
                result['valid'] = False
            elif not (0 <= value <= 1):
                result['warnings'].append(f'Threshold {key} should be in range [0, 1]')
    
    return result


def validate_export_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate export parameters
    
    Args:
        params: Export parameters
        
    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate export format
    if 'format' in params:
        valid_formats = ['geotiff', 'shapefile', 'csv', 'json', 'pdf']
        if params['format'] not in valid_formats:
            result['errors'].append(f'Unsupported export format. Valid: {", ".join(valid_formats)}')
            result['valid'] = False
    
    # Validate resolution
    if 'resolution' in params:
        resolution = params['resolution']
        if not isinstance(resolution, (int, float)) or resolution <= 0:
            result['errors'].append('Resolution must be a positive number')
            result['valid'] = False
        elif resolution > 1000:
            result['warnings'].append('High resolution may result in large file sizes')
    
    # Validate coordinate system
    if 'coordinate_system' in params:
        cs = params['coordinate_system']
        valid_cs = ['WGS84', 'UTM', 'Web Mercator']
        if cs not in valid_cs:
            result['warnings'].append(f'Coordinate system {cs} may not be supported')
    
    return result


def _is_valid_date(date_string: str) -> bool:
    """
    Check if date string is valid ISO format
    
    Args:
        date_string: Date string
        
    Returns:
        True if date is valid
    """
    try:
        datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


def sanitize_string(input_string: str, max_length: int = 255) -> str:
    """
    Sanitize string from potentially dangerous characters
    
    Args:
        input_string: Input string
        max_length: Maximum length
        
    Returns:
        Sanitized string
    """
    if not isinstance(input_string, str):
        return ''
    
    # Remove HTML tags
    cleaned = re.sub(r'<[^>]+>', '', input_string)
    
    # Remove special characters
    cleaned = re.sub(r'[<>"\'\&]', '', cleaned)
    
    # Limit length
    cleaned = cleaned[:max_length]
    
    return cleaned.strip()


def validate_email(email: str) -> bool:
    """
    Validate email address
    
    Args:
        email: Email address
        
    Returns:
        True if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate geographic coordinates
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        True if coordinates are valid
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180