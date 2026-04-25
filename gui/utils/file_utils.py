"""
File utilities for GOP GUI application
"""

import os
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def validate_file_format(file_path: str, supported_formats: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate file format
    
    Args:
        file_path: Path to the file
        supported_formats: List of supported formats
        
    Returns:
        Validation result dictionary
    """
    if supported_formats is None:
        supported_formats = ['.bil', '.hdr', '.tif', '.tiff', '.dat', '.png', '.jpg', '.jpeg']
    
    result = {
        'valid': False,
        'error': None,
        'file_info': {}
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            result['error'] = 'File does not exist'
            return result
        
        # Get file information
        file_path_obj = Path(file_path)
        file_size = file_path_obj.stat().st_size
        file_ext = file_path_obj.suffix.lower()
        
        result['file_info'] = {
            'name': file_path_obj.name,
            'size': file_size,
            'extension': file_ext,
            'mime_type': mimetypes.guess_type(file_path)[0],
            'created_time': datetime.fromtimestamp(file_path_obj.stat().st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(file_path_obj.stat().st_mtime).isoformat()
        }
        
        # Check format
        if file_ext not in supported_formats:
            result['error'] = f'Unsupported format: {file_ext}. Supported: {", ".join(supported_formats)}'
            return result
        
        # Check size (maximum 10GB)
        max_size = 10 * 1024 * 1024 * 1024
        if file_size > max_size:
            result['error'] = f'File too large: {format_file_size(file_size)}. Maximum: {format_file_size(max_size)}'
            return result
        
        # Additional validation for specific formats
        if file_ext in ['.bil', '.hdr']:
            if not _validate_hyperspectral_file(file_path):
                result['error'] = 'File is not valid hyperspectral data'
                return result
        
        result['valid'] = True
        
    except Exception as e:
        result['error'] = f'Error validating file: {str(e)}'
    
    return result


def format_file_size(size_bytes: int) -> str:
    """
    Format file size to human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get file metadata
    
    Args:
        file_path: Path to the file
        
    Returns:
        File metadata dictionary
    """
    try:
        file_path_obj = Path(file_path)
        stat = file_path_obj.stat()
        
        metadata = {
            'name': file_path_obj.name,
            'path': str(file_path_obj.absolute()),
            'size': stat.st_size,
            'size_formatted': format_file_size(stat.st_size),
            'extension': file_path_obj.suffix.lower(),
            'mime_type': mimetypes.guess_type(file_path)[0],
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed_time': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'is_readable': os.access(file_path, os.R_OK),
            'is_writable': os.access(file_path, os.W_OK),
        }
        
        # Add format-specific information
        if file_path_obj.suffix.lower() in ['.bil', '.hdr']:
            metadata.update(_get_hyperspectral_metadata(file_path))
        elif file_path_obj.suffix.lower() in ['.tif', '.tiff']:
            metadata.update(_get_geotiff_metadata(file_path))
        
        return metadata
        
    except Exception as e:
        return {
            'error': f'Error getting metadata: {str(e)}',
            'path': file_path
        }


def _validate_hyperspectral_file(file_path: str) -> bool:
    """
    Validate hyperspectral file
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is valid
    """
    try:
        # Basic validation - full validation requires GDAL/spectral
        file_path_obj = Path(file_path)
        
        # Check for HDR file for BIL
        if file_path_obj.suffix.lower() == '.bil':
            hdr_file = file_path_obj.with_suffix('.hdr')
            if not hdr_file.exists():
                return False
        
        # Check minimum size
        if file_path_obj.stat().st_size < 1024:  # Minimum 1KB
            return False
        
        return True
        
    except Exception:
        return False


def _get_hyperspectral_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get hyperspectral file metadata
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hyperspectral file metadata
    """
    metadata = {
        'file_type': 'hyperspectral',
        'sensor_type': 'unknown',
        'bands_count': 0,
        'wavelength_range': None
    }
    
    try:
        file_path_obj = Path(file_path)
        
        # Try to read HDR file
        if file_path_obj.suffix.lower() == '.hdr':
            hdr_content = file_path_obj.read_text()
            # Simple HDR file parser
            if 'samples' in hdr_content.lower():
                # Extract basic information from HDR
                metadata['samples'] = _extract_hdr_value(hdr_content, 'samples')
                metadata['lines'] = _extract_hdr_value(hdr_content, 'lines')
                metadata['bands'] = _extract_hdr_value(hdr_content, 'bands')
                metadata['bands_count'] = metadata.get('bands', 0)
        
        elif file_path_obj.suffix.lower() == '.bil':
            # Search for corresponding HDR file
            hdr_file = file_path_obj.with_suffix('.hdr')
            if hdr_file.exists():
                return _get_hyperspectral_metadata(str(hdr_file))
        
    except Exception:
        pass
    
    return metadata


def _get_geotiff_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get GeoTIFF file metadata
    
    Args:
        file_path: Path to the file
        
    Returns:
        GeoTIFF file metadata
    """
    metadata = {
        'file_type': 'geotiff',
        'coordinate_system': 'unknown',
        'pixel_size': None,
        'bounds': None
    }
    
    try:
        # Full metadata extraction requires GDAL/rasterio
        # Basic implementation here
        file_path_obj = Path(file_path)
        
        # Check size to estimate resolution
        size_mb = file_path_obj.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            metadata['estimated_resolution'] = 'high'
        elif size_mb > 10:
            metadata['estimated_resolution'] = 'medium'
        else:
            metadata['estimated_resolution'] = 'low'
        
    except Exception:
        pass
    
    return metadata


def _extract_hdr_value(hdr_content: str, key: str) -> Optional[int]:
    """
    Extract numeric value from HDR file
    
    Args:
        hdr_content: HDR file content
        key: Key to search for
        
    Returns:
        Numeric value or None
    """
    try:
        lines = hdr_content.split('\n')
        for line in lines:
            if key.lower() in line.lower() and '=' in line:
                value_str = line.split('=')[1].strip()
                return int(value_str)
    except (ValueError, IndexError):
        pass
    return None


def create_safe_filename(filename: str) -> str:
    """
    Create safe filename
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    
    # Remove invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(safe_name) > 255:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:255-len(ext)] + ext
    
    return safe_name


def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure directory exists
    
    Args:
        directory: Path to directory
        
    Returns:
        True if directory exists or was created
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_directory_size(directory: str) -> int:
    """
    Get directory size in bytes
    
    Args:
        directory: Path to directory
        
    Returns:
        Size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception:
        pass
    return total_size


def cleanup_old_files(directory: str, max_age_days: int = 30) -> int:
    """
    Clean up old files in directory
    
    Args:
        directory: Path to directory
        max_age_days: Maximum file age in days
        
    Returns:
        Number of deleted files
    """
    deleted_count = 0
    cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
    
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.unlink(file_path)
                        deleted_count += 1
                except Exception:
                    pass
    except Exception:
        pass
    
    return deleted_count


def sanitize_project_name(name: str) -> str:
    """
    Sanitize project name to create a safe folder name.
    
    Replaces path-unsafe characters with underscores, strips leading/trailing
    whitespace and dots, and ensures the name is not empty.
    
    Args:
        name: Original project name from GUI
        
    Returns:
        Sanitized project name safe for use as a folder name
    """
    import re
    import string
    
    # Strip leading/trailing whitespace and dots
    sanitized = name.strip().strip('.')
    
    # Replace path-unsafe characters with underscores
    # Characters: \ / : * ? " < > | and control characters (0-31)
    unsafe_chars = r'[\\/:*?"<>|' + ''.join(chr(i) for i in range(32)) + ']'
    sanitized = re.sub(unsafe_chars, '_', sanitized)
    
    # Also replace multiple consecutive dots with single underscore
    sanitized = re.sub(r'\.\.+', '_', sanitized)
    
    return sanitized