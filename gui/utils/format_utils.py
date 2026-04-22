"""
Formatting utilities for GOP GUI application
"""

from datetime import datetime


def format_date(date_str: str) -> str:
    """
    Format date string to international format.
    
    Args:
        date_str: Date string in ISO format
        
    Returns:
        Formatted date string
    """
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return date_str


def format_file_size(size_bytes: int) -> str:
    """
    Format file size for display.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def get_stage_display_name(stage_key: str) -> str:
    """
    Convert stage key to user-friendly display name.
    
    Args:
        stage_key: Stage key
        
    Returns:
        Display name
    """
    stage_names = {
        "preprocessing": "Preprocessing",
        "orthophoto": "Orthophoto Generation",
        "Not started": "Not started"
    }
    return stage_names.get(stage_key, stage_key)