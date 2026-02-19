"""
Компоненты интерфейса для GUI GOP
"""

from .layout import create_main_layout
from .navigation import create_navigation
from .sidebar import create_sidebar
from .dashboard import create_dashboard
from .data_upload import create_data_upload_component
from .visualization import create_visualization_component
from .callbacks import register_callbacks

__all__ = [
    "create_main_layout",
    "create_navigation", 
    "create_sidebar",
    "create_dashboard",
    "create_data_upload_component",
    "create_visualization_component",
    "register_callbacks"
]