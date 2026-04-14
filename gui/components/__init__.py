"""
GUI components for GOP (Hyperspectral Processing and Plant Analysis)

This module provides all the Dash components used in the GOP GUI application.
"""

from .layout import create_main_layout
from .sidebar import create_sidebar
from .dashboard import create_dashboard
from .data_upload import create_data_upload_component
from .visualization import create_visualization_component
from .project_detail import create_project_detail
from .callbacks import register_callbacks
from .navigation import create_navigation
from .documentation import create_documentation_component

__all__ = [
    "create_main_layout",
    "create_sidebar",
    "create_dashboard",
    "create_data_upload_component",
    "create_visualization_component",
    "create_project_detail",
    "register_callbacks",
    "create_navigation",
    "create_documentation_component"
]