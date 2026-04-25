"""
Enhanced file picker component for GOP GUI application with OS-native file dialog.

This component provides a standard OS file dialog for selecting files to add to a project,
which is more familiar to users than the server-side file browser.
"""

from pathlib import Path
from typing import List, Dict, Any

import dash_bootstrap_components as dbc
from dash import html, dcc

from gui.utils.format_utils import format_file_size

from gui.config import config


# Supported file extensions for orthophoto / hyperspectral processing
SUPPORTED_EXTENSIONS = {
    '.bil', '.hdr', '.tif', '.tiff', '.dat',
    '.png', '.jpg', '.jpeg', '.geotiff',
}


def create_enhanced_file_picker() -> html.Div:
    """
    Create an enhanced file picker component with OS-native file dialog.
    
    The component includes:
    - Button to open OS file dialog
    - Display of selected files
    - "Add selected files" button
    
    Returns:
        Dash HTML component.
    """
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                # File upload area with OS dialog
                html.Div([
                    dcc.Upload(
                        id='enhanced-file-picker-upload',
                        children=html.Div([
                            html.I(className="fas fa-folder-open fa-2x mb-2"),
                            html.P("Click to select files from your computer", className="mb-1"),
                            html.P("or drag and drop files here", className="text-muted small mb-2"),
                            html.P("Supported formats: BIL/HDR, TIFF, DAT, PNG, JPG", 
                                  className="text-muted small mb-0"),
                        ]),
                        multiple=True,
                        className="upload-area p-4 border border-dashed rounded text-center cursor-pointer"
                    ),
                ], className="mb-3"),
                
                # Selected files summary
                html.Div(id="enhanced-file-picker-selection-summary", className="mt-2"),
                
                # Add button
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-plus me-2"), "Add Selected Files to Project"],
                        id="add-enhanced-files-btn",
                        color="success",
                        className="w-100",
                        disabled=True,
                    ),
                ], className="mt-3"),
            ]),
        ]),
        
        # Store for selected file data
        dcc.Store(id='enhanced-file-picker-store', data=[]),
    ], id="enhanced-file-picker-container")


def format_selected_files_summary(selected_files: List[Dict[str, Any]]) -> html.Div:
    """
    Format a summary of selected files for display.
    
    Args:
        selected_files: List of selected file dictionaries with name, size, content
        
    Returns:
        Dash HTML component showing file summary.
    """
    if not selected_files:
        return html.Div()
    
    total_size = sum(f.get('size', 0) for f in selected_files)
    file_count = len(selected_files)
    
    # Create list of files
    file_items = []
    for file_info in selected_files:
        file_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-file me-2"),
                        html.Span(file_info.get('name', 'Unknown'), className="fw-bold"),
                        html.Span(f"  ({format_file_size(file_info.get('size', 0))})", 
                                 className="text-muted ms-2 small"),
                    ], className="d-flex align-items-center"),
                ], className="d-flex align-items-center justify-content-between"),
            ], className="py-2")
        )
    
    return html.Div([
        html.Hr(className="my-3"),
        html.H6(f"Selected Files ({file_count})", className="mb-2"),
        dbc.ListGroup(file_items, flush=True),
        html.Div([
            html.Strong("Total Size: "),
            html.Span(format_file_size(total_size), className="text-muted"),
        ], className="mt-2 text-end"),
    ])