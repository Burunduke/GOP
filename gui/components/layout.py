"""
Main layout for GOP GUI application

This module provides the main application layout including data stores,
URL routing, modals, and the overall page structure.
"""

from typing import Any
import dash_bootstrap_components as dbc
from dash import html, dcc

from .sidebar import create_sidebar
from .dashboard import create_dashboard
from .data_upload import create_data_upload_component
from .visualization import create_visualization_component
from .documentation import create_documentation_component, create_documentation_layout


def create_main_layout() -> html.Div:
    """
    Create main application layout.
    
    Returns:
        Main layout component
    """
    return html.Div([
        # Data stores
        dcc.Store(id='session-store', storage_type='session'),
        dcc.Store(id='project-store'),
        dcc.Store(id='processing-store'),
        dcc.Store(id='projects-store'),
        dcc.Store(id='current-project-store'),
        
        # URL routing
        dcc.Location(id='url', refresh=False),
        
        # Main container
        html.Div([
            # Sidebar
            create_sidebar(),  # Will be populated dynamically via callbacks
            
            # Main content
            html.Div(id='page-content', className="main-content flex-grow-1 p-4"),
        ], className="d-flex main-container"),
        
        # Modal windows
        _create_modals(),
        
        # Notifications
        dbc.Toast(id="notification-toast", is_open=False, duration=4000),
        
        # Progress update interval
        dcc.Interval(
            id='progress-interval',
            interval=1000,  # 1 second
            n_intervals=0,
            disabled=True
        ),
    ], className="app-container")


def _create_modals() -> html.Div:
    """Create modal windows."""
    return html.Div([
        # Create project modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Create New Project")),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Project Name", html_for="project-name-input"),
                            dbc.Input(
                                id="project-name-input",
                                placeholder="Enter project name",
                                type="text"
                            ),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Description", html_for="project-description-input"),
                            dbc.Textarea(
                                id="project-description-input",
                                placeholder="Enter project description (optional)",
                                rows=3
                            ),
                        ]),
                    ], className="mt-3"),
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Create", id="create-project-btn", color="primary", className="me-2"),
                dbc.Button("Cancel", id="cancel-create-project", color="secondary")
            ])
        ], id="create-project-modal", centered=True, size="lg"),
        
        # File upload modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("File Upload")),
            dbc.ModalBody([
                html.P("Select hyperspectral data to upload:"),
                html.P("Supported formats: BIL/HDR, TIFF, DAT", className="text-muted small"),
                
                dcc.Upload(
                    id='file-upload',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                        html.P("Drag and drop files here or click to select"),
                        html.P("Maximum file size: 10GB", className="text-muted small")
                    ]),
                    multiple=True,
                    className="upload-area p-4 border border-dashed rounded text-center"
                ),
                
                html.Div(id='upload-file-list', className="mt-3"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Upload", id="upload-files-modal-btn", color="primary", className="me-2"),
                dbc.Button("Cancel", id="cancel-upload", color="secondary")
            ])
        ], id="upload-files-modal", centered=True, size="lg"),
        
        # Processing settings modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Processing Settings")),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Sensor Type", html_for="sensor-type-select"),
                            dbc.Select(
                                id="sensor-type-select",
                                options=[
                                    {"label": "Hyperspectral", "value": "hyperspectral"},
                                    {"label": "Multispectral", "value": "multispectral"},
                                ],
                                value="hyperspectral"
                            ),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Vegetation Indices", html_for="indices-select"),
                            dcc.Dropdown(
                                id="indices-select",
                                options=[
                                    {"label": "NDVI", "value": "NDVI"},
                                    {"label": "EVI", "value": "EVI"},
                                    {"label": "SAVI", "value": "SAVI"},
                                ],
                                value=["NDVI", "EVI"],
                                multi=True
                            ),
                        ]),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id="processing-options",
                                options=[
                                    {"label": "Apply atmospheric correction", "value": "atmospheric_correction"},
                                ],
                                value=["atmospheric_correction"],
                            ),
                        ]),
                    ], className="mt-3"),
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Start Processing", id="start-processing-btn", color="primary", className="me-2"),
                dbc.Button("Cancel", id="cancel-processing", color="secondary")
            ])
        ], id="processing-settings-modal", centered=True, size="lg"),
        
        # Delete project modal
        dbc.Modal([
            dbc.ModalHeader("Delete Project"),
            dbc.ModalBody("Are you sure you want to delete this project?"),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-delete-project-btn", className="me-2"),
                dbc.Button("Delete", id="confirm-delete-project-btn", color="danger"),
            ]),
        ], id="delete-project-modal", is_open=False),
    ])

