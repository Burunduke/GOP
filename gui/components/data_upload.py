"""
Data upload component for GOP GUI application

This module provides components for uploading hyperspectral data files.
"""

from typing import Dict, Any
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_data_upload_component() -> html.Div:
    """
    Create data upload component.
    
    Returns:
        Data upload layout component
    """
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H5("Hyperspectral Data Upload", className="mb-0"),
            ]),
            dbc.CardBody([
                # Information about supported formats
                dbc.Alert([
                    html.H6("Supported Formats:", className="alert-heading"),
                    html.Ul([
                        html.Li("BIL/HDR - standard hyperspectral data format"),
                        html.Li("TIFF/TIFF - geospatial images"),
                        html.Li("DAT - raw spectrometer data"),
                    ]),
                    html.H6("Data Sources:", className="alert-heading mt-3"),
                    html.Ul([
                        html.Li(html.A("NASA EarthData", href="https://search.earthdata.nasa.gov/", target="_blank")),
                        html.Li(html.A("GLIHT Data", href="https://glihtdata.gsfc.nasa.gov/", target="_blank")),
                        html.Li(html.A("Open Aerial Map", href="https://map.openaerialmap.org/", target="_blank")),
                        html.Li(html.A("AVIRIS Data", href="https://popo.jpl.nasa.gov/mmgis-aviris/", target="_blank")),
                    ])
                ], color="info", className="mb-4"),
                
                # Upload area
                html.Div([
                    dcc.Upload(
                        id='data-upload',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt fa-3x mb-3 text-primary"),
                            html.H4("Drag and drop files here"),
                            html.P("or click to select files", className="text-muted"),
                            html.P("Maximum size: 10GB", className="text-muted small"),
                        ]),
                        multiple=True,
                        className="upload-area p-5 border border-dashed rounded text-center",
                        style=upload_style()
                    ),
                ], className="mb-4"),
                
                # Uploaded files list
                html.Div([
                    html.H6("Uploaded Files:", className="mb-3"),
                    html.Div(id='uploaded-files-list', children=[
                        html.P("No files uploaded yet", className="text-muted text-center py-3")
                    ]),
                ]),
                
                # File information
                html.Div(id='file-info', className="mt-3"),
                
                # Action buttons
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-check me-2"), "Start Processing"],
                        id="start-processing-from-upload",
                        color="success",
                        className="me-2",
                        disabled=True
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-trash me-2"), "Clear All"],
                        id="clear-uploaded-files",
                        color="outline-danger",
                        disabled=True
                    ),
                ], className="mt-4"),
            ])
        ])
    ], id="data-upload-container")


def upload_style() -> Dict[str, str]:
    """Styles for upload widget."""
    return {
        'borderWidth': '2px',
        'borderStyle': 'dashed',
        'borderRadius': '10px',
        'backgroundColor': '#f8f9fa',
        'cursor': 'pointer',
        'transition': 'all 0.3s ease'
    }


def create_file_list_item(filename: str, filesize: int, upload_time: str) -> dbc.ListGroupItem:
    """Create uploaded file list item."""
    return dbc.ListGroupItem([
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-file me-2 text-primary"),
                    html.Strong(filename),
                ], className="d-flex align-items-center mb-1"),
                html.Div([
                    html.Span(f"Size: {format_filesize(filesize)}", className="text-muted me-3"),
                    html.Span(f"Uploaded: {upload_time}", className="text-muted"),
                ], className="small"),
            ], className="flex-grow-1"),
            html.Div([
                dbc.Button(
                    html.I(className="fas fa-times"),
                    color="link",
                    size="sm",
                    className="text-danger p-0",
                    title="Delete file"
                )
            ])
        ], className="d-flex align-items-center justify-content-between")
    ], className="mb-2")


def format_filesize(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"