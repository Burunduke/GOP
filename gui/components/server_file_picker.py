"""
Server-side file browser component for GOP GUI application.

Provides a full file browser UI that lets users navigate directories on the
server, select files, and add them to a project. Files are copied at the
filesystem level using shutil.copy2, which uses constant memory regardless
of file size — solving the OOM issue with browser-based uploads.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import dash_bootstrap_components as dbc
from dash import html, dcc

from gui.utils.format_utils import format_file_size

from gui.config import config


# Supported file extensions for orthophoto / hyperspectral processing
SUPPORTED_EXTENSIONS = {
    '.bil', '.hdr', '.tif', '.tiff', '.dat',
    '.png', '.jpg', '.jpeg', '.geotiff',
}


def list_directory(directory: str) -> Dict[str, Any]:
    """
    List contents of a directory, returning folders and supported files.

    Args:
        directory: Absolute or relative path to list.

    Returns:
        Dict with keys:
            - path: resolved absolute path of the directory
            - parent: parent directory path (or None if at root)
            - folders: list of dicts {name, path}
            - files: list of dicts {name, path, size, size_formatted, extension}
    """
    dir_path = Path(directory).resolve()

    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

    parent = str(dir_path.parent) if dir_path.parent != dir_path else None

    folders: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []

    try:
        for item in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith('.'):
                continue  # skip hidden files/dirs

            if item.is_dir():
                folders.append({
                    'name': item.name,
                    'path': str(item),
                })
            elif item.is_file():
                ext = item.suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    try:
                        size = item.stat().st_size
                        files.append({
                            'name': item.name,
                            'path': str(item),
                            'size': size,
                            'size_formatted': _format_file_size(size),
                            'extension': ext,
                        })
                    except OSError:
                        continue
    except PermissionError:
        pass

    return {
        'path': str(dir_path),
        'parent': parent,
        'folders': folders,
        'files': files,
    }


def get_default_browse_root() -> str:
    """Return the default root directory for the file browser."""
    app_config = config['default']
    return str(Path(app_config.UPLOAD_FOLDER).resolve())


def create_server_file_picker() -> html.Div:
    """
    Create the server-side file browser component.

    The component includes:
    - Breadcrumb-style path display with navigation
    - "Go up" button to navigate to parent directory
    - Folder list (clickable to navigate into)
    - File list with checkboxes for selection
    - "Add selected files" button

    Returns:
        Dash HTML component.
    """
    default_root = get_default_browse_root()

    return html.Div([
        # Store for current browsing path
        dcc.Store(id='file-browser-current-path', data=default_root),

        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-folder-open me-2"),
                    "Select Files",
                ], className="mb-0"),
            ]),
            dbc.CardBody([
                # Current path display + navigation
                html.Div([
                    html.Div([
                        dbc.Button(
                            [html.I(className="fas fa-arrow-up me-1"), "Up"],
                            id="file-browser-go-up-btn",
                            color="outline-secondary",
                            size="sm",
                            className="me-2",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-home me-1"), "Root"],
                            id="file-browser-go-root-btn",
                            color="outline-secondary",
                            size="sm",
                            className="me-2",
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-sync-alt me-1"), "Refresh"],
                            id="file-browser-refresh-btn",
                            color="outline-secondary",
                            size="sm",
                        ),
                    ], className="d-flex align-items-center mb-2"),

                    # Breadcrumb path
                    html.Div(
                        id="file-browser-breadcrumb",
                        className="mb-2",
                    ),
                ]),

                # Directory contents
                html.Div(
                    id="file-browser-contents",
                    children=[
                        html.P(
                            "Loading...",
                            className="text-muted text-center py-3",
                        )
                    ],
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),

                # Selected files summary
                html.Div(id="file-browser-selection-summary", className="mt-2"),

                # Add button
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-plus me-2"), "Add Selected Files to Project"],
                        id="add-server-files-btn",
                        color="success",
                        className="w-100",
                        disabled=True,
                    ),
                ], className="mt-3"),
            ]),
        ]),
    ], id="server-file-picker-container")


def render_breadcrumb(current_path: str, root_path: str) -> html.Div:
    """
    Render a breadcrumb-style path display for the current directory.

    Args:
        current_path: Current directory path.
        root_path: Root directory path (upload folder).

    Returns:
        Dash HTML component showing the current path.
    """
    current = Path(current_path).resolve()
    root = Path(root_path).resolve()

    # Build path parts relative to root
    try:
        relative = current.relative_to(root)
        if str(relative) == '.':
            parts = [root.name]
        else:
            parts = [root.name] + list(relative.parts)
    except ValueError:
        # current_path is outside root — show absolute
        parts = list(current.parts)

    # Build breadcrumb-style spans
    breadcrumb_items = []
    for i, part in enumerate(parts):
        if i > 0:
            breadcrumb_items.append(
                html.Span(" / ", className="text-muted mx-1")
            )
        if i == len(parts) - 1:
            # Active (current) item — bold
            breadcrumb_items.append(
                html.Span(part or "/", className="fw-bold")
            )
        else:
            breadcrumb_items.append(
                html.Span(part or "/", className="text-muted")
            )

    return html.Div([
        html.Small([
            html.I(className="fas fa-folder-open me-2 text-warning"),
            *breadcrumb_items,
        ], className="d-flex align-items-center"),
    ], className="py-1 px-2 bg-light rounded")


def render_directory_contents(listing: Dict[str, Any]) -> html.Div:
    """
    Render directory contents as a list of folders and files.

    Args:
        listing: Result from list_directory().

    Returns:
        Dash component with folder and file items.
    """
    items = []

    # Folders
    for folder in listing['folders']:
        items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-folder text-warning me-2"),
                        html.Span(folder['name'], className="fw-bold"),
                    ], className="d-flex align-items-center"),
                ], className="d-flex align-items-center justify-content-between"),
            ],
                id={'type': 'file-browser-folder', 'index': folder['path']},
                action=True,
                className="py-2 cursor-pointer",
                style={"cursor": "pointer"},
            )
        )

    # Files
    if listing['files']:
        file_options = []
        for f in listing['files']:
            ext_icon = _get_file_icon(f['extension'])
            label = html.Span([
                html.I(className=f"{ext_icon} me-2"),
                html.Span(f['name']),
                html.Span(f"  ({f['size_formatted']})", className="text-muted ms-2 small"),
            ])
            file_options.append({
                'label': label,
                'value': f['path'],
            })

        items.append(
            html.Div([
                dbc.Checklist(
                    id="server-files-checklist",
                    options=file_options,
                    value=[],
                    className="file-checklist",
                ),
            ], className="mt-1")
        )
    elif not listing['folders']:
        items.append(
            html.Div([
                html.P(
                    "This directory is empty or contains no supported files.",
                    className="text-muted text-center py-3 mb-0",
                ),
                html.P([
                    "Supported formats: ",
                    html.Code(", ".join(sorted(SUPPORTED_EXTENSIONS))),
                ], className="text-muted text-center small"),
            ])
        )

    if listing['folders'] and not listing['files']:
        # Add a note that there are no files but there are subdirectories
        items.append(
            html.Div([
                html.Hr(className="my-2"),
                html.P(
                    "No supported files in this directory. Browse subdirectories above.",
                    className="text-muted text-center small mb-0 py-1",
                ),
            ])
        )

    return html.Div([
        dbc.ListGroup(items, flush=True) if listing['folders'] else html.Div(items),
    ] if listing['folders'] else items)


def _get_file_icon(extension: str) -> str:
    """Return a Font Awesome icon class for a file extension."""
    icon_map = {
        '.tif': 'fas fa-map text-success',
        '.tiff': 'fas fa-map text-success',
        '.geotiff': 'fas fa-map text-success',
        '.bil': 'fas fa-layer-group text-primary',
        '.hdr': 'fas fa-file-alt text-info',
        '.dat': 'fas fa-database text-secondary',
        '.png': 'fas fa-image text-success',
        '.jpg': 'fas fa-image text-success',
        '.jpeg': 'fas fa-image text-success',
    }
    return icon_map.get(extension, 'fas fa-file text-muted')


def _format_file_size(size_bytes: int) -> str:
    """Format file size for display."""
    return format_file_size(size_bytes)
