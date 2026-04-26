"""
Project detail page component for GOP GUI application

This module provides the detailed project view with tabs for overview, files,
processing, and results.
"""

from typing import Any, Dict, Optional
import dash_bootstrap_components as dbc
from dash import html

from gui.utils.format_utils import format_date, format_file_size, get_stage_display_name

from gui.components.enhanced_file_picker import create_enhanced_file_picker


def _get_run_display_name(run_data: dict) -> str:
    """
    Get display name for a run based on run_folder_name or fallback to index.
    
    Args:
        run_data: Run data dictionary
        
    Returns:
        Display name for the run
    """
    run_folder_name = run_data.get("run_folder_name")
    if run_folder_name and run_folder_name.startswith("run_"):
        try:
            run_number = int(run_folder_name.split("_")[1])
            return f"Run {run_number}"
        except (ValueError, IndexError):
            # Invalid format, fall back to default
            pass
    # Fallback to generic naming for legacy runs or if no folder name
    return f"Run (Legacy)"

def create_project_detail(project: Optional[Dict[str, Any]] = None) -> html.Div:
    """
    Create project detail page.
    
    Args:
        project: Project data dictionary
    Returns:
        Project detail layout component
    """
    if project is None:
        project = {
            "id": "",
            "name": "No project selected",
            "description": "",
            "status": "new",
            "created_at": "",
            "updated_at": "",
            "files": [],
            "processing_config": {},
            "current_stage": None,
            "progress": 0.0,
            "processing_history": [],
            "tags": []
}
    # Project header
    project_header = html.Div([
        dbc.Row([
            dbc.Col([
                html.H2(project.get("name", "Untitled"), className="mb-2"),
                html.P(project.get("description", "") or "No description",
                       className="text-muted mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Badge(
                            project.get("status_display", "New"),
                            color=project.get("status_color", "secondary"),
                            className="me-2"
                        ),
                        html.Small(f"Created: {format_date(project.get('created_at', ''))}",
                                  className="text-muted me-3"),
                        html.Small(f"Updated: {format_date(project.get('updated_at', ''))}",
                                  className="text-muted"),
                    ], width=8),
                    dbc.Col([
                        html.Div([
                            html.Span(f"Files: {len(project.get('files', []))}",
                                     className="text-muted me-3"),
                            html.Span(f"Size: {format_file_size(sum(f.get('file_size', 0) for f in project.get('files', [])))}",
                                     className="text-muted"),
                        ], className="text-end"),
                    ], width=4),
                ]),
            ]),
        ]),
    ], className="mb-4")
    
    # Project tabs
    tabs = dbc.Tabs([
        # Overview tab
        dbc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Project Information"),
                            dbc.CardBody([
                                html.Div([
                                    html.Strong("Project ID:"),
                                    html.P(project.get("id", "-"), className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Status:"),
                                    html.P(project.get("status_display", "-"), className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Created:"),
                                    html.P(format_date(project.get("created_at", "-")),
                                           className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Last Updated:"),
                                    html.P(format_date(project.get("updated_at", "-")),
                                           className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("File Count:"),
                                    html.P(str(len(project.get("files", []))), className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Total Size:"),
                                    html.P(format_file_size(sum(f.get("file_size", 0) for f in project.get("files", []))),
                                           className="text-muted"),
                                ]),
                            ])
                        ], className="mb-4"),
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Processing Statistics"),
                            dbc.CardBody([
                                html.Div([
                                    html.Strong("Processing Runs:"),
                                    html.P(str(len(project.get("processing_history", []))),
                                           className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Current Stage:"),
                                    html.Div([
                                        html.Span(
                                            get_stage_display_name(project.get("current_stage", "Not started") or "Not started"),
                                            id="current-stage-display",
                                            className="fw-bold me-2"
                                        ),
                                        html.Span(
                                            f" ({project.get('progress', 0):.1f}%)" if project.get("current_stage") else "",
                                            id="current-stage-progress",
                                            className="text-muted"
                                        ),
                                        dbc.Badge(
                                            "Done" if project.get("progress", 0) == 100 and project.get("current_stage")
                                            else "Run" if project.get("current_stage") and project.get("status") == "run"
                                            else "Pending" if not project.get("current_stage") and project.get("status") in ["ready", "new"]
                                            else "Error" if project.get("status") == "error"
                                            else "Cancelled" if project.get("status") == "cancelled"
                                            else "Pending",
                                            color="success" if project.get("progress", 0) == 100 and project.get("current_stage")
                                            else "warning" if project.get("current_stage") and project.get("status") == "run"
                                            else "secondary" if not project.get("current_stage") and project.get("status") in ["ready", "new"]
                                            else "danger" if project.get("status") == "error"
                                            else "dark" if project.get("status") == "cancelled"
                                            else "secondary",
                                            className="ms-2",
                                            id="current-stage-badge"
                                        )
                                    ], className="d-flex align-items-center"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Progress:"),
                                    dbc.Progress(
                                        id="overview-progress-bar",
                                        value=project.get("progress", 0),
                                        max=100,
                                        label=f"{project.get('progress', 0)}%",
                                        className="mb-2"
                                    ),
                                ]),
                            ])
                        ]),
                    ], width=6),
                ]),
            ])
        ], label="Overview", tab_id="overview-tab"),
        
        # Files tab
        dbc.Tab([
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                                            html.H5("Project Files", className="mb-0"),
                                        ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody([
                        dbc.ListGroup([
                            *[
                                dbc.ListGroupItem([
                                    html.Div([
                                        html.Div([
                                            html.H6(file.get("filename", "Untitled"),
                                                   className="mb-1"),
                                            html.Div([
                                                html.P(f"Type: гиперспектральный ({'RGB' if file.get('file_type') == 'rgb' else 'HS'}) | "
                                                       f"Size: {format_file_size(file.get('file_size', 0))} | "
                                                       f"Uploaded: {format_date(file.get('upload_date', ''))}",
                                                       className="mb-1 text-muted small"),
                                            ], className="d-flex align-items-center"),
                                        ], className="flex-grow-1"),
                                        html.Div([
                                            dbc.Button(
                                                html.I(className="fas fa-times"),
                                                id={"type": "project-file-delete",
                                                     "index": file.get("id", "")},
                                                color="outline-danger",
                                                size="sm",
                                                className="delete-file-btn"
                                            ),
                                        ], className="ms-3 d-flex align-items-center"),
                                    ], className="d-flex align-items-center")
                                ])
                                for file in project.get("files", [])
                            ]
                        ], flush=True),
                        
                        
                        # Enhanced file picker (OS-native file dialog)
                        create_enhanced_file_picker(),
                    ])
                ]),
            ])
        ], label="Files", tab_id="files-tab"),
        # Processing tab
        dbc.Tab([
            html.Div([
                dbc.Card([
                    dbc.CardHeader("Processing Configuration"),
                    dbc.CardBody([
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Processing Stages", className="mb-3"),
                                    dbc.Checklist(
                                        id="stage-checkboxes",
                                        options=[
                                            {"label": "Preprocessing", "value": "preprocessing"},
                                            {"label": "Orthophoto", "value": "orthophoto"},
                                        ],
                                        value=project.get("processing_config", {}).get("stages", []),
                                        inline=False,
                                        className="mb-3"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.H6("Settings", className="mb-3"),
                                    html.Div([
                                        dbc.Checklist(
                                            id="processing-options",
                                            options=[
                                                {"label": "Apply atmospheric correction",
                                                 "value": "atmospheric_correction"},
                                            ],
                                            value=["atmospheric_correction"],
                                        ),
                                    ]),
                                ], width=6),
                            ]),
                        ]),
                        
                        # Processing progress
                        html.Div(id="processing-progress-section", children=[
                            html.Hr(className="my-4"),
                            html.H6("Processing Progress", className="mb-3"),
                            dbc.Progress(
                                id="project-processing-progress",
                                value=project.get("progress", 0),
                                max=100,
                                label=f"{project.get('progress', 0)}%",
                                className="mb-3"
                            ),
                            html.Div([
                                dbc.Button(
                                    [html.I(className="fas fa-play me-2"), "Start Processing"],
                                    id="project-start-processing-btn",
                                    color="primary",
                                    className="me-2"
                                ),
                                dbc.Button(
                                    [html.I(className="fas fa-stop me-2"), "Stop"],
                                    id="project-cancel-processing-btn",
                                    color="secondary",
                                    disabled=True
                                ),
                            ], className="text-center"),
                        ]),
                    ])
                ]),
            ])
        ], label="Processing", tab_id="processing-tab"),
        
        # Results tab
        dbc.Tab([
            html.Div([
                dbc.Card([
                    dbc.CardHeader("Processing History"),
                    dbc.CardBody([
                        dbc.ListGroup(
                            id="processing-history-list",
                            children=[
                                *[
                                    dbc.ListGroupItem([
                                        html.Div([
                                            html.Div([
                                                html.H6(_get_run_display_name(run), className="mb-1"),
                                                html.P(f"Start: {format_date(run.get('start_time', ''))} | "
                                                       f"Status: {run.get('status', 'unknown')}",
                                                       className="mb-1 text-muted small"),
                                                html.Div([
                                                    dbc.Badge(
                                                        "Done" if run.get("status") == "completed"
                                                        else "Run" if run.get("status") == "running"
                                                        else "Error" if run.get("status") == "error"
                                                        else "Cancelled",
                                                        color="success" if run.get("status") == "completed"
                                                        else "warning" if run.get("status") == "running"
                                                        else "danger" if run.get("status") == "error"
                                                        else "secondary",
                                                        className="me-2"
                                                    ),
                                                    html.Small(f"Duration: {run.get('total_duration_seconds', 0) if run.get('total_duration_seconds') is not None else 0:.1f} sec",
                                                              className="text-muted"),
                                                ]),
                                            ], className="flex-grow-1"),
                                            html.Div([
                                                dbc.Button(
                                                    "View",
                                                    id={"type": "view-run-results",
                                                         "index": run.get("run_id", "")},
                                                    color="outline-primary",
                                                    size="sm"
                                                ),
                                            ], className="ms-3"),
                                        ], className="d-flex align-items-center")
                                    ])
                                    for i, run in enumerate(project.get("processing_history", []))
                                ]
                            ],
                            flush=True
                        ),
                        
                        html.Div(id="run-results-details", className="mt-4"),
                    ])
                ]),
            ])
        ], label="Results", tab_id="results-tab"),
    ], id="project-detail-tabs", active_tab="overview-tab")
    
    return html.Div([
        project_header,
        tabs
    ], className="project-detail")