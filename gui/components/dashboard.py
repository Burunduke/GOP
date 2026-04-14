"""
Dashboard component for GOP GUI application

This module provides the main dashboard view showing project statistics and recent projects.
"""

from typing import Any, Dict, List, Optional
import dash_bootstrap_components as dbc
from dash import html
from datetime import datetime


def create_dashboard(
    statistics: Optional[Dict[str, Any]] = None,
    all_projects: Optional[List[Dict[str, Any]]] = None
) -> html.Div:
    """
    Create project management dashboard.
    
    Args:
        statistics: Project statistics dictionary
        all_projects: List of all project dictionaries
        
    Returns:
        Dashboard layout component
    """
    if statistics is None:
        statistics = {"total_projects": 0, "status_counts": {}, "total_files": 0, "total_size_mb": 0}
    if all_projects is None:
        all_projects = []
    
    # Format date in international format
    def format_date(date_str: str) -> str:
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return date_str
    
    # Sort projects by update date (newest first)
    sorted_projects = sorted(all_projects, key=lambda x: x.get('updated_at', ''), reverse=True)
    
    return html.Div([
        # Page header
        html.Div([
            html.H2("Project Management Dashboard", className="mb-4"),
            html.P("Manage all hyperspectral analysis projects",
                   className="text-muted mb-4"),
        ]),
        
        # Statistics cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-project-diagram fa-2x text-primary mb-3"),
                            html.H4(str(statistics.get("total_projects", 0)), className="card-title"),
                            html.P("Total Projects", className="card-text text-muted"),
                        ], className="text-center")
                    ])
                ], color="light", outline=True)
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-file-upload fa-2x text-success mb-3"),
                            html.H4(str(statistics.get("total_files", 0)), className="card-title"),
                            html.P("Files Uploaded", className="card-text text-muted"),
                        ], className="text-center")
                    ])
                ], color="light", outline=True)
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-cogs fa-2x text-warning mb-3"),
                            html.H4(str(statistics.get("status_counts", {}).get("processing", 0)),
                                   className="card-title"),
                            html.P("Active Tasks", className="card-text text-muted"),
                        ], className="text-center")
                    ])
                ], color="light", outline=True)
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-check-circle fa-2x text-info mb-3"),
                            html.H4(str(statistics.get("status_counts", {}).get("completed", 0)),
                                   className="card-title"),
                            html.P("Completed Analyses", className="card-text text-muted"),
                        ], className="text-center")
                    ])
                ], color="light", outline=True)
            ], width=3),
        ], className="mb-4"),
        
        # Main content area with projects and documentation
        dbc.Row([
            # Projects column (reduced width)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5(f"All Projects ({len(sorted_projects)})", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        dbc.ListGroup([
                            *[
                                dbc.ListGroupItem([
                                    html.Div([
                                        html.Div([
                                            html.H6(project.get("name", "Untitled"),
                                                   className="mb-1"),
                                            html.P(project.get("description", "") or "No description",
                                                   className="mb-1 text-muted small"),
                                            html.Div([
                                                dbc.Badge(
                                                    project.get("status_display", "New"),
                                                    color=project.get("status_color", "secondary"),
                                                    className="me-2"
                                                ),
                                                html.Span(f"{len(project.get('files', []))} file(s)",
                                                         className="text-muted small me-3"),
                                                html.Span(format_date(project.get("updated_at", "")),
                                                         className="text-muted small"),
                                            ], className="d-flex align-items-center")
                                        ], className="flex-grow-1"),
                                        html.Div([
                                            dbc.Button("Open", size="sm",
                                                       color="outline-primary",
                                                       id={"type": "dashboard-project-btn",
                                                            "index": project.get("id", "")})
                                        ], className="ms-3")
                                    ], className="d-flex align-items-center")
                                ], action=True)
                                for project in sorted_projects
                            ]
                        ], flush=True),
                        
                    ])
                ])
            ], width=8),  # Reduced from 12 to 8
            
            # Documentation column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Documentation", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        html.P("Quick access to project documentation:", className="text-muted"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Div([
                                    html.I(className="fas fa-book me-2 text-primary"),
                                    html.Span("User Guide", className="fw-bold")
                                ]),
                                html.Small("Complete user manual", className="text-muted d-block mt-1")
                            ], action=True, href="/docs/user-guide", target="_blank"),
                            
                            dbc.ListGroupItem([
                                html.Div([
                                    html.I(className="fas fa-code me-2 text-success"),
                                    html.Span("API Documentation", className="fw-bold")
                                ]),
                                html.Small("Technical API reference", className="text-muted d-block mt-1")
                            ], action=True, href="/docs/api", target="_blank"),
                            
                            dbc.ListGroupItem([
                                html.Div([
                                    html.I(className="fas fa-question-circle me-2 text-info"),
                                    html.Span("FAQ", className="fw-bold")
                                ]),
                                html.Small("Frequently asked questions", className="text-muted d-block mt-1")
                            ], action=True, href="/docs/faq", target="_blank"),
                            
                            dbc.ListGroupItem([
                                html.Div([
                                    html.I(className="fas fa-graduation-cap me-2 text-warning"),
                                    html.Span("Examples", className="fw-bold")
                                ]),
                                html.Small("Code examples and tutorials", className="text-muted d-block mt-1")
                            ], action=True, href="/docs/examples", target="_blank"),
                        ], flush=True),
                        
                        html.Hr(),
                        
                        html.Div([
                            html.H6("Quick Actions", className="mb-3"),
                            dbc.Button("Create New Project", color="primary", size="sm", className="me-2"),
                            dbc.Button("Upload Files", color="outline-secondary", size="sm"),
                        ], className="mt-3")
                    ])
                ], className="h-100")
            ], width=4),  # Documentation column
        ]),
    ], className="dashboard")