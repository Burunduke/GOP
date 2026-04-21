"""
Sidebar component for GOP GUI application
"""

from typing import Optional, List, Dict, Any
import dash_bootstrap_components as dbc
from dash import html
from datetime import datetime


def create_sidebar(
    projects: Optional[List[Dict[str, Any]]] = None,
    statistics: Optional[Dict[str, Any]] = None
) -> html.Div:
    """Create sidebar with real project data
    
    Args:
        projects: List of project dictionaries with project data
        statistics: Dictionary containing project statistics
        
    Returns:
        html.Div: Sidebar component with project navigation and documentation links
    """
    if projects is None:
        projects = []
    if statistics is None:
        statistics = {"total_projects": 0, "status_counts": {}, "total_files": 0}
    
    # Date formatting function
    def format_date(date_str: str) -> str:
        """Format date string to international format
        
        Args:
            date_str: Date string in ISO format
            
        Returns:
            str: Formatted date string or original if formatting fails
        """
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, AttributeError):
            return date_str
    
    return html.Div([
        
        # Control panel header
        html.Div([
            dbc.NavLink(
                html.H4("Control Panel", className="mb-3 text-center text-decoration-none text-dark"),
                href="/dashboard",
                className="text-decoration-none",
                id="sidebar-dashboard"
            ),
            dbc.Button(
                [html.I(className="fas fa-plus me-2"), "New Project"],
                id="new-project-btn",
                color="primary",
                className="w-100 mb-4",
                outline=True
            ),
        ], className="px-3 pt-4"),
        
        # Projects list (main navigation element)
        html.Div([
            html.H6("Projects", className="mb-3"),
            dbc.ListGroup([
                *[
                    dbc.ListGroupItem([
                        html.Div([
                            html.H6(project.get("name", "Untitled"), className="mb-1"),
                            html.P(project.get("description", "") or "No description",
                                   className="mb-1 text-muted small"),
                            html.Div([
                                dbc.Badge(
                                    project.get("status_display", "New"),
                                    color=project.get("status_color", "secondary"),
                                    className="me-1"
                                ),
                                html.Span(f"{len(project.get('files', []))} file(s)",
                                         className="text-muted small"),
                                html.Span(f"{format_date(project.get('updated_at', ''))}",
                                         className="text-muted small ms-2"),
                            ], className="d-flex align-items-center")
                        ])
                    ], action=True,
                       id={"type": "project-item", "index": project.get("id", "")},
                       n_clicks=0,
                       className="mb-2")
                    for project in projects
                ]
            ], flush=True),
        ], className="px-3"),
        
        # Contact information section (keep only email)
        html.Div([
            html.Hr(className="my-3"),
            html.H6("Contact", className="mb-3"),
            dbc.Nav([
                dbc.NavLink(
                    [html.I(className="fas fa-envelope me-2"), "st087204@student.spbu.ru"],
                    href="mailto:st087204@student.spbu.ru",
                    className="small text-decoration-none"
                ),
            ], vertical=True, pills=True, className="flex-column"),
        ], className="px-3 mb-4"),

        # System information (at the bottom)
        html.Div([
            html.Hr(className="my-3"),
            html.Div([
                html.P("GOP GUI v1.0.0", className="text-muted small text-center mb-1"),
                html.P("Hyperspectral Analysis", className="text-muted small text-center"),
            ])
        ], className="px-3 mt-auto"),
        
    ], id="sidebar", className="sidebar bg-light border-end", style={
        "width": "300px",
        "min-height": "100vh",  # Full screen height
        "overflow-y": "auto"
    })