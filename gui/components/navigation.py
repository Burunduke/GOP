"""
Navigation component for GOP GUI application

This module provides the main navigation bar for the application.
"""

import dash_bootstrap_components as dbc
from dash import html


def create_navigation() -> dbc.NavbarSimple:
    """
    Create navigation bar.
    
    Returns:
        Navigation bar component
    """
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="#", id="nav-dashboard")),
            dbc.NavItem(dbc.NavLink("Projects", href="#", id="nav-projects")),
        ],
        brand=html.A("GOP - Hyperspectral Analysis", href="#", id="nav-brand", className="navbar-brand"),
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-0",
        sticky="top",
    )