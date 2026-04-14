"""
Documentation component for displaying Markdown documentation.

This module provides components for displaying various types of documentation
including user guides, API documentation, and FAQs.
"""

import os
from typing import Dict, Optional
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_documentation_component(doc_type: str = "user_guide") -> html.Div:
    """
    Create documentation display component.
    
    Args:
        doc_type: Type of documentation to display (user_guide, faq, api)
        
    Returns:
        Documentation layout component
    """
    
    # Define documentation file paths
    doc_paths: Dict[str, str] = {
        "user_guide": "docs/USER_GUIDE.md",
        "faq": "docs/FAQ.md",
        "api": "docs/api/_build/html/index.html"
    }
    
    file_path = doc_paths.get(doc_type)
    
    if not file_path or not os.path.exists(file_path):
        return html.Div([
            html.H3("Documentation Not Found"),
            html.P(f"File {file_path} does not exist.")
        ], className="p-4")
    
    # For HTML API documentation
    if doc_type == "api":
        return html.Div([
            html.H3("API Documentation", className="mb-4"),
            html.Iframe(
                src="/docs/api/_build/html/index.html",
                style={
                    "width": "100%",
                    "height": "800px",
                    "border": "none"
                }
            )
        ], className="p-4")
    
    # For Markdown documentation
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Define titles
        titles: Dict[str, str] = {
            "user_guide": "User Guide",
            "faq": "Frequently Asked Questions"
        }
        title = titles.get(doc_type, "Documentation")
        
        return html.Div([
            html.H3(title, className="mb-4"),
            html.Div(
                html.Div(
                    dcc.Markdown(markdown_content),
                    className="card-body"
                ),
                className="card"
            )
        ], className="p-4")
        
    except Exception as e:
        return html.Div([
            html.H3("Documentation Loading Error"),
            html.P(f"Failed to load documentation: {str(e)}")
        ], className="p-4")


def create_documentation_layout() -> html.Div:
    """Create layout for documentation page."""
    return html.Div([
        dcc.Location(id='doc-url', refresh=False),
        html.Div(id='doc-content')
    ])