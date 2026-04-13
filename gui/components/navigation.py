"""
Компонент навигации для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html


def create_navigation():
    """Создание навигационной панели"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Панель управления", href="#", id="nav-dashboard")),
            dbc.NavItem(dbc.NavLink("Проекты", href="#", id="nav-projects")),
            dbc.NavItem(dbc.NavLink("Загрузка данных", href="#", id="nav-upload")),
            dbc.NavItem(dbc.NavLink("Обработка", href="#", id="nav-processing")),
            dbc.NavItem(dbc.NavLink("Анализ", href="#", id="nav-analysis")),
        ],
        brand=html.A("GOP - Гиперспектральный анализ", href="#", id="nav-brand", className="navbar-brand"),
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-0",
        sticky="top",
    )