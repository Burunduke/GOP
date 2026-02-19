"""
Компонент навигации для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html


def create_navigation():
    """Создание навигационной панели"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Проекты", href="#", id="nav-projects")),
            dbc.NavItem(dbc.NavLink("Загрузка данных", href="#", id="nav-upload")),
            dbc.NavItem(dbc.NavLink("Обработка", href="#", id="nav-processing")),
            dbc.NavItem(dbc.NavLink("Анализ", href="#", id="nav-analysis")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Настройки", header=True),
                    dbc.DropdownMenuItem("Профиль", href="#"),
                    dbc.DropdownMenuItem("Конфигурация", href="#"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Выход", href="#"),
                ],
                nav=True,
                in_navbar=True,
                label="Меню",
            ),
        ],
        brand="GOP - Гиперспектральный анализ",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-0",
        sticky="top",
    )