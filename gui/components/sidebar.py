"""
Компонент боковой панели для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html
from datetime import datetime


def create_sidebar(projects=None, statistics=None):
    """Создание боковой панели с реальными данными проектов"""
    if projects is None:
        projects = []
    if statistics is None:
        statistics = {"total_projects": 0, "status_counts": {}, "total_files": 0}
    
    # Форматирование даты в русском формате
    def format_date(date_str):
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%d.%m.%Y %H:%M")
        except:
            return date_str
    
    return html.Div([
        
        # Кнопки действий
        html.Div([
            dbc.Button(
                [html.I(className="fas fa-plus me-2"), "Новый проект"],
                id="new-project-btn",
                color="primary",
                className="w-100 mb-2 mt-3",
                outline=True
            ),
            dbc.Button(
                [html.I(className="fas fa-upload me-2"), "Загрузить файлы"],
                id="upload-files-btn",
                color="success",
                className="w-100 mb-2",
                outline=True
            ),
            dbc.Button(
                [html.I(className="fas fa-cogs me-2"), "Настройки обработки"],
                id="processing-settings-btn",
                color="info",
                className="w-100 mb-2",
                outline=True
            ),
        ], className="px-3 mb-4"),
        
        # Список проектов
        html.Div([
            html.H6("Проекты", className="mb-3"),
            dbc.ListGroup([
                *[
                    dbc.ListGroupItem([
                        html.Div([
                            html.H6(project.get("name", "Без названия"), className="mb-1"),
                            html.P(project.get("description", "") or "Описание отсутствует", 
                                   className="mb-1 text-muted small"),
                            html.Div([
                                dbc.Badge(
                                    project.get("status_display", "Новый"), 
                                    color=project.get("status_color", "secondary"), 
                                    className="me-1"
                                ),
                                html.Span(f"{len(project.get('files', []))} файл(ов)", 
                                         className="text-muted small"),
                                html.Span(f"{format_date(project.get('updated_at', ''))}", 
                                         className="text-muted small ms-2"),
                            ], className="d-flex align-items-center")
                        ])
                    ], action=True, href="#", 
                       id={"type": "project-item", "index": project.get("id", "")}, 
                       n_clicks=0)
                    for project in projects
                ]
            ], flush=True),
        ], className="px-3 mb-4"),
        
        # Статистика
        html.Div([
            html.H6("Статистика", className="mb-3"),
            html.Div([
                html.Div([
                    html.Span("Всего проектов:", className="text-muted"),
                    html.Span(str(statistics.get("total_projects", 0)), 
                             className="fw-bold ms-2"),
                ], className="d-flex justify-content-between mb-2"),
                html.Div([
                    html.Span("Активных задач:", className="text-muted"),
                    html.Span(str(statistics.get("status_counts", {}).get("processing", 0)), 
                             className="fw-bold ms-2 text-warning"),
                ], className="d-flex justify-content-between mb-2"),
                html.Div([
                    html.Span("Загружено файлов:", className="text-muted"),
                    html.Span(str(statistics.get("total_files", 0)), 
                             className="fw-bold ms-2"),
                ], className="d-flex justify-content-between"),
            ])
        ], className="px-3 mb-4"),
        
        # Секция помощи
        html.Div([
            html.Hr(className="my-3"),
            html.H6("Помощь", className="mb-3"),
            dbc.Nav([
                dbc.NavLink(
                    [html.I(className="fas fa-book me-2"), "Документация API"],
                    href="/docs/api",
                    id="nav-api-docs",
                    className="small text-decoration-none"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-user me-2"), "Руководство пользователя"],
                    href="/docs/user-guide",
                    id="nav-user-guide",
                    className="small text-decoration-none"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-question-circle me-2"), "Часто задаваемые вопросы"],
                    href="/docs/faq",
                    id="nav-faq",
                    className="small text-decoration-none"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-envelope me-2"), "st087204@student.spbu.ru"],
                    href="mailto:st087204@student.spbu.ru",
                    className="small text-decoration-none"
                ),
            ], vertical=True, pills=True, className="flex-column"),
        ], className="px-3 mb-4"),

        # Информация о системе (в самом низу)
        html.Div([
            html.Hr(className="my-3"),
            html.Div([
                html.P("GOP GUI v1.0.0", className="text-muted small text-center mb-1"),
                html.P("Гиперспектральный анализ", className="text-muted small text-center"),
            ])
        ], className="px-3 mt-auto"),
        
    ], id="sidebar", className="sidebar bg-light border-end", style={
        "width": "300px",
        "min-height": "calc(100vh - 56px)",  # Высота экрана минус высота навбара
        "overflow-y": "auto"
    })