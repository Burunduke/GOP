"""
Компонент дашборда для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html
from datetime import datetime


def create_dashboard(statistics=None, recent_projects=None):
    """Создание дашборда с реальной статистикой проектов"""
    if statistics is None:
        statistics = {"total_projects": 0, "status_counts": {}, "total_files": 0, "total_size_mb": 0}
    if recent_projects is None:
        recent_projects = []
    
    # Форматирование даты в русском формате
    def format_date(date_str):
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%d.%m.%Y")
        except:
            return date_str
    
    return html.Div([
        # Заголовок страницы
        html.Div([
            html.H2("Панель управления", className="mb-4"),
            html.P("Добро пожаловать в систему гиперспектрального анализа растений GOP", 
                   className="text-muted mb-4"),
        ]),
        
        # Карточки статистики
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-project-diagram fa-2x text-primary mb-3"),
                            html.H4(str(statistics.get("total_projects", 0)), className="card-title"),
                            html.P("Всего проектов", className="card-text text-muted"),
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
                            html.P("Загружено файлов", className="card-text text-muted"),
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
                            html.P("Активных задач", className="card-text text-muted"),
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
                            html.P("Завершено анализов", className="card-text text-muted"),
                        ], className="text-center")
                    ])
                ], color="light", outline=True)
            ], width=3),
        ], className="mb-4"),
        
        # Последние проекты
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Последние проекты", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        dbc.ListGroup([
                            *[
                                dbc.ListGroupItem([
                                    html.Div([
                                        html.Div([
                                            html.H6(project.get("name", "Без названия"), 
                                                   className="mb-1"),
                                            html.P(project.get("description", "") or "Описание отсутствует", 
                                                   className="mb-1 text-muted small"),
                                            html.Div([
                                                dbc.Badge(
                                                    project.get("status_display", "Новый"), 
                                                    color=project.get("status_color", "secondary"), 
                                                    className="me-2"
                                                ),
                                                html.Span(format_date(project.get("updated_at", "")), 
                                                         className="text-muted small"),
                                            ])
                                        ], className="flex-grow-1"),
                                        html.Div([
                                            dbc.Button("Открыть", size="sm", 
                                                       color="outline-primary",
                                                       id={"type": "dashboard-project-btn", 
                                                            "index": project.get("id", "")})
                                        ], className="ms-3")
                                    ], className="d-flex align-items-center")
                                ], action=True)
                                for project in recent_projects
                            ]
                        ], flush=True),
                        
                    ])
                ])
            ], width=12),
            
        ]),
    ], className="dashboard")