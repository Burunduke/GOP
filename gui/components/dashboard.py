"""
Компонент дашборда для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html


def create_dashboard():
    """Создание дашборда"""
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
                            html.H4("2", className="card-title"),
                            html.P("Активных проекта", className="card-text text-muted"),
                        ], className="text-center")
                    ])
                ], color="light", outline=True)
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-file-upload fa-2x text-success mb-3"),
                            html.H4("5", className="card-title"),
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
                            html.H4("1", className="card-title"),
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
                            html.H4("3", className="card-title"),
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
                            dbc.ListGroupItem([
                                html.Div([
                                    html.Div([
                                        html.H6("Анализ поля пшеницы", className="mb-1"),
                                        html.P("NDVI анализ для оценки состояния посевов", 
                                               className="mb-1 text-muted small"),
                                        html.Div([
                                            dbc.Badge("Завершен", color="success", className="me-2"),
                                            html.Span("15.01.2024", className="text-muted small"),
                                        ])
                                    ], className="flex-grow-1"),
                                    html.Div([
                                        dbc.Button("Открыть", size="sm", color="outline-primary")
                                    ], className="ms-3")
                                ], className="d-flex align-items-center")
                            ], action=True),
                            
                            dbc.ListGroupItem([
                                html.Div([
                                    html.Div([
                                        html.H6("Тестирование индексов", className="mb-1"),
                                        html.P("Сравнительный анализ различных вегетационных индексов", 
                                               className="mb-1 text-muted small"),
                                        html.Div([
                                            dbc.Badge("В обработке", color="warning", className="me-2"),
                                            html.Span("16.01.2024", className="text-muted small"),
                                        ])
                                    ], className="flex-grow-1"),
                                    html.Div([
                                        dbc.Button("Открыть", size="sm", color="outline-primary")
                                    ], className="ms-3")
                                ], className="d-flex align-items-center")
                            ], action=True),
                        ], flush=True),
                        
                    ])
                ])
            ], width=12),
            
        ]),
    ], className="dashboard")