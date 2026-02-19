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
                        
                        html.Div([
                            dbc.Button("Все проекты", color="link", className="p-0")
                        ], className="text-center mt-3")
                    ])
                ])
            ], width=8),
            
            dbc.Col([
                # Быстрые действия
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Быстрые действия", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        html.Div([
                            dbc.Button(
                                [html.I(className="fas fa-plus me-2"), "Новый проект"],
                                id="quick-new-project-btn",
                                color="primary",
                                className="w-100 mb-2"
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-upload me-2"), "Загрузить данные"],
                                id="quick-upload-btn",
                                color="success",
                                className="w-100 mb-2"
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-chart-line me-2"), "Анализ индексов"],
                                id="quick-analysis-btn",
                                color="info",
                                className="w-100 mb-2"
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-download me-2"), "Экспорт результатов"],
                                id="quick-export-btn",
                                color="secondary",
                                className="w-100"
                            ),
                        ])
                    ])
                ], className="mb-3"),
                
                # Справка
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Справка", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        html.P([
                            "Нужна помощь с использованием системы? ",
                            html.A("Посетите документацию", href="#", className="text-primary"),
                        ], className="small"),
                        html.P([
                            html.I(className="fas fa-book me-1"),
                            "Руководство пользователя",
                        ], className="small mb-2"),
                        html.P([
                            html.I(className="fas fa-question-circle me-1"),
                            "Часто задаваемые вопросы",
                        ], className="small mb-2"),
                        html.P([
                            html.I(className="fas fa-envelope me-1"),
                            "Поддержка: support@gop.ru",
                        ], className="small"),
                    ])
                ])
            ], width=4),
        ]),
    ], className="dashboard")