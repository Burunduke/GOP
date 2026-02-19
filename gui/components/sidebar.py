"""
Компонент боковой панели для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html


def create_sidebar():
    """Создание боковой панели"""
    return html.Div([
        # Заголовок боковой панели
        html.Div([
            html.H5("Панель управления", className="text-center mb-4"),
            html.Hr(className="my-2"),
        ], className="p-3"),
        
        # Кнопки действий
        html.Div([
            dbc.Button(
                [html.I(className="fas fa-plus me-2"), "Новый проект"],
                id="new-project-btn",
                color="primary",
                className="w-100 mb-2",
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
                dbc.ListGroupItem([
                    html.Div([
                        html.H6("Демо проект 1", className="mb-1"),
                        html.P("Анализ поля пшеницы", className="mb-1 text-muted small"),
                        html.Div([
                            dbc.Badge("Завершен", color="success", className="me-1"),
                            html.Span("3 файла", className="text-muted small"),
                        ])
                    ])
                ], action=True, href="#"),
                dbc.ListGroupItem([
                    html.Div([
                        html.H6("Демо проект 2", className="mb-1"),
                        html.P("Тестирование индексов", className="mb-1 text-muted small"),
                        html.Div([
                            dbc.Badge("В обработке", color="warning", className="me-1"),
                            html.Span("2 файла", className="text-muted small"),
                        ])
                    ])
                ], action=True, href="#"),
            ], flush=True),
        ], className="px-3 mb-4"),
        
        # Статистика
        html.Div([
            html.H6("Статистика", className="mb-3"),
            html.Div([
                html.Div([
                    html.Span("Всего проектов:", className="text-muted"),
                    html.Span("2", className="fw-bold ms-2"),
                ], className="d-flex justify-content-between mb-2"),
                html.Div([
                    html.Span("Активных задач:", className="text-muted"),
                    html.Span("1", className="fw-bold ms-2 text-warning"),
                ], className="d-flex justify-content-between mb-2"),
                html.Div([
                    html.Span("Загружено файлов:", className="text-muted"),
                    html.Span("5", className="fw-bold ms-2"),
                ], className="d-flex justify-content-between"),
            ])
        ], className="px-3 mb-4"),
        
        # Информация о системе
        html.Div([
            html.Hr(className="my-3"),
            html.Div([
                html.P("GOP GUI v1.0.0", className="text-muted small text-center mb-1"),
                html.P("Гиперспектральный анализ", className="text-muted small text-center"),
            ])
        ], className="px-3"),
        
    ], className="sidebar bg-light border-end", style={
        "width": "300px",
        "min-height": "calc(100vh - 56px)",  # Высота экрана минус высота навбара
        "overflow-y": "auto"
    })