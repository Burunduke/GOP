"""
Компонент визуализации для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objs as go


def create_visualization_component():
    """Создание компонента визуализации"""
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H5("Визуализация данных", className="mb-0"),
            ]),
            dbc.CardBody([
                # Панель управления визуализацией
                dbc.Row([
                    dbc.Col([
                        html.Label("Тип визуализации:"),
                        dcc.Dropdown(
                            id='visualization-type',
                            options=[
                                {'label': 'Индексное изображение', 'value': 'index_map'},
                                {'label': 'Гистограмма распределения', 'value': 'histogram'},
                                {'label': 'Спектральный профиль', 'value': 'spectral_profile'},
                                {'label': '3D визуализация', 'value': '3d_visualization'},
                            ],
                            value='index_map',
                            className="mb-3"
                        )
                    ], width=4),
                    
                    dbc.Col([
                        html.Label("Вегетационный индекс:"),
                        dcc.Dropdown(
                            id='index-selector',
                            options=[
                                {'label': 'NDVI', 'value': 'NDVI'},
                                {'label': 'EVI', 'value': 'EVI'},
                                {'label': 'SAVI', 'value': 'SAVI'},
                            ],
                            value='NDVI',
                            className="mb-3"
                        )
                    ], width=4),
                    
                    dbc.Col([
                        html.Label("Цветовая схема:"),
                        dcc.Dropdown(
                            id='colormap-selector',
                            options=[
                                {'label': 'Viridis', 'value': 'viridis'},
                                {'label': 'Plasma', 'value': 'plasma'},
                                {'label': 'RdYlGn', 'value': 'RdYlGn'},
                                {'label': 'RdYlBu', 'value': 'RdYlBu'},
                            ],
                            value='viridis',
                            className="mb-3"
                        )
                    ], width=4),
                ], className="mb-4"),
                
                # Область визуализации
                html.Div([
                    dcc.Graph(
                        id='main-visualization',
                        figure=create_empty_figure(),
                        style={'height': '500px'}
                    )
                ], className="mb-4"),
                
                # Панель инструментов
                dbc.Row([
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button(
                                [html.I(className="fas fa-search-plus me-1"), "Увеличить"],
                                id="zoom-in-btn",
                                size="sm",
                                outline=True
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-search-minus me-1"), "Уменьшить"],
                                id="zoom-out-btn",
                                size="sm",
                                outline=True
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-expand me-1"), "Во весь экран"],
                                id="fullscreen-btn",
                                size="sm",
                                outline=True
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-download me-1"), "Скачать"],
                                id="download-visualization-btn",
                                size="sm",
                                outline=True
                            ),
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        html.Div([
                            html.Label("Прозрачность:", className="me-2"),
                            dcc.Slider(
                                id='transparency-slider',
                                min=0,
                                max=1,
                                step=0.1,
                                value=0.8,
                                marks={0: '0%', 0.5: '50%', 1: '100%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], className="d-flex align-items-center")
                    ], width=6),
                ], className="mb-4"),
                
                # Информационная панель
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Статистика изображения", className="card-title"),
                                html.Div(id='image-stats', children=[
                                    html.P("Загрузите данные для отображения статистики", 
                                           className="text-muted small")
                                ])
                            ])
                        ], color="light", outline=True)
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Информация о пикселе", className="card-title"),
                                html.Div(id='pixel-info', children=[
                                    html.P("Кликните по изображению для получения информации", 
                                           className="text-muted small")
                                ])
                            ])
                        ], color="light", outline=True)
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Легенда", className="card-title"),
                                html.Div(id='legend-info', children=[
                                    html.P("Легенда будет отображена здесь", 
                                           className="text-muted small")
                                ])
                            ])
                        ], color="light", outline=True)
                    ], width=4),
                ]),
            ])
        ])
    ], id="visualization-container")


def create_empty_figure():
    """Создание пустой фигуры для визуализации"""
    return {
        'data': [],
        'layout': {
            'title': 'Загрузите данные для начала визуализации',
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'paper_bgcolor': '#f8f9fa',
            'plot_bgcolor': '#f8f9fa',
            'height': 500,
            'annotations': [
                {
                    'text': 'Перетащите файлы данных в область загрузки',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'middle',
                    'font': {'size': 16, 'color': '#6c757d'}
                }
            ]
        }
    }


def create_index_map_figure(data, index_name, colormap='viridis'):
    """Создание фигуры для индексного изображения"""
    # Временная реализация - будет заменена реальными данными
    import numpy as np
    
    # Создание тестовых данных
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    z = np.random.rand(100, 100)
    
    return {
        'data': [
            go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=colormap,
                name=index_name,
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Значение: %{z:.3f}<extra></extra>'
            )
        ],
        'layout': {
            'title': f'Карта индекса {index_name}',
            'xaxis': {'title': 'Координата X'},
            'yaxis': {'title': 'Координата Y'},
            'height': 500,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
    }


def create_histogram_figure(data, index_name):
    """Создание фигуры для гистограммы распределения"""
    # Временная реализация
    import numpy as np
    
    # Создание тестовых данных
    values = np.random.normal(0.5, 0.2, 1000)
    
    return {
        'data': [
            go.Histogram(
                x=values,
                nbinsx=50,
                name=index_name,
                marker_color='rgba(55, 128, 191, 0.7)',
                hovertemplate='Диапазон: %{x}<br>Количество: %{y}<extra></extra>'
            )
        ],
        'layout': {
            'title': f'Распределение значений индекса {index_name}',
            'xaxis': {'title': 'Значение индекса'},
            'yaxis': {'title': 'Частота'},
            'height': 500,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
    }