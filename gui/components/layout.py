"""
Основной layout для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html, dcc

from .navigation import create_navigation
from .sidebar import create_sidebar
from .dashboard import create_dashboard
from .data_upload import create_data_upload_component
from .visualization import create_visualization_component
from .documentation import create_documentation_component, create_documentation_layout


def create_main_layout():
    """Создание главного макета приложения"""
    return html.Div([
        # Хранилища данных
        dcc.Store(id='session-store', storage_type='session'),
        dcc.Store(id='project-store'),
        dcc.Store(id='processing-store'),
        dcc.Store(id='projects-store'),
        dcc.Store(id='current-project-store'),
        
        # URL routing
        dcc.Location(id='url', refresh=False),
        
        # Навигационная панель
        create_navigation(),
        
        # Основной контейнер
        html.Div([
            # Боковая панель
            create_sidebar(),  # Will be populated dynamically via callbacks
            
            # Основное содержимое
            html.Div(id='page-content', className="main-content flex-grow-1 p-4"),
        ], className="d-flex main-container"),
        
        # Модальные окна
        _create_modals(),
        
        # Уведомления
        dbc.Toast(id="notification-toast", is_open=False, duration=4000),
        
        # Интервал для обновления прогресса
        dcc.Interval(
            id='progress-interval',
            interval=1000,  # 1 секунда
            n_intervals=0,
            disabled=True
        ),
    ], className="app-container")


def _create_modals():
    """Создание модальных окон"""
    return html.Div([
        # Модальное окно создания проекта
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Создание нового проекта")),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Название проекта", html_for="project-name-input"),
                            dbc.Input(
                                id="project-name-input",
                                placeholder="Введите название проекта",
                                type="text"
                            ),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Описание", html_for="project-description-input"),
                            dbc.Textarea(
                                id="project-description-input",
                                placeholder="Введите описание проекта (необязательно)",
                                rows=3
                            ),
                        ]),
                    ], className="mt-3"),
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Создать", id="create-project-btn", color="primary", className="me-2"),
                dbc.Button("Отмена", id="cancel-create-project", color="secondary")
            ])
        ], id="create-project-modal", centered=True, size="lg"),
        
        # Модальное окно загрузки файлов
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Загрузка файлов")),
            dbc.ModalBody([
                html.P("Выберите гиперспектральные данные для загрузки:"),
                html.P("Поддерживаемые форматы: BIL/HDR, TIFF, DAT", className="text-muted small"),
                
                dcc.Upload(
                    id='file-upload',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                        html.P("Перетащите файлы сюда или нажмите для выбора"),
                        html.P("Максимальный размер файла: 10GB", className="text-muted small")
                    ]),
                    multiple=True,
                    className="upload-area p-4 border border-dashed rounded text-center"
                ),
                
                html.Div(id='upload-file-list', className="mt-3"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Загрузить", id="upload-files-modal-btn", color="primary", className="me-2"),
                dbc.Button("Отмена", id="cancel-upload", color="secondary")
            ])
        ], id="upload-files-modal", centered=True, size="lg"),
        
        # Модальное окно настроек обработки
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Настройки обработки")),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Тип сенсора", html_for="sensor-type-select"),
                            dbc.Select(
                                id="sensor-type-select",
                                options=[
                                    {"label": "Гиперспектральный", "value": "hyperspectral"},
                                    {"label": "Мультиспектральный", "value": "multispectral"},
                                ],
                                value="hyperspectral"
                            ),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Вегетационные индексы", html_for="indices-select"),
                            dcc.Dropdown(
                                id="indices-select",
                                options=[
                                    {"label": "NDVI", "value": "NDVI"},
                                    {"label": "EVI", "value": "EVI"},
                                    {"label": "SAVI", "value": "SAVI"},
                                ],
                                value=["NDVI", "EVI"],
                                multi=True
                            ),
                        ]),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id="processing-options",
                                options=[
                                    {"label": "Применить коррекцию атмосферы", "value": "atmospheric_correction"},
                                    {"label": "Удалить шум", "value": "denoising"},
                                    {"label": "Сегментация растений", "value": "segmentation"},
                                ],
                                value=["atmospheric_correction", "denoising"],
                            ),
                        ]),
                    ], className="mt-3"),
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Начать обработку", id="start-processing-btn", color="primary", className="me-2"),
                dbc.Button("Отмена", id="cancel-processing", color="secondary")
            ])
        ], id="processing-settings-modal", centered=True, size="lg"),
        
        # Модальное окно удаления проекта
        dbc.Modal([
            dbc.ModalHeader("Удаление проекта"),
            dbc.ModalBody("Вы уверены, что хотите удалить этот проект?"),
            dbc.ModalFooter([
                dbc.Button("Отмена", id="cancel-delete-project-btn", className="me-2"),
                dbc.Button("Удалить", id="confirm-delete-project-btn", color="danger"),
            ]),
        ], id="delete-project-modal", is_open=False),
    ])

