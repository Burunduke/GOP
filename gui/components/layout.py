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


def create_main_layout():
    """Создание главного макета приложения"""
    return html.Div([
        # Хранилища данных
        dcc.Store(id='session-store', storage_type='session'),
        dcc.Store(id='project-store'),
        dcc.Store(id='processing-store'),
        
        # Навигационная панель
        create_navigation(),
        
        # Основной контейнер
        html.Div([
            # Боковая панель
            create_sidebar(),
            
            # Основное содержимое
            html.Div(id='main-content', children=[
                create_dashboard()
            ], className="main-content flex-grow-1 p-4"),
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
                    dbcFormGroup([
                        dbc.Label("Название проекта", html_for="project-name-input"),
                        dbc.Input(
                            id="project-name-input",
                            placeholder="Введите название проекта",
                            type="text"
                        ),
                    ]),
                    dbcFormGroup([
                        dbc.Label("Описание", html_for="project-description-input"),
                        dbc.Textarea(
                            id="project-description-input",
                            placeholder="Введите описание проекта (необязательно)",
                            rows=3
                        ),
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
                dbc.Button("Загрузить", id="upload-files-btn", color="primary", className="me-2"),
                dbc.Button("Отмена", id="cancel-upload", color="secondary")
            ])
        ], id="upload-files-modal", centered=True, size="lg"),
        
        # Модальное окно настроек обработки
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Настройки обработки")),
            dbc.ModalBody([
                dbc.Form([
                    dbcFormGroup([
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
                    dbcFormGroup([
                        dbc.Label("Вегетационные индексы", html_for="indices-select"),
                        dbc.Select(
                            id="indices-select",
                            options=[
                                {"label": "NDVI", "value": "NDVI"},
                                {"label": "EVI", "value": "EVI"},
                                {"label": "SAVI", "value": "SAVI"},
                            ],
                            value=["NDVI", "EVI"],
                            multi=True
                        ),
                    ], className="mt-3"),
                    dbcFormGroup([
                        dbc.Checklist(
                            id="processing-options",
                            options=[
                                {"label": "Применить коррекцию атмосферы", "value": "atmospheric_correction"},
                                {"label": "Удалить шум", "value": "denoising"},
                                {"label": "Сегментация растений", "value": "segmentation"},
                            ],
                            value=["atmospheric_correction", "denoising"],
                        ),
                    ], className="mt-3"),
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Начать обработку", id="start-processing-btn", color="primary", className="me-2"),
                dbc.Button("Отмена", id="cancel-processing", color="secondary")
            ])
        ], id="processing-settings-modal", centered=True, size="lg"),
    ])


# Вспомогательный компонент для группировки форм
def dbcFormGroup(children, **kwargs):
    """Обертка для dbc.FormGroup для совместимости"""
    return html.Div(children, className="mb-3", **kwargs)