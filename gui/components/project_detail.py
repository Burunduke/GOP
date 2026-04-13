"""
Компонент детальной страницы проекта для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
from datetime import datetime


def create_project_detail(project=None):
    """Создание детальной страницы проекта"""
    if project is None:
        project = {
            "id": "",
            "name": "Проект не выбран",
            "description": "",
            "status": "new",
            "created_at": "",
            "updated_at": "",
            "files": [],
            "processing_config": {},
            "current_stage": None,
            "progress": 0.0,
            "processing_history": [],
            "tags": []
        }
    
    # Форматирование даты в русском формате
    def format_date(date_str):
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%d.%m.%Y %H:%M")
        except:
            return date_str
    
    # Форматирование размера файла
    def format_file_size(size_bytes):
        if size_bytes == 0:
            return "0 Б"
        size_names = ["Б", "КБ", "МБ", "ГБ", "ТБ"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"
    
    # Заголовок проекта
    project_header = html.Div([
        dbc.Row([
            dbc.Col([
                html.H2(project.get("name", "Без названия"), className="mb-2"),
                html.P(project.get("description", "") or "Описание отсутствует", 
                       className="text-muted mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Badge(
                            project.get("status_display", "Новый"), 
                            color=project.get("status_color", "secondary"),
                            className="me-2"
                        ),
                        html.Small(f"Создан: {format_date(project.get('created_at', ''))}", 
                                  className="text-muted me-3"),
                        html.Small(f"Обновлён: {format_date(project.get('updated_at', ''))}", 
                                  className="text-muted"),
                    ], width=8),
                    dbc.Col([
                        html.Div([
                            html.Span(f"Файлов: {len(project.get('files', []))}", 
                                     className="text-muted me-3"),
                            html.Span(f"Размер: {format_file_size(project.get('total_file_size', 0))}", 
                                     className="text-muted"),
                        ], className="text-end"),
                    ], width=4),
                ]),
            ]),
        ]),
    ], className="mb-4")
    
    # Вкладки проекта
    tabs = dbc.Tabs([
        # Вкладка "Обзор"
        dbc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Информация о проекте"),
                            dbc.CardBody([
                                html.Div([
                                    html.Strong("ID проекта:"),
                                    html.P(project.get("id", "-"), className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Статус:"),
                                    html.P(project.get("status_display", "-"), className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Дата создания:"),
                                    html.P(format_date(project.get("created_at", "-")), 
                                           className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Последнее обновление:"),
                                    html.P(format_date(project.get("updated_at", "-")), 
                                           className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Количество файлов:"),
                                    html.P(str(len(project.get("files", []))), className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Общий размер:"),
                                    html.P(format_file_size(project.get("total_file_size", 0)), 
                                           className="text-muted"),
                                ]),
                            ])
                        ], className="mb-4"),
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Статистика обработки"),
                            dbc.CardBody([
                                html.Div([
                                    html.Strong("Запусков обработки:"),
                                    html.P(str(len(project.get("processing_history", []))), 
                                           className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Текущий этап:"),
                                    html.P(project.get("current_stage", "Не запущено"), 
                                           className="text-muted"),
                                ], className="mb-3"),
                                html.Div([
                                    html.Strong("Прогресс:"),
                                    dbc.Progress(
                                        value=project.get("progress", 0),
                                        max=100,
                                        label=f"{project.get('progress', 0)}%",
                                        className="mb-2"
                                    ),
                                ]),
                            ])
                        ]),
                    ], width=6),
                ]),
            ])
        ], label="Обзор", tab_id="overview-tab"),
        
        # Вкладка "Файлы"
        dbc.Tab([
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Файлы проекта", className="mb-0"),
                        dbc.Button(
                            [html.I(className="fas fa-plus me-2"), "Добавить файлы"],
                            id="add-files-btn",
                            color="primary",
                            size="sm"
                        ),
                    ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody([
                        dbc.ListGroup([
                            *[
                                dbc.ListGroupItem([
                                    html.Div([
                                        html.Div([
                                            html.H6(file.get("filename", "Без названия"), 
                                                   className="mb-1"),
                                            html.P(f"Тип: {file.get('file_type', 'unknown')} | "
                                                   f"Размер: {format_file_size(file.get('file_size', 0))} | "
                                                   f"Загружен: {format_date(file.get('upload_date', ''))}", 
                                                   className="mb-1 text-muted small"),
                                        ], className="flex-grow-1"),
                                        html.Div([
                                            dbc.Button(
                                                html.I(className="fas fa-trash"),
                                                id={"type": "project-file-delete", 
                                                     "index": file.get("id", "")},
                                                color="outline-danger",
                                                size="sm",
                                                className="me-2"
                                            ),
                                            dbc.Button(
                                                html.I(className="fas fa-download"),
                                                id={"type": "project-file-download", 
                                                     "index": file.get("id", "")},
                                                color="outline-primary",
                                                size="sm"
                                            ),
                                        ], className="ms-3"),
                                    ], className="d-flex align-items-center")
                                ])
                                for file in project.get("files", [])
                            ]
                        ], flush=True),
                        
                        # Область загрузки файлов
                        dcc.Upload(
                            id='project-file-upload',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                                html.P("Перетащите файлы сюда или нажмите для выбора"),
                                html.P("Поддерживаемые форматы: BIL/HDR, TIFF, DAT", 
                                       className="text-muted small")
                            ]),
                            multiple=True,
                            className="upload-area p-4 border border-dashed rounded text-center mt-3"
                        ),
                    ])
                ]),
            ])
        ], label="Файлы", tab_id="files-tab"),
        
        # Вкладка "Обработка"
        dbc.Tab([
            html.Div([
                dbc.Card([
                    dbc.CardHeader("Конфигурация обработки"),
                    dbc.CardBody([
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Этапы обработки", className="mb-3"),
                                    dbc.Checklist(
                                        id="stage-checkboxes",
                                        options=[
                                            {"label": "Предобработка", "value": "preprocessing"},
                                            {"label": "Ортофото", "value": "orthophoto"},
                                            {"label": "Сегментация", "value": "segmentation"},
                                            {"label": "Расчёт индексов", "value": "indices"},
                                            {"label": "Оценка", "value": "assessment"},
                                            {"label": "Анализ", "value": "analysis"},
                                        ],
                                        value=project.get("processing_config", {}).get("stages", []),
                                        inline=False,
                                        className="mb-3"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.H6("Настройки", className="mb-3"),
                                    html.Div([
                                        dbc.Label("Вегетационные индексы", 
                                                  html_for="indices-select"),
                                        dcc.Dropdown(
                                            id="indices-select",
                                            options=[
                                                {"label": "NDVI", "value": "NDVI"},
                                                {"label": "EVI", "value": "EVI"},
                                                {"label": "SAVI", "value": "SAVI"},
                                                {"label": "GNDVI", "value": "GNDVI"},
                                                {"label": "MCARI", "value": "MCARI"},
                                                {"label": "RENDVI", "value": "RENDVI"},
                                            ],
                                            value=project.get("processing_config", {})
                                                .get("indices", {})
                                                .get("selected_indices", ["NDVI", "EVI"]),
                                            multi=True,
                                            className="mb-3"
                                        ),
                                        dbc.Checklist(
                                            id="processing-options",
                                            options=[
                                                {"label": "Применить коррекцию атмосферы", 
                                                 "value": "atmospheric_correction"},
                                                {"label": "Удалить шум", "value": "denoising"},
                                                {"label": "Сегментация растений", 
                                                 "value": "segmentation"},
                                            ],
                                            value=["atmospheric_correction", "denoising"],
                                        ),
                                    ]),
                                ], width=6),
                            ]),
                        ]),
                        
                        # Прогресс обработки
                        html.Div(id="processing-progress-section", children=[
                            html.Hr(className="my-4"),
                            html.H6("Прогресс обработки", className="mb-3"),
                            dbc.Progress(
                                id="project-processing-progress",
                                value=project.get("progress", 0),
                                max=100,
                                label=f"{project.get('progress', 0)}%",
                                className="mb-3"
                            ),
                            html.Div([
                                dbc.Button(
                                    [html.I(className="fas fa-play me-2"), "Запустить обработку"],
                                    id="project-start-processing-btn",
                                    color="primary",
                                    className="me-2"
                                ),
                                dbc.Button(
                                    [html.I(className="fas fa-stop me-2"), "Остановить"],
                                    id="project-cancel-processing-btn",
                                    color="secondary",
                                    disabled=True
                                ),
                            ], className="text-center"),
                        ]),
                    ])
                ]),
            ])
        ], label="Обработка", tab_id="processing-tab"),
        
        # Вкладка "Результаты"
        dbc.Tab([
            html.Div([
                dbc.Card([
                    dbc.CardHeader("История обработки"),
                    dbc.CardBody([
                        dbc.ListGroup([
                            *[
                                dbc.ListGroupItem([
                                    html.Div([
                                        html.Div([
                                            html.H6(f"Запуск {i+1}", className="mb-1"),
                                            html.P(f"Начало: {format_date(run.get('start_time', ''))} | "
                                                   f"Статус: {run.get('status', 'unknown')}", 
                                                   className="mb-1 text-muted small"),
                                            html.Div([
                                                dbc.Badge(
                                                    "Завершено" if run.get("status") == "completed" 
                                                    else "В процессе" if run.get("status") == "running"
                                                    else "Ошибка" if run.get("status") == "error"
                                                    else "Отменено",
                                                    color="success" if run.get("status") == "completed"
                                                    else "warning" if run.get("status") == "running"
                                                    else "danger" if run.get("status") == "error"
                                                    else "secondary",
                                                    className="me-2"
                                                ),
                                                html.Small(f"Длительность: {run.get('total_duration_seconds', 0):.1f} сек", 
                                                          className="text-muted"),
                                            ]),
                                        ], className="flex-grow-1"),
                                        html.Div([
                                            dbc.Button(
                                                "Просмотреть",
                                                id={"type": "view-run-results", 
                                                     "index": run.get("run_id", "")},
                                                color="outline-primary",
                                                size="sm"
                                            ),
                                        ], className="ms-3"),
                                    ], className="d-flex align-items-center")
                                ])
                                for i, run in enumerate(project.get("processing_history", []))
                            ]
                        ], flush=True),
                        
                        html.Div(id="run-results-details", className="mt-4"),
                    ])
                ]),
            ])
        ], label="Результаты", tab_id="results-tab"),
    ], id="project-detail-tabs", active_tab="overview-tab")
    
    return html.Div([
        project_header,
        tabs
    ], className="project-detail")