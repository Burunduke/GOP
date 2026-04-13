"""
Компонент загрузки данных для GUI приложения GOP
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_data_upload_component():
    """Создание компонента загрузки данных"""
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H5("Загрузка гиперспектральных данных", className="mb-0"),
            ]),
            dbc.CardBody([
                # Информация о поддерживаемых форматах
                dbc.Alert([
                    html.H6("Поддерживаемые форматы:", className="alert-heading"),
                    html.Ul([
                        html.Li("BIL/HDR - стандартный формат гиперспектральных данных"),
                        html.Li("TIFF/TIFF - геопространственные изображения"),
                        html.Li("DAT - сырые данные спектрометра"),
                    ]),
                    html.H6("Источники данных:", className="alert-heading mt-3"),
                    html.Ul([
                        html.Li(html.A("NASA EarthData", href="https://search.earthdata.nasa.gov/", target="_blank")),
                        html.Li(html.A("GLIHT Data", href="https://glihtdata.gsfc.nasa.gov/", target="_blank")),
                        html.Li(html.A("Open Aerial Map", href="https://map.openaerialmap.org/", target="_blank")),
                        html.Li(html.A("AVIRIS Data", href="https://popo.jpl.nasa.gov/mmgis-aviris/", target="_blank")),
                    ])
                ], color="info", className="mb-4"),
                
                # Область загрузки
                html.Div([
                    dcc.Upload(
                        id='data-upload',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt fa-3x mb-3 text-primary"),
                            html.H4("Перетащите файлы сюда"),
                            html.P("или нажмите для выбора файлов", className="text-muted"),
                            html.P("Максимальный размер: 10GB", className="text-muted small"),
                        ]),
                        multiple=True,
                        className="upload-area p-5 border border-dashed rounded text-center",
                        style=upload_style()
                    ),
                ], className="mb-4"),
                
                # Список загруженных файлов
                html.Div([
                    html.H6("Загруженные файлы:", className="mb-3"),
                    html.Div(id='uploaded-files-list', children=[
                        html.P("Файлы еще не загружены", className="text-muted text-center py-3")
                    ]),
                ]),
                
                # Информация о файлах
                html.Div(id='file-info', className="mt-3"),
                
                # Кнопки действий
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-check me-2"), "Начать обработку"],
                        id="start-processing-from-upload",
                        color="success",
                        className="me-2",
                        disabled=True
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-trash me-2"), "Очистить"],
                        id="clear-uploaded-files",
                        color="outline-danger",
                        disabled=True
                    ),
                ], className="mt-4"),
            ])
        ])
    ], id="data-upload-container")


def upload_style():
    """Стили для виджета загрузки"""
    return {
        'borderWidth': '2px',
        'borderStyle': 'dashed',
        'borderRadius': '10px',
        'backgroundColor': '#f8f9fa',
        'cursor': 'pointer',
        'transition': 'all 0.3s ease'
    }


def create_file_list_item(filename, filesize, upload_time):
    """Создание элемента списка загруженных файлов"""
    return dbc.ListGroupItem([
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-file me-2 text-primary"),
                    html.Strong(filename),
                ], className="d-flex align-items-center mb-1"),
                html.Div([
                    html.Span(f"Размер: {format_filesize(filesize)}", className="text-muted me-3"),
                    html.Span(f"Загружен: {upload_time}", className="text-muted"),
                ], className="small"),
            ], className="flex-grow-1"),
            html.Div([
                dbc.Button(
                    html.I(className="fas fa-times"),
                    color="link",
                    size="sm",
                    className="text-danger p-0",
                    title="Удалить файл"
                )
            ])
        ], className="d-flex align-items-center justify-content-between")
    ], className="mb-2")


def format_filesize(size_bytes):
    """Форматирование размера файла"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"