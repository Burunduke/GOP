"""
Callbacks для GUI приложения GOP
"""

import json
import base64
import io
from datetime import datetime
from dash import Input, Output, State, callback_context, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from .data_upload import create_file_list_item, format_filesize
from .visualization import create_index_map_figure, create_histogram_figure, create_empty_figure
from .dashboard import create_dashboard


def register_callbacks(app):
    """Регистрация всех колбэков приложения"""
    
    # Навигация
    @app.callback(
        Output('main-content', 'children'),
        [Input('nav-projects', 'n_clicks'),
         Input('nav-upload', 'n_clicks'),
         Input('nav-processing', 'n_clicks'),
         Input('nav-analysis', 'n_clicks'),
         Input('nav-brand', 'n_clicks'),
         Input('sidebar-dashboard', 'n_clicks')],
        prevent_initial_call=True
    )
    def navigate_to_page(projects_clicks, upload_clicks, processing_clicks, analysis_clicks, brand_clicks, dashboard_clicks):
        """Обработка навигации между страницами"""
        ctx = callback_context
        if not ctx.triggered:
            return html.Div("Выберите раздел меню")
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'nav-projects':
            return create_projects_page()
        elif button_id == 'nav-upload':
            from .data_upload import create_data_upload_component
            return create_data_upload_component()
        elif button_id == 'nav-processing':
            return create_processing_page()
        elif button_id == 'nav-analysis':
            from .visualization import create_visualization_component
            return create_visualization_component()
        elif button_id == 'nav-brand' or button_id == 'sidebar-dashboard':
            return create_dashboard()
        
        return html.Div("Страница в разработке")
    
    # Модальные окна
    @app.callback(
        Output('create-project-modal', 'is_open'),
        [Input('new-project-btn', 'n_clicks'),
         Input('quick-new-project-btn', 'n_clicks'),
         Input('create-project-btn', 'n_clicks'),
         Input('cancel-create-project', 'n_clicks')],
        [State('create-project-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_create_project_modal(new_btn, quick_btn, create_btn, cancel_btn, is_open):
        """Управление модальным окном создания проекта"""
        if new_btn or quick_btn:
            return True
        elif create_btn or cancel_btn:
            return False
        return is_open
    
    @app.callback(
        Output('upload-files-modal', 'is_open'),
        [Input('upload-files-btn', 'n_clicks'),
         Input('quick-upload-btn', 'n_clicks'),
         Input('upload-files-modal-btn', 'n_clicks'),
         Input('cancel-upload', 'n_clicks')],
        [State('upload-files-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_upload_files_modal(sidebar_btn, quick_btn, modal_btn, cancel_btn, is_open):
        """Управление модальным окном загрузки файлов"""
        if sidebar_btn or quick_btn:
            return True
        elif upload_btn or cancel_btn:
            return False
        return is_open
    
    @app.callback(
        Output('processing-settings-modal', 'is_open'),
        [Input('processing-settings-btn', 'n_clicks'),
         Input('quick-analysis-btn', 'n_clicks'),
         Input('start-processing-btn', 'n_clicks'),
         Input('cancel-processing', 'n_clicks')],
        [State('processing-settings-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_processing_settings_modal(sidebar_btn, quick_btn, start_btn, cancel_btn, is_open):
        """Управление модальным окном настроек обработки"""
        if sidebar_btn or quick_btn:
            return True
        elif start_btn or cancel_btn:
            return False
        return is_open
    
    # Загрузка файлов
    @app.callback(
        [Output('uploaded-files-list', 'children'),
         Output('file-info', 'children'),
         Output('start-processing-from-upload', 'disabled')],
        [Input('file-upload', 'contents')],
        [State('file-upload', 'filename'),
         State('file-upload', 'last_modified')],
        prevent_initial_call=True
    )
    def handle_file_upload(contents, filenames, last_modified):
        """Обработка загрузки файлов"""
        if not contents:
            return html.P("Файлы еще не загружены", className="text-muted text-center py-3"), "", True
        
        file_items = []
        total_size = 0
        
        for i, (content, filename, modified) in enumerate(zip(contents, filenames, last_modified)):
            # Расчет размера файла (приблизительно)
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            file_size = len(decoded)
            total_size += file_size
            
            # Форматирование времени загрузки
            upload_time = datetime.fromtimestamp(modified / 1000).strftime('%H:%M:%S')
            
            file_items.append(
                create_file_list_item(filename, file_size, upload_time)
            )
        
        file_info = dbc.Alert([
            html.H6("Информация о загруженных файлах:", className="alert-heading"),
            html.P(f"Всего файлов: {len(filenames)}"),
            html.P(f"Общий размер: {format_filesize(total_size)}"),
        ], color="success")
        
        return file_items, file_info, False
    
    # Визуализация
    @app.callback(
        Output('main-visualization', 'figure'),
        [Input('visualization-type', 'value'),
         Input('index-selector', 'value'),
         Input('colormap-selector', 'value')],
        prevent_initial_call=True
    )
    def update_visualization(viz_type, index_name, colormap):
        """Обновление визуализации"""
        if viz_type == 'index_map':
            return create_index_map_figure(None, index_name, colormap)
        elif viz_type == 'histogram':
            return create_histogram_figure(None, index_name)
        else:
            return create_empty_figure()
    
    # Уведомления
    @app.callback(
        Output('notification-toast', 'is_open'),
        [Input('create-project-btn', 'n_clicks')],
        [State('project-name-input', 'value'),
         State('notification-toast', 'is_open')],
        prevent_initial_call=True
    )
    def show_notification(create_click, project_name, is_open):
        """Показ уведомлений"""
        if create_click and project_name:
            return True
        return False
    
    # Прогресс обработки
    @app.callback(
        Output('progress-interval', 'disabled'),
        [Input('start-processing-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_progress_interval(start_click):
        """Включение/выключение интервала прогресса"""
        if start_click:
            return False  # Включить интервал
        return True  # Выключить интервал


def create_projects_page():
    """Создание страницы проектов"""
    return html.Div([
        html.H2("Проекты", className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Управление проектами"),
                        html.P("Здесь будет отображаться список ваших проектов"),
                        dbc.Button("Создать новый проект", color="primary", className="mt-3")
                    ])
                ])
            ])
        ])
    ])


def create_processing_page():
    """Создание страницы обработки"""
    return html.Div([
        html.H2("Обработка данных", className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Настройки обработки"),
                        html.P("Здесь будут настройки для обработки гиперспектральных данных"),
                        dbc.Button("Начать обработку", color="success", className="mt-3")
                    ])
                ])
            ])
        ])
    ])