"""
Callbacks для GUI приложения GOP с интеграцией управления проектами.
"""

import json
import logging
from datetime import datetime

from dash import Input, Output, State, callback_context, no_update, ALL, MATCH, html
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc

from gui.components.dashboard import create_dashboard
from gui.components.sidebar import create_sidebar
from gui.components.documentation import create_documentation_component
from gui.components.project_detail import create_project_detail
from gui.models.project import Project, ProjectStatus, PipelineStage

logger = logging.getLogger(__name__)


def register_callbacks(app, project_manager=None, pipeline_executor=None):
    """Register all application callbacks."""
    
    # === 1. Page routing callback ===
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
        State("current-project-store", "data"),
    )
    def display_page(pathname, current_project_data):
        """Route to appropriate page based on URL."""
        if pathname is None:
            raise PreventUpdate
        
        if pathname == "/" or pathname == "/dashboard":
            if project_manager:
                stats = project_manager.get_statistics()
                recent = project_manager.get_recent_projects(5)
                recent_dicts = [p.to_dict() for p in recent]
                return create_dashboard(statistics=stats, recent_projects=recent_dicts)
            return create_dashboard()
        
        elif pathname == "/docs/api":
            return create_documentation_component('api')
        elif pathname == "/docs/user-guide":
            return create_documentation_component('user_guide')
        elif pathname == "/docs/faq":
            return create_documentation_component('faq')
        
        elif pathname.startswith("/project/"):
            project_id = pathname.split("/project/")[-1]
            if project_manager:
                project = project_manager.get_project(project_id)
                if project:
                    return create_project_detail(project.to_dict())
            return create_project_detail(None)
        
        elif pathname == "/projects":
            return _create_projects_page()
        
        # Default
        if project_manager:
            stats = project_manager.get_statistics()
            recent = project_manager.get_recent_projects(5)
            recent_dicts = [p.to_dict() for p in recent]
            return create_dashboard(statistics=stats, recent_projects=recent_dicts)
        return create_dashboard()
    
    # === 2. Sidebar update callback ===
    @app.callback(
        Output("sidebar", "children"),
        Input("projects-store", "data"),
    )
    def update_sidebar(projects_data):
        """Update sidebar with current project data."""
        if project_manager:
            projects = project_manager.list_projects()
            projects_dicts = [p.to_dict() for p in projects]
            stats = project_manager.get_statistics()
            return create_sidebar(projects=projects_dicts, statistics=stats)
        return create_sidebar()
    
    # === 3. Navigation callback (keep existing) ===
    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        [Input('nav-dashboard', 'n_clicks'),
         Input('nav-projects', 'n_clicks'),
         Input('nav-brand', 'n_clicks'),
         Input('nav-api-docs', 'n_clicks'),
         Input('nav-user-guide', 'n_clicks'),
         Input('nav-faq', 'n_clicks')],
        prevent_initial_call=True
    )
    def navigate_to_page(dashboard_clicks, projects_clicks, brand_clicks, api_docs_clicks, user_guide_clicks, faq_clicks):
        """Обработка навигации между страницами"""
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        
        # Check if this is an initial call (all clicks are None)
        all_clicks = [dashboard_clicks, projects_clicks, brand_clicks, api_docs_clicks, user_guide_clicks, faq_clicks]
        
        if all(click is None for click in all_clicks):
            return dash.no_update
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Filter out non-navigation button clicks (like project items)
        valid_nav_buttons = ['nav-dashboard', 'nav-projects', 'nav-brand', 'nav-api-docs', 'nav-user-guide', 'nav-faq']
        
        if button_id not in valid_nav_buttons:
            return dash.no_update
        
        # DEBUG: Log which navigation button was clicked
        print(f"[DEBUG] Navigation button clicked: {button_id}")
        
        if button_id == 'nav-dashboard' or button_id == 'nav-brand':
            return '/'
        elif button_id == 'nav-projects':
            return '/projects'
        elif button_id == 'nav-api-docs':
            return '/docs/api'
        elif button_id == 'nav-user-guide':
            return '/docs/user-guide'
        elif button_id == 'nav-faq':
            return '/docs/faq'
        else:
            return '/'
    
    # === 4. Active nav highlighting (keep existing) ===
    @app.callback(
        [Output('nav-dashboard', 'active'),
         Output('nav-projects', 'active')],
        [Input('nav-dashboard', 'n_clicks'),
         Input('nav-projects', 'n_clicks'),
         Input('nav-brand', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_active_tab(dashboard_clicks, projects_clicks, brand_clicks):
        """Обновление подсветки активной вкладки"""
        ctx = callback_context
        if not ctx.triggered:
            return True, False  # Dashboard активен по умолчанию
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'nav-dashboard' or button_id == 'nav-brand':
            return True, False
        elif button_id == 'nav-projects':
            return False, True
        
        return True, False  # По умолчанию dashboard

    # === 5. Create project modal ===
    @app.callback(
        Output("create-project-modal", "is_open"),
        [Input("new-project-btn", "n_clicks"),
         Input("create-project-btn", "n_clicks"),
         Input("cancel-create-project", "n_clicks")],
        [State("create-project-modal", "is_open"),
         State("project-name-input", "value"),
         State("project-description-input", "value")],
        prevent_initial_call=True,
    )
    def toggle_create_project_modal(new_btn, create_btn, cancel_btn, is_open, name, description):
        """Handle create project modal open/close and project creation."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Debug logging to help identify the issue
        print(f"[DEBUG] Modal trigger: {trigger_id}, clicks: new={new_btn}, create={create_btn}, cancel={cancel_btn}")
        
        # Additional safety check: ensure we only respond to actual button clicks
        if trigger_id == "new-project-btn":
            # Only open if new-project-btn was actually clicked and we have a valid click count
            if new_btn is not None and new_btn > 0:
                return True
            return no_update
        
        if trigger_id == "create-project-btn":
            if name and project_manager:
                project_manager.create_project(name=name, description=description or "")
            return False

        if trigger_id == "cancel-create-project":
            return False
        
        # If we get here, it's an unexpected trigger - don't change modal state
        return no_update
        
        return not is_open
    
    # === 6. Refresh projects store after creation ===
    @app.callback(
        Output("projects-store", "data"),
        [Input("create-project-modal", "is_open"),
         Input("delete-project-modal", "is_open"),
         Input("url", "pathname")],
        prevent_initial_call=True,
    )
    def refresh_projects_store(create_modal_open, delete_modal_open, pathname):
        """Refresh projects store when modals close or navigation happens."""
        if project_manager:
            projects = project_manager.list_projects()
            return [p.to_dict() for p in projects]
        return []
    
    # === 7. New project button click -> navigate to projects page ===
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("new-project-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def navigate_to_projects_page(n_clicks):
        """Navigate to projects page when new project button is clicked."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        
        return "/projects"
    
    # === 8. Project item click -> navigate to project detail ===
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input({"type": "project-item", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def navigate_to_project(n_clicks):
        """Navigate to project detail when project item is clicked."""
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks):
            raise PreventUpdate
        
        trigger = ctx.triggered[0]
        # Extract project ID from the trigger
        prop_id = trigger["prop_id"]
        # prop_id format: '{"index":"project-id","type":"project-item"}.n_clicks'
        id_str = prop_id.split(".")[0]
        id_dict = json.loads(id_str)
        project_id = id_dict["index"]
        
        return f"/project/{project_id}"
    
    # === 8. Dashboard project button click -> navigate to project detail ===
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input({"type": "dashboard-project-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def navigate_to_project_from_dashboard(n_clicks):
        """Navigate to project detail when dashboard project button is clicked."""
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks):
            raise PreventUpdate
        
        trigger = ctx.triggered[0]
        prop_id = trigger["prop_id"]
        id_str = prop_id.split(".")[0]
        id_dict = json.loads(id_str)
        project_id = id_dict["index"]
        
        return f"/project/{project_id}"
    
    # === 9. Start processing callback ===
    @app.callback(
        [Output("project-processing-progress", "value"),
         Output("project-processing-progress", "label"),
         Output("project-processing-progress", "style")],
        Input("project-start-processing-btn", "n_clicks"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def start_project_processing(n_clicks, pathname):
        """Start processing for the current project."""
        if not n_clicks or not pathname or not pathname.startswith("/project/") or not pipeline_executor:
            raise PreventUpdate
        
        project_id = pathname.split("/project/")[-1]
        if project_id:
            pipeline_executor.execute_project(project_id)
            return 0, "Запуск...", {"display": "block"}
        
        raise PreventUpdate
    
    # === 10. Cancel processing callback ===
    @app.callback(
        Output("notification-toast", "is_open", allow_duplicate=True),
        Input("project-cancel-processing-btn", "n_clicks"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def cancel_project_processing(n_clicks, pathname):
        """Cancel processing for the current project."""
        if not n_clicks or not pathname or not pathname.startswith("/project/") or not pipeline_executor:
            raise PreventUpdate
        
        project_id = pathname.split("/project/")[-1]
        if project_id:
            pipeline_executor.cancel_project(project_id)
            return True
        
        raise PreventUpdate
    
    # === 11. Processing progress polling ===
    @app.callback(
        Output("page-content", "children", allow_duplicate=True),
        Input("progress-interval", "n_intervals"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def update_processing_progress(n_intervals, pathname):
        """Poll for processing progress updates."""
        if not pathname or not pathname.startswith("/project/") or not project_manager:
            raise PreventUpdate
        
        project_id = pathname.split("/project/")[-1]
        project = project_manager.get_project(project_id)
        if project and project.status == ProjectStatus.PROCESSING.value:
            return create_project_detail(project.to_dict())
        
        raise PreventUpdate
    
    # === 12. Delete project callbacks ===
    @app.callback(
        Output("delete-project-modal", "is_open"),
        [Input({"type": "project-delete-btn", "index": ALL}, "n_clicks"),
         Input("confirm-delete-project-btn", "n_clicks"),
         Input("cancel-delete-project-btn", "n_clicks")],
        State("delete-project-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_delete_modal(delete_clicks, confirm_click, cancel_click, is_open):
        """Handle delete project modal."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger = ctx.triggered[0]["prop_id"]
        
        if "confirm-delete-project-btn" in trigger:
            # Actually delete - handled in separate callback
            return False
        elif "cancel-delete-project-btn" in trigger:
            return False
        elif "project-delete-btn" in trigger and any(c for c in (delete_clicks or []) if c):
            return True
        
        return is_open
    
    # === 13. File upload to project ===
    @app.callback(
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Input("project-file-upload", "contents"),
        State("project-file-upload", "filename"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def handle_file_upload(contents, filenames, pathname):
        """Handle file upload to current project."""
        if not contents or not pathname or not pathname.startswith("/project/"):
            raise PreventUpdate
        
        project_id = pathname.split("/project/")[-1]
        
        if project_manager and filenames:
            import base64
            for content, filename in zip(
                contents if isinstance(contents, list) else [contents],
                filenames if isinstance(filenames, list) else [filenames]
            ):
                # Decode base64 content
                content_type, content_string = content.split(",")
                decoded = base64.b64decode(content_string)
                project_manager.add_file_to_project(project_id, filename, decoded)
            
            return True, f"Загружено файлов: {len(filenames) if isinstance(filenames, list) else 1}"
        
        raise PreventUpdate
    
    # === Keep existing callbacks that still work ===
    
    # Модальные окна
    @app.callback(
        Output('upload-files-modal', 'is_open'),
        [Input('upload-files-btn', 'n_clicks'),
         Input('upload-files-modal-btn', 'n_clicks'),
         Input('cancel-upload', 'n_clicks')],
        [State('upload-files-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_upload_files_modal(sidebar_btn, modal_btn, cancel_btn, is_open):
        """Управление модальным окном загрузки файлов"""
        if sidebar_btn:
            return True
        elif modal_btn or cancel_btn:
            return False
        return is_open
    
    @app.callback(
        Output('processing-settings-modal', 'is_open'),
        [Input('processing-settings-btn', 'n_clicks'),
         Input('start-processing-btn', 'n_clicks'),
         Input('cancel-processing', 'n_clicks')],
        [State('processing-settings-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_processing_settings_modal(sidebar_btn, start_btn, cancel_btn, is_open):
        """Управление модальным окном настроек обработки"""
        if sidebar_btn:
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
    def handle_file_upload_legacy(contents, filenames, last_modified):
        """Обработка загрузки файлов (legacy)"""
        if not contents:
            return "Файлы еще не загружены", "", True
        
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
                html.Div([
                    html.H6(filename, className="mb-1"),
                    html.P(f"Размер: {file_size} байт | Время: {upload_time}", 
                           className="text-muted small"),
                ], className="border-bottom pb-2 mb-2")
            )
        
        file_info = dbc.Alert([
            html.H6("Информация о загруженных файлах:", className="alert-heading"),
            html.P(f"Всего файлов: {len(filenames)}"),
            html.P(f"Общий размер: {total_size} байт"),
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
        from .visualization import create_index_map_figure, create_histogram_figure, create_empty_figure
        
        if viz_type == 'index_map':
            return create_index_map_figure(None, index_name, colormap)
        elif viz_type == 'histogram':
            return create_histogram_figure(None, index_name)
        else:
            return create_empty_figure()
    
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


def _create_projects_page():
    """Создание страницы проектов"""
    return html.Div([
        html.H2("Проекты", className="mb-4"),
        
        # Карточки проектов
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Анализ поля пшеницы", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        html.P("NDVI анализ для оценки состояния посевов", className="text-muted"),
                        html.Div([
                            dbc.Badge("Завершен", color="success", className="me-2"),
                            html.Span("15.01.2024", className="text-muted"),
                        ], className="mb-3"),
                        dbc.Button("Открыть проект", color="primary", size="sm")
                    ])
                ], className="h-100")
            ], width=6, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Тестирование индексов", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        html.P("Сравнительный анализ различных вегетационных индексов", className="text-muted"),
                        html.Div([
                            dbc.Badge("В обработке", color="warning", className="me-2"),
                            html.Span("16.01.2024", className="text-muted"),
                        ], className="mb-3"),
                        dbc.Button("Открыть проект", color="primary", size="sm")
                    ])
                ], className="h-100")
            ], width=6, className="mb-3"),
        ]),
        
        # Кнопка создания нового проекта
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-plus me-2"), "Создать новый проект"],
                    color="primary",
                    size="lg",
                    className="w-100"
                )
            ], width=6, className="mx-auto mt-4")
        ])
    ])

