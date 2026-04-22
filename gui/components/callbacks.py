"""
Callbacks for GOP GUI application with project management integration.

This module contains all Dash callbacks that handle user interactions and state management
for the GOP GUI application, including project management, file uploads, and processing.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from gui.utils.format_utils import get_stage_display_name

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

def register_callbacks(
    app: dash.Dash,
    project_manager: Optional[Any] = None,
    pipeline_executor: Optional[Any] = None
) -> None:
    """
    Register all application callbacks.
    
    Args:
        app: Dash application instance
        project_manager: Project manager service (optional)
        pipeline_executor: Pipeline executor service (optional)
    """
    
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
        
        # Show documentation for docs/api since we now have documentation integrated in dashboard
        if pathname == "/docs/api":
            return create_documentation_component('api')
        
        if pathname == "/" or pathname == "/dashboard":
            if project_manager:
                stats = project_manager.get_statistics()
                all_projects = project_manager.list_projects()
                all_projects_dicts = [p.to_dict() for p in all_projects]
                return create_dashboard(statistics=stats, all_projects=all_projects_dicts)
            return create_dashboard()
        
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
        
        # Default - always show project management panel
        if project_manager:
            stats = project_manager.get_statistics()
            all_projects = project_manager.list_projects()
            all_projects_dicts = [p.to_dict() for p in all_projects]
            return create_dashboard(statistics=stats, all_projects=all_projects_dicts)
        return create_dashboard()
    
    # === 2. URL redirect callback for unrecognized paths ===
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("url", "pathname"),
        prevent_initial_call=True
    )
    def redirect_unrecognized_paths(pathname):
        """Redirect unrecognized paths to dashboard."""
        if pathname is None:
            raise PreventUpdate
        
        # List of valid paths
        valid_paths = ["/", "/dashboard", "/docs/api", "/docs/user-guide", "/docs/faq"]
        
        # Check if path starts with /project/
        if pathname.startswith("/project/"):
            return dash.no_update
        
        # If path is not recognized, redirect to dashboard
        if pathname not in valid_paths:
            return "/dashboard"
        
        return dash.no_update
    
    # === 3. Sidebar update callback ===
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
    def toggle_create_project_modal(
        new_btn: Optional[int],
        create_btn: Optional[int],
        cancel_btn: Optional[int],
        is_open: bool,
        name: Optional[str],
        description: Optional[str]
    ) -> Union[bool, Any]:
        """Handle create project modal open/close and project creation."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Debug logging to help identify the issue
        logger.debug(f"Modal trigger: {trigger_id}, clicks: new={new_btn}, create={create_btn}, cancel={cancel_btn}")
        
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
    
    # === 7. Refresh projects store after creation ===
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
    
    # === 8. New project button click -> navigate to projects page ===
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("new-project-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def navigate_to_projects_page(n_clicks):
        """Navigate to dashboard when new project button is clicked."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        
        return "/"
    
    # === 9. Project item click -> navigate to project detail ===
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
    
    # === 10. Dashboard project button click -> navigate to project detail ===
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
    
    # === 11. Start processing callback ===
    @app.callback(
        Output("project-processing-progress", "style"),
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
            return {"display": "block"}
        
        raise PreventUpdate
    
    # === 12. Cancel processing callback ===
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
        [Output("overview-progress-bar", "value"),
         Output("overview-progress-bar", "label"),
         Output("project-processing-progress", "value"),
         Output("project-processing-progress", "label"),
         Output("current-stage-display", "children"),
         Output("current-stage-progress", "children"),
         Output("current-stage-badge", "children"),
         Output("current-stage-badge", "color"),
         Output("progress-interval", "disabled")],
        Input("progress-interval", "n_intervals"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def update_processing_progress(n_intervals, pathname):
        """Poll for processing progress updates and synchronize progress bars."""
        if not pathname or not pathname.startswith("/project/") or not project_manager:
            raise PreventUpdate
        
        project_id = pathname.split("/project/")[-1]
        project = project_manager.get_project(project_id)
        if project:
            progress_value = project.progress
            progress_label = f"{progress_value:.1f}%"
            
            # Get current stage information
            current_stage = get_stage_display_name(project.current_stage or "Not started")
            stage_progress = f" ({progress_value:.1f}%)" if project.current_stage else ""
            
            # Determine badge text and color
            if progress_value == 100 and project.current_stage:
                badge_text = "Done"
                badge_color = "success"
            elif project.current_stage and project.status == "run":
                badge_text = "Run"
                badge_color = "warning"
            elif not project.current_stage and project.status in ["ready", "new"]:
                badge_text = "Pending"
                badge_color = "secondary"
            elif project.status == "error":
                badge_text = "Error"
                badge_color = "danger"
            elif project.status == "cancelled":
                badge_text = "Cancelled"
                badge_color = "dark"
            else:
                badge_text = "Pending"
                badge_color = "secondary"
            
            # Check if processing is complete to disable interval
            from gui.models.project import ProjectStatus
            interval_disabled = project.status != ProjectStatus.RUN.value
            
            return (
                progress_value,      # overview progress value
                progress_label,      # overview progress label
                progress_value,      # processing progress value
                progress_label,      # processing progress label
                current_stage,       # current stage display
                stage_progress,      # current stage progress
                badge_text,          # badge text
                badge_color,         # badge color
                interval_disabled    # disable interval when processing is complete
            )
        
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
    
    # === 13. File browser: navigate directory / refresh ===
    @app.callback(
        Output("file-browser-contents", "children"),
        Output("file-browser-breadcrumb", "children"),
        Output("file-browser-current-path", "data"),
        Input("file-browser-refresh-btn", "n_clicks"),
        Input("file-browser-go-up-btn", "n_clicks"),
        Input("file-browser-go-root-btn", "n_clicks"),
        Input({"type": "file-browser-folder", "index": ALL}, "n_clicks"),
        State("file-browser-current-path", "data"),
        prevent_initial_call=True,
    )
    def navigate_file_browser(refresh_clicks, up_clicks, root_clicks,
                              folder_clicks, current_path):
        """Handle file browser navigation: refresh, go up, go root, enter folder."""
        from gui.components.server_file_picker import (
            list_directory, render_directory_contents, render_breadcrumb,
            get_default_browse_root,
        )

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]["prop_id"]
        target_path = current_path

        if "file-browser-go-root-btn" in trigger:
            target_path = get_default_browse_root()
        elif "file-browser-go-up-btn" in trigger:
            from pathlib import Path
            parent = str(Path(current_path).parent)
            target_path = parent
        elif "file-browser-folder" in trigger:
            # Extract folder path from pattern-matching trigger
            if any(c for c in (folder_clicks or []) if c):
                prop_id = trigger.split(".")[0]
                id_dict = json.loads(prop_id)
                target_path = id_dict["index"]
            else:
                raise PreventUpdate
        elif "file-browser-refresh-btn" in trigger:
            target_path = current_path
        else:
            raise PreventUpdate

        try:
            listing = list_directory(target_path)
            contents = render_directory_contents(listing)
            breadcrumb = render_breadcrumb(listing['path'], get_default_browse_root())
            return contents, breadcrumb, listing['path']
        except Exception as e:
            logger.error(f"Error browsing directory {target_path}: {e}")
            error_msg = dbc.Alert(
                f"Error browsing directory: {e}",
                color="danger", className="py-2 small",
            )
            return error_msg, html.Span(str(target_path)), current_path

    # === 14. File browser: enable/disable add button ===
    @app.callback(
        Output("add-server-files-btn", "disabled"),
        Input("server-files-checklist", "value"),
        prevent_initial_call=True,
    )
    def toggle_add_server_files_btn(selected_files):
        """Enable the 'Add' button when at least one file is selected."""
        if selected_files and len(selected_files) > 0:
            return False
        return True

    # === 15. File browser: add selected files to project ===
    @app.callback(
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Input("add-server-files-btn", "n_clicks"),
        State("server-files-checklist", "value"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def add_server_files_to_project(n_clicks, selected_paths, pathname):
        """
        Add server-side files to the current project using filesystem copy.

        This is the key fix for the OOM issue: instead of reading file contents
        into memory (as dcc.Upload does via base64), we use shutil.copy2 which
        copies the file at the OS level with constant memory usage.
        """
        if not n_clicks or not selected_paths or not pathname:
            raise PreventUpdate

        if not pathname.startswith("/project/"):
            raise PreventUpdate

        project_id = pathname.split("/project/")[-1]

        if not project_manager:
            raise PreventUpdate

        added_count = 0
        errors = []

        for file_path in selected_paths:
            try:
                project_manager.add_file_by_server_path(
                    project_id=project_id,
                    source_path=file_path,
                    copy=True,  # Copy, don't move — keep original in uploads
                )
                added_count += 1
            except FileNotFoundError as e:
                errors.append(f"File not found: {file_path}")
                logger.error(f"Server file not found: {e}")
            except Exception as e:
                errors.append(f"Error adding {file_path}: {e}")
                logger.error(f"Error adding server file {file_path}: {e}")

        # Build notification message
        if errors:
            msg = f"Added {added_count} file(s). Errors: {'; '.join(errors)}"
        else:
            msg = f"Successfully added {added_count} file(s)"

        # Refresh project detail page
        project = project_manager.get_project(project_id)
        page_content = no_update
        if project:
            page_content = create_project_detail(project.to_dict())

        return True, msg, page_content

    # === Keep existing callbacks that still work ===
    
    # === 16. File deletion callback ===
    @app.callback(
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Input({"type": "project-file-delete", "index": ALL}, "n_clicks"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def delete_project_file(n_clicks, pathname):
        """Delete a file from the current project."""
        if not any(n_clicks) or not pathname or not pathname.startswith("/project/") or not project_manager:
            raise PreventUpdate
        
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Get the file ID from the triggered button
        trigger = ctx.triggered[0]
        prop_id = trigger["prop_id"]
        id_str = prop_id.split(".")[0]
        id_dict = json.loads(id_str)
        file_id = id_dict["index"]
        
        # Get project ID from URL
        project_id = pathname.split("/project/")[-1]
        
        # Delete the file
        success = project_manager.remove_file_from_project(project_id, file_id)
        
        if success:
            message = "File deleted successfully"
        else:
            message = "Error deleting file"
        
        # Refresh project detail page
        project = project_manager.get_project(project_id)
        page_content = no_update
        if project:
            page_content = create_project_detail(project.to_dict())
        
        return True, message, page_content


    # Modal windows
    
    # Modal windows
    @app.callback(
        Output('upload-files-modal', 'is_open'),
        [Input('upload-files-btn', 'n_clicks'),
         Input('upload-files-modal-btn', 'n_clicks'),
         Input('cancel-upload', 'n_clicks')],
        [State('upload-files-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_upload_files_modal(
        sidebar_btn: Optional[int],
        modal_btn: Optional[int],
        cancel_btn: Optional[int],
        is_open: bool
    ) -> bool:
        """Manage file upload modal window."""
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
    def toggle_processing_settings_modal(
        sidebar_btn: Optional[int],
        start_btn: Optional[int],
        cancel_btn: Optional[int],
        is_open: bool
    ) -> bool:
        """Manage processing settings modal window."""
        if sidebar_btn:
            return True
        elif start_btn or cancel_btn:
            return False
        return is_open
    
        # Processing progress
        @app.callback(
        Output('progress-interval', 'disabled'),
        [Input('start-processing-btn', 'n_clicks'),
         Input('project-start-processing-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_progress_interval(modal_start_click: Optional[int],
                                  project_start_click: Optional[int]) -> bool:
        """Enable/disable progress interval."""
        if modal_start_click or project_start_click:
            return False  # Enable interval
        return True  # Disable interval

    # === Dynamic results history updates ===
    @app.callback(
        Output("processing-history-list", "children"),
        Input("progress-interval", "n_intervals"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def update_results_history(n_intervals, pathname):
        """Update results history dynamically without page refresh."""
        if not pathname or not pathname.startswith("/project/") or not project_manager:
            raise PreventUpdate
        
        project_id = pathname.split("/project/")[-1]
        project = project_manager.get_project(project_id)
        if project:
            # Format date helper function
            def format_date(date_str: str) -> str:
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    return dt.strftime("%Y-%m-%d %H:%M")
                except:
                    return date_str
            
            # Create history items
            history_items = []
            for i, run in enumerate(project.processing_history):
                item = dbc.ListGroupItem([
                    html.Div([
                        html.Div([
                            html.H6(f"Run {i+1}", className="mb-1"),
                            html.P(f"Start: {format_date(run.get('start_time', ''))} | "
                                   f"Status: {run.get('status', 'unknown')}",
                                   className="mb-1 text-muted small"),
                            html.Div([
                                dbc.Badge(
                                    "Completed" if run.get("status") == "completed"
                                    else "Running" if run.get("status") == "running"
                                    else "Error" if run.get("status") == "error"
                                    else "Cancelled",
                                    color="success" if run.get("status") == "completed"
                                    else "warning" if run.get("status") == "running"
                                    else "danger" if run.get("status") == "error"
                                    else "secondary",
                                    className="me-2"
                                ),
                                html.Small(f"Duration: {run.get('total_duration_seconds', 0):.1f} sec",
                                          className="text-muted"),
                            ]),
                        ], className="flex-grow-1"),
                        html.Div([
                            dbc.Button(
                                "View",
                                id={"type": "view-run-results",
                                     "index": run.get("run_id", "")},
                                color="outline-primary",
                                size="sm"
                            ),
                        ], className="ms-3"),
                    ], className="d-flex align-items-center")
                ])
                history_items.append(item)
            
            return history_items
        
        raise PreventUpdate
