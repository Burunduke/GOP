"""
Callbacks for GOP GUI application with project management integration.

This module contains all Dash callbacks that handle user interactions and state management
for the GOP GUI application, including project management, file uploads, and processing.
"""

import json
import logging
import base64
from datetime import datetime
from typing import Any, Optional, Union
from pathlib import Path

from gui.utils.format_utils import get_stage_display_name, format_date, format_file_size

from dash import Input, Output, State, callback_context, no_update, ALL, html
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc

from gui.components.dashboard import create_dashboard
from gui.components.sidebar import create_sidebar
from gui.components.documentation import create_documentation_component
from gui.components.project_detail import create_project_detail

logger = logging.getLogger(__name__)


def _get_run_display_name(run_data: dict) -> str:
    """
    Get display name for a run based on run_folder_name or fallback to index.
    
    Args:
        run_data: Run data dictionary
        
    Returns:
        Display name for the run
    """
    run_folder_name = run_data.get("run_folder_name")
    if run_folder_name and run_folder_name.startswith("run_"):
        try:
            run_number = int(run_folder_name.split("_")[1])
            return f"Run {run_number}"
        except (ValueError, IndexError):
            # Invalid format, fall back to default
            pass
    # Fallback to generic naming for legacy runs or if no folder name
    return "Run (Legacy)"

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
                all_projects_dicts = project_manager.list_projects_dicts()
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
            all_projects_dicts = project_manager.list_projects_dicts()
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
            projects_dicts = project_manager.list_projects_dicts()
            stats = project_manager.get_statistics()
            return create_sidebar(projects=projects_dicts, statistics=stats)
        return create_sidebar()
    

    
    # === 5. Create project modal ===
    @app.callback(
        [Output("create-project-modal", "is_open"),
         Output("project-creation-error-store", "data")],
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
                return True, None  # Clear any previous error when opening modal
            return no_update, dash.no_update
        
        if trigger_id == "create-project-btn":
            if name and project_manager:
                result = project_manager.create_project_safe(name=name, description=description or "")
                # Check if there was an error
                if "error" in result:
                    # Return the error to be displayed
                    return True, result["error"]  # Keep modal open and pass error
                else:
                    # Success - close modal and clear error
                    return False, None
            return False, dash.no_update

        if trigger_id == "cancel-create-project":
            return False, None
        
        # If we get here, it's an unexpected trigger - don't change modal state
        return no_update, dash.no_update
    
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
            return project_manager.list_projects_dicts()
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
            pipeline_executor.start_project_safe(project_id)
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
        if pipeline_executor and project_manager:
            status_dict = pipeline_executor.get_status_dict(project_id)
            if "error" not in status_dict and "status" in status_dict:
                progress_value = status_dict["progress"]
                progress_label = f"{progress_value:.1f}%"
                
                # Get current stage information
                current_stage = get_stage_display_name(status_dict["stage"] or "Not started")
                stage_progress = f" ({progress_value:.1f}%)" if status_dict["stage"] else ""
                
                # Determine badge text and color
                if progress_value == 100 and status_dict["stage"]:
                    badge_text = "Done"
                    badge_color = "success"
                elif status_dict["stage"] and status_dict["status"] == "run":
                    badge_text = "Run"
                    badge_color = "warning"
                elif not status_dict["stage"] and status_dict["status"] in ["ready", "new"]:
                    badge_text = "Pending"
                    badge_color = "secondary"
                elif status_dict["status"] == "error":
                    badge_text = "Error"
                    badge_color = "danger"
                elif status_dict["status"] == "cancelled":
                    badge_text = "Cancelled"
                    badge_color = "dark"
                else:
                    badge_text = "Pending"
                    badge_color = "secondary"
                
                # Check if processing is complete to disable interval
                from gui.models.project import ProjectStatus
                interval_disabled = status_dict["status"] != ProjectStatus.RUN.value
                
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
        elif project_manager:
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
    
    # === Display project creation errors ===
    @app.callback(
        [Output("notification-toast", "is_open", allow_duplicate=True),
         Output("notification-toast", "children", allow_duplicate=True)],
        Input("project-creation-error-store", "data"),
        prevent_initial_call=True,
    )
    def display_project_creation_error(error_message):
        """Display project creation error in notification toast."""
        if error_message:
            return True, error_message
        return dash.no_update, dash.no_update

    # === Display project creation errors in modal ===
    @app.callback(
        Output("project-creation-error-display", "children"),
        Input("project-creation-error-store", "data"),
    )
    def display_project_creation_error_in_modal(error_message):
        """Display project creation error in the create project modal."""
        if error_message:
            return dbc.Alert(error_message, color="danger", className="mb-0")
        return ""

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
                            html.H6(_get_run_display_name(run), className="mb-1"),
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
                                html.Small(f"Duration: {run.get('total_duration_seconds', 0) if run.get('total_duration_seconds') is not None else 0:.1f} sec",
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

    # === Enhanced file picker callbacks ===
    @app.callback(
        [Output('enhanced-file-picker-store', 'data'),
         Output('enhanced-file-picker-selection-summary', 'children'),
         Output('add-enhanced-files-btn', 'disabled')],
        Input('enhanced-file-picker-upload', 'contents'),
        [State('enhanced-file-picker-upload', 'filename'),
         State('enhanced-file-picker-upload', 'last_modified'),
         State('enhanced-file-picker-store', 'data')],
        prevent_initial_call=True,
    )
    def update_enhanced_file_picker(contents_list, filenames_list, dates_list, current_files):
        """
        Update the enhanced file picker with selected files.
        
        Args:
            contents_list: List of file contents (base64 encoded)
            filenames_list: List of file names
            dates_list: List of file modification dates
            current_files: Current list of selected files in store
            
        Returns:
            Updated file store data, summary component, and button disabled state
        """
        from gui.components.enhanced_file_picker import format_selected_files_summary, SUPPORTED_EXTENSIONS
        
        if contents_list is None or filenames_list is None:
            raise PreventUpdate
        
        # Create list of new files
        new_files = []
        for i, (content, filename) in enumerate(zip(contents_list, filenames_list)):
            # Check file extension
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
                
            # Calculate size from base64 content
            # Remove data URL prefix if present
            if content.startswith('data:'):
                content_data = content.split(',')[1]
            else:
                content_data = content
                
            # Approximate size (base64 is ~4/3 the size of binary data)
            try:
                size = len(base64.b64decode(content_data))
            except Exception:
                size = 0
                
            new_files.append({
                'name': filename,
                'content': content,
                'size': size,
                'last_modified': dates_list[i] if dates_list else None
            })
        
        # Combine with existing files (avoiding duplicates)
        updated_files = list(current_files)  # Start with existing files
        existing_names = {f['name'] for f in current_files}
        
        # Add new files that aren't already selected
        for new_file in new_files:
            if new_file['name'] not in existing_names:
                updated_files.append(new_file)
        
        # Create summary
        summary = format_selected_files_summary(updated_files)
        
        # Button is disabled if no files selected
        button_disabled = len(updated_files) == 0
        
        return updated_files, summary, button_disabled
    
    @app.callback(
        [Output("notification-toast", "is_open", allow_duplicate=True),
         Output("notification-toast", "children", allow_duplicate=True),
         Output("page-content", "children", allow_duplicate=True)],
        Input("add-enhanced-files-btn", "n_clicks"),
        [State('enhanced-file-picker-store', 'data'),
         State("url", "pathname")],
        prevent_initial_call=True,
    )
    def add_enhanced_files_to_project(n_clicks, selected_files, pathname):
        """
        Add files selected through the enhanced file picker to the current project.
        
        Args:
            n_clicks: Button click count
            selected_files: List of selected files with content
            pathname: Current URL pathname
            
        Returns:
            Notification state, message, and updated page content
        """
        from gui.components.project_detail import create_project_detail
        from dash import no_update
        
        if not n_clicks or not selected_files or not pathname:
            raise PreventUpdate

        if not pathname.startswith("/project/"):
            raise PreventUpdate

        project_id = pathname.split("/project/")[-1]

        if not project_manager:
            raise PreventUpdate

        added_count = 0
        errors = []

        # Process each selected file
        for file_info in selected_files:
            try:
                filename = file_info.get('name', 'unknown')
                content = file_info.get('content', '')
                
                # Decode base64 content
                if content.startswith('data:'):
                    content_data = content.split(',')[1]
                else:
                    content_data = content
                    
                file_bytes = base64.b64decode(content_data)
                
                # Add file to project
                project_manager.add_file_to_project(
                    project_id=project_id,
                    filename=filename,
                    file_content=file_bytes,
                    file_type=None  # Will be auto-detected
                )
                added_count += 1
            except Exception as e:
                errors.append(f"Error adding {file_info.get('name', 'Unknown')}: {str(e)}")
                logger.error(f"Error adding enhanced file {file_info.get('name', 'Unknown')}: {e}")

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

    @app.callback(
        Output('enhanced-file-picker-store', 'data', allow_duplicate=True),
        Input("add-enhanced-files-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_enhanced_file_picker_store(n_clicks):
        """Clear the enhanced file picker store after files are added."""
        if not n_clicks:
            raise PreventUpdate
        return []


# Register callbacks function end
