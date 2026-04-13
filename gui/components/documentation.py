"""
Компонент для отображения документации в формате Markdown
"""

import os
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_documentation_component(doc_type="user_guide"):
    """Создание компонента для отображения документации"""
    
    # Определение пути к файлу документации
    doc_paths = {
        "user_guide": "docs/USER_GUIDE.md",
        "faq": "docs/FAQ.md",
        "api": "docs/api/_build/html/index.html"
    }
    
    file_path = doc_paths.get(doc_type)
    
    if not file_path or not os.path.exists(file_path):
        return html.Div([
            html.H3("Документация не найдена"),
            html.P(f"Файл {file_path} не существует.")
        ], className="p-4")
    
    # Для HTML документации API
    if doc_type == "api":
        return html.Div([
            html.H3("Документация API", className="mb-4"),
            html.Iframe(
                src="/docs/api/_build/html/index.html",
                style={
                    "width": "100%",
                    "height": "800px",
                    "border": "none"
                }
            )
        ], className="p-4")
    
    # Для Markdown документации
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Определение заголовка
        titles = {
            "user_guide": "Руководство пользователя",
            "faq": "Часто задаваемые вопросы"
        }
        title = titles.get(doc_type, "Документация")
        
        return html.Div([
            html.H3(title, className="mb-4"),
            html.Div(
                html.Div(
                    dcc.Markdown(markdown_content),
                    className="card-body"
                ),
                className="card"
            )
        ], className="p-4")
        
    except Exception as e:
        return html.Div([
            html.H3("Ошибка загрузки документации"),
            html.P(f"Не удалось загрузить документацию: {str(e)}")
        ], className="p-4")


def create_documentation_layout():
    """Создание layout для страницы документации"""
    return html.Div([
        dcc.Location(id='doc-url', refresh=False),
        html.Div(id='doc-content')
    ])