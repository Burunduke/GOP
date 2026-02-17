"""
Утилиты для визуализации результатов
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
from .image_utils import apply_colormap, normalize_image


def visualize_indices(indices_dict, output_path=None, figsize=(15, 10)):
    """
    Визуализация вегетационных индексов
    
    Args:
        indices_dict (dict): Словарь с индексами {имя: массив}
        output_path (str, optional): Путь для сохранения изображения
        figsize (tuple): Размер фигуры
        
    Returns:
        matplotlib.figure.Figure: Объект фигуры
    """
    n_indices = len(indices_dict)
    if n_indices == 0:
        return None
    
    # Определение сетки
    cols = min(3, n_indices)
    rows = (n_indices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_indices == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (index_name, index_data) in enumerate(indices_dict.items()):
        row = i // cols
        col = i % cols
        
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # Нормализация данных
        normalized = normalize_image(index_data, method='minmax')
        
        # Визуализация
        im = ax.imshow(normalized, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f'{index_name}', fontsize=12)
        ax.axis('off')
        
        # Добавление цветовой шкалы
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Скрытие лишних осей
    for i in range(n_indices, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            fig.delaxes(axes[col])
        else:
            fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comparison_plot(original_image, segmentation_mask, indices_dict, 
                          output_path=None, figsize=(20, 12)):
    """
    Создание сравнительного графика с исходным изображением, маской и индексами
    
    Args:
        original_image (numpy.ndarray): Исходное изображение
        segmentation_mask (numpy.ndarray): Маска сегментации
        indices_dict (dict): Словарь с индексами
        output_path (str, optional): Путь для сохранения
        figsize (tuple): Размер фигуры
        
    Returns:
        matplotlib.figure.Figure: Объект фигуры
    """
    # Выбор до 3 лучших индексов для отображения
    selected_indices = dict(list(indices_dict.items())[:3])
    
    n_plots = 2 + len(selected_indices)  # оригинал + маска + индексы
    cols = min(4, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    
    # Исходное изображение
    row = plot_idx // cols
    col = plot_idx % cols
    if rows == 1:
        ax = axes[col]
    else:
        ax = axes[row, col]
    
    if len(original_image.shape) == 3:
        ax.imshow(original_image)
    else:
        ax.imshow(original_image, cmap='gray')
    ax.set_title('Исходное изображение', fontsize=12)
    ax.axis('off')
    plot_idx += 1
    
    # Маска сегментации
    row = plot_idx // cols
    col = plot_idx % cols
    if rows == 1:
        ax = axes[col]
    else:
        ax = axes[row, col]
    
    ax.imshow(segmentation_mask, cmap='tab20')
    ax.set_title('Маска сегментации', fontsize=12)
    ax.axis('off')
    plot_idx += 1
    
    # Вегетационные индексы
    for index_name, index_data in selected_indices.items():
        row = plot_idx // cols
        col = plot_idx % cols
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        normalized = normalize_image(index_data, method='minmax')
        im = ax.imshow(normalized, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f'{index_name}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plot_idx += 1
    
    # Скрытие лишних осей
    for i in range(plot_idx, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            fig.delaxes(axes[col])
        else:
            fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_index_histogram(index_data, index_name, output_path=None, bins=50):
    """
    Построение гистограммы значений индекса
    
    Args:
        index_data (numpy.ndarray): Данные индекса
        index_name (str): Название индекса
        output_path (str, optional): Путь для сохранения
        bins (int): Количество бинов
        
    Returns:
        matplotlib.figure.Figure: Объект фигуры
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Удаление NaN значений
    valid_data = index_data[~np.isnan(index_data)]
    
    ax.hist(valid_data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(f'Распределение значений индекса {index_name}', fontsize=14)
    ax.set_xlabel('Значение индекса', fontsize=12)
    ax.set_ylabel('Частота', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Добавление статистики
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_val:.3f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±σ: {std_val:.3f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
    
    ax.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_plant_condition_chart(plant_condition_data, output_path=None):
    """
    Создание диаграммы состояния растений
    
    Args:
        plant_condition_data (dict): Данные о состоянии растений
        output_path (str, optional): Путь для сохранения
        
    Returns:
        matplotlib.figure.Figure: Объект фигуры
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Классификация состояния
    classification = plant_condition_data.get('classification', {})
    class_name = classification.get('class', 'Неизвестно')
    confidence = classification.get('score', 0)
    
    # Круговая диаграмма классификации
    labels = [class_name, 'Другое']
    sizes = [confidence, 1 - confidence]
    colors = ['#2ecc71', '#ecf0f1']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Классификация состояния растений', fontsize=14)
    
    # Столбчатая диаграмма индексов
    indices = plant_condition_data.get('indices', {})
    if indices:
        index_names = list(indices.keys())
        index_values = list(indices.values())
        
        bars = ax2.bar(index_names, index_values, color='skyblue', alpha=0.7)
        ax2.set_title('Нормализованные значения индексов', fontsize=14)
        ax2.set_ylabel('Нормализованное значение', fontsize=12)
        ax2.set_ylim(0, 1)
        
        # Добавление значений на столбцы
        for bar, value in zip(bars, index_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Поворот меток
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'Нет данных об индексах', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Нормализованные значения индексов', fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_processing_workflow_chart(workflow_steps, output_path=None):
    """
    Создание диаграммы рабочего процесса обработки
    
    Args:
        workflow_steps (list): Список шагов обработки
        output_path (str, optional): Путь для сохранения
        
    Returns:
        matplotlib.figure.Figure: Объект фигуры
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Создание горизонтальной диаграммы процесса
    y_pos = np.arange(len(workflow_steps))
    
    bars = ax.barh(y_pos, [1] * len(workflow_steps), color='lightblue', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(workflow_steps)
    ax.invert_yaxis()  # Сверху вниз
    ax.set_xlabel('Прогресс', fontsize=12)
    ax.set_title('Рабочий процесс обработки', fontsize=14)
    
    # Удаление оси X
    ax.set_xticks([])
    
    # Добавление номеров шагов
    for i, (bar, step) in enumerate(zip(bars, workflow_steps)):
        width = bar.get_width()
        ax.text(width / 2, bar.get_y() + bar.get_height() / 2,
                f'{i+1}', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig