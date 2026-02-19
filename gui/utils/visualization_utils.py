"""
Утилиты визуализации для GUI приложения GOP
"""

import numpy as np
import plotly.graph_objs as go
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def create_colormap(name: str, n_colors: int = 256) -> List[Tuple[float, float, float]]:
    """
    Создание цветовой карты
    
    Args:
        name: Название цветовой схемы
        n_colors: Количество цветов
        
    Returns:
        Список RGB кортежей
    """
    try:
        if name == 'viridis':
            cmap = plt.cm.viridis
        elif name == 'plasma':
            cmap = plt.cm.plasma
        elif name == 'inferno':
            cmap = plt.cm.inferno
        elif name == 'magma':
            cmap = plt.cm.magma
        elif name == 'RdYlGn':
            cmap = plt.cm.RdYlGn
        elif name == 'RdYlBu':
            cmap = plt.cm.RdYlBu
        elif name == 'RdBu':
            cmap = plt.cm.RdBu
        elif name == 'coolwarm':
            cmap = plt.cm.coolwarm
        elif name == 'terrain':
            cmap = plt.cm.terrain
        else:
            cmap = plt.cm.viridis  # По умолчанию
        
        # Получение цветов и конвертация в RGB
        colors = []
        for i in range(n_colors):
            rgba = cmap(i / (n_colors - 1))
            rgb = (rgba[0], rgba[1], rgba[2])  # Удаление альфа-канала
            colors.append(rgb)
        
        return colors
        
    except Exception:
        # Возврат базовой цветовой схемы в случае ошибки
        return [(i/255, i/255, i/255) for i in range(n_colors)]


def apply_colormap(data: np.ndarray, colormap_name: str = 'viridis', 
                   vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """
    Применение цветовой карты к данным
    
    Args:
        data: Входные данные
        colormap_name: Название цветовой схемы
        vmin: Минимальное значение для нормализации
        vmax: Максимальное значение для нормализации
        
    Returns:
        RGB массив
    """
    try:
        # Нормализация данных
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        
        # Обработка случая с одинаковыми значениями
        if vmax == vmin:
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = (data - vmin) / (vmax - vmin)
        
        # Применение цветовой карты
        if colormap_name == 'viridis':
            cmap = plt.cm.viridis
        elif colormap_name == 'plasma':
            cmap = plt.cm.plasma
        elif colormap_name == 'RdYlGn':
            cmap = plt.cm.RdYlGn
        elif colormap_name == 'RdYlBu':
            cmap = plt.cm.RdYlBu
        else:
            cmap = plt.cm.viridis
        
        rgb_array = cmap(normalized_data)
        
        # Удаление альфа-канала
        return rgb_array[:, :, :3]
        
    except Exception as e:
        print(f"Ошибка применения цветовой карты: {e}")
        # Возврат градации серого в случае ошибки
        gray_data = np.zeros((data.shape[0], data.shape[1], 3))
        gray_data[:, :, 0] = gray_data[:, :, 1] = gray_data[:, :, 2] = data
        return gray_data


def create_heatmap_figure(data: np.ndarray, x_coords: Optional[np.ndarray] = None,
                         y_coords: Optional[np.ndarray] = None, 
                         colorscale: str = 'viridis',
                         title: str = "Тепловая карта") -> go.Figure:
    """
    Создание фигуры тепловой карты для Plotly
    
    Args:
        data: 2D массив данных
        x_coords: Координаты по оси X
        y_coords: Координаты по оси Y
        colorscale: Цветовая схема
        title: Заголовок
        
    Returns:
        Фигура Plotly
    """
    try:
        # Создание координат если не предоставлены
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])
        
        # Конвертация цветовой схемы matplotlib в plotly
        plotly_colorscales = {
            'viridis': 'Viridis',
            'plasma': 'Plasma',
            'inferno': 'Inferno',
            'magma': 'Magma',
            'RdYlGn': 'RdYlGn',
            'RdYlBu': 'RdYlBu',
            'RdBu': 'RdBu',
            'coolwarm': 'RdBu',
            'terrain': 'Earth'
        }
        
        plotly_colorscale = plotly_colorscales.get(colorscale, 'Viridis')
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_coords,
            y=y_coords,
            colorscale=plotly_colorscale,
            colorbar=dict(title="Значение"),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Значение: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Координата X",
            yaxis_title="Координата Y",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Ошибка создания тепловой карты: {e}")
        # Возврат пустой фигуры
        return go.Figure()


def create_histogram_figure(data: np.ndarray, bins: int = 50, 
                          title: str = "Гистограмма распределения",
                          x_label: str = "Значение",
                          y_label: str = "Частота") -> go.Figure:
    """
    Создание фигуры гистограммы для Plotly
    
    Args:
        data: 1D массив данных
        bins: Количество бинов
        title: Заголовок
        x_label: Подпись оси X
        y_label: Подпись оси Y
        
    Returns:
        Фигура Plotly
    """
    try:
        # Удаление NaN значений
        clean_data = data[~np.isnan(data)]
        
        fig = go.Figure(data=go.Histogram(
            x=clean_data,
            nbinsx=bins,
            marker_color='rgba(55, 128, 191, 0.7)',
            hovertemplate='Диапазон: %{x}<br>Количество: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Ошибка создания гистограммы: {e}")
        return go.Figure()


def create_scatter_figure(x_data: np.ndarray, y_data: np.ndarray,
                         title: str = "Диаграмма рассеяния",
                         x_label: str = "X",
                         y_label: str = "Y") -> go.Figure:
    """
    Создание фигуры диаграммы рассеяния для Plotly
    
    Args:
        x_data: Данные по оси X
        y_data: Данные по оси Y
        title: Заголовок
        x_label: Подпись оси X
        y_label: Подпись оси Y
        
    Returns:
        Фигура Plotly
    """
    try:
        # Удаление NaN значений
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        clean_x = x_data[mask]
        clean_y = y_data[mask]
        
        fig = go.Figure(data=go.Scatter(
            x=clean_x,
            y=clean_y,
            mode='markers',
            marker=dict(
                size=5,
                color=clean_y,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=y_label)
            ),
            hovertemplate=f'{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Ошибка создания диаграммы рассеяния: {e}")
        return go.Figure()


def create_spectral_profile_figure(wavelengths: np.ndarray, 
                                 reflectance: np.ndarray,
                                 title: str = "Спектральный профиль") -> go.Figure:
    """
    Создание фигуры спектрального профиля для Plotly
    
    Args:
        wavelengths: Длины волн
        reflectance: Значения отражения
        title: Заголовок
        
    Returns:
        Фигура Plotly
    """
    try:
        fig = go.Figure(data=go.Scatter(
            x=wavelengths,
            y=reflectance,
            mode='lines',
            line=dict(color='blue', width=2),
            hovertemplate='Длина волны: %{x} нм<br>Отражение: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Длина волны (нм)",
            yaxis_title="Отражение",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Ошибка создания спектрального профиля: {e}")
        return go.Figure()


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Расчет базовой статистики для данных
    
    Args:
        data: Входные данные
        
    Returns:
        Словарь со статистикой
    """
    try:
        # Удаление NaN значений
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'count': 0
            }
        
        return {
            'mean': float(np.mean(clean_data)),
            'std': float(np.std(clean_data)),
            'min': float(np.min(clean_data)),
            'max': float(np.max(clean_data)),
            'median': float(np.median(clean_data)),
            'count': int(len(clean_data))
        }
        
    except Exception as e:
        print(f"Ошибка расчета статистики: {e}")
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'count': 0
        }


def create_legend_items(colormap_name: str, vmin: float, vmax: float, 
                       n_items: int = 5) -> List[Dict[str, Any]]:
    """
    Создание элементов легенды для цветовой карты
    
    Args:
        colormap_name: Название цветовой схемы
        vmin: Минимальное значение
        vmax: Максимальное значение
        n_items: Количество элементов легенды
        
    Returns:
        Список элементов легенды
    """
    try:
        colors = create_colormap(colormap_name, n_items)
        values = np.linspace(vmin, vmax, n_items)
        
        legend_items = []
        for i, (value, color) in enumerate(zip(values, colors)):
            legend_items.append({
                'value': float(value),
                'color': f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})',
                'label': f'{value:.3f}'
            })
        
        return legend_items
        
    except Exception as e:
        print(f"Ошибка создания легенды: {e}")
        return []


def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Нормализация данных
    
    Args:
        data: Входные данные
        method: Метод нормализации ('minmax', 'zscore', 'robust')
        
    Returns:
        Нормализованные данные
    """
    try:
        # Удаление NaN значений для расчета статистики
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return data
        
        if method == 'minmax':
            vmin, vmax = np.min(clean_data), np.max(clean_data)
            if vmax == vmin:
                return np.zeros_like(data)
            return (data - vmin) / (vmax - vmin)
        
        elif method == 'zscore':
            mean, std = np.mean(clean_data), np.std(clean_data)
            if std == 0:
                return np.zeros_like(data)
            return (data - mean) / std
        
        elif method == 'robust':
            median, mad = np.median(clean_data), np.median(np.abs(clean_data - np.median(clean_data)))
            if mad == 0:
                return np.zeros_like(data)
            return (data - median) / mad
        
        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")
            
    except Exception as e:
        print(f"Ошибка нормализации данных: {e}")
        return data


def create_colorbar_config(colormap_name: str, title: str = "Значение") -> Dict[str, Any]:
    """
    Создание конфигурации цветовой полосы для Plotly
    
    Args:
        colormap_name: Название цветовой схемы
        title: Заголовок цветовой полосы
        
    Returns:
        Конфигурация цветовой полосы
    """
    # Конвертация цветовой схемы matplotlib в plotly
    plotly_colorscales = {
        'viridis': 'Viridis',
        'plasma': 'Plasma',
        'inferno': 'Inferno',
        'magma': 'Magma',
        'RdYlGn': 'RdYlGn',
        'RdYlBu': 'RdYlBu',
        'RdBu': 'RdBu',
        'coolwarm': 'RdBu',
        'terrain': 'Earth'
    }
    
    plotly_colorscale = plotly_colorscales.get(colormap_name, 'Viridis')
    
    return {
        'title': title,
        'titleside': 'right',
        'tickmode': 'auto',
        'ticks': 'outside',
        'showticklabels': True,
        'tickfont': {'size': 10},
        'len': 0.7,
        'lenmode': 'fraction',
        'x': 1.02,
        'xanchor': 'left',
        'y': 0.5,
        'yanchor': 'middle'
    }