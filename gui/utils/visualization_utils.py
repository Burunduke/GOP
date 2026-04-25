"""
Visualization utilities for GOP GUI application
"""

import numpy as np
import plotly.graph_objs as go
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt


def create_colormap(name: str, n_colors: int = 256) -> List[Tuple[float, float, float]]:
    """
    Create color map
    
    Args:
        name: Color scheme name
        n_colors: Number of colors
        
    Returns:
        List of RGB tuples
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
            cmap = plt.cm.viridis  # Default
        
        # Get colors and convert to RGB
        colors = []
        for i in range(n_colors):
            rgba = cmap(i / (n_colors - 1))
            rgb = (rgba[0], rgba[1], rgba[2])  # Remove alpha channel
            colors.append(rgb)
        
        return colors
        
    except Exception:
        # Return basic color scheme in case of error
        return [(i/255, i/255, i/255) for i in range(n_colors)]


def apply_colormap(data: np.ndarray, colormap_name: str = 'viridis', 
                   vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """
    Apply color map to data
    
    Args:
        data: Input data
        colormap_name: Color scheme name
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        
    Returns:
        RGB array
    """
    try:
        # Normalize data
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        
        # Handle case with identical values
        if vmax == vmin:
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = (data - vmin) / (vmax - vmin)
        
        # Apply color map
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
        
        # Remove alpha channel
        return rgb_array[:, :, :3]
        
    except Exception as e:
        print(f"Error applying color map: {e}")
        # Return grayscale in case of error
        gray_data = np.zeros((data.shape[0], data.shape[1], 3))
        gray_data[:, :, 0] = gray_data[:, :, 1] = gray_data[:, :, 2] = data
        return gray_data


def create_heatmap_figure(data: np.ndarray, x_coords: Optional[np.ndarray] = None,
                         y_coords: Optional[np.ndarray] = None, 
                         colorscale: str = 'viridis',
                         title: str = "Heatmap") -> go.Figure:
    """
    Create heatmap figure for Plotly
    
    Args:
        data: 2D data array
        x_coords: X-axis coordinates
        y_coords: Y-axis coordinates
        colorscale: Color scheme
        title: Title
        
    Returns:
        Plotly figure
    """
    try:
        # Create coordinates if not provided
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])
        
        # Convert matplotlib color scheme to plotly
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
            colorbar=dict(title="Value"),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        # Return empty figure
        return go.Figure()


def create_histogram_figure(data: np.ndarray, bins: int = 50, 
                          title: str = "Distribution Histogram",
                          x_label: str = "Value",
                          y_label: str = "Frequency") -> go.Figure:
    """
    Create histogram figure for Plotly
    
    Args:
        data: 1D data array
        bins: Number of bins
        title: Title
        x_label: X-axis label
        y_label: Y-axis label
        
    Returns:
        Plotly figure
    """
    try:
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        
        fig = go.Figure(data=go.Histogram(
            x=clean_data,
            nbinsx=bins,
            marker_color='rgba(55, 128, 191, 0.7)',
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating histogram: {e}")
        return go.Figure()


def create_scatter_figure(x_data: np.ndarray, y_data: np.ndarray,
                         title: str = "Scatter Plot",
                         x_label: str = "X",
                         y_label: str = "Y") -> go.Figure:
    """
    Create scatter plot figure for Plotly
    
    Args:
        x_data: X-axis data
        y_data: Y-axis data
        title: Title
        x_label: X-axis label
        y_label: Y-axis label
        
    Returns:
        Plotly figure
    """
    try:
        # Remove NaN values
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
        print(f"Error creating scatter plot: {e}")
        return go.Figure()




def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for data
    
    Args:
        data: Input data
        
    Returns:
        Dictionary with statistics
    """
    try:
        # Remove NaN values
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
        print(f"Error calculating statistics: {e}")
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
    Create legend items for color map
    
    Args:
        colormap_name: Color scheme name
        vmin: Minimum value
        vmax: Maximum value
        n_items: Number of legend items
        
    Returns:
        List of legend items
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
        print(f"Error creating legend: {e}")
        return []


def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize data
    
    Args:
        data: Input data
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized data
    """
    try:
        # Remove NaN values for statistics calculation
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
            raise ValueError(f"Unknown normalization method: {method}")
            
    except Exception as e:
        print(f"Error normalizing data: {e}")
        return data


def create_colorbar_config(colormap_name: str, title: str = "Value") -> Dict[str, Any]:
    """
    Create color bar configuration for Plotly
    
    Args:
        colormap_name: Color scheme name
        title: Color bar title
        
    Returns:
        Color bar configuration
    """
    # Convert matplotlib color scheme to plotly
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