"""
Visualization utilities for the GOP project.

This module provides visualization functions for orthophoto data and basic data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Optional, Tuple, Any, List, Union
from numpy.typing import NDArray
from .image_utils import apply_colormap, normalize_image

# Type aliases for better type safety
VisualizationData = Dict[str, NDArray[np.float32]]
Figure = plt.Figure
Axes = plt.Axes


def visualize_orthophoto(
    orthophoto_data: NDArray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Orthophoto View"
) -> Optional[Figure]:
    """
    Visualize orthophoto data.

    Args:
        orthophoto_data: Orthophoto data array
        output_path: Path to save the image
        figsize: Figure size
        title: Plot title

    Returns:
        Figure object or None if no data provided
    """
    if orthophoto_data.size == 0:
        raise ValueError("orthophoto_data cannot be empty")

    fig, ax = plt.subplots(figsize=figsize)

    # Handle different data formats
    if len(orthophoto_data.shape) == 3 and orthophoto_data.shape[2] >= 3:
        # RGB or multispectral data - use first 3 bands for RGB
        rgb_data = orthophoto_data[:, :, :3]
        # Normalize for display
        rgb_data = (rgb_data - np.min(rgb_data)) / (np.max(rgb_data) - np.min(rgb_data))
        ax.imshow(rgb_data)
    else:
        # Single band data - use colormap
        normalized = normalize_image(orthophoto_data, method="minmax")
        im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def create_data_histogram(
    data: NDArray,
    title: str = "Data Distribution",
    output_path: Optional[str] = None,
    bins: int = 50,
) -> plt.Figure:
    """
    Plot histogram of data values with statistical information.

    Args:
        data: Data array
        title: Plot title
        output_path: Path to save the plot
        bins: Number of histogram bins

    Returns:
        Figure object
    """
    if data.size == 0:
        raise ValueError("data cannot be empty")
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove NaN values
    valid_data = data[~np.isnan(data)]

    ax.hist(valid_data, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)
    ax.axvline(
        float(mean_val),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.3f}",
    )
    ax.axvline(
        float(mean_val + std_val),
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"±σ: {std_val:.3f}",
    )
    ax.axvline(float(mean_val - std_val), color="orange", linestyle="--", alpha=0.7)

    ax.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def create_processing_workflow_chart(
    workflow_steps: List[str], output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create processing workflow chart showing analysis steps.

    Args:
        workflow_steps: List of processing steps
        output_path: Path to save the chart

    Returns:
        Figure object
    """
    if not workflow_steps:
        raise ValueError("workflow_steps cannot be empty")
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create horizontal process chart
    y_pos = np.arange(len(workflow_steps))

    bars = ax.barh(y_pos, [1] * len(workflow_steps), color="lightblue", alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(workflow_steps)
    ax.invert_yaxis()  # Top to bottom
    ax.set_xlabel("Progress", fontsize=12)
    ax.set_title("Processing Workflow", fontsize=14)

    # Remove X axis
    ax.set_xticks([])

    # Add step numbers
    for i, (bar, step) in enumerate(zip(bars, workflow_steps)):
        width = bar.get_width()
        ax.text(
            width / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{i+1}",
            ha="center",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def create_rgb_composite(
    data: NDArray,
    band_indices: Tuple[int, int, int] = (0, 1, 2),
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[Figure]:
    """
    Create RGB composite from multispectral data.

    Args:
        data: Multispectral data array
        band_indices: Tuple of (R, G, B) band indices
        output_path: Path to save the image
        figsize: Figure size

    Returns:
        Figure object or None if data cannot be processed
    """
    if data.size == 0:
        raise ValueError("data cannot be empty")
    
    if len(data.shape) != 3 or data.shape[2] < 3:
        raise ValueError("Data must have at least 3 bands for RGB composite")
    
    # Extract RGB bands
    r_band = data[:, :, band_indices[0]]
    g_band = data[:, :, band_indices[1]]
    b_band = data[:, :, band_indices[2]]
    
    # Normalize each band
    r_norm = normalize_image(r_band, method="minmax")
    g_norm = normalize_image(g_band, method="minmax")
    b_norm = normalize_image(b_band, method="minmax")
    
    # Create RGB composite
    rgb_composite = np.stack([r_norm, g_norm, b_norm], axis=-1)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb_composite)
    ax.set_title(f"RGB Composite (Bands {band_indices})", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


__all__ = [
    "visualize_orthophoto",
    "create_data_histogram",
    "create_processing_workflow_chart",
    "create_rgb_composite",
]
