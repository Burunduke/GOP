"""
Visualization utilities for the GOP project.

This module provides visualization functions for vegetation indices, comparison plots,
histograms, and various charts for plant condition analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
from typing import Dict, Optional, Tuple, Any, List, Union
from numpy.typing import NDArray
from .image_utils import apply_colormap, normalize_image

# Type aliases for better type safety
VisualizationData = Dict[str, NDArray[np.float32]]
Figure = plt.Figure
Axes = plt.Axes


def visualize_indices(
    indices_dict: VisualizationData,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> Optional[Figure]:
    """
    Visualize vegetation indices in a grid layout.

    Args:
        indices_dict: Dictionary with indices {name: array}
        output_path: Path to save the image
        figsize: Figure size

    Returns:
        Figure object or None if no indices provided
    """
    n_indices = len(indices_dict)
    if n_indices == 0:
        raise ValueError("indices_dict cannot be empty")

    # Determine grid layout
    cols = min(3, n_indices)
    rows = (n_indices + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_indices == 1:
        axes = [axes]
    elif rows == 1 and hasattr(axes, 'reshape'):
        axes = axes.reshape(1, -1)

    for i, (index_name, index_data) in enumerate(indices_dict.items()):
        row = i // cols
        col = i % cols

        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        # Normalize data
        normalized = normalize_image(index_data, method="minmax")

        # Visualization
        im = ax.imshow(normalized, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"{index_name}", fontsize=12)
        ax.axis("off")

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for i in range(n_indices, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            fig.delaxes(axes[col])
        else:
            fig.delaxes(axes[row, col])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def create_comparison_plot(
    original_image: NDArray,
    segmentation_mask: NDArray,
    indices_dict: Dict[str, NDArray],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12),
) -> plt.Figure:
    """
    Create comparison plot with original image, segmentation mask, and indices.

    Args:
        original_image: Original image array
        segmentation_mask: Segmentation mask array
        indices_dict: Dictionary with indices
        output_path: Path to save the plot
        figsize: Figure size

    Returns:
        Figure object
    """
    # Validate inputs
    if original_image.size == 0:
        raise ValueError("original_image cannot be empty")
    if segmentation_mask.size == 0:
        raise ValueError("segmentation_mask cannot be empty")
    
    # Select up to 3 best indices for display
    selected_indices = dict(list(indices_dict.items())[:3])

    n_plots = 2 + len(selected_indices)  # original + mask + indices
    cols = min(4, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif rows == 1 and hasattr(axes, 'reshape'):
        axes = axes.reshape(1, -1)

    plot_idx = 0

    # Original image
    row = plot_idx // cols
    col = plot_idx % cols
    if rows == 1:
        ax = axes[col]
    else:
        ax = axes[row, col]

    if len(original_image.shape) == 3:
        ax.imshow(original_image)
    else:
        ax.imshow(original_image, cmap="gray")
    ax.set_title("Original Image", fontsize=12)
    ax.axis("off")
    plot_idx += 1

    # Segmentation mask
    row = plot_idx // cols
    col = plot_idx % cols
    if rows == 1:
        ax = axes[col]
    else:
        ax = axes[row, col]

    ax.imshow(segmentation_mask, cmap="tab20")
    ax.set_title("Segmentation Mask", fontsize=12)
    ax.axis("off")
    plot_idx += 1

    # Vegetation indices
    for index_name, index_data in selected_indices.items():
        row = plot_idx // cols
        col = plot_idx % cols
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        normalized = normalize_image(index_data, method="minmax")
        im = ax.imshow(normalized, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"{index_name}", fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            fig.delaxes(axes[col])
        else:
            fig.delaxes(axes[row, col])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_index_histogram(
    index_data: NDArray,
    index_name: str,
    output_path: Optional[str] = None,
    bins: int = 50,
) -> plt.Figure:
    """
    Plot histogram of index values with statistical information.

    Args:
        index_data: Index data array
        index_name: Name of the index
        output_path: Path to save the plot
        bins: Number of histogram bins

    Returns:
        Figure object
    """
    if index_data.size == 0:
        raise ValueError("index_data cannot be empty")
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove NaN values
    valid_data = index_data[~np.isnan(index_data)]

    ax.hist(valid_data, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
    ax.set_title(f"Distribution of {index_name} Index Values", fontsize=14)
    ax.set_xlabel("Index Value", fontsize=12)
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


def create_plant_condition_chart(
    plant_condition_data: Dict[str, Any], output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create plant condition chart with classification and index values.

    Args:
        plant_condition_data: Plant condition data dictionary
        output_path: Path to save the chart

    Returns:
        Figure object
    """
    # Validate required data
    if "classification" not in plant_condition_data:
        raise ValueError("plant_condition_data must contain 'classification' key")
    
    classification = plant_condition_data["classification"]
    if "class" not in classification or "score" not in classification:
        raise ValueError("classification must contain 'class' and 'score' keys")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Classification data
    classification = plant_condition_data.get("classification", {})
    class_name = classification.get("class", "Unknown")
    confidence = classification.get("score", 0)

    # Pie chart for classification
    labels = [class_name, "Other"]
    sizes = [confidence, 1 - confidence]
    colors = ["#2ecc71", "#ecf0f1"]

    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Plant Condition Classification", fontsize=14)

    # Bar chart for indices
    indices = plant_condition_data.get("indices", {})
    if indices:
        index_names = list(indices.keys())
        index_values = list(indices.values())

        bars = ax2.bar(index_names, index_values, color="skyblue", alpha=0.7)
        ax2.set_title("Normalized Index Values", fontsize=14)
        ax2.set_ylabel("Normalized Value", fontsize=12)
        ax2.set_ylim(0, 1)

        # Add values on bars
        for bar, value in zip(bars, index_values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # Rotate labels
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    else:
        ax2.text(
            0.5,
            0.5,
            "No index data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Normalized Index Values", fontsize=14)

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


__all__ = [
    "visualize_indices",
    "create_comparison_plot",
    "plot_index_histogram",
    "create_plant_condition_chart",
    "create_processing_workflow_chart",
]
