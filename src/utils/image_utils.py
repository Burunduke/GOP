"""
Image utility functions for the GOP project.

This module provides image processing utilities including loading, saving, resizing,
normalization, and various image enhancement operations.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional, List
from numpy.typing import NDArray

# Type aliases for better type safety
ImageData = NDArray[np.uint8]
GrayImage = NDArray[np.uint8]
RGBImage = NDArray[np.uint8]
RGBAImage = NDArray[np.uint8]
FloatImage = NDArray[np.float32]


def load_image(image_path: str, mode: str = "RGB") -> ImageData:
    """
    Load image from file.

    Args:
        image_path: Path to the image file
        mode: Loading mode ('RGB', 'L', 'RGBA')

    Returns:
        Image as numpy array

    Raises:
        ValueError: If unsupported mode is provided
    """
    if mode == "RGB":
        image = Image.open(image_path).convert("RGB")
        return np.array(image)
    elif mode == "L":
        image = Image.open(image_path).convert("L")
        return np.array(image)
    elif mode == "RGBA":
        image = Image.open(image_path).convert("RGBA")
        return np.array(image)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def save_image(image_array: ImageData, output_path: str) -> None:
    """
    Save image array to file.

    Args:
        image_array: Image array to save
        output_path: Path for saving the image
    """
    if len(image_array.shape) == 3:
        # RGB image
        image = Image.fromarray(image_array.astype(np.uint8))
    else:
        # Grayscale image
        image = Image.fromarray(image_array.astype(np.uint8), mode="L")

    image.save(output_path)


def resize_image(
    image: NDArray, target_size: Tuple[int, int], interpolation: int = cv2.INTER_LINEAR
) -> NDArray:
    """
    Resize image to target dimensions.

    Args:
        image: Input image array
        target_size: Target size (width, height)
        interpolation: Interpolation method

    Returns:
        Resized image array
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_image(image: NDArray, method: str = "minmax") -> NDArray:
    """
    Normalize image using specified method.

    Args:
        image: Input image array
        method: Normalization method ('minmax', 'zscore')

    Returns:
        Normalized image array

    Raises:
        ValueError: If unsupported normalization method is provided
    """
    if method == "minmax":
        # Min-max normalization [0, 1]
        min_val: float = np.min(image)
        max_val: float = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(image)
    elif method == "zscore":
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        else:
            return np.zeros_like(image)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def apply_colormap(image: NDArray, colormap: int = cv2.COLORMAP_JET) -> NDArray:
    """
    Apply colormap to grayscale image.

    Args:
        image: Input grayscale image
        colormap: OpenCV colormap constant

    Returns:
        Image with applied colormap
    """
    # Normalize to [0, 255] range
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(normalized, colormap)


def blend_images(image1: NDArray, image2: NDArray, alpha: float = 0.5) -> NDArray:
    """
    Blend two images using weighted average.

    Args:
        image1: First image
        image2: Second image
        alpha: Weight of first image [0, 1]

    Returns:
        Blended image
    """
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)


def create_thumbnail(image_path: str, size: Tuple[int, int] = (256, 256)) -> NDArray:
    """
    Create thumbnail from image file.

    Args:
        image_path: Path to the image file
        size: Thumbnail size

    Returns:
        Thumbnail image array
    """
    image = load_image(image_path)
    return resize_image(image, size)


def calculate_histogram(image: NDArray, bins: int = 256) -> tuple:
    """
    Calculate image histogram.

    Args:
        image: Input image array
        bins: Number of histogram bins

    Returns:
        Tuple of (histogram, bin_edges)
    """
    if len(image.shape) == 3:
        # RGB image - calculate for each channel
        histograms = []
        for i in range(3):
            hist, bins = np.histogram(
                image[:, :, i].flatten(), bins=bins, range=(0, 256)
            )
            histograms.append(hist)
        return histograms, bins
    else:
        # Grayscale image
        hist, bins = np.histogram(image.flatten(), bins=bins, range=(0, 256))
        return hist, bins


def enhance_contrast(image: NDArray, method: str = "histogram_eq") -> NDArray:
    """
    Enhance image contrast using specified method.

    Args:
        image: Input image array
        method: Enhancement method ('histogram_eq', 'clahe')

    Returns:
        Contrast-enhanced image

    Raises:
        ValueError: If unsupported enhancement method is provided
    """
    if method == "histogram_eq":
        if len(image.shape) == 3:
            # RGB - convert to YUV, equalize Y channel, convert back
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            return cv2.equalizeHist(image)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            return clahe.apply(image)
    else:
        raise ValueError(f"Unsupported enhancement method: {method}")


def remove_noise(image: NDArray, method: str = "gaussian") -> NDArray:
    """
    Remove noise from image using specified method.

    Args:
        image: Input image array
        method: Noise removal method ('gaussian', 'bilateral', 'median')

    Returns:
        Denoised image

    Raises:
        ValueError: If unsupported noise removal method is provided
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == "median":
        return cv2.medianBlur(image, 5)
    else:
        raise ValueError(f"Unsupported noise removal method: {method}")


__all__ = [
    "load_image",
    "save_image",
    "resize_image",
    "normalize_image",
    "apply_colormap",
    "blend_images",
    "create_thumbnail",
    "calculate_histogram",
    "enhance_contrast",
    "remove_noise",
]
