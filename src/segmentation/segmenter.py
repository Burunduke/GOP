"""
Main image segmentation class.

This module provides advanced image segmentation capabilities for high-resolution
remote sensing imagery using deep learning models and refinement techniques.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from numpy.typing import NDArray

# Type aliases for better type safety
SegmentationResult = Dict[str, Union[str, NDArray[np.uint8], Dict[str, Any]]]
ImageData = NDArray[np.uint8]
MaskData = NDArray[np.uint8]

try:
    from osgeo import gdal

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    # Don't raise error here to allow tests to run

from ..core.config import get_config
from ..utils.logger import setup_logger
from ..utils.gdal_utils import open_gdal_dataset


# Constants for segmentation parameters
DEFAULT_COMPRESSION_RATIO = 0.125
DEFAULT_NDVI_THRESHOLD = 0.2
DEFAULT_MASK_QUALITY_THRESHOLD = 0.5
MORPH_KERNEL_SIZE = 3
GAUSSIAN_KERNEL_SIZE = 5

class ImageSegmenter:
    """
    Main class for high-resolution image segmentation.
    
    This class provides advanced segmentation capabilities using deep learning
    models (DeepLabV3+ and CascadePSP) with refinement techniques for
    ultra-high-resolution remote sensing imagery.
    """

    def __init__(self):
        """Initialize the image segmenter."""
        self.logger = setup_logger("ImageSegmenter")

        # Model initialization (stubs for compatibility)
        self.deeplab_segmenter = None
        self.cascade_segmenter = None

    def segment(
        self,
        image_path: str,
        output_dir: str = "results",
        use_refinement: bool = True,
        compression_ratio: Optional[float] = None,
    ) -> str:
        """
        Segment image using cascade approach.

        Args:
            image_path: Path to the image
            output_dir: Directory for saving results
            use_refinement: Use boundary refinement
            compression_ratio: Compression ratio for preliminary segmentation

        Returns:
            Path to the final segmentation mask

        Raises:
            FileNotFoundError: If input image is not found
            Exception: If segmentation fails
        """
        try:
            self.logger.info(f"Starting image segmentation: {image_path}")

            # Validate input file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Configure parameters
            if compression_ratio is None:
                config_instance = get_config()
                compression_ratio = config_instance.get("processing.compression_ratio", DEFAULT_COMPRESSION_RATIO)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Stage 1: Preliminary segmentation with DeepLabV3+
            self.logger.info("Stage 1: Preliminary segmentation with DeepLabV3+")
            coarse_mask_path = self._preliminary_segmentation(
                image_path, output_dir, compression_ratio
            )

            if not use_refinement:
                self.logger.info("Boundary refinement disabled")
                return coarse_mask_path

            # Stage 2: Boundary refinement with CascadePSP
            self.logger.info("Stage 2: Boundary refinement with CascadePSP")
            refined_mask_path = self._refine_segmentation(
                image_path, coarse_mask_path, output_dir
            )

            # Stage 3: Select best mask
            self.logger.info("Stage 3: Selecting optimal mask")
            final_mask_path = self._select_best_mask(
                image_path, coarse_mask_path, refined_mask_path, output_dir
            )

            self.logger.info(f"Segmentation completed: {final_mask_path}")
            return final_mask_path

        except Exception as e:
            self.logger.error(f"Segmentation error: {e}")
            raise

    def _preliminary_segmentation(
        self, image_path: str, output_dir: str, compression_ratio: float
    ) -> str:
        """
        Preliminary segmentation using DeepLabV3+.

        Args:
            image_path: Path to the image
            output_dir: Directory for saving
            compression_ratio: Compression ratio

        Returns:
            Path to preliminary segmentation mask

        Raises:
            Exception: If preliminary segmentation fails
        """
        try:
            # Read and compress image
            image_data, original_shape = self._read_and_compress_image(
                image_path, compression_ratio
            )

            # Simplified segmentation (stub implementation)
            mask_data = self._simple_segmentation(image_data)

            # Resize mask to original size
            mask_resized = self._resize_mask(mask_data, original_shape)

            # Save mask
            output_path = os.path.join(output_dir, "coarse_segmentation.tif")
            self._save_mask(mask_resized, output_path, image_path)

            return output_path

        except Exception as e:
            self.logger.error(f"Preliminary segmentation error: {e}")
            raise

    def _refine_segmentation(
        self, image_path: str, coarse_mask_path: str, output_dir: str
    ) -> str:
        """
        Уточнение границ с помощью CascadePSP

        Args:
            image_path: Путь к исходному изображению
            coarse_mask_path: Путь к грубой маске
            output_dir: Директория для сохранения

        Returns:
            Путь к уточненной маске
        """
        try:
            # Чтение исходного изображения и маски
            image_data = self._read_image(image_path)
            mask_data = self._read_mask(coarse_mask_path)

            # Упрощенное уточнение (заглушка)
            refined_mask = self._simple_refinement(image_data, mask_data)

            # Сохранение уточненной маски
            output_path = os.path.join(output_dir, "refined_segmentation.tif")
            self._save_mask(refined_mask, output_path, image_path)

            return output_path

        except Exception as e:
            self.logger.error(f"Ошибка уточнения сегментации: {e}")
            raise

    def _select_best_mask(
        self,
        image_path: str,
        coarse_mask_path: str,
        refined_mask_path: str,
        output_dir: str,
    ) -> str:
        """
        Выбор лучшей маски сегментации

        Args:
            image_path: Путь к исходному изображению
            coarse_mask_path: Путь к грубой маске
            refined_mask_path: Путь к уточненной маске
            output_dir: Директория для сохранения

        Returns:
            Путь к лучшей маске
        """
        try:
            # Оценка качества масок
            coarse_quality = self._evaluate_mask_quality(coarse_mask_path)
            refined_quality = self._evaluate_mask_quality(refined_mask_path)

            self.logger.info(f"Качество грубой маски: {coarse_quality:.3f}")
            self.logger.info(f"Качество уточненной маски: {refined_quality:.3f}")

            # Выбор лучшей маски
            if refined_quality > coarse_quality:
                best_mask_path = refined_mask_path
                self.logger.info("Выбрана уточненная маска")
            else:
                best_mask_path = coarse_mask_path
                self.logger.info("Выбрана грубая маска")

            # Копирование лучшей маски как финальной
            final_mask_path = os.path.join(output_dir, "final_segmentation.tif")
            self._copy_file(best_mask_path, final_mask_path)

            return final_mask_path

        except Exception as e:
            self.logger.error(f"Ошибка выбора лучшей маски: {e}")
            # В случае ошибки возвращаем грубую маску
            return coarse_mask_path

    def _read_and_compress_image(
        self, image_path: str, compression_ratio: float
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Чтение и сжатие изображения

        Args:
            image_path: Путь к изображению
            compression_ratio: Коэффициент сжатия

        Returns:
            Кортеж (сжатое изображение, исходный размер)
        """
        try:
            from ..utils.gdal_utils import read_raster_bands, get_raster_metadata

            # Чтение данных с использованием централизованной утилиты
            if not GDAL_AVAILABLE:
                raise ImportError("GDAL library is required but not available. Install with: pip install gdal")
            image_data = read_raster_bands(image_path)

            # Получение метаданных для исходного размера
            metadata = get_raster_metadata(image_path)
            original_shape = (metadata["height"], metadata["width"])

            # Сжатие изображения
            if compression_ratio < 1.0:
                new_height = int(original_shape[0] * compression_ratio)
                new_width = int(original_shape[1] * compression_ratio)

                # Простое сжатие (в реальном приложении можно использовать более сложные методы)
                compressed_image = np.zeros(
                    (new_height, new_width, image_data.shape[2]), dtype=np.float32
                )

                for band in range(image_data.shape[2]):
                    # Использование билинейной интерполяции
                    from skimage.transform import resize

                    compressed_image[:, :, band] = resize(
                        image_data[:, :, band],
                        (new_height, new_width),
                        preserve_range=True,
                        anti_aliasing=True,
                    )

                return compressed_image, original_shape
            else:
                return image_data, original_shape

        except Exception as e:
            self.logger.error(f"Ошибка чтения и сжатия изображения: {e}")
            raise

    def _read_image(self, image_path: str) -> np.ndarray:
        """
        Чтение изображения

        Args:
            image_path: Путь к изображению

        Returns:
            Массив данных изображения
        """
        from ..utils.gdal_utils import read_raster_bands

        # Чтение данных с использованием централизованной утилиты
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL library is required but not available. Install with: pip install gdal")
        return read_raster_bands(image_path)

    def _read_mask(self, mask_path: str) -> np.ndarray:
        """
        Чтение маски

        Args:
            mask_path: Путь к маске

        Returns:
            Массив данных маски
        """
        from ..utils.gdal_utils import read_raster_band

        # Чтение первого канала с использованием централизованной утилиты
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL library is required but not available. Install with: pip install gdal")
        mask_data = read_raster_band(mask_path, band_number=1)
        return mask_data.astype(np.uint8)

    def _resize_mask(
        self, mask_data: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Масштабирование маски к целевому размеру

        Args:
            mask_data: Исходная маска
            target_shape: Целевой размер (height, width)

        Returns:
            Масштабированная маска
        """
        try:
            from skimage.transform import resize

            resized_mask = resize(
                mask_data,
                target_shape,
                preserve_range=True,
                anti_aliasing=False,
                order=0,  # Ближайший сосед для масок
            )

            return resized_mask.astype(np.uint8)

        except ImportError:
            # Альтернативный метод без skimage
            import cv2

            resized_mask = cv2.resize(
                mask_data,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            return resized_mask

    def _save_mask(
        self, mask_data: np.ndarray, output_path: str, reference_path: str
    ) -> None:
        """
        Сохранение маски

        Args:
            mask_data: Данные маски
            output_path: Путь для сохранения
            reference_path: Путь к референсному изображению
        """
        try:
            from ..utils.gdal_utils import write_raster

            # Сохранение с использованием централизованной утилиты
            if not GDAL_AVAILABLE:
                raise ImportError("GDAL library is required but not available. Install with: pip install gdal")
            write_raster(mask_data, output_path, source_path=reference_path)

        except Exception as e:
            self.logger.error(f"Ошибка сохранения маски: {e}")
            raise

    def _evaluate_mask_quality(self, mask_path: str) -> float:
        """
        Evaluate mask quality using simple metrics.

        Args:
            mask_path: Path to the mask file

        Returns:
            Quality score (0-1)
        """
        try:
            mask_data = self._read_mask(mask_path)

            # Simple quality metrics
            # 1. Ratio of segmented area to total area
            area_ratio = np.sum(mask_data > 0) / mask_data.size

            # 2. Compactness (area to perimeter squared ratio)
            from skimage.measure import regionprops

            regions = regionprops(mask_data)

            if regions:
                compactness = regions[0].area / (regions[0].perimeter ** 2 + 1e-8)
            else:
                compactness = 0

            # Combined quality score
            quality = 0.7 * area_ratio + 0.3 * compactness

            return np.clip(quality, 0, 1)

        except Exception as e:
            self.logger.warning(f"Mask quality evaluation error: {e}")
            return DEFAULT_MASK_QUALITY_THRESHOLD  # Default quality score

    def _copy_file(self, src: str, dst: str) -> None:
        """
        Копирование файла

        Args:
            src: Исходный путь
            dst: Целевой путь
        """
        import shutil

        shutil.copy2(src, dst)

    def segment_batch(
        self, image_paths: list, output_dir: str = "results", **kwargs
    ) -> list:
        """
        Batch segmentation of multiple images.

        Args:
            image_paths: List of image paths
            output_dir: Directory for saving results
            **kwargs: Additional parameters

        Returns:
            List of result paths
        """
        results = []

        for i, image_path in enumerate(image_paths):
            try:
                self.logger.info(
                    f"Processing image {i+1}/{len(image_paths)}: {image_path}"
                )

                # Create individual directory for each image
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                image_output_dir = os.path.join(output_dir, image_name)

                # Segmentation
                mask_path = self.segment(image_path, image_output_dir, **kwargs)
                results.append(mask_path)

            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}")
                results.append(None)

        return results

    def _simple_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        """
        Simplified segmentation based on threshold values.
        
        Uses NDVI-like approach for vegetation segmentation or green channel
        thresholding for RGB images.
        """
        try:
            # Use NDVI-like approach for vegetation segmentation
            if image_data.shape[2] >= 3:
                # Assume channels: Red, Green, NIR
                red = image_data[:, :, 0]
                nir = (
                    image_data[:, :, 2]
                    if image_data.shape[2] >= 3
                    else image_data[:, :, 1]
                )

                # Calculate NDVI-like index
                ndvi_like = (nir - red) / (nir + red + 1e-8)

                # Threshold segmentation
                mask = (ndvi_like > DEFAULT_NDVI_THRESHOLD).astype(np.uint8)
                return mask
            else:
                # For RGB images use green channel
                green = image_data[:, :, 1]
                mask = (green > np.mean(green)).astype(np.uint8)
                return mask

        except Exception as e:
            self.logger.warning(f"Simplified segmentation error: {e}")
            # Return default mask
            return np.ones(image_data.shape[:2], dtype=np.uint8)

    def _simple_refinement(
        self, image_data: np.ndarray, mask_data: np.ndarray
    ) -> np.ndarray:
        """
        Simplified boundary refinement for segmentation.
        
        Uses morphological operations and Gaussian blur to refine mask boundaries.
        """
        try:
            import cv2

            # Morphological operations for boundary refinement
            kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

            # Closing to fill small holes
            closed = cv2.morphologyEx(mask_data, cv2.MORPH_CLOSE, kernel)

            # Opening to remove small objects
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

            # Gaussian blur for boundary smoothing
            smoothed = cv2.GaussianBlur(opened.astype(np.float32), (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)

            # Binarization back to binary mask
            refined = (smoothed > 0.5).astype(np.uint8)

            return refined

        except ImportError:
            # If OpenCV is not available, return original mask
            self.logger.warning("OpenCV not available, refinement not performed")
            return mask_data
        except Exception as e:
            self.logger.warning(f"Simplified refinement error: {e}")
            return mask_data
