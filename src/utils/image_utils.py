"""
Утилиты для работы с изображениями
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path, mode='RGB'):
    """
    Загрузка изображения
    
    Args:
        image_path (str): Путь к изображению
        mode (str): Режим загрузки ('RGB', 'L', 'RGBA')
        
    Returns:
        numpy.ndarray: Изображение в виде массива
    """
    if mode == 'RGB':
        image = Image.open(image_path).convert('RGB')
        return np.array(image)
    elif mode == 'L':
        image = Image.open(image_path).convert('L')
        return np.array(image)
    elif mode == 'RGBA':
        image = Image.open(image_path).convert('RGBA')
        return np.array(image)
    else:
        raise ValueError(f"Неподдерживаемый режим: {mode}")


def save_image(image_array, output_path):
    """
    Сохранение изображения
    
    Args:
        image_array (numpy.ndarray): Массив изображения
        output_path (str): Путь для сохранения
    """
    if len(image_array.shape) == 3:
        # RGB изображение
        image = Image.fromarray(image_array.astype(np.uint8))
    else:
        # Оттенки серого
        image = Image.fromarray(image_array.astype(np.uint8), mode='L')
    
    image.save(output_path)


def resize_image(image, target_size, interpolation=cv2.INTER_LINEAR):
    """
    Изменение размера изображения
    
    Args:
        image (numpy.ndarray): Входное изображение
        target_size (tuple): Целевой размер (width, height)
        interpolation: Метод интерполяции
        
    Returns:
        numpy.ndarray: Измененное изображение
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_image(image, method='minmax'):
    """
    Нормализация изображения
    
    Args:
        image (numpy.ndarray): Входное изображение
        method (str): Метод нормализации ('minmax', 'zscore')
        
    Returns:
        numpy.ndarray: Нормализованное изображение
    """
    if method == 'minmax':
        # Минимаксная нормализация [0, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(image)
    elif method == 'zscore':
        # Z-нормализация
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        else:
            return np.zeros_like(image)
    else:
        raise ValueError(f"Неподдерживаемый метод нормализации: {method}")


def apply_colormap(image, colormap=cv2.COLORMAP_JET):
    """
    Применение цветовой карты к изображению в оттенках серого
    
    Args:
        image (numpy.ndarray): Входное изображение в оттенках серого
        colormap: Цветовая карта OpenCV
        
    Returns:
        numpy.ndarray: Изображение с примененной цветовой картой
    """
    # Нормализация в диапазон [0, 255]
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(normalized, colormap)


def blend_images(image1, image2, alpha=0.5):
    """
    Смешивание двух изображений
    
    Args:
        image1 (numpy.ndarray): Первое изображение
        image2 (numpy.ndarray): Второе изображение
        alpha (float): Вес первого изображения [0, 1]
        
    Returns:
        numpy.ndarray: Смешанное изображение
    """
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)


def create_thumbnail(image_path, size=(256, 256)):
    """
    Создание миниатюры изображения
    
    Args:
        image_path (str): Путь к изображению
        size (tuple): Размер миниатюры
        
    Returns:
        numpy.ndarray: Миниатюра
    """
    image = load_image(image_path)
    return resize_image(image, size)


def calculate_histogram(image, bins=256):
    """
    Расчет гистограммы изображения
    
    Args:
        image (numpy.ndarray): Входное изображение
        bins (int): Количество бинов
        
    Returns:
        tuple: (гистограмма, границы бинов)
    """
    if len(image.shape) == 3:
        # RGB изображение - расчет для каждого канала
        histograms = []
        for i in range(3):
            hist, bins = np.histogram(image[:, :, i].flatten(), bins=bins, range=(0, 256))
            histograms.append(hist)
        return histograms, bins
    else:
        # Оттенки серого
        hist, bins = np.histogram(image.flatten(), bins=bins, range=(0, 256))
        return hist, bins


def enhance_contrast(image, method='histogram_eq'):
    """
    Улучшение контраста изображения
    
    Args:
        image (numpy.ndarray): Входное изображение
        method (str): Метод улучшения ('histogram_eq', 'clahe')
        
    Returns:
        numpy.ndarray: Улучшенное изображение
    """
    if method == 'histogram_eq':
        if len(image.shape) == 3:
            # RGB - преобразование в YUV, эквализация Y, обратное преобразование
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            return cv2.equalizeHist(image)
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            return clahe.apply(image)
    else:
        raise ValueError(f"Неподдерживаемый метод улучшения: {method}")


def remove_noise(image, method='gaussian'):
    """
    Удаление шума из изображения
    
    Args:
        image (numpy.ndarray): Входное изображение
        method (str): Метод удаления шума ('gaussian', 'bilateral', 'median')
        
    Returns:
        numpy.ndarray: Очищенное изображение
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    else:
        raise ValueError(f"Неподдерживаемый метод удаления шума: {method}")