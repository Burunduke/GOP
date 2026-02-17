#!/usr/bin/env python3
"""
Примеры интеграции GOP с другими научными библиотеками
Демонстрация использования OpenCV, scikit-learn, pandas и других библиотек

Этот пример показывает:
- Интеграцию с OpenCV для обработки изображений
- Использование scikit-learn для машинного обучения
- Работа с pandas для анализа данных
- Интеграцию с matplotlib/seaborn для визуализации
- Использование scipy для научных вычислений
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Добавление src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.pipeline import Pipeline
from src.processing.hyperspectral import HyperspectralProcessor
from src.indices.calculator import VegetationIndexCalculator
from src.segmentation.segmenter import ImageSegmenter
from src.utils.logger import setup_logger


def main():
    """Основная функция примеров интеграции"""
    
    # Настройка логирования
    logger = setup_logger('GOP_Integration', level=logging.INFO)
    logger.info("Начало демонстрации интеграции с другими библиотеками")
    
    try:
        # Создание выходной директории
        output_dir = "results/integration_examples"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        
        # Создание примера данных
        data_path = "data/integration_sample.bil"
        if not os.path.exists(data_path):
            create_integration_sample_data(data_path)
        
        # Пример 1: Интеграция с OpenCV
        logger.info("Пример 1: Интеграция с OpenCV")
        opencv_integration_example(data_path, output_dir)
        
        # Пример 2: Интеграция с scikit-learn
        logger.info("Пример 2: Интеграция с scikit-learn")
        sklearn_integration_example(data_path, output_dir)
        
        # Пример 3: Интеграция с pandas
        logger.info("Пример 3: Интеграция с pandas")
        pandas_integration_example(data_path, output_dir)
        
        # Пример 4: Интеграция с scipy
        logger.info("Пример 4: Интеграция с scipy")
        scipy_integration_example(data_path, output_dir)
        
        # Пример 5: Комплексная интеграция
        logger.info("Пример 5: Комплексная интеграция")
        comprehensive_integration_example(data_path, output_dir)
        
        print("\n" + "="*60)
        print("ПРИМЕРЫ ИНТЕГРАЦИИ ЗАВЕРШЕНЫ")
        print("="*60)
        print(f"Результаты сохранены в: {output_dir}")
        
    except Exception as e:
        logger.error(f"Ошибка в примерах интеграции: {e}")
        print(f"Ошибка: {e}")
        return 1
    
    return 0


def create_integration_sample_data(output_path: str):
    """Создание примера данных для демонстрации интеграции"""
    try:
        # Создание директории
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Параметры данных
        height, width, bands = 150, 150, 100
        wavelengths = np.linspace(400, 1000, bands)
        
        # Создание сложных гиперспектральных данных
        image_data = np.zeros((height, width, bands), dtype=np.float32)
        
        # Создание различных классов объектов
        for i in range(height):
            for j in range(width):
                # Определение класса пикселя
                if (i - 75)**2 + (j - 75)**2 < 900:  # Центральная область - здоровая растительность
                    spectrum = create_class_spectrum('healthy_vegetation', wavelengths)
                elif (i - 30)**2 + (j - 30)**2 < 400:  # Область стресса
                    spectrum = create_class_spectrum('stressed_vegetation', wavelengths)
                elif (i - 120)**2 + (j - 120)**2 < 400:  # Область болезни
                    spectrum = create_class_spectrum('diseased_vegetation', wavelengths)
                elif (i - 30)**2 + (j - 120)**2 < 400:  # Почва
                    spectrum = create_class_spectrum('soil', wavelengths)
                else:  # Фон
                    spectrum = create_class_spectrum('background', wavelengths)
                
                image_data[i, j, :] = spectrum
                
                # Добавление шума
                image_data[i, j, :] += np.random.normal(0, 0.01, bands)
        
        # Сохранение данных
        try:
            from osgeo import gdal
            
            # Создание HDR файла
            hdr_path = output_path.replace('.bil', '.hdr')
            with open(hdr_path, 'w') as f:
                f.write(f"ENVI\n")
                f.write(f"description = {{Integration sample data}}\n")
                f.write(f"samples = {width}\n")
                f.write(f"lines = {height}\n")
                f.write(f"bands = {bands}\n")
                f.write(f"data type = 4\n")
                f.write(f"interleave = bsq\n")
                f.write(f"byte order = 0\n")
                f.write(f"wavelength = {{")
                f.write(", ".join([f"{w:.1f}" for w in wavelengths]))
                f.write("}}\n")
            
            # Создание BIL файла
            driver = gdal.GetDriverByName('ENVI')
            dataset = driver.Create(output_path, width, height, bands, gdal.GDT_Float32)
            
            for band in range(bands):
                band_data = dataset.GetRasterBand(band + 1)
                band_data.WriteArray(image_data[:, :, band])
            
            dataset = None
            
        except ImportError:
            # Альтернативный метод сохранения
            np.save(output_path.replace('.bil', '.npy'), image_data)
            np.save(output_path.replace('.bil', '_wavelengths.npy'), wavelengths)
        
        print(f"Созданы примеры данных для интеграции: {output_path}")
        
    except Exception as e:
        print(f"Ошибка создания примера данных: {e}")


def create_class_spectrum(class_type: str, wavelengths: np.ndarray) -> np.ndarray:
    """Создание спектральной сигнатуры для класса"""
    spectrum = np.zeros_like(wavelengths)
    
    if class_type == 'healthy_vegetation':
        # Здоровая растительность
        green_peak = 0.4 * np.exp(-((wavelengths - 550) / 40) ** 2)
        red_edge = 0.6 / (1 + np.exp(-(wavelengths - 720) / 20))
        nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.7, 0)
        spectrum = 0.1 + green_peak + red_edge + nir_plateau
        
    elif class_type == 'stressed_vegetation':
        # Стрессовая растительность
        green_peak = 0.2 * np.exp(-((wavelengths - 550) / 50) ** 2)
        red_edge = 0.3 / (1 + np.exp(-(wavelengths - 720) / 25))
        nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.4, 0)
        spectrum = 0.15 + green_peak + red_edge + nir_plateau
        
    elif class_type == 'diseased_vegetation':
        # Больная растительность
        green_peak = 0.15 * np.exp(-((wavelengths - 550) / 60) ** 2)
        red_edge = 0.2 / (1 + np.exp(-(wavelengths - 720) / 30))
        nir_plateau = np.where((wavelengths >= 750) & (wavelengths <= 900), 0.3, 0)
        spectrum = 0.2 + green_peak + red_edge + nir_plateau
        
    elif class_type == 'soil':
        # Почва
        spectrum = 0.15 + 0.05 * np.sin((wavelengths - 400) / 200)
        
    else:  # background
        # Фон
        spectrum = 0.1 + 0.02 * np.random.random(len(wavelengths))
    
    return spectrum


def opencv_integration_example(data_path: str, output_dir: str):
    """Пример интеграции с OpenCV"""
    try:
        import cv2
        
        print("\n--- Интеграция с OpenCV ---")
        
        # Загрузка данных
        processor = HyperspectralProcessor()
        dataset, image_data, wavelengths = processor._read_hyperspectral_data(data_path)
        
        # Создание RGB композита для OpenCV
        blue_idx = np.argmin(np.abs(wavelengths - 450))
        green_idx = np.argmin(np.abs(wavelengths - 550))
        red_idx = np.argmin(np.abs(wavelengths - 650))
        
        rgb_image = np.stack([
            image_data[:, :, red_idx],
            image_data[:, :, green_idx],
            image_data[:, :, blue_idx]
        ], axis=2)
        
        # Нормализация для OpenCV
        rgb_normalized = np.zeros_like(rgb_image)
        for i in range(3):
            band = rgb_image[:, :, i]
            band_min, band_max = np.percentile(band, [2, 98])
            if band_max > band_min:
                rgb_normalized[:, :, i] = (band - band_min) / (band_max - band_min)
        
        # Конвертация в uint8 для OpenCV
        rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
        
        # Применение фильтров OpenCV
        # 1. Гауссово размытие
        blurred = cv2.GaussianBlur(rgb_uint8, (5, 5), 0)
        
        # 2. Детектор границ Canny
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 3. Морфологические операции
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 4. Сегментация с помощью k-means
        pixels = rgb_uint8.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 4
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(rgb_uint8.shape)
        
        # Визуализация результатов
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            axes[0, 0].imshow(rgb_uint8)
            axes[0, 0].set_title('Оригинал')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(blurred)
            axes[0, 1].set_title('Гауссово размытие')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(edges, cmap='gray')
            axes[0, 2].set_title('Детектор границ Canny')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(opened, cmap='gray')
            axes[1, 0].set_title('Морфологическое открытие')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(closed, cmap='gray')
            axes[1, 1].set_title('Морфологическое закрытие')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(segmented_image)
            axes[1, 2].set_title('K-means сегментация')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'plots', 'opencv_integration.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Результаты интеграции с OpenCV сохранены: {plot_path}")
            
        except ImportError:
            print("Matplotlib не доступен для визуализации")
        
        # Сохранение результатов обработки
        cv2.imwrite(os.path.join(output_dir, 'opencv_blurred.jpg'), blurred)
        cv2.imwrite(os.path.join(output_dir, 'opencv_edges.jpg'), edges)
        cv2.imwrite(os.path.join(output_dir, 'opencv_segmented.jpg'), segmented_image)
        
        print("Интеграция с OpenCV завершена успешно")
        
    except ImportError:
        print("OpenCV не установлен. Пропуск примера интеграции с OpenCV.")
    except Exception as e:
        print(f"Ошибка в интеграции с OpenCV: {e}")


def sklearn_integration_example(data_path: str, output_dir: str):
    """Пример интеграции с scikit-learn"""
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.preprocessing import StandardScaler
        
        print("\n--- Интеграция с scikit-learn ---")
        
        # Загрузка данных
        processor = HyperspectralProcessor()
        dataset, image_data, wavelengths = processor._read_hyperspectral_data(data_path)
        
        # Подготовка данных для машинного обучения
        height, width, bands = image_data.shape
        pixels = image_data.reshape(-1, bands)
        
        # Создание меток классов на основе спектральных характеристик
        labels = create_spectral_labels(image_data, wavelengths)
        
        # 1. PCA для уменьшения размерности
        print("Выполнение PCA...")
        pca = PCA(n_components=10)
        pca_pixels = pca.fit_transform(pixels)
        
        # Визуализация объясненной дисперсии
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(np.cumsum(pca.explained_variance_ratio_), 'bo-')
            plt.xlabel('Количество компонентов')
            plt.ylabel('Кумулятивная объясненная дисперсия')
            plt.title('Анализ главных компонентов')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_dir, 'plots', 'pca_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Анализ PCA сохранен: {plot_path}")
            
        except ImportError:
            print("Matplotlib не доступен для визуализации")
        
        # 2. Кластеризация с K-means
        print("Выполнение кластеризации K-means...")
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans_labels = kmeans.fit_predict(pca_pixels)
        
        # 3. Кластеризация с DBSCAN
        print("Выполнение кластеризации DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        dbscan_labels = dbscan.fit_predict(pca_pixels)
        
        # 4. Классификация с Random Forest
        print("Обучение классификатора Random Forest...")
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            pca_pixels, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Масштабирование данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение классификатора
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Оценка классификатора
        y_pred = rf_classifier.predict(X_test_scaled)
        
        # Сохранение отчета классификации
        report = classification_report(y_test, y_pred)
        report_path = os.path.join(output_dir, 'classification_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("ОТЧЕТ КЛАССИФИКАЦИИ RANDOM FOREST\n")
            f.write("="*50 + "\n\n")
            f.write(report)
            f.write(f"\n\nТочность на обучающей выборке: {rf_classifier.score(X_train_scaled, y_train):.3f}")
            f.write(f"\nТочность на тестовой выборке: {rf_classifier.score(X_test_scaled, y_test):.3f}")
        
        print(f"Отчет классификации сохранен: {report_path}")
        
        # Важность признаков
        feature_importance = rf_classifier.feature_importances_
        
        # Визуализация результатов кластеризации и классификации
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # K-means кластеризация
            kmeans_image = kmeans_labels.reshape(height, width)
            axes[0, 0].imshow(kmeans_image, cmap='tab10')
            axes[0, 0].set_title('K-means кластеризация')
            axes[0, 0].axis('off')
            
            # DBSCAN кластеризация
            dbscan_image = dbscan_labels.reshape(height, width)
            axes[0, 1].imshow(dbscan_image, cmap='tab10')
            axes[0, 1].set_title('DBSCAN кластеризация')
            axes[0, 1].axis('off')
            
            # Исходные метки
            labels_image = labels.reshape(height, width)
            axes[1, 0].imshow(labels_image, cmap='tab10')
            axes[1, 0].set_title('Исходные классы')
            axes[1, 0].axis('off')
            
            # Важность признаков
            axes[1, 1].bar(range(len(feature_importance)), feature_importance)
            axes[1, 1].set_title('Важность признаков (PCA компоненты)')
            axes[1, 1].set_xlabel('Компонент PCA')
            axes[1, 1].set_ylabel('Важность')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'plots', 'sklearn_integration.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Результаты интеграции с scikit-learn сохранены: {plot_path}")
            
        except ImportError:
            print("Matplotlib не доступен для визуализации")
        
        # Сохранение модели
        import joblib
        model_path = os.path.join(output_dir, 'models', 'random_forest_model.pkl')
        joblib.dump(rf_classifier, model_path)
        
        scaler_path = os.path.join(output_dir, 'models', 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        print(f"Модель сохранена: {model_path}")
        print("Интеграция с scikit-learn завершена успешно")
        
    except ImportError:
        print("scikit-learn не установлен. Пропуск примера интеграции с scikit-learn.")
    except Exception as e:
        print(f"Ошибка в интеграции с scikit-learn: {e}")


def pandas_integration_example(data_path: str, output_dir: str):
    """Пример интеграции с pandas"""
    try:
        import pandas as pd
        
        print("\n--- Интеграция с pandas ---")
        
        # Загрузка данных
        processor = HyperspectralProcessor()
        dataset, image_data, wavelengths = processor._read_hyperspectral_data(data_path)
        
        # Расчет вегетационных индексов
        calculator = VegetationIndexCalculator()
        
        # Извлечение каналов
        blue_idx = np.argmin(np.abs(wavelengths - 450))
        green_idx = np.argmin(np.abs(wavelengths - 550))
        red_idx = np.argmin(np.abs(wavelengths - 650))
        nir_idx = np.argmin(np.abs(wavelengths - 800))
        
        blue = image_data[:, :, blue_idx].flatten()
        green = image_data[:, :, green_idx].flatten()
        red = image_data[:, :, red_idx].flatten()
        nir = image_data[:, :, nir_idx].flatten()
        
        # Расчет индексов
        ndvi = (nir - red) / (nir + red + 1e-8)
        gndvi = (nir - green) / (nir + green + 1e-8)
        ndwi = (green - nir) / (green + nir + 1e-8)
        
        # Создание DataFrame
        df = pd.DataFrame({
            'Blue': blue,
            'Green': green,
            'Red': red,
            'NIR': nir,
            'NDVI': ndvi,
            'GNDVI': gndvi,
            'NDWI': ndwi,
            'X': np.repeat(np.arange(image_data.shape[1]), image_data.shape[0]),
            'Y': np.tile(np.arange(image_data.shape[0]), image_data.shape[1])
        })
        
        # Фильтрация валидных пикселей
        df_valid = df[(df['NDVI'] > -1) & (df['NDVI'] < 1)].copy()
        
        # 1. Статистический анализ
        print("Выполнение статистического анализа...")
        stats = df_valid[['NDVI', 'GNDVI', 'NDWI']].describe()
        
        # 2. Корреляционный анализ
        correlation_matrix = df_valid[['Blue', 'Green', 'Red', 'NIR', 'NDVI', 'GNDVI', 'NDWI']].corr()
        
        # 3. Группировка по пространственным областям
        df_valid['Region'] = pd.cut(df_valid['X'], bins=5, labels=['Запад', 'С-З', 'Центр', 'С-В', 'Восток'])
        regional_stats = df_valid.groupby('Region')[['NDVI', 'GNDVI', 'NDWI']].mean()
        
        # 4. Классификация пикселей по NDVI
        def classify_ndvi(ndvi_value):
            if ndvi_value < 0:
                return 'Вода/Снег'
            elif ndvi_value < 0.2:
                return 'Бедная растительность/Почва'
            elif ndvi_value < 0.5:
                return 'Умеренная растительность'
            else:
                return 'Плотная растительность'
        
        df_valid['Vegetation_Class'] = df_valid['NDVI'].apply(classify_ndvi)
        class_counts = df_valid['Vegetation_Class'].value_counts()
        
        # Сохранение результатов в Excel
        with pd.ExcelWriter(os.path.join(output_dir, 'pandas_analysis.xlsx')) as writer:
            stats.to_excel(writer, sheet_name='Статистика')
            correlation_matrix.to_excel(writer, sheet_name='Корреляции')
            regional_stats.to_excel(writer, sheet_name='Региональная статистика')
            class_counts.to_excel(writer, sheet_name='Классы растительности')
            df_valid.head(1000).to_excel(writer, sheet_name='Пример данных', index=False)
        
        # Визуализация результатов
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Настройка стиля
            sns.set_style("whitegrid")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Распределение NDVI
            sns.histplot(df_valid['NDVI'], bins=50, kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Распределение NDVI')
            axes[0, 0].set_xlabel('NDVI')
            axes[0, 0].set_ylabel('Частота')
            
            # Корреляционная матрица
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
            axes[0, 1].set_title('Корреляционная матрица')
            
            # Региональная статистика
            regional_stats.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Средние значения по регионам')
            axes[1, 0].set_ylabel('Среднее значение')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Классы растительности
            class_counts.plot(kind='pie', autopct='%1.1f%%', ax=axes[1, 1])
            axes[1, 1].set_title('Распределение классов растительности')
            axes[1, 1].set_ylabel('')
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'plots', 'pandas_integration.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Результаты интеграции с pandas сохранены: {plot_path}")
            
        except ImportError:
            print("Matplotlib/Seaborn не доступны для визуализации")
        
        # Сохранение DataFrame в CSV
        csv_path = os.path.join(output_dir, 'vegetation_indices.csv')
        df_valid.to_csv(csv_path, index=False)
        
        print(f"Данные сохранены в CSV: {csv_path}")
        print(f"Анализ сохранен в Excel: {output_dir}/pandas_analysis.xlsx")
        print("Интеграция с pandas завершена успешно")
        
    except ImportError:
        print("pandas не установлен. Пропуск примера интеграции с pandas.")
    except Exception as e:
        print(f"Ошибка в интеграции с pandas: {e}")


def scipy_integration_example(data_path: str, output_dir: str):
    """Пример интеграции с scipy"""
    try:
        from scipy import ndimage, signal, stats, spatial
        from scipy.interpolate import interp1d
        
        print("\n--- Интеграция с scipy ---")
        
        # Загрузка данных
        processor = HyperspectralProcessor()
        dataset, image_data, wavelengths = processor._read_hyperspectral_data(data_path)
        
        # 1. Пространственная фильтрация с scipy.ndimage
        print("Применение пространственной фильтрации...")
        
        # Выбор канала для демонстрации
        red_channel = image_data[:, :, np.argmin(np.abs(wavelengths - 650))]
        
        # Гауссов фильтр
        gaussian_filtered = ndimage.gaussian_filter(red_channel, sigma=2)
        
        # Медианный фильтр
        median_filtered = ndimage.median_filter(red_channel, size=3)
        
        # Фильтр Собеля для выделения границ
        sobel_filtered = ndimage.sobel(red_channel)
        
        # 2. Спектральная интерполяция
        print("Выполнение спектральной интерполяции...")
        
        # Создание новой сетки длин волн
        new_wavelengths = np.linspace(400, 1000, 200)
        
        # Интерполяция спектральных данных
        interpolated_data = np.zeros((image_data.shape[0], image_data.shape[1], len(new_wavelengths)))
        
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                spectrum = image_data[i, j, :]
                f = interp1d(wavelengths, spectrum, kind='cubic', fill_value='extrapolate')
                interpolated_data[i, j, :] = f(new_wavelengths)
        
        # 3. Статистический анализ с scipy.stats
        print("Выполнение статистического анализа...")
        
        # Расчет NDVI
        red_idx = np.argmin(np.abs(wavelengths - 650))
        nir_idx = np.argmin(np.abs(wavelengths - 800))
        
        ndvi = (image_data[:, :, nir_idx] - image_data[:, :, red_idx]) / \
               (image_data[:, :, nir_idx] + image_data[:, :, red_idx] + 1e-8)
        
        # Статистические тесты
        ndvi_flat = ndvi.flatten()
        ndvi_valid = ndvi_flat[~np.isnan(ndvi_flat) & (ndvi_flat > -1) & (ndvi_flat < 1)]
        
        # Тест на нормальность
        normality_test = stats.normaltest(ndvi_valid)
        
        # Описательная статистика
        descriptive_stats = stats.describe(ndvi_valid)
        
        # 4. Кластерный анализ с scipy.spatial
        print("Выполнение кластерного анализа...")
        
        # Выбор случайных пикселей для кластеризации
        n_samples = min(1000, len(ndvi_valid))
        sample_indices = np.random.choice(len(ndvi_valid), n_samples, replace=False)
        
        # Подготовка данных (спектральные признаки)
        sample_pixels = image_data.reshape(-1, image_data.shape[2])[sample_indices]
        
        # Расчет матрицы расстояний
        distance_matrix = spatial.distance.pdist(sample_pixels)
        
        # Иерархическая кластеризация
        from scipy.cluster.hierarchy import linkage, dendrogram
        
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Визуализация результатов
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Пространственная фильтрация
            axes[0, 0].imshow(red_channel, cmap='gray')
            axes[0, 0].set_title('Оригинал')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(gaussian_filtered, cmap='gray')
            axes[0, 1].set_title('Гауссов фильтр')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(sobel_filtered, cmap='gray')
            axes[0, 2].set_title('Фильтр Собеля')
            axes[0, 2].axis('off')
            
            # Спектральная интерполяция
            center_spectrum = image_data[75, 75, :]
            interpolated_spectrum = interpolated_data[75, 75, :]
            
            axes[1, 0].plot(wavelengths, center_spectrum, 'b-', label='Оригинал')
            axes[1, 0].plot(new_wavelengths, interpolated_spectrum, 'r--', label='Интерполяция')
            axes[1, 0].set_xlabel('Длина волны (нм)')
            axes[1, 0].set_ylabel('Отражение')
            axes[1, 0].set_title('Спектральная интерполяция')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Распределение NDVI
            axes[1, 1].hist(ndvi_valid, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title(f'Распределение NDVI\nТест нормальности: p={normality_test.pvalue:.4f}')
            axes[1, 1].set_xlabel('NDVI')
            axes[1, 1].set_ylabel('Частота')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Дендрограмма (ограниченная выборка)
            from scipy.cluster.hierarchy import dendrogram
            truncated = dendrogram(linkage_matrix, truncate_mode='lastp', p=12, ax=axes[1, 2])
            axes[1, 2].set_title('Иерархическая кластеризация')
            axes[1, 2].set_xlabel('Кластер')
            axes[1, 2].set_ylabel('Расстояние')
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'plots', 'scipy_integration.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Результаты интеграции с scipy сохранены: {plot_path}")
            
        except ImportError:
            print("Matplotlib не доступен для визуализации")
        
        # Сохранение статистического отчета
        report_path = os.path.join(output_dir, 'scipy_statistical_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("СТАТИСТИЧЕСКИЙ ОТЧЕТ SCIPY\n")
            f.write("="*40 + "\n\n")
            
            f.write("1. ТЕСТ НА НОРМАЛЬНОСТЬ NDVI\n")
            f.write("-" * 30 + "\n")
            f.write(f"Статистика: {normality_test.statistic:.4f}\n")
            f.write(f"P-значение: {normality_test.pvalue:.6f}\n")
            f.write(f"Нормальность: {'Да' if normality_test.pvalue > 0.05 else 'Нет'}\n\n")
            
            f.write("2. ОПИСАТЕЛЬНАЯ СТАТИСТИКА NDVI\n")
            f.write("-" * 30 + "\n")
            f.write(f"Количество наблюдений: {descriptive_stats.nobs}\n")
            f.write(f"Минимум: {descriptive_stats.minmax[0]:.4f}\n")
            f.write(f"Максимум: {descriptive_stats.minmax[1]:.4f}\n")
            f.write(f"Среднее: {descriptive_stats.mean:.4f}\n")
            f.write(f"Дисперсия: {descriptive_stats.variance:.6f}\n")
            f.write(f"Асимметрия: {descriptive_stats.skewness:.4f}\n")
            f.write(f"Эксцесс: {descriptive_stats.kurtosis:.4f}\n\n")
            
            f.write("3. СПЕКТРАЛЬНАЯ ИНТЕРПОЛЯЦИЯ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Исходных каналов: {len(wavelengths)}\n")
            f.write(f"Интерполированных каналов: {len(new_wavelengths)}\n")
            f.write(f"Спектральный диапазон: {min(wavelengths):.1f} - {max(wavelengths):.1f} нм\n")
        
        print(f"Статистический отчет сохранен: {report_path}")
        print("Интеграция с scipy завершена успешно")
        
    except ImportError:
        print("scipy не установлен. Пропуск примера интеграции с scipy.")
    except Exception as e:
        print(f"Ошибка в интеграции с scipy: {e}")


def comprehensive_integration_example(data_path: str, output_dir: str):
    """Комплексный пример интеграции всех библиотек"""
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import cv2
        
        print("\n--- Комплексная интеграция ---")
        
        # Загрузка и предобработка данных
        processor = HyperspectralProcessor()
        dataset, image_data, wavelengths = processor._read_hyperspectral_data(data_path)
        
        # 1. Предобработка с OpenCV
        # Создание RGB композита
        blue_idx = np.argmin(np.abs(wavelengths - 450))
        green_idx = np.argmin(np.abs(wavelengths - 550))
        red_idx = np.argmin(np.abs(wavelengths - 650))
        
        rgb_image = np.stack([
            image_data[:, :, red_idx],
            image_data[:, :, green_idx],
            image_data[:, :, blue_idx]
        ], axis=2)
        
        # Нормализация
        rgb_normalized = np.zeros_like(rgb_image)
        for i in range(3):
            band = rgb_image[:, :, i]
            band_min, band_max = np.percentile(band, [2, 98])
            if band_max > band_min:
                rgb_normalized[:, :, i] = (band - band_min) / (band_max - band_min)
        
        # Улучшение контраста с OpenCV
        rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
        enhanced = cv2.convertScaleAbs(rgb_uint8, alpha=1.2, beta=10)
        
        # 2. Извлечение признаков
        # Спектральные признаки
        spectral_features = image_data.reshape(-1, image_data.shape[2])
        
        # Текстурные признаки (упрощенные)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        
        # Вычисление локальных бинарных шаблонов (упрощенная версия)
        def compute_lbp(image, radius=1):
            lbp = np.zeros_like(image)
            for i in range(radius, image.shape[0] - radius):
                for j in range(radius, image.shape[1] - radius):
                    center = image[i, j]
                    code = 0
                    for k in range(8):
                        angle = 2 * np.pi * k / 8
                        x = i + radius * np.cos(angle)
                        y = j + radius * np.sin(angle)
                        x, y = int(x), int(y)
                        if image[x, y] >= center:
                            code |= (1 << k)
                    lbp[i, j] = code
            return lbp
        
        lbp = compute_lbp(gray)
        texture_features = lbp.reshape(-1, 1)
        
        # Вегетационные индексы
        nir = image_data[:, :, np.argmin(np.abs(wavelengths - 800))]
        red = image_data[:, :, red_idx]
        green = image_data[:, :, green_idx]
        
        ndvi = (nir - red) / (nir + red + 1e-8)
        gndvi = (nir - green) / (nir + green + 1e-8)
        
        index_features = np.column_stack([ndvi.flatten(), gndvi.flatten()])
        
        # Объединение всех признаков
        all_features = np.column_stack([
            spectral_features,
            texture_features,
            index_features
        ])
        
        # 3. Создание меток с использованием пороговых значений NDVI
        ndvi_flat = ndvi.flatten()
        labels = np.zeros_like(ndvi_flat, dtype=int)
        
        labels[ndvi_flat < 0.1] = 0  # Почва/негатив
        labels[(ndvi_flat >= 0.1) & (ndvi_flat < 0.3)] = 1  # Бедная растительность
        labels[(ndvi_flat >= 0.3) & (ndvi_flat < 0.6)] = 2  # Умеренная растительность
        labels[ndvi_flat >= 0.6] = 3  # Хорошая растительность
        
        # Фильтрация валидных данных
        valid_mask = ~np.isnan(all_features).any(axis=1) & (labels >= 0)
        features_valid = all_features[valid_mask]
        labels_valid = labels[valid_mask]
        
        # 4. Обучение модели с scikit-learn
        X_train, X_test, y_train, y_test = train_test_split(
            features_valid, labels_valid, test_size=0.3, random_state=42, stratify=labels_valid
        )
        
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        
        # Оценка модели
        train_accuracy = rf_classifier.score(X_train, y_train)
        test_accuracy = rf_classifier.score(X_test, y_test)
        
        # Предсказание для всего изображения
        full_predictions = rf_classifier.predict(all_features)
        prediction_map = full_predictions.reshape(image_data.shape[0], image_data.shape[1])
        
        # 5. Анализ результатов с pandas
        class_names = ['Почва', 'Бедная растительность', 'Умеренная растительность', 'Хорошая растительность']
        
        # Создание DataFrame с результатами
        results_df = pd.DataFrame({
            'X': np.repeat(np.arange(image_data.shape[1]), image_data.shape[0]),
            'Y': np.tile(np.arange(image_data.shape[0]), image_data.shape[1]),
            'NDVI': ndvi_flat,
            'GNDVI': gndvi.flatten(),
            'Predicted_Class': full_predictions,
            'Class_Name': [class_names[i] for i in full_predictions]
        })
        
        # Статистика по классам
        class_statistics = results_df.groupby('Class_Name').agg({
            'NDVI': ['mean', 'std', 'count'],
            'GNDVI': ['mean', 'std']
        }).round(4)
        
        # 6. Визуализация комплексных результатов
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Исходное изображение
            axes[0, 0].imshow(rgb_normalized)
            axes[0, 0].set_title('Исходное изображение')
            axes[0, 0].axis('off')
            
            # NDVI
            im1 = axes[0, 1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[0, 1].set_title('NDVI')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # Карта классификации
            im2 = axes[0, 2].imshow(prediction_map, cmap='tab10')
            axes[0, 2].set_title('Карта классификации')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # Распределение NDVI по классам
            for class_id, class_name in enumerate(class_names):
                class_data = results_df[results_df['Predicted_Class'] == class_id]['NDVI']
                axes[1, 0].hist(class_data, alpha=0.6, label=class_name, bins=20)
            
            axes[1, 0].set_title('Распределение NDVI по классам')
            axes[1, 0].set_xlabel('NDVI')
            axes[1, 0].set_ylabel('Частота')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Точность модели
            accuracy_data = {'Обучающая выборка': train_accuracy, 'Тестовая выборка': test_accuracy}
            axes[1, 1].bar(accuracy_data.keys(), accuracy_data.values())
            axes[1, 1].set_title('Точность классификации')
            axes[1, 1].set_ylabel('Точность')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Статистика по классам
            class_means = results_df.groupby('Class_Name')['NDVI'].mean()
            axes[1, 2].bar(class_means.index, class_means.values)
            axes[1, 2].set_title('Средний NDVI по классам')
            axes[1, 2].set_ylabel('Средний NDVI')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'plots', 'comprehensive_integration.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Комплексные результаты сохранены: {plot_path}")
            
        except ImportError:
            print("Matplotlib не доступен для визуализации")
        
        # Сохранение результатов
        results_df.to_csv(os.path.join(output_dir, 'comprehensive_results.csv'), index=False)
        class_statistics.to_excel(os.path.join(output_dir, 'class_statistics.xlsx'))
        
        # Сохранение модели
        import joblib
        joblib.dump(rf_classifier, os.path.join(output_dir, 'models', 'comprehensive_model.pkl'))
        
        # Сохранение отчета
        report_path = os.path.join(output_dir, 'comprehensive_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("КОМПЛЕКСНЫЙ ОТЧЕТ ИНТЕГРАЦИИ\n")
            f.write("="*40 + "\n\n")
            
            f.write("1. ИСПОЛЬЗОВАННЫЕ БИБЛИОТЕКИ\n")
            f.write("-" * 30 + "\n")
            f.write("- OpenCV: предобработка изображений\n")
            f.write("- scikit-learn: машинное обучение\n")
            f.write("- pandas: анализ данных\n")
            f.write("- scipy: научные вычисления\n\n")
            
            f.write("2. ХАРАКТЕРИСТИКИ МОДЕЛИ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Точность на обучающей выборке: {train_accuracy:.4f}\n")
            f.write(f"Точность на тестовой выборке: {test_accuracy:.4f}\n")
            f.write(f"Количество признаков: {all_features.shape[1]}\n")
            f.write(f"Количество классов: {len(class_names)}\n\n")
            
            f.write("3. СТАТИСТИКА ПО КЛАССАМ\n")
            f.write("-" * 30 + "\n")
            f.write(str(class_statistics))
        
        print(f"Комплексный отчет сохранен: {report_path}")
        print("Комплексная интеграция завершена успешно")
        
    except ImportError as e:
        print(f"Отсутствуют необходимые библиотеки: {e}")
    except Exception as e:
        print(f"Ошибка в комплексной интеграции: {e}")


def create_spectral_labels(image_data: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Создание меток классов на основе спектральных характеристик"""
    height, width, bands = image_data.shape
    labels = np.zeros(height * width, dtype=int)
    
    # Расчет NDVI для классификации
    red_idx = np.argmin(np.abs(wavelengths - 650))
    nir_idx = np.argmin(np.abs(wavelengths - 800))
    
    ndvi = (image_data[:, :, nir_idx] - image_data[:, :, red_idx]) / \
           (image_data[:, :, nir_idx] + image_data[:, :, red_idx] + 1e-8)
    
    ndvi_flat = ndvi.flatten()
    
    # Классификация на основе NDVI
    labels[ndvi_flat < 0.1] = 0  # Почва
    labels[(ndvi_flat >= 0.1) & (ndvi_flat < 0.3)] = 1  # Бедная растительность
    labels[(ndvi_flat >= 0.3) & (ndvi_flat < 0.6)] = 2  # Умеренная растительность
    labels[ndvi_flat >= 0.6] = 3  # Хорошая растительность
    
    return labels


if __name__ == '__main__':
    sys.exit(main())