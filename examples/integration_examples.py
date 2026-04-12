#!/usr/bin/env python3
"""
Integration Examples with Other Scientific Libraries
Demonstrating GOP integration with OpenCV, scikit-learn, pandas, and other libraries

This example shows:
- Integration with OpenCV for image processing
- Using scikit-learn for machine learning
- Working with pandas for data analysis
- Integration with matplotlib/seaborn for visualization
- Using scipy for scientific computing
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.pipeline import Pipeline
from src.processing.hyperspectral import HyperspectralProcessor
from src.indices.calculator import VegetationIndexCalculator
from src.segmentation.segmenter import ImageSegmenter
from src.utils.logger import setup_logger


def create_integration_sample_data(file_path):
    """Create sample data for integration examples"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create synthetic hyperspectral data
        data = np.random.rand(256, 256, 224).astype(np.float32)
        wavelengths = np.linspace(400, 2500, 224)
        
        # Add realistic patterns
        for i, wl in enumerate(wavelengths):
            if 500 <= wl <= 700:
                data[:, :, i] *= 0.1 + 0.2 * np.sin(wl/100)
            elif 700 <= wl <= 1300:
                data[:, :, i] *= 0.4 + 0.3 * np.cos(wl/200)
            elif 1300 <= wl <= 2500:
                data[:, :, i] *= 0.05 + 0.1 * np.sin(wl/300)
        
        data.tofile(file_path)
        
        # Create header
        header_file = file_path.replace('.bil', '.hdr')
        with open(header_file, 'w') as f:
            f.write("ENVI\n")
            f.write("samples = 256\n")
            f.write("lines = 256\n")
            f.write("bands = 224\n")
            f.write("header offset = 0\n")
            f.write("file type = ENVI Standard\n")
            f.write("data type = 4\n")
            f.write("interleave = bil\n")
            f.write("byte order = 0\n")
            f.write("wavelength = {}".format(",".join(map(str, wavelengths))))
        
        print(f"Created integration sample data: {file_path}")
        
    except Exception as e:
        print(f"Error creating sample data: {e}")


def opencv_integration_example(data_path, output_dir):
    """Example of integrating with OpenCV"""
    try:
        import cv2
        
        print("\nOpenCV Integration Example")
        print("-" * 40)
        
        # Load data using GOP
        processor = HyperspectralProcessor()
        data = processor.load_data(data_path)
        
        if data is None:
            print("Failed to load data")
            return
        
        # Extract RGB bands for OpenCV processing
        # Assuming bands 29, 19, 9 correspond to R, G, B
        rgb_data = data[:, :, [29, 19, 9]]
        rgb_normalized = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min())
        rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
        
        # Apply OpenCV operations
        # Gaussian blur
        blurred = cv2.GaussianBlur(rgb_uint8, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Save results
        cv2.imwrite(os.path.join(output_dir, 'opencv_rgb.png'), rgb_uint8)
        cv2.imwrite(os.path.join(output_dir, 'opencv_blurred.png'), blurred)
        cv2.imwrite(os.path.join(output_dir, 'opencv_edges.png'), edges)
        
        print("OpenCV processing completed")
        print("Saved: opencv_rgb.png, opencv_blurred.png, opencv_edges.png")
        
    except ImportError:
        print("OpenCV not available. Install with: pip install opencv-python")
    except Exception as e:
        print(f"Error in OpenCV integration: {e}")


def sklearn_integration_example(data_path, output_dir):
    """Example of integrating with scikit-learn"""
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        print("\nscikit-learn Integration Example")
        print("-" * 40)
        
        # Load data using GOP
        processor = HyperspectralProcessor()
        data = processor.load_data(data_path)
        
        if data is None:
            print("Failed to load data")
            return
        
        # Reshape data for machine learning
        height, width, bands = data.shape
        X = data.reshape(-1, bands)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        
        # Reshape back to image
        clustered_image = labels.reshape(height, width)
        
        # Save results
        np.save(os.path.join(output_dir, 'sklearn_pca.npy'), X_pca)
        np.save(os.path.join(output_dir, 'sklearn_clusters.npy'), clustered_image)
        
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
        print("scikit-learn processing completed")
        
    except ImportError:
        print("scikit-learn not available. Install with: pip install scikit-learn")
    except Exception as e:
        print(f"Error in scikit-learn integration: {e}")


def pandas_integration_example(data_path, output_dir):
    """Example of integrating with pandas"""
    try:
        import pandas as pd
        
        print("\npandas Integration Example")
        print("-" * 40)
        
        # Calculate vegetation indices using GOP
        calculator = VegetationIndexCalculator()
        processor = HyperspectralProcessor()
        data = processor.load_data(data_path)
        
        if data is None:
            print("Failed to load data")
            return
        
        indices = calculator.calculate_indices(
            data=data,
            indices=['NDVI', 'GNDVI', 'EVI', 'NDWI']
        )
        
        # Create pandas DataFrame with index statistics
        stats_data = []
        for index_name, index_data in indices.items():
            if index_data is not None:
                stats_data.append({
                    'Index': index_name,
                    'Mean': np.mean(index_data),
                    'Std': np.std(index_data),
                    'Min': np.min(index_data),
                    'Max': np.max(index_data),
                    'Median': np.median(index_data)
                })
        
        df = pd.DataFrame(stats_data)
        
        # Perform additional analysis
        df['Range'] = df['Max'] - df['Min']
        df['CV'] = df['Std'] / df['Mean']  # Coefficient of variation
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'vegetation_indices_stats.csv')
        df.to_csv(csv_path, index=False)
        
        print("Vegetation index statistics:")
        print(df)
        print(f"\nSaved to: {csv_path}")
        
    except ImportError:
        print("pandas not available. Install with: pip install pandas")
    except Exception as e:
        print(f"Error in pandas integration: {e}")


def scipy_integration_example(data_path, output_dir):
    """Example of integrating with scipy"""
    try:
        from scipy import stats
        from scipy.ndimage import gaussian_filter
        from scipy.signal import savgol_filter
        
        print("\nSciPy Integration Example")
        print("-" * 40)
        
        # Load data using GOP
        processor = HyperspectralProcessor()
        data = processor.load_data(data_path)
        
        if data is None:
            print("Failed to load data")
            return
        
        # Extract spectral signature from a pixel
        spectral_signature = data[100, 100, :]  # Center pixel
        
        # Apply Savitzky-Golay filter for spectral smoothing
        smoothed_spectrum = savgol_filter(spectral_signature, window_length=11, polyorder=3)
        
        # Apply Gaussian filter for spatial smoothing
        spatial_smoothed = gaussian_filter(data[:, :, 100], sigma=1.0)
        
        # Statistical analysis
        mean_spectrum = np.mean(data, axis=(0, 1))
        skewness = stats.skew(mean_spectrum)
        kurtosis = stats.kurtosis(mean_spectrum)
        
        # Save results
        np.save(os.path.join(output_dir, 'scipy_smoothed_spectrum.npy'), smoothed_spectrum)
        np.save(os.path.join(output_dir, 'scipy_spatial_smoothed.npy'), spatial_smoothed)
        
        print(f"Spectral skewness: {skewness:.3f}")
        print(f"Spectral kurtosis: {kurtosis:.3f}")
        print("SciPy processing completed")
        
    except ImportError:
        print("SciPy not available. Install with: pip install scipy")
    except Exception as e:
        print(f"Error in SciPy integration: {e}")


def main():
    """Main function for integration examples"""
    
    # Setup logging
    logger = setup_logger('GOP_Integration', level=logging.INFO)
    logger.info("Starting integration examples with other libraries")
    
    try:
        # Create output directory
        output_dir = "results/integration_examples"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        
        # Create sample data
        data_path = "data/integration_sample.bil"
        if not os.path.exists(data_path):
            create_integration_sample_data(data_path)
        
        # Run integration examples
        print("="*60)
        print("INTEGRATION EXAMPLES")
        print("="*60)
        
        # Example 1: OpenCV integration
        opencv_integration_example(data_path, output_dir)
        
        # Example 2: scikit-learn integration
        sklearn_integration_example(data_path, output_dir)
        
        # Example 3: pandas integration
        pandas_integration_example(data_path, output_dir)
        
        # Example 4: SciPy integration
        scipy_integration_example(data_path, output_dir)
        
        print("\n" + "="*60)
        print("Integration examples completed successfully!")
        print("="*60)
        
        print("\nSummary of integrations demonstrated:")
        print("- OpenCV: Image processing and edge detection")
        print("- scikit-learn: PCA and clustering")
        print("- pandas: Data analysis and statistics")
        print("- SciPy: Signal processing and statistical analysis")
        
    except Exception as e:
        logger.error(f"Error in integration examples: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())