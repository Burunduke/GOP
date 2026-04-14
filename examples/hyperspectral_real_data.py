#!/usr/bin/env python3
"""
Hyperspectral Data Loading Example
using GOP Library v2.0

This example demonstrates:
- Loading hyperspectral data
- Creating orthophotos
- Basic data visualization
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.pipeline import Pipeline
from src.processing.hyperspectral import HyperspectralProcessor
from src.utils.logger import setup_logger


def create_sample_data(file_path):
    """Create synthetic hyperspectral data for demonstration"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create synthetic hyperspectral data (256x256 pixels, 224 bands)
        data = np.random.rand(256, 256, 224).astype(np.float32)
        
        # Add realistic spectral signature patterns
        wavelengths = np.linspace(400, 2500, 224)
        
        # Simulate vegetation spectral signature
        for i, wl in enumerate(wavelengths):
            if 500 <= wl <= 700:  # Photosynthetic active radiation
                data[:, :, i] *= 0.1 + 0.2 * np.sin(wl/100)
            elif 700 <= wl <= 1300:  # Near-infrared plateau
                data[:, :, i] *= 0.4 + 0.3 * np.cos(wl/200)
            elif 1300 <= wl <= 2500:  # Short-wave infrared
                data[:, :, i] *= 0.05 + 0.1 * np.sin(wl/300)
        
        # Save as binary file
        data.tofile(file_path)
        
        # Create header file
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
        
        print(f"Created sample data: {file_path}")
        print(f"Created header file: {header_file}")
        
    except Exception as e:
        print(f"Error creating sample data: {e}")


def plot_orthophoto_preview(data, output_dir):
    """Create a preview of the orthophoto data"""
    try:
        if data is None:
            print("No data available for plotting")
            return
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create a simple RGB preview (using first 3 bands)
        if len(data.shape) >= 3 and data.shape[2] >= 3:
            rgb_preview = data[:, :, :3]
            
            # Normalize for display
            rgb_preview = (rgb_preview - np.min(rgb_preview)) / (np.max(rgb_preview) - np.min(rgb_preview))
            
            plt.figure(figsize=(10, 8))
            plt.imshow(rgb_preview)
            plt.title('Orthophoto Preview (RGB Composite)')
            plt.colorbar(label='Normalized Reflectance')
            plt.axis('off')
            
            # Save plot
            plt.savefig(os.path.join(plots_dir, 'orthophoto_preview.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Orthophoto preview saved: {plots_dir}/orthophoto_preview.png")
            
    except Exception as e:
        print(f"Error creating orthophoto preview: {e}")


def main():
    """Main function for hyperspectral data loading example"""
    
    # Setup logging
    logger = setup_logger('GOP_RealData', level=logging.INFO)
    logger.info("Starting hyperspectral data loading example")
    
    try:
        # Path to input data (replace with your path)
        input_path = "data/real_hyperspectral_data.bil"
        output_dir = "results/hyperspectral_real_data"
        
        # Check if input data exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            logger.info("Creating sample data for demonstration")
            create_sample_data(input_path)
            logger.info(f"Sample data created: {input_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Initialize pipeline
        logger.info("Initializing pipeline")
        pipeline = Pipeline()
        
        # Process data to create orthophoto
        logger.info(f"Processing file: {input_path}")
        results = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            sensor_type='Hyperspectral',
            compression_ratio=0.125
        )
        
        # Create orthophoto preview
        logger.info("Generating orthophoto preview")
        if 'processed_data' in results and results['processed_data'] is not None:
            plot_orthophoto_preview(results['processed_data'], output_dir)
        
        # Save results
        logger.info("Saving results")
        results_file = os.path.join(output_dir, 'processing_results.json')
        pipeline.save_results(results_file)
        
        # Display results
        print("\n" + "="*60)
        print("DATA LOADING RESULTS")
        print("="*60)
        
        print(f"Input file: {results['input_path']}")
        print(f"Orthophoto: {results['orthophoto_path']}")
        print(f"Output directory: {output_dir}")
        print(f"Sensor type: {results['sensor_type']}")
        print(f"Data size: {results['processed_data']['shape']}")
        print(f"Number of bands: {results['processed_data']['bands']}")
        
        print("\n" + "="*60)
        print("Data loading example completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in data loading example: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())