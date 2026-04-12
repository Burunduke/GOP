#!/usr/bin/env python3
"""
Advanced Hyperspectral Data Processing Example
using GOP Scientific Library v2.0

This example demonstrates:
- Loading and preprocessing real hyperspectral data
- Applying various correction methods
- Spectral analysis and visualization
- Calculating specialized indices
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
from src.indices.calculator import VegetationIndexCalculator
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


def plot_spectral_signatures(processor, output_dir):
    """Plot spectral signatures from processed data"""
    try:
        # Get spectral data
        spectral_data = processor.get_spectral_data()
        if spectral_data is None:
            print("No spectral data available for plotting")
            return
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot average spectral signature
        plt.figure(figsize=(12, 8))
        
        if hasattr(spectral_data, 'wavelengths') and hasattr(spectral_data, 'reflectance'):
            wavelengths = spectral_data.wavelengths
            reflectance = spectral_data.reflectance
            
            # Plot average spectrum
            avg_spectrum = np.mean(reflectance, axis=(0, 1))
            plt.plot(wavelengths, avg_spectrum, 'b-', linewidth=2, label='Average Spectrum')
            
            # Highlight key spectral regions
            plt.axvspan(400, 700, alpha=0.2, color='green', label='Visible (400-700nm)')
            plt.axvspan(700, 1300, alpha=0.2, color='red', label='NIR (700-1300nm)')
            plt.axvspan(1300, 2500, alpha=0.2, color='orange', label='SWIR (1300-2500nm)')
            
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Reflectance')
            plt.title('Average Spectral Signature')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(plots_dir, 'spectral_signature.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Spectral signature plot saved: {plots_dir}/spectral_signature.png")
            
    except Exception as e:
        print(f"Error plotting spectral signatures: {e}")


def main():
    """Main function for advanced hyperspectral processing example"""
    
    # Setup logging
    logger = setup_logger('GOP_RealData', level=logging.INFO)
    logger.info("Starting advanced hyperspectral data processing example")
    
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
        
        # Initialize components
        logger.info("Initializing hyperspectral processor")
        processor = HyperspectralProcessor()
        calculator = VegetationIndexCalculator()
        
        # Load and preprocess data
        logger.info(f"Loading data from: {input_path}")
        
        # Apply atmospheric corrections
        logger.info("Applying atmospheric corrections")
        corrected_data = processor.apply_atmospheric_correction(
            input_path=input_path,
            method='empirical_line'
        )
        
        # Apply denoising
        logger.info("Applying denoising")
        denoised_data = processor.apply_denoising(
            data=corrected_data,
            method='wavelet'
        )
        
        # Calculate vegetation indices
        logger.info("Calculating vegetation indices")
        indices = calculator.calculate_indices(
            data=denoised_data,
            indices=['NDVI', 'GNDVI', 'EVI', 'SAVI', 'MSAVI', 'NDWI', 'MSI']
        )
        
        # Plot spectral signatures
        logger.info("Generating spectral plots")
        plot_spectral_signatures(processor, output_dir)
        
        # Save processed data
        logger.info("Saving processed data")
        processor.save_processed_data(
            data=denoised_data,
            output_path=os.path.join(output_dir, 'processed_data.npy')
        )
        
        # Save indices
        for index_name, index_data in indices.items():
            np.save(os.path.join(output_dir, f'{index_name}.npy'), index_data)
        
        # Display results
        print("\n" + "="*60)
        print("ADVANCED PROCESSING RESULTS")
        print("="*60)
        
        print(f"Input file: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"Processed data shape: {denoised_data.shape if denoised_data is not None else 'N/A'}")
        print(f"Calculated indices: {list(indices.keys())}")
        
        # Display index statistics
        for index_name, index_data in indices.items():
            if index_data is not None:
                print(f"\n{index_name}:")
                print(f"  Min: {np.min(index_data):.3f}")
                print(f"  Max: {np.max(index_data):.3f}")
                print(f"  Mean: {np.mean(index_data):.3f}")
                print(f"  Std: {np.std(index_data):.3f}")
        
        print("\n" + "="*60)
        print("Advanced processing example completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in advanced processing example: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())