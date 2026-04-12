#!/usr/bin/env python3
"""
Orthophoto Analysis and Vegetation Index Calculation Example
using GOP Scientific Library v2.0

This example demonstrates:
- Loading and analyzing orthophotos
- Calculating various vegetation indices
- Spatial analysis of indices
- Visualization of results
- Plant condition classification

Note: Orthophoto generation is not fully implemented in this version.
This example works with pre-existing orthophoto data.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.pipeline import Pipeline
from src.processing.orthophoto import OrthophotoProcessor
from src.indices.calculator import VegetationIndexCalculator
from src.segmentation.segmenter import ImageSegmenter
from src.utils.logger import setup_logger


def create_sample_orthophoto(file_path):
    """Create a sample orthophoto for demonstration"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create a synthetic RGB orthophoto (512x512 pixels)
        orthophoto = np.random.rand(512, 512, 3).astype(np.float32)
        
        # Add realistic patterns (vegetation, soil, water)
        # Vegetation areas (green)
        orthophoto[100:300, 100:300, 1] *= 1.5  # Enhance green channel
        
        # Soil areas (brown)
        orthophoto[350:450, 50:150, 0] *= 1.2  # Enhance red channel
        orthophoto[350:450, 50:150, 2] *= 0.8  # Reduce blue channel
        
        # Water areas (blue)
        orthophoto[50:150, 350:450, 2] *= 1.5  # Enhance blue channel
        
        # Normalize to 0-255 range for image
        orthophoto = np.clip(orthophoto * 255, 0, 255).astype(np.uint8)
        
        # Save as TIFF (requires PIL)
        try:
            from PIL import Image
            img = Image.fromarray(orthophoto)
            img.save(file_path)
            print(f"Created sample orthophoto: {file_path}")
        except ImportError:
            # Fallback: save as numpy array
            np.save(file_path.replace('.tif', '.npy'), orthophoto)
            print(f"Created sample orthophoto (numpy format): {file_path.replace('.tif', '.npy')}")
            
    except Exception as e:
        print(f"Error creating sample orthophoto: {e}")


def analyze_orthophoto(orthophoto_path, output_dir):
    """Analyze orthophoto and calculate vegetation indices"""
    try:
        # Initialize components
        processor = OrthophotoProcessor()
        calculator = VegetationIndexCalculator()
        segmenter = ImageSegmenter()
        
        # Load orthophoto
        print(f"Loading orthophoto: {orthophoto_path}")
        orthophoto = processor.load_orthophoto(orthophoto_path)
        
        if orthophoto is None:
            print("Failed to load orthophoto")
            return None
        
        # Convert to appropriate format if needed
        if hasattr(processor, 'convert_to_float'):
            orthophoto = processor.convert_to_float(orthophoto)
        
        # Calculate vegetation indices
        print("Calculating vegetation indices...")
        indices = calculator.calculate_indices(
            data=orthophoto,
            indices=['NDVI', 'GNDVI', 'EVI', 'SAVI', 'NDWI']
        )
        
        # Perform segmentation
        print("Performing image segmentation...")
        segmentation_result = segmenter.segment_image(orthophoto)
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate visualizations
        generate_visualizations(orthophoto, indices, segmentation_result, plots_dir)
        
        return {
            'orthophoto': orthophoto,
            'indices': indices,
            'segmentation': segmentation_result
        }
        
    except Exception as e:
        print(f"Error analyzing orthophoto: {e}")
        return None


def generate_visualizations(orthophoto, indices, segmentation, plots_dir):
    """Generate visualization plots for orthophoto analysis"""
    try:
        # Plot 1: Original orthophoto
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        if orthophoto.shape[-1] == 3:  # RGB image
            plt.imshow(orthophoto)
        else:
            plt.imshow(orthophoto, cmap='gray')
        plt.title('Original Orthophoto')
        plt.axis('off')
        
        # Plot 2-4: Vegetation indices
        for i, (index_name, index_data) in enumerate(list(indices.items())[:3]):
            plt.subplot(2, 3, i + 2)
            if index_data is not None:
                plt.imshow(index_data, cmap='viridis')
                plt.title(f'{index_name}')
                plt.colorbar()
                plt.axis('off')
        
        # Plot 5: Segmentation result
        plt.subplot(2, 3, 5)
        if segmentation is not None:
            plt.imshow(segmentation, cmap='tab10')
            plt.title('Segmentation Result')
            plt.colorbar()
            plt.axis('off')
        
        # Plot 6: Index statistics
        plt.subplot(2, 3, 6)
        index_stats = []
        index_names = []
        for index_name, index_data in indices.items():
            if index_data is not None:
                index_stats.append(np.mean(index_data))
                index_names.append(index_name)
        
        if index_stats:
            plt.bar(range(len(index_stats)), index_stats)
            plt.xticks(range(len(index_stats)), index_names, rotation=45)
            plt.title('Average Index Values')
            plt.ylabel('Mean Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'orthophoto_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plot saved: {plots_dir}/orthophoto_analysis.png")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")


def main():
    """Main function for orthophoto analysis example"""
    
    # Setup logging
    logger = setup_logger('GOP_Orthophoto', level=logging.INFO)
    logger.info("Starting orthophoto analysis and vegetation index calculation example")
    
    try:
        # Path to orthophoto data
        orthophoto_path = "data/sample_orthophoto.tif"
        output_dir = "results/orthophoto_analysis"
        
        # Check if orthophoto exists
        if not os.path.exists(orthophoto_path):
            logger.error(f"Orthophoto not found: {orthophoto_path}")
            logger.info("Creating sample orthophoto")
            create_sample_orthophoto(orthophoto_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze orthophoto
        logger.info("Starting orthophoto analysis")
        results = analyze_orthophoto(orthophoto_path, output_dir)
        
        if results is None:
            logger.error("Orthophoto analysis failed")
            return 1
        
        # Display results
        print("\n" + "="*60)
        print("ORTHOPHOTO ANALYSIS RESULTS")
        print("="*60)
        
        print(f"Orthophoto file: {orthophoto_path}")
        print(f"Output directory: {output_dir}")
        print(f"Orthophoto shape: {results['orthophoto'].shape}")
        print(f"Calculated indices: {list(results['indices'].keys())}")
        
        # Display index statistics
        for index_name, index_data in results['indices'].items():
            if index_data is not None:
                print(f"\n{index_name}:")
                print(f"  Min: {np.min(index_data):.3f}")
                print(f"  Max: {np.max(index_data):.3f}")
                print(f"  Mean: {np.mean(index_data):.3f}")
                print(f"  Std: {np.std(index_data):.3f}")
        
        # Note about ODM integration
        print("\n" + "-"*60)
        print("NOTE: Orthophoto generation using OpenDroneMap (ODM)")
        print("is not fully implemented in this version.")
        print("This example works with pre-existing orthophoto data.")
        print("For ODM integration, additional setup is required.")
        print("-"*60)
        
        print("\n" + "="*60)
        print("Orthophoto analysis example completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in orthophoto analysis example: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())