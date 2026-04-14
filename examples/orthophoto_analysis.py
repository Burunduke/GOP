#!/usr/bin/env python3
"""
Orthophoto Creation Example
using GOP Library v2.0

This example demonstrates:
- Loading hyperspectral data
- Creating orthophotos
- Basic visualization

Note: Orthophoto generation is not fully implemented in this version.
This example works with pre-existing orthophoto data.
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
from src.processing.orthophoto import OrthophotoProcessor
from src.utils.logger import setup_logger




def create_orthophoto(input_path, output_dir):
    """Create orthophoto from hyperspectral data"""
    try:
        # Initialize pipeline
        pipeline = Pipeline()
        
        # Process data to create orthophoto
        print(f"Processing file: {input_path}")
        results = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            sensor_type='Hyperspectral',
            compression_ratio=0.125
        )
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate orthophoto preview
        generate_orthophoto_preview(results, plots_dir)
        
        return results
        
    except Exception as e:
        print(f"Error creating orthophoto: {e}")
        return None


def generate_orthophoto_preview(results, plots_dir):
    """Generate orthophoto preview visualization"""
    try:
        if 'processed_data' not in results or results['processed_data'] is None:
            print("No processed data available for preview")
            return
        
        data = results['processed_data']
        
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
        print(f"Error generating orthophoto preview: {e}")


def main():
    """Main function for orthophoto creation example"""
    
    # Setup logging
    logger = setup_logger('GOP_Orthophoto', level=logging.INFO)
    logger.info("Starting orthophoto creation example")
    
    try:
        # Path to input data
        input_path = "data/sample_field.bil"
        output_dir = "results/orthophoto_creation"
        
        # Check if input data exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            logger.info("Please provide a valid path to hyperspectral data")
            return 1
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create orthophoto
        logger.info("Starting orthophoto creation")
        results = create_orthophoto(input_path, output_dir)
        
        if results is None:
            logger.error("Orthophoto analysis failed")
            return 1
        
        # Display results
        print("\n" + "="*60)
        print("ORTHOPHOTO CREATION RESULTS")
        print("="*60)
        
        print(f"Input file: {results['input_path']}")
        print(f"Orthophoto: {results['orthophoto_path']}")
        print(f"Output directory: {output_dir}")
        print(f"Sensor type: {results['sensor_type']}")
        print(f"Data size: {results['processed_data']['shape']}")
        print(f"Number of bands: {results['processed_data']['bands']}")
        
        # Note about ODM integration
        print("\n" + "-"*60)
        print("NOTE: Orthophoto generation using OpenDroneMap (ODM)")
        print("is not fully implemented in this version.")
        print("This example works with pre-existing orthophoto data.")
        print("For ODM integration, additional setup is required.")
        print("-"*60)
        
        print("\n" + "="*60)
        print("Orthophoto creation example completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in orthophoto creation example: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())