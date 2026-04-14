#!/usr/bin/env python3
"""
Basic Hyperspectral Data Processing Example
using GOP Library v2.0

This example demonstrates:
- Loading hyperspectral data
- Creating orthophotos
- Saving results
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.pipeline import Pipeline
from src.core.config import get_config
from src.utils.logger import setup_logger


def main():
    """Main function for basic processing example"""
    
    # Setup logging
    logger = setup_logger('GOP_Example', level=logging.INFO)
    logger.info("Starting basic hyperspectral data processing example")
    
    try:
        # Path to input data (replace with your path)
        input_path = "data/sample_field.bil"
        output_dir = "results/basic_processing"
        
        # Check if input data exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            logger.info("Please provide a valid path to hyperspectral data")
            logger.info("You can download sample data or create synthetic data")
            logger.info("For real data, use formats like .bil, .hdr, .tif")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize pipeline
        logger.info("Initializing scientific pipeline")
        pipeline = Pipeline()
        
        # Process data to create orthophoto
        logger.info(f"Processing file: {input_path}")
        results = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            sensor_type='Hyperspectral',
            compression_ratio=0.125
        )
        
        # Display results
        logger.info("Processing completed successfully")
        print("\n" + "="*60)
        print("PROCESSING RESULTS")
        print("="*60)
        
        print(f"Input file: {results['input_path']}")
        print(f"Orthophoto: {results['orthophoto_path']}")
        print(f"Sensor type: {results['sensor_type']}")
        print(f"Data size: {results['processed_data']['shape']}")
        print(f"Number of bands: {results['processed_data']['bands']}")
        
        
        # Save results
        results_file = os.path.join(output_dir, 'processing_results.json')
        pipeline.save_results(results_file)
        print(f"\nResults saved: {results_file}")
        
        
        print("\n" + "="*60)
        print("Basic example completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in basic example: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())