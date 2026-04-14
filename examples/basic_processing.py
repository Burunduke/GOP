#!/usr/bin/env python3
"""
Basic Hyperspectral Data Processing Example
using GOP Scientific Library v2.0

This example demonstrates:
- Loading hyperspectral data
- Applying basic corrections
- Calculating vegetation indices
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
        
        # Process data with scientific analysis
        logger.info(f"Processing file: {input_path}")
        results = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            sensor_type='Hyperspectral',
            selected_indices=['GNDVI', 'MCARI', 'NDWI', 'MSI', 'SIPI2'],
            use_refinement=True,
            compression_ratio=0.125
        )
        
        # Display results
        logger.info("Processing completed successfully")
        print("\n" + "="*60)
        print("PROCESSING RESULTS")
        print("="*60)
        
        print(f"Input file: {results['input_path']}")
        print(f"Orthophoto: {results['orthophoto_path']}")
        print(f"Segmentation mask: {results['segmentation_mask']}")
        print(f"Sensor type: {results['sensor_type']}")
        print(f"Data size: {results['processed_data']['shape']}")
        print(f"Number of bands: {results['processed_data']['bands']}")
        
        # Plant condition analysis
        plant_condition = results.get('plant_condition', {})
        if 'classification' in plant_condition:
            classification = plant_condition['classification']
            print(f"\nPLANT CONDITION:")
            print(f"  Class: {classification['class']}")
            print(f"  Description: {classification['description']}")
            print(f"  Score: {classification['overall_score']:.3f}")
            print(f"  Confidence: {classification['confidence']:.2f}")
        
        # Scientific analysis
        scientific_analysis = results.get('scientific_analysis', {})
        if scientific_analysis:
            print(f"\nSCIENTIFIC ANALYSIS:")
            
            # Index statistics
            if 'index_statistics' in scientific_analysis:
                stats = scientific_analysis['index_statistics']
                print(f"  Calculated indices: {len(stats)}")
                for index_name, index_stats in list(stats.items())[:3]:
                    print(f"    {index_name}: mean={index_stats['mean']:.3f}, std={index_stats['std']:.3f}")
            
            # Correlation analysis
            if 'correlation_analysis' in scientific_analysis:
                corr_analysis = scientific_analysis['correlation_analysis']
                if 'strong_correlations' in corr_analysis:
                    strong_corr = corr_analysis['strong_correlations']
                    print(f"  Strong correlations: {len(strong_corr)}")
                    for corr in strong_corr[:3]:
                        print(f"    {corr['index1']} - {corr['index2']}: {corr['correlation']:.3f}")
            
            # Spatial analysis
            if 'spatial_analysis' in scientific_analysis:
                spatial = scientific_analysis['spatial_analysis']
                if 'overall' in spatial:
                    overall_spatial = spatial['overall']
                    print(f"  Spatial autocorrelation: {overall_spatial.get('spatial_autocorrelation', 0):.3f}")
        
        # Save results
        results_file = os.path.join(output_dir, 'processing_results.json')
        pipeline.save_results(results_file)
        print(f"\nResults saved: {results_file}")
        
        # Export scientific data
        pipeline.export_scientific_data(output_dir)
        print(f"Scientific data exported: {output_dir}/scientific_export/")
        
        # Data quality
        data_quality = results['processed_data'].get('data_quality', {})
        if data_quality and 'overall_quality' in data_quality:
            quality = data_quality['overall_quality']
            print(f"\nDATA QUALITY:")
            print(f"  Overall score: {quality.get('quality_score', 0):.3f}")
            print(f"  Average SNR: {quality.get('average_snr', 0):.2f}")
        
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