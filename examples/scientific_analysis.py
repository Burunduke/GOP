#!/usr/bin/env python3
"""
Scientific Analysis Pipeline Example
using GOP Scientific Library v2.0

This example demonstrates:
- Full scientific pipeline from src/core/pipeline.py
- Statistical analysis of hyperspectral data
- Correlation analysis between vegetation indices
- Plant condition assessment
- Advanced data quality metrics
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
from src.indices.calculator import VegetationIndexCalculator
from src.processing.hyperspectral import HyperspectralProcessor
from src.utils.logger import setup_logger


def create_research_data(file_path):
    """Create synthetic research data for scientific analysis"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create more realistic hyperspectral data for research
        # 512x512 pixels, 224 bands (AVIRIS-like)
        data = np.random.rand(512, 512, 224).astype(np.float32)
        
        wavelengths = np.linspace(400, 2500, 224)
        
        # Add realistic vegetation patterns with different health levels
        # Healthy vegetation area
        healthy_area = data[100:300, 100:300, :]
        for i, wl in enumerate(wavelengths):
            if 500 <= wl <= 700:  # Photosynthetic active radiation
                healthy_area[:, :, i] *= 0.08 + 0.15 * np.sin(wl/100)
            elif 700 <= wl <= 1300:  # Near-infrared plateau
                healthy_area[:, :, i] *= 0.45 + 0.25 * np.cos(wl/200)
            elif 1300 <= wl <= 2500:  # Short-wave infrared
                healthy_area[:, :, i] *= 0.03 + 0.08 * np.sin(wl/300)
        
        # Stressed vegetation area
        stressed_area = data[350:450, 50:150, :]
        for i, wl in enumerate(wavelengths):
            if 500 <= wl <= 700:
                stressed_area[:, :, i] *= 0.12 + 0.18 * np.sin(wl/100)
            elif 700 <= wl <= 1300:
                stressed_area[:, :, i] *= 0.25 + 0.15 * np.cos(wl/200)
            elif 1300 <= wl <= 2500:
                stressed_area[:, :, i] *= 0.08 + 0.12 * np.sin(wl/300)
        
        # Save as binary file
        data.tofile(file_path)
        
        # Create header file
        header_file = file_path.replace('.bil', '.hdr')
        with open(header_file, 'w') as f:
            f.write("ENVI\n")
            f.write("samples = 512\n")
            f.write("lines = 512\n")
            f.write("bands = 224\n")
            f.write("header offset = 0\n")
            f.write("file type = ENVI Standard\n")
            f.write("data type = 4\n")
            f.write("interleave = bil\n")
            f.write("byte order = 0\n")
            f.write("wavelength = {}".format(",".join(map(str, wavelengths))))
        
        print(f"Created research data: {file_path}")
        
    except Exception as e:
        print(f"Error creating research data: {e}")


def perform_scientific_analysis(pipeline, results, output_dir):
    """Perform comprehensive scientific analysis"""
    try:
        # Create analysis directory
        analysis_dir = os.path.join(output_dir, 'scientific_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        plots_dir = os.path.join(analysis_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get scientific analysis results
        scientific_analysis = results.get('scientific_analysis', {})
        
        # 1. Statistical Analysis
        print("\n1. STATISTICAL ANALYSIS")
        print("-" * 40)
        
        if 'index_statistics' in scientific_analysis:
            stats = scientific_analysis['index_statistics']
            
            # Create statistical summary plot
            plt.figure(figsize=(12, 8))
            
            index_names = list(stats.keys())
            means = [stats[name]['mean'] for name in index_names]
            stds = [stats[name]['std'] for name in index_names]
            
            plt.bar(range(len(index_names)), means, yerr=stds, 
                   capsize=5, alpha=0.7, color='skyblue')
            plt.xticks(range(len(index_names)), index_names, rotation=45)
            plt.ylabel('Mean Value')
            plt.title('Vegetation Index Statistics')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'index_statistics.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print statistics
            for name in index_names[:5]:  # Show first 5 indices
                stat = stats[name]
                print(f"{name}:")
                print(f"  Mean: {stat['mean']:.4f}")
                print(f"  Std: {stat['std']:.4f}")
                print(f"  Min: {stat['min']:.4f}")
                print(f"  Max: {stat['max']:.4f}")
        
        # 2. Correlation Analysis
        print("\n2. CORRELATION ANALYSIS")
        print("-" * 40)
        
        if 'correlation_analysis' in scientific_analysis:
            corr_analysis = scientific_analysis['correlation_analysis']
            
            if 'correlation_matrix' in corr_analysis:
                corr_matrix = corr_analysis['correlation_matrix']
                
                # Create correlation heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                           center=0, fmt='.2f')
                plt.title('Vegetation Index Correlation Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print("Correlation matrix generated")
            
            if 'strong_correlations' in corr_analysis:
                strong_corr = corr_analysis['strong_correlations']
                print(f"Strong correlations found: {len(strong_corr)}")
                for corr in strong_corr[:3]:  # Show top 3
                    print(f"  {corr['index1']} - {corr['index2']}: {corr['correlation']:.3f}")
        
        # 3. Spatial Analysis
        print("\n3. SPATIAL ANALYSIS")
        print("-" * 40)
        
        if 'spatial_analysis' in scientific_analysis:
            spatial = scientific_analysis['spatial_analysis']
            
            if 'overall' in spatial:
                overall = spatial['overall']
                print(f"Spatial autocorrelation: {overall.get('spatial_autocorrelation', 0):.3f}")
                print(f"Spatial heterogeneity: {overall.get('spatial_heterogeneity', 0):.3f}")
        
        # 4. Plant Condition Assessment
        print("\n4. PLANT CONDITION ASSESSMENT")
        print("-" * 40)
        
        plant_condition = results.get('plant_condition', {})
        if 'classification' in plant_condition:
            classification = plant_condition['classification']
            print(f"Plant condition class: {classification['class']}")
            print(f"Description: {classification['description']}")
            print(f"Overall score: {classification['overall_score']:.3f}")
            print(f"Confidence: {classification['confidence']:.2f}")
        
        # 5. Data Quality Assessment
        print("\n5. DATA QUALITY ASSESSMENT")
        print("-" * 40)
        
        data_quality = results['processed_data'].get('data_quality', {})
        if 'overall_quality' in data_quality:
            quality = data_quality['overall_quality']
            print(f"Overall quality score: {quality.get('quality_score', 0):.3f}")
            print(f"Average SNR: {quality.get('average_snr', 0):.2f}")
            print(f"Data completeness: {quality.get('completeness', 0):.1f}%")
        
        print(f"\nScientific analysis results saved to: {analysis_dir}")
        
    except Exception as e:
        print(f"Error in scientific analysis: {e}")


def main():
    """Main function for scientific analysis example"""
    
    # Setup logging
    logger = setup_logger('GOP_Scientific', level=logging.INFO)
    logger.info("Starting scientific analysis pipeline example")
    
    try:
        # Path to research data
        input_path = "data/research_field.bil"
        output_dir = "results/scientific_analysis"
        
        # Check if data exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            logger.info("Creating research data for analysis")
            create_research_data(input_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Initialize pipeline
        logger.info("Initializing scientific pipeline")
        pipeline = Pipeline()
        
        # Process data with comprehensive analysis
        logger.info(f"Processing research data: {input_path}")
        results = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            sensor_type='Hyperspectral',
            selected_indices=['NDVI', 'GNDVI', 'EVI', 'SAVI', 'MSAVI', 'NDWI', 'MSI', 'SIPI2'],
            use_refinement=True,
            compression_ratio=0.1,
            enable_scientific_analysis=True
        )
        
        # Perform scientific analysis
        logger.info("Performing comprehensive scientific analysis")
        perform_scientific_analysis(pipeline, results, output_dir)
        
        # Save results
        results_file = os.path.join(output_dir, 'scientific_results.json')
        pipeline.save_results(results_file)
        
        # Export scientific data
        pipeline.export_scientific_data(output_dir)
        
        # Display summary
        print("\n" + "="*60)
        print("SCIENTIFIC ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Input file: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"Results file: {results_file}")
        print(f"Scientific data exported to: {output_dir}/scientific_export/")
        
        print("\nAnalysis includes:")
        print("- Statistical analysis of vegetation indices")
        print("- Correlation analysis between indices")
        print("- Spatial pattern analysis")
        print("- Plant condition assessment")
        print("- Data quality metrics")
        
        print("\n" + "="*60)
        print("Scientific analysis example completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in scientific analysis example: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())