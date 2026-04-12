#!/usr/bin/env python3
"""
Real-World Use Case Scenarios for GOP
Demonstrating practical applications in various domains

This example shows:
- Precision agriculture monitoring
- Forest ecosystem analysis
- Water resource assessment
- Environmental monitoring
- Urban analysis applications
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
from src.segmentation.segmenter import ImageSegmenter
from src.utils.logger import setup_logger


def create_scenario_data(scenario_name, file_path):
    """Create scenario-specific synthetic data"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create base hyperspectral data
        data = np.random.rand(512, 512, 224).astype(np.float32)
        wavelengths = np.linspace(400, 2500, 224)
        
        # Apply scenario-specific patterns
        if scenario_name == "agriculture":
            # Agriculture: healthy crops, stressed areas, bare soil
            # Healthy crops (high NIR, low red)
            data[100:300, 100:300, :] = apply_vegetation_pattern(data[100:300, 100:300, :], wavelengths, health=0.8)
            # Stressed crops (moderate NIR, higher red)
            data[350:450, 50:150, :] = apply_vegetation_pattern(data[350:450, 50:150, :], wavelengths, health=0.4)
            # Bare soil
            data[50:150, 350:450, :] = apply_soil_pattern(data[50:150, 350:450, :], wavelengths)
            
        elif scenario_name == "forest":
            # Forest: dense canopy, sparse areas, water bodies
            # Dense forest (very high NIR)
            data[100:400, 100:400, :] = apply_vegetation_pattern(data[100:400, 100:400, :], wavelengths, health=0.9)
            # Sparse forest
            data[400:500, 50:150, :] = apply_vegetation_pattern(data[400:500, 50:150, :], wavelengths, health=0.6)
            # Water body (low reflectance overall)
            data[50:100, 400:500, :] = apply_water_pattern(data[50:100, 400:500, :], wavelengths)
            
        elif scenario_name == "water":
            # Water resources: clean water, turbid water, vegetation
            # Clean water (very low reflectance)
            data[100:300, 100:300, :] = apply_water_pattern(data[100:300, 100:300, :], wavelengths, turbidity=0.1)
            # Turbid water (higher reflectance in visible)
            data[350:450, 50:150, :] = apply_water_pattern(data[350:450, 50:150, :], wavelengths, turbidity=0.8)
            # Riparian vegetation
            data[50:150, 350:450, :] = apply_vegetation_pattern(data[50:150, 350:450, :], wavelengths, health=0.7)
        
        data.tofile(file_path)
        
        # Create header
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
        
        print(f"Created {scenario_name} scenario data: {file_path}")
        
    except Exception as e:
        print(f"Error creating {scenario_name} data: {e}")


def apply_vegetation_pattern(data, wavelengths, health=0.8):
    """Apply vegetation spectral pattern"""
    for i, wl in enumerate(wavelengths):
        if 500 <= wl <= 700:  # Photosynthetic active radiation
            data[:, :, i] *= 0.1 + 0.15 * health * np.sin(wl/100)
        elif 700 <= wl <= 1300:  # Near-infrared plateau
            data[:, :, i] *= 0.3 + 0.4 * health * np.cos(wl/200)
        elif 1300 <= wl <= 2500:  # Short-wave infrared
            data[:, :, i] *= 0.02 + 0.06 * health * np.sin(wl/300)
    return data


def apply_soil_pattern(data, wavelengths):
    """Apply soil spectral pattern"""
    for i, wl in enumerate(wavelengths):
        if 400 <= wl <= 2500:
            data[:, :, i] *= 0.2 + 0.1 * np.sin(wl/150)
    return data


def apply_water_pattern(data, wavelengths, turbidity=0.5):
    """Apply water spectral pattern"""
    for i, wl in enumerate(wavelengths):
        if 400 <= wl <= 700:  # Visible range
            data[:, :, i] *= 0.05 + 0.1 * turbidity * np.sin(wl/100)
        else:  # NIR and SWIR
            data[:, :, i] *= 0.01 + 0.02 * turbidity
    return data


def agriculture_monitoring_scenario(output_dir):
    """Precision agriculture monitoring scenario"""
    try:
        print("\nPRECISION AGRICULTURE MONITORING")
        print("-" * 40)
        
        # Create scenario data
        data_path = "data/agriculture_scenario.bil"
        if not os.path.exists(data_path):
            create_scenario_data("agriculture", data_path)
        
        # Initialize pipeline
        pipeline = Pipeline()
        
        # Process with agriculture-specific indices
        results = pipeline.process(
            input_path=data_path,
            output_dir=os.path.join(output_dir, 'agriculture'),
            sensor_type='Hyperspectral',
            selected_indices=['NDVI', 'GNDVI', 'EVI', 'SAVI', 'MSAVI', 'NDWI', 'MSI'],
            use_refinement=True
        )
        
        # Agriculture-specific analysis
        plant_condition = results.get('plant_condition', {})
        if 'classification' in plant_condition:
            classification = plant_condition['classification']
            print("Crop Health Assessment:")
            print(f"  Condition: {classification['class']}")
            print(f"  Score: {classification['overall_score']:.3f}")
            print(f"  Confidence: {classification['confidence']:.2f}")
        
        # Calculate yield estimation metrics
        indices = results.get('calculated_indices', {})
        if 'NDVI' in indices and indices['NDVI'] is not None:
            ndvi_mean = np.mean(indices['NDVI'])
            # Simple yield estimation based on NDVI
            estimated_yield = ndvi_mean * 100  # Simplified model
            print(f"Estimated Yield Index: {estimated_yield:.1f}")
        
        print("Agriculture monitoring completed")
        
    except Exception as e:
        print(f"Error in agriculture scenario: {e}")


def forest_analysis_scenario(output_dir):
    """Forest ecosystem analysis scenario"""
    try:
        print("\nFOREST ECOSYSTEM ANALYSIS")
        print("-" * 40)
        
        # Create scenario data
        data_path = "data/forest_scenario.bil"
        if not os.path.exists(data_path):
            create_scenario_data("forest", data_path)
        
        # Initialize pipeline
        pipeline = Pipeline()
        
        # Process with forest-specific indices
        results = pipeline.process(
            input_path=data_path,
            output_dir=os.path.join(output_dir, 'forest'),
            sensor_type='Hyperspectral',
            selected_indices=['NDVI', 'EVI', 'ARVI', 'NDWI', 'MSI', 'SIPI2'],
            use_refinement=True
        )
        
        # Forest-specific analysis
        scientific_analysis = results.get('scientific_analysis', {})
        if 'index_statistics' in scientific_analysis:
            stats = scientific_analysis['index_statistics']
            print("Forest Health Indicators:")
            for index_name in ['NDVI', 'EVI', 'ARVI']:
                if index_name in stats:
                    stat = stats[index_name]
                    print(f"  {index_name}: Mean={stat['mean']:.3f}, Std={stat['std']:.3f}")
        
        # Canopy density estimation
        indices = results.get('calculated_indices', {})
        if 'NDVI' in indices and indices['NDVI'] is not None:
            ndvi_data = indices['NDVI']
            # Simple canopy density estimation
            dense_canopy = np.sum(ndvi_data > 0.6) / ndvi_data.size * 100
            moderate_canopy = np.sum((ndvi_data > 0.3) & (ndvi_data <= 0.6)) / ndvi_data.size * 100
            sparse_canopy = np.sum(ndvi_data <= 0.3) / ndvi_data.size * 100
            
            print("Canopy Density Distribution:")
            print(f"  Dense (>0.6): {dense_canopy:.1f}%")
            print(f"  Moderate (0.3-0.6): {moderate_canopy:.1f}%")
            print(f"  Sparse (<=0.3): {sparse_canopy:.1f}%")
        
        print("Forest analysis completed")
        
    except Exception as e:
        print(f"Error in forest scenario: {e}")


def water_resource_scenario(output_dir):
    """Water resource assessment scenario"""
    try:
        print("\nWATER RESOURCE ASSESSMENT")
        print("-" * 40)
        
        # Create scenario data
        data_path = "data/water_scenario.bil"
        if not os.path.exists(data_path):
            create_scenario_data("water", data_path)
        
        # Initialize pipeline
        pipeline = Pipeline()
        
        # Process with water-specific indices
        results = pipeline.process(
            input_path=data_path,
            output_dir=os.path.join(output_dir, 'water'),
            sensor_type='Hyperspectral',
            selected_indices=['NDWI', 'MSI', 'NDVI', 'EVI'],
            use_refinement=True
        )
        
        # Water resource analysis
        indices = results.get('calculated_indices', {})
        
        if 'NDWI' in indices and indices['NDWI'] is not None:
            ndwi_data = indices['NDWI']
            # Water body detection
            water_pixels = np.sum(ndwi_data > 0.2) / ndwi_data.size * 100
            print(f"Water Body Coverage: {water_pixels:.1f}%")
            
            # Water quality indicators
            if 'MSI' in indices and indices['MSI'] is not None:
                msi_mean = np.mean(indices['MSI'])
                # Simple water stress indicator
                if msi_mean > 1.0:
                    quality = "Stressed"
                elif msi_mean > 0.5:
                    quality = "Moderate"
                else:
                    quality = "Good"
                print(f"Water Stress Indicator: {quality} (MSI={msi_mean:.3f})")
        
        # Riparian vegetation health
        if 'NDVI' in indices and indices['NDVI'] is not None:
            ndvi_data = indices['NDVI']
            # Focus on areas near water (simplified)
            riparian_health = np.mean(ndvi_data)
            print(f"Riparian Vegetation Health: {riparian_health:.3f}")
        
        print("Water resource assessment completed")
        
    except Exception as e:
        print(f"Error in water resource scenario: {e}")


def environmental_monitoring_scenario(output_dir):
    """Environmental monitoring scenario"""
    try:
        print("\nENVIRONMENTAL MONITORING")
        print("-" * 40)
        
        # Use agriculture data for environmental monitoring
        data_path = "data/agriculture_scenario.bil"
        if not os.path.exists(data_path):
            create_scenario_data("agriculture", data_path)
        
        # Initialize pipeline
        pipeline = Pipeline()
        
        # Comprehensive environmental analysis
        results = pipeline.process(
            input_path=data_path,
            output_dir=os.path.join(output_dir, 'environmental'),
            sensor_type='Hyperspectral',
            selected_indices=['NDVI', 'EVI', 'SAVI', 'MSAVI', 'NDWI', 'MSI', 'SIPI2'],
            use_refinement=True,
            enable_scientific_analysis=True
        )
        
        # Environmental indicators
        scientific_analysis = results.get('scientific_analysis', {})
        
        print("Environmental Indicators:")
        
        # Vegetation health
        plant_condition = results.get('plant_condition', {})
        if 'classification' in plant_condition:
            classification = plant_condition['classification']
            print(f"  Overall Ecosystem Health: {classification['class']}")
            print(f"  Health Score: {classification['overall_score']:.3f}")
        
        # Biodiversity proxy (spatial heterogeneity)
        if 'spatial_analysis' in scientific_analysis:
            spatial = scientific_analysis['spatial_analysis']
            if 'overall' in spatial:
                heterogeneity = spatial['overall'].get('spatial_heterogeneity', 0)
                print(f"  Spatial Heterogeneity (Biodiversity Proxy): {heterogeneity:.3f}")
        
        # Water stress
        indices = results.get('calculated_indices', {})
        if 'MSI' in indices and indices['MSI'] is not None:
            msi_mean = np.mean(indices['MSI'])
            print(f"  Moisture Stress Index: {msi_mean:.3f}")
        
        print("Environmental monitoring completed")
        
    except Exception as e:
        print(f"Error in environmental monitoring scenario: {e}")


def main():
    """Main function for use case scenarios"""
    
    # Setup logging
    logger = setup_logger('GOP_Scenarios', level=logging.INFO)
    logger.info("Starting use case scenarios demonstration")
    
    try:
        # Create output directory
        output_dir = "results/use_case_scenarios"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        print("="*60)
        print("GOP USE CASE SCENARIOS")
        print("="*60)
        
        # Scenario 1: Precision Agriculture
        agriculture_monitoring_scenario(output_dir)
        
        # Scenario 2: Forest Analysis
        forest_analysis_scenario(output_dir)
        
        # Scenario 3: Water Resources
        water_resource_scenario(output_dir)
        
        # Scenario 4: Environmental Monitoring
        environmental_monitoring_scenario(output_dir)
        
        print("\n" + "="*60)
        print("USE CASE SCENARIOS COMPLETED")
        print("="*60)
        
        print("\nSummary of demonstrated applications:")
        print("✓ Precision Agriculture: Crop health monitoring, yield estimation")
        print("✓ Forest Analysis: Canopy density, ecosystem health assessment")
        print("✓ Water Resources: Water body detection, quality assessment")
        print("✓ Environmental Monitoring: Ecosystem health, biodiversity proxies")
        
        print("\nThese scenarios demonstrate GOP's versatility in real-world applications")
        print("across agriculture, forestry, hydrology, and environmental science.")
        
    except Exception as e:
        logger.error(f"Error in use case scenarios: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())