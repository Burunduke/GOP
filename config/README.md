# GOP Configuration Documentation

## Overview

This document provides complete documentation for the GOP (Geospatial Orthophoto Processing) configuration system. The application uses a hierarchical configuration approach with YAML files and environment variables.

## Configuration Files

### Main Configuration File
- **File**: [`config/config.yaml`](config/config.yaml)
- **Format**: YAML
- **Purpose**: Main application configuration with scientific processing parameters

### Environment Configuration
- **File**: [`.env.example`](.env.example) (template) → `.env` (actual)
- **Format**: Key-value pairs
- **Purpose**: Environment-specific settings and secrets

## Configuration Structure

### Processing Configuration
```yaml
processing:
  max_image_size: 15000           # Maximum image size in pixels
  compression_ratio: 0.125        # Compression ratio for intermediate files
  batch_size: 32                  # Processing batch size
  num_workers: 4                  # Number of parallel workers
  orthophoto_resolution: 0.05     # Orthophoto resolution in meters
  dem_resolution: 0.1             # DEM resolution in meters
  feature_quality: "high"         # Feature extraction quality
  matcher_neighbors: 8            # Number of neighbors for feature matching
  odm_timeout: 7200               # OpenDroneMap timeout in seconds (2 hours)
```

#### Radiometric Correction
```yaml
radiometric_correction:
  method: "empirical_line"        # dark_current, empirical_line, flat_field
  dark_percentile: 1              # Percentile for dark pixel detection
  bright_percentile: 99           # Percentile for bright pixel detection
```

#### Atmospheric Correction
```yaml
atmospheric_correction:
  enabled: true
  method: "simplified"            # simplified, empirical_line, modtran
```

#### Noise Reduction
```yaml
noise_reduction:
  method: "pca"                   # pca, mnf, wavelet, savgol
  n_components: 0.95              # PCA components ratio
  wavelet_type: "db4"             # Wavelet type for wavelet denoising
  wavelet_levels: 2               # Wavelet decomposition levels
  savgol_window: 11               # Savitzky-Golay window size
  savgol_polyorder: 3             # Savitzky-Golay polynomial order
```

### Segmentation Configuration
```yaml
segmentation:
  model_path: "models/segmentation/best_deeplabv3plus_resnet50_voc_os16.pth"
  device: "auto"                  # auto, cpu, cuda
  confidence_threshold: 0.5       # Minimum confidence for segmentation
```

#### Cascade PSP (CascadePSP)
```yaml
cascade_psp:
  enabled: true
  l_parameter: 500                # L parameter for CascadePSP
  refinement_threshold: 0.7       # Refinement threshold
```

### Vegetation Indices
```yaml
indices:
  sensor_types: ["RGB", "Multispectral", "Hyperspectral"]
  
  # Scientific index classification
  index_groups:
    greenness: ["GNDVI", "MCARI", "MNLI", "OSAVI", "TVI", "NDVI"]
    stress: ["SIPI2", "mARI", "PRI", "CRI"]
    water: ["NDWI", "MSI", "WI", "NDII"]
    pigment: ["CARI", "PSRI", "SIPI"]
    structure: ["MSR", "MSAVI", "TVI"]
  
  default_indices: ["GNDVI", "MCARI", "MNLI", "OSAVI", "TVI", "SIPI2", "mARI", "NDWI", "MSI"]
```

### Scientific Analysis
```yaml
scientific_analysis:
  enabled: true
  
  statistics:
    confidence_level: 0.95        # Statistical confidence level
    outlier_detection: true       # Enable outlier detection
    outlier_method: "iqr"         # iqr, zscore, isolation_forest
  
  correlation:
    method: "pearson"             # pearson, spearman, kendall
    threshold: 0.7                # Correlation threshold
    significance_test: true       # Statistical significance testing
  
  spatial:
    morans_i: true                # Moran's spatial autocorrelation
    hotspot_analysis: true        # Hotspot analysis
    fragmentation_index: true     # Landscape fragmentation index
    spatial_autocorrelation: true # General spatial autocorrelation
```

### Output Configuration
```yaml
output:
  results_dir: "results"          # Output directory for results
  save_intermediate: true         # Save intermediate processing files
  output_format: "GeoTIFF"        # Output file format
  
  scientific_reports:
    enabled: true
    format: "json"                # json, csv, excel
    include_statistics: true      # Include statistical analysis
    include_correlations: true    # Include correlation matrices
    include_spatial_analysis: true # Include spatial analysis
```

### Performance Configuration
```yaml
performance:
  memory:
    max_memory_usage: "8GB"       # Maximum memory usage
    chunk_size: 1024              # Processing chunk size
    memory_mapping: true          # Use memory mapping for large files
  
  parallel:
    enabled: true                 # Enable parallel processing
    max_workers: 4                # Maximum parallel workers
    chunk_processing: true        # Process data in chunks
  
  cache:
    enabled: true                 # Enable caching
    cache_dir: "cache"            # Cache directory
    max_cache_size: "1GB"         # Maximum cache size
    max_memory_entries: 100       # Maximum memory entries
    ttl: 3600                     # Time to live in seconds (1 hour)
    cleanup_interval: 86400       # Cache cleanup interval (24 hours)
    compression: true             # Enable cache compression
    stats_enabled: true           # Enable cache statistics
```

### Logging Configuration
```yaml
logging:
  level: "INFO"                   # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "logs/gop.log"           # Log file path
  max_size: "10MB"               # Maximum log file size
  backup_count: 5                 # Number of backup files
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  scientific_logging:
    enabled: true
    log_processing_steps: true    # Log detailed processing steps
    log_quality_metrics: true     # Log quality metrics
    log_performance_metrics: true # Log performance metrics
```

### Environment Variables (.env)

```bash
# Debug mode (True/False)
DEBUG=False

# Secret key for session management
SECRET_KEY=your-secure-secret-key-here

# Server configuration
HOST=0.0.0.0
PORT=8050

# Database configuration
DATABASE_URL=postgresql://username:password@localhost/gop_db

# Cache configuration
REDIS_URL=redis://localhost:6379/0

# File upload settings
MAX_UPLOAD_SIZE=100MB
UPLOAD_FOLDER=./uploads

# Processing settings
CACHE_ENABLED=True
CACHE_DIR=./cache

# Logging level
LOG_LEVEL=INFO

# External services
ODM_PATH=/opt/opendronemap

# Security settings
CSRF_ENABLED=True
SESSION_TIMEOUT=3600
```

## Configuration Loading Order

1. **Default Values** - Hardcoded in [`src/core/config.py`](src/core/config.py)
2. **YAML Configuration** - [`config/config.yaml`](config/config.yaml)
3. **Environment Variables** - `.env` file
4. **Command Line Arguments** - Runtime overrides

## Validation and Quality Control

```yaml
validation:
  enabled: true
  
  data_validation:
    check_missing_values: true    # Check for missing data
    check_outliers: true          # Check for statistical outliers
    check_spectral_consistency: true # Check spectral consistency
    min_snr: 10                   # Minimum signal-to-noise ratio
  
  result_validation:
    check_georeference: true      # Validate georeferencing
    check_projection: true        # Validate coordinate system
    check_data_range: true        # Validate data value ranges
    check_nodata_values: true     # Check for no-data values
```

## External Tools Integration

```yaml
external_tools:
  opendronemap:
    enabled: true
    auto_detect: true
    fallback_to_gdal: true
  
  gdal:
    config_options:
      GDAL_CACHEMAX: "512"        # GDAL cache size in MB
      GDAL_DATA: "/usr/share/gdal" # GDAL data directory
      CPL_DEBUG: "OFF"            # GDAL debug mode
```

## Experimental Features

```yaml
experimental:
  enabled: false                  # Enable experimental features
  
  machine_learning:
    enabled: false
    auto_classification: false    # Automatic classification
    anomaly_detection: false      # Anomaly detection
  
  cloud_processing:
    enabled: false
    provider: "aws"               # aws, gcp, azure
    auto_scaling: false           # Auto-scaling capability
```

## Best Practices

### Performance Optimization
1. Configure `batch_size` and `num_workers` based on available memory
2. Use `memory_mapping: true` for large datasets
3. Enable caching for repeated operations
4. Set appropriate `chunk_size` for memory-constrained environments

### Quality Control
1. Enable all validation checks for production use
2. Set appropriate `confidence_threshold` for segmentation
3. Configure outlier detection based on data characteristics

### Scientific Accuracy
1. Choose appropriate correction methods for your sensor type
2. Validate spectral calibration settings
3. Configure statistical analysis parameters for your research needs

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max_memory_usage` or `batch_size`
2. **Slow Processing**: Increase `num_workers` or enable caching
3. **Segmentation Failures**: Adjust `confidence_threshold`
4. **Validation Errors**: Check input data quality and format

### Configuration Validation

The configuration system includes built-in validation. Invalid settings will generate warnings or errors during startup.

## Version Information

- **Configuration Version**: 2.0.0
- **Last Updated**: 2026-04-12
- **Compatibility**: GOP v2.0.0+

For detailed API documentation, see [API Reference](../docs/api/index.rst).