# GOP - Geospatial Orthophoto Processing

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/indykovdm/GOP/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Scientific](https://img.shields.io/badge/purpose-scientific-orange.svg)](https://github.com/indykovdm/GOP)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](docs/api/_build/html/index.html)

**Version 2.0.0 - Scientific Data Processing Foundation**

Scientific library for hyperspectral data processing with a focus on data loading and orthophoto creation. Provides the foundation for geospatial data processing workflows.

## 🚀 Quick Start

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src.core.pipeline; print('GOP successfully installed')"
```

### First Run

```bash
# Run processing example
python examples/basic_processing.py
```

## 📚 Documentation

### For Users
- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions
- **[User Guide](docs/USER_GUIDE.md)** - How to use GOP
- **[Configuration Guide](config/README.md)** - Complete configuration documentation

### For Developers
- **[Developer Guide](docs/DEVELOPER.md)** - API documentation and development
- **[Testing Guide](docs/TESTING.md)** - Testing and CI/CD
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System architecture

### Scientific Materials
- **[Examples](examples/README.md)** - Usage examples and tutorials
- **[Technical Notes](docs/research/TECHNICAL_NOTES.md)** - Technical research documentation
- **[API Reference](docs/api/_build/html/index.html)** - Complete API documentation

## ✨ Current Features

### 🎯 Data Processing Foundation
- **Hyperspectral data loading** - Support for BIL/HDR and TIFF formats
- **Orthophoto creation** - Integration with OpenDroneMap and GDAL
- **Data validation** - Comprehensive input validation and error handling
- **Configuration system** - Flexible YAML-based configuration

### 🔒 Code Quality
- **Type safety** - Full type annotations throughout codebase
- **Error handling** - Robust exception hierarchy for stable operation
- **Input validation** - Strict data validation and sanitization
- **Modular architecture** - Clean separation of concerns

### 📊 Core Infrastructure
- **Pipeline architecture** - Extensible processing pipeline design
- **Caching system** - Basic caching for performance optimization
- **Logging system** - Comprehensive logging for debugging
- **Utility functions** - Mathematical and file operation utilities

## 🏗️ Architecture

### Core Modules
- **`src/core/`** - Main processing pipeline and configuration
- **`src/processing/`** - Data processing algorithms
  - **`hyperspectral/`** - Hyperspectral data loading and validation
  - **`orthophoto/`** - Orthophoto creation utilities
- **`src/utils/`** - Helper functions and utilities

### Enhanced Utilities
- **`math_utils`** - Safe mathematical operations with error handling
- **`validators`** - Comprehensive data validation system
- **`gdal_utils`** - GDAL integration utilities
- **`image_utils`** - Image processing utilities
- **`exceptions`** - Hierarchical exception system

## 🔧 Configuration

GOP uses a hierarchical configuration system:

```yaml
# config/config.yaml
processing:
  max_image_size: 15000
  batch_size: 32
  num_workers: 4

performance:
  cache:
    enabled: true
    max_cache_size: "1GB"
    ttl: 3600
```

See [Configuration Documentation](config/README.md) for full details.

## 🧪 Examples

### Basic Processing
```python
from src.core.pipeline import Pipeline

pipeline = Pipeline()
results = pipeline.process(
    input_path="data/sample.bil",
    output_dir="results"
)
```

### Data Loading
```python
from src.processing.hyperspectral import HyperspectralProcessor

processor = HyperspectralProcessor()
data = processor.load_data("data/sample.bil")
```

See [Examples Directory](examples/README.md) for comprehensive examples.

## 🛠️ Development

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Code formatting
black src/ tests/ examples/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Documentation
```bash
# Generate API documentation
cd docs/api && make html
```

## 📋 Requirements

### Core Dependencies
All dependencies included in [`requirements.txt`](requirements.txt):
- **Scientific libraries**: NumPy, SciPy
- **Image processing**: OpenCV, scikit-image
- **Geodata**: GDAL, rasterio
- **GUI**: Dash, Flask
- **Utilities**: PyYAML, tqdm

### Development Dependencies
- **Testing**: pytest, pytest-cov
- **Formatting**: black, flake8
- **Typing**: mypy
- **Documentation**: sphinx

### System Requirements
- **Python**: 3.9+
- **RAM**: 8GB+ (16GB recommended)
- **Disk space**: 2GB+ for dependencies
- **Operating system**: Linux, macOS, Windows

## 🤝 Contributing

We welcome contributions! Please review:
- [Contributing Guide](docs/api/contributing.rst)
- [Code of Conduct](CODE_OF_CONDUCT.md)

### Development Environment Setup
```bash
# Fork and clone
gh repo fork indykovdm/GOP --clone
cd GOP

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install all dependencies (including dev)
pip install -r requirements.txt
pip install -e .

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Scientific References
- Mitrofanov E.P., Petrushin A.F. "Use of aerial photography data for precision agricultural technologies"

### Technical Acknowledgments
- OpenDroneMap community
- GDAL/OGR development team
- Scientific Python ecosystem

## 📞 Contact

- **Author**: Dmitry Indykov
- **Email**: indykovdm@example.com
- **Repository**: https://github.com/indykovdm/GOP
- **Documentation**: https://indykovdm.github.io/GOP/

---

**GOP v2.0.0** - Foundation for scientific geospatial data processing.