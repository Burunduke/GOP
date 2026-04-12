#!/bin/bash

# Script for generating documentation before commit

set -e

echo "Generating documentation..."

# Check dependencies
if ! python -c "import sphinx" 2>/dev/null; then
    echo "❌ Sphinx not installed. Install dependencies:"
    echo "pip install sphinx sphinx_rtd_theme sphinx_autodoc_typehints"
    exit 1
fi

# Generate documentation
cd docs
python generate_docs.py --clean --build

# Check if generation was successful
if [ ! -f "api/_build/html/index.html" ]; then
    echo "❌ Documentation generation error"
    exit 1
fi

echo "✅ Documentation successfully generated"
echo "📄 Available at: docs/api/_build/html/index.html"