#!/usr/bin/env python3
"""
Main entry point for GOP - Hyperspectral Processing and Plant Analysis
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from gui.app.app import main as gui_main
except ImportError as e:
    print(f"Error importing GUI modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Launch the GUI application"""
    print("Starting GOP GUI application...")
    gui_main()


if __name__ == '__main__':
    main()