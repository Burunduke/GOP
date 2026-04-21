#!/usr/bin/env python3
"""
Script for running all tests in the GOP project
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def discover_and_run_tests() -> int:
    """Discover and run all tests"""
    # Test directory
    test_dir = Path(__file__).parent

    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern="test_*.py")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


def run_specific_test(test_module: str) -> int:
    """Run specific test module"""
    try:
        suite = unittest.TestLoader().loadTestsFromName(test_module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1
    except Exception as e:
        print(f"Error running test {test_module}: {e}")
        return 1


def main() -> int:
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Run GOP project tests")
    parser.add_argument(
        "--module",
        "-m",
        help="Run specific test module",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--list", "-l", action="store_true", help="Show available test modules"
    )

    args = parser.parse_args()

    # Show available tests
    if args.list:
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob("test_*.py"))
        print("Available test modules:")
        for test_file in test_files:
            module_name = test_file.stem
            print(f"  {module_name}")
        return 0

    # Set verbosity level
    if args.verbose:
        verbosity = 2
    else:
        verbosity = 1

    # Run specific module
    if args.module:
        return run_specific_test(args.module)

    # Run all tests
    print("Running all GOP project tests...")
    print("=" * 50)

    return discover_and_run_tests()


if __name__ == "__main__":
    sys.exit(main())
