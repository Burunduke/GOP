"""
Performance benchmarks for processing operations
"""

import time
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.processing.hyperspectral.corrections import HyperspectralCorrections
from src.processing.hyperspectral.denoising import HyperspectralDenoising


def benchmark_corrections() -> Dict[str, float]:
    """Benchmark correction operations performance."""
    # Create test data
    data = np.random.rand(500, 500, 10).astype(np.float32)
    corrections = HyperspectralCorrections()

    methods = ["dark_current", "empirical_line", "flat_field"]
    results = {}

    for method in methods:
        start = time.time()
        iterations = 5

        for _ in range(iterations):
            corrected = corrections.radiometric_correction(data, method)

        elapsed = time.time() - start

        results[method] = {
            "operation": f"Radiometric correction ({method})",
            "iterations": iterations,
            "total_time": elapsed,
            "time_per_iteration": elapsed / iterations,
            "data_size": f"{data.shape[0]}x{data.shape[1]}x{data.shape[2]}",
        }

    return results


def benchmark_denoising() -> Dict[str, float]:
    """Benchmark denoising operations performance."""
    # Create test data with some noise
    data = np.random.rand(300, 300, 20).astype(np.float32)
    # Add some noise
    noise = np.random.normal(0, 0.1, data.shape).astype(np.float32)
    noisy_data = data + noise

    denoising = HyperspectralDenoising()
    methods = ["pca", "savgol"]
    results = {}

    for method in methods:
        start = time.time()
        iterations = 3  # Fewer iterations due to longer processing time

        for _ in range(iterations):
            denoised = denoising.advanced_noise_reduction(noisy_data, method)

        elapsed = time.time() - start

        results[method] = {
            "operation": f"Denoising ({method})",
            "iterations": iterations,
            "total_time": elapsed,
            "time_per_iteration": elapsed / iterations,
            "data_size": f"{noisy_data.shape[0]}x{noisy_data.shape[1]}x{noisy_data.shape[2]}",
        }

    return results


def benchmark_atmospheric_correction() -> Dict[str, float]:
    """Benchmark atmospheric correction performance."""
    # Create test data
    data = np.random.rand(400, 400, 15).astype(np.float32)
    corrections = HyperspectralCorrections()

    start = time.time()
    iterations = 5

    for _ in range(iterations):
        corrected = corrections.atmospheric_correction(data)

    elapsed = time.time() - start

    return {
        "operation": "Atmospheric correction",
        "iterations": iterations,
        "total_time": elapsed,
        "time_per_iteration": elapsed / iterations,
        "data_size": f"{data.shape[0]}x{data.shape[1]}x{data.shape[2]}",
    }


def benchmark_cache_performance() -> Dict[str, float]:
    """Benchmark cache performance."""
    from src.processing.hyperspectral.cache import HyperspectralCache, lru_cache

    # Test data
    data = np.random.rand(100, 100).astype(np.float32)

    # Test cache class
    cache = HyperspectralCache(cache_enabled=True)

    # Function to cache
    def expensive_operation(x, factor=2.0):
        time.sleep(0.01)  # Simulate expensive operation
        return x * factor

    # Without cache
    start = time.time()
    for _ in range(10):
        result = expensive_operation(data)
    time_without_cache = time.time() - start

    # With cache
    start = time.time()
    for _ in range(10):
        result = cache.get_or_compute(
            data, "expensive_operation", expensive_operation, factor=2.0
        )
    time_with_cache = time.time() - start

    # Test LRU decorator
    @lru_cache(maxsize=10, ttl=3600)
    def cached_operation(x, factor=2.0):
        time.sleep(0.01)
        return x * factor

    start = time.time()
    for _ in range(10):
        result = cached_operation(data)
    time_with_decorator = time.time() - start

    cache_stats = cache.get_stats()

    return {
        "operation": "Cache performance",
        "time_without_cache": time_without_cache / 10,
        "time_with_cache": time_with_cache / 10,
        "time_with_decorator": time_with_decorator / 10,
        "cache_hit_rate": cache_stats.get("hit_rate", 0),
        "speedup_with_cache": time_without_cache / time_with_cache,
        "speedup_with_decorator": time_without_cache / time_with_decorator,
    }


def benchmark_memory_efficiency() -> Dict[str, float]:
    """Benchmark memory efficiency of operations."""
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Test different data types
    data_types = [np.float32, np.float64]
    sizes = [(500, 500), (1000, 1000)]

    results = {}

    for dtype in data_types:
        for size in sizes:
            # Memory before
            memory_before = process.memory_info().rss / 1024 / 1024

            # Create array
            data = np.random.rand(*size).astype(dtype)

            # Memory after creation
            memory_after = process.memory_info().rss / 1024 / 1024

            # Perform operations
            result1 = data * 2.0
            result2 = np.exp(data)
            result3 = np.sqrt(data)

            # Memory after operations
            memory_after_ops = process.memory_info().rss / 1024 / 1024

            # Clean up
            del data, result1, result2, result3

            key = f"{dtype.__name__}_{size[0]}x{size[1]}"
            results[key] = {
                "data_type": dtype.__name__,
                "size": f"{size[0]}x{size[1]}",
                "memory_usage_mb": memory_after - memory_before,
                "memory_after_ops_mb": memory_after_ops - memory_before,
                "expected_memory_mb": (size[0] * size[1] * np.dtype(dtype).itemsize)
                / 1024
                / 1024,
            }

    return results


def run_all_benchmarks() -> List[Dict[str, float]]:
    """Run all processing benchmarks and return results."""
    print("Running processing performance benchmarks...")
    print("=" * 60)

    results = []

    # Corrections benchmark
    print("\n1. Corrections Benchmark")
    corrections_results = benchmark_corrections()
    for method, result in corrections_results.items():
        results.append(result)
        print(f"   {method}: {result['time_per_iteration']:.4f}s per iteration")

    # Denoising benchmark
    print("\n2. Denoising Benchmark")
    denoising_results = benchmark_denoising()
    for method, result in denoising_results.items():
        results.append(result)
        print(f"   {method}: {result['time_per_iteration']:.4f}s per iteration")

    # Atmospheric correction benchmark
    print("\n3. Atmospheric Correction Benchmark")
    atmos_result = benchmark_atmospheric_correction()
    results.append(atmos_result)
    print(f"   Time per iteration: {atmos_result['time_per_iteration']:.4f}s")

    # Cache benchmark
    print("\n4. Cache Performance Benchmark")
    try:
        cache_result = benchmark_cache_performance()
        results.append(cache_result)
        print(f"   Without cache: {cache_result['time_without_cache']:.4f}s")
        print(f"   With cache: {cache_result['time_with_cache']:.4f}s")
        print(f"   Speedup: {cache_result['speedup_with_cache']:.1f}x")
    except Exception as e:
        print(f"   Cache benchmark failed: {e}")

    # Memory efficiency benchmark
    print("\n5. Memory Efficiency Benchmark")
    try:
        memory_results = benchmark_memory_efficiency()
        for key, result in memory_results.items():
            results.append(result)
            print(
                f"   {key}: {result['memory_usage_mb']:.1f} MB (expected: {result['expected_memory_mb']:.1f} MB)"
            )
    except ImportError:
        print("   psutil not available, skipping memory benchmark")

    print("\n" + "=" * 60)
    print("Processing benchmarks completed!")

    return results


def save_benchmark_results(
    results: List[Dict[str, float]], filename: str = "processing_benchmark_results.json"
) -> None:
    """Save benchmark results to JSON file."""
    import json
    from datetime import datetime

    results_with_metadata = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": sys.platform,
        },
        "benchmarks": results,
    }

    with open(filename, "w") as f:
        json.dump(results_with_metadata, f, indent=2)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    results = run_all_benchmarks()
    save_benchmark_results(results)

    # Print summary
    print("\nSummary:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('operation', 'Unknown operation')}")
        if "time_per_iteration" in result:
            print(f"   Performance: {result['time_per_iteration']:.4f}s per iteration")
        if "speedup_with_cache" in result:
            print(f"   Cache speedup: {result['speedup_with_cache']:.1f}x")
        if "memory_usage_mb" in result:
            print(f"   Memory usage: {result['memory_usage_mb']:.1f} MB")
