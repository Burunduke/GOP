"""
Performance benchmarks for vegetation indices calculations
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.indices.definitions import IndexDefinitions


def benchmark_ndvi_calculation() -> Dict[str, float]:
    """Benchmark NDVI calculation performance."""
    # Create test data
    nir = np.random.rand(1000, 1000).astype(np.float32)
    red = np.random.rand(1000, 1000).astype(np.float32)

    start = time.time()
    iterations = 10

    for _ in range(iterations):
        ndvi = (nir - red) / (nir + red + 1e-8)

    elapsed = time.time() - start

    return {
        "operation": "NDVI calculation",
        "iterations": iterations,
        "total_time": elapsed,
        "time_per_iteration": elapsed / iterations,
        "data_size": f"{nir.shape[0]}x{nir.shape[1]}",
        "data_type": str(nir.dtype),
    }


def benchmark_multiple_indices() -> Dict[str, float]:
    """Benchmark multiple vegetation indices calculation."""
    # Create test bands
    bands = {
        "NIR": np.random.rand(500, 500).astype(np.float32),
        "Red": np.random.rand(500, 500).astype(np.float32),
        "Green": np.random.rand(500, 500).astype(np.float32),
        "Blue": np.random.rand(500, 500).astype(np.float32),
        "RedEdge": np.random.rand(500, 500).astype(np.float32),
    }

    indices = ["NDVI", "GNDVI", "EVI", "SAVI", "MSAVI"]
    definitions = IndexDefinitions()

    start = time.time()
    iterations = 5

    for _ in range(iterations):
        for index_name in indices:
            try:
                definitions.calculate_index(index_name, bands)
            except Exception as e:
                print(f"Warning: Could not calculate {index_name}: {e}")

    elapsed = time.time() - start

    return {
        "operation": "Multiple indices calculation",
        "indices": indices,
        "iterations": iterations,
        "total_time": elapsed,
        "time_per_iteration": elapsed / iterations,
        "time_per_index": elapsed / (iterations * len(indices)),
        "data_size": f"{bands['NIR'].shape[0]}x{bands['NIR'].shape[1]}",
    }


def benchmark_vectorized_operations() -> Dict[str, float]:
    """Benchmark vectorized vs non-vectorized operations."""
    data = np.random.rand(1000, 1000).astype(np.float32)
    mask = data > 0.5
    factor = 2.0

    # Vectorized operation
    start = time.time()
    for _ in range(10):
        result_vectorized = np.where(mask, data * factor, 0)
    time_vectorized = time.time() - start

    # Non-vectorized operation (for comparison)
    start = time.time()
    for _ in range(10):
        result_loop = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if mask[i, j]:
                    result_loop[i, j] = data[i, j] * factor
    time_loop = time.time() - start

    return {
        "operation": "Vectorized vs Loop operations",
        "vectorized_time": time_vectorized / 10,
        "loop_time": time_loop / 10,
        "speedup": time_loop / time_vectorized,
        "data_size": f"{data.shape[0]}x{data.shape[1]}",
    }


def benchmark_memory_usage() -> Dict[str, float]:
    """Benchmark memory usage patterns."""
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Memory before
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # Create large array
    large_array = np.random.rand(2000, 2000, 10).astype(np.float32)

    # Memory after creation
    memory_after_creation = process.memory_info().rss / 1024 / 1024

    # Process data
    processed = large_array * 2.0

    # Memory after processing
    memory_after_processing = process.memory_info().rss / 1024 / 1024

    # Clean up
    del large_array, processed

    return {
        "operation": "Memory usage benchmark",
        "memory_before_mb": memory_before,
        "memory_after_creation_mb": memory_after_creation,
        "memory_after_processing_mb": memory_after_processing,
        "memory_increase_mb": memory_after_creation - memory_before,
        "array_size": f"{2000}x{2000}x10",
        "array_memory_mb": (2000 * 2000 * 10 * 4) / 1024 / 1024,  # 4 bytes per float32
    }


def run_all_benchmarks() -> List[Dict[str, float]]:
    """Run all benchmarks and return results."""
    print("Running performance benchmarks...")
    print("=" * 60)

    results = []

    # NDVI benchmark
    print("\n1. NDVI Calculation Benchmark")
    ndvi_result = benchmark_ndvi_calculation()
    results.append(ndvi_result)
    print(f"   Time per iteration: {ndvi_result['time_per_iteration']:.4f}s")
    print(f"   Data size: {ndvi_result['data_size']}")

    # Multiple indices benchmark
    print("\n2. Multiple Indices Benchmark")
    multi_result = benchmark_multiple_indices()
    results.append(multi_result)
    print(f"   Time per index: {multi_result['time_per_index']:.4f}s")
    print(f"   Indices: {', '.join(multi_result['indices'])}")

    # Vectorized operations benchmark
    print("\n3. Vectorized Operations Benchmark")
    vec_result = benchmark_vectorized_operations()
    results.append(vec_result)
    print(f"   Vectorized time: {vec_result['vectorized_time']:.4f}s")
    print(f"   Loop time: {vec_result['loop_time']:.4f}s")
    print(f"   Speedup: {vec_result['speedup']:.1f}x")

    # Memory usage benchmark
    try:
        print("\n4. Memory Usage Benchmark")
        mem_result = benchmark_memory_usage()
        results.append(mem_result)
        print(f"   Memory increase: {mem_result['memory_increase_mb']:.1f} MB")
        print(f"   Expected array size: {mem_result['array_memory_mb']:.1f} MB")
    except ImportError:
        print("   psutil not available, skipping memory benchmark")

    print("\n" + "=" * 60)
    print("Benchmarks completed!")

    return results


def save_benchmark_results(
    results: List[Dict[str, float]], filename: str = "benchmark_results.json"
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
        print(f"{i}. {result['operation']}")
        if "time_per_iteration" in result:
            print(f"   Performance: {result['time_per_iteration']:.4f}s per iteration")
        if "speedup" in result:
            print(f"   Speedup: {result['speedup']:.1f}x")
        if "memory_increase_mb" in result:
            print(f"   Memory: +{result['memory_increase_mb']:.1f} MB")
