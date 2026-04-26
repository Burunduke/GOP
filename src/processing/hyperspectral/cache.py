"""
Caching module for hyperspectral data processing.

This module provides LRU caching functionality for hyperspectral data processing
operations, including both in-memory and disk-based caching.

NOTE: This cache is not used by default after the streaming refactor.
For new code, consider per-band caching if needed.
"""

import os
import hashlib
import pickle
import time
import logging
import gzip
import functools
from typing import Any, Dict, Optional, Callable
from collections import OrderedDict


class LRUCache:
    """LRU cache with size and time-to-live (TTL) limitations."""

    def __init__(self, maxsize: int = 100, ttl: int = 3600) -> None:
        """
        Initialize LRU cache.

        Args:
            maxsize: Maximum number of items in cache
            ttl: Time-to-live for cache items in seconds
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self._cache:
            return None

        # Check TTL
        if time.time() - self._timestamps[key] > self.ttl:
            self._remove(key)
            return None

        # Move item to end (most recent)
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        """Add item to cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                # Remove oldest item
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)

        self._cache[key] = value
        self._timestamps[key] = time.time()

    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._timestamps.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


def lru_cache(maxsize: int = 100, ttl: int = 3600) -> Callable:
    """
    Decorator for LRU caching of functions.

    Args:
        maxsize: Maximum number of items in cache
        ttl: Time-to-live for cache items in seconds

    Returns:
        Decorated function with caching
    """

    def decorator(func: Callable) -> Callable:
        cache = LRUCache(maxsize=maxsize, ttl=ttl)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key based on arguments
            key_parts = [str(arg) for arg in args]
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = hashlib.md5("".join(key_parts).encode()).hexdigest()

            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Save to cache
            cache.set(key, result)

            return result

        return wrapper

    return decorator


class HyperspectralCache:
    """Class for caching hyperspectral data processing results."""

    def __init__(self, cache_enabled: bool = True, cache_dir: Optional[str] = None) -> None:
        """
        Initialize cache.

        Args:
            cache_enabled: Enable result caching
            cache_dir: Cache directory (default: ~/.gop_cache)
        """
        self.cache_enabled = cache_enabled
        self.logger = logging.getLogger(__name__)

        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.gop_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Internal cache for fast operations
        self._memory_cache = LRUCache(maxsize=100, ttl=3600)
        self._cache_stats = {"hits": 0, "misses": 0, "disk_hits": 0, "disk_misses": 0}

        self.logger.info(
            f"Cache initialized. Directory: {self.cache_dir}, enabled: {cache_enabled}"
        )

    def _get_cache_key(self, data: Any, method_name: str, **kwargs) -> str:
        """
        Generate cache key based on data and parameters.

        Args:
            data: Input data
            method_name: Method name
            **kwargs: Additional parameters

        Returns:
            Cache key
        """
        try:
            # Create hash based on data and parameters
            if hasattr(data, "shape") and hasattr(data, "dtype"):
                # For numpy arrays use shape, dtype and checksum
                data_hash = hashlib.md5()
                data_hash.update(str(data.shape).encode())
                data_hash.update(str(data.dtype).encode())
                data_hash.update(hashlib.md5(data.tobytes()).hexdigest().encode())
            else:
                data_hash = hashlib.md5(str(data).encode())

            # Add method parameters
            params_str = str(sorted(kwargs.items()))
            params_hash = hashlib.md5(params_str.encode()).hexdigest()

            # Combine all hashes
            combined = f"{method_name}_{data_hash.hexdigest()}_{params_hash}"
            return hashlib.md5(combined.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Error generating cache key: {e}")
            # Return simple key based on time
            return f"{method_name}_{time.time()}"

    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get path to cache file.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl.gz")

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Get data from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None
        """
        if not self.cache_enabled:
            return None

        try:
            # First check memory
            cached_data = self._memory_cache.get(cache_key)
            if cached_data is not None:
                self._cache_stats["hits"] += 1
                return cached_data

            # Then check disk
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                try:
                    with gzip.open(cache_path, "rb") as f:
                        data = pickle.load(f)
                    # Save to memory for fast access
                    self._memory_cache.set(cache_key, data)
                    self._cache_stats["hits"] += 1
                    self._cache_stats["disk_hits"] += 1
                    return data
                except Exception as e:
                    self.logger.warning(f"Error reading cache file {cache_key}: {e}")

            self._cache_stats["misses"] += 1
            self._cache_stats["disk_misses"] += 1
            return None

        except Exception as e:
            self.logger.warning(f"Error reading cache {cache_key}: {e}")
            self._cache_stats["misses"] += 1
            return None

    def set(self, cache_key: str, data: Any) -> bool:
        """
        Save data to cache.

        Args:
            cache_key: Cache key
            data: Data to save

        Returns:
            True if successful, False otherwise
        """
        if not self.cache_enabled:
            return False

        try:
            # Save to memory
            self._memory_cache.set(cache_key, data)

            # Save to disk with compression
            cache_path = self._get_cache_path(cache_key)
            with gzip.open(cache_path, "wb") as f:
                pickle.dump(data, f)

            return True

        except Exception as e:
            self.logger.warning(f"Error writing to cache {cache_key}: {e}")
            return False

    def get_or_compute(
        self, data: Any, method_name: str, compute_func: Callable, **kwargs
    ) -> Any:
        """
        Get data from cache or compute it.

        Args:
            data: Input data
            method_name: Method name
            compute_func: Function to compute result
            **kwargs: Additional parameters

        Returns:
            Result from cache or computed
        """
        cache_key = self._get_cache_key(data, method_name, **kwargs)

        # Try to get from cache
        cached_result = self.get(cache_key)
        if cached_result is not None:
            self.logger.info(f"Result for {method_name} retrieved from cache")
            return cached_result

        # Compute result
        result = compute_func(data, **kwargs)

        # Save to cache
        self.set(cache_key, result)

        return result

    def clear(self) -> None:
        """Clear cache."""
        try:
            # Clear memory
            self._memory_cache.clear()

            # Clear disk
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".pkl.gz"):
                    os.remove(os.path.join(self.cache_dir, filename))

            self._cache_stats = {
                "hits": 0,
                "misses": 0,
                "disk_hits": 0,
                "disk_misses": 0,
            }
            self.logger.info("Cache cleared")

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (
            self._cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )

        disk_hit_rate = 0
        if self._cache_stats["disk_hits"] + self._cache_stats["disk_misses"] > 0:
            disk_hit_rate = self._cache_stats["disk_hits"] / (
                self._cache_stats["disk_hits"] + self._cache_stats["disk_misses"]
            )

        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "disk_hits": self._cache_stats["disk_hits"],
            "disk_misses": self._cache_stats["disk_misses"],
            "hit_rate": hit_rate,
            "disk_hit_rate": disk_hit_rate,
            "memory_cache_size": self._memory_cache.size(),
            "memory_cache_maxsize": self._memory_cache.maxsize,
            "cache_dir": self.cache_dir,
        }

    def cleanup_old_files(self, max_age_days: int = 30) -> int:
        """
        Clean up old cache files.

        Args:
            max_age_days: Maximum file age in days

        Returns:
            Number of deleted files
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            deleted_count = 0

            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".pkl.gz"):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_age = current_time - os.path.getmtime(file_path)

                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        deleted_count += 1

            self.logger.info(f"Deleted {deleted_count} old cache files")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Error cleaning old cache files: {e}")
            return 0

    def get_cache_size(self) -> Dict[str, Any]:
        """
        Get cache size information.

        Returns:
            Dictionary with size information
        """
        try:
            total_size = 0
            file_count = 0

            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".pkl.gz"):
                    file_path = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(file_path)
                    file_count += 1

            return {
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "file_count": file_count,
            }

        except Exception as e:
            self.logger.error(f"Error getting cache size: {e}")
            return {"total_size_bytes": 0, "total_size_mb": 0, "file_count": 0}
