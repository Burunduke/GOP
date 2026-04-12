"""
Tests for hyperspectral cache module
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import pytest

from src.processing.hyperspectral.cache import HyperspectralCache


class TestHyperspectralCache(unittest.TestCase):
    """Test cases for HyperspectralCache class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = HyperspectralCache(cache_enabled=True, cache_dir=self.temp_dir)

        # Create test data
        self.test_data = np.random.rand(10, 10, 5).astype(np.float32)
        self.test_key = "test_data_key"

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test cache initialization"""
        self.assertIsInstance(self.cache, HyperspectralCache)
        self.assertTrue(self.cache.cache_enabled)
        self.assertEqual(self.cache.cache_dir, self.temp_dir)

    def test_initialization_disabled(self):
        """Test cache initialization with disabled cache"""
        cache = HyperspectralCache(cache_enabled=False)
        self.assertFalse(cache.cache_enabled)

    def test_get_cache_key(self):
        """Test cache key generation"""
        key = self.cache._get_cache_key(
            self.test_data, "test_function", param1="value1", param2=123
        )
        self.assertIsInstance(key, str)
        self.assertGreater(len(key), 0)

    def test_set_and_get_data(self):
        """Test setting and getting data from cache"""
        # Set data
        result = self.cache.set(self.test_key, self.test_data)
        self.assertTrue(result)

        # Get data
        retrieved_data = self.cache.get(self.test_key)

        np.testing.assert_array_equal(retrieved_data, self.test_data)

    def test_get_nonexistent_key(self):
        """Test getting non-existent key"""
        result = self.cache.get("nonexistent_key")
        self.assertIsNone(result)

    def test_get_or_compute(self):
        """Test get_or_compute functionality"""

        def compute_func(data, multiplier=1):
            return data * multiplier

        # First call should compute and cache
        result1 = self.cache.get_or_compute(
            self.test_data, "test_func", compute_func, multiplier=2
        )
        expected = self.test_data * 2
        np.testing.assert_array_equal(result1, expected)

        # Second call should get from cache
        result2 = self.cache.get_or_compute(
            self.test_data, "test_func", compute_func, multiplier=2
        )
        np.testing.assert_array_equal(result2, expected)

    def test_clear_cache(self):
        """Test clearing the cache"""
        # Set some data
        self.cache.set(self.test_key, self.test_data)

        # Verify data exists
        result = self.cache.get(self.test_key)
        self.assertIsNotNone(result)

        # Clear cache
        self.cache.clear()

        # Verify data is gone
        result = self.cache.get(self.test_key)
        self.assertIsNone(result)

    def test_cache_statistics(self):
        """Test cache statistics"""
        # Set some data
        self.cache.set(self.test_key, self.test_data)
        self.cache.set("key2", np.array([1, 2, 3]))

        # Get some data to generate hits
        self.cache.get(self.test_key)
        self.cache.get("nonexistent_key")  # This should be a miss

        stats = self.cache.get_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)
        self.assertIn("memory_cache_size", stats)
        self.assertIn("cache_dir", stats)

        self.assertGreaterEqual(stats["hits"], 1)
        self.assertGreaterEqual(stats["misses"], 1)

    def test_cache_disabled_behavior(self):
        """Test cache behavior when disabled"""
        cache = HyperspectralCache(cache_enabled=False)

        # Set should return False when disabled
        result = cache.set(self.test_key, self.test_data)
        self.assertFalse(result)

        # Get should return None when disabled
        result = cache.get(self.test_key)
        self.assertIsNone(result)

    def test_cache_size_info(self):
        """Test cache size information"""
        # Set some data
        self.cache.set(self.test_key, self.test_data)

        size_info = self.cache.get_cache_size()

        self.assertIsInstance(size_info, dict)
        self.assertIn("total_size_bytes", size_info)
        self.assertIn("total_size_mb", size_info)
        self.assertIn("file_count", size_info)

        self.assertGreaterEqual(size_info["file_count"], 1)

    def test_cleanup_old_files(self):
        """Test cleanup of old cache files"""
        # Set some data
        self.cache.set(self.test_key, self.test_data)

        # Cleanup files older than 0 days (should remove all)
        deleted_count = self.cache.cleanup_old_files(max_age_days=0)

        self.assertGreaterEqual(deleted_count, 1)

    def test_cache_key_uniqueness(self):
        """Test that different data generates different cache keys"""
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        key1 = self.cache._get_cache_key(data1, "test_func")
        key2 = self.cache._get_cache_key(data2, "test_func")

        self.assertNotEqual(key1, key2)

    def test_cache_key_with_parameters(self):
        """Test cache key generation with different parameters"""
        key1 = self.cache._get_cache_key(self.test_data, "test_func", param1="value1")
        key2 = self.cache._get_cache_key(self.test_data, "test_func", param1="value2")

        self.assertNotEqual(key1, key2)

    def test_cache_persistence(self):
        """Test that cache persists between instances"""
        # Set data with first cache instance
        cache1 = HyperspectralCache(cache_enabled=True, cache_dir=self.temp_dir)
        cache1.set(self.test_key, self.test_data)

        # Create new cache instance with same directory
        cache2 = HyperspectralCache(cache_enabled=True, cache_dir=self.temp_dir)

        # Should be able to get data from second instance
        retrieved = cache2.get(self.test_key)
        np.testing.assert_array_equal(retrieved, self.test_data)

    def test_memory_cache_priority(self):
        """Test that memory cache is checked first"""
        # Set data
        self.cache.set(self.test_key, self.test_data)

        # Get data twice - second should come from memory cache
        result1 = self.cache.get(self.test_key)
        result2 = self.cache.get(self.test_key)

        np.testing.assert_array_equal(result1, self.test_data)
        np.testing.assert_array_equal(result2, self.test_data)

    def test_cache_with_large_data(self):
        """Test caching of large data arrays"""
        large_data = np.random.rand(100, 100, 50).astype(np.float32)

        result = self.cache.set("large_data", large_data)
        self.assertTrue(result)

        retrieved = self.cache.get("large_data")
        np.testing.assert_array_equal(retrieved, large_data)


if __name__ == "__main__":
    unittest.main()
