"""
Cache manager for GOP GUI application
"""

import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheManager:
    """Manager for data caching"""
    
    def __init__(self, redis_url: str = 'redis://localhost:6379/0',
                 cache_dir: str = 'data/cache') -> None:
        """
        Initialize cache manager
        
        Args:
            redis_url: URL for Redis connection
            cache_dir: Directory for file cache
        """
        self.redis_url = redis_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Local in-memory cache
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = 3600  # 1 hour default
        
        # Initialize Redis if available
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                # Test connection
                self.redis_client.ping()
                self.cache_mode = "redis"
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.cache_mode = "file"
        else:
            self.cache_mode = "file"
    
    def get(self, key: str, use_local_cache: bool = True) -> Optional[Any]:
        """
        Get data from cache
        
        Args:
            key: Cache key
            use_local_cache: Use local cache
            
        Returns:
            Cached data or None
        """
        # Try to get from local cache
        if use_local_cache and key in self.local_cache:
            cached_item = self.local_cache[key]
            if self._is_valid(cached_item):
                return cached_item['data']
            else:
                del self.local_cache[key]
        
        # Try to get from Redis
        if self.cache_mode == "redis" and self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    cached_item = pickle.loads(cached_data)
                    if self._is_valid(cached_item):
                        # Save to local cache
                        if use_local_cache:
                            self.local_cache[key] = cached_item
                        return cached_item['data']
                    else:
                        self.redis_client.delete(key)
            except (pickle.PickleError, redis.RedisError):
                pass
        
        # Try to get from file cache
        if self.cache_mode == "file":
            return self._get_from_file_cache(key)
        
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """
        Save data to cache
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if ttl is None:
            ttl = self.default_ttl
        
        cache_item = {
            'data': data,
            'created_at': datetime.now().isoformat(),
            'ttl': ttl
        }
        
        success = True
        
        # Save to local cache
        self.local_cache[key] = cache_item
        
        # Save to Redis
        if self.cache_mode == "redis" and self.redis_client:
            try:
                serialized_data = pickle.dumps(cache_item)
                self.redis_client.setex(key, ttl, serialized_data)
            except (pickle.PickleError, redis.RedisError):
                success = False
        
        # Save to file cache
        if self.cache_mode == "file":
            success = self._set_to_file_cache(key, cache_item, ttl)
        
        return success
    
    def delete(self, key: str) -> bool:
        """
        Delete data from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        # Delete from local cache
        if key in self.local_cache:
            del self.local_cache[key]
        
        success = True
        
        # Delete from Redis
        if self.cache_mode == "redis" and self.redis_client:
            try:
                self.redis_client.delete(key)
            except redis.RedisError:
                success = False
        
        # Delete from file cache
        if self.cache_mode == "file":
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    success = False
        
        return success
    
    def clear(self) -> bool:
        """
        Clear entire cache
        
        Returns:
            True if successful
        """
        # Clear local cache
        self.local_cache.clear()
        
        success = True
        
        # Clear Redis
        if self.cache_mode == "redis" and self.redis_client:
            try:
                self.redis_client.flushdb()
            except redis.RedisError:
                success = False
        
        # Clear file cache
        if self.cache_mode == "file":
            try:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
            except Exception:
                success = False
        
        return success
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information
        
        Returns:
            Cache status information
        """
        info = {
            'cache_mode': self.cache_mode,
            'local_cache_size': len(self.local_cache),
            'default_ttl': self.default_ttl
        }
        
        if self.cache_mode == "redis" and self.redis_client:
            try:
                info['redis_info'] = self.redis_client.info()
                info['redis_db_size'] = self.redis_client.dbsize()
            except redis.RedisError:
                info['redis_info'] = 'Connection error'
        
        if self.cache_mode == "file":
            cache_files = list(self.cache_dir.glob("*.cache"))
            info['file_cache_size'] = len(cache_files)
            total_size = sum(f.stat().st_size for f in cache_files)
            info['file_cache_bytes'] = total_size
            info['file_cache_mb'] = total_size / (1024 * 1024)
        
        return info
    
    def cache_with_ttl(self, ttl: int = None):
        """
        Decorator for caching function results
        
        Args:
            ttl: Cache time to live in seconds
            
        Returns:
            Decorator
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Create cache key based on function name and arguments
                key = self._create_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def _is_valid(self, cache_item: Dict[str, Any]) -> bool:
        """Check if cached data is valid"""
        try:
            created_at = datetime.fromisoformat(cache_item['created_at'])
            expires_at = created_at + timedelta(seconds=cache_item['ttl'])
            return datetime.now() < expires_at
        except (KeyError, ValueError):
            return False
    
    def _get_cache_file_path(self, key: str) -> Path:
        """Get cache file path"""
        # Use hash for safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_from_file_cache(self, key: str) -> Optional[Any]:
        """Get data from file cache"""
        cache_file = self._get_cache_file_path(key)
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_item = pickle.load(f)
            
            if self._is_valid(cache_item):
                return cache_item['data']
            else:
                cache_file.unlink()
                return None
        except (pickle.PickleError, FileNotFoundError, ValueError):
            # Delete corrupted file
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    def _set_to_file_cache(self, key: str, cache_item: Dict[str, Any], ttl: int) -> bool:
        """Save data to file cache"""
        cache_file = self._get_cache_file_path(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_item, f)
            return True
        except (pickle.PickleError, IOError):
            return False
    
    def _create_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key based on function and arguments"""
        # Serialize arguments to create unique key
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # If serialization failed, use string representation
            args_str = str(args)
            kwargs_str = str(kwargs)
        
        key_data = f"{func_name}:{args_str}:{kwargs_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries
        
        Returns:
            Number of deleted entries
        """
        deleted_count = 0
        
        # Clean up local cache
        expired_keys = []
        for key, cache_item in self.local_cache.items():
            if not self._is_valid(cache_item):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.local_cache[key]
            deleted_count += 1
        
        # Clean up file cache
        if self.cache_mode == "file":
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_item = pickle.load(f)
                    
                    if not self._is_valid(cache_item):
                        cache_file.unlink()
                        deleted_count += 1
                except (pickle.PickleError, ValueError):
                    # Remove corrupted files
                    try:
                        cache_file.unlink()
                        deleted_count += 1
                    except:
                        pass
        
        return deleted_count