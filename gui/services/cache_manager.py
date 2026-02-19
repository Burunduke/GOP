"""
Менеджер кэширования для GUI приложения GOP
"""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheManager:
    """Менеджер для кэширования данных"""
    
    def __init__(self, redis_url: str = 'redis://localhost:6379/0', 
                 cache_dir: str = 'data/cache'):
        """
        Инициализация менеджера кэша
        
        Args:
            redis_url: URL для подключения к Redis
            cache_dir: Директория для файлового кэша
        """
        self.redis_url = redis_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Локальный кэш в памяти
        self.local_cache = {}
        self.default_ttl = 3600  # 1 час по умолчанию
        
        # Инициализация Redis если доступен
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                # Проверка соединения
                self.redis_client.ping()
                self.cache_mode = "redis"
            except Exception as e:
                print(f"Не удалось подключиться к Redis: {e}")
                self.cache_mode = "file"
        else:
            self.cache_mode = "file"
    
    def get(self, key: str, use_local_cache: bool = True) -> Optional[Any]:
        """
        Получение данных из кэша
        
        Args:
            key: Ключ кэша
            use_local_cache: Использовать локальный кэш
            
        Returns:
            Кэшированные данные или None
        """
        # Попытка получить из локального кэша
        if use_local_cache and key in self.local_cache:
            cached_item = self.local_cache[key]
            if self._is_valid(cached_item):
                return cached_item['data']
            else:
                del self.local_cache[key]
        
        # Попытка получить из Redis
        if self.cache_mode == "redis" and self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    cached_item = pickle.loads(cached_data)
                    if self._is_valid(cached_item):
                        # Сохранение в локальный кэш
                        if use_local_cache:
                            self.local_cache[key] = cached_item
                        return cached_item['data']
                    else:
                        self.redis_client.delete(key)
            except (pickle.PickleError, redis.RedisError):
                pass
        
        # Попытка получить из файлового кэша
        if self.cache_mode == "file":
            return self._get_from_file_cache(key)
        
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """
        Сохранение данных в кэш
        
        Args:
            key: Ключ кэша
            data: Данные для кэширования
            ttl: Время жизни в секундах
            
        Returns:
            True если успешно
        """
        if ttl is None:
            ttl = self.default_ttl
        
        cache_item = {
            'data': data,
            'created_at': datetime.now().isoformat(),
            'ttl': ttl
        }
        
        success = True
        
        # Сохранение в локальный кэш
        self.local_cache[key] = cache_item
        
        # Сохранение в Redis
        if self.cache_mode == "redis" and self.redis_client:
            try:
                serialized_data = pickle.dumps(cache_item)
                self.redis_client.setex(key, ttl, serialized_data)
            except (pickle.PickleError, redis.RedisError):
                success = False
        
        # Сохранение в файловый кэш
        if self.cache_mode == "file":
            success = self._set_to_file_cache(key, cache_item, ttl)
        
        return success
    
    def delete(self, key: str) -> bool:
        """
        Удаление данных из кэша
        
        Args:
            key: Ключ кэша
            
        Returns:
            True если успешно
        """
        # Удаление из локального кэша
        if key in self.local_cache:
            del self.local_cache[key]
        
        success = True
        
        # Удаление из Redis
        if self.cache_mode == "redis" and self.redis_client:
            try:
                self.redis_client.delete(key)
            except redis.RedisError:
                success = False
        
        # Удаление из файлового кэша
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
        Очистка всего кэша
        
        Returns:
            True если успешно
        """
        # Очистка локального кэша
        self.local_cache.clear()
        
        success = True
        
        # Очистка Redis
        if self.cache_mode == "redis" and self.redis_client:
            try:
                self.redis_client.flushdb()
            except redis.RedisError:
                success = False
        
        # Очистка файлового кэша
        if self.cache_mode == "file":
            try:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
            except Exception:
                success = False
        
        return success
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Получение информации о кэше
        
        Returns:
            Информация о состоянии кэша
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
                info['redis_info'] = 'Ошибка подключения'
        
        if self.cache_mode == "file":
            cache_files = list(self.cache_dir.glob("*.cache"))
            info['file_cache_size'] = len(cache_files)
            total_size = sum(f.stat().st_size for f in cache_files)
            info['file_cache_bytes'] = total_size
            info['file_cache_mb'] = total_size / (1024 * 1024)
        
        return info
    
    def cache_with_ttl(self, ttl: int = None):
        """
        Декоратор для кэширования результатов функций
        
        Args:
            ttl: Время жизни кэша в секундах
            
        Returns:
            Декоратор
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Создание ключа кэша на основе имени функции и аргументов
                key = self._create_cache_key(func.__name__, args, kwargs)
                
                # Попытка получить из кэша
                cached_result = self.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Выполнение функции и кэширование результата
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def _is_valid(self, cache_item: Dict[str, Any]) -> bool:
        """Проверка валидности кэшированных данных"""
        try:
            created_at = datetime.fromisoformat(cache_item['created_at'])
            expires_at = created_at + timedelta(seconds=cache_item['ttl'])
            return datetime.now() < expires_at
        except (KeyError, ValueError):
            return False
    
    def _get_cache_file_path(self, key: str) -> Path:
        """Получение пути к файлу кэша"""
        # Использование хэша для безопасного имени файла
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_from_file_cache(self, key: str) -> Optional[Any]:
        """Получение данных из файлового кэша"""
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
            # Удаление поврежденного файла
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    def _set_to_file_cache(self, key: str, cache_item: Dict[str, Any], ttl: int) -> bool:
        """Сохранение данных в файловый кэш"""
        cache_file = self._get_cache_file_path(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_item, f)
            return True
        except (pickle.PickleError, IOError):
            return False
    
    def _create_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Создание ключа кэша на основе функции и аргументов"""
        # Сериализация аргументов для создания уникального ключа
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Если не удалось сериализовать, используем строковое представление
            args_str = str(args)
            kwargs_str = str(kwargs)
        
        key_data = f"{func_name}:{args_str}:{kwargs_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cleanup_expired(self) -> int:
        """
        Очистка просроченных записей кэша
        
        Returns:
            Количество удаленных записей
        """
        deleted_count = 0
        
        # Очистка локального кэша
        expired_keys = []
        for key, cache_item in self.local_cache.items():
            if not self._is_valid(cache_item):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.local_cache[key]
            deleted_count += 1
        
        # Очистка файлового кэша
        if self.cache_mode == "file":
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_item = pickle.load(f)
                    
                    if not self._is_valid(cache_item):
                        cache_file.unlink()
                        deleted_count += 1
                except (pickle.PickleError, ValueError):
                    # Удаление поврежденных файлов
                    try:
                        cache_file.unlink()
                        deleted_count += 1
                    except:
                        pass
        
        return deleted_count