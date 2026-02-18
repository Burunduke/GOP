"""
Модуль кэширования для гиперспектральной обработки
"""

import os
import hashlib
import pickle
import time
import logging
from typing import Any, Dict, Optional


class HyperspectralCache:
    """Класс для кэширования результатов обработки гиперспектральных данных"""
    
    def __init__(self, cache_enabled: bool = True, cache_dir: str = None):
        """
        Инициализация кэша
        
        Args:
            cache_enabled: Включить кэширование результатов
            cache_dir: Директория для кэша (по умолчанию ~/.gop_cache)
        """
        self.cache_enabled = cache_enabled
        self.logger = logging.getLogger(__name__)
        
        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.gop_cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Внутренний кэш для быстрых операций
        self._memory_cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        self.logger.info(f"Кэш инициализирован. Директория: {self.cache_dir}, включен: {cache_enabled}")
    
    def _get_cache_key(self, data: Any, method_name: str, **kwargs) -> str:
        """
        Генерация ключа кэша на основе данных и параметров
        
        Args:
            data: Входные данные
            method_name: Имя метода
            **kwargs: Дополнительные параметры
            
        Returns:
            Ключ кэша
        """
        try:
            # Создание хэша на основе данных и параметров
            if hasattr(data, 'shape') and hasattr(data, 'dtype'):
                # Для numpy массивов используем форму, тип данных и контрольную сумму
                data_hash = hashlib.md5()
                data_hash.update(str(data.shape).encode())
                data_hash.update(str(data.dtype).encode())
                data_hash.update(hashlib.md5(data.tobytes()).hexdigest().encode())
            else:
                data_hash = hashlib.md5(str(data).encode())
            
            # Добавляем параметры метода
            params_str = str(sorted(kwargs.items()))
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            # Комбинируем все хэши
            combined = f"{method_name}_{data_hash.hexdigest()}_{params_hash}"
            return hashlib.md5(combined.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Ошибка генерации ключа кэша: {e}")
            # Возвращаем простой ключ на основе времени
            return f"{method_name}_{time.time()}"

    def _get_cache_path(self, cache_key: str) -> str:
        """
        Получение пути к файлу кэша
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            Путь к файлу кэша
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Получение данных из кэша
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            Кэшированные данные или None
        """
        if not self.cache_enabled:
            return None
        
        try:
            # Сначала проверяем память
            if cache_key in self._memory_cache:
                self._cache_stats['hits'] += 1
                return self._memory_cache[cache_key]
            
            # Затем проверяем диск
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    # Сохраняем в память для быстрого доступа
                    self._memory_cache[cache_key] = data
                    self._cache_stats['hits'] += 1
                    return data
            
            self._cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            self.logger.warning(f"Ошибка чтения кэша {cache_key}: {e}")
            self._cache_stats['misses'] += 1
            return None

    def set(self, cache_key: str, data: Any) -> bool:
        """
        Сохранение данных в кэш
        
        Args:
            cache_key: Ключ кэша
            data: Данные для сохранения
            
        Returns:
            True если успешно, иначе False
        """
        if not self.cache_enabled:
            return False
        
        try:
            # Сохраняем в память
            self._memory_cache[cache_key] = data
            
            # Сохраняем на диск
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Ошибка записи в кэш {cache_key}: {e}")
            return False

    def get_or_compute(self, data: Any, method_name: str, compute_func, **kwargs) -> Any:
        """
        Получение данных из кэша или вычисление
        
        Args:
            data: Входные данные
            method_name: Имя метода
            compute_func: Функция для вычисления результата
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат из кэша или вычисленный
        """
        cache_key = self._get_cache_key(data, method_name, **kwargs)
        
        # Пытаемся получить из кэша
        cached_result = self.get(cache_key)
        if cached_result is not None:
            self.logger.info(f"Результат для {method_name} получен из кэша")
            return cached_result
        
        # Вычисляем результат
        result = compute_func(data, **kwargs)
        
        # Сохраняем в кэш
        self.set(cache_key, result)
        
        return result

    def clear(self) -> None:
        """Очистка кэша"""
        try:
            # Очистка памяти
            self._memory_cache.clear()
            
            # Очистка диска
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            
            self._cache_stats = {'hits': 0, 'misses': 0}
            self.logger.info("Кэш очищен")
            
        except Exception as e:
            self.logger.error(f"Ошибка очистки кэша: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики кэша
        
        Returns:
            Словарь со статистикой кэша
        """
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = self._cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'hit_rate': hit_rate,
            'memory_cache_size': len(self._memory_cache),
            'cache_dir': self.cache_dir
        }

    def cleanup_old_files(self, max_age_days: int = 30) -> int:
        """
        Очистка старых файлов кэша
        
        Args:
            max_age_days: Максимальный возраст файлов в днях
            
        Returns:
            Количество удаленных файлов
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            deleted_count = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        deleted_count += 1
            
            self.logger.info(f"Удалено {deleted_count} старых файлов кэша")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Ошибка очистки старых файлов кэша: {e}")
            return 0

    def get_cache_size(self) -> Dict[str, Any]:
        """
        Получение информации о размере кэша
        
        Returns:
            Словарь с информацией о размере
        """
        try:
            total_size = 0
            file_count = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения размера кэша: {e}")
            return {'total_size_bytes': 0, 'total_size_mb': 0, 'file_count': 0}