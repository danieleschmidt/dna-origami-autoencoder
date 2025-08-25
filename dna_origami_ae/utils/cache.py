"""Caching utilities for performance optimization."""

import time
import hashlib
import pickle
import json
from typing import Any, Dict, Optional, Callable, Union, List
from pathlib import Path
from functools import wraps
import threading
from dataclasses import dataclass
import numpy as np

from .helpers import logger, serialize_numpy


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.cache = {}
        self.access_order = []
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    def _get_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            try:
                return len(pickle.dumps(obj))
            except:
                return 1000  # Default estimate
    
    def _evict_if_needed(self):
        """Evict items if cache is too full."""
        while (len(self.cache) >= self.max_size or 
               self.stats.total_size_bytes >= self.max_memory_bytes):
            
            if not self.access_order:
                break
                
            # Remove least recently used item
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                removed_item = self.cache.pop(oldest_key)
                self.stats.total_size_bytes -= self._get_object_size(removed_item['value'])
                self.stats.evictions += 1
                logger.debug(f"Evicted cache item: {oldest_key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                
                # Check if item has expired
                item = self.cache[key]
                if item['expires_at'] and time.time() > item['expires_at']:
                    self.cache.pop(key)
                    self.access_order.remove(key)
                    self.stats.total_size_bytes -= self._get_object_size(item['value'])
                    self.stats.misses += 1
                    return None
                
                self.stats.hits += 1
                return item['value']
            else:
                self.stats.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        with self.lock:
            expires_at = time.time() + ttl if ttl else None
            item_size = self._get_object_size(value)
            
            # Remove existing item if present
            if key in self.cache:
                old_item = self.cache[key]
                self.stats.total_size_bytes -= self._get_object_size(old_item['value'])
                self.access_order.remove(key)
            
            # Add new item
            self.cache[key] = {
                'value': value,
                'created_at': time.time(),
                'expires_at': expires_at
            }
            self.access_order.append(key)
            self.stats.total_size_bytes += item_size
            
            # Evict if needed
            self._evict_if_needed()
    
    def clear(self):
        """Clear all cache items."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats = CacheStats()
    
    def size(self) -> int:
        """Get cache size."""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            return self.stats


class PersistentCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: Union[str, Path] = ".cache", 
                 max_age_hours: float = 24.0):
        """Initialize persistent cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cached items in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
        self.stats = CacheStats()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_expired(self, file_path: Path) -> bool:
        """Check if cache file is expired."""
        if not file_path.exists():
            return True
        
        age = time.time() - file_path.stat().st_mtime
        return age > self.max_age_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from persistent cache."""
        cache_path = self._get_cache_path(key)
        
        if self._is_expired(cache_path):
            self.stats.misses += 1
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.stats.hits += 1
            logger.debug(f"Cache hit: {key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache for {key}: {e}")
            self.stats.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in persistent cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Cached: {key}")
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")
    
    def clear(self):
        """Clear all cached files."""
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        self.stats = CacheStats()
    
    def cleanup_expired(self):
        """Remove expired cache files."""
        removed_count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            if self._is_expired(cache_file):
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove expired cache file {cache_file}: {e}")
        
        logger.info(f"Cleaned up {removed_count} expired cache files")


class SmartCache:
    """Smart cache that combines memory and disk caching."""
    
    def __init__(self, 
                 memory_size: int = 1000,
                 memory_mb: float = 100.0,
                 disk_cache_dir: Union[str, Path] = ".cache",
                 disk_max_age_hours: float = 24.0):
        """Initialize smart cache."""
        self.memory_cache = LRUCache(memory_size, memory_mb)
        self.disk_cache = PersistentCache(disk_cache_dir, disk_max_age_hours)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Store in memory cache for faster access
            self.memory_cache.put(key, value)
            return value
        
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put item in both caches."""
        self.memory_cache.put(key, value, ttl)
        self.disk_cache.put(key, value)
    
    def clear(self):
        """Clear both caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for both caches."""
        return {
            'memory': self.memory_cache.get_stats(),
            'disk': self.disk_cache.stats
        }


# Global cache instances
_memory_cache = LRUCache(max_size=1000, max_memory_mb=200.0)
_smart_cache = SmartCache()


def cached(cache_key_func: Optional[Callable] = None,
           ttl: Optional[float] = None,
           cache_type: str = 'memory'):
    """Decorator for caching function results.
    
    Args:
        cache_key_func: Function to generate cache key from args/kwargs
        ttl: Time to live in seconds
        cache_type: Type of cache ('memory', 'disk', 'smart')
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default key from function name and arguments
                arg_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{hashlib.md5(arg_str.encode()).hexdigest()}"
            
            # Select cache
            if cache_type == 'memory':
                cache = _memory_cache
            elif cache_type == 'smart':
                cache = _smart_cache
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            
            if cache_type == 'memory':
                cache.put(cache_key, result, ttl)
            else:
                cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_dna_sequence(sequence: str) -> str:
    """Cache key generator for DNA sequences."""
    return f"dna_seq:{hashlib.sha256(sequence.encode()).hexdigest()}"


def cache_image_data(image_array: np.ndarray, name: str = "") -> str:
    """Cache key generator for image data."""
    array_hash = hashlib.sha256(image_array.tobytes()).hexdigest()
    return f"image:{name}:{array_hash}"


def cache_structure_coords(coords: np.ndarray, atom_types: List[str]) -> str:
    """Cache key generator for structure coordinates."""
    coords_hash = hashlib.sha256(coords.tobytes()).hexdigest()
    types_hash = hashlib.sha256(str(atom_types).encode()).hexdigest()
    return f"structure:{coords_hash}:{types_hash}"


class CacheManager:
    """Manage all caches in the application."""
    
    def __init__(self):
        self.caches = {
            'memory': _memory_cache,
            'smart': _smart_cache
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        for name, cache in self.caches.items():
            if hasattr(cache, 'get_stats'):
                stats[name] = cache.get_stats()
        return stats
    
    def clear_all_caches(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        logger.info("Cleared all caches")
    
    def cleanup_expired(self):
        """Clean up expired cache entries."""
        for cache in self.caches.values():
            if hasattr(cache, 'cleanup_expired'):
                cache.cleanup_expired()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage for all caches."""
        usage = {}
        for name, cache in self.caches.items():
            if hasattr(cache, 'stats'):
                usage[name] = cache.stats.total_size_bytes
            elif hasattr(cache, 'get_stats'):
                stats = cache.get_stats()
                if isinstance(stats, dict):
                    total = sum(s.total_size_bytes for s in stats.values() 
                              if hasattr(s, 'total_size_bytes'))
                    usage[name] = total
                else:
                    usage[name] = stats.total_size_bytes
        return usage
    
    def optimize_caches(self):
        """Optimize cache performance."""
        # Clean up expired entries
        self.cleanup_expired()
        
        # Log cache statistics
        stats = self.get_cache_stats()
        for name, cache_stats in stats.items():
            if isinstance(cache_stats, dict):
                for cache_type, stat in cache_stats.items():
                    logger.info(f"Cache {name}.{cache_type}: "
                              f"Hit rate: {stat.hit_rate:.2%}, "
                              f"Size: {stat.hits + stat.misses} items")
            else:
                logger.info(f"Cache {name}: "
                          f"Hit rate: {cache_stats.hit_rate:.2%}, "
                          f"Size: {cache_stats.hits + cache_stats.misses} items")


# Global cache manager
cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return cache_manager


# Convenient caching decorators for specific use cases
def cache_sequence_analysis(ttl: float = 3600):
    """Cache sequence analysis results for 1 hour."""
    return cached(
        cache_key_func=lambda seq: cache_dna_sequence(seq.sequence if hasattr(seq, 'sequence') else str(seq)),
        ttl=ttl,
        cache_type='smart'
    )


def cache_image_processing(ttl: float = 1800):
    """Cache image processing results for 30 minutes."""
    return cached(
        cache_key_func=lambda img, *args: cache_image_data(
            img.data if hasattr(img, 'data') else img, 
            img.name if hasattr(img, 'name') else ""
        ),
        ttl=ttl,
        cache_type='smart'
    )


def cache_simulation_results(ttl: float = 7200):
    """Cache simulation results for 2 hours."""
    return cached(
        cache_key_func=lambda coords, *args: cache_structure_coords(
            coords.positions if hasattr(coords, 'positions') else coords,
            coords.atom_types if hasattr(coords, 'atom_types') else []
        ),
        ttl=ttl,
        cache_type='smart'
    )