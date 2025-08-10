"""
Generation 3 Performance Optimization System
High-performance caching, concurrent processing, and resource management
"""

import asyncio
import concurrent.futures
import functools
import hashlib
import json
import multiprocessing
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import psutil
import redis

from .logger import get_logger


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Caching
    enable_redis_cache: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600  # 1 hour
    max_cache_memory: int = 500 * 1024 * 1024  # 500MB
    
    # Concurrent processing  
    max_workers: Optional[int] = None
    use_process_pool: bool = True
    enable_gpu_acceleration: bool = False
    
    # Resource monitoring
    memory_threshold: float = 0.85  # 85% memory usage threshold
    cpu_threshold: float = 0.90     # 90% CPU usage threshold
    
    # Batch processing
    batch_size: int = 100
    prefetch_size: int = 10


class HighPerformanceCache:
    """High-performance multi-level cache system."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = get_logger('cache')
        
        # Memory cache (L1)
        self._memory_cache = {}
        self._memory_stats = defaultdict(int)
        self._cache_lock = threading.RLock()
        
        # Redis cache (L2) 
        self._redis = None
        if config.enable_redis_cache:
            try:
                self._redis = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    decode_responses=True
                )
                self._redis.ping()
                self.logger.info("Connected to Redis cache")
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
                self._redis = None
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0, 
            'redis_hits': 0,
            'memory_hits': 0,
            'evictions': 0
        }
    
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        # Create deterministic key from function name and arguments
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Convert args to hashable format
        serializable_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                serializable_args.append(hashlib.md5(arg.tobytes()).hexdigest())
            elif hasattr(arg, '__dict__'):
                serializable_args.append(str(sorted(arg.__dict__.items())))
            else:
                serializable_args.append(str(arg))
        
        serializable_kwargs = {k: str(v) for k, v in sorted(kwargs.items())}
        
        key_data = {
            'func': func_name,
            'args': serializable_args,
            'kwargs': serializable_kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"dna_cache:{hashlib.sha256(key_str.encode()).hexdigest()[:16]}"
    
    def get(self, key: str) -> tuple:
        """Get value from cache (returns found, value)."""
        # Check memory cache first (L1)
        with self._cache_lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if time.time() < entry['expires_at']:
                    self.stats['hits'] += 1
                    self.stats['memory_hits'] += 1
                    return True, entry['value']
                else:
                    del self._memory_cache[key]
        
        # Check Redis cache (L2)
        if self._redis:
            try:
                cached_data = self._redis.get(key)
                if cached_data:
                    value = json.loads(cached_data)
                    # Store in memory cache for faster access
                    self._store_memory_cache(key, value)
                    self.stats['hits'] += 1
                    self.stats['redis_hits'] += 1
                    return True, value
            except Exception as e:
                self.logger.warning(f"Redis get failed: {e}")
        
        self.stats['misses'] += 1
        return False, None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache."""
        ttl = ttl or self.config.cache_ttl
        
        # Store in memory cache (L1)
        self._store_memory_cache(key, value, ttl)
        
        # Store in Redis cache (L2)
        if self._redis:
            try:
                serialized = json.dumps(value, default=str)
                self._redis.setex(key, ttl, serialized)
            except Exception as e:
                self.logger.warning(f"Redis set failed: {e}")
    
    def _store_memory_cache(self, key: str, value: Any, ttl: int = None) -> None:
        """Store value in memory cache with size management."""
        ttl = ttl or self.config.cache_ttl
        
        with self._cache_lock:
            # Check memory usage and evict if necessary
            current_memory = psutil.Process().memory_info().rss
            if current_memory > self.config.max_cache_memory:
                self._evict_memory_cache()
            
            self._memory_cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl,
                'created_at': time.time()
            }
    
    def _evict_memory_cache(self) -> None:
        """Evict old entries from memory cache."""
        current_time = time.time()
        
        # Remove expired entries first
        expired_keys = [
            k for k, v in self._memory_cache.items() 
            if current_time >= v['expires_at']
        ]
        
        for key in expired_keys:
            del self._memory_cache[key]
            self.stats['evictions'] += 1
        
        # If still too much memory, remove oldest entries
        if len(self._memory_cache) > 1000:  # Keep max 1000 entries
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1]['created_at']
            )
            
            to_remove = len(sorted_entries) - 1000
            for key, _ in sorted_entries[:to_remove]:
                del self._memory_cache[key]
                self.stats['evictions'] += 1
    
    def clear(self) -> None:
        """Clear all caches."""
        with self._cache_lock:
            self._memory_cache.clear()
        
        if self._redis:
            try:
                keys = self._redis.keys("dna_cache:*")
                if keys:
                    self._redis.delete(*keys)
            except Exception as e:
                self.logger.warning(f"Redis clear failed: {e}")
        
        self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests > 0:
            hit_rate = self.stats['hits'] / total_requests
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self._memory_cache),
            'redis_connected': self._redis is not None
        }


class ConcurrentProcessor:
    """High-performance concurrent processing system."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = get_logger('concurrent')
        
        # Determine optimal worker count
        self.max_workers = config.max_workers or min(32, (psutil.cpu_count() or 4) + 4)
        
        # Thread pool for I/O bound tasks
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Process pool for CPU bound tasks
        if config.use_process_pool:
            self.process_executor = ProcessPoolExecutor(
                max_workers=min(psutil.cpu_count() or 4, self.max_workers)
            )
        else:
            self.process_executor = None
        
        self.logger.info(f"Initialized concurrent processor with {self.max_workers} workers")
    
    def submit_io_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit I/O bound task to thread pool."""
        return self.thread_executor.submit(func, *args, **kwargs)
    
    def submit_cpu_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit CPU bound task to process pool."""
        if self.process_executor:
            return self.process_executor.submit(func, *args, **kwargs)
        else:
            return self.thread_executor.submit(func, *args, **kwargs)
    
    def map_concurrent(self, func: Callable, items: List[Any], 
                      cpu_bound: bool = True, batch_size: Optional[int] = None) -> List[Any]:
        """Map function over items concurrently."""
        batch_size = batch_size or self.config.batch_size
        
        if len(items) <= batch_size:
            # Process small batches directly
            executor = self.process_executor if cpu_bound else self.thread_executor
            if executor:
                futures = [executor.submit(func, item) for item in items]
                return [future.result() for future in futures]
        
        # Process large datasets in batches
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            executor = self.process_executor if cpu_bound else self.thread_executor
            
            if executor:
                futures = [executor.submit(func, item) for item in batch]
                batch_results = [future.result() for future in futures]
                results.extend(batch_results)
            else:
                # Fallback to sequential processing
                batch_results = [func(item) for item in batch]
                results.extend(batch_results)
        
        return results
    
    def shutdown(self) -> None:
        """Shutdown all executors."""
        self.thread_executor.shutdown(wait=True)
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
        self.logger.info("Concurrent processor shutdown complete")


class ResourceMonitor:
    """Monitor and manage system resources."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = get_logger('resources')
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval: float) -> None:
        """Resource monitoring loop."""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                disk_info = psutil.disk_usage('/')
                
                # Log metrics
                self.logger.info(
                    "System metrics",
                    extra={
                        'operation': 'resource_monitor',
                        'metrics': {
                            'cpu_percent': cpu_percent,
                            'memory_percent': memory_info.percent,
                            'memory_available_gb': memory_info.available / (1024**3),
                            'disk_percent': disk_info.percent,
                            'disk_free_gb': disk_info.free / (1024**3)
                        }
                    }
                )
                
                # Check thresholds and warn if exceeded
                if memory_info.percent / 100 > self.config.memory_threshold:
                    self.logger.warning(
                        f"Memory usage high: {memory_info.percent:.1f}%",
                        extra={'operation': 'resource_alert'}
                    )
                
                if cpu_percent / 100 > self.config.cpu_threshold:
                    self.logger.warning(
                        f"CPU usage high: {cpu_percent:.1f}%",
                        extra={'operation': 'resource_alert'}
                    )
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system resource statistics."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'cpu_count': psutil.cpu_count(),
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total_gb': psutil.disk_usage('/').total / (1024**3),
                'free_gb': psutil.disk_usage('/').free / (1024**3),
                'percent': psutil.disk_usage('/').percent
            }
        }


class PerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.logger = get_logger('performance')
        
        # Initialize subsystems
        self.cache = HighPerformanceCache(self.config)
        self.concurrent = ConcurrentProcessor(self.config)
        self.resources = ResourceMonitor(self.config)
        
        # Start monitoring
        self.resources.start_monitoring()
        
        self.logger.info("Performance optimizer initialized")
    
    def cached(self, ttl: Optional[int] = None):
        """Decorator for automatic caching of function results."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.cache._generate_key(func, args, kwargs)
                
                # Try to get from cache
                found, cached_value = self.cache.get(cache_key)
                if found:
                    return cached_value
                
                # Compute result
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Store in cache
                self.cache.set(cache_key, result, ttl)
                
                self.logger.debug(
                    f"Cached function result: {func.__name__}",
                    extra={
                        'operation': 'cache_store',
                        'metrics': {'duration': duration}
                    }
                )
                
                return result
            return wrapper
        return decorator
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    cpu_bound: bool = True, batch_size: Optional[int] = None) -> List[Any]:
        """Execute function over items in parallel."""
        start_time = time.time()
        
        results = self.concurrent.map_concurrent(
            func, items, cpu_bound=cpu_bound, batch_size=batch_size
        )
        
        duration = time.time() - start_time
        
        self.logger.info(
            f"Parallel processing completed: {len(items)} items",
            extra={
                'operation': 'parallel_map',
                'metrics': {
                    'item_count': len(items),
                    'duration': duration,
                    'items_per_second': len(items) / duration if duration > 0 else 0,
                    'cpu_bound': cpu_bound
                }
            }
        )
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'cache': self.cache.get_stats(),
            'resources': self.resources.get_current_stats(),
            'config': {
                'max_workers': self.concurrent.max_workers,
                'use_process_pool': self.config.use_process_pool,
                'cache_enabled': self.config.enable_redis_cache
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown performance optimizer."""
        self.resources.stop_monitoring()
        self.concurrent.shutdown()
        self.logger.info("Performance optimizer shutdown complete")


# Global performance optimizer instance
_global_optimizer = None

def get_performance_optimizer(config: Optional[PerformanceConfig] = None) -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(config)
    return _global_optimizer

def cached(ttl: Optional[int] = None):
    """Decorator for caching function results."""
    return get_performance_optimizer().cached(ttl)

def parallel_map(func: Callable, items: List[Any], cpu_bound: bool = True) -> List[Any]:
    """Execute function over items in parallel."""
    return get_performance_optimizer().parallel_map(func, items, cpu_bound)