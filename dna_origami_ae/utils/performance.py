"""Performance optimization utilities for DNA origami autoencoder."""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import psutil
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from contextlib import contextmanager
import pickle
import hashlib
import redis
from pathlib import Path

from .logger import get_logger, dna_logger


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    
    # Threading and multiprocessing
    max_worker_threads: int = min(32, mp.cpu_count() * 2)
    max_worker_processes: int = mp.cpu_count()
    
    # Caching
    enable_memory_cache: bool = True
    enable_redis_cache: bool = False
    memory_cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # GPU acceleration
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Batch processing
    optimal_batch_size: int = 32
    max_batch_size: int = 128
    
    # Resource monitoring
    memory_usage_threshold: float = 0.8
    cpu_usage_threshold: float = 0.9
    
    # Auto-scaling
    scale_up_threshold: float = 0.7
    scale_down_threshold: float = 0.3
    min_replicas: int = 1
    max_replicas: int = 10


class AdaptiveCache:
    """Adaptive caching system with memory and Redis support."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize adaptive cache."""
        self.config = config
        self.logger = get_logger('performance')
        
        # Memory cache
        if config.enable_memory_cache:
            self._memory_cache = {}
            self._cache_access_count = {}
            self._cache_access_time = {}
            self._cache_lock = threading.RLock()
        
        # Redis cache
        self.redis_client = None
        if config.enable_redis_cache:
            try:
                import redis
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                self.logger.info("Redis cache enabled")
            except Exception as e:
                self.logger.warning(f"Redis not available: {e}")
                self.redis_client = None
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        # Create deterministic hash of arguments
        key_data = {
            'function': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(key_str).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        # Try memory cache first
        if self.config.enable_memory_cache:
            with self._cache_lock:
                if key in self._memory_cache:
                    self._cache_access_count[key] = self._cache_access_count.get(key, 0) + 1
                    self._cache_access_time[key] = time.time()
                    return self._memory_cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                self.logger.warning(f"Redis get failed: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        ttl = ttl or self.config.cache_ttl_seconds
        
        # Memory cache
        if self.config.enable_memory_cache:
            with self._cache_lock:
                # Evict LRU items if cache is full
                if len(self._memory_cache) >= self.config.memory_cache_size:
                    self._evict_lru_items()
                
                self._memory_cache[key] = value
                self._cache_access_count[key] = 1
                self._cache_access_time[key] = time.time()
        
        # Redis cache
        if self.redis_client:
            try:
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                self.redis_client.setex(key, ttl, data)
            except Exception as e:
                self.logger.warning(f"Redis set failed: {e}")
    
    def _evict_lru_items(self) -> None:
        """Evict least recently used items."""
        # Remove 25% of least recently used items
        items_to_remove = len(self._memory_cache) // 4
        
        # Sort by access time
        sorted_items = sorted(
            self._cache_access_time.items(),
            key=lambda x: x[1]
        )
        
        for key, _ in sorted_items[:items_to_remove]:
            self._memory_cache.pop(key, None)
            self._cache_access_count.pop(key, None)
            self._cache_access_time.pop(key, None)
    
    def invalidate(self, pattern: str = None) -> None:
        """Invalidate cache entries."""
        if pattern:
            # Pattern-based invalidation
            keys_to_remove = []
            
            if self.config.enable_memory_cache:
                with self._cache_lock:
                    for key in self._memory_cache:
                        if pattern in key:
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        self._memory_cache.pop(key, None)
                        self._cache_access_count.pop(key, None)
                        self._cache_access_time.pop(key, None)
            
            if self.redis_client:
                try:
                    keys = self.redis_client.keys(f"*{pattern}*")
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    self.logger.warning(f"Redis invalidation failed: {e}")
        else:
            # Clear all
            if self.config.enable_memory_cache:
                with self._cache_lock:
                    self._memory_cache.clear()
                    self._cache_access_count.clear()
                    self._cache_access_time.clear()
            
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    self.logger.warning(f"Redis flush failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'memory_cache_enabled': self.config.enable_memory_cache,
            'redis_cache_enabled': self.redis_client is not None,
        }
        
        if self.config.enable_memory_cache:
            with self._cache_lock:
                stats.update({
                    'memory_cache_size': len(self._memory_cache),
                    'memory_cache_max_size': self.config.memory_cache_size,
                    'total_access_count': sum(self._cache_access_count.values())
                })
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats['redis_used_memory'] = info.get('used_memory', 0)
                stats['redis_keyspace_hits'] = info.get('keyspace_hits', 0)
                stats['redis_keyspace_misses'] = info.get('keyspace_misses', 0)
            except:
                pass
        
        return stats


class ResourceMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize resource monitor."""
        self.config = config
        self.logger = get_logger('performance')
        self._monitoring = False
        self._monitor_thread = None
        self._resource_history = []
        self._max_history = 1000
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self.get_current_metrics()
                
                # Add to history
                self._resource_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics
                })
                
                # Limit history size
                if len(self._resource_history) > self._max_history:
                    self._resource_history = self._resource_history[-self._max_history:]
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
        }
        
        # GPU metrics (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics.update({
                    'gpu_utilization': gpu.load * 100,
                    'gpu_memory_percent': gpu.memoryUtil * 100,
                    'gpu_temperature': gpu.temperature
                })
        except:
            pass
        
        return metrics
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check if metrics exceed thresholds."""
        if metrics['memory_percent'] > self.config.memory_usage_threshold * 100:
            dna_logger.log_security_event(
                'high_memory_usage',
                {'memory_percent': metrics['memory_percent']},
                'WARN'
            )
        
        if metrics['cpu_percent'] > self.config.cpu_usage_threshold * 100:
            dna_logger.log_security_event(
                'high_cpu_usage',
                {'cpu_percent': metrics['cpu_percent']},
                'WARN'
            )
    
    def get_resource_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get resource history for specified minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [
            entry for entry in self._resource_history
            if entry['timestamp'] > cutoff_time
        ]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self._resource_history:
            return {'error': 'No monitoring data available'}
        
        recent_metrics = [entry['metrics'] for entry in self._resource_history[-60:]]
        
        report = {
            'monitoring_duration_minutes': len(self._resource_history) * 5 / 60,
            'current_metrics': self.get_current_metrics(),
            'averages': {},
            'peaks': {},
            'recommendations': []
        }
        
        # Calculate averages and peaks
        for metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
            values = [m.get(metric, 0) for m in recent_metrics]
            if values:
                report['averages'][metric] = np.mean(values)
                report['peaks'][metric] = np.max(values)
        
        # Generate recommendations
        avg_cpu = report['averages'].get('cpu_percent', 0)
        avg_memory = report['averages'].get('memory_percent', 0)
        
        if avg_cpu > 80:
            report['recommendations'].append("Consider adding more CPU cores or optimizing computations")
        
        if avg_memory > 80:
            report['recommendations'].append("Consider increasing memory or optimizing memory usage")
        
        return report


class ConcurrentProcessor:
    """Concurrent processing for compute-intensive operations."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize concurrent processor."""
        self.config = config
        self.logger = get_logger('performance')
        
        # Thread pool for I/O bound tasks
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_worker_threads
        )
        
        # Process pool for CPU bound tasks
        self.process_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=config.max_worker_processes
        )
    
    async def process_batch_async(self, 
                                  items: List[Any], 
                                  processor_func: Callable,
                                  batch_size: Optional[int] = None,
                                  use_processes: bool = False) -> List[Any]:
        """Process items in batches asynchronously."""
        batch_size = batch_size or self.config.optimal_batch_size
        executor = self.process_executor if use_processes else self.thread_executor
        
        # Split into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Submit batch processing tasks
        loop = asyncio.get_event_loop()
        tasks = []
        
        for batch in batches:
            task = loop.run_in_executor(
                executor,
                self._process_batch_wrapper,
                processor_func,
                batch
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        flattened_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing error: {result}")
                continue
            flattened_results.extend(result)
        
        return flattened_results
    
    def _process_batch_wrapper(self, processor_func: Callable, batch: List[Any]) -> List[Any]:
        """Wrapper for batch processing."""
        return [processor_func(item) for item in batch]
    
    def process_parallel(self, 
                        items: List[Any], 
                        processor_func: Callable,
                        use_processes: bool = False) -> List[Any]:
        """Process items in parallel (synchronous)."""
        executor = self.process_executor if use_processes else self.thread_executor
        
        # Submit all tasks
        futures = [executor.submit(processor_func, item) for item in items]
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=300):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Parallel processing error: {e}")
        
        return results
    
    def shutdown(self):
        """Shutdown executors."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class AutoScaler:
    """Automatic scaling based on load metrics."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize auto-scaler."""
        self.config = config
        self.logger = get_logger('performance')
        self.current_replicas = config.min_replicas
        self.scaling_history = []
    
    def should_scale_up(self, current_load: float) -> bool:
        """Check if should scale up."""
        return (current_load > self.config.scale_up_threshold and 
                self.current_replicas < self.config.max_replicas)
    
    def should_scale_down(self, current_load: float) -> bool:
        """Check if should scale down."""
        return (current_load < self.config.scale_down_threshold and 
                self.current_replicas > self.config.min_replicas)
    
    def scale_up(self) -> int:
        """Scale up replicas."""
        if self.current_replicas < self.config.max_replicas:
            self.current_replicas += 1
            self._log_scaling_event('scale_up')
        return self.current_replicas
    
    def scale_down(self) -> int:
        """Scale down replicas."""
        if self.current_replicas > self.config.min_replicas:
            self.current_replicas -= 1
            self._log_scaling_event('scale_down')
        return self.current_replicas
    
    def _log_scaling_event(self, event_type: str):
        """Log scaling events."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'replicas': self.current_replicas
        }
        self.scaling_history.append(event)
        
        dna_logger.log_security_event(
            f'autoscaling_{event_type}',
            {'replicas': self.current_replicas}
        )
    
    def get_scaling_recommendation(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Get scaling recommendation based on metrics."""
        # Calculate combined load score
        cpu_load = metrics.get('cpu_percent', 0) / 100
        memory_load = metrics.get('memory_percent', 0) / 100
        combined_load = (cpu_load + memory_load) / 2
        
        if self.should_scale_up(combined_load):
            return 'scale_up'
        elif self.should_scale_down(combined_load):
            return 'scale_down'
        
        return None


# Decorators for performance optimization
def cached(ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, '_cache'):
                wrapper._cache = global_cache
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = global_cache._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = wrapper._cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            wrapper._cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def async_batch_process(batch_size: int = None, use_processes: bool = False):
    """Decorator for async batch processing."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(items: List[Any], *args, **kwargs):
            processor = global_concurrent_processor
            
            def process_item(item):
                return func(item, *args, **kwargs)
            
            return await processor.process_batch_async(
                items, 
                process_item,
                batch_size=batch_size,
                use_processes=use_processes
            )
        return wrapper
    return decorator


@contextmanager
def performance_monitor(operation: str):
    """Context manager for performance monitoring."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    logger = get_logger('performance')
    logger.info(f"Starting operation: {operation}")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.info(
            f"Operation completed: {operation}",
            extra={
                'metrics': {
                    'duration_seconds': duration,
                    'memory_delta_mb': memory_delta,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory
                }
            }
        )


# Global instances
performance_config = PerformanceConfig()
global_cache = AdaptiveCache(performance_config)
resource_monitor = ResourceMonitor(performance_config)
global_concurrent_processor = ConcurrentProcessor(performance_config)
auto_scaler = AutoScaler(performance_config)

# Start monitoring
resource_monitor.start_monitoring()