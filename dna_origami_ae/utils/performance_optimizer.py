"""Performance optimization and scaling for DNA origami encoding."""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json
import logging
from collections import defaultdict, deque
import weakref

from ..models.image_data import ImageData
from ..models.dna_sequence import DNASequence


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    data: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl_seconds)
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class AdaptiveCache:
    """Adaptive caching system with intelligent eviction."""
    
    def __init__(self, 
                 max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
                 max_entries: int = 1000,
                 default_ttl: Optional[int] = None):
        """Initialize adaptive cache."""
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = deque()  # For LRU tracking
        self._current_size = 0
        self._lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_evictions': 0,
            'ttl_evictions': 0
        }
        
        # Adaptive parameters
        self._hit_rates = deque(maxlen=100)  # Track recent hit rates
        self._optimal_ttl = default_ttl
        
        # Background cleanup
        self._cleanup_thread = None
        self._cleanup_interval = 60  # seconds
        self._shutdown = False
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        while not self._shutdown:
            try:
                self._cleanup_expired()
                time.sleep(self._cleanup_interval)
            except Exception:
                pass  # Ignore cleanup errors
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key, reason='ttl')
    
    def _calculate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        try:
            return len(pickle.dumps(data))
        except Exception:
            # Fallback estimation
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (list, tuple)):
                return sum(self._calculate_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in data.items())
            else:
                return 1024  # Default estimate
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = (args, tuple(sorted(kwargs.items())))
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        with self._lock:
            while (self._current_size > self.max_size_bytes or 
                   len(self._cache) > self.max_entries):
                
                if not self._access_order:
                    break
                
                lru_key = self._access_order.popleft()
                if lru_key in self._cache:
                    self._remove_entry(lru_key, reason='size')
    
    def _remove_entry(self, key: str, reason: str = 'manual') -> None:
        """Remove cache entry and update statistics."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_size -= entry.size_bytes
            del self._cache[key]
            
            # Remove from access order
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
            
            # Update statistics
            self.stats['evictions'] += 1
            if reason == 'size':
                self.stats['size_evictions'] += 1
            elif reason == 'ttl':
                self.stats['ttl_evictions'] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self.stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key, reason='ttl')
                self.stats['misses'] += 1
                return None
            
            # Update access
            entry.update_access()
            
            # Move to end of access order (most recent)
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
            self._access_order.append(key)
            
            self.stats['hits'] += 1
            return entry.data
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Put item in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(data)
            
            # Check if item is too large
            if size_bytes > self.max_size_bytes:
                return  # Cannot cache
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create cache entry
            entry = CacheEntry(
                data=data,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl
            )
            
            # Add to cache
            self._cache[key] = entry
            self._current_size += size_bytes
            self._access_order.append(key)
            
            # Evict if necessary
            self._evict_lru()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self.stats,
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'current_entries': len(self._cache),
                'current_size_bytes': self._current_size,
                'size_utilization': self._current_size / self.max_size_bytes,
                'entry_utilization': len(self._cache) / self.max_entries
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_size = 0
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup thread."""
        self._shutdown = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)


class ConcurrentProcessor:
    """Concurrent processing for DNA encoding operations."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 enable_async: bool = True,
                 chunk_size: int = 1000):
        """Initialize concurrent processor."""
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.enable_async = enable_async
        self.chunk_size = chunk_size
        
        # Thread pool for I/O bound operations
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="dna_encoder"
        )
        
        # Process pool for CPU bound operations
        self.process_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(self.max_workers, mp.cpu_count() or 1)
        )
        
        self.logger = logging.getLogger(__name__)
    
    def process_chunks_parallel(self, 
                              chunks: List[Any],
                              process_func: Callable,
                              use_processes: bool = False,
                              **kwargs) -> List[Any]:
        """Process chunks in parallel."""
        if len(chunks) <= 1:
            return [process_func(chunk, **kwargs) for chunk in chunks]
        
        executor = self.process_executor if use_processes else self.thread_executor
        
        try:
            futures = [
                executor.submit(process_func, chunk, **kwargs)
                for chunk in chunks
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=300):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Chunk processing failed: {e}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            return [process_func(chunk, **kwargs) for chunk in chunks]
    
    async def process_chunks_async(self,
                                 chunks: List[Any],
                                 async_func: Callable,
                                 **kwargs) -> List[Any]:
        """Process chunks asynchronously."""
        if not self.enable_async:
            return [await async_func(chunk, **kwargs) for chunk in chunks]
        
        tasks = [async_func(chunk, **kwargs) for chunk in chunks]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Async processing failed: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Async processing failed: {e}")
            return [None] * len(chunks)
    
    def shutdown(self) -> None:
        """Shutdown all executors."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class AutoScaler:
    """Auto-scaling system for processing resources."""
    
    def __init__(self, 
                 min_workers: int = 1,
                 max_workers: Optional[int] = None,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 evaluation_window: int = 10):
        """Initialize auto-scaler."""
        self.min_workers = min_workers
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) * 2)
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.evaluation_window = evaluation_window
        
        # Current state
        self.current_workers = min_workers
        self.load_history = deque(maxlen=evaluation_window)
        self.last_scale_time = datetime.now()
        self.min_scale_interval = timedelta(seconds=30)
        
        # Statistics
        self.scale_events = []
        
    def record_load(self, load_metric: float) -> None:
        """Record load metric (0.0 to 1.0)."""
        self.load_history.append({
            'load': load_metric,
            'timestamp': datetime.now()
        })
    
    def get_scaling_decision(self) -> Dict[str, Any]:
        """Get scaling decision based on current load."""
        if len(self.load_history) < self.evaluation_window:
            return {'action': 'wait', 'current_workers': self.current_workers}
        
        # Calculate average load
        avg_load = np.mean([entry['load'] for entry in self.load_history])
        
        # Check if enough time has passed since last scaling
        time_since_scale = datetime.now() - self.last_scale_time
        if time_since_scale < self.min_scale_interval:
            return {'action': 'wait', 'current_workers': self.current_workers}
        
        # Scaling decision
        if avg_load > self.scale_up_threshold and self.current_workers < self.max_workers:
            new_workers = min(self.current_workers + 1, self.max_workers)
            return {
                'action': 'scale_up',
                'current_workers': self.current_workers,
                'new_workers': new_workers,
                'avg_load': avg_load
            }
        elif avg_load < self.scale_down_threshold and self.current_workers > self.min_workers:
            new_workers = max(self.current_workers - 1, self.min_workers)
            return {
                'action': 'scale_down',
                'current_workers': self.current_workers,
                'new_workers': new_workers,
                'avg_load': avg_load
            }
        else:
            return {'action': 'maintain', 'current_workers': self.current_workers}
    
    def apply_scaling(self, decision: Dict[str, Any]) -> bool:
        """Apply scaling decision."""
        if decision['action'] in ['scale_up', 'scale_down']:
            self.current_workers = decision['new_workers']
            self.last_scale_time = datetime.now()
            
            # Record scaling event
            self.scale_events.append({
                'timestamp': datetime.now(),
                'action': decision['action'],
                'from_workers': decision['current_workers'],
                'to_workers': decision['new_workers'],
                'avg_load': decision.get('avg_load', 0.0)
            })
            
            return True
        
        return False


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self,
                 enable_caching: bool = True,
                 enable_parallel: bool = True,
                 enable_autoscaling: bool = True,
                 cache_size_mb: int = 100):
        """Initialize performance optimizer."""
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.enable_autoscaling = enable_autoscaling
        
        # Initialize components
        if enable_caching:
            self.cache = AdaptiveCache(
                max_size_bytes=cache_size_mb * 1024 * 1024,
                default_ttl=3600  # 1 hour
            )
        else:
            self.cache = None
        
        if enable_parallel:
            self.processor = ConcurrentProcessor()
        else:
            self.processor = None
        
        if enable_autoscaling:
            self.autoscaler = AutoScaler()
        else:
            self.autoscaler = None
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self.optimization_stats = {
            'cache_saves': 0,
            'parallel_speedups': 0,
            'autoscale_events': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def cached_operation(self, 
                        cache_key: str,
                        operation: Callable,
                        *args,
                        ttl: Optional[int] = None,
                        **kwargs) -> Any:
        """Execute operation with caching."""
        if not self.cache:
            return operation(*args, **kwargs)
        
        # Try cache first
        result = self.cache.get(cache_key)
        if result is not None:
            self.optimization_stats['cache_saves'] += 1
            return result
        
        # Execute operation
        result = operation(*args, **kwargs)
        
        # Cache result
        self.cache.put(cache_key, result, ttl=ttl)
        
        return result
    
    def parallel_operation(self,
                         data_chunks: List[Any],
                         operation: Callable,
                         use_processes: bool = False,
                         **kwargs) -> List[Any]:
        """Execute operation in parallel."""
        if not self.processor or len(data_chunks) <= 1:
            return [operation(chunk, **kwargs) for chunk in data_chunks]
        
        start_time = time.time()
        
        # Parallel execution
        results = self.processor.process_chunks_parallel(
            chunks=data_chunks,
            process_func=operation,
            use_processes=use_processes,
            **kwargs
        )
        
        parallel_time = time.time() - start_time
        
        # Estimate sequential time (rough)
        estimated_sequential_time = parallel_time * len(data_chunks)
        
        if estimated_sequential_time > parallel_time * 1.2:  # 20% speedup threshold
            self.optimization_stats['parallel_speedups'] += 1
        
        return results
    
    def adaptive_batch_size(self, 
                          total_items: int,
                          operation_name: str,
                          base_batch_size: int = 100) -> int:
        """Calculate adaptive batch size based on performance history."""
        if operation_name not in self.operation_times:
            return base_batch_size
        
        recent_times = self.operation_times[operation_name][-10:]  # Last 10 operations
        
        if not recent_times:
            return base_batch_size
        
        avg_time_per_item = np.mean(recent_times)
        
        # Adjust batch size based on performance
        if avg_time_per_item < 0.01:  # Very fast operations
            return min(base_batch_size * 2, total_items)
        elif avg_time_per_item > 0.1:  # Slow operations
            return max(base_batch_size // 2, 1)
        else:
            return base_batch_size
    
    def record_operation_time(self, 
                            operation_name: str,
                            execution_time: float) -> None:
        """Record operation execution time."""
        self.operation_times[operation_name].append(execution_time)
        
        # Keep only recent times
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][-50:]
        
        # Update autoscaler if enabled
        if self.autoscaler:
            # Convert execution time to load metric (normalized)
            load_metric = min(execution_time / 10.0, 1.0)  # Assume 10s is 100% load
            self.autoscaler.record_load(load_metric)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        report = {
            'optimization_stats': self.optimization_stats.copy(),
            'cache_stats': self.cache.get_stats() if self.cache else {},
            'autoscaler_stats': {
                'current_workers': self.autoscaler.current_workers if self.autoscaler else 1,
                'scale_events': len(self.autoscaler.scale_events) if self.autoscaler else 0
            },
            'operation_performance': {}
        }
        
        # Add operation performance summary
        for op_name, times in self.operation_times.items():
            if times:
                report['operation_performance'][op_name] = {
                    'count': len(times),
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times)
                }
        
        return report
    
    def shutdown(self) -> None:
        """Shutdown all optimization components."""
        if self.cache:
            self.cache.shutdown()
        
        if self.processor:
            self.processor.shutdown()


def performance_optimized(cache_key_func: Optional[Callable] = None,
                        enable_parallel: bool = False,
                        use_processes: bool = False):
    """Decorator for performance optimization."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get or create optimizer
            if not hasattr(wrapper, '_optimizer'):
                wrapper._optimizer = PerformanceOptimizer()
            
            optimizer = wrapper._optimizer
            
            # Generate cache key
            if cache_key_func and optimizer.cache:
                cache_key = cache_key_func(*args, **kwargs)
                return optimizer.cached_operation(cache_key, func, *args, **kwargs)
            
            # Record timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                optimizer.record_operation_time(func.__name__, execution_time)
        
        return wrapper
    return decorator