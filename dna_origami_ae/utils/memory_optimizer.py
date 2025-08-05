"""Memory optimization utilities for large-scale processing."""

import gc
import psutil
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union, Iterator, Tuple
from contextlib import contextmanager
import threading
from dataclasses import dataclass
from functools import wraps
import warnings

from .helpers import logger, format_size


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_mb: float
    
    def __str__(self) -> str:
        return (f"Memory: {self.used_mb:.1f}/{self.total_mb:.1f} MB "
                f"({self.percent_used:.1f}%), "
                f"Process: {self.process_mb:.1f} MB")


class MemoryMonitor:
    """Monitor memory usage and provide optimization suggestions."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        """Initialize memory monitor.
        
        Args:
            warning_threshold: Warn when memory usage exceeds this fraction
            critical_threshold: Critical alert when memory usage exceeds this fraction
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.peak_usage = 0.0
        self.warning_callback: Optional[Callable] = None
        self.critical_callback: Optional[Callable] = None
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        total_mb = memory.total / (1024 ** 2)
        available_mb = memory.available / (1024 ** 2)
        used_mb = memory.used / (1024 ** 2)
        percent_used = memory.percent / 100.0
        process_mb = process_memory.rss / (1024 ** 2)
        
        # Update peak usage
        self.peak_usage = max(self.peak_usage, percent_used)
        
        return MemoryStats(
            total_mb=total_mb,
            available_mb=available_mb,
            used_mb=used_mb,
            percent_used=percent_used,
            process_mb=process_mb
        )
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check for memory pressure and return recommendations."""
        stats = self.get_memory_stats()
        
        recommendations = []
        alert_level = "normal"
        
        if stats.percent_used > self.critical_threshold:
            alert_level = "critical"
            recommendations.extend([
                "Memory usage is critically high",
                "Consider reducing batch sizes",
                "Free unused variables with del",
                "Force garbage collection",
                "Consider processing data in smaller chunks"
            ])
            
            if self.critical_callback:
                self.critical_callback(stats)
        
        elif stats.percent_used > self.warning_threshold:
            alert_level = "warning"
            recommendations.extend([
                "Memory usage is high",
                "Monitor memory usage closely",
                "Consider optimizing data structures"
            ])
            
            if self.warning_callback:
                self.warning_callback(stats)
        
        return {
            'stats': stats,
            'alert_level': alert_level,
            'recommendations': recommendations,
            'peak_usage': self.peak_usage
        }
    
    def set_callbacks(self, warning_callback: Optional[Callable] = None,
                     critical_callback: Optional[Callable] = None):
        """Set callbacks for memory alerts."""
        self.warning_callback = warning_callback
        self.critical_callback = critical_callback


@contextmanager
def memory_limit(max_memory_mb: float):
    """Context manager to enforce memory limits."""
    monitor = MemoryMonitor()
    
    def check_limit():
        stats = monitor.get_memory_stats()
        if stats.process_mb > max_memory_mb:
            raise MemoryError(f"Process memory usage ({stats.process_mb:.1f} MB) "
                            f"exceeds limit ({max_memory_mb:.1f} MB)")
    
    try:
        check_limit()
        yield monitor
    finally:
        check_limit()


def memory_efficient_array_ops():
    """Configure numpy for memory-efficient operations."""
    # Use memory mapping for large arrays when possible
    np.seterr(over='warn')  # Warn on overflow instead of error
    
    # Set up memory mapping defaults
    if hasattr(np, 'memmap'):
        logger.info("Memory mapping available for large arrays")


class ChunkedArrayProcessor:
    """Process large arrays in memory-efficient chunks."""
    
    def __init__(self, chunk_size_mb: float = 100.0, overlap: int = 0):
        """Initialize chunked processor.
        
        Args:
            chunk_size_mb: Target chunk size in MB
            overlap: Number of elements to overlap between chunks
        """
        self.chunk_size_mb = chunk_size_mb
        self.overlap = overlap
        self.monitor = MemoryMonitor()
    
    def calculate_chunk_size(self, array: np.ndarray, target_mb: float) -> int:
        """Calculate optimal chunk size for array."""
        element_size = array.itemsize
        elements_per_mb = (1024 ** 2) / element_size
        chunk_elements = int(target_mb * elements_per_mb)
        
        # Ensure chunk size is reasonable
        chunk_elements = min(chunk_elements, len(array))
        chunk_elements = max(chunk_elements, 1)
        
        return chunk_elements
    
    def process_array_chunked(self, array: np.ndarray, 
                            process_func: Callable,
                            combine_func: Optional[Callable] = None) -> np.ndarray:
        """Process large array in chunks.
        
        Args:
            array: Large array to process
            process_func: Function to apply to each chunk
            combine_func: Function to combine chunk results
        
        Returns:
            Processed array
        """
        chunk_size = self.calculate_chunk_size(array, self.chunk_size_mb)
        
        if chunk_size >= len(array):
            # Array is small enough to process all at once
            return process_func(array)
        
        logger.info(f"Processing array of {len(array)} elements in chunks of {chunk_size}")
        
        results = []
        
        for start in range(0, len(array), chunk_size - self.overlap):
            end = min(start + chunk_size, len(array))
            chunk = array[start:end]
            
            # Process chunk
            try:
                result = process_func(chunk)
                results.append(result)
                
                # Check memory after each chunk
                pressure = self.monitor.check_memory_pressure()
                if pressure['alert_level'] == 'critical':
                    logger.warning("Critical memory pressure detected, forcing garbage collection")
                    gc.collect()
                
            except MemoryError as e:
                logger.error(f"Memory error processing chunk {start}:{end}: {e}")
                # Try to recover by forcing garbage collection
                gc.collect()
                raise
            
            if end >= len(array):
                break
        
        # Combine results
        if combine_func:
            return combine_func(results)
        elif results and isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        else:
            return results


def optimize_numpy_memory():
    """Optimize numpy memory usage."""
    # Set numpy to use less memory for small arrays
    np.seterr(all='warn')
    
    # Use float32 instead of float64 when precision allows
    logger.info("Consider using float32 instead of float64 for memory savings")


class MemoryEfficientDataLoader:
    """Load and process data with memory efficiency."""
    
    def __init__(self, batch_size: int = 32, prefetch_batches: int = 2):
        """Initialize memory-efficient data loader.
        
        Args:
            batch_size: Size of each batch
            prefetch_batches: Number of batches to prefetch
        """
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.monitor = MemoryMonitor()
    
    def load_batches(self, data_source: Union[List, Iterator], 
                    preprocess_func: Optional[Callable] = None) -> Iterator:
        """Load data in memory-efficient batches.
        
        Args:
            data_source: Source of data (list or iterator)
            preprocess_func: Optional preprocessing function
        
        Yields:
            Batches of processed data
        """
        if isinstance(data_source, list):
            data_iter = iter(data_source)
        else:
            data_iter = data_source
        
        batch = []
        
        for item in data_iter:
            if preprocess_func:
                item = preprocess_func(item)
            
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
                
                # Check memory pressure
                pressure = self.monitor.check_memory_pressure()
                if pressure['alert_level'] == 'warning':
                    logger.debug("Memory pressure detected, yielding smaller batches")
                    self.batch_size = max(1, self.batch_size // 2)
        
        # Yield remaining items
        if batch:
            yield batch


def memory_profile(func: Callable) -> Callable:
    """Decorator to profile memory usage of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        
        # Get initial memory
        initial_stats = monitor.get_memory_stats()
        logger.debug(f"Memory before {func.__name__}: {initial_stats}")
        
        try:
            result = func(*args, **kwargs)
            
            # Get final memory
            final_stats = monitor.get_memory_stats()
            memory_delta = final_stats.process_mb - initial_stats.process_mb
            
            logger.info(f"Memory usage for {func.__name__}: "
                       f"{memory_delta:+.1f} MB "
                       f"(Peak: {monitor.peak_usage:.1%})")
            
            return result
            
        except Exception as e:
            # Check if memory-related error
            final_stats = monitor.get_memory_stats()
            if isinstance(e, MemoryError) or 'memory' in str(e).lower():
                logger.error(f"Memory error in {func.__name__}: {final_stats}")
            raise
    
    return wrapper


def optimize_array_dtype(array: np.ndarray, preserve_precision: bool = True) -> np.ndarray:
    """Optimize array data type for memory efficiency.
    
    Args:
        array: Input array
        preserve_precision: Whether to preserve numerical precision
    
    Returns:
        Array with optimized dtype
    """
    if array.dtype == np.float64:
        if not preserve_precision or np.allclose(array, array.astype(np.float32)):
            logger.debug("Converting float64 to float32 for memory savings")
            return array.astype(np.float32)
    
    elif array.dtype == np.int64:
        max_val = np.max(np.abs(array))
        if max_val < 2**31:
            logger.debug("Converting int64 to int32 for memory savings")
            return array.astype(np.int32)
        elif max_val < 2**15:
            logger.debug("Converting int64 to int16 for memory savings")
            return array.astype(np.int16)
        elif max_val < 2**7:
            logger.debug("Converting int64 to int8 for memory savings")
            return array.astype(np.int8)
    
    return array


class MemoryPool:
    """Simple memory pool for reusing arrays."""
    
    def __init__(self, max_size: int = 100):
        """Initialize memory pool.
        
        Args:
            max_size: Maximum number of arrays to pool
        """
        self.max_size = max_size
        self.pools = {}  # dtype -> list of arrays
        self.lock = threading.Lock()
    
    def get_array(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Get array from pool or create new one.
        
        Args:
            shape: Required array shape
            dtype: Required array data type
        
        Returns:
            Array with requested shape and dtype
        """
        with self.lock:
            key = (shape, dtype)
            if key in self.pools and self.pools[key]:
                array = self.pools[key].pop()
                array.fill(0)  # Clear array
                return array
            else:
                return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray):
        """Return array to pool for reuse.
        
        Args:
            array: Array to return to pool
        """
        with self.lock:
            key = (array.shape, array.dtype)
            if key not in self.pools:
                self.pools[key] = []
            
            if len(self.pools[key]) < self.max_size:
                self.pools[key].append(array)
    
    def clear(self):
        """Clear all pooled arrays."""
        with self.lock:
            self.pools.clear()
            gc.collect()


# Global memory pool instance
memory_pool = MemoryPool()


@contextmanager
def temporary_array(shape: Tuple[int, ...], dtype: np.dtype = np.float32):
    """Context manager for temporary arrays using memory pool."""
    array = memory_pool.get_array(shape, dtype)
    try:
        yield array
    finally:
        memory_pool.return_array(array)


def suggest_memory_optimizations(array: np.ndarray) -> List[str]:
    """Suggest memory optimizations for array."""
    suggestions = []
    
    # Check data type
    if array.dtype == np.float64:
        suggestions.append("Consider using float32 instead of float64")
    elif array.dtype == np.int64:
        max_val = np.max(np.abs(array))
        if max_val < 2**31:
            suggestions.append("Consider using int32 instead of int64")
    
    # Check sparsity
    zeros = np.count_nonzero(array == 0)
    sparsity = zeros / array.size
    if sparsity > 0.5:
        suggestions.append(f"Array is {sparsity:.1%} sparse, consider sparse matrices")
    
    # Check memory usage
    memory_mb = array.nbytes / (1024 ** 2)
    if memory_mb > 100:
        suggestions.append(f"Large array ({memory_mb:.1f} MB), consider chunked processing")
    
    return suggestions


def memory_efficient_copy(array: np.ndarray, 
                         target_dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Create memory-efficient copy of array."""
    if target_dtype and target_dtype != array.dtype:
        # Convert dtype during copy
        return array.astype(target_dtype, copy=True)
    else:
        # Regular copy
        return array.copy()


class GarbageCollectionManager:
    """Manage garbage collection for memory optimization."""
    
    def __init__(self, auto_collect_threshold: int = 1000):
        """Initialize GC manager.
        
        Args:
            auto_collect_threshold: Automatically collect after this many allocations
        """
        self.auto_collect_threshold = auto_collect_threshold
        self.allocation_count = 0
        self.lock = threading.Lock()
    
    def maybe_collect(self):
        """Conditionally run garbage collection."""
        with self.lock:
            self.allocation_count += 1
            if self.allocation_count >= self.auto_collect_threshold:
                collected = gc.collect()
                self.allocation_count = 0
                if collected > 0:
                    logger.debug(f"Garbage collected {collected} objects")
    
    def force_collect(self) -> int:
        """Force garbage collection."""
        with self.lock:
            collected = gc.collect()
            self.allocation_count = 0
            logger.info(f"Forced garbage collection: {collected} objects collected")
            return collected


# Global GC manager
gc_manager = GarbageCollectionManager()


# Convenience functions
def monitor_memory() -> MemoryStats:
    """Get current memory statistics."""
    monitor = MemoryMonitor()
    return monitor.get_memory_stats()


def free_memory():
    """Free up memory by running garbage collection."""
    return gc_manager.force_collect()