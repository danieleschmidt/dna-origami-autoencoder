"""Parallel processing utilities for performance optimization."""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional, Dict, Union, Iterator, Tuple
import time
import threading
from functools import partial
import numpy as np
from dataclasses import dataclass

from .helpers import logger, Timer


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: Optional[int] = None
    chunk_size: int = 1
    use_threads: bool = False  # True for I/O bound, False for CPU bound
    timeout: Optional[float] = None
    progress_callback: Optional[Callable[[int, int], None]] = None
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(32, (os.cpu_count() or 1) + 4)


class ProgressTracker:
    """Thread-safe progress tracking."""
    
    def __init__(self, total_tasks: int, callback: Optional[Callable[[int, int], None]] = None):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.callback = callback
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress counter."""
        with self.lock:
            self.completed_tasks += increment
            if self.callback:
                self.callback(self.completed_tasks, self.total_tasks)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self.lock:
            elapsed = time.time() - self.start_time
            rate = self.completed_tasks / elapsed if elapsed > 0 else 0
            eta = (self.total_tasks - self.completed_tasks) / rate if rate > 0 else 0
            
            return {
                'completed': self.completed_tasks,
                'total': self.total_tasks,
                'progress': self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0,
                'elapsed_time': elapsed,
                'rate': rate,
                'eta': eta
            }


def parallel_map(func: Callable, iterable: List[Any], 
                config: Optional[ParallelConfig] = None) -> List[Any]:
    """Apply function to items in parallel.
    
    Args:
        func: Function to apply to each item
        iterable: List of items to process
        config: Parallel processing configuration
    
    Returns:
        List of results in same order as input
    """
    if config is None:
        config = ParallelConfig()
    
    if not iterable:
        return []
    
    # Single item or very small list - don't use parallel processing
    if len(iterable) <= 1 or config.max_workers == 1:
        return [func(item) for item in iterable]
    
    executor_class = ThreadPoolExecutor if config.use_threads else ProcessPoolExecutor
    progress_tracker = ProgressTracker(len(iterable), config.progress_callback)
    
    with Timer(f"Parallel processing {len(iterable)} items"):
        with executor_class(max_workers=config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(func, item): i 
                for i, item in enumerate(iterable)
            }
            
            # Collect results in order
            results = [None] * len(iterable)
            
            try:
                for future in as_completed(future_to_index, timeout=config.timeout):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                        progress_tracker.update()
                    except Exception as e:
                        logger.error(f"Task {index} failed: {e}")
                        results[index] = None
                        progress_tracker.update()
            
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}")
                # Cancel remaining futures
                for future in future_to_index:
                    future.cancel()
                raise
    
    return results


def parallel_batch_process(func: Callable, items: List[Any], 
                          batch_size: int, config: Optional[ParallelConfig] = None) -> List[Any]:
    """Process items in parallel batches.
    
    Args:
        func: Function that takes a batch of items
        items: List of items to process
        batch_size: Size of each batch
        config: Parallel processing configuration
    
    Returns:
        Flattened list of results
    """
    if config is None:
        config = ParallelConfig()
    
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    # Process batches in parallel
    batch_results = parallel_map(func, batches, config)
    
    # Flatten results
    results = []
    for batch_result in batch_results:
        if isinstance(batch_result, list):
            results.extend(batch_result)
        else:
            results.append(batch_result)
    
    return results


def parallel_reduce(func: Callable, iterable: List[Any], 
                   reduce_func: Callable, initial_value: Any = None,
                   config: Optional[ParallelConfig] = None) -> Any:
    """Map-reduce pattern with parallel map phase.
    
    Args:
        func: Function to apply to each item (map phase)
        iterable: List of items to process
        reduce_func: Function to combine results (reduce phase)
        initial_value: Initial value for reduction
        config: Parallel processing configuration
    
    Returns:
        Reduced result
    """
    # Parallel map phase
    mapped_results = parallel_map(func, iterable, config)
    
    # Sequential reduce phase
    result = initial_value
    for item in mapped_results:
        if item is not None:
            if result is None:
                result = item
            else:
                result = reduce_func(result, item)
    
    return result


def chunked_parallel_process(func: Callable, large_array: np.ndarray,
                           chunk_size: int, overlap: int = 0,
                           config: Optional[ParallelConfig] = None) -> np.ndarray:
    """Process large numpy array in parallel chunks.
    
    Args:
        func: Function that processes array chunks
        large_array: Large array to process
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        config: Parallel processing configuration
    
    Returns:
        Processed array
    """
    if config is None:
        config = ParallelConfig()
    
    # Create overlapping chunks
    chunks = []
    chunk_indices = []
    
    for start in range(0, len(large_array), chunk_size - overlap):
        end = min(start + chunk_size, len(large_array))
        chunk = large_array[start:end]
        chunks.append(chunk)
        chunk_indices.append((start, end))
        
        if end >= len(large_array):
            break
    
    # Process chunks in parallel
    processed_chunks = parallel_map(func, chunks, config)
    
    # Combine results (handling overlap)
    if overlap == 0:
        # Simple concatenation
        return np.concatenate(processed_chunks)
    else:
        # Handle overlap by averaging overlapping regions
        result = np.zeros_like(large_array)
        counts = np.zeros(len(large_array))
        
        for i, (start, end) in enumerate(chunk_indices):
            if processed_chunks[i] is not None:
                chunk_length = end - start
                result[start:end] += processed_chunks[i][:chunk_length]
                counts[start:end] += 1
        
        # Average overlapping regions
        result = np.divide(result, counts, out=np.zeros_like(result), where=counts!=0)
        return result


class ParallelTaskManager:
    """Manage complex parallel processing workflows."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.tasks = {}
        self.results = {}
        self.dependencies = {}
        self.completed = set()
        self.lock = threading.Lock()
    
    def add_task(self, task_id: str, func: Callable, args: tuple = (), 
                 kwargs: dict = None, dependencies: List[str] = None):
        """Add a task to the workflow.
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            dependencies: List of task IDs that must complete first
        """
        with self.lock:
            self.tasks[task_id] = {
                'func': func,
                'args': args,
                'kwargs': kwargs or {}
            }
            self.dependencies[task_id] = dependencies or []
    
    def can_execute(self, task_id: str) -> bool:
        """Check if task can be executed (all dependencies completed)."""
        return all(dep in self.completed for dep in self.dependencies[task_id])
    
    def get_ready_tasks(self) -> List[str]:
        """Get list of tasks ready to execute."""
        ready = []
        for task_id in self.tasks:
            if task_id not in self.completed and self.can_execute(task_id):
                ready.append(task_id)
        return ready
    
    def execute_workflow(self, config: Optional[ParallelConfig] = None) -> Dict[str, Any]:
        """Execute all tasks respecting dependencies.
        
        Returns:
            Dictionary mapping task IDs to results
        """
        if config is None:
            config = ParallelConfig()
        
        executor_class = ThreadPoolExecutor if config.use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            active_futures = {}
            
            while len(self.completed) < len(self.tasks):
                # Submit ready tasks
                ready_tasks = self.get_ready_tasks()
                
                for task_id in ready_tasks:
                    if task_id not in active_futures:
                        task = self.tasks[task_id]
                        future = executor.submit(task['func'], *task['args'], **task['kwargs'])
                        active_futures[task_id] = future
                        logger.debug(f"Submitted task: {task_id}")
                
                # Check for completed tasks
                completed_futures = []
                for task_id, future in active_futures.items():
                    if future.done():
                        try:
                            result = future.result()
                            self.results[task_id] = result
                            self.completed.add(task_id)
                            completed_futures.append(task_id)
                            logger.debug(f"Completed task: {task_id}")
                        except Exception as e:
                            logger.error(f"Task {task_id} failed: {e}")
                            self.results[task_id] = None
                            self.completed.add(task_id)
                            completed_futures.append(task_id)
                
                # Remove completed futures
                for task_id in completed_futures:
                    del active_futures[task_id]
                
                # Short sleep to avoid busy waiting
                if active_futures:
                    time.sleep(0.01)
        
        return self.results


def parallel_dna_encoding(sequences: List[str], encoder_func: Callable,
                         config: Optional[ParallelConfig] = None) -> List[Any]:
    """Encode DNA sequences in parallel."""
    if config is None:
        config = ParallelConfig()
    
    logger.info(f"Encoding {len(sequences)} DNA sequences in parallel")
    
    # For DNA encoding, use process-based parallelism for CPU-intensive work
    config.use_threads = False
    
    return parallel_map(encoder_func, sequences, config)


def parallel_image_processing(images: List[Any], process_func: Callable,
                            config: Optional[ParallelConfig] = None) -> List[Any]:
    """Process images in parallel."""
    if config is None:
        config = ParallelConfig()
    
    logger.info(f"Processing {len(images)} images in parallel")
    
    # Image processing is typically CPU-bound
    config.use_threads = False
    
    return parallel_map(process_func, images, config)


def parallel_simulation_batch(structures: List[Any], simulation_func: Callable,
                            config: Optional[ParallelConfig] = None) -> List[Any]:
    """Run molecular simulations in parallel."""
    if config is None:
        config = ParallelConfig()
    
    logger.info(f"Running {len(structures)} simulations in parallel")
    
    # Simulations are CPU-intensive and may benefit from fewer workers
    # to avoid memory pressure
    config.max_workers = min(config.max_workers or 1, 4)
    config.use_threads = False
    
    return parallel_map(simulation_func, structures, config)


def adaptive_parallel_config(task_type: str, data_size: int) -> ParallelConfig:
    """Create adaptive parallel configuration based on task type and data size.
    
    Args:
        task_type: Type of task ('cpu_bound', 'io_bound', 'memory_intensive')
        data_size: Size of data to process
    
    Returns:
        Optimized parallel configuration
    """
    cpu_count = os.cpu_count() or 1
    
    if task_type == 'cpu_bound':
        # Use all CPU cores for CPU-bound tasks
        max_workers = cpu_count
        use_threads = False
        chunk_size = max(1, data_size // (cpu_count * 4))
    
    elif task_type == 'io_bound':
        # Use more threads for I/O bound tasks
        max_workers = min(32, cpu_count * 4)
        use_threads = True
        chunk_size = max(1, data_size // max_workers)
    
    elif task_type == 'memory_intensive':
        # Use fewer workers to avoid memory pressure
        max_workers = max(1, cpu_count // 2)
        use_threads = False
        chunk_size = max(1, data_size // max_workers)
    
    else:
        # Default configuration
        max_workers = cpu_count
        use_threads = False
        chunk_size = max(1, data_size // max_workers)
    
    return ParallelConfig(
        max_workers=max_workers,
        use_threads=use_threads,
        chunk_size=chunk_size
    )


def benchmark_parallel_performance(func: Callable, test_data: List[Any],
                                 configs: List[ParallelConfig]) -> Dict[str, Any]:
    """Benchmark different parallel configurations.
    
    Args:
        func: Function to benchmark
        test_data: Test data to process
        configs: List of configurations to test
    
    Returns:
        Benchmark results
    """
    results = {}
    
    # Sequential baseline
    with Timer("Sequential") as sequential_timer:
        sequential_result = [func(item) for item in test_data]
    
    results['sequential'] = {
        'time': sequential_timer.duration,
        'throughput': len(test_data) / sequential_timer.duration
    }
    
    # Test each parallel configuration
    for i, config in enumerate(configs):
        config_name = f"parallel_{i}"
        
        try:
            with Timer(config_name) as parallel_timer:
                parallel_result = parallel_map(func, test_data, config)
            
            # Verify results match
            results_match = len(parallel_result) == len(sequential_result)
            
            results[config_name] = {
                'time': parallel_timer.duration,
                'throughput': len(test_data) / parallel_timer.duration,
                'speedup': sequential_timer.duration / parallel_timer.duration,
                'results_match': results_match,
                'config': config
            }
            
        except Exception as e:
            results[config_name] = {
                'error': str(e),
                'config': config
            }
    
    return results


# Convenient aliases for common parallel patterns
parallel_encode = partial(parallel_dna_encoding)
parallel_process_images = partial(parallel_image_processing)
parallel_simulate = partial(parallel_simulation_batch)