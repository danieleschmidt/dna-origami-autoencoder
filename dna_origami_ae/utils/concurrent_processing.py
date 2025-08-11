"""Advanced concurrent processing with adaptive scaling and load balancing."""

import threading
import multiprocessing
import asyncio
import queue
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
import psutil
import resource
import weakref
from collections import defaultdict, deque
import numpy as np

from .logger import get_logger
from .advanced_error_handling import robust_execution


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ProcessingMode(Enum):
    """Processing execution modes."""
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"
    AUTO = "auto"


@dataclass
class Task:
    """Task representation with metadata."""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[Exception] = None
    result: Any = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    @property
    def is_completed(self) -> bool:
        return self.completed_at is not None
    
    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class ResourceMonitor:
    """Monitor system resources for adaptive scaling."""
    
    def __init__(self):
        self.cpu_history = deque(maxlen=60)  # Last 60 measurements
        self.memory_history = deque(maxlen=60)
        self.load_history = deque(maxlen=60)
        self._lock = threading.Lock()
        
    def update(self):
        """Update resource measurements."""
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else cpu_percent / 100
        
        with self._lock:
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory_percent)
            self.load_history.append(load_avg)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        
        with self._lock:
            if not self.cpu_history:
                return {'cpu': 0.0, 'memory': 0.0, 'load': 0.0}
            
            return {
                'cpu': self.cpu_history[-1],
                'memory': self.memory_history[-1],
                'load': self.load_history[-1]
            }
    
    def get_average_metrics(self, window: int = 10) -> Dict[str, float]:
        """Get average resource metrics over window."""
        
        with self._lock:
            if not self.cpu_history:
                return {'cpu': 0.0, 'memory': 0.0, 'load': 0.0}
            
            recent_cpu = list(self.cpu_history)[-window:]
            recent_memory = list(self.memory_history)[-window:]
            recent_load = list(self.load_history)[-window:]
            
            return {
                'cpu': sum(recent_cpu) / len(recent_cpu),
                'memory': sum(recent_memory) / len(recent_memory),
                'load': sum(recent_load) / len(recent_load)
            }
    
    def should_scale_up(self, thresholds: Dict[str, float]) -> bool:
        """Check if scaling up is needed."""
        
        avg_metrics = self.get_average_metrics()
        
        return (avg_metrics['cpu'] > thresholds.get('cpu_scale_up', 80) or
                avg_metrics['memory'] > thresholds.get('memory_scale_up', 85) or
                avg_metrics['load'] > thresholds.get('load_scale_up', 2.0))
    
    def should_scale_down(self, thresholds: Dict[str, float]) -> bool:
        """Check if scaling down is possible."""
        
        avg_metrics = self.get_average_metrics()
        
        return (avg_metrics['cpu'] < thresholds.get('cpu_scale_down', 30) and
                avg_metrics['memory'] < thresholds.get('memory_scale_down', 50) and
                avg_metrics['load'] < thresholds.get('load_scale_down', 0.5))


class AdaptiveThreadPool:
    """Thread pool with adaptive scaling based on load."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 20,
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3):
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_workers = min_workers
        self.executor = ThreadPoolExecutor(max_workers=self.current_workers, 
                                         thread_name_prefix="adaptive_worker")
        
        self.task_queue = queue.Queue()
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        self.resource_monitor = ResourceMonitor()
        self._lock = threading.Lock()
        self._shutdown = False
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_and_scale, daemon=True)
        self.monitor_thread.start()
        
        self.logger = get_logger("adaptive_thread_pool")
    
    def _monitor_and_scale(self):
        """Monitor load and scale workers accordingly."""
        
        while not self._shutdown:
            try:
                time.sleep(5)  # Check every 5 seconds
                
                self.resource_monitor.update()
                
                with self._lock:
                    if self.active_tasks == 0:
                        continue
                    
                    load_ratio = self.active_tasks / self.current_workers
                    
                    # Scale up if load is high
                    if (load_ratio > self.scale_up_threshold and 
                        self.current_workers < self.max_workers):
                        
                        new_workers = min(self.max_workers, 
                                        self.current_workers + max(1, int(self.current_workers * 0.5)))
                        
                        self._resize_pool(new_workers)
                        self.logger.info(f"Scaled up to {new_workers} workers (load: {load_ratio:.2f})")
                    
                    # Scale down if load is low
                    elif (load_ratio < self.scale_down_threshold and 
                          self.current_workers > self.min_workers):
                        
                        new_workers = max(self.min_workers,
                                        self.current_workers - max(1, int(self.current_workers * 0.25)))
                        
                        self._resize_pool(new_workers)
                        self.logger.info(f"Scaled down to {new_workers} workers (load: {load_ratio:.2f})")
                        
            except Exception as e:
                self.logger.error(f"Error in scaling monitor: {e}")
    
    def _resize_pool(self, new_size: int):
        """Resize the thread pool."""
        
        # Create new executor with new size
        old_executor = self.executor
        self.executor = ThreadPoolExecutor(max_workers=new_size, 
                                         thread_name_prefix="adaptive_worker")
        self.current_workers = new_size
        
        # Gracefully shutdown old executor
        old_executor.shutdown(wait=False)
    
    def submit(self, func: Callable, *args, **kwargs) -> Future:
        """Submit task to thread pool."""
        
        with self._lock:
            self.active_tasks += 1
        
        def task_wrapper():
            try:
                result = func(*args, **kwargs)
                with self._lock:
                    self.completed_tasks += 1
                return result
            except Exception as e:
                with self._lock:
                    self.failed_tasks += 1
                raise e
            finally:
                with self._lock:
                    self.active_tasks -= 1
        
        return self.executor.submit(task_wrapper)
    
    def map(self, func: Callable, iterable: Iterator, timeout: Optional[float] = None) -> Iterator:
        """Map function over iterable using thread pool."""
        
        futures = [self.submit(func, item) for item in iterable]
        
        for future in as_completed(futures, timeout=timeout):
            yield future.result()
    
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool."""
        
        self._shutdown = True
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        
        with self._lock:
            return {
                'current_workers': self.current_workers,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'active_tasks': self.active_tasks,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'load_ratio': self.active_tasks / self.current_workers if self.current_workers > 0 else 0,
                'resource_metrics': self.resource_monitor.get_current_metrics()
            }


class PriorityTaskScheduler:
    """Task scheduler with priority queues and load balancing."""
    
    def __init__(self, max_concurrent_tasks: int = 50):
        
        # Priority queues
        self.task_queues = {
            TaskPriority.CRITICAL: queue.PriorityQueue(),
            TaskPriority.HIGH: queue.PriorityQueue(),
            TaskPriority.NORMAL: queue.PriorityQueue(),
            TaskPriority.LOW: queue.PriorityQueue()
        }
        
        # Task tracking
        self.tasks = {}  # task_id -> Task
        self.running_tasks = {}  # task_id -> Future
        self.completed_tasks = deque(maxlen=1000)
        
        # Executors
        self.thread_pool = AdaptiveThreadPool()
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, multiprocessing.cpu_count()))
        
        # Configuration
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_count = 0
        
        # Synchronization
        self._lock = threading.Lock()
        self._shutdown = False
        
        # Task processing thread
        self.scheduler_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.scheduler_thread.start()
        
        self.logger = get_logger("priority_task_scheduler")
    
    def submit_task(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: Optional[float] = None, processing_mode: ProcessingMode = ProcessingMode.AUTO,
                   max_retries: int = 3, **kwargs) -> str:
        """Submit task to scheduler."""
        
        task_id = f"task_{int(time.time() * 1000000)}_{threading.get_ident()}"
        
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Determine processing mode if AUTO
        if processing_mode == ProcessingMode.AUTO:
            processing_mode = self._determine_processing_mode(func, args, kwargs)
        
        task.kwargs['_processing_mode'] = processing_mode
        
        with self._lock:
            self.tasks[task_id] = task
        
        # Add to appropriate priority queue
        queue_item = (time.time(), task_id)  # Use timestamp for queue ordering
        self.task_queues[priority].put(queue_item)
        
        self.logger.debug(f"Submitted task {task_id} with priority {priority.name}")
        
        return task_id
    
    def _determine_processing_mode(self, func: Callable, args: tuple, kwargs: dict) -> ProcessingMode:
        """Automatically determine best processing mode for task."""
        
        # Check if function is CPU-intensive
        func_name = getattr(func, '__name__', str(func))
        
        cpu_intensive_keywords = ['encode', 'decode', 'compress', 'optimize', 'simulate', 'calculate']
        is_cpu_intensive = any(keyword in func_name.lower() for keyword in cpu_intensive_keywords)
        
        # Check argument sizes (large data suggests process-based)
        total_size = 0
        for arg in args:
            if isinstance(arg, (list, tuple)):
                total_size += len(arg)
            elif isinstance(arg, np.ndarray):
                total_size += arg.nbytes
            elif hasattr(arg, '__len__'):
                total_size += len(str(arg))
        
        large_data = total_size > 1024 * 1024  # 1MB threshold
        
        # Decision logic
        if is_cpu_intensive and large_data:
            return ProcessingMode.PROCESS
        elif is_cpu_intensive:
            return ProcessingMode.THREAD
        else:
            return ProcessingMode.THREAD
    
    def _process_tasks(self):
        """Main task processing loop."""
        
        while not self._shutdown:
            try:
                # Get next task from priority queues
                task = self._get_next_task()
                
                if task is None:
                    time.sleep(0.1)  # No tasks available
                    continue
                
                # Check if we can run more tasks
                with self._lock:
                    if self.running_count >= self.max_concurrent_tasks:
                        # Put task back and wait
                        queue_item = (time.time(), task.task_id)
                        self.task_queues[task.priority].put(queue_item)
                        time.sleep(0.5)
                        continue
                
                # Execute task
                self._execute_task(task)
                
            except Exception as e:
                self.logger.error(f"Error in task processing: {e}")
                time.sleep(1)
    
    def _get_next_task(self) -> Optional[Task]:
        """Get next task from priority queues."""
        
        # Check queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            
            queue_obj = self.task_queues[priority]
            
            try:
                timestamp, task_id = queue_obj.get_nowait()
                
                with self._lock:
                    if task_id in self.tasks:
                        return self.tasks[task_id]
                    
            except queue.Empty:
                continue
        
        return None
    
    def _execute_task(self, task: Task):
        """Execute a task using appropriate executor."""
        
        task.started_at = time.time()
        
        with self._lock:
            self.running_count += 1
        
        # Wrap task execution with error handling and timeout
        @robust_execution(max_attempts=task.max_retries + 1)
        def execute_with_error_handling():
            return task.func(*task.args, **task.kwargs)
        
        # Select executor based on processing mode
        processing_mode = task.kwargs.get('_processing_mode', ProcessingMode.THREAD)
        
        if processing_mode == ProcessingMode.PROCESS:
            future = self.process_pool.submit(execute_with_error_handling)
        else:  # THREAD or ASYNC
            future = self.thread_pool.submit(execute_with_error_handling)
        
        # Store future for tracking
        with self._lock:
            self.running_tasks[task.task_id] = future
        
        # Handle completion asynchronously
        def handle_completion(fut: Future):
            try:
                task.result = fut.result(timeout=task.timeout)
                task.completed_at = time.time()
                
                self.logger.debug(f"Task {task.task_id} completed in {task.duration:.3f}s")
                
            except Exception as e:
                task.error = e
                task.completed_at = time.time()
                
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    self.logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                    
                    # Resubmit for retry
                    queue_item = (time.time(), task.task_id)
                    self.task_queues[task.priority].put(queue_item)
                    return
                else:
                    self.logger.error(f"Task {task.task_id} failed permanently: {e}")
            
            finally:
                with self._lock:
                    self.running_count -= 1
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]
                    
                    # Move to completed tasks
                    self.completed_tasks.append(task)
        
        # Add completion callback
        future.add_done_callback(handle_completion)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                status = "running" if task_id in self.running_tasks else "pending"
                if task.is_completed:
                    status = "completed" if task.error is None else "failed"
                
                return {
                    'task_id': task_id,
                    'status': status,
                    'priority': task.priority.name,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'duration': task.duration,
                    'retry_count': task.retry_count,
                    'error': str(task.error) if task.error else None
                }
        
        return None
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result (blocking)."""
        
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    
                    if task.is_completed:
                        if task.error:
                            raise task.error
                        return task.result
                    
                    # Check timeout
                    if timeout and (time.time() - start_time) > timeout:
                        raise TimeoutError(f"Task {task_id} timed out")
            
            time.sleep(0.1)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        
        with self._lock:
            # Cancel if running
            if task_id in self.running_tasks:
                future = self.running_tasks[task_id]
                cancelled = future.cancel()
                
                if cancelled:
                    del self.running_tasks[task_id]
                    if task_id in self.tasks:
                        del self.tasks[task_id]
                
                return cancelled
            
            # Remove from queues if pending
            if task_id in self.tasks:
                del self.tasks[task_id]
                return True
        
        return False
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        
        with self._lock:
            queue_sizes = {
                priority.name: queue_obj.qsize()
                for priority, queue_obj in self.task_queues.items()
            }
            
            completed_count = len(self.completed_tasks)
            failed_count = sum(1 for task in self.completed_tasks if task.error)
            
            return {
                'running_tasks': self.running_count,
                'queue_sizes': queue_sizes,
                'total_tasks': len(self.tasks),
                'completed_tasks': completed_count,
                'failed_tasks': failed_count,
                'success_rate': (completed_count - failed_count) / completed_count if completed_count > 0 else 0,
                'thread_pool_stats': self.thread_pool.get_stats()
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown scheduler."""
        
        self._shutdown = True
        self.thread_pool.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)


class BatchProcessor:
    """Batch processing with automatic batching and load balancing."""
    
    def __init__(self, scheduler: PriorityTaskScheduler):
        self.scheduler = scheduler
        self.batch_configs = {}  # function -> batch config
        self.pending_items = defaultdict(list)  # function -> list of items
        self.batch_timers = {}  # function -> timer
        self._lock = threading.Lock()
        
        self.logger = get_logger("batch_processor")
    
    def register_batch_function(self, func: Callable, batch_size: int = 10,
                              max_wait_time: float = 1.0, 
                              priority: TaskPriority = TaskPriority.NORMAL):
        """Register function for batch processing."""
        
        self.batch_configs[func] = {
            'batch_size': batch_size,
            'max_wait_time': max_wait_time,
            'priority': priority
        }
        
        self.logger.info(f"Registered batch function {func.__name__} "
                        f"(batch_size={batch_size}, max_wait={max_wait_time}s)")
    
    def submit_for_batch(self, func: Callable, item: Any) -> str:
        """Submit item for batch processing."""
        
        if func not in self.batch_configs:
            raise ValueError(f"Function {func.__name__} not registered for batch processing")
        
        config = self.batch_configs[func]
        item_id = f"batch_item_{int(time.time() * 1000000)}"
        
        with self._lock:
            self.pending_items[func].append((item_id, item))
            
            # Check if batch is ready
            if len(self.pending_items[func]) >= config['batch_size']:
                self._process_batch(func)
            else:
                # Set timer if not already set
                if func not in self.batch_timers:
                    timer = threading.Timer(config['max_wait_time'], 
                                          lambda: self._process_batch(func))
                    timer.start()
                    self.batch_timers[func] = timer
        
        return item_id
    
    def _process_batch(self, func: Callable):
        """Process pending batch for function."""
        
        with self._lock:
            if func not in self.pending_items or not self.pending_items[func]:
                return
            
            # Get batch items
            batch_items = self.pending_items[func].copy()
            self.pending_items[func].clear()
            
            # Cancel timer
            if func in self.batch_timers:
                self.batch_timers[func].cancel()
                del self.batch_timers[func]
        
        if not batch_items:
            return
        
        config = self.batch_configs[func]
        
        # Create batch processing function
        def process_batch():
            items = [item for item_id, item in batch_items]
            return func(items)
        
        # Submit batch to scheduler
        task_id = self.scheduler.submit_task(
            process_batch,
            priority=config['priority']
        )
        
        self.logger.debug(f"Submitted batch of {len(batch_items)} items for {func.__name__}")
        
        return task_id


class ConcurrentProcessor:
    """High-level concurrent processor with automatic optimization."""
    
    def __init__(self, max_workers: int = None):
        
        if max_workers is None:
            max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
        
        # Initialize components
        self.scheduler = PriorityTaskScheduler(max_concurrent_tasks=max_workers)
        self.batch_processor = BatchProcessor(self.scheduler)
        
        # Processing statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        self._stats_lock = threading.Lock()
        
        self.logger = get_logger("concurrent_processor")
    
    def process_single(self, func: Callable, *args, 
                      priority: TaskPriority = TaskPriority.NORMAL,
                      timeout: Optional[float] = None,
                      processing_mode: ProcessingMode = ProcessingMode.AUTO,
                      **kwargs) -> str:
        """Process single item asynchronously."""
        
        with self._stats_lock:
            self.stats['tasks_submitted'] += 1
        
        return self.scheduler.submit_task(
            func, *args, 
            priority=priority,
            timeout=timeout,
            processing_mode=processing_mode,
            **kwargs
        )
    
    def process_batch(self, func: Callable, items: List[Any],
                     priority: TaskPriority = TaskPriority.NORMAL,
                     batch_size: Optional[int] = None,
                     processing_mode: ProcessingMode = ProcessingMode.AUTO) -> List[str]:
        """Process batch of items."""
        
        if batch_size is None:
            batch_size = min(len(items), 50)  # Default batch size
        
        task_ids = []
        
        # Split into batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Create batch processing function
            def process_batch_items(batch_items=batch):
                return [func(item) for item in batch_items]
            
            task_id = self.scheduler.submit_task(
                process_batch_items,
                priority=priority,
                processing_mode=processing_mode
            )
            
            task_ids.append(task_id)
        
        return task_ids
    
    def map_concurrent(self, func: Callable, items: List[Any],
                      max_workers: Optional[int] = None,
                      timeout: Optional[float] = None) -> List[Any]:
        """Concurrent map operation."""
        
        if max_workers is None:
            max_workers = min(len(items), 20)
        
        # Submit all tasks
        task_ids = []
        for item in items:
            task_id = self.process_single(func, item, timeout=timeout)
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            try:
                result = self.scheduler.get_result(task_id, timeout=timeout)
                results.append(result)
                
                with self._stats_lock:
                    self.stats['tasks_completed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                results.append(None)
                
                with self._stats_lock:
                    self.stats['tasks_failed'] += 1
        
        return results
    
    def submit_pipeline(self, pipeline_functions: List[Callable], initial_data: Any,
                       priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit processing pipeline."""
        
        def execute_pipeline():
            data = initial_data
            
            for func in pipeline_functions:
                data = func(data)
            
            return data
        
        return self.scheduler.submit_task(execute_pipeline, priority=priority)
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result."""
        return self.scheduler.get_result(task_id, timeout)
    
    def wait_for_completion(self, task_ids: List[str], 
                          timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for multiple tasks to complete."""
        
        results = {}
        start_time = time.time()
        
        for task_id in task_ids:
            remaining_timeout = None
            if timeout:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
            
            try:
                results[task_id] = self.get_result(task_id, remaining_timeout)
            except Exception as e:
                results[task_id] = {'error': str(e)}
        
        return results
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        
        with self._stats_lock:
            base_stats = self.stats.copy()
        
        scheduler_stats = self.scheduler.get_scheduler_stats()
        
        return {
            'processing_stats': base_stats,
            'scheduler_stats': scheduler_stats,
            'system_resources': psutil.virtual_memory()._asdict()
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown processor."""
        self.scheduler.shutdown(wait=wait)


# Global concurrent processor instance
_global_processor = None
_processor_lock = threading.Lock()


def get_concurrent_processor() -> ConcurrentProcessor:
    """Get global concurrent processor instance."""
    
    global _global_processor
    
    with _processor_lock:
        if _global_processor is None:
            _global_processor = ConcurrentProcessor()
        
        return _global_processor


# Convenience decorators
def concurrent_task(priority: TaskPriority = TaskPriority.NORMAL,
                   processing_mode: ProcessingMode = ProcessingMode.AUTO):
    """Decorator to make function execute concurrently."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            processor = get_concurrent_processor()
            task_id = processor.process_single(func, *args, 
                                             priority=priority,
                                             processing_mode=processing_mode,
                                             **kwargs)
            return processor.get_result(task_id)
        
        return wrapper
    
    return decorator


def batch_process(batch_size: int = 10, max_wait_time: float = 1.0,
                 priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator for batch processing."""
    
    def decorator(func):
        processor = get_concurrent_processor()
        processor.batch_processor.register_batch_function(
            func, batch_size, max_wait_time, priority
        )
        
        def wrapper(items):
            if isinstance(items, list):
                # Batch processing
                return processor.batch_processor.submit_for_batch(func, items)
            else:
                # Single item
                return processor.batch_processor.submit_for_batch(func, [items])
        
        return wrapper
    
    return decorator