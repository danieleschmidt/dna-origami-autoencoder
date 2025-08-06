"""
Distributed Processing and Task Management for DNA Origami AutoEncoder

Provides distributed task execution, load balancing, auto-scaling,
and cluster management capabilities.
"""

import asyncio
import logging
import json
import time
import uuid
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from enum import Enum
import multiprocessing
import threading
import queue
from contextlib import contextmanager

import numpy as np


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerType(Enum):
    """Types of distributed workers."""
    PROCESS = "process"      # Multi-process for CPU-bound tasks
    THREAD = "thread"        # Multi-thread for I/O-bound tasks
    ASYNC = "async"          # Async coroutines
    REMOTE = "remote"        # Remote workers (future)


@dataclass
class TaskDefinition:
    """Definition of a distributed task."""
    task_id: str
    function: Callable
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher number = higher priority
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    worker_type: WorkerType = WorkerType.PROCESS
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a distributed task."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    worker_id: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None


class ResourceManager:
    """Manages computational resources and load balancing."""
    
    def __init__(self):
        self.cpu_cores = multiprocessing.cpu_count()
        self.available_memory = self._get_available_memory()
        self.gpu_devices = self._detect_gpu_devices()
        
        # Resource allocation tracking
        self.cpu_usage = 0
        self.memory_usage = 0
        self.gpu_usage = {device: 0 for device in self.gpu_devices}
        
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def _get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            # Fallback estimate
            return 8 * 1024 * 1024 * 1024  # 8GB default
    
    def _detect_gpu_devices(self) -> List[str]:
        """Detect available GPU devices."""
        gpu_devices = []
        
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_devices.append(f"cuda:{i}")
        except ImportError:
            pass
            
        return gpu_devices
    
    def can_allocate_resources(self, requirements: Dict[str, Any]) -> bool:
        """Check if required resources can be allocated."""
        with self.lock:
            # Check CPU cores
            if requirements.get('cpu_cores', 1) > (self.cpu_cores - self.cpu_usage):
                return False
                
            # Check memory
            required_memory = requirements.get('memory', 0)
            if required_memory > (self.available_memory - self.memory_usage):
                return False
                
            # Check GPU
            required_gpu = requirements.get('gpu_device')
            if required_gpu and required_gpu in self.gpu_usage:
                if self.gpu_usage[required_gpu] >= 1.0:  # Assume max 1 task per GPU
                    return False
                    
            return True
    
    def allocate_resources(self, task_id: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources for a task."""
        with self.lock:
            if not self.can_allocate_resources(requirements):
                raise ResourceError("Insufficient resources available")
                
            allocation = {}
            
            # Allocate CPU
            cpu_cores = requirements.get('cpu_cores', 1)
            self.cpu_usage += cpu_cores
            allocation['cpu_cores'] = cpu_cores
            
            # Allocate memory
            memory = requirements.get('memory', 0)
            self.memory_usage += memory
            allocation['memory'] = memory
            
            # Allocate GPU
            gpu_device = requirements.get('gpu_device')
            if gpu_device and gpu_device in self.gpu_usage:
                self.gpu_usage[gpu_device] += 1.0
                allocation['gpu_device'] = gpu_device
                
            self.logger.debug(f"Allocated resources for task {task_id}: {allocation}")
            return allocation
    
    def release_resources(self, task_id: str, allocation: Dict[str, Any]) -> None:
        """Release allocated resources."""
        with self.lock:
            # Release CPU
            cpu_cores = allocation.get('cpu_cores', 0)
            self.cpu_usage = max(0, self.cpu_usage - cpu_cores)
            
            # Release memory
            memory = allocation.get('memory', 0)
            self.memory_usage = max(0, self.memory_usage - memory)
            
            # Release GPU
            gpu_device = allocation.get('gpu_device')
            if gpu_device and gpu_device in self.gpu_usage:
                self.gpu_usage[gpu_device] = max(0, self.gpu_usage[gpu_device] - 1.0)
                
            self.logger.debug(f"Released resources for task {task_id}: {allocation}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource usage status."""
        with self.lock:
            return {
                "cpu": {
                    "total_cores": self.cpu_cores,
                    "used_cores": self.cpu_usage,
                    "utilization": self.cpu_usage / self.cpu_cores
                },
                "memory": {
                    "total_bytes": self.available_memory,
                    "used_bytes": self.memory_usage,
                    "utilization": self.memory_usage / self.available_memory
                },
                "gpu": {
                    "devices": list(self.gpu_devices),
                    "usage": self.gpu_usage.copy()
                }
            }


class TaskQueue:
    """Priority-based task queue with load balancing."""
    
    def __init__(self, maxsize: int = 0):
        self.queue = queue.PriorityQueue(maxsize=maxsize)
        self.pending_tasks = {}
        self.lock = threading.Lock()
        
    def put_task(self, task: TaskDefinition) -> None:
        """Add task to queue with priority."""
        # Use negative priority for correct ordering (higher number = higher priority)
        priority_item = (-task.priority, time.time(), task.task_id, task)
        
        self.queue.put(priority_item)
        
        with self.lock:
            self.pending_tasks[task.task_id] = task
    
    def get_task(self, timeout: Optional[float] = None) -> Optional[TaskDefinition]:
        """Get next highest priority task."""
        try:
            priority_item = self.queue.get(timeout=timeout)
            _, _, task_id, task = priority_item
            
            with self.lock:
                self.pending_tasks.pop(task_id, None)
                
            return task
            
        except queue.Empty:
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self.lock:
            if task_id in self.pending_tasks:
                # Mark as cancelled (actual removal from queue is complex)
                self.pending_tasks[task_id].metadata['cancelled'] = True
                return True
            return False
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()
    
    def get_pending_tasks(self) -> List[str]:
        """Get list of pending task IDs."""
        with self.lock:
            return list(self.pending_tasks.keys())


class DistributedWorkerPool:
    """Pool of distributed workers for task execution."""
    
    def __init__(self, worker_type: WorkerType, max_workers: int = None):
        self.worker_type = worker_type
        self.max_workers = max_workers or self._get_default_worker_count()
        self.active_tasks = {}
        self.worker_stats = {}
        
        # Initialize worker pools
        if worker_type == WorkerType.PROCESS:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        elif worker_type == WorkerType.THREAD:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        elif worker_type == WorkerType.ASYNC:
            self.semaphore = asyncio.Semaphore(self.max_workers)
        else:
            raise ValueError(f"Unsupported worker type: {worker_type}")
            
        self.logger = logging.getLogger(__name__)
        
    def _get_default_worker_count(self) -> int:
        """Get default number of workers based on type."""
        if self.worker_type == WorkerType.PROCESS:
            return multiprocessing.cpu_count()
        elif self.worker_type == WorkerType.THREAD:
            return min(32, multiprocessing.cpu_count() * 2)
        elif self.worker_type == WorkerType.ASYNC:
            return 100  # High concurrency for async
        else:
            return 4
    
    def submit_task(self, task: TaskDefinition) -> 'asyncio.Future':
        """Submit task for execution."""
        
        if self.worker_type == WorkerType.ASYNC:
            return asyncio.create_task(self._execute_async_task(task))
        else:
            future = self.executor.submit(self._execute_sync_task, task)
            self.active_tasks[task.task_id] = future
            return future
    
    async def _execute_async_task(self, task: TaskDefinition) -> TaskResult:
        """Execute task using async worker."""
        
        async with self.semaphore:
            return await self._execute_task_with_monitoring(task)
    
    def _execute_sync_task(self, task: TaskDefinition) -> TaskResult:
        """Execute task using sync worker."""
        
        return asyncio.run(self._execute_task_with_monitoring(task))
    
    async def _execute_task_with_monitoring(self, task: TaskDefinition) -> TaskResult:
        """Execute task with comprehensive monitoring."""
        
        task_result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=time.time()
        )
        
        try:
            # Check for cancellation
            if task.metadata.get('cancelled', False):
                task_result.status = TaskStatus.CANCELLED
                return task_result
            
            # Execute with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self._execute_function(task.function, task.args, task.kwargs),
                    timeout=task.timeout
                )
            else:
                result = await self._execute_function(task.function, task.args, task.kwargs)
            
            task_result.result = result
            task_result.status = TaskStatus.COMPLETED
            
        except asyncio.TimeoutError:
            task_result.status = TaskStatus.FAILED
            task_result.error = f"Task timed out after {task.timeout}s"
            
        except Exception as e:
            task_result.status = TaskStatus.FAILED
            task_result.error = str(e)
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            task_result.end_time = time.time()
            task_result.execution_time = task_result.end_time - task_result.start_time
            
            # Clean up
            self.active_tasks.pop(task.task_id, None)
            
        return task_result
    
    async def _execute_function(self, func: Callable, args: Tuple, kwargs: Dict) -> Any:
        """Execute function with proper async handling."""
        
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            return future.cancel()
        return False
    
    def get_active_task_count(self) -> int:
        """Get number of currently active tasks."""
        return len(self.active_tasks)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool."""
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=wait)


class AutoScaler:
    """Automatic scaling of worker resources based on load."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 16, 
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.2):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_workers = min_workers
        self.last_scale_time = time.time()
        self.scale_cooldown = 60.0  # seconds
        
        self.metrics_window = []
        self.window_size = 10
        
        self.logger = logging.getLogger(__name__)
    
    def update_metrics(self, queue_size: int, active_tasks: int) -> None:
        """Update scaling metrics."""
        
        utilization = active_tasks / max(self.current_workers, 1)
        load_pressure = queue_size + active_tasks
        
        metrics = {
            'timestamp': time.time(),
            'utilization': utilization,
            'load_pressure': load_pressure,
            'queue_size': queue_size,
            'active_tasks': active_tasks
        }
        
        self.metrics_window.append(metrics)
        if len(self.metrics_window) > self.window_size:
            self.metrics_window.pop(0)
    
    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed."""
        
        if self.current_workers >= self.max_workers:
            return False
            
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
            
        if len(self.metrics_window) < 3:
            return False
            
        # Check recent utilization trend
        recent_utilization = [m['utilization'] for m in self.metrics_window[-3:]]
        avg_utilization = sum(recent_utilization) / len(recent_utilization)
        
        return avg_utilization > self.scale_up_threshold
    
    def should_scale_down(self) -> bool:
        """Determine if scaling down is needed."""
        
        if self.current_workers <= self.min_workers:
            return False
            
        if time.time() - self.last_scale_time < self.scale_cooldown * 2:  # Longer cooldown for scale down
            return False
            
        if len(self.metrics_window) < 5:
            return False
            
        # Check sustained low utilization
        recent_utilization = [m['utilization'] for m in self.metrics_window[-5:]]
        avg_utilization = sum(recent_utilization) / len(recent_utilization)
        
        return avg_utilization < self.scale_down_threshold
    
    def get_scaling_decision(self) -> Tuple[str, int]:
        """Get scaling decision and target worker count."""
        
        if self.should_scale_up():
            new_count = min(self.max_workers, self.current_workers * 2)
            return "scale_up", new_count
        elif self.should_scale_down():
            new_count = max(self.min_workers, self.current_workers // 2)
            return "scale_down", new_count
        else:
            return "no_change", self.current_workers
    
    def apply_scaling(self, new_worker_count: int) -> None:
        """Apply scaling decision."""
        
        old_count = self.current_workers
        self.current_workers = new_worker_count
        self.last_scale_time = time.time()
        
        self.logger.info(f"Scaled workers: {old_count} -> {new_worker_count}")


class DistributedTaskManager:
    """Main distributed task management system."""
    
    def __init__(self, max_queue_size: int = 1000):
        self.task_queue = TaskQueue(maxsize=max_queue_size)
        self.resource_manager = ResourceManager()
        self.worker_pools = {}
        self.auto_scaler = AutoScaler()
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_results = {}
        
        # Management thread
        self.running = False
        self.management_thread = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default worker pools
        self._initialize_worker_pools()
    
    def _initialize_worker_pools(self) -> None:
        """Initialize default worker pools."""
        
        self.worker_pools[WorkerType.PROCESS] = DistributedWorkerPool(
            WorkerType.PROCESS, 
            max_workers=multiprocessing.cpu_count()
        )
        
        self.worker_pools[WorkerType.THREAD] = DistributedWorkerPool(
            WorkerType.THREAD,
            max_workers=min(32, multiprocessing.cpu_count() * 2)
        )
        
        self.worker_pools[WorkerType.ASYNC] = DistributedWorkerPool(
            WorkerType.ASYNC,
            max_workers=100
        )
    
    def start(self) -> None:
        """Start the task management system."""
        
        if not self.running:
            self.running = True
            self.management_thread = threading.Thread(target=self._management_loop, daemon=True)
            self.management_thread.start()
            self.logger.info("Distributed task manager started")
    
    def stop(self) -> None:
        """Stop the task management system."""
        
        if self.running:
            self.running = False
            if self.management_thread:
                self.management_thread.join(timeout=5.0)
            
            # Shutdown worker pools
            for pool in self.worker_pools.values():
                pool.shutdown()
            
            self.logger.info("Distributed task manager stopped")
    
    def submit_task(self, func: Callable, *args, priority: int = 0, 
                   worker_type: WorkerType = WorkerType.PROCESS,
                   timeout: Optional[float] = None,
                   resource_requirements: Optional[Dict[str, Any]] = None,
                   **kwargs) -> str:
        """Submit a task for distributed execution."""
        
        task_id = str(uuid.uuid4())
        
        task = TaskDefinition(
            task_id=task_id,
            function=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            worker_type=worker_type,
            resource_requirements=resource_requirements or {}
        )
        
        self.task_queue.put_task(task)
        self.active_tasks[task_id] = task
        
        self.logger.debug(f"Submitted task {task_id}")
        return task_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result of a completed task."""
        
        if task_id in self.task_results:
            return self.task_results[task_id]
        
        # Wait for completion if timeout specified
        if timeout:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if task_id in self.task_results:
                    return self.task_results[task_id]
                time.sleep(0.1)
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        
        # Try to cancel from queue
        if self.task_queue.cancel_task(task_id):
            self.active_tasks.pop(task_id, None)
            return True
        
        # Try to cancel from worker pools
        for pool in self.worker_pools.values():
            if pool.cancel_task(task_id):
                return True
        
        return False
    
    def _management_loop(self) -> None:
        """Main management loop for task scheduling and resource management."""
        
        while self.running:
            try:
                self._process_pending_tasks()
                self._update_scaling_metrics()
                self._apply_auto_scaling()
                time.sleep(1.0)  # Management loop interval
                
            except Exception as e:
                self.logger.error(f"Error in management loop: {e}")
                time.sleep(5.0)
    
    def _process_pending_tasks(self) -> None:
        """Process pending tasks from the queue."""
        
        # Get next task
        task = self.task_queue.get_task(timeout=0.1)
        if not task:
            return
        
        # Check if task was cancelled
        if task.metadata.get('cancelled', False):
            self._complete_task(task.task_id, TaskResult(
                task_id=task.task_id,
                status=TaskStatus.CANCELLED
            ))
            return
        
        # Check resource availability
        if not self.resource_manager.can_allocate_resources(task.resource_requirements):
            # Put task back in queue
            self.task_queue.put_task(task)
            return
        
        # Allocate resources
        try:
            allocation = self.resource_manager.allocate_resources(
                task.task_id, 
                task.resource_requirements
            )
        except Exception as e:
            self.logger.error(f"Resource allocation failed for task {task.task_id}: {e}")
            self._complete_task(task.task_id, TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Resource allocation failed: {e}"
            ))
            return
        
        # Submit to appropriate worker pool
        worker_pool = self.worker_pools.get(task.worker_type)
        if not worker_pool:
            self.logger.error(f"No worker pool available for type: {task.worker_type}")
            self.resource_manager.release_resources(task.task_id, allocation)
            return
        
        try:
            future = worker_pool.submit_task(task)
            
            # Setup completion callback
            def task_completed(fut):
                try:
                    result = fut.result()
                    self._complete_task(task.task_id, result)
                    self.resource_manager.release_resources(task.task_id, allocation)
                except Exception as e:
                    self.logger.error(f"Task completion callback failed: {e}")
            
            if hasattr(future, 'add_done_callback'):
                future.add_done_callback(task_completed)
            else:
                # For asyncio tasks, create monitoring task
                asyncio.create_task(self._monitor_async_task(future, task.task_id, allocation))
                
        except Exception as e:
            self.logger.error(f"Task submission failed for {task.task_id}: {e}")
            self.resource_manager.release_resources(task.task_id, allocation)
            self._complete_task(task.task_id, TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Task submission failed: {e}"
            ))
    
    async def _monitor_async_task(self, future: asyncio.Task, task_id: str, allocation: Dict[str, Any]) -> None:
        """Monitor async task completion."""
        
        try:
            result = await future
            self._complete_task(task_id, result)
        except Exception as e:
            self.logger.error(f"Async task monitoring failed: {e}")
        finally:
            self.resource_manager.release_resources(task_id, allocation)
    
    def _complete_task(self, task_id: str, result: TaskResult) -> None:
        """Handle task completion."""
        
        self.active_tasks.pop(task_id, None)
        self.task_results[task_id] = result
        self.completed_tasks[task_id] = time.time()
        
        self.logger.debug(f"Task {task_id} completed with status: {result.status}")
    
    def _update_scaling_metrics(self) -> None:
        """Update metrics for auto-scaling."""
        
        queue_size = self.task_queue.get_queue_size()
        active_task_count = sum(
            pool.get_active_task_count() 
            for pool in self.worker_pools.values()
        )
        
        self.auto_scaler.update_metrics(queue_size, active_task_count)
    
    def _apply_auto_scaling(self) -> None:
        """Apply auto-scaling decisions."""
        
        scaling_decision, target_workers = self.auto_scaler.get_scaling_decision()
        
        if scaling_decision != "no_change":
            # In a full implementation, this would dynamically adjust worker pools
            # For now, just log the decision
            self.logger.info(f"Auto-scaling decision: {scaling_decision} to {target_workers} workers")
            self.auto_scaler.apply_scaling(target_workers)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the task management system."""
        
        return {
            "queue_size": self.task_queue.get_queue_size(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "worker_pools": {
                worker_type.value: pool.get_active_task_count()
                for worker_type, pool in self.worker_pools.items()
            },
            "resource_status": self.resource_manager.get_resource_status(),
            "auto_scaler": {
                "current_workers": self.auto_scaler.current_workers,
                "min_workers": self.auto_scaler.min_workers,
                "max_workers": self.auto_scaler.max_workers
            }
        }


class ResourceError(Exception):
    """Exception raised when resources cannot be allocated."""
    pass


# Global task manager instance
_task_manager = None

def get_task_manager() -> DistributedTaskManager:
    """Get the global distributed task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = DistributedTaskManager()
        _task_manager.start()
    return _task_manager

# Convenience functions for common operations
def submit_cpu_task(func: Callable, *args, priority: int = 0, **kwargs) -> str:
    """Submit a CPU-bound task."""
    return get_task_manager().submit_task(
        func, *args, 
        priority=priority,
        worker_type=WorkerType.PROCESS,
        **kwargs
    )

def submit_io_task(func: Callable, *args, priority: int = 0, **kwargs) -> str:
    """Submit an I/O-bound task."""
    return get_task_manager().submit_task(
        func, *args,
        priority=priority,
        worker_type=WorkerType.THREAD,
        **kwargs
    )

def submit_async_task(func: Callable, *args, priority: int = 0, **kwargs) -> str:
    """Submit an async task."""
    return get_task_manager().submit_task(
        func, *args,
        priority=priority,
        worker_type=WorkerType.ASYNC,
        **kwargs
    )

def submit_gpu_task(func: Callable, *args, gpu_device: str = "cuda:0", 
                   priority: int = 0, **kwargs) -> str:
    """Submit a GPU-accelerated task."""
    return get_task_manager().submit_task(
        func, *args,
        priority=priority,
        worker_type=WorkerType.PROCESS,
        resource_requirements={"gpu_device": gpu_device},
        **kwargs
    )