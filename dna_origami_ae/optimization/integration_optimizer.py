"""Integrated optimization system combining all performance components."""

import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from ..utils.advanced_caching import get_multi_level_cache
from ..utils.concurrent_processing import get_concurrent_processor
from ..utils.gpu_acceleration import get_gpu_accelerator
from ..utils.distributed import get_task_manager, WorkerType
from ..utils.advanced_logging import get_advanced_logger
from ..utils.performance import performance_monitor
from ..models.image_data import ImageData


class OptimizationLevel(Enum):
    """Optimization levels from basic to maximum performance."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    MAXIMUM = "maximum"


class OptimizationMode(Enum):
    """Optimization modes for different use cases."""
    THROUGHPUT = "throughput"    # Maximize images/second
    LATENCY = "latency"         # Minimize response time
    BALANCED = "balanced"       # Balance throughput and latency
    MEMORY = "memory"           # Minimize memory usage
    ACCURACY = "accuracy"       # Maximize encoding accuracy


@dataclass
class OptimizationConfig:
    """Configuration for integrated optimization."""
    level: OptimizationLevel = OptimizationLevel.INTERMEDIATE
    mode: OptimizationMode = OptimizationMode.BALANCED
    
    # Component settings
    enable_caching: bool = True
    enable_gpu_acceleration: bool = True
    enable_distributed_processing: bool = True
    enable_concurrent_processing: bool = True
    
    # Performance targets
    target_throughput: Optional[float] = None  # images/second
    target_latency: Optional[float] = None     # seconds per image
    max_memory_usage: Optional[int] = None     # bytes
    
    # Adaptive settings
    enable_auto_tuning: bool = True
    tuning_interval: float = 300.0             # seconds
    performance_window: int = 100              # samples for performance tracking


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    throughput: float = 0.0              # images/second
    avg_latency: float = 0.0            # seconds per image
    memory_usage: int = 0               # bytes
    gpu_utilization: float = 0.0        # 0-1
    cache_hit_rate: float = 0.0         # 0-1
    error_rate: float = 0.0             # 0-1
    cpu_utilization: float = 0.0        # 0-1
    queue_length: int = 0               # pending tasks
    
    # Historical data
    samples: List[Dict[str, float]] = field(default_factory=list)
    
    def update_sample(self, **metrics):
        """Update with new performance sample."""
        sample = {
            'timestamp': time.time(),
            **metrics
        }
        self.samples.append(sample)
        
        # Keep only recent samples
        if len(self.samples) > 1000:
            self.samples = self.samples[-1000:]
        
        # Update current metrics
        for key, value in metrics.items():
            if hasattr(self, key):
                setattr(self, key, value)


class IntegratedOptimizer:
    """Main integrated optimization system."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = get_advanced_logger("integrated_optimizer")
        
        # Initialize components
        self.cache = get_multi_level_cache() if self.config.enable_caching else None
        self.concurrent_processor = get_concurrent_processor() if self.config.enable_concurrent_processing else None
        self.gpu_accelerator = get_gpu_accelerator() if self.config.enable_gpu_acceleration else None
        self.task_manager = get_task_manager() if self.config.enable_distributed_processing else None
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.optimization_history = []
        
        # Auto-tuning
        self.auto_tuner = None
        if self.config.enable_auto_tuning:
            self.auto_tuner = AutoTuner(self)
        
        # Threading
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._monitoring = False
        
        self.logger.info(f"Integrated optimizer initialized with level: {self.config.level.value}")
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self._monitor_thread.start()
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_performance(self):
        """Background performance monitoring."""
        while self._monitoring:
            try:
                self._collect_performance_metrics()
                
                # Auto-tune if enabled
                if self.auto_tuner:
                    self.auto_tuner.update()
                
                time.sleep(10.0)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(30.0)
    
    def _collect_performance_metrics(self):
        """Collect current performance metrics."""
        metrics = {}
        
        try:
            # GPU metrics
            if self.gpu_accelerator:
                gpu_info = self.gpu_accelerator.get_device_info()
                metrics['gpu_utilization'] = gpu_info.get('memory_info', {}).get('utilization', 0.0)
            
            # Cache metrics  
            if self.cache:
                cache_stats = self.cache.get_comprehensive_stats()
                metrics['cache_hit_rate'] = cache_stats.get('overall_hit_rate', 0.0)
                metrics['memory_usage'] = cache_stats.get('total_memory_usage', 0)
            
            # Concurrent processing metrics
            if self.concurrent_processor:
                proc_stats = self.concurrent_processor.get_comprehensive_stats()
                metrics['cpu_utilization'] = proc_stats.get('scheduler_stats', {}).get('thread_pool_stats', {}).get('load_ratio', 0.0)
                metrics['queue_length'] = proc_stats.get('scheduler_stats', {}).get('running_tasks', 0)
            
            # Task manager metrics
            if self.task_manager:
                task_status = self.task_manager.get_status()
                metrics['queue_length'] += task_status.get('queue_size', 0)
            
            # Update metrics
            self.metrics.update_sample(**metrics)
            
        except Exception as e:
            self.logger.error(f"Metrics collection error: {e}")
    
    @performance_monitor
    def optimize_image_encoding(self, images: List[ImageData], 
                               encoding_func: Callable,
                               **kwargs) -> List[str]:
        """Optimize image encoding using all available techniques."""
        start_time = time.time()
        
        try:
            # Choose optimization strategy based on config
            if self.config.level == OptimizationLevel.BASIC:
                return self._basic_optimization(images, encoding_func, **kwargs)
            elif self.config.level == OptimizationLevel.INTERMEDIATE:
                return self._intermediate_optimization(images, encoding_func, **kwargs)
            elif self.config.level == OptimizationLevel.ADVANCED:
                return self._advanced_optimization(images, encoding_func, **kwargs)
            else:  # MAXIMUM
                return self._maximum_optimization(images, encoding_func, **kwargs)
                
        finally:
            # Update performance metrics
            execution_time = time.time() - start_time
            throughput = len(images) / execution_time if execution_time > 0 else 0
            avg_latency = execution_time / len(images) if len(images) > 0 else 0
            
            self.metrics.update_sample(
                throughput=throughput,
                avg_latency=avg_latency,
                batch_size=len(images)
            )
    
    def _basic_optimization(self, images: List[ImageData], encoding_func: Callable, **kwargs) -> List[str]:
        """Basic optimization with simple caching."""
        results = []
        
        for image in images:
            # Simple caching if available
            cache_key = f"encode_{hash(image.data.tobytes())}"
            
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    results.append(cached_result)
                    continue
            
            # Encode image
            result = encoding_func(image, **kwargs)
            results.append(result)
            
            # Cache result
            if self.cache:
                self.cache.set(cache_key, result)
        
        return results
    
    def _intermediate_optimization(self, images: List[ImageData], encoding_func: Callable, **kwargs) -> List[str]:
        """Intermediate optimization with concurrent processing."""
        if not self.concurrent_processor:
            return self._basic_optimization(images, encoding_func, **kwargs)
        
        # Use concurrent processing
        task_ids = []
        
        for image in images:
            cache_key = f"encode_{hash(image.data.tobytes())}"
            
            # Check cache first
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    task_ids.append(('cached', cached_result))
                    continue
            
            # Submit to concurrent processor
            task_id = self.concurrent_processor.process_single(
                encoding_func, image, **kwargs
            )
            task_ids.append(('task', task_id, cache_key))
        
        # Collect results
        results = []
        for item in task_ids:
            if item[0] == 'cached':
                results.append(item[1])
            else:
                _, task_id, cache_key = item
                result = self.concurrent_processor.get_result(task_id)
                results.append(result)
                
                # Cache result
                if self.cache:
                    self.cache.set(cache_key, result)
        
        return results
    
    def _advanced_optimization(self, images: List[ImageData], encoding_func: Callable, **kwargs) -> List[str]:
        """Advanced optimization with GPU acceleration and intelligent batching."""
        if not self.gpu_accelerator or self.config.mode == OptimizationMode.MEMORY:
            return self._intermediate_optimization(images, encoding_func, **kwargs)
        
        # Batch processing with GPU acceleration
        batch_size = self._determine_optimal_batch_size(images)
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = self._process_gpu_batch(batch, encoding_func, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def _maximum_optimization(self, images: List[ImageData], encoding_func: Callable, **kwargs) -> List[str]:
        """Maximum optimization using all available techniques."""
        if not self.task_manager:
            return self._advanced_optimization(images, encoding_func, **kwargs)
        
        # Distributed processing with dynamic load balancing
        return self._distributed_optimization(images, encoding_func, **kwargs)
    
    def _process_gpu_batch(self, batch: List[ImageData], encoding_func: Callable, **kwargs) -> List[str]:
        """Process batch using GPU acceleration."""
        batch_images = [img.data for img in batch]
        
        try:
            # Use GPU accelerated encoding
            results = self.gpu_accelerator.accelerate_image_encoding(
                batch_images, encoding_func
            )
            
            # Cache results
            if self.cache:
                for i, (image, result) in enumerate(zip(batch, results)):
                    cache_key = f"encode_{hash(image.data.tobytes())}"
                    self.cache.set(cache_key, result)
            
            return results
            
        except Exception as e:
            self.logger.warning(f"GPU batch processing failed: {e}")
            # Fallback to concurrent processing
            return self._process_concurrent_batch(batch, encoding_func, **kwargs)
    
    def _process_concurrent_batch(self, batch: List[ImageData], encoding_func: Callable, **kwargs) -> List[str]:
        """Process batch using concurrent processing."""
        if not self.concurrent_processor:
            return [encoding_func(img, **kwargs) for img in batch]
        
        # Submit batch tasks
        task_ids = []
        for image in batch:
            task_id = self.concurrent_processor.process_single(
                encoding_func, image, **kwargs
            )
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = self.concurrent_processor.get_result(task_id)
            results.append(result)
        
        return results
    
    def _distributed_optimization(self, images: List[ImageData], encoding_func: Callable, **kwargs) -> List[str]:
        """Distributed optimization across multiple workers."""
        # Determine optimal distribution strategy
        chunk_size = max(1, len(images) // (self.task_manager.auto_scaler.current_workers * 2))
        
        task_ids = []
        for i in range(0, len(images), chunk_size):
            chunk = images[i:i+chunk_size]
            
            # Submit distributed task
            task_id = self.task_manager.submit_task(
                self._process_image_chunk,
                chunk,
                encoding_func,
                worker_type=WorkerType.PROCESS,
                priority=1,
                **kwargs
            )
            task_ids.append(task_id)
        
        # Collect results
        all_results = []
        for task_id in task_ids:
            chunk_results = self.task_manager.get_task_result(task_id, timeout=300.0)
            if chunk_results and chunk_results.status.value == "completed":
                all_results.extend(chunk_results.result)
            else:
                self.logger.error(f"Distributed task {task_id} failed or timed out")
        
        return all_results
    
    def _process_image_chunk(self, chunk: List[ImageData], encoding_func: Callable, **kwargs) -> List[str]:
        """Process a chunk of images (for distributed processing)."""
        # This runs in a separate process
        results = []
        
        for image in chunk:
            try:
                result = encoding_func(image, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Image encoding failed in chunk: {e}")
                results.append("")  # Empty result for failed encoding
        
        return results
    
    def _determine_optimal_batch_size(self, images: List[ImageData]) -> int:
        """Determine optimal batch size based on configuration and resources."""
        if self.config.mode == OptimizationMode.LATENCY:
            return 1  # Process one at a time for minimum latency
        
        elif self.config.mode == OptimizationMode.MEMORY:
            return min(4, len(images))  # Small batches to minimize memory
        
        elif self.config.mode == OptimizationMode.THROUGHPUT:
            # Large batches for maximum throughput
            if self.gpu_accelerator:
                return min(64, len(images))
            else:
                return min(16, len(images))
        
        else:  # BALANCED
            return min(32, len(images))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self._lock:
            report = {
                'current_metrics': {
                    'throughput': self.metrics.throughput,
                    'avg_latency': self.metrics.avg_latency,
                    'memory_usage': self.metrics.memory_usage,
                    'gpu_utilization': self.metrics.gpu_utilization,
                    'cache_hit_rate': self.metrics.cache_hit_rate,
                    'cpu_utilization': self.metrics.cpu_utilization,
                    'queue_length': self.metrics.queue_length
                },
                'configuration': {
                    'level': self.config.level.value,
                    'mode': self.config.mode.value,
                    'components_enabled': {
                        'caching': self.config.enable_caching,
                        'gpu_acceleration': self.config.enable_gpu_acceleration,
                        'distributed_processing': self.config.enable_distributed_processing,
                        'concurrent_processing': self.config.enable_concurrent_processing
                    }
                },
                'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
                'component_stats': {}
            }
            
            # Add component-specific stats
            if self.cache:
                report['component_stats']['cache'] = self.cache.get_comprehensive_stats()
            
            if self.gpu_accelerator:
                report['component_stats']['gpu'] = self.gpu_accelerator.get_device_info()
            
            if self.concurrent_processor:
                report['component_stats']['concurrent'] = self.concurrent_processor.get_comprehensive_stats()
            
            if self.task_manager:
                report['component_stats']['distributed'] = self.task_manager.get_status()
            
            return report
    
    def tune_configuration(self, target_throughput: Optional[float] = None,
                          target_latency: Optional[float] = None) -> OptimizationConfig:
        """Tune configuration for specific performance targets."""
        if self.auto_tuner:
            return self.auto_tuner.tune_for_targets(target_throughput, target_latency)
        else:
            self.logger.warning("Auto-tuning not enabled")
            return self.config


class AutoTuner:
    """Automatic performance tuning system."""
    
    def __init__(self, optimizer: IntegratedOptimizer):
        self.optimizer = optimizer
        self.logger = get_advanced_logger("auto_tuner")
        self.last_tune_time = time.time()
        self.tuning_history = []
    
    def update(self):
        """Update auto-tuning based on current performance."""
        current_time = time.time()
        
        if (current_time - self.last_tune_time) < self.optimizer.config.tuning_interval:
            return
        
        # Analyze recent performance
        if len(self.optimizer.metrics.samples) < 10:
            return  # Not enough data
        
        recent_samples = self.optimizer.metrics.samples[-10:]
        avg_throughput = np.mean([s.get('throughput', 0) for s in recent_samples])
        avg_latency = np.mean([s.get('avg_latency', 0) for s in recent_samples])
        
        # Check if tuning is needed
        needs_tuning = False
        
        if self.optimizer.config.target_throughput:
            if avg_throughput < self.optimizer.config.target_throughput * 0.9:
                needs_tuning = True
        
        if self.optimizer.config.target_latency:
            if avg_latency > self.optimizer.config.target_latency * 1.1:
                needs_tuning = True
        
        if needs_tuning:
            self._apply_automatic_tuning(avg_throughput, avg_latency)
            self.last_tune_time = current_time
    
    def _apply_automatic_tuning(self, current_throughput: float, current_latency: float):
        """Apply automatic tuning adjustments."""
        self.logger.info(f"Auto-tuning: throughput={current_throughput:.2f}, latency={current_latency:.4f}")
        
        # Simple tuning rules
        if self.optimizer.config.target_throughput and current_throughput < self.optimizer.config.target_throughput:
            # Need more throughput
            if self.optimizer.config.level != OptimizationLevel.MAXIMUM:
                self.logger.info("Increasing optimization level for throughput")
                levels = list(OptimizationLevel)
                current_idx = levels.index(self.optimizer.config.level)
                if current_idx < len(levels) - 1:
                    self.optimizer.config.level = levels[current_idx + 1]
        
        if self.optimizer.config.target_latency and current_latency > self.optimizer.config.target_latency:
            # Need lower latency
            if self.optimizer.config.mode != OptimizationMode.LATENCY:
                self.logger.info("Switching to latency-focused mode")
                self.optimizer.config.mode = OptimizationMode.LATENCY
        
        # Record tuning action
        self.tuning_history.append({
            'timestamp': time.time(),
            'throughput': current_throughput,
            'latency': current_latency,
            'level': self.optimizer.config.level.value,
            'mode': self.optimizer.config.mode.value
        })
        
        # Keep only recent history
        if len(self.tuning_history) > 100:
            self.tuning_history = self.tuning_history[-100:]
    
    def tune_for_targets(self, target_throughput: Optional[float] = None,
                        target_latency: Optional[float] = None) -> OptimizationConfig:
        """Tune configuration for specific targets."""
        config = self.optimizer.config
        
        if target_throughput:
            config.target_throughput = target_throughput
            
            # High throughput requires advanced optimization
            if target_throughput > 100:  # images/second
                config.level = OptimizationLevel.MAXIMUM
                config.mode = OptimizationMode.THROUGHPUT
                config.enable_gpu_acceleration = True
                config.enable_distributed_processing = True
        
        if target_latency:
            config.target_latency = target_latency
            
            # Low latency requires latency-focused mode
            if target_latency < 0.1:  # seconds
                config.mode = OptimizationMode.LATENCY
                config.level = OptimizationLevel.ADVANCED
        
        self.logger.info(f"Tuned configuration: level={config.level.value}, mode={config.mode.value}")
        return config


class SystemOptimizer:
    """System-level optimization coordinator."""
    
    def __init__(self):
        self.optimizers = {}
        self.global_metrics = PerformanceMetrics()
        self.logger = get_advanced_logger("system_optimizer")
    
    def create_optimizer(self, name: str, config: Optional[OptimizationConfig] = None) -> IntegratedOptimizer:
        """Create a new optimizer instance."""
        optimizer = IntegratedOptimizer(config)
        self.optimizers[name] = optimizer
        self.logger.info(f"Created optimizer '{name}'")
        return optimizer
    
    def get_optimizer(self, name: str) -> Optional[IntegratedOptimizer]:
        """Get existing optimizer by name."""
        return self.optimizers.get(name)
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get system-wide performance metrics."""
        system_metrics = {
            'optimizers': {},
            'global_performance': {},
            'resource_usage': {}
        }
        
        # Aggregate metrics from all optimizers
        total_throughput = 0.0
        total_latency = 0.0
        active_optimizers = 0
        
        for name, optimizer in self.optimizers.items():
            optimizer_report = optimizer.get_performance_report()
            system_metrics['optimizers'][name] = optimizer_report['current_metrics']
            
            if optimizer_report['current_metrics']['throughput'] > 0:
                total_throughput += optimizer_report['current_metrics']['throughput']
                total_latency += optimizer_report['current_metrics']['avg_latency']
                active_optimizers += 1
        
        # Global metrics
        if active_optimizers > 0:
            system_metrics['global_performance'] = {
                'total_throughput': total_throughput,
                'average_latency': total_latency / active_optimizers,
                'active_optimizers': active_optimizers
            }
        
        return system_metrics
    
    def optimize_system_configuration(self) -> Dict[str, OptimizationConfig]:
        """Optimize configurations across all optimizers."""
        optimized_configs = {}
        
        for name, optimizer in self.optimizers.items():
            if optimizer.auto_tuner:
                config = optimizer.tune_configuration()
                optimized_configs[name] = config
        
        self.logger.info(f"Optimized {len(optimized_configs)} optimizer configurations")
        return optimized_configs


# Global system optimizer instance
_system_optimizer = None
_system_lock = threading.Lock()


def get_system_optimizer() -> SystemOptimizer:
    """Get global system optimizer instance."""
    global _system_optimizer
    
    with _system_lock:
        if _system_optimizer is None:
            _system_optimizer = SystemOptimizer()
        return _system_optimizer


def create_optimized_pipeline(name: str, 
                            level: OptimizationLevel = OptimizationLevel.INTERMEDIATE,
                            mode: OptimizationMode = OptimizationMode.BALANCED) -> IntegratedOptimizer:
    """Create an optimized processing pipeline."""
    config = OptimizationConfig(level=level, mode=mode)
    system_optimizer = get_system_optimizer()
    return system_optimizer.create_optimizer(name, config)