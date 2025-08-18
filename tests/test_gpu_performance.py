"""
GPU and performance testing utilities for DNA-Origami-AutoEncoder.

This module provides comprehensive testing for GPU acceleration,
performance benchmarks, and resource utilization monitoring.
"""

import time
import psutil
import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

from .test_config import TestMarkers, TestUtilities, TEST_CONFIG


@dataclass
class PerformanceMetrics:
    """Container for performance test results."""
    
    execution_time_ms: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float] = None
    cpu_utilization_percent: float = 0.0
    gpu_utilization_percent: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    
    def passes_thresholds(self, 
                         max_time_ms: float = TEST_CONFIG.performance_threshold_ms,
                         max_memory_mb: float = TEST_CONFIG.memory_threshold_mb) -> bool:
        """Check if performance metrics pass defined thresholds."""
        return (self.execution_time_ms <= max_time_ms and 
                self.memory_usage_mb <= max_memory_mb)


class PerformanceMonitor:
    """Monitor system and GPU performance during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
    
    @contextmanager
    def monitor_performance(self, num_samples: Optional[int] = None):
        """Context manager to monitor performance metrics."""
        # Initial measurements
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_gpu_memory = None
        
        if self.gpu_available:
            torch.cuda.empty_cache()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        try:
            yield
        finally:
            # Final measurements
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            end_gpu_memory = None
            
            if self.gpu_available:
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Calculate metrics
            execution_time_ms = (end_time - start_time) * 1000
            memory_usage_mb = end_memory - start_memory
            gpu_memory_mb = None
            
            if self.gpu_available and end_gpu_memory is not None:
                gpu_memory_mb = end_gpu_memory - (start_gpu_memory or 0)
            
            throughput = None
            if num_samples and execution_time_ms > 0:
                throughput = num_samples / (execution_time_ms / 1000)
            
            # Store metrics as instance variable for retrieval
            self.last_metrics = PerformanceMetrics(
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                gpu_memory_mb=gpu_memory_mb,
                cpu_utilization_percent=self.process.cpu_percent(),
                throughput_samples_per_sec=throughput
            )
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        if not self.gpu_available:
            return {'available': False}
        
        gpu_info = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'allocated_memory_mb': torch.cuda.memory_allocated() / 1024**2,
            'cached_memory_mb': torch.cuda.memory_reserved() / 1024**2,
            'cuda_version': torch.version.cuda,
            'pytorch_cuda_version': torch.version.cuda
        }
        
        return gpu_info


class GPUTestUtilities:
    """Utilities for GPU-specific testing."""
    
    @staticmethod
    def assert_cuda_available():
        """Assert CUDA is available, skip test otherwise."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    
    @staticmethod
    def assert_sufficient_gpu_memory(required_gb: float = 1.0):
        """Assert sufficient GPU memory is available."""
        GPUTestUtilities.assert_cuda_available()
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory - torch.cuda.memory_allocated()
        available_gb = available_memory / 1024**3
        
        if available_gb < required_gb:
            pytest.skip(f"Insufficient GPU memory: {available_gb:.1f}GB < {required_gb}GB")
    
    @staticmethod
    def cleanup_gpu_memory():
        """Clean up GPU memory between tests."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def create_test_tensor_gpu(shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create a test tensor on GPU."""
        GPUTestUtilities.assert_cuda_available()
        TestUtilities.set_random_seeds()
        return torch.randn(shape, dtype=dtype, device='cuda')
    
    @staticmethod
    def benchmark_gpu_operation(operation: Callable, 
                               iterations: int = TEST_CONFIG.benchmark_iterations,
                               warmup_iterations: int = 3) -> PerformanceMetrics:
        """Benchmark a GPU operation."""
        GPUTestUtilities.assert_cuda_available()
        
        monitor = PerformanceMonitor()
        
        # Warmup iterations
        for _ in range(warmup_iterations):
            operation()
            torch.cuda.synchronize()
        
        # Benchmark iterations
        with monitor.monitor_performance(num_samples=iterations):
            for _ in range(iterations):
                operation()
                torch.cuda.synchronize()
        
        return monitor.last_metrics


# Performance test decorators
def performance_test(max_time_ms: float = TEST_CONFIG.performance_threshold_ms,
                    max_memory_mb: float = TEST_CONFIG.memory_threshold_mb):
    """Decorator for performance tests with automatic threshold checking."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            with monitor.monitor_performance():
                result = test_func(*args, **kwargs)
            
            metrics = monitor.last_metrics
            assert metrics.passes_thresholds(max_time_ms, max_memory_mb), \
                f"Performance test failed: {metrics.execution_time_ms:.1f}ms > {max_time_ms}ms " \
                f"or {metrics.memory_usage_mb:.1f}MB > {max_memory_mb}MB"
            
            return result
        return wrapper
    return decorator


def gpu_test(min_memory_gb: float = 1.0):
    """Decorator for GPU tests with memory requirement checking."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            GPUTestUtilities.assert_sufficient_gpu_memory(min_memory_gb)
            try:
                result = test_func(*args, **kwargs)
            finally:
                GPUTestUtilities.cleanup_gpu_memory()
            return result
        return wrapper
    return decorator


# Test fixtures for performance testing
@pytest.fixture
def performance_monitor():
    """Fixture providing a performance monitor."""
    return PerformanceMonitor()


@pytest.fixture
def gpu_test_data():
    """Fixture providing GPU test data."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    data = {
        'small_tensor': GPUTestUtilities.create_test_tensor_gpu((100, 100)),
        'medium_tensor': GPUTestUtilities.create_test_tensor_gpu((1000, 1000)),
        'batch_tensors': [GPUTestUtilities.create_test_tensor_gpu((32, 64)) for _ in range(16)]
    }
    
    yield data
    
    # Cleanup
    GPUTestUtilities.cleanup_gpu_memory()


@pytest.fixture(scope="session")
def gpu_info():
    """Session-scoped fixture providing GPU information."""
    monitor = PerformanceMonitor()
    return monitor.get_gpu_info()


# Example performance tests
class TestGPUPerformance:
    """Example GPU performance tests."""
    
    @TestMarkers.GPU
    @TestMarkers.PERFORMANCE
    def test_gpu_tensor_operations(self, gpu_test_data, performance_monitor):
        """Test basic GPU tensor operations performance."""
        tensor_a = gpu_test_data['medium_tensor']
        tensor_b = gpu_test_data['medium_tensor']
        
        with performance_monitor.monitor_performance(num_samples=100):
            for _ in range(100):
                result = torch.matmul(tensor_a, tensor_b)
                torch.cuda.synchronize()
        
        metrics = performance_monitor.last_metrics
        assert metrics.execution_time_ms < 5000, "GPU matrix multiplication too slow"
        assert metrics.throughput_samples_per_sec > 10, "GPU throughput too low"
    
    @TestMarkers.GPU
    @TestMarkers.PERFORMANCE
    @gpu_test(min_memory_gb=2.0)
    def test_gpu_memory_efficiency(self, gpu_test_data):
        """Test GPU memory usage efficiency."""
        initial_memory = torch.cuda.memory_allocated()
        
        # Perform memory-intensive operations
        large_tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000, device='cuda')
            large_tensors.append(tensor)
        
        peak_memory = torch.cuda.memory_allocated()
        memory_increase = (peak_memory - initial_memory) / 1024**2  # MB
        
        # Clean up
        del large_tensors
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        
        # Assert memory was properly cleaned up
        assert final_memory <= initial_memory + 10 * 1024**2, "GPU memory leak detected"
        assert memory_increase < 500, f"Excessive GPU memory usage: {memory_increase:.1f}MB"
    
    @TestMarkers.PERFORMANCE
    @performance_test(max_time_ms=1000, max_memory_mb=100)
    def test_cpu_performance_baseline(self):
        """Test CPU performance baseline."""
        # CPU-intensive operation
        data = np.random.randn(1000, 1000)
        result = np.linalg.eigvals(data)
        assert len(result) == 1000


# Utility functions for integration with main test suite
def run_performance_benchmarks():
    """Run comprehensive performance benchmarks."""
    monitor = PerformanceMonitor()
    gpu_info = monitor.get_gpu_info()
    
    results = {
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3,
            'gpu_info': gpu_info
        },
        'benchmarks': {}
    }
    
    return results


# Export main utilities
__all__ = [
    'PerformanceMetrics',
    'PerformanceMonitor', 
    'GPUTestUtilities',
    'performance_test',
    'gpu_test',
    'TestGPUPerformance'
]