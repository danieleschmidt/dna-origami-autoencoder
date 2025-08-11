"""Integration tests for the optimization system."""

import pytest
import numpy as np
import time
import threading
from typing import List
from unittest.mock import Mock, patch

from dna_origami_ae.models.image_data import ImageData
from dna_origami_ae.optimization.integration_optimizer import (
    IntegratedOptimizer, OptimizationConfig, OptimizationLevel, OptimizationMode,
    SystemOptimizer, get_system_optimizer, create_optimized_pipeline
)


class TestIntegratedOptimizer:
    """Test cases for integrated optimization system."""
    
    @pytest.fixture
    def sample_images(self) -> List[ImageData]:
        """Create sample test images."""
        images = []
        for i in range(10):
            data = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            images.append(ImageData(
                data=data,
                id=f"test_image_{i}",
                metadata={'test': True}
            ))
        return images
    
    @pytest.fixture
    def mock_encoding_func(self):
        """Mock encoding function for testing."""
        def encode_func(image_data, **kwargs):
            # Simulate encoding by creating a DNA sequence based on image hash
            image_hash = hash(image_data.data.tobytes())
            return f"ATGC{abs(image_hash) % 10000:04d}CGTA"
        return encode_func
    
    def test_basic_optimization_level(self, sample_images, mock_encoding_func):
        """Test basic optimization level."""
        config = OptimizationConfig(
            level=OptimizationLevel.BASIC,
            enable_caching=True,
            enable_gpu_acceleration=False,
            enable_distributed_processing=False
        )
        
        optimizer = IntegratedOptimizer(config)
        
        # First run - should cache results
        start_time = time.time()
        results1 = optimizer.optimize_image_encoding(sample_images, mock_encoding_func)
        first_duration = time.time() - start_time
        
        assert len(results1) == len(sample_images)
        assert all(result.startswith('ATGC') for result in results1)
        
        # Second run - should use cache and be faster
        start_time = time.time()
        results2 = optimizer.optimize_image_encoding(sample_images, mock_encoding_func)
        second_duration = time.time() - start_time
        
        assert results1 == results2
        assert second_duration < first_duration  # Should be faster due to caching
        
        optimizer.stop_monitoring()
    
    def test_intermediate_optimization_level(self, sample_images, mock_encoding_func):
        """Test intermediate optimization level with concurrent processing."""
        config = OptimizationConfig(
            level=OptimizationLevel.INTERMEDIATE,
            enable_concurrent_processing=True,
            enable_caching=True
        )
        
        optimizer = IntegratedOptimizer(config)
        
        results = optimizer.optimize_image_encoding(sample_images, mock_encoding_func)
        
        assert len(results) == len(sample_images)
        assert all(result.startswith('ATGC') for result in results)
        
        # Check that metrics were collected
        assert optimizer.metrics.throughput > 0
        assert optimizer.metrics.avg_latency > 0
        
        optimizer.stop_monitoring()
    
    @pytest.mark.skipif(not pytest.gpu_available, reason="GPU not available")
    def test_advanced_optimization_level(self, sample_images, mock_encoding_func):
        """Test advanced optimization level with GPU acceleration."""
        config = OptimizationConfig(
            level=OptimizationLevel.ADVANCED,
            enable_gpu_acceleration=True,
            enable_concurrent_processing=True
        )
        
        optimizer = IntegratedOptimizer(config)
        
        results = optimizer.optimize_image_encoding(sample_images, mock_encoding_func)
        
        assert len(results) == len(sample_images)
        assert all(result.startswith('ATGC') for result in results)
        
        optimizer.stop_monitoring()
    
    def test_maximum_optimization_level(self, sample_images, mock_encoding_func):
        """Test maximum optimization level with distributed processing."""
        config = OptimizationConfig(
            level=OptimizationLevel.MAXIMUM,
            enable_distributed_processing=True,
            enable_gpu_acceleration=True,
            enable_concurrent_processing=True
        )
        
        optimizer = IntegratedOptimizer(config)
        
        results = optimizer.optimize_image_encoding(sample_images, mock_encoding_func)
        
        assert len(results) == len(sample_images)
        
        optimizer.stop_monitoring()
    
    def test_throughput_optimization_mode(self, sample_images, mock_encoding_func):
        """Test throughput-focused optimization mode."""
        config = OptimizationConfig(
            mode=OptimizationMode.THROUGHPUT,
            level=OptimizationLevel.INTERMEDIATE
        )
        
        optimizer = IntegratedOptimizer(config)
        
        start_time = time.time()
        results = optimizer.optimize_image_encoding(sample_images, mock_encoding_func)
        duration = time.time() - start_time
        
        assert len(results) == len(sample_images)
        
        # Should achieve reasonable throughput
        throughput = len(sample_images) / duration
        assert throughput > 5  # At least 5 images per second
        
        optimizer.stop_monitoring()
    
    def test_latency_optimization_mode(self, sample_images, mock_encoding_func):
        """Test latency-focused optimization mode."""
        config = OptimizationConfig(
            mode=OptimizationMode.LATENCY,
            level=OptimizationLevel.INTERMEDIATE
        )
        
        optimizer = IntegratedOptimizer(config)
        
        # Process single image to test latency
        single_image = [sample_images[0]]
        
        start_time = time.time()
        results = optimizer.optimize_image_encoding(single_image, mock_encoding_func)
        latency = time.time() - start_time
        
        assert len(results) == 1
        assert latency < 1.0  # Should be fast for single image
        
        optimizer.stop_monitoring()
    
    def test_memory_optimization_mode(self, sample_images, mock_encoding_func):
        """Test memory-focused optimization mode."""
        config = OptimizationConfig(
            mode=OptimizationMode.MEMORY,
            level=OptimizationLevel.INTERMEDIATE,
            enable_gpu_acceleration=False  # Avoid GPU memory usage
        )
        
        optimizer = IntegratedOptimizer(config)
        
        results = optimizer.optimize_image_encoding(sample_images, mock_encoding_func)
        
        assert len(results) == len(sample_images)
        
        # Memory mode should use smaller batches
        batch_size = optimizer._determine_optimal_batch_size(sample_images)
        assert batch_size <= 4
        
        optimizer.stop_monitoring()
    
    def test_performance_monitoring(self, sample_images, mock_encoding_func):
        """Test performance monitoring and metrics collection."""
        config = OptimizationConfig(level=OptimizationLevel.INTERMEDIATE)
        optimizer = IntegratedOptimizer(config)
        
        # Process images to generate metrics
        optimizer.optimize_image_encoding(sample_images, mock_encoding_func)
        
        # Wait for monitoring to collect metrics
        time.sleep(0.5)
        
        # Check metrics were collected
        assert optimizer.metrics.throughput >= 0
        assert optimizer.metrics.avg_latency >= 0
        assert len(optimizer.metrics.samples) > 0
        
        # Test performance report
        report = optimizer.get_performance_report()
        assert 'current_metrics' in report
        assert 'configuration' in report
        assert 'component_stats' in report
        
        optimizer.stop_monitoring()
    
    def test_auto_tuning_disabled(self, sample_images, mock_encoding_func):
        """Test operation with auto-tuning disabled."""
        config = OptimizationConfig(
            enable_auto_tuning=False,
            level=OptimizationLevel.BASIC
        )
        
        optimizer = IntegratedOptimizer(config)
        
        assert optimizer.auto_tuner is None
        
        results = optimizer.optimize_image_encoding(sample_images, mock_encoding_func)
        assert len(results) == len(sample_images)
        
        optimizer.stop_monitoring()
    
    def test_auto_tuning_enabled(self, sample_images, mock_encoding_func):
        """Test operation with auto-tuning enabled."""
        config = OptimizationConfig(
            enable_auto_tuning=True,
            target_throughput=100.0,
            tuning_interval=0.1  # Very short for testing
        )
        
        optimizer = IntegratedOptimizer(config)
        
        assert optimizer.auto_tuner is not None
        
        # Process images multiple times to trigger auto-tuning
        for _ in range(5):
            optimizer.optimize_image_encoding(sample_images[:2], mock_encoding_func)
            time.sleep(0.05)
        
        # Allow time for auto-tuning
        time.sleep(0.2)
        
        assert len(optimizer.auto_tuner.tuning_history) >= 0
        
        optimizer.stop_monitoring()
    
    def test_configuration_tuning(self, sample_images, mock_encoding_func):
        """Test configuration tuning for performance targets."""
        config = OptimizationConfig(enable_auto_tuning=True)
        optimizer = IntegratedOptimizer(config)
        
        # Test tuning for high throughput
        tuned_config = optimizer.tune_configuration(target_throughput=200.0)
        assert tuned_config.target_throughput == 200.0
        assert tuned_config.level in [OptimizationLevel.ADVANCED, OptimizationLevel.MAXIMUM]
        
        # Test tuning for low latency
        tuned_config = optimizer.tune_configuration(target_latency=0.05)
        assert tuned_config.target_latency == 0.05
        assert tuned_config.mode == OptimizationMode.LATENCY
        
        optimizer.stop_monitoring()
    
    def test_error_handling_in_encoding(self, sample_images):
        """Test error handling during encoding."""
        def failing_encode_func(image_data, **kwargs):
            if image_data.id == "test_image_5":
                raise ValueError("Simulated encoding failure")
            return "ATGCATGC"
        
        config = OptimizationConfig(level=OptimizationLevel.BASIC)
        optimizer = IntegratedOptimizer(config)
        
        # Should handle errors gracefully
        with pytest.raises(ValueError):
            optimizer.optimize_image_encoding(sample_images, failing_encode_func)
        
        optimizer.stop_monitoring()
    
    def test_concurrent_access(self, sample_images, mock_encoding_func):
        """Test thread safety with concurrent access."""
        config = OptimizationConfig(level=OptimizationLevel.INTERMEDIATE)
        optimizer = IntegratedOptimizer(config)
        
        results = []
        errors = []
        
        def worker():
            try:
                result = optimizer.optimize_image_encoding(sample_images[:3], mock_encoding_func)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert len(results) == 3
        assert all(len(result) == 3 for result in results)
        
        optimizer.stop_monitoring()


class TestSystemOptimizer:
    """Test cases for system-level optimization."""
    
    def test_system_optimizer_singleton(self):
        """Test that system optimizer is a singleton."""
        optimizer1 = get_system_optimizer()
        optimizer2 = get_system_optimizer()
        
        assert optimizer1 is optimizer2
    
    def test_create_and_get_optimizer(self):
        """Test creating and retrieving optimizers."""
        system = get_system_optimizer()
        
        config = OptimizationConfig(level=OptimizationLevel.BASIC)
        optimizer = system.create_optimizer("test_optimizer", config)
        
        assert optimizer is not None
        assert system.get_optimizer("test_optimizer") is optimizer
        assert system.get_optimizer("nonexistent") is None
    
    def test_system_performance_aggregation(self, sample_images, mock_encoding_func):
        """Test system-wide performance aggregation."""
        system = get_system_optimizer()
        
        # Create multiple optimizers
        opt1 = system.create_optimizer("optimizer1", OptimizationConfig(level=OptimizationLevel.BASIC))
        opt2 = system.create_optimizer("optimizer2", OptimizationConfig(level=OptimizationLevel.INTERMEDIATE))
        
        # Generate some activity
        opt1.optimize_image_encoding(sample_images[:3], mock_encoding_func)
        opt2.optimize_image_encoding(sample_images[3:6], mock_encoding_func)
        
        # Get system performance
        performance = system.get_system_performance()
        
        assert 'optimizers' in performance
        assert 'global_performance' in performance
        assert 'optimizer1' in performance['optimizers']
        assert 'optimizer2' in performance['optimizers']
        
        opt1.stop_monitoring()
        opt2.stop_monitoring()
    
    def test_system_configuration_optimization(self):
        """Test system-wide configuration optimization."""
        system = get_system_optimizer()
        
        # Create optimizers with auto-tuning
        config1 = OptimizationConfig(enable_auto_tuning=True)
        config2 = OptimizationConfig(enable_auto_tuning=True)
        
        opt1 = system.create_optimizer("opt1", config1)
        opt2 = system.create_optimizer("opt2", config2)
        
        # Optimize configurations
        optimized_configs = system.optimize_system_configuration()
        
        assert len(optimized_configs) <= 2  # May be empty if no tuning needed
        
        opt1.stop_monitoring()
        opt2.stop_monitoring()
    
    def test_create_optimized_pipeline(self):
        """Test creating optimized pipeline with convenience function."""
        pipeline = create_optimized_pipeline(
            "test_pipeline",
            level=OptimizationLevel.ADVANCED,
            mode=OptimizationMode.THROUGHPUT
        )
        
        assert pipeline is not None
        assert pipeline.config.level == OptimizationLevel.ADVANCED
        assert pipeline.config.mode == OptimizationMode.THROUGHPUT
        
        pipeline.stop_monitoring()


@pytest.fixture
def mock_encoding_func():
    """Mock encoding function for testing."""
    def encode_func(image_data, **kwargs):
        # Simulate encoding delay
        time.sleep(0.001)
        image_hash = hash(image_data.data.tobytes())
        return f"ATGC{abs(image_hash) % 10000:04d}CGTA"
    return encode_func


@pytest.fixture
def sample_images() -> List[ImageData]:
    """Create sample test images."""
    images = []
    for i in range(10):
        data = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        images.append(ImageData(
            data=data,
            id=f"test_image_{i}",
            metadata={'test': True}
        ))
    return images


# Mark GPU tests appropriately
pytest.gpu_available = False
try:
    import torch
    if torch.cuda.is_available():
        pytest.gpu_available = True
except ImportError:
    pass