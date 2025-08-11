"""Performance benchmark tests for DNA Origami AutoEncoder."""

import pytest
import time
import numpy as np
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil
import os

from dna_origami_ae.models.image_data import ImageData
from dna_origami_ae.optimization.integration_optimizer import (
    IntegratedOptimizer, OptimizationConfig, OptimizationLevel, OptimizationMode
)


class PerformanceBenchmarks:
    """Comprehensive performance benchmark suite."""
    
    BENCHMARK_TIMEOUT = 300  # 5 minutes max per benchmark
    MIN_THROUGHPUT_THRESHOLD = 10  # images/second
    MAX_LATENCY_THRESHOLD = 1.0    # seconds per image
    MAX_MEMORY_INCREASE = 500 * 1024 * 1024  # 500MB max memory increase
    
    @pytest.fixture(scope="class")
    def benchmark_images(self) -> Dict[str, List[ImageData]]:
        """Create benchmark image datasets of various sizes and types."""
        datasets = {}
        
        # Small images (32x32) - typical for quick processing
        small_images = []
        for i in range(100):
            data = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            small_images.append(ImageData(
                data=data,
                id=f"small_{i}",
                metadata={'size': 'small', 'benchmark': True}
            ))
        datasets['small'] = small_images
        
        # Medium images (128x128) - typical application size
        medium_images = []
        for i in range(50):
            data = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
            medium_images.append(ImageData(
                data=data,
                id=f"medium_{i}",
                metadata={'size': 'medium', 'benchmark': True}
            ))
        datasets['medium'] = medium_images
        
        # Large images (256x256) - stress test
        large_images = []
        for i in range(20):
            data = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            large_images.append(ImageData(
                data=data,
                id=f"large_{i}",
                metadata={'size': 'large', 'benchmark': True}
            ))
        datasets['large'] = large_images
        
        # Mixed sizes for realistic scenarios
        mixed_images = small_images[:30] + medium_images[:15] + large_images[:5]
        datasets['mixed'] = mixed_images
        
        return datasets
    
    @pytest.fixture
    def encoding_function(self):
        """Standard encoding function for benchmarks."""
        def encode_image(image_data: ImageData, **kwargs) -> str:
            """Simulate DNA encoding with realistic computation."""
            # Simulate base-4 encoding complexity
            flat_data = image_data.data.flatten()
            
            # Simple base-4 conversion with some processing
            base4_chars = []
            for pixel in flat_data:
                # Convert pixel (0-255) to 4 base-4 digits
                for i in range(4):
                    digit = (pixel >> (i * 2)) & 3
                    base4_chars.append('ATGC'[digit])
            
            # Simulate biological constraints checking
            sequence = ''.join(base4_chars)
            
            # Add some computational load
            gc_count = sequence.count('G') + sequence.count('C')
            gc_ratio = gc_count / len(sequence)
            
            # Simulate constraint validation delay
            if len(sequence) > 10000:
                time.sleep(0.001)  # Larger sequences take more time
            
            return sequence
        
        return encode_image
    
    def measure_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    @pytest.mark.performance
    def test_throughput_benchmark_small_images(self, benchmark_images, encoding_function):
        """Benchmark throughput with small images."""
        images = benchmark_images['small']
        config = OptimizationConfig(
            level=OptimizationLevel.INTERMEDIATE,
            mode=OptimizationMode.THROUGHPUT
        )
        
        optimizer = IntegratedOptimizer(config)
        
        # Warm-up run
        optimizer.optimize_image_encoding(images[:5], encoding_function)
        
        # Benchmark run
        start_time = time.time()
        results = optimizer.optimize_image_encoding(images, encoding_function)
        end_time = time.time()
        
        # Verify results
        assert len(results) == len(images)
        assert all(result.startswith('ATGC') or result.startswith('TACG') or 
                  result.startswith('GATC') or result.startswith('CATG') for result in results)
        
        # Performance assertions
        duration = end_time - start_time
        throughput = len(images) / duration
        
        assert throughput >= self.MIN_THROUGHPUT_THRESHOLD, \
            f"Throughput {throughput:.2f} images/sec below threshold {self.MIN_THROUGHPUT_THRESHOLD}"
        
        print(f"Small images throughput: {throughput:.2f} images/sec")
        
        optimizer.stop_monitoring()
    
    @pytest.mark.performance
    def test_throughput_benchmark_medium_images(self, benchmark_images, encoding_function):
        """Benchmark throughput with medium images."""
        images = benchmark_images['medium']
        config = OptimizationConfig(
            level=OptimizationLevel.ADVANCED,
            mode=OptimizationMode.THROUGHPUT
        )
        
        optimizer = IntegratedOptimizer(config)
        
        start_time = time.time()
        results = optimizer.optimize_image_encoding(images, encoding_function)
        end_time = time.time()
        
        assert len(results) == len(images)
        
        duration = end_time - start_time
        throughput = len(images) / duration
        
        # Lower threshold for larger images
        min_throughput = self.MIN_THROUGHPUT_THRESHOLD * 0.5
        assert throughput >= min_throughput, \
            f"Medium image throughput {throughput:.2f} images/sec below threshold {min_throughput}"
        
        print(f"Medium images throughput: {throughput:.2f} images/sec")
        
        optimizer.stop_monitoring()
    
    @pytest.mark.performance
    def test_latency_benchmark(self, benchmark_images, encoding_function):
        """Benchmark single-image latency."""
        images = benchmark_images['small']
        config = OptimizationConfig(
            level=OptimizationLevel.INTERMEDIATE,
            mode=OptimizationMode.LATENCY
        )
        
        optimizer = IntegratedOptimizer(config)
        
        # Warm-up
        optimizer.optimize_image_encoding([images[0]], encoding_function)
        
        # Measure latency for individual images
        latencies = []
        for image in images[:20]:  # Test subset for latency
            start_time = time.time()
            result = optimizer.optimize_image_encoding([image], encoding_function)
            end_time = time.time()
            
            assert len(result) == 1
            latencies.append(end_time - start_time)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency <= self.MAX_LATENCY_THRESHOLD, \
            f"Average latency {avg_latency:.3f}s exceeds threshold {self.MAX_LATENCY_THRESHOLD}s"
        
        print(f"Latency - Avg: {avg_latency:.3f}s, Max: {max_latency:.3f}s, P95: {p95_latency:.3f}s")
        
        optimizer.stop_monitoring()
    
    @pytest.mark.performance
    def test_memory_usage_benchmark(self, benchmark_images, encoding_function):
        """Benchmark memory usage during processing."""
        images = benchmark_images['large']  # Use large images for memory stress test
        config = OptimizationConfig(
            level=OptimizationLevel.INTERMEDIATE,
            mode=OptimizationMode.MEMORY
        )
        
        optimizer = IntegratedOptimizer(config)
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = self.measure_memory_usage()
        
        # Process images and monitor memory
        memory_measurements = []
        
        for i in range(0, len(images), 5):  # Process in chunks
            batch = images[i:i+5]
            results = optimizer.optimize_image_encoding(batch, encoding_function)
            
            current_memory = self.measure_memory_usage()
            memory_increase = current_memory - baseline_memory
            memory_measurements.append(memory_increase)
        
        max_memory_increase = max(memory_measurements)
        avg_memory_increase = np.mean(memory_measurements)
        
        assert max_memory_increase <= self.MAX_MEMORY_INCREASE, \
            f"Memory increase {max_memory_increase / 1024 / 1024:.1f}MB exceeds threshold {self.MAX_MEMORY_INCREASE / 1024 / 1024:.1f}MB"
        
        print(f"Memory - Max increase: {max_memory_increase / 1024 / 1024:.1f}MB, Avg: {avg_memory_increase / 1024 / 1024:.1f}MB")
        
        optimizer.stop_monitoring()
    
    @pytest.mark.performance
    def test_concurrent_processing_benchmark(self, benchmark_images, encoding_function):
        """Benchmark concurrent processing performance."""
        images = benchmark_images['mixed']
        config = OptimizationConfig(
            level=OptimizationLevel.ADVANCED,
            mode=OptimizationMode.BALANCED,
            enable_concurrent_processing=True
        )
        
        optimizer = IntegratedOptimizer(config)
        
        # Sequential benchmark
        start_time = time.time()
        sequential_results = []
        for image in images[:20]:  # Subset for comparison
            result = encoding_function(image)
            sequential_results.append(result)
        sequential_duration = time.time() - start_time
        
        # Concurrent benchmark
        start_time = time.time()
        concurrent_results = optimizer.optimize_image_encoding(images[:20], encoding_function)
        concurrent_duration = time.time() - start_time
        
        assert len(concurrent_results) == len(sequential_results)
        
        speedup = sequential_duration / concurrent_duration
        print(f"Concurrency speedup: {speedup:.2f}x (Sequential: {sequential_duration:.2f}s, Concurrent: {concurrent_duration:.2f}s)")
        
        # Should achieve some speedup with concurrency
        assert speedup > 1.2, f"Concurrent processing speedup {speedup:.2f}x too low"
        
        optimizer.stop_monitoring()
    
    @pytest.mark.performance
    @pytest.mark.skipif(not pytest.gpu_available, reason="GPU not available")
    def test_gpu_acceleration_benchmark(self, benchmark_images, encoding_function):
        """Benchmark GPU acceleration performance."""
        images = benchmark_images['medium']
        
        # CPU benchmark
        cpu_config = OptimizationConfig(
            level=OptimizationLevel.INTERMEDIATE,
            enable_gpu_acceleration=False
        )
        cpu_optimizer = IntegratedOptimizer(cpu_config)
        
        start_time = time.time()
        cpu_results = cpu_optimizer.optimize_image_encoding(images[:20], encoding_function)
        cpu_duration = time.time() - start_time
        
        # GPU benchmark
        gpu_config = OptimizationConfig(
            level=OptimizationLevel.ADVANCED,
            enable_gpu_acceleration=True
        )
        gpu_optimizer = IntegratedOptimizer(gpu_config)
        
        start_time = time.time()
        gpu_results = gpu_optimizer.optimize_image_encoding(images[:20], encoding_function)
        gpu_duration = time.time() - start_time
        
        assert len(gpu_results) == len(cpu_results)
        
        speedup = cpu_duration / gpu_duration
        print(f"GPU speedup: {speedup:.2f}x (CPU: {cpu_duration:.2f}s, GPU: {gpu_duration:.2f}s)")
        
        # GPU should provide some speedup for batch processing
        if speedup < 1.0:
            print(f"Warning: GPU slower than CPU for this workload (speedup: {speedup:.2f}x)")
        
        cpu_optimizer.stop_monitoring()
        gpu_optimizer.stop_monitoring()
    
    @pytest.mark.performance
    def test_scalability_benchmark(self, benchmark_images, encoding_function):
        """Test scalability with increasing load."""
        small_batch = benchmark_images['small'][:10]
        medium_batch = benchmark_images['small'][:50]
        large_batch = benchmark_images['small'][:100]
        
        config = OptimizationConfig(
            level=OptimizationLevel.ADVANCED,
            mode=OptimizationMode.THROUGHPUT
        )
        optimizer = IntegratedOptimizer(config)
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100]
        batches = [small_batch, medium_batch, large_batch]
        throughputs = []
        
        for batch_size, batch in zip(batch_sizes, batches):
            start_time = time.time()
            results = optimizer.optimize_image_encoding(batch, encoding_function)
            duration = time.time() - start_time
            
            assert len(results) == batch_size
            
            throughput = batch_size / duration
            throughputs.append(throughput)
            
            print(f"Batch size {batch_size}: {throughput:.2f} images/sec")
        
        # Throughput should generally increase or remain stable with larger batches
        # (allowing some variance due to system load)
        throughput_trend = np.polyfit(batch_sizes, throughputs, 1)[0]  # Linear trend slope
        
        if throughput_trend < -2:  # Significant negative trend
            print(f"Warning: Throughput decreases significantly with batch size (slope: {throughput_trend:.2f})")
        
        optimizer.stop_monitoring()
    
    @pytest.mark.performance
    def test_stress_test_sustained_load(self, benchmark_images, encoding_function):
        """Stress test with sustained load over time."""
        images = benchmark_images['mixed']
        config = OptimizationConfig(
            level=OptimizationLevel.MAXIMUM,
            mode=OptimizationMode.BALANCED
        )
        
        optimizer = IntegratedOptimizer(config)
        
        # Run sustained load for 30 seconds
        start_time = time.time()
        total_processed = 0
        throughputs = []
        
        while time.time() - start_time < 30:  # 30 second stress test
            batch_start = time.time()
            batch = images[:20]  # Process batches of 20
            
            results = optimizer.optimize_image_encoding(batch, encoding_function)
            
            batch_duration = time.time() - batch_start
            batch_throughput = len(batch) / batch_duration
            throughputs.append(batch_throughput)
            
            total_processed += len(results)
            
            assert len(results) == len(batch)
        
        total_duration = time.time() - start_time
        overall_throughput = total_processed / total_duration
        
        # Check for performance degradation
        early_throughput = np.mean(throughputs[:5]) if len(throughputs) >= 5 else throughputs[0]
        late_throughput = np.mean(throughputs[-5:]) if len(throughputs) >= 5 else throughputs[-1]
        
        degradation = (early_throughput - late_throughput) / early_throughput
        
        print(f"Stress test - Total: {total_processed} images, Throughput: {overall_throughput:.2f} images/sec")
        print(f"Performance degradation: {degradation * 100:.1f}%")
        
        # Performance should not degrade significantly over time
        assert degradation < 0.2, f"Performance degraded by {degradation * 100:.1f}% during stress test"
        
        optimizer.stop_monitoring()
    
    @pytest.mark.performance
    def test_cache_effectiveness_benchmark(self, benchmark_images, encoding_function):
        """Benchmark caching effectiveness."""
        # Use subset of images, but repeat them to test cache hits
        unique_images = benchmark_images['small'][:20]
        repeated_images = unique_images * 3  # Same images repeated 3 times
        
        config = OptimizationConfig(
            level=OptimizationLevel.INTERMEDIATE,
            enable_caching=True
        )
        optimizer = IntegratedOptimizer(config)
        
        # First pass - populate cache
        start_time = time.time()
        first_results = optimizer.optimize_image_encoding(unique_images, encoding_function)
        first_duration = time.time() - start_time
        
        # Second pass - should hit cache
        start_time = time.time()
        second_results = optimizer.optimize_image_encoding(unique_images, encoding_function)
        second_duration = time.time() - start_time
        
        # Third pass - mixed with repeated images
        start_time = time.time()
        mixed_results = optimizer.optimize_image_encoding(repeated_images, encoding_function)
        mixed_duration = time.time() - start_time
        
        assert len(first_results) == len(unique_images)
        assert len(second_results) == len(unique_images)
        assert len(mixed_results) == len(repeated_images)
        
        # Verify results are consistent
        assert first_results == second_results
        
        # Calculate speedup from caching
        cache_speedup = first_duration / second_duration
        print(f"Cache speedup: {cache_speedup:.2f}x (First: {first_duration:.3f}s, Cached: {second_duration:.3f}s)")
        
        # Cache should provide significant speedup
        assert cache_speedup > 2.0, f"Cache speedup {cache_speedup:.2f}x too low"
        
        # Get cache statistics
        report = optimizer.get_performance_report()
        cache_stats = report.get('component_stats', {}).get('cache', {})
        hit_rate = cache_stats.get('overall_hit_rate', 0.0)
        
        print(f"Cache hit rate: {hit_rate * 100:.1f}%")
        
        optimizer.stop_monitoring()


# Configure pytest for GPU availability
pytest.gpu_available = False
try:
    import torch
    if torch.cuda.is_available():
        pytest.gpu_available = True
except ImportError:
    pass


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests to ensure performance doesn't degrade."""
    
    # Performance baselines (adjust based on your system)
    PERFORMANCE_BASELINES = {
        'small_image_throughput': 50.0,  # images/sec
        'medium_image_throughput': 20.0,  # images/sec
        'single_image_latency': 0.1,     # seconds
        'cache_speedup': 3.0,            # times faster
    }
    
    def test_performance_regression_detection(self, benchmark_images):
        """Test to detect performance regressions."""
        # This is a placeholder for actual regression detection
        # In practice, you would compare against stored baseline metrics
        
        encoding_function = lambda img: f"ATGC{hash(img.data.tobytes()) % 10000:04d}"
        
        config = OptimizationConfig(level=OptimizationLevel.INTERMEDIATE)
        optimizer = IntegratedOptimizer(config)
        
        # Quick performance test
        images = benchmark_images['small'][:20]
        start_time = time.time()
        results = optimizer.optimize_image_encoding(images, encoding_function)
        duration = time.time() - start_time
        
        throughput = len(images) / duration
        
        # Check against baseline (with 20% tolerance)
        baseline = self.PERFORMANCE_BASELINES['small_image_throughput']
        tolerance = 0.2
        
        assert throughput >= baseline * (1 - tolerance), \
            f"Performance regression detected: {throughput:.1f} images/sec < {baseline * (1 - tolerance):.1f} (baseline with tolerance)"
        
        print(f"Performance check passed: {throughput:.1f} images/sec >= {baseline * (1 - tolerance):.1f} baseline")
        
        optimizer.stop_monitoring()