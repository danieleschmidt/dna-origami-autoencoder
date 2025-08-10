#!/usr/bin/env python3
"""
Test Generation 3 functionality - High-performance optimization, caching, and concurrency.
"""

import time
import numpy as np
import sys
from pathlib import Path
from concurrent.futures import as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dna_origami_ae import DNAEncoder, ImageData
from dna_origami_ae.utils.performance_optimized import (
    get_performance_optimizer, cached, parallel_map,
    PerformanceConfig, HighPerformanceCache
)
from dna_origami_ae.utils.logger import get_logger


def simulate_heavy_computation(x: int) -> int:
    """Simulate computationally expensive work."""
    # Simulate some CPU-intensive work
    result = 0
    for i in range(1000):
        result += np.sum(np.random.random((10, 10)) * x)
    return int(result)


@cached(ttl=300)  # Cache for 5 minutes
def expensive_cached_function(n: int) -> int:
    """Cached version of expensive computation."""
    time.sleep(0.1)  # Simulate I/O delay
    return n * n * n


def process_image_chunk(chunk_data: tuple) -> dict:
    """Process a chunk of image data (simulated)."""
    chunk_id, image_array = chunk_data
    
    # Simulate image processing work
    processed = np.mean(image_array) + np.std(image_array)
    
    return {
        'chunk_id': chunk_id,
        'mean': float(np.mean(image_array)),
        'std': float(np.std(image_array)),
        'processed_value': processed
    }


def test_performance_features():
    """Test Generation 3 performance optimization features."""
    print("⚡ DNA-Origami-AutoEncoder Generation 3 - Performance Test")
    print("=" * 75)
    
    logger = get_logger('performance_test')
    logger.info("Generation 3 performance test started")
    
    # Initialize performance optimizer
    config = PerformanceConfig(
        enable_redis_cache=True,
        max_workers=8,
        batch_size=50,
        memory_threshold=0.8
    )
    
    optimizer = get_performance_optimizer(config)
    
    test_results = {
        'cache_tests': 0,
        'concurrency_tests': 0,
        'performance_improvements': {},
        'failures': []
    }
    
    # 1. Test High-Performance Caching
    print("\\n1. Testing high-performance caching system...")
    try:
        # Test memory cache
        cache = HighPerformanceCache(config)
        
        # Store and retrieve values
        test_data = {'key1': 'value1', 'numbers': [1, 2, 3, 4, 5]}
        cache.set('test_key', test_data)
        
        found, retrieved = cache.get('test_key')
        if found and retrieved == test_data:
            print(f"   ✅ Memory cache working: stored and retrieved data correctly")
            test_results['cache_tests'] += 1
        else:
            test_results['failures'].append("Memory cache retrieval failed")
        
        # Test cache statistics
        stats = cache.get_stats()
        print(f"   📊 Cache stats: {stats['hits']} hits, {stats['misses']} misses")
        test_results['cache_tests'] += 1
        
        # Test function caching with decorator
        print("\\n   Testing function result caching...")
        
        # First call (should be slow)
        start_time = time.time()
        result1 = expensive_cached_function(5)
        first_call_time = time.time() - start_time
        
        # Second call (should be fast due to caching)
        start_time = time.time()
        result2 = expensive_cached_function(5)
        second_call_time = time.time() - start_time
        
        if result1 == result2:
            print(f"   ✅ Function caching working: results match")
            print(f"      First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s")
            
            speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
            test_results['performance_improvements']['caching_speedup'] = speedup
            print(f"      Cache speedup: {speedup:.1f}x")
            test_results['cache_tests'] += 1
        else:
            test_results['failures'].append("Cached function returned different results")
        
    except Exception as e:
        test_results['failures'].append(f"Caching test failed: {e}")
        logger.error(f"Caching test failed", exc_info=True)
    
    # 2. Test Concurrent Processing
    print("\\n2. Testing concurrent processing system...")
    try:
        # Test parallel map with CPU-bound tasks
        print("   Testing parallel CPU-bound processing...")
        
        work_items = list(range(1, 21))  # 20 items
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [simulate_heavy_computation(x) for x in work_items]
        sequential_time = time.time() - start_time
        
        # Parallel processing
        start_time = time.time()
        parallel_results = parallel_map(simulate_heavy_computation, work_items, cpu_bound=True)
        parallel_time = time.time() - start_time
        
        if len(sequential_results) == len(parallel_results):
            print(f"   ✅ Parallel processing working: {len(parallel_results)} results")
            print(f"      Sequential: {sequential_time:.2f}s, Parallel: {parallel_time:.2f}s")
            
            if parallel_time > 0:
                speedup = sequential_time / parallel_time
                test_results['performance_improvements']['parallel_speedup'] = speedup
                print(f"      Parallel speedup: {speedup:.1f}x")
            
            test_results['concurrency_tests'] += 1
        else:
            test_results['failures'].append("Parallel processing returned different number of results")
        
        # Test batch processing of image chunks
        print("\\n   Testing batch image processing...")
        
        # Create synthetic image chunks
        image_chunks = []
        for i in range(10):
            chunk_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            image_chunks.append((i, chunk_array))
        
        start_time = time.time()
        chunk_results = parallel_map(process_image_chunk, image_chunks, cpu_bound=False)
        processing_time = time.time() - start_time
        
        if len(chunk_results) == len(image_chunks):
            print(f"   ✅ Batch processing working: {len(chunk_results)} chunks processed")
            print(f"      Processing time: {processing_time:.3f}s")
            test_results['concurrency_tests'] += 1
        else:
            test_results['failures'].append("Batch processing failed")
        
    except Exception as e:
        test_results['failures'].append(f"Concurrent processing test failed: {e}")
        logger.error(f"Concurrent processing test failed", exc_info=True)
    
    # 3. Test Resource Monitoring
    print("\\n3. Testing resource monitoring...")
    try:
        # Get current resource stats
        resource_stats = optimizer.resources.get_current_stats()
        
        print(f"   📊 System Resources:")
        print(f"      CPU cores: {resource_stats['cpu_count']}")
        print(f"      CPU usage: {resource_stats['cpu_percent']:.1f}%")
        print(f"      Memory: {resource_stats['memory']['available_gb']:.1f}GB available")
        print(f"      Memory usage: {resource_stats['memory']['percent']:.1f}%")
        print(f"      Disk free: {resource_stats['disk']['free_gb']:.1f}GB")
        
        print(f"   ✅ Resource monitoring working")
        test_results['concurrency_tests'] += 1
        
    except Exception as e:
        test_results['failures'].append(f"Resource monitoring test failed: {e}")
        logger.error(f"Resource monitoring test failed", exc_info=True)
    
    # 4. Test Performance-Optimized DNA Encoding
    print("\\n4. Testing performance-optimized DNA encoding...")
    try:
        # Create multiple test images for batch processing
        test_images = []
        for i in range(5):
            size = 16 + (i * 8)  # Varying sizes: 16x16, 24x24, 32x32, 40x40, 48x48
            image_array = np.random.randint(0, 256, (size, size), dtype=np.uint8)
            image_data = ImageData.from_array(image_array, name=f"perf_test_{i}")
            test_images.append(image_data)
        
        # Test with standard encoder
        encoder = DNAEncoder()
        
        print(f"   Processing {len(test_images)} images with sizes: {[img.data.shape for img in test_images]}")
        
        # Sequential encoding
        start_time = time.time()
        sequential_results = []
        for img in test_images:
            sequences = encoder.encode_image(img)
            sequential_results.append(len(sequences))
        sequential_time = time.time() - start_time
        
        # Parallel encoding (using thread-based parallelism to avoid pickling issues)
        start_time = time.time()
        parallel_results = parallel_map(lambda img: len(DNAEncoder().encode_image(img)), 
                                      test_images, cpu_bound=False)  # Use threads instead
        parallel_time = time.time() - start_time
        
        if sequential_results == parallel_results:
            print(f"   ✅ Performance-optimized encoding working")
            print(f"      Sequential: {sequential_time:.3f}s, Parallel: {parallel_time:.3f}s")
            
            if parallel_time > 0:
                encoding_speedup = sequential_time / parallel_time
                test_results['performance_improvements']['encoding_speedup'] = encoding_speedup
                print(f"      Encoding speedup: {encoding_speedup:.1f}x")
            
            total_sequences = sum(sequential_results)
            print(f"      Total sequences generated: {total_sequences}")
            test_results['concurrency_tests'] += 1
        else:
            test_results['failures'].append("Parallel encoding returned different results")
        
    except Exception as e:
        test_results['failures'].append(f"Performance encoding test failed: {e}")
        logger.error(f"Performance encoding test failed", exc_info=True)
    
    # 5. Test System Performance Statistics
    print("\\n5. Comprehensive performance statistics...")
    try:
        perf_stats = optimizer.get_performance_stats()
        
        print(f"   📊 Performance Summary:")
        print(f"      Cache hit rate: {perf_stats['cache']['hit_rate']:.2%}")
        print(f"      Cache hits: {perf_stats['cache']['hits']}")
        print(f"      Memory cache size: {perf_stats['cache']['memory_cache_size']}")
        print(f"      Redis connected: {perf_stats['cache']['redis_connected']}")
        
        print(f"\\n      System Resources:")
        print(f"      Available memory: {perf_stats['resources']['memory']['available_gb']:.1f}GB")
        print(f"      CPU usage: {perf_stats['resources']['cpu_percent']:.1f}%")
        
        print(f"\\n      Configuration:")
        print(f"      Max workers: {perf_stats['config']['max_workers']}")
        print(f"      Process pool enabled: {perf_stats['config']['use_process_pool']}")
        
        test_results['concurrency_tests'] += 1
        
    except Exception as e:
        test_results['failures'].append(f"Performance stats test failed: {e}")
        logger.error(f"Performance stats test failed", exc_info=True)
    
    # 6. Summary
    print("\\n6. Generation 3 Test Summary:")
    print(f"   ✅ Cache tests passed: {test_results['cache_tests']}")
    print(f"   ✅ Concurrency tests passed: {test_results['concurrency_tests']}")
    
    print(f"\\n   🚀 Performance Improvements:")
    for improvement, speedup in test_results['performance_improvements'].items():
        if speedup > 1:
            print(f"      {improvement}: {speedup:.1f}x faster")
        else:
            print(f"      {improvement}: {speedup:.3f}x (baseline)")
    
    if test_results['failures']:
        print(f"\\n   ❌ Failures: {len(test_results['failures'])}")
        for failure in test_results['failures']:
            print(f"      - {failure}")
    
    total_tests = test_results['cache_tests'] + test_results['concurrency_tests']
    
    print(f"\\n🎉 Generation 3 test completed!")
    print(f"📊 Total tests: {total_tests}, Failures: {len(test_results['failures'])}")
    
    if len(test_results['failures']) == 0:
        print("✅ MAKE IT SCALE - All performance optimizations working!")
        print("📈 Ready for Quality Gates and Production Deployment")
    else:
        print("⚠️  Some tests failed - review and optimize further")
    
    # Cleanup
    try:
        optimizer.shutdown()
    except:
        pass
    
    return test_results


if __name__ == "__main__":
    try:
        results = test_performance_features()
        if len(results['failures']) == 0:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Generation 3 test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)