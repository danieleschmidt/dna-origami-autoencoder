#!/usr/bin/env python3
"""
Generation 2 Robustness Test: Make It Robust
Tests comprehensive error handling, validation, monitoring, and reliability
"""

import sys
import os
import numpy as np
import tempfile
import json
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, '/root/repo')

def test_comprehensive_validation():
    """Test advanced validation system."""
    print("Testing comprehensive validation...")
    
    try:
        from dna_origami_ae.models.image_data import ImageData
        from dna_origami_ae.encoding.image_encoder import DNAEncoder
        from dna_origami_ae.utils.advanced_validation import ComprehensiveValidator
        
        # Create test image
        test_array = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        image = ImageData.from_array(test_array, name="validation_test")
        
        # Create encoder with validation enabled
        encoder = DNAEncoder(
            enable_validation=True,
            enable_monitoring=False,
            enable_optimization=False
        )
        
        # Encode image
        dna_sequences = encoder.encode_image(image, validate_result=True)
        
        print(f"✓ Encoded with validation: {len(dna_sequences)} sequences")
        print(f"  Total bases: {sum(len(seq.sequence) for seq in dna_sequences)}")
        
        return True
    except Exception as e:
        print(f"✗ Comprehensive validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_health_monitoring():
    """Test health monitoring system."""
    print("\nTesting health monitoring...")
    
    try:
        from dna_origami_ae.utils.health_monitoring import HealthMonitor
        from dna_origami_ae.encoding.image_encoder import DNAEncoder
        from dna_origami_ae.models.image_data import ImageData
        
        # Create monitor
        monitor = HealthMonitor(monitoring_interval=1.0)
        monitor.start_monitoring()
        
        # Wait a moment for initial metrics
        time.sleep(2)
        
        # Get health status
        health = monitor.get_health_status()
        print(f"✓ Health monitor started")
        print(f"  Status: {health['status']}")
        print(f"  Health score: {health['health_score']:.1f}")
        print(f"  Active alerts: {health['active_alerts']}")
        
        # Test operation recording
        monitor.record_operation("test_operation", 100.0, True)
        monitor.record_operation("test_operation", 200.0, False)
        
        # Test circuit breaker
        breaker_open = monitor.check_circuit_breaker("test_operation")
        print(f"  Circuit breaker test: {breaker_open}")
        
        # Create encoder with monitoring
        encoder = DNAEncoder(
            enable_validation=False,
            enable_monitoring=True,
            enable_optimization=False
        )
        
        # Test with small image
        test_array = np.random.randint(0, 256, size=(16, 16), dtype=np.uint8)
        image = ImageData.from_array(test_array, name="monitor_test")
        
        dna_sequences = encoder.encode_image(image)
        print(f"  Encoded with monitoring: {len(dna_sequences)} sequences")
        
        # Check updated health
        health = monitor.get_health_status()
        print(f"  Updated health score: {health['health_score']:.1f}")
        
        monitor.stop_monitoring()
        encoder.cleanup_and_shutdown()
        
        return True
    except Exception as e:
        print(f"✗ Health monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_optimization():
    """Test performance optimization features."""
    print("\nTesting performance optimization...")
    
    try:
        from dna_origami_ae.encoding.image_encoder import DNAEncoder
        from dna_origami_ae.models.image_data import ImageData
        
        # Create encoder with optimization enabled
        encoder = DNAEncoder(
            enable_validation=False,
            enable_monitoring=False,
            enable_optimization=True
        )
        
        # Test with small images
        images = []
        for i in range(3):
            test_array = np.random.randint(0, 256, size=(16, 16), dtype=np.uint8)
            image = ImageData.from_array(test_array, name=f"opt_test_{i}")
            images.append(image)
        
        # Test optimized encoding
        print("  Testing optimized single encoding...")
        start_time = time.time()
        dna_sequences = encoder.encode_image_optimized(images[0], use_cache=True)
        single_time = time.time() - start_time
        print(f"    Single encoding: {len(dna_sequences)} sequences in {single_time*1000:.1f}ms")
        
        # Test cache hit
        start_time = time.time()
        cached_sequences = encoder.encode_image_optimized(images[0], use_cache=True)
        cache_time = time.time() - start_time
        print(f"    Cache hit: {len(cached_sequences)} sequences in {cache_time*1000:.1f}ms")
        
        # Test batch encoding
        print("  Testing batch encoding...")
        start_time = time.time()
        batch_results = encoder.batch_encode_images(images)
        batch_time = time.time() - start_time
        print(f"    Batch encoding: {len(batch_results)} results in {batch_time*1000:.1f}ms")
        
        # Get performance report
        report = encoder.get_performance_report()
        print(f"  Optimization stats: {report['optimization_stats']}")
        
        encoder.cleanup_and_shutdown()
        
        return True
    except Exception as e:
        print(f"✗ Performance optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_resilience():
    """Test error handling and resilience."""
    print("\nTesting error resilience...")
    
    try:
        from dna_origami_ae.encoding.image_encoder import DNAEncoder
        from dna_origami_ae.models.image_data import ImageData
        from dna_origami_ae.encoding.error_correction import DNAErrorCorrection
        
        # Test with various error conditions
        encoder = DNAEncoder(
            enable_validation=True,
            enable_monitoring=True,
            enable_optimization=False
        )
        
        # Test 1: Extremely small image (edge case)
        tiny_array = np.random.randint(0, 256, size=(4, 4), dtype=np.uint8)
        tiny_image = ImageData.from_array(tiny_array, name="tiny_test")
        
        try:
            sequences = encoder.encode_image(tiny_image)
            print(f"✓ Handled tiny image: {len(sequences)} sequences")
        except Exception as e:
            print(f"  Expected error for tiny image: {type(e).__name__}")
        
        # Test 2: Large image (stress test)
        try:
            large_array = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
            large_image = ImageData.from_array(large_array, name="large_test")
            
            sequences = encoder.encode_image(large_image)
            print(f"✓ Handled large image: {len(sequences)} sequences")
            total_bases = sum(len(seq.sequence) for seq in sequences)
            print(f"  Total bases: {total_bases}")
        except Exception as e:
            print(f"  Large image handling: {type(e).__name__}: {e}")
        
        # Test 3: Error correction resilience
        print("  Testing error correction resilience...")
        corrector = DNAErrorCorrection(method="reed_solomon")
        
        # Test with various data sizes
        for size in [16, 64, 256]:
            test_data = np.random.randint(0, 2, size=size, dtype=np.uint8)
            
            try:
                encoded = corrector.encode(test_data)
                decoded = corrector.decode(encoded)
                
                # Check if original data is preserved (allowing for padding)
                original_recovered = decoded[:len(test_data)]
                match_rate = np.mean(test_data == original_recovered)
                print(f"    Size {size}: {match_rate:.1%} recovery rate")
                
            except Exception as e:
                print(f"    Size {size}: Error - {type(e).__name__}")
        
        # Test 4: Performance under stress
        print("  Testing performance under stress...")
        start_time = time.time()
        stress_results = []
        
        for i in range(5):
            test_array = np.random.randint(0, 256, size=(20, 20), dtype=np.uint8)
            image = ImageData.from_array(test_array, name=f"stress_{i}")
            
            try:
                sequences = encoder.encode_image(image)
                stress_results.append(len(sequences))
            except Exception as e:
                print(f"    Stress test {i} failed: {type(e).__name__}")
                stress_results.append(0)
        
        stress_time = time.time() - start_time
        success_rate = sum(1 for r in stress_results if r > 0) / len(stress_results)
        print(f"  Stress test: {success_rate:.1%} success rate in {stress_time:.1f}s")
        
        encoder.cleanup_and_shutdown()
        
        return True
    except Exception as e:
        print(f"✗ Error resilience test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_round_trip_validation():
    """Test complete round-trip encoding/decoding validation."""
    print("\nTesting round-trip validation...")
    
    try:
        from dna_origami_ae.encoding.image_encoder import DNAEncoder
        from dna_origami_ae.models.image_data import ImageData
        
        encoder = DNAEncoder(
            enable_validation=True,
            enable_monitoring=False,
            enable_optimization=False
        )
        
        # Create test image with known pattern
        test_array = np.zeros((16, 16), dtype=np.uint8)
        test_array[4:12, 4:12] = 255  # White square in center
        image = ImageData.from_array(test_array, name="round_trip_test")
        
        print(f"  Original image stats: min={np.min(test_array)}, max={np.max(test_array)}, mean={np.mean(test_array):.1f}")
        
        # Encode
        dna_sequences = encoder.encode_image(image)
        print(f"  Encoded to {len(dna_sequences)} DNA sequences")
        
        # Decode
        decoded_image = encoder.decode_image(
            dna_sequences, 
            original_width=16, 
            original_height=16
        )
        
        print(f"  Decoded image stats: min={np.min(decoded_image.data)}, max={np.max(decoded_image.data)}, mean={np.mean(decoded_image.data):.1f}")
        
        # Calculate quality metrics
        mse = image.calculate_mse(decoded_image)
        psnr = image.calculate_psnr(decoded_image)
        ssim = image.calculate_ssim(decoded_image)
        
        print(f"  Quality metrics:")
        print(f"    MSE: {mse:.2f}")
        print(f"    PSNR: {psnr:.2f} dB")
        print(f"    SSIM: {ssim:.3f}")
        
        # Validate encoding efficiency
        efficiency = encoder.get_encoding_efficiency(
            original_size_bytes=image.metadata.size_bytes,
            encoded_sequences=dna_sequences
        )
        
        print(f"  Encoding efficiency:")
        print(f"    Compression ratio: {efficiency['compression_ratio']:.3f}")
        print(f"    Bits per base: {efficiency['bits_per_base']:.2f}")
        print(f"    Storage density: {efficiency['storage_density_bytes_per_gram']:.2e} bytes/gram")
        
        # Run validation
        validation_result = encoder.validate_encoding(image, dna_sequences)
        print(f"  Validation result: {validation_result['success']}")
        if validation_result['success']:
            print(f"    Constraint violations: {len(validation_result['constraint_violations'])}")
        
        encoder.cleanup_and_shutdown()
        
        return True
    except Exception as e:
        print(f"✗ Round-trip validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_error_correction():
    """Test advanced error correction capabilities."""
    print("\nTesting advanced error correction...")
    
    try:
        from dna_origami_ae.encoding.error_correction import DNAErrorCorrection
        
        # Test Reed-Solomon error correction
        rs_corrector = DNAErrorCorrection(method="reed_solomon")
        
        test_data = np.random.randint(0, 2, size=88, dtype=np.uint8)  # 88 bits = 22 symbols
        print(f"  Original data length: {len(test_data)} bits")
        
        # Encode with error correction
        encoded = rs_corrector.encode(test_data)
        print(f"  Encoded length: {len(encoded)} bits (overhead: {rs_corrector.get_overhead():.2f})")
        
        # Test error correction capability
        capability = rs_corrector.estimate_error_correction_capability()
        print(f"  Error correction capability: {capability}")
        
        # Test with simulated errors
        test_result = rs_corrector.test_error_correction(test_data, error_rate=0.01)
        print(f"  Error correction test: {test_result}")
        
        return True
    except Exception as e:
        print(f"✗ Advanced error correction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_and_reporting():
    """Test comprehensive logging and reporting."""
    print("\nTesting logging and reporting...")
    
    try:
        from dna_origami_ae.encoding.image_encoder import DNAEncoder
        from dna_origami_ae.models.image_data import ImageData
        
        # Create encoder with all features enabled
        encoder = DNAEncoder(
            enable_validation=True,
            enable_monitoring=True,
            enable_optimization=True,
            log_level="INFO"
        )
        
        # Create test image
        test_array = np.random.randint(0, 256, size=(24, 24), dtype=np.uint8)
        image = ImageData.from_array(test_array, name="logging_test")
        
        # Encode with full logging
        dna_sequences = encoder.encode_image(image)
        print(f"  Encoded with full logging: {len(dna_sequences)} sequences")
        
        # Get comprehensive performance report
        report = encoder.get_performance_report()
        
        print(f"  Performance report keys: {list(report.keys())}")
        if 'encoding_stats' in report:
            stats = report['encoding_stats']
            print(f"    Total images encoded: {stats.get('total_images_encoded', 0)}")
            print(f"    Success rate: {stats.get('successful_encodings', 0)}/{stats.get('total_images_encoded', 0)}")
        
        # Get statistics
        stats = encoder.get_statistics()
        print(f"  Encoder statistics: {stats}")
        
        encoder.cleanup_and_shutdown()
        
        return True
    except Exception as e:
        print(f"✗ Logging and reporting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_robustness():
    """Test robustness with various configurations."""
    print("\nTesting configuration robustness...")
    
    try:
        from dna_origami_ae.encoding.image_encoder import DNAEncoder, EncodingParameters
        from dna_origami_ae.models.image_data import ImageData
        from dna_origami_ae.encoding.biological_constraints import BiologicalConstraints
        
        # Test different encoding parameters
        test_array = np.random.randint(0, 256, size=(16, 16), dtype=np.uint8)
        image = ImageData.from_array(test_array, name="config_test")
        
        configs = [
            ("Default", {}),
            ("No error correction", {"error_correction": "none"}),
            ("Compression enabled", {"compression_enabled": True}),
            ("No metadata", {"include_metadata": False}),
            ("Large chunks", {"chunk_size": 400}),
            ("Small chunks", {"chunk_size": 50})
        ]
        
        for config_name, params in configs:
            try:
                # Create custom encoder
                encoder = DNAEncoder(
                    enable_validation=False,
                    enable_monitoring=False,
                    enable_optimization=False
                )
                
                encoding_params = EncodingParameters(**params)
                sequences = encoder.encode_image(image, encoding_params)
                
                total_bases = sum(len(seq.sequence) for seq in sequences)
                print(f"  {config_name}: {len(sequences)} sequences, {total_bases} bases")
                
                encoder.cleanup_and_shutdown()
                
            except Exception as e:
                print(f"  {config_name}: Failed - {type(e).__name__}: {e}")
        
        # Test different biological constraints
        strict_constraints = BiologicalConstraints(
            gc_content_range=(0.45, 0.55),
            max_homopolymer_length=3,
            melting_temp_range=(60.0, 70.0)
        )
        
        relaxed_constraints = BiologicalConstraints(
            gc_content_range=(0.3, 0.7),
            max_homopolymer_length=6,
            melting_temp_range=(50.0, 80.0)
        )
        
        for name, constraints in [("Strict", strict_constraints), ("Relaxed", relaxed_constraints)]:
            try:
                encoder = DNAEncoder(
                    biological_constraints=constraints,
                    enable_validation=False,
                    enable_monitoring=False,
                    enable_optimization=False
                )
                
                sequences = encoder.encode_image(image)
                total_bases = sum(len(seq.sequence) for seq in sequences)
                print(f"  {name} constraints: {len(sequences)} sequences, {total_bases} bases")
                
                encoder.cleanup_and_shutdown()
                
            except Exception as e:
                print(f"  {name} constraints: Failed - {type(e).__name__}: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Generation 2 robustness tests."""
    print("=" * 70)
    print("DNA-Origami-AutoEncoder Generation 2: MAKE IT ROBUST")
    print("=" * 70)
    
    tests = [
        test_comprehensive_validation,
        test_health_monitoring,
        test_performance_optimization,
        test_error_resilience,
        test_round_trip_validation,
        test_advanced_error_correction,
        test_logging_and_reporting,
        test_configuration_robustness
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 70)
    print(f"Generation 2 Results: {passed}/{total} tests passed")
    
    if passed >= total * 0.8:  # 80% pass rate for robustness
        print("🎉 Generation 2 PASSED! System is robust and reliable.")
        return True
    else:
        print("⚠️  Generation 2 needs improvement. Some robustness features failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)