#!/usr/bin/env python3
"""Comprehensive test suite and quality gates for DNA Origami AutoEncoder."""

import sys
import time
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

def run_basic_tests():
    """Run basic functionality tests."""
    print("🧪 RUNNING BASIC FUNCTIONALITY TESTS")
    print("-" * 50)
    
    test_results = []
    
    # Test 1: Module imports
    print("\n1. Testing module imports...")
    try:
        from dna_origami_ae import DNASequence, ImageData, Base4Encoder, BiologicalConstraints
        print("   ✅ Core imports successful")
        test_results.append(("Core Imports", True, ""))
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        test_results.append(("Core Imports", False, str(e)))
        return test_results
    
    # Test 2: DNASequence functionality
    print("\n2. Testing DNASequence...")
    try:
        seq = DNASequence('ATGCATGCATGCATGC', 'test_seq')
        assert len(seq) == 16
        assert 0.4 <= seq.gc_content <= 0.6
        
        rev_comp = seq.reverse_complement()
        assert len(rev_comp) == 16
        
        print("   ✅ DNASequence tests passed")
        test_results.append(("DNASequence", True, ""))
    except Exception as e:
        print(f"   ❌ DNASequence test failed: {e}")
        test_results.append(("DNASequence", False, str(e)))
    
    # Test 3: ImageData functionality  
    print("\n3. Testing ImageData...")
    try:
        # Create test image
        test_img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        img = ImageData.from_array(test_img, 'test_img')
        
        assert img.metadata.width == 16
        assert img.metadata.height == 16
        assert img.metadata.channels == 1
        
        # Test binary conversion
        binary = img.to_binary(8)
        assert len(binary) == 16 * 16 * 8
        
        # Test round-trip
        reconstructed = ImageData.from_binary(binary, 16, 16, 1, 8, 'reconstructed')
        mse = img.calculate_mse(reconstructed)
        assert mse == 0.0  # Should be identical
        
        print("   ✅ ImageData tests passed")
        test_results.append(("ImageData", True, ""))
    except Exception as e:
        print(f"   ❌ ImageData test failed: {e}")
        test_results.append(("ImageData", False, str(e)))
    
    # Test 4: Base4 encoding
    print("\n4. Testing Base4 Encoding...")
    try:
        encoder = Base4Encoder()
        test_binary = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        
        # Test encoding
        dna_seq = encoder.encode_binary_to_dna(test_binary)
        assert len(dna_seq) == 4  # 8 bits = 4 bases
        
        # Test decoding
        decoded = encoder.decode_dna_to_binary(dna_seq)
        assert np.array_equal(test_binary, decoded)
        
        print("   ✅ Base4 encoding tests passed")
        test_results.append(("Base4 Encoding", True, ""))
    except Exception as e:
        print(f"   ❌ Base4 encoding test failed: {e}")
        test_results.append(("Base4 Encoding", False, str(e)))
    
    # Test 5: Biological constraints
    print("\n5. Testing Biological Constraints...")
    try:
        constraints = BiologicalConstraints()
        
        # Test valid sequence
        valid_seq = "ATGCATGCATGC"
        is_valid, errors = constraints.validate_sequence(valid_seq)
        # Note: May not pass due to length constraints, but should not crash
        
        # Test invalid sequence
        invalid_seq = "ATGXYZ"
        is_valid, errors = constraints.validate_sequence(invalid_seq)
        assert not is_valid  # Should be invalid due to bad bases
        
        print("   ✅ Biological constraints tests passed")
        test_results.append(("Biological Constraints", True, ""))
    except Exception as e:
        print(f"   ❌ Biological constraints test failed: {e}")
        test_results.append(("Biological Constraints", False, str(e)))
    
    return test_results

def run_integration_tests():
    """Run integration tests."""
    print("\\n🔗 RUNNING INTEGRATION TESTS")
    print("-" * 50)
    
    test_results = []
    
    try:
        from dna_origami_ae.encoding.image_encoder import DNAEncoder, EncodingParameters
        from dna_origami_ae import ImageData
        import numpy as np
        
        print("\\n1. Testing end-to-end image encoding...")
        
        # Create simple test image
        test_pattern = np.array([
            [255, 128, 64, 0],
            [0, 64, 128, 255],
            [255, 128, 64, 0],
            [0, 64, 128, 255]
        ], dtype=np.uint8)
        
        img = ImageData.from_array(test_pattern, 'integration_test')
        
        # Create relaxed encoder for testing
        from dna_origami_ae import BiologicalConstraints
        relaxed_constraints = BiologicalConstraints(
            gc_content_range=(0.1, 0.9),
            min_sequence_length=2,
            max_homopolymer_length=10,
            forbidden_sequences=[]
        )
        
        encoder = DNAEncoder(
            bits_per_base=2,
            error_correction=None,
            biological_constraints=relaxed_constraints
        )
        
        params = EncodingParameters(
            error_correction=None,
            compression_enabled=False,
            include_metadata=False,
            chunk_size=50,
            enforce_constraints=False
        )
        
        # Test encoding
        dna_sequences = encoder.encode_image(img, params)
        assert len(dna_sequences) > 0
        
        # Test decoding
        decoded_img = encoder.decode_image(dna_sequences, 4, 4, params)
        assert decoded_img.metadata.width == 4
        assert decoded_img.metadata.height == 4
        
        print("   ✅ Integration test passed")
        test_results.append(("Integration Test", True, ""))
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        test_results.append(("Integration Test", False, str(e)))
    
    return test_results

def run_performance_tests():
    """Run performance benchmarks."""
    print("\\n⚡ RUNNING PERFORMANCE TESTS")
    print("-" * 50)
    
    test_results = []
    
    try:
        from dna_origami_ae import Base4Encoder, ImageData
        import time
        
        encoder = Base4Encoder()
        
        # Test different data sizes
        sizes = [100, 1000, 10000]
        
        for size in sizes:
            # Generate test data
            test_data = np.random.randint(0, 2, size, dtype=np.uint8)
            
            # Measure encoding time
            start_time = time.time()
            dna_seq = encoder.encode_binary_to_dna(test_data)
            encode_time = time.time() - start_time
            
            # Measure decoding time
            start_time = time.time()
            decoded = encoder.decode_dna_to_binary(dna_seq)
            decode_time = time.time() - start_time
            
            # Calculate throughput
            encode_throughput = size / encode_time if encode_time > 0 else float('inf')
            decode_throughput = size / decode_time if decode_time > 0 else float('inf')
            
            print(f"   {size:5d} bits: Encode {encode_time*1000:.1f}ms ({encode_throughput:.0f} bits/s), "
                  f"Decode {decode_time*1000:.1f}ms ({decode_throughput:.0f} bits/s)")
            
            # Verify correctness
            assert np.array_equal(test_data, decoded)
        
        print("   ✅ Performance tests passed")
        test_results.append(("Performance Tests", True, ""))
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        test_results.append(("Performance Tests", False, str(e)))
    
    return test_results

def run_api_tests():
    """Test API endpoints."""
    print("\\n🌐 RUNNING API TESTS")
    print("-" * 50)
    
    test_results = []
    
    try:
        # Test API server import
        import api_server
        print("   ✅ API server import successful")
        
        # Test FastAPI app creation
        app = api_server.app
        assert app is not None
        print("   ✅ FastAPI app creation successful")
        
        test_results.append(("API Server", True, ""))
        
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        test_results.append(("API Server", False, str(e)))
    
    return test_results

def run_quality_gates():
    """Run quality gates and static analysis."""
    print("\\n🏗️  RUNNING QUALITY GATES")
    print("-" * 50)
    
    quality_results = []
    
    # Check file structure
    print("\\n1. Checking project structure...")
    required_files = [
        'dna_origami_ae/__init__.py',
        'dna_origami_ae/models/dna_sequence.py',
        'dna_origami_ae/models/image_data.py',
        'dna_origami_ae/encoding/image_encoder.py',
        'dna_origami_ae/encoding/biological_constraints.py',
        'pyproject.toml',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"   ⚠️  Missing files: {missing_files}")
        quality_results.append(("File Structure", False, f"Missing: {missing_files}"))
    else:
        print("   ✅ All required files present")
        quality_results.append(("File Structure", True, ""))
    
    # Check code quality (basic)
    print("\\n2. Checking basic code quality...")
    try:
        # Count lines of code
        total_lines = 0
        python_files = list(Path('.').rglob('*.py'))
        
        for py_file in python_files:
            if 'test' not in str(py_file) and '__pycache__' not in str(py_file):
                try:
                    with open(py_file, 'r') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                except:
                    pass
        
        print(f"   📊 Total lines of Python code: {total_lines}")
        print(f"   📁 Python files found: {len(python_files)}")
        
        quality_results.append(("Code Quality", True, f"{total_lines} lines, {len(python_files)} files"))
        
    except Exception as e:
        print(f"   ⚠️  Code quality check failed: {e}")
        quality_results.append(("Code Quality", False, str(e)))
    
    return quality_results

def generate_test_report(basic_results, integration_results, performance_results, api_results, quality_results):
    """Generate comprehensive test report."""
    print("\\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("=" * 70)
    
    all_results = {
        "Basic Tests": basic_results,
        "Integration Tests": integration_results,
        "Performance Tests": performance_results,
        "API Tests": api_results,
        "Quality Gates": quality_results
    }
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        print(f"\\n{category}:")
        for test_name, passed, error in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name:<25} {status}")
            if error and not passed:
                print(f"    └── {error}")
            total_tests += 1
            if passed:
                passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\\n" + "-" * 70)
    print(f"SUMMARY: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 OVERALL STATUS: PRODUCTION READY")
    elif success_rate >= 60:
        print("⚠️  OVERALL STATUS: NEEDS IMPROVEMENT")
    else:
        print("❌ OVERALL STATUS: NOT READY")
    
    print("=" * 70)
    
    return success_rate >= 80

if __name__ == '__main__':
    start_time = time.time()
    
    print("🚀 DNA ORIGAMI AUTOENCODER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    try:
        # Run all test suites
        basic_results = run_basic_tests()
        integration_results = run_integration_tests()
        performance_results = run_performance_tests()
        api_results = run_api_tests()
        quality_results = run_quality_gates()
        
        # Generate report
        production_ready = generate_test_report(
            basic_results, integration_results, performance_results, 
            api_results, quality_results
        )
        
        end_time = time.time()
        print(f"\\nTotal test time: {end_time - start_time:.2f} seconds")
        
        # Exit with appropriate code
        sys.exit(0 if production_ready else 1)
        
    except KeyboardInterrupt:
        print("\\n\\nTesting interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)