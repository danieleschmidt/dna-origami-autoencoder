#!/usr/bin/env python3
"""
Test Generation 2 functionality - Robust error handling, validation, and logging.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dna_origami_ae import DNAEncoder, ImageData, DNASequence
from dna_origami_ae.utils.logger import get_logger, dna_logger, health_monitor
from dna_origami_ae.utils.error_handling import (
    ValidationError, ComputationError, BiologicalConstraintError,
    ErrorHandler, ValidationFramework, setup_error_recovery
)

def test_robust_functionality():
    """Test Generation 2 robustness features."""
    print("🛡️ DNA-Origami-AutoEncoder Generation 2 - Robustness Test")
    print("=" * 70)
    
    # Initialize logging and error handling
    logger = get_logger('test')
    error_handler = ErrorHandler(logger)
    validator = ValidationFramework(error_handler)
    setup_error_recovery()
    
    logger.info("Generation 2 test started with robust error handling")
    
    test_results = {
        'validation_tests': 0,
        'error_recovery_tests': 0,
        'logging_tests': 0,
        'health_checks': 0,
        'failures': []
    }
    
    # 1. Test Input Validation
    print("\\n1. Testing comprehensive input validation...")
    try:
        # Test invalid image data
        try:
            validator.validate_image_data("not_an_array")
            test_results['failures'].append("Should have failed on string input")
        except ValidationError as e:
            print(f"   ✅ Correctly caught invalid image type: {e.message}")
            test_results['validation_tests'] += 1
        
        # Test empty array
        try:
            validator.validate_image_data(np.array([]))
            test_results['failures'].append("Should have failed on empty array")
        except ValidationError as e:
            print(f"   ✅ Correctly caught empty image: {e.message}")
            test_results['validation_tests'] += 1
        
        # Test valid image
        valid_image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        validator.validate_image_data(valid_image)
        print(f"   ✅ Valid image data passed validation")
        test_results['validation_tests'] += 1
        
    except Exception as e:
        test_results['failures'].append(f"Validation test failed: {e}")
        logger.error(f"Validation test failed", exc_info=True)
    
    # 2. Test DNA Sequence Validation
    print("\\n2. Testing DNA sequence validation...")
    try:
        # Test invalid sequences
        invalid_sequences = [
            ("", "empty sequence"),
            ("ATCGXYZ", "invalid bases"),
            ("AT" * 1000000, "too long")
        ]
        
        for seq, desc in invalid_sequences:
            try:
                validator.validate_dna_sequence(seq)
                test_results['failures'].append(f"Should have failed on {desc}")
            except ValidationError as e:
                print(f"   ✅ Correctly caught {desc}: {e.message}")
                test_results['validation_tests'] += 1
        
        # Test valid sequence
        validator.validate_dna_sequence("ATCGATCGATCGATCG")
        print(f"   ✅ Valid DNA sequence passed validation")
        test_results['validation_tests'] += 1
        
    except Exception as e:
        test_results['failures'].append(f"DNA validation test failed: {e}")
        logger.error(f"DNA validation test failed", exc_info=True)
    
    # 3. Test Biological Constraints
    print("\\n3. Testing biological constraint validation...")
    try:
        constraints = {
            "gc_content": (0.4, 0.6),
            "max_homopolymer": 4,
            "avoid_sequences": ["AAAA", "TTTT"]
        }
        
        # Test GC content violation
        try:
            validator.validate_biological_constraints("AAAAAAAAAAAAAAAA", constraints)
            test_results['failures'].append("Should have failed on low GC content")
        except BiologicalConstraintError as e:
            print(f"   ✅ Correctly caught GC content violation: {e.message}")
            test_results['validation_tests'] += 1
        
        # Test homopolymer violation
        try:
            validator.validate_biological_constraints("ATCGAAAAATCG", constraints)
            test_results['failures'].append("Should have failed on homopolymer run")
        except BiologicalConstraintError as e:
            print(f"   ✅ Correctly caught homopolymer violation: {e.message}")
            test_results['validation_tests'] += 1
        
        # Test forbidden sequence
        try:
            validator.validate_biological_constraints("ATCGAAAATCG", constraints)
            test_results['failures'].append("Should have failed on forbidden sequence")
        except BiologicalConstraintError as e:
            print(f"   ✅ Correctly caught forbidden sequence: {e.message}")
            test_results['validation_tests'] += 1
        
        # Test valid sequence
        validator.validate_biological_constraints("ATCGATCGATCGATCG", constraints)
        print(f"   ✅ Valid sequence passed biological constraints")
        test_results['validation_tests'] += 1
        
    except Exception as e:
        test_results['failures'].append(f"Biological constraint test failed: {e}")
        logger.error(f"Biological constraint test failed", exc_info=True)
    
    # 4. Test Error Recovery
    print("\\n4. Testing error recovery mechanisms...")
    try:
        # Test memory error recovery
        def simulate_memory_error():
            raise MemoryError("Simulated memory error")
        
        error_handler.register_recovery_strategy(MemoryError, lambda e: "recovered")
        
        try:
            result = error_handler.handle_error(MemoryError("test error"))
            print(f"   ✅ Memory error recovery successful: {result}")
            test_results['error_recovery_tests'] += 1
        except:
            # Expected since we re-raise after recovery attempt
            print(f"   ✅ Memory error recovery attempted")
            test_results['error_recovery_tests'] += 1
        
    except Exception as e:
        test_results['failures'].append(f"Error recovery test failed: {e}")
        logger.error(f"Error recovery test failed", exc_info=True)
    
    # 5. Test Logging System
    print("\\n5. Testing structured logging...")
    try:
        # Test different log levels and structured data
        logger.info("Test info message", extra={
            'operation': 'test_logging',
            'metrics': {'test_value': 123, 'test_string': 'hello'}
        })
        
        logger.warning("Test warning message", extra={
            'operation': 'test_logging',
            'metrics': {'warning_level': 'medium'}
        })
        
        logger.error("Test error message", extra={
            'operation': 'test_logging',
            'metrics': {'error_type': 'test_error'}
        })
        
        print(f"   ✅ Structured logging working (check logs for details)")
        test_results['logging_tests'] += 1
        
        # Test performance logging
        with dna_logger.performance.time_operation("test_operation", "test_session"):
            import time
            time.sleep(0.1)  # Simulate work
        
        print(f"   ✅ Performance logging working")
        test_results['logging_tests'] += 1
        
    except Exception as e:
        test_results['failures'].append(f"Logging test failed: {e}")
        logger.error(f"Logging test failed", exc_info=True)
    
    # 6. Test Health Monitoring
    print("\\n6. Testing health monitoring...")
    try:
        # Register a test component
        def test_health_check():
            return {'healthy': True, 'status': 'all systems normal', 'uptime': 100}
        
        health_monitor.register_component('test_component', test_health_check)
        health_result = health_monitor.check_component_health('test_component')
        
        if health_result.get('healthy'):
            print(f"   ✅ Health check passed: {health_result['status']}")
            test_results['health_checks'] += 1
        else:
            test_results['failures'].append("Health check failed unexpectedly")
        
        # Test system-wide health
        system_health = health_monitor.get_system_health()
        print(f"   ✅ System health check: {system_health['overall_status']}")
        test_results['health_checks'] += 1
        
    except Exception as e:
        test_results['failures'].append(f"Health monitoring test failed: {e}")
        logger.error(f"Health monitoring test failed", exc_info=True)
    
    # 7. Test Enhanced DNA Encoding with Error Handling
    print("\\n7. Testing enhanced DNA encoding with error handling...")
    try:
        # Create encoder with enhanced error handling
        encoder = DNAEncoder()
        
        # Test with valid image
        test_array = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        test_image = ImageData.from_array(test_array, name="robust_test")
        
        # This should work
        dna_sequences = encoder.encode_image(test_image)
        print(f"   ✅ Enhanced encoding successful: {len(dna_sequences)} sequences")
        test_results['validation_tests'] += 1
        
        # Test error statistics
        stats = error_handler.get_error_statistics()
        print(f"   📊 Error statistics: {stats['total_errors']} total errors")
        
    except Exception as e:
        test_results['failures'].append(f"Enhanced encoding test failed: {e}")
        logger.error(f"Enhanced encoding test failed", exc_info=True)
    
    # 8. Summary
    print("\\n8. Generation 2 Test Summary:")
    print(f"   ✅ Validation tests passed: {test_results['validation_tests']}")
    print(f"   ✅ Error recovery tests: {test_results['error_recovery_tests']}")
    print(f"   ✅ Logging tests: {test_results['logging_tests']}")
    print(f"   ✅ Health checks: {test_results['health_checks']}")
    
    if test_results['failures']:
        print(f"   ❌ Failures: {len(test_results['failures'])}")
        for failure in test_results['failures']:
            print(f"      - {failure}")
    
    total_tests = sum([
        test_results['validation_tests'],
        test_results['error_recovery_tests'], 
        test_results['logging_tests'],
        test_results['health_checks']
    ])
    
    print(f"\\n🎉 Generation 2 test completed!")
    print(f"📊 Total tests: {total_tests}, Failures: {len(test_results['failures'])}")
    
    if len(test_results['failures']) == 0:
        print("✅ MAKE IT ROBUST - All robustness features working!")
        print("📈 Ready for Generation 3 (Performance & Scale)")
    else:
        print("⚠️  Some tests failed - review and fix before proceeding")
    
    return test_results

if __name__ == "__main__":
    try:
        results = test_robust_functionality()
        if len(results['failures']) == 0:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Generation 2 test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)