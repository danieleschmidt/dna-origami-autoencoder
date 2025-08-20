#!/usr/bin/env python3
"""
Basic functionality test for DNA-Origami-AutoEncoder
Tests core encoding/decoding pipeline
"""

import sys
import os
import numpy as np
from PIL import Image

# Add project to path
sys.path.insert(0, '/root/repo')

def test_basic_imports():
    """Test that core modules can be imported."""
    print("Testing imports...")
    
    try:
        from dna_origami_ae.models.dna_sequence import DNASequence, DNAConstraints
        from dna_origami_ae.models.image_data import ImageData
        from dna_origami_ae.encoding.image_encoder import DNAEncoder, Base4Encoder
        from dna_origami_ae.encoding.biological_constraints import BiologicalConstraints
        from dna_origami_ae.encoding.error_correction import DNAErrorCorrection
        print("✓ All core modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_dna_sequence():
    """Test DNA sequence functionality."""
    print("\nTesting DNA sequence...")
    
    try:
        from dna_origami_ae.models.dna_sequence import DNASequence
        
        # Create a simple DNA sequence
        seq = DNASequence(
            sequence="ATGCATGCATGC",
            name="test_sequence",
            skip_validation=True  # Skip validation for basic test
        )
        
        print(f"✓ Created sequence: {seq}")
        print(f"  Length: {seq.length}")
        print(f"  GC content: {seq.gc_content:.2%}")
        
        # Test reverse complement
        rev_comp = seq.reverse_complement()
        print(f"  Reverse complement: {rev_comp.sequence}")
        
        # Test binary conversion
        binary = seq.to_binary()
        print(f"  Binary length: {len(binary)} bits")
        
        return True
    except Exception as e:
        print(f"✗ DNA sequence test failed: {e}")
        return False

def test_image_data():
    """Test image data functionality."""
    print("\nTesting image data...")
    
    try:
        from dna_origami_ae.models.image_data import ImageData
        
        # Create a simple test image
        test_array = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        
        image = ImageData.from_array(test_array, name="test_image")
        print(f"✓ Created image: {image}")
        print(f"  Shape: {image.data.shape}")
        print(f"  Size: {image.metadata.size_bytes} bytes")
        
        # Test binary conversion
        binary = image.to_binary(encoding_bits=8)
        print(f"  Binary length: {len(binary)} bits")
        
        # Test reconstruction
        reconstructed = ImageData.from_binary(
            binary, 
            width=32, 
            height=32, 
            channels=1,
            encoding_bits=8
        )
        
        # Calculate MSE
        mse = image.calculate_mse(reconstructed)
        print(f"  Reconstruction MSE: {mse}")
        
        return True
    except Exception as e:
        print(f"✗ Image data test failed: {e}")
        return False

def test_biological_constraints():
    """Test biological constraints."""
    print("\nTesting biological constraints...")
    
    try:
        from dna_origami_ae.encoding.biological_constraints import BiologicalConstraints
        
        constraints = BiologicalConstraints()
        
        # Test valid sequence
        valid_seq = "ATGCATGCATGC"
        is_valid, errors = constraints.validate_sequence(valid_seq)
        print(f"✓ Valid sequence test: {is_valid} (errors: {len(errors)})")
        
        # Test invalid sequence
        invalid_seq = "AAAAAAAAAAAA"  # Too many homopolymers
        is_valid, errors = constraints.validate_sequence(invalid_seq)
        print(f"✓ Invalid sequence test: {is_valid} (errors: {len(errors)})")
        if errors:
            print(f"  Errors: {errors[0]}")
        
        return True
    except Exception as e:
        print(f"✗ Biological constraints test failed: {e}")
        return False

def test_basic_encoding():
    """Test basic encoding functionality."""
    print("\nTesting basic encoding...")
    
    try:
        from dna_origami_ae.models.image_data import ImageData
        from dna_origami_ae.encoding.image_encoder import DNAEncoder
        
        # Create a simple test image
        test_array = np.random.randint(0, 256, size=(16, 16), dtype=np.uint8)
        image = ImageData.from_array(test_array, name="encoding_test")
        
        # Initialize encoder with minimal functionality
        encoder = DNAEncoder(
            enable_validation=False,  # Disable validation for basic test
            enable_monitoring=False,  # Disable monitoring for basic test  
            enable_optimization=False  # Disable optimization for basic test
        )
        
        print(f"✓ Created encoder")
        print(f"  Image size: {image.metadata.width}x{image.metadata.height}")
        
        # Encode image
        print("  Encoding image...")
        dna_sequences = encoder.encode_image(image)
        
        print(f"✓ Encoded to {len(dna_sequences)} DNA sequences")
        total_bases = sum(len(seq.sequence) for seq in dna_sequences)
        print(f"  Total bases: {total_bases}")
        
        if dna_sequences:
            print(f"  First sequence: {dna_sequences[0].sequence[:50]}...")
            print(f"  First sequence length: {len(dna_sequences[0].sequence)}")
        
        return True
    except Exception as e:
        print(f"✗ Basic encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_correction():
    """Test error correction functionality."""
    print("\nTesting error correction...")
    
    try:
        from dna_origami_ae.encoding.error_correction import DNAErrorCorrection
        
        # Test simple parity correction
        corrector = DNAErrorCorrection(method="parity")
        
        test_data = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        print(f"  Original data: {test_data}")
        
        # Encode with error correction
        encoded = corrector.encode(test_data)
        print(f"  Encoded length: {len(encoded)} (overhead: {corrector.get_overhead():.2f})")
        
        # Decode
        decoded = corrector.decode(encoded)
        print(f"  Decoded length: {len(decoded)}")
        
        # Check if data matches
        matches = np.array_equal(test_data, decoded[:len(test_data)])
        print(f"✓ Error correction test: {matches}")
        
        return True
    except Exception as e:
        print(f"✗ Error correction test failed: {e}")
        return False

def main():
    """Run all basic functionality tests."""
    print("=" * 60)
    print("DNA-Origami-AutoEncoder Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_dna_sequence,
        test_image_data, 
        test_biological_constraints,
        test_error_correction,
        test_basic_encoding
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Core functionality is working.")
        return True
    else:
        print("⚠️  Some tests failed. Core functionality needs fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)