#!/usr/bin/env python3
"""
Generation 1 Basic Functionality Test - MAKE IT WORK
Tests basic models and core functionality without complex dependencies.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_generation1_basic():
    """Test Generation 1 basic functionality."""
    print("🚀 Testing Generation 1: MAKE IT WORK")
    
    # Test basic imports
    from dna_origami_ae import ImageData
    from dna_origami_ae.models.image_data import ImageMetadata
    from dna_origami_ae.models.dna_sequence import DNASequence
    from dna_origami_ae.models.origami_structure import OrigamiStructure
    print("✅ Basic model imports successful")
    
    # Test ImageData creation
    test_data = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    metadata = ImageMetadata(width=28, height=28, channels=1, format="grayscale")
    image = ImageData(data=test_data, metadata=metadata, name="mnist_sample")
    
    assert image.metadata.total_pixels == 784
    assert image.metadata.aspect_ratio == 1.0
    assert image.data.shape == (28, 28)
    print("✅ ImageData model working")
    
    # Test DNASequence creation
    dna_seq = DNASequence(sequence="ATCGATCGATCG", name="test_sequence")
    assert len(dna_seq) == 12
    assert dna_seq.gc_content > 0
    print("✅ DNASequence model working")
    
    # Test OrigamiStructure creation with valid DNA sequences
    scaffold = DNASequence(sequence="ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG", name="scaffold")
    staples = [DNASequence(sequence="ATCGATCGATCGATCGATCG", name=f"staple_{i}") for i in range(3)]
    origami = OrigamiStructure(
        scaffold=scaffold, 
        staples=staples, 
        name="test_origami",
        dimensions=(50, 50)
    )
    assert origami.staple_count == 3
    print("✅ OrigamiStructure model working")
    
    print("🎉 Generation 1 basic functionality: SUCCESS")
    return True

def test_basic_encoding_concept():
    """Test basic DNA encoding concept."""
    print("\n🧬 Testing basic encoding concept")
    
    # Simple 2-bit encoding: 00=A, 01=T, 10=G, 11=C
    def encode_byte_to_dna(byte_val):
        """Convert single byte to DNA sequence using 2-bit encoding."""
        bases = ['A', 'T', 'G', 'C']
        dna = ""
        for i in range(4):  # 8 bits / 2 bits per base = 4 bases
            base_idx = (byte_val >> (6 - i*2)) & 0b11
            dna += bases[base_idx]
        return dna
    
    # Test encoding
    test_byte = 0b11100100  # 228 in decimal
    dna_result = encode_byte_to_dna(test_byte)
    expected = "CGTA"  # 11 10 01 00 = C G T A (corrected)
    
    assert dna_result == expected, f"Expected {expected}, got {dna_result}"
    print(f"✅ Basic encoding: byte {test_byte} -> DNA {dna_result}")
    
    return True

if __name__ == "__main__":
    try:
        test_generation1_basic()
        test_basic_encoding_concept()
        print("\n🌟 All Generation 1 tests PASSED")
        exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)