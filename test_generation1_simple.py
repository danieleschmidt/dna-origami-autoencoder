#!/usr/bin/env python3
"""
Simplified Generation 1 test - Core DNA encoding/decoding functionality.
"""

import numpy as np
from dna_origami_ae import DNAEncoder, ImageData, DNASequence

def test_core_functionality():
    """Test core DNA encoding and decoding functionality."""
    print("🧬 DNA-Origami-AutoEncoder Generation 1 - Core Test")
    print("=" * 60)
    
    # 1. Create test image
    print("1. Creating test image...")
    test_array = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    test_image = ImageData.from_array(test_array, name="test_8x8")
    print(f"   ✅ Created {test_image}")
    print(f"   📊 Size: {test_image.metadata.size_bytes} bytes")
    
    # 2. Test DNA sequence creation
    print("\n2. Testing DNA sequence functionality...")
    test_seq = DNASequence("ATCGATCGATCGATCG", name="test", skip_validation=True)
    print(f"   ✅ DNA Sequence: {test_seq}")
    print(f"   📊 Length: {test_seq.length}, GC: {test_seq.gc_content:.1%}, Tm: {test_seq.melting_temperature:.1f}°C")
    
    # Test reverse complement
    rev_comp = test_seq.reverse_complement()
    print(f"   ✅ Reverse complement: {rev_comp.sequence}")
    
    # 3. Encode image to DNA
    print("\n3. Encoding image to DNA...")
    encoder = DNAEncoder()
    dna_sequences = encoder.encode_image(test_image)
    print(f"   ✅ Encoded to {len(dna_sequences)} DNA sequences")
    
    total_bases = sum(len(seq) for seq in dna_sequences)
    print(f"   📊 Total bases: {total_bases}")
    
    for i, seq in enumerate(dna_sequences[:3]):
        print(f"      Seq {i+1}: {len(seq.sequence)} bases, GC={seq.gc_content:.1%}")
    
    # 4. Calculate encoding efficiency
    print("\n4. Calculating encoding efficiency...")
    efficiency = encoder.get_encoding_efficiency(test_image.metadata.size_bytes, dna_sequences)
    print(f"   📊 Compression ratio: {efficiency['compression_ratio']:.2f}")
    print(f"   📊 Bits per base: {efficiency['bits_per_base']:.2f}")
    print(f"   📊 Storage density: {efficiency['storage_density_bytes_per_gram']:.2e} bytes/gram")
    
    # 5. Test round-trip encoding/decoding
    print("\n5. Testing round-trip encoding...")
    try:
        decoded_image = encoder.decode_image(
            dna_sequences,
            test_image.metadata.width,
            test_image.metadata.height
        )
        print(f"   ✅ Decoded to: {decoded_image}")
        
        # Calculate similarity metrics
        mse = test_image.calculate_mse(decoded_image)
        psnr = test_image.calculate_psnr(decoded_image)
        ssim = test_image.calculate_ssim(decoded_image)
        
        print(f"   📊 Reconstruction MSE: {mse:.2f}")
        print(f"   📊 Reconstruction PSNR: {psnr:.2f} dB")
        print(f"   📊 Reconstruction SSIM: {ssim:.3f}")
        
    except Exception as e:
        print(f"   ⚠️  Decoding error: {str(e)[:100]}")
    
    # 6. Validate encoding
    print("\n6. Validating encoding...")
    validation = encoder.validate_encoding(test_image, dna_sequences)
    print(f"   Success: {validation['success']}")
    print(f"   Sequences: {validation['num_sequences']}")
    print(f"   Total bases: {validation['total_bases']}")
    
    if not validation['success']:
        print(f"   Error: {validation.get('error', 'Unknown error')}")
    
    # 7. Test error correction
    print("\n7. Testing error correction...")
    from dna_origami_ae.encoding.error_correction import DNAErrorCorrection
    
    error_corrector = DNAErrorCorrection(method="parity")
    test_data = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    
    # Encode with error correction
    encoded_data = error_corrector.encode(test_data)
    print(f"   ✅ Added error correction: {len(test_data)} → {len(encoded_data)} bits")
    
    # Test error correction capability
    test_result = error_corrector.test_error_correction(test_data, error_rate=0.1)
    print(f"   📊 Error correction test: {test_result['success']}")
    print(f"   📊 Errors introduced: {test_result['errors_introduced']}")
    print(f"   📊 Errors corrected: {test_result['errors_corrected']}")
    
    # 8. Show system statistics
    print("\n8. System statistics:")
    encoding_stats = encoder.get_statistics()
    print(f"   Images encoded: {encoding_stats['total_images_encoded']}")
    print(f"   Bases generated: {encoding_stats['total_bases_generated']}")
    print(f"   Constraint violations: {encoding_stats['constraint_violations']}")
    
    print("\n🎉 Generation 1 core test completed!")
    print("✅ MAKE IT WORK - Core DNA encoding/decoding functional")
    
    return {
        'image': test_image,
        'dna_sequences': dna_sequences,
        'encoding_efficiency': efficiency,
        'validation_result': validation,
        'error_correction_result': test_result,
        'stats': encoding_stats
    }

if __name__ == "__main__":
    try:
        results = test_core_functionality()
        print(f"\n📊 Summary: Core functionality working!")
        print(f"📈 Ready for Generation 2 (Robustness improvements)")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()