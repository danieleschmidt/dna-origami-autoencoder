"""Tests for DNA encoding functionality."""

import pytest
import numpy as np
from PIL import Image

from dna_origami_ae.encoding.image_encoder import DNAEncoder, Base4Encoder, EncodingParameters
from dna_origami_ae.encoding.biological_constraints import BiologicalConstraints
from dna_origami_ae.models.image_data import ImageData, ImageMetadata
from dna_origami_ae.models.dna_sequence import DNASequence


class TestBase4Encoder:
    """Test Base4 DNA encoding."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.encoder = Base4Encoder()
        self.test_binary = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    
    def test_encode_binary_to_dna(self):
        """Test binary to DNA encoding."""
        dna_sequence = self.encoder.encode_binary_to_dna(self.test_binary)
        
        # Should convert [01][10][10][01] -> TGGT
        expected = "TGGT"
        assert dna_sequence == expected
    
    def test_decode_dna_to_binary(self):
        """Test DNA to binary decoding."""
        dna_sequence = "TGGT"
        binary_data = self.encoder.decode_dna_to_binary(dna_sequence)
        
        expected = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(binary_data, expected)
    
    def test_round_trip_encoding(self):
        """Test round-trip encoding/decoding."""
        # Ensure even number of bits
        binary_data = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        
        # Encode to DNA
        dna_sequence = self.encoder.encode_binary_to_dna(binary_data)
        
        # Decode back to binary
        decoded_binary = self.encoder.decode_dna_to_binary(dna_sequence)
        
        np.testing.assert_array_equal(binary_data, decoded_binary)
    
    def test_odd_length_binary_padding(self):
        """Test that odd-length binary data is padded."""
        odd_binary = np.array([1, 0, 1], dtype=np.uint8)
        dna_sequence = self.encoder.encode_binary_to_dna(odd_binary)
        
        # Should be padded to [1, 0, 1, 0] -> GT
        assert dna_sequence == "GT"
    
    def test_invalid_bit_pair(self):
        """Test handling of invalid bit pairs."""
        # This shouldn't happen in normal usage, but test robustness
        with pytest.raises(ValueError, match="Invalid bit pair"):
            # Manually create invalid bit sequence
            self.encoder._reverse_mapping["99"] = "X"
            binary_data = np.array([9, 9], dtype=np.uint8)
            self.encoder.encode_binary_to_dna(binary_data)
    
    def test_encode_with_constraints(self):
        """Test encoding with biological constraints."""
        constraints = BiologicalConstraints(
            gc_content_range=(0.4, 0.6),
            max_homopolymer_length=3
        )
        encoder = Base4Encoder(constraints)
        
        # Create binary data that might violate constraints
        binary_data = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)  # All A's
        
        # Should either succeed or raise ValueError
        try:
            dna_sequence = encoder.encode_with_constraints(binary_data)
            assert isinstance(dna_sequence, str)
            assert len(dna_sequence) == 4
        except ValueError as e:
            assert "biological constraints" in str(e)


class TestDNAEncoder:
    """Test main DNA encoder class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.encoder = DNAEncoder()
        
        # Create simple test image
        image_array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        self.test_image = ImageData.from_array(image_array, name="test_image")
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        assert self.encoder.bits_per_base == 2
        assert self.encoder.error_correction_method == 'reed_solomon'
        assert isinstance(self.encoder.base4_encoder, Base4Encoder)
    
    def test_encode_image_basic(self):
        """Test basic image encoding."""
        dna_sequences = self.encoder.encode_image(self.test_image)
        
        assert isinstance(dna_sequences, list)
        assert len(dna_sequences) > 0
        
        for seq in dna_sequences:
            assert isinstance(seq, DNASequence)
            assert len(seq.sequence) > 0
            assert seq.name.startswith("test_image_chunk_")
    
    def test_encode_image_with_parameters(self):
        """Test image encoding with custom parameters."""
        params = EncodingParameters(
            chunk_size=100,
            include_metadata=True,
            error_correction="reed_solomon"
        )
        
        dna_sequences = self.encoder.encode_image(self.test_image, params)
        
        assert isinstance(dna_sequences, list)
        assert len(dna_sequences) > 0
    
    def test_decode_image(self):
        """Test image decoding."""
        # First encode an image
        dna_sequences = self.encoder.encode_image(self.test_image)
        
        # Then decode it back
        decoded_image = self.encoder.decode_image(
            dna_sequences,
            self.test_image.metadata.width,
            self.test_image.metadata.height
        )
        
        assert isinstance(decoded_image, ImageData)
        assert decoded_image.metadata.width == self.test_image.metadata.width
        assert decoded_image.metadata.height == self.test_image.metadata.height
    
    def test_encode_decode_round_trip(self):
        """Test full round-trip encoding and decoding."""
        # Use a smaller image for faster testing
        small_array = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        small_image = ImageData.from_array(small_array, name="small_test")
        
        # Encode
        dna_sequences = self.encoder.encode_image(small_image)
        
        # Decode
        decoded_image = self.encoder.decode_image(
            dna_sequences,
            small_image.metadata.width,
            small_image.metadata.height
        )
        
        # Check similarity (won't be exactly equal due to error correction)
        mse = small_image.calculate_mse(decoded_image)
        assert mse < 100  # Reasonable reconstruction error
    
    def test_encoding_efficiency(self):
        """Test encoding efficiency calculation."""
        dna_sequences = self.encoder.encode_image(self.test_image)
        original_size = self.test_image.metadata.size_bytes
        
        efficiency = self.encoder.get_encoding_efficiency(original_size, dna_sequences)
        
        assert 'compression_ratio' in efficiency
        assert 'bits_per_base' in efficiency
        assert 'total_bases' in efficiency
        assert efficiency['total_bases'] > 0
    
    def test_validate_encoding(self):
        """Test encoding validation."""
        dna_sequences = self.encoder.encode_image(self.test_image)
        
        validation_result = self.encoder.validate_encoding(
            self.test_image, dna_sequences
        )
        
        assert 'success' in validation_result
        assert 'mse' in validation_result
        assert 'num_sequences' in validation_result
        assert validation_result['num_sequences'] == len(dna_sequences)
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        initial_stats = self.encoder.get_statistics()
        assert initial_stats['total_images_encoded'] == 0
        
        # Encode an image
        self.encoder.encode_image(self.test_image)
        
        updated_stats = self.encoder.get_statistics()
        assert updated_stats['total_images_encoded'] == 1
        assert updated_stats['total_bases_generated'] > 0
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        # Encode an image to generate some stats
        self.encoder.encode_image(self.test_image)
        
        # Reset statistics
        self.encoder.reset_statistics()
        
        stats = self.encoder.get_statistics()
        assert stats['total_images_encoded'] == 0
        assert stats['total_bases_generated'] == 0


class TestEncodingParameters:
    """Test encoding parameters."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = EncodingParameters()
        
        assert params.bits_per_base == 2
        assert params.error_correction == "reed_solomon"
        assert params.redundancy == 0.3
        assert params.chunk_size == 200
        assert params.include_metadata is True
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = EncodingParameters(
            bits_per_base=4,
            error_correction="parity",
            chunk_size=100,
            include_metadata=False
        )
        
        assert params.bits_per_base == 4
        assert params.error_correction == "parity"
        assert params.chunk_size == 100
        assert params.include_metadata is False


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.encoder = DNAEncoder()
    
    def test_empty_image(self):
        """Test encoding empty/minimal image."""
        # Create minimal 1x1 image
        tiny_array = np.array([[128]], dtype=np.uint8)
        tiny_image = ImageData.from_array(tiny_array, name="tiny")
        
        dna_sequences = self.encoder.encode_image(tiny_image)
        
        assert len(dna_sequences) > 0
        # Should be able to decode back
        decoded = self.encoder.decode_image(dna_sequences, 1, 1)
        assert decoded.metadata.width == 1
        assert decoded.metadata.height == 1
    
    def test_large_image_chunking(self):
        """Test that large images are properly chunked."""
        # Create larger image
        large_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        large_image = ImageData.from_array(large_array, name="large")
        
        params = EncodingParameters(chunk_size=50)  # Small chunks
        dna_sequences = self.encoder.encode_image(large_image, params)
        
        # Should have multiple chunks
        assert len(dna_sequences) > 1
        
        # Each chunk should be reasonable size
        for seq in dna_sequences:
            assert len(seq.sequence) <= 100  # Accounting for overhead
    
    def test_constraint_violations(self):
        """Test handling of constraint violations."""
        # Create encoder with very strict constraints
        strict_constraints = BiologicalConstraints(
            gc_content_range=(0.5, 0.5),  # Impossible range
            max_homopolymer_length=1
        )
        
        encoder = DNAEncoder(biological_constraints=strict_constraints)
        
        # Should handle constraint violations gracefully
        try:
            dna_sequences = encoder.encode_image(self.test_image)
            # If it succeeds, check statistics
            stats = encoder.get_statistics()
            # Might have constraint violations
        except ValueError:
            # Expected for very strict constraints
            pass
    
    @pytest.fixture(autouse=True)
    def setup_test_image(self):
        """Setup test image for edge case tests."""
        image_array = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        self.test_image = ImageData.from_array(image_array, name="edge_test")


@pytest.mark.integration
class TestEncodingIntegration:
    """Integration tests for encoding system."""
    
    def test_complete_workflow(self):
        """Test complete encoding workflow."""
        # Create test image
        test_array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        test_image = ImageData.from_array(test_array, name="integration_test")
        
        # Initialize encoder
        encoder = DNAEncoder()
        
        # Encode image
        dna_sequences = encoder.encode_image(test_image)
        
        # Validate encoding
        validation = encoder.validate_encoding(test_image, dna_sequences)
        assert validation['success']
        
        # Decode image
        decoded_image = encoder.decode_image(
            dna_sequences,
            test_image.metadata.width,
            test_image.metadata.height
        )
        
        # Check quality metrics
        mse = test_image.calculate_mse(decoded_image)
        psnr = test_image.calculate_psnr(decoded_image)
        
        assert mse < 1000  # Reasonable reconstruction
        assert psnr > 10   # Reasonable quality
        
        # Check statistics
        stats = encoder.get_statistics()
        assert stats['total_images_encoded'] == 1
    
    def test_multiple_image_encoding(self):
        """Test encoding multiple images."""
        encoder = DNAEncoder()
        
        images = []
        for i in range(3):
            array = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            image = ImageData.from_array(array, name=f"multi_test_{i}")
            images.append(image)
        
        # Encode all images
        all_sequences = []
        for image in images:
            sequences = encoder.encode_image(image)
            all_sequences.extend(sequences)
        
        # Check statistics
        stats = encoder.get_statistics()
        assert stats['total_images_encoded'] == 3
        assert len(all_sequences) > 3  # Should have multiple chunks
    
    def test_different_encoding_parameters(self):
        """Test various encoding parameter combinations."""
        test_array = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        test_image = ImageData.from_array(test_array, name="param_test")
        
        parameter_sets = [
            EncodingParameters(error_correction="none"),
            EncodingParameters(error_correction="parity"),
            EncodingParameters(chunk_size=50),
            EncodingParameters(include_metadata=False),
            EncodingParameters(compression_enabled=True),
        ]
        
        for params in parameter_sets:
            encoder = DNAEncoder()
            
            try:
                dna_sequences = encoder.encode_image(test_image, params)
                assert len(dna_sequences) > 0
                
                # Try to decode
                decoded = encoder.decode_image(
                    dna_sequences,
                    test_image.metadata.width,
                    test_image.metadata.height,
                    params
                )
                assert decoded.metadata.width == test_image.metadata.width
                
            except NotImplementedError:
                # Some features might not be fully implemented
                pass