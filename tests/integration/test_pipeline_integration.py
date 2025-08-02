"""
Integration tests for the complete DNA-Origami-AutoEncoder pipeline.

This module tests the integration between different components of the system,
ensuring that data flows correctly through the entire pipeline from image
input to reconstructed output.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from pathlib import Path

from tests.conftest import (
    assert_arrays_close,
    assert_dna_sequence_valid,
    assert_image_shape,
    skip_if_slow,
)


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline integration."""
    
    def test_image_to_dna_to_origami_pipeline(
        self,
        sample_image,
        mock_encoder,
        mock_origami_designer,
        temp_dir
    ):
        """Test image encoding to origami design pipeline."""
        # Step 1: Encode image to DNA
        dna_sequence = mock_encoder.encode_image(sample_image)
        assert isinstance(dna_sequence, str)
        assert len(dna_sequence) > 0
        assert_dna_sequence_valid(dna_sequence)
        
        # Step 2: Design origami from DNA sequence
        origami_design = mock_origami_designer.design_origami(
            dna_sequence,
            target_shape="rectangle",
            dimensions=(100, 100)
        )
        
        # Verify origami design structure
        assert "scaffold" in origami_design
        assert "staples" in origami_design
        assert "structure" in origami_design
        assert isinstance(origami_design["staples"], list)
        assert len(origami_design["staples"]) > 0
        
        # Verify data consistency
        total_sequence_length = sum(len(staple) for staple in origami_design["staples"])
        assert total_sequence_length > 0
        
        # Test file I/O integration
        output_file = temp_dir / "origami_design.json"
        # Mock file writing
        mock_encoder.save_design = Mock()
        mock_encoder.save_design(origami_design, output_file)
        mock_encoder.save_design.assert_called_once_with(origami_design, output_file)

    def test_origami_to_simulation_pipeline(
        self,
        sample_origami_design,
        mock_simulator,
        test_config
    ):
        """Test origami design to simulation pipeline."""
        # Configure simulation parameters
        sim_params = {
            "temperature": 300,
            "salt_concentration": 0.5,
            "time_steps": test_config.get("simulation_steps", 1000),
            "save_interval": 100,
        }
        
        # Run simulation
        simulation_result = mock_simulator.simulate_folding(
            sample_origami_design,
            **sim_params
        )
        
        # Verify simulation output structure
        assert "final_structure" in simulation_result
        assert "trajectory" in simulation_result
        assert "energy" in simulation_result
        
        # Verify data types and shapes
        final_structure = simulation_result["final_structure"]
        assert isinstance(final_structure, np.ndarray)
        assert final_structure.ndim == 2
        assert final_structure.shape[1] == 3  # x, y, z coordinates
        
        trajectory = simulation_result["trajectory"]
        assert isinstance(trajectory, np.ndarray)
        assert trajectory.ndim == 3  # (time, particles, coordinates)
        
        energy = simulation_result["energy"]
        assert isinstance(energy, np.ndarray)
        assert energy.ndim == 1

    def test_simulation_to_decoding_pipeline(
        self,
        sample_trajectory,
        mock_decoder,
        test_config
    ):
        """Test simulation to neural decoding pipeline."""
        # Extract final structure from trajectory
        final_structure = sample_trajectory["positions"][-1]
        
        # Decode structure to image
        reconstructed_image = mock_decoder.decode_structure(final_structure)
        
        # Verify output
        assert isinstance(reconstructed_image, np.ndarray)
        assert reconstructed_image.dtype == np.uint8
        assert reconstructed_image.ndim == 2
        
        expected_shape = test_config["image_size"]
        assert_image_shape(reconstructed_image, expected_shape)
        
        # Verify pixel value range
        assert 0 <= reconstructed_image.min() <= reconstructed_image.max() <= 255

    def test_complete_pipeline_integration(
        self,
        sample_image,
        mock_encoder,
        mock_origami_designer,
        mock_simulator,
        mock_decoder,
        performance_tracker
    ):
        """Test complete pipeline from image input to reconstruction."""
        performance_tracker.start()
        
        # Step 1: Image to DNA encoding
        dna_sequence = mock_encoder.encode_image(sample_image)
        assert_dna_sequence_valid(dna_sequence)
        
        # Step 2: DNA to origami design
        origami_design = mock_origami_designer.design_origami(dna_sequence)
        assert "staples" in origami_design
        
        # Step 3: Origami simulation
        simulation_result = mock_simulator.simulate_folding(origami_design)
        final_structure = simulation_result["final_structure"]
        
        # Step 4: Structure decoding
        reconstructed_image = mock_decoder.decode_structure(final_structure)
        
        performance_tracker.stop()
        
        # Verify end-to-end consistency
        assert sample_image.shape == reconstructed_image.shape
        assert sample_image.dtype == reconstructed_image.dtype
        
        # Performance verification
        assert performance_tracker.duration is not None
        assert performance_tracker.duration < 10.0  # Should complete within 10 seconds for mocks


@pytest.mark.integration
class TestComponentInterfaces:
    """Test interfaces between major components."""
    
    def test_encoder_designer_interface(
        self,
        sample_image,
        mock_encoder,
        mock_origami_designer
    ):
        """Test interface between encoder and origami designer."""
        # Encode image
        dna_sequence = mock_encoder.encode_image(sample_image)
        
        # Test that designer accepts encoder output
        try:
            origami_design = mock_origami_designer.design_origami(dna_sequence)
            interface_compatible = True
        except Exception:
            interface_compatible = False
        
        assert interface_compatible, "Encoder output not compatible with designer input"
        assert "staples" in origami_design

    def test_designer_simulator_interface(
        self,
        sample_origami_design,
        mock_simulator
    ):
        """Test interface between origami designer and simulator."""
        # Test that simulator accepts designer output
        try:
            simulation_result = mock_simulator.simulate_folding(sample_origami_design)
            interface_compatible = True
        except Exception:
            interface_compatible = False
        
        assert interface_compatible, "Designer output not compatible with simulator input"
        assert "final_structure" in simulation_result

    def test_simulator_decoder_interface(
        self,
        sample_trajectory,
        mock_decoder
    ):
        """Test interface between simulator and decoder."""
        final_structure = sample_trajectory["positions"][-1]
        
        # Test that decoder accepts simulator output
        try:
            reconstructed = mock_decoder.decode_structure(final_structure)
            interface_compatible = True
        except Exception:
            interface_compatible = False
        
        assert interface_compatible, "Simulator output not compatible with decoder input"
        assert isinstance(reconstructed, np.ndarray)


@pytest.mark.integration
class TestDataFlowIntegration:
    """Test data flow and transformations between components."""
    
    def test_data_shape_consistency(
        self,
        sample_image,
        mock_encoder,
        mock_origami_designer,
        mock_simulator,
        mock_decoder
    ):
        """Test that data shapes remain consistent through pipeline."""
        original_shape = sample_image.shape
        
        # Run through pipeline
        dna_sequence = mock_encoder.encode_image(sample_image)
        origami_design = mock_origami_designer.design_origami(dna_sequence)
        simulation_result = mock_simulator.simulate_folding(origami_design)
        final_structure = simulation_result["final_structure"]
        reconstructed = mock_decoder.decode_structure(final_structure)
        
        # Verify shape consistency
        assert reconstructed.shape == original_shape, "Shape not preserved through pipeline"

    def test_data_type_consistency(
        self,
        sample_image,
        mock_encoder,
        mock_decoder
    ):
        """Test that data types are handled consistently."""
        # Verify input types
        assert isinstance(sample_image, np.ndarray)
        assert sample_image.dtype == np.uint8
        
        # Mock pipeline preserves data types
        dna_sequence = mock_encoder.encode_image(sample_image)
        assert isinstance(dna_sequence, str)
        
        # Mock final output maintains correct type
        mock_structure = np.random.randn(1000, 3)
        reconstructed = mock_decoder.decode_structure(mock_structure)
        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.dtype == np.uint8

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_batch_processing_integration(
        self,
        batch_size,
        test_config,
        mock_encoder,
        mock_decoder
    ):
        """Test batch processing through pipeline components."""
        # Generate batch of images
        height, width = test_config["image_size"]
        image_batch = np.random.randint(
            0, 256, (batch_size, height, width), dtype=np.uint8
        )
        
        # Mock batch encoding
        mock_encoder.encode_batch = Mock()
        dna_sequences = ["ATCG" * 100] * batch_size
        mock_encoder.encode_batch.return_value = dna_sequences
        
        sequences = mock_encoder.encode_batch(image_batch)
        assert len(sequences) == batch_size
        
        # Mock batch decoding
        mock_decoder.decode_batch = Mock()
        reconstructed_batch = np.random.randint(
            0, 256, (batch_size, height, width), dtype=np.uint8
        )
        mock_decoder.decode_batch.return_value = reconstructed_batch
        
        mock_structures = [np.random.randn(1000, 3) for _ in range(batch_size)]
        reconstructed = mock_decoder.decode_batch(mock_structures)
        
        assert reconstructed.shape == image_batch.shape


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleIntegration:
    """Test integration with larger datasets and longer processing."""
    
    @skip_if_slow
    def test_multiple_image_pipeline(
        self,
        test_config,
        mock_encoder,
        mock_origami_designer,
        mock_simulator,
        mock_decoder
    ):
        """Test pipeline with multiple images."""
        num_images = 10
        height, width = test_config["image_size"]
        
        # Generate multiple test images
        images = [
            np.random.randint(0, 256, (height, width), dtype=np.uint8)
            for _ in range(num_images)
        ]
        
        results = []
        for image in images:
            # Process each image through pipeline
            dna_sequence = mock_encoder.encode_image(image)
            origami_design = mock_origami_designer.design_origami(dna_sequence)
            simulation_result = mock_simulator.simulate_folding(origami_design)
            final_structure = simulation_result["final_structure"]
            reconstructed = mock_decoder.decode_structure(final_structure)
            
            results.append({
                "original": image,
                "reconstructed": reconstructed,
                "dna_length": len(dna_sequence)
            })
        
        # Verify all images processed successfully
        assert len(results) == num_images
        for result in results:
            assert result["original"].shape == result["reconstructed"].shape
            assert result["dna_length"] > 0

    @skip_if_slow
    def test_memory_usage_integration(
        self,
        performance_tracker,
        sample_image,
        mock_encoder,
        mock_decoder
    ):
        """Test memory usage during integration pipeline."""
        performance_tracker.start()
        
        # Process multiple times to test memory accumulation
        for _ in range(5):
            dna_sequence = mock_encoder.encode_image(sample_image)
            mock_structure = np.random.randn(5000, 3)  # Larger structure
            reconstructed = mock_decoder.decode_structure(mock_structure)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        performance_tracker.stop()
        
        # Memory usage should be reasonable
        memory_delta = performance_tracker.memory_delta
        assert memory_delta is not None
        # Allow up to 100MB memory increase for integration tests
        assert memory_delta < 100, f"Memory usage too high: {memory_delta:.2f} MB"


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling across component interfaces."""
    
    def test_invalid_input_propagation(
        self,
        mock_encoder,
        mock_origami_designer
    ):
        """Test that invalid inputs are handled gracefully."""
        # Test with invalid image (wrong shape)
        invalid_image = np.array([[1, 2], [3, 4]])  # Too small
        
        # Mock encoder should handle gracefully
        mock_encoder.encode_image.side_effect = ValueError("Invalid image dimensions")
        
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            mock_encoder.encode_image(invalid_image)

    def test_component_failure_handling(
        self,
        sample_image,
        mock_encoder,
        mock_origami_designer
    ):
        """Test handling of component failures in pipeline."""
        # Set up encoder to succeed
        dna_sequence = "ATCG" * 100
        mock_encoder.encode_image.return_value = dna_sequence
        
        # Set up designer to fail
        mock_origami_designer.design_origami.side_effect = RuntimeError("Simulation failed")
        
        # Test that failure is propagated correctly
        encoded = mock_encoder.encode_image(sample_image)
        assert encoded == dna_sequence
        
        with pytest.raises(RuntimeError, match="Simulation failed"):
            mock_origami_designer.design_origami(encoded)

    def test_partial_pipeline_recovery(
        self,
        sample_image,
        mock_encoder,
        mock_origami_designer,
        temp_dir
    ):
        """Test recovery from partial pipeline execution."""
        # Run first part successfully
        dna_sequence = mock_encoder.encode_image(sample_image)
        
        # Save intermediate result
        intermediate_file = temp_dir / "intermediate.txt"
        intermediate_file.write_text(dna_sequence)
        
        # Test loading intermediate result
        loaded_sequence = intermediate_file.read_text()
        assert loaded_sequence == dna_sequence
        
        # Continue pipeline from intermediate result
        origami_design = mock_origami_designer.design_origami(loaded_sequence)
        assert "staples" in origami_design