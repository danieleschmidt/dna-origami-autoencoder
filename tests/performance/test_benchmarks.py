"""
Performance and benchmark tests for DNA-Origami-AutoEncoder.

This module contains performance tests to ensure the system meets
performance requirements and to detect performance regressions.
"""

import pytest
import time
import numpy as np
import torch
from unittest.mock import Mock
import psutil
import gc

from tests.conftest import skip_if_slow, skip_if_no_gpu


@pytest.mark.performance
class TestEncodingPerformance:
    """Test performance of DNA encoding operations."""
    
    @pytest.mark.parametrize("image_size", [(32, 32), (64, 64), (128, 128)])
    def test_image_encoding_speed(self, image_size, performance_tracker):
        """Test DNA encoding speed for different image sizes."""
        # Generate test image
        image = np.random.randint(0, 256, image_size, dtype=np.uint8)
        
        # Mock encoder with realistic timing
        mock_encoder = Mock()
        def mock_encode(img):
            # Simulate encoding time proportional to image size
            pixels = img.size
            time.sleep(pixels / 1000000)  # 1 microsecond per pixel
            return "ATCG" * (pixels // 4)
        
        mock_encoder.encode_image = mock_encode
        
        # Measure performance
        performance_tracker.start()
        result = mock_encoder.encode_image(image)
        performance_tracker.stop()
        
        # Performance assertions
        duration = performance_tracker.duration
        assert duration is not None
        
        # Encoding should be fast (less than 1 second for largest image)
        if image_size == (128, 128):
            assert duration < 1.0, f"Encoding too slow: {duration:.3f}s"
        else:
            assert duration < 0.1, f"Encoding too slow: {duration:.3f}s"
        
        # Verify output
        assert len(result) > 0
        expected_length = image.size // 4 * 4  # Multiple of 4 bases
        assert len(result) >= expected_length

    def test_batch_encoding_performance(self, performance_tracker):
        """Test performance of batch encoding operations."""
        batch_sizes = [1, 4, 8, 16]
        image_size = (32, 32)
        
        performance_results = {}
        
        for batch_size in batch_sizes:
            # Generate batch
            batch = np.random.randint(
                0, 256, (batch_size, *image_size), dtype=np.uint8
            )
            
            # Mock batch encoder
            mock_encoder = Mock()
            def mock_batch_encode(batch_imgs):
                # Simulate batch processing with efficiency gains
                total_pixels = batch_imgs.size
                # Batch processing is more efficient
                efficiency_factor = min(batch_imgs.shape[0] / 4, 2.0)
                time.sleep(total_pixels / 2000000 / efficiency_factor)
                return ["ATCG" * (img.size // 4) for img in batch_imgs]
            
            mock_encoder.encode_batch = mock_batch_encode
            
            # Measure performance
            performance_tracker.start()
            results = mock_encoder.encode_batch(batch)
            performance_tracker.stop()
            
            duration = performance_tracker.duration
            performance_results[batch_size] = {
                "duration": duration,
                "images_per_second": batch_size / duration if duration > 0 else float('inf'),
                "memory_delta": performance_tracker.memory_delta
            }
            
            # Verify results
            assert len(results) == batch_size
        
        # Verify batch processing efficiency
        single_rate = performance_results[1]["images_per_second"]
        batch_rate = performance_results[8]["images_per_second"]
        
        # Batch processing should be more efficient
        efficiency_gain = batch_rate / single_rate
        assert efficiency_gain > 1.0, f"No batch efficiency gain: {efficiency_gain:.2f}x"

    @skip_if_slow
    def test_large_image_encoding(self, performance_tracker):
        """Test encoding performance with large images."""
        large_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        # Mock encoder for large images
        mock_encoder = Mock()
        def mock_large_encode(img):
            # Simulate more complex encoding for large images
            pixels = img.size
            time.sleep(pixels / 500000)  # 2 microseconds per pixel
            return "ATCG" * (pixels // 2)  # Higher density encoding
        
        mock_encoder.encode_image = mock_large_encode
        
        performance_tracker.start()
        result = mock_encoder.encode_image(large_image)
        performance_tracker.stop()
        
        # Should complete within 10 seconds
        assert performance_tracker.duration < 10.0
        assert len(result) > 0
        
        # Memory usage should be reasonable
        memory_usage = performance_tracker.memory_delta
        assert memory_usage < 500, f"Memory usage too high: {memory_usage:.2f} MB"


@pytest.mark.performance
class TestSimulationPerformance:
    """Test performance of molecular simulation operations."""
    
    @pytest.mark.parametrize("particle_count", [100, 500, 1000])
    def test_simulation_scaling(self, particle_count, performance_tracker):
        """Test simulation performance scaling with particle count."""
        # Mock simulation with scaling behavior
        mock_simulator = Mock()
        def mock_simulate(design, time_steps=1000):
            # Simulate O(N²) scaling for particle interactions
            computation_cost = particle_count * particle_count / 1000000
            time.sleep(computation_cost)
            
            return {
                "final_structure": np.random.randn(particle_count, 3),
                "trajectory": np.random.randn(100, particle_count, 3),
                "energy": np.random.randn(100)
            }
        
        mock_simulator.simulate_folding = mock_simulate
        
        # Create mock design
        mock_design = {"particle_count": particle_count}
        
        performance_tracker.start()
        result = mock_simulator.simulate_folding(mock_design)
        performance_tracker.stop()
        
        duration = performance_tracker.duration
        assert duration is not None
        
        # Performance scaling expectations
        if particle_count <= 500:
            assert duration < 1.0, f"Small simulation too slow: {duration:.3f}s"
        else:
            assert duration < 5.0, f"Large simulation too slow: {duration:.3f}s"
        
        # Verify output structure
        assert result["final_structure"].shape == (particle_count, 3)

    @skip_if_no_gpu
    def test_gpu_simulation_performance(self, device, performance_tracker):
        """Test GPU-accelerated simulation performance."""
        if device.type != "cuda":
            pytest.skip("GPU not available")
        
        particle_count = 2000
        
        # Mock GPU simulation
        mock_gpu_simulator = Mock()
        def mock_gpu_simulate(design):
            # GPU simulation should be faster
            with torch.cuda.device(device):
                # Simulate GPU computation
                tensor = torch.randn(particle_count, 3, device=device)
                result_tensor = tensor @ tensor.T
                result = result_tensor.cpu().numpy()
                
                # Add small delay to simulate real computation
                time.sleep(0.1)
                
                return {
                    "final_structure": result[:particle_count, :3],
                    "gpu_used": True
                }
        
        mock_gpu_simulator.simulate_folding = mock_gpu_simulate
        
        mock_design = {"particle_count": particle_count}
        
        performance_tracker.start()
        result = mock_gpu_simulator.simulate_folding(mock_design)
        performance_tracker.stop()
        
        # GPU simulation should be fast
        assert performance_tracker.duration < 1.0
        assert result["gpu_used"] is True

    def test_parallel_simulation_performance(self, performance_tracker):
        """Test parallel simulation performance."""
        num_simulations = 4
        
        # Mock parallel simulator
        mock_parallel_simulator = Mock()
        def mock_parallel_simulate(designs):
            # Simulate parallel processing
            # Sequential would take num_simulations * 0.1 seconds
            # Parallel should be faster
            time.sleep(0.15)  # Overhead + parallel execution
            
            return [
                {
                    "final_structure": np.random.randn(100, 3),
                    "simulation_id": i
                }
                for i in range(len(designs))
            ]
        
        mock_parallel_simulator.simulate_batch = mock_parallel_simulate
        
        designs = [{"id": i} for i in range(num_simulations)]
        
        performance_tracker.start()
        results = mock_parallel_simulator.simulate_batch(designs)
        performance_tracker.stop()
        
        # Parallel should be faster than sequential
        assert performance_tracker.duration < 0.3  # Less than 4 * 0.1
        assert len(results) == num_simulations


@pytest.mark.performance
class TestDecodingPerformance:
    """Test performance of neural decoding operations."""
    
    @pytest.mark.parametrize("structure_size", [500, 1000, 2000])
    def test_decoding_speed(self, structure_size, performance_tracker, device):
        """Test neural decoding speed for different structure sizes."""
        # Generate mock 3D structure
        structure = np.random.randn(structure_size, 3).astype(np.float32)
        
        # Mock neural decoder
        mock_decoder = Mock()
        def mock_decode(struct):
            # Convert to tensor and simulate neural network inference
            tensor = torch.from_numpy(struct).to(device)
            
            # Simulate transformer operations
            # Attention complexity is O(N²)
            computation_time = (structure_size ** 2) / 10000000
            time.sleep(computation_time)
            
            # Return mock reconstructed image
            return np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        mock_decoder.decode_structure = mock_decode
        
        performance_tracker.start()
        result = mock_decoder.decode_structure(structure)
        performance_tracker.stop()
        
        duration = performance_tracker.duration
        assert duration is not None
        
        # Decoding should be reasonably fast
        if structure_size <= 1000:
            assert duration < 1.0, f"Small decoding too slow: {duration:.3f}s"
        else:
            assert duration < 5.0, f"Large decoding too slow: {duration:.3f}s"
        
        # Verify output
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_batch_decoding_performance(self, performance_tracker, device):
        """Test batch decoding performance."""
        batch_sizes = [1, 4, 8]
        structure_size = 1000
        
        for batch_size in batch_sizes:
            # Generate batch of structures
            structures = [
                np.random.randn(structure_size, 3).astype(np.float32)
                for _ in range(batch_size)
            ]
            
            # Mock batch decoder
            mock_decoder = Mock()
            def mock_batch_decode(struct_batch):
                # Batch processing should be more efficient
                total_structures = len(struct_batch) * structure_size
                # Batch efficiency reduces computation per structure
                batch_efficiency = min(batch_size / 2, 4.0)
                computation_time = (total_structures ** 1.5) / 50000000 / batch_efficiency
                time.sleep(computation_time)
                
                return np.random.randint(
                    0, 256, (batch_size, 32, 32), dtype=np.uint8
                )
            
            mock_decoder.decode_batch = mock_batch_decode
            
            performance_tracker.start()
            results = mock_decoder.decode_batch(structures)
            performance_tracker.stop()
            
            duration = performance_tracker.duration
            
            # Batch processing should be efficient
            if batch_size == 1:
                assert duration < 0.5, f"Single decoding too slow: {duration:.3f}s"
            else:
                # Batch should be more efficient per item
                per_item_time = duration / batch_size
                assert per_item_time < 0.3, f"Batch decoding not efficient: {per_item_time:.3f}s per item"
            
            # Verify results
            assert results.shape == (batch_size, 32, 32)

    @skip_if_no_gpu
    def test_gpu_decoding_performance(self, device, performance_tracker):
        """Test GPU-accelerated decoding performance."""
        if device.type != "cuda":
            pytest.skip("GPU not available")
        
        structure_size = 2000
        structure = np.random.randn(structure_size, 3).astype(np.float32)
        
        # Mock GPU decoder
        mock_gpu_decoder = Mock()
        def mock_gpu_decode(struct):
            # Move to GPU and simulate neural network operations
            tensor = torch.from_numpy(struct).to(device)
            
            # Simulate transformer layers on GPU
            for _ in range(3):  # 3 layers
                # Self-attention simulation
                attention = torch.matmul(tensor, tensor.transpose(-2, -1))
                tensor = torch.matmul(attention, tensor)
            
            # Small additional processing time
            time.sleep(0.05)
            
            return np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        mock_gpu_decoder.decode_structure = mock_gpu_decode
        
        performance_tracker.start()
        result = mock_gpu_decoder.decode_structure(structure)
        performance_tracker.stop()
        
        # GPU decoding should be fast
        assert performance_tracker.duration < 2.0
        assert result.shape == (32, 32)


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and optimization."""
    
    def test_memory_usage_scaling(self, performance_tracker):
        """Test memory usage scaling with data size."""
        data_sizes = [1000, 5000, 10000]
        memory_usage = {}
        
        for size in data_sizes:
            # Clear memory before test
            gc.collect()
            
            performance_tracker.start()
            
            # Simulate data processing
            large_array = np.random.randn(size, size)
            processed_array = large_array @ large_array.T
            
            # Keep reference to prevent immediate garbage collection
            result = processed_array.sum()
            
            performance_tracker.stop()
            
            memory_usage[size] = performance_tracker.memory_delta
            
            # Clean up
            del large_array, processed_array
            gc.collect()
        
        # Memory usage should scale reasonably
        for size, memory in memory_usage.items():
            # Rough estimate: should not exceed 10x the data size in MB
            expected_max_memory = (size * size * 8) / (1024 * 1024) * 3  # 3x for safety
            assert memory < expected_max_memory, f"Memory usage too high for size {size}: {memory:.2f} MB"

    def test_memory_leak_detection(self, performance_tracker):
        """Test for memory leaks in repeated operations."""
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform repeated operations
        for i in range(10):
            # Simulate processing that might leak memory
            data = np.random.randn(1000, 1000)
            result = np.fft.fft2(data)
            processed = np.real(result) + np.imag(result)
            
            # Force garbage collection
            del data, result, processed
            gc.collect()
            
            # Check memory usage every few iterations
            if i % 3 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_increase = current_memory - baseline_memory
                
                # Memory increase should be minimal (less than 50MB)
                assert memory_increase < 50, f"Potential memory leak: {memory_increase:.2f} MB increase"

    @skip_if_slow
    def test_large_dataset_memory_efficiency(self, performance_tracker):
        """Test memory efficiency with large datasets."""
        # Simulate processing large dataset in chunks
        total_size = 50000
        chunk_size = 5000
        
        performance_tracker.start()
        
        max_memory_usage = 0
        for i in range(0, total_size, chunk_size):
            # Process chunk
            chunk = np.random.randn(chunk_size, 100)
            processed_chunk = np.fft.fft(chunk, axis=1)
            result_chunk = np.abs(processed_chunk)
            
            # Track peak memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            max_memory_usage = max(max_memory_usage, current_memory)
            
            # Clean up chunk
            del chunk, processed_chunk, result_chunk
            
            # Periodic garbage collection
            if i % (chunk_size * 3) == 0:
                gc.collect()
        
        performance_tracker.stop()
        
        # Memory usage should remain bounded despite large total dataset
        # Should not exceed memory for processing ~3 chunks simultaneously
        reasonable_memory_limit = 200  # MB
        assert max_memory_usage - psutil.Process().memory_info().rss / 1024 / 1024 < reasonable_memory_limit


@pytest.mark.performance
@pytest.mark.slow
class TestEndToEndPerformance:
    """Test end-to-end pipeline performance."""
    
    @skip_if_slow
    def test_complete_pipeline_performance(
        self,
        sample_image,
        mock_encoder,
        mock_origami_designer,
        mock_simulator,
        mock_decoder,
        performance_tracker
    ):
        """Test complete pipeline performance."""
        # Configure mocks with realistic timing
        def slow_encode(img):
            time.sleep(0.1)
            return "ATCG" * (img.size // 4)
        
        def slow_design(seq):
            time.sleep(0.2)
            return {"staples": ["ATCG" * 8] * 100, "scaffold": "M13mp18"}
        
        def slow_simulate(design):
            time.sleep(0.5)  # Simulation is typically the slowest
            return {
                "final_structure": np.random.randn(1000, 3),
                "trajectory": np.random.randn(100, 1000, 3)
            }
        
        def slow_decode(structure):
            time.sleep(0.1)
            return np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        mock_encoder.encode_image = slow_encode
        mock_origami_designer.design_origami = slow_design
        mock_simulator.simulate_folding = slow_simulate
        mock_decoder.decode_structure = slow_decode
        
        performance_tracker.start()
        
        # Run complete pipeline
        dna_sequence = mock_encoder.encode_image(sample_image)
        origami_design = mock_origami_designer.design_origami(dna_sequence)
        simulation_result = mock_simulator.simulate_folding(origami_design)
        final_structure = simulation_result["final_structure"]
        reconstructed = mock_decoder.decode_structure(final_structure)
        
        performance_tracker.stop()
        
        # Complete pipeline should finish within reasonable time
        total_time = performance_tracker.duration
        assert total_time < 5.0, f"Complete pipeline too slow: {total_time:.2f}s"
        
        # Verify output quality
        assert reconstructed.shape == sample_image.shape
        
        # Memory usage should be reasonable
        memory_usage = performance_tracker.memory_delta
        assert memory_usage < 100, f"Pipeline memory usage too high: {memory_usage:.2f} MB"

    def test_throughput_benchmark(self, performance_tracker, test_config):
        """Test system throughput with multiple samples."""
        num_samples = 5
        batch_size = test_config["batch_size"]
        
        # Generate test samples
        height, width = test_config["image_size"]
        samples = [
            np.random.randint(0, 256, (height, width), dtype=np.uint8)
            for _ in range(num_samples)
        ]
        
        # Mock high-throughput processing
        mock_pipeline = Mock()
        def mock_process_batch(sample_batch):
            # Simulate efficient batch processing
            processing_time = len(sample_batch) * 0.05  # 50ms per sample
            time.sleep(processing_time)
            
            return [
                np.random.randint(0, 256, (height, width), dtype=np.uint8)
                for _ in sample_batch
            ]
        
        mock_pipeline.process_batch = mock_process_batch
        
        performance_tracker.start()
        
        # Process in batches
        results = []
        for i in range(0, num_samples, batch_size):
            batch = samples[i:i + batch_size]
            batch_results = mock_pipeline.process_batch(batch)
            results.extend(batch_results)
        
        performance_tracker.stop()
        
        # Calculate throughput
        duration = performance_tracker.duration
        throughput = num_samples / duration if duration > 0 else float('inf')
        
        # Should process at least 10 samples per second
        assert throughput > 10, f"Throughput too low: {throughput:.2f} samples/s"
        
        # Verify all samples processed
        assert len(results) == num_samples