"""
Test configuration and utilities for DNA-Origami-AutoEncoder.

This module provides centralized configuration, fixtures, and utilities
for all tests in the DNA-Origami-AutoEncoder project.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import pytest
import numpy as np
import torch
from dataclasses import dataclass

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class TestConfig:
    """Central configuration for all tests."""
    
    # Test data settings
    test_data_size: int = 100
    test_image_size: tuple = (32, 32)
    test_batch_size: int = 16
    
    # Molecular simulation settings  
    test_simulation_steps: int = 1000
    test_temperature: float = 300.0
    test_salt_concentration: float = 0.5
    
    # Neural network settings
    test_hidden_dim: int = 64
    test_num_heads: int = 4
    test_num_layers: int = 2
    
    # Performance test settings
    benchmark_iterations: int = 10
    performance_threshold_ms: float = 1000.0
    memory_threshold_mb: float = 512.0
    
    # GPU settings
    use_gpu: bool = torch.cuda.is_available()
    gpu_memory_fraction: float = 0.3
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Timeout settings (seconds)
    unit_test_timeout: int = 30
    integration_test_timeout: int = 300
    performance_test_timeout: int = 600
    slow_test_timeout: int = 1800
    
    # Coverage settings
    min_coverage_percentage: float = 80.0
    
    # File paths
    test_data_dir: Path = PROJECT_ROOT / "tests" / "fixtures"
    temp_output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize derived settings."""
        if self.temp_output_dir is None:
            self.temp_output_dir = Path(tempfile.mkdtemp(prefix="dna_origami_ae_test_"))
        
        # Create test directories
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_output_dir.mkdir(parents=True, exist_ok=True)

# Global test configuration instance
TEST_CONFIG = TestConfig()

class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def set_random_seeds(seed: int = TEST_CONFIG.random_seed):
        """Set random seeds for reproducible tests."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def create_test_image(size: tuple = TEST_CONFIG.test_image_size) -> np.ndarray:
        """Create a test image for encoding tests."""
        TestUtilities.set_random_seeds()
        image = np.random.randint(0, 256, size=size, dtype=np.uint8)
        return image
    
    @staticmethod
    def create_test_dna_sequence(length: int = 1000) -> str:
        """Create a random DNA sequence for testing."""
        TestUtilities.set_random_seeds()
        bases = ['A', 'T', 'G', 'C']
        return ''.join(np.random.choice(bases, size=length))
    
    @staticmethod
    def create_test_origami_coordinates(num_bases: int = 100) -> np.ndarray:
        """Create test 3D coordinates for origami structure."""
        TestUtilities.set_random_seeds()
        return np.random.rand(num_bases, 3) * 100  # 100nm box
    
    @staticmethod
    def assert_dna_sequence_valid(sequence: str):
        """Assert that a DNA sequence is valid."""
        valid_bases = set('ATGC')
        assert all(base in valid_bases for base in sequence.upper()), \
            f"Invalid DNA sequence: {sequence[:50]}..."
    
    @staticmethod
    def assert_image_shape_valid(image: np.ndarray, expected_shape: tuple):
        """Assert that an image has the expected shape."""
        assert image.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {image.shape}"
    
    @staticmethod
    def assert_gpu_available():
        """Skip test if GPU is not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU not available")
    
    @staticmethod
    def assert_memory_usage_reasonable(max_mb: float = TEST_CONFIG.memory_threshold_mb):
        """Assert that memory usage is within reasonable bounds."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < max_mb, f"Memory usage {memory_mb:.1f}MB exceeds threshold {max_mb}MB"

class MockDataGenerator:
    """Generate mock data for testing."""
    
    @staticmethod
    def generate_test_dataset(num_samples: int = TEST_CONFIG.test_data_size):
        """Generate a test dataset for ML training."""
        TestUtilities.set_random_seeds()
        
        images = np.random.randint(
            0, 256, 
            size=(num_samples, *TEST_CONFIG.test_image_size), 
            dtype=np.uint8
        )
        
        # Generate corresponding DNA sequences
        dna_sequences = []
        for _ in range(num_samples):
            seq_length = np.random.randint(500, 2000)
            dna_sequences.append(TestUtilities.create_test_dna_sequence(seq_length))
        
        return {
            'images': images,
            'dna_sequences': dna_sequences,
            'metadata': {
                'num_samples': num_samples,
                'image_size': TEST_CONFIG.test_image_size,
                'generated_with_seed': TEST_CONFIG.random_seed
            }
        }
    
    @staticmethod
    def generate_origami_structure_data(num_structures: int = 10):
        """Generate mock origami structure data."""
        TestUtilities.set_random_seeds()
        
        structures = []
        for i in range(num_structures):
            num_bases = np.random.randint(50, 500)
            coordinates = TestUtilities.create_test_origami_coordinates(num_bases)
            sequence = TestUtilities.create_test_dna_sequence(num_bases)
            
            structures.append({
                'id': f'test_structure_{i}',
                'coordinates': coordinates,
                'sequence': sequence,
                'num_bases': num_bases
            })
        
        return structures

class TestMarkers:
    """Centralized test markers for pytest."""
    
    UNIT = pytest.mark.unit
    INTEGRATION = pytest.mark.integration
    PERFORMANCE = pytest.mark.performance
    SLOW = pytest.mark.slow
    GPU = pytest.mark.gpu
    EXPERIMENTAL = pytest.mark.experimental
    
    # Domain-specific markers
    MOLECULAR_DYNAMICS = pytest.mark.molecular_dynamics
    NEURAL_NETWORK = pytest.mark.neural_network
    WET_LAB = pytest.mark.wet_lab
    ENCODING = pytest.mark.encoding
    DECODING = pytest.mark.decoding
    DESIGN = pytest.mark.design
    
    # Test type markers
    SMOKE = pytest.mark.smoke
    REGRESSION = pytest.mark.regression
    
    # Special markers with conditions
    @staticmethod
    def skipif_no_gpu(reason: str = "CUDA GPU not available"):
        """Skip test if GPU is not available."""
        return pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason=reason
        )
    
    @staticmethod
    def skipif_slow(reason: str = "Slow test skipped"):
        """Skip slow tests unless explicitly requested."""
        return pytest.mark.skipif(
            not pytest.config.getoption("--runslow", default=False),
            reason=reason
        )

# Environment variables for testing
TEST_ENV = {
    'DNA_ORIGAMI_AE_TEST_MODE': 'true',
    'DNA_ORIGAMI_AE_LOG_LEVEL': 'DEBUG',
    'DNA_ORIGAMI_AE_RANDOM_SEED': str(TEST_CONFIG.random_seed),
    'CUDA_VISIBLE_DEVICES': '0' if TEST_CONFIG.use_gpu else '',
    'PYTHONPATH': str(PROJECT_ROOT),
}

# Apply test environment variables
for key, value in TEST_ENV.items():
    os.environ[key] = value

# Test fixtures directory structure
FIXTURES_STRUCTURE = {
    'images': TEST_CONFIG.test_data_dir / 'images',
    'dna_sequences': TEST_CONFIG.test_data_dir / 'dna_sequences',
    'origami_structures': TEST_CONFIG.test_data_dir / 'origami_structures',
    'simulation_data': TEST_CONFIG.test_data_dir / 'simulation_data',
    'model_weights': TEST_CONFIG.test_data_dir / 'model_weights',
    'protocols': TEST_CONFIG.test_data_dir / 'protocols',
}

# Create fixture directories
for fixture_type, path in FIXTURES_STRUCTURE.items():
    path.mkdir(parents=True, exist_ok=True)

# Export commonly used items
__all__ = [
    'TEST_CONFIG',
    'TestUtilities', 
    'MockDataGenerator',
    'TestMarkers',
    'PROJECT_ROOT',
    'FIXTURES_STRUCTURE',
]