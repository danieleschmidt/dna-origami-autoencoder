"""
Pytest configuration and shared fixtures for DNA-Origami-AutoEncoder tests.

This module contains:
- Global pytest configuration
- Shared fixtures for all test modules
- Common test utilities and helpers
- Test data generation functions
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, Optional
import warnings

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Configure warnings
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to unit tests
        if 'unit' in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add 'integration' marker to integration tests
        elif 'integration' in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add 'performance' marker to performance tests
        elif 'performance' in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add 'gpu' marker to tests that require GPU
        if 'gpu' in item.name.lower() or 'cuda' in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Add 'slow' marker to tests with long runtime
        if any(keyword in item.name.lower() for keyword in ['simulation', 'training', 'benchmark']):
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Environment and Configuration Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root) -> Path:
    """Return the test data directory."""
    data_dir = project_root / "tests" / "fixtures"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Return test configuration dictionary."""
    return {
        "random_seed": 42,
        "test_data_size": 100,
        "image_size": (32, 32),
        "dna_sequence_length": 1000,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "device": "cpu",  # Use CPU for tests unless GPU tests are specifically marked
    }


@pytest.fixture(autouse=True)
def setup_test_environment(test_config):
    """Setup test environment for each test."""
    # Set random seeds
    np.random.seed(test_config["random_seed"])
    torch.manual_seed(test_config["random_seed"])
    
    # Ensure CPU-only mode for most tests
    if not pytest.current_item.get_closest_marker("gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


# =============================================================================
# Data Generation Fixtures
# =============================================================================

@pytest.fixture
def sample_image(test_config) -> np.ndarray:
    """Generate a sample grayscale image for testing."""
    height, width = test_config["image_size"]
    np.random.seed(test_config["random_seed"])
    return np.random.randint(0, 256, (height, width), dtype=np.uint8)


@pytest.fixture
def sample_image_batch(test_config) -> np.ndarray:
    """Generate a batch of sample images for testing."""
    batch_size = test_config["batch_size"]
    height, width = test_config["image_size"]
    np.random.seed(test_config["random_seed"])
    return np.random.randint(0, 256, (batch_size, height, width), dtype=np.uint8)


@pytest.fixture
def sample_dna_sequence(test_config) -> str:
    """Generate a sample DNA sequence for testing."""
    length = test_config["dna_sequence_length"]
    np.random.seed(test_config["random_seed"])
    bases = ['A', 'T', 'G', 'C']
    return ''.join(np.random.choice(bases, size=length))


@pytest.fixture
def sample_origami_design() -> Dict[str, Any]:
    """Generate a sample origami design for testing."""
    return {
        "scaffold": "M13mp18",
        "scaffold_length": 7249,
        "staples": [
            {"sequence": "ATCGATCGATCGATCG", "start": 0, "end": 15},
            {"sequence": "GCTAGCTAGCTAGCTA", "start": 16, "end": 31},
        ],
        "shape": "rectangle",
        "dimensions": (100, 100),  # nm
        "crossovers": [(10, 20), (30, 40)],
    }


@pytest.fixture
def sample_trajectory() -> Dict[str, Any]:
    """Generate a sample molecular dynamics trajectory for testing."""
    n_frames = 100
    n_particles = 1000
    n_dimensions = 3
    
    np.random.seed(42)
    positions = np.random.randn(n_frames, n_particles, n_dimensions) * 10
    
    return {
        "positions": positions,
        "n_frames": n_frames,
        "n_particles": n_particles,
        "timestep": 0.001,  # ps
        "temperature": 300,  # K
        "box_size": [100, 100, 100],  # Angstroms
    }


# =============================================================================
# Component Fixtures
# =============================================================================

@pytest.fixture
def mock_encoder():
    """Create a mock DNA encoder for testing."""
    encoder = Mock()
    encoder.encode_image.return_value = "ATCGATCGATCG" * 100
    encoder.encode_with_constraints.return_value = ["ATCGATCGATCG" * 25] * 4
    return encoder


@pytest.fixture
def mock_decoder():
    """Create a mock neural decoder for testing."""
    decoder = Mock()
    decoder.decode_structure.return_value = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    decoder.predict.return_value = torch.randn(1, 1, 32, 32)
    return decoder


@pytest.fixture
def mock_simulator():
    """Create a mock molecular dynamics simulator for testing."""
    simulator = Mock()
    simulator.simulate_folding.return_value = {
        "final_structure": np.random.randn(1000, 3),
        "trajectory": np.random.randn(100, 1000, 3),
        "energy": np.random.randn(100),
    }
    return simulator


@pytest.fixture
def mock_origami_designer():
    """Create a mock origami designer for testing."""
    designer = Mock()
    designer.design_origami.return_value = {
        "scaffold": "M13mp18",
        "staples": ["ATCG" * 8] * 200,
        "structure": np.random.randn(7249, 3),
    }
    return designer


# =============================================================================
# GPU and Hardware Fixtures
# =============================================================================

@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def device(gpu_available, request) -> torch.device:
    """Return appropriate device for testing."""
    if request.node.get_closest_marker("gpu") and gpu_available:
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cuda_context():
    """Context manager for CUDA operations in tests."""
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            yield
    else:
        pytest.skip("CUDA not available")


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_file(temp_dir) -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    temp_file_path = temp_dir / "test_file.tmp"
    temp_file_path.touch()
    yield temp_file_path


@pytest.fixture
def sample_data_files(temp_dir, test_data_dir):
    """Create sample data files for testing."""
    files = {}
    
    # Create sample image file
    image_file = temp_dir / "sample_image.npy"
    sample_image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    np.save(image_file, sample_image)
    files["image"] = image_file
    
    # Create sample DNA sequence file
    sequence_file = temp_dir / "sample_sequence.txt"
    sequence_file.write_text("ATCGATCGATCG" * 100)
    files["sequence"] = sequence_file
    
    # Create sample configuration file
    config_file = temp_dir / "sample_config.json"
    config_file.write_text('{"param1": "value1", "param2": 42}')
    files["config"] = config_file
    
    return files


# =============================================================================
# Network and API Fixtures
# =============================================================================

@pytest.fixture
def mock_api_response():
    """Create a mock API response for testing."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"status": "success", "data": {}}
    response.text = '{"status": "success", "data": {}}'
    return response


# =============================================================================
# Database and Storage Fixtures
# =============================================================================

@pytest.fixture
def temp_database(temp_dir):
    """Create a temporary database for testing."""
    db_path = temp_dir / "test.db"
    # Initialize database if needed
    return db_path


# =============================================================================
# Performance and Profiling Fixtures
# =============================================================================

@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    import time
    import psutil
    
    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            self.end_time = time.time()
            self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        @property
        def memory_delta(self):
            if self.start_memory and self.end_memory:
                return self.end_memory - self.start_memory
            return None
    
    return PerformanceTracker()


# =============================================================================
# Cleanup and Teardown
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    
    # Clear CUDA cache if GPU tests were run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reset random seeds
    np.random.seed(None)
    torch.manual_seed(torch.initial_seed())


# =============================================================================
# Test Utilities
# =============================================================================

def assert_arrays_close(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert that two numpy arrays are close within tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_dna_sequence_valid(sequence: str):
    """Assert that a DNA sequence contains only valid bases."""
    valid_bases = set('ATGC')
    sequence_bases = set(sequence.upper())
    assert sequence_bases.issubset(valid_bases), f"Invalid bases found: {sequence_bases - valid_bases}"


def assert_image_shape(image: np.ndarray, expected_shape: tuple):
    """Assert that an image has the expected shape."""
    assert image.shape == expected_shape, f"Expected shape {expected_shape}, got {image.shape}"


def assert_tensor_device(tensor: torch.Tensor, expected_device: torch.device):
    """Assert that a tensor is on the expected device."""
    assert tensor.device == expected_device, f"Expected device {expected_device}, got {tensor.device}"


# =============================================================================
# Skip Conditions
# =============================================================================

skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)

skip_if_no_oxdna = pytest.mark.skipif(
    shutil.which("oxDNA") is None,
    reason="oxDNA not installed"
)

skip_if_slow = pytest.mark.skipif(
    os.environ.get("SKIP_SLOW_TESTS", "false").lower() == "true",
    reason="Slow tests disabled"
)


# =============================================================================
# Parametrize Helpers
# =============================================================================

image_sizes = pytest.mark.parametrize("image_size", [(16, 16), (32, 32), (64, 64)])
dna_encoding_methods = pytest.mark.parametrize("method", ["base4", "goldman", "church"])
origami_shapes = pytest.mark.parametrize("shape", ["rectangle", "triangle", "hexagon"])
simulation_engines = pytest.mark.parametrize("engine", ["oxdna2", "lammps"])


# Make common imports available
__all__ = [
    "assert_arrays_close",
    "assert_dna_sequence_valid", 
    "assert_image_shape",
    "assert_tensor_device",
    "skip_if_no_gpu",
    "skip_if_no_oxdna",
    "skip_if_slow",
    "image_sizes",
    "dna_encoding_methods",
    "origami_shapes",
    "simulation_engines",
]