# Testing Documentation

This directory contains documentation for the DNA-Origami-AutoEncoder testing infrastructure.

## Test Organization

The test suite is organized into several categories:

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Fast execution (< 1 second per test)
- No external dependencies
- High code coverage target (>90%)

### Integration Tests (`tests/integration/`)
- Test component interactions
- End-to-end workflow validation
- Medium execution time (< 30 seconds per test)
- May use mock external services

### Performance Tests (`tests/performance/`)
- Benchmark critical operations
- Memory usage validation
- Scalability testing
- Regression detection

### Test Fixtures (`tests/fixtures/`)
- Reusable test data
- Mock objects and utilities
- Sample datasets
- Configuration files

## Running Tests

### Basic Usage
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run tests with specific markers
pytest -m unit
pytest -m integration
pytest -m performance
pytest -m gpu  # GPU-required tests
pytest -m slow  # Long-running tests
```

### Coverage Analysis
```bash
# Run tests with coverage
pytest --cov=dna_origami_ae

# Generate HTML coverage report
pytest --cov=dna_origami_ae --cov-report=html

# Coverage with branch analysis
pytest --cov=dna_origami_ae --cov-branch
```

### Performance Testing
```bash
# Run performance tests only
pytest -m performance

# Run with performance profiling
pytest --benchmark-only

# Skip slow tests
pytest -m "not slow"
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery settings
- Markers for test categorization
- Coverage configuration
- Logging setup
- Timeout settings

### Coverage Configuration (`.coveragerc`)
- Source code inclusion/exclusion
- Branch coverage settings
- Report formatting
- Minimum coverage thresholds

### Fixtures (`tests/conftest.py`)
- Shared test fixtures
- Test environment setup
- Mock object configuration
- Performance tracking utilities

## Test Data Management

### Synthetic Data Generation
The test suite generates synthetic data for reproducible testing:

```python
# Generate test images
sample_image = generate_test_image(32, 32, seed=42)

# Generate DNA sequences
sample_sequence = generate_test_dna_sequence(length=1000, seed=42)

# Generate molecular trajectories
sample_trajectory = generate_test_trajectory(
    n_frames=100, 
    n_particles=1000, 
    seed=42
)
```

### Real Experimental Data
For validation against real experimental data:
- Store in `tests/fixtures/experimental/`
- Use Git LFS for large files
- Include metadata and provenance
- Respect data usage agreements

## Test Utilities

### Assertion Helpers
```python
from tests.conftest import (
    assert_arrays_close,
    assert_dna_sequence_valid,
    assert_image_shape,
    assert_tensor_device
)

# Numerical comparisons
assert_arrays_close(actual, expected, rtol=1e-5)

# DNA sequence validation
assert_dna_sequence_valid("ATCGATCG")

# Shape verification
assert_image_shape(image, (32, 32))

# Device verification (for GPU tests)
assert_tensor_device(tensor, torch.device("cuda"))
```

### Skip Conditions
```python
from tests.conftest import (
    skip_if_no_gpu,
    skip_if_no_oxdna,
    skip_if_slow
)

@skip_if_no_gpu
def test_gpu_operation():
    # Only runs if GPU is available
    pass

@skip_if_slow
def test_long_running_operation():
    # Skipped if SKIP_SLOW_TESTS=true
    pass
```

### Parametrized Tests
```python
from tests.conftest import (
    image_sizes,
    dna_encoding_methods,
    origami_shapes
)

@image_sizes
def test_with_different_sizes(image_size):
    # Runs with (16,16), (32,32), (64,64)
    pass

@dna_encoding_methods
def test_encoding_methods(method):
    # Runs with "base4", "goldman", "church"
    pass
```

## Performance Testing

### Memory Profiling
```python
def test_memory_usage(performance_tracker):
    performance_tracker.start()
    
    # Code under test
    result = expensive_operation()
    
    performance_tracker.stop()
    
    # Assert memory usage
    assert performance_tracker.memory_delta < 100  # MB
```

### Timing Benchmarks
```python
def test_processing_speed(performance_tracker):
    performance_tracker.start()
    
    # Code under test
    process_large_dataset()
    
    performance_tracker.stop()
    
    # Assert timing requirements
    assert performance_tracker.duration < 5.0  # seconds
```

### Scalability Testing
```python
@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_scaling_behavior(size, performance_tracker):
    performance_tracker.start()
    
    # Test with different sizes
    result = process_data_of_size(size)
    
    performance_tracker.stop()
    
    # Assert scaling behavior
    expected_time = size * 0.001  # Linear scaling
    assert performance_tracker.duration < expected_time * 2
```

## Mock Objects and Test Doubles

### Component Mocks
The test suite provides pre-configured mocks for major components:

```python
def test_with_mocks(
    mock_encoder,
    mock_decoder,
    mock_simulator,
    mock_origami_designer
):
    # Use pre-configured mocks
    dna_sequence = mock_encoder.encode_image(sample_image)
    origami = mock_origami_designer.design_origami(dna_sequence)
    simulation = mock_simulator.simulate_folding(origami)
    result = mock_decoder.decode_structure(simulation["final_structure"])
```

### Custom Mocks
```python
from unittest.mock import Mock, patch

def test_with_custom_mock():
    custom_mock = Mock()
    custom_mock.method.return_value = "expected_result"
    
    with patch('module.function', custom_mock.method):
        result = function_under_test()
        assert result == "expected_result"
```

## Continuous Integration

### GitHub Actions Integration
Tests are automatically run on:
- Pull requests
- Pushes to main branch
- Scheduled runs (nightly)

### Test Matrix
Tests run across multiple configurations:
- Python versions: 3.9, 3.10, 3.11
- Operating systems: Ubuntu, macOS, Windows
- Dependency versions: minimal, latest

### Performance Regression Detection
- Benchmark results tracked over time
- Alerts for significant performance degradation
- Performance reports in PR comments

## Best Practices

### Writing Tests

1. **Follow AAA Pattern**
   ```python
   def test_function():
       # Arrange
       input_data = setup_test_data()
       
       # Act
       result = function_under_test(input_data)
       
       # Assert
       assert result == expected_output
   ```

2. **Use Descriptive Names**
   ```python
   def test_dna_encoder_handles_invalid_image_dimensions():
       # Clear what is being tested
       pass
   ```

3. **Test Edge Cases**
   ```python
   @pytest.mark.parametrize("input_value", [
       0,           # minimum
       255,         # maximum
       -1,          # below minimum
       256,         # above maximum
       None,        # null value
       "invalid",   # wrong type
   ])
   def test_edge_cases(input_value):
       # Test boundary conditions
       pass
   ```

4. **Use Fixtures for Setup**
   ```python
   @pytest.fixture
   def complex_test_data():
       # Expensive setup code
       return setup_complex_data()
   
   def test_with_fixture(complex_test_data):
       # Test uses pre-setup data
       pass
   ```

### Performance Testing Guidelines

1. **Set Realistic Expectations**
   - Base thresholds on actual requirements
   - Consider hardware variations
   - Allow reasonable margins

2. **Test Multiple Scenarios**
   - Different data sizes
   - Various input types
   - Edge cases and stress conditions

3. **Monitor Trends**
   - Track performance over time
   - Detect gradual degradation
   - Compare against baselines

### Test Maintenance

1. **Keep Tests Updated**
   - Update tests when code changes
   - Maintain test dependencies
   - Remove obsolete tests

2. **Review Test Coverage**
   - Aim for high coverage (>80%)
   - Focus on critical paths
   - Don't test trivial code

3. **Optimize Test Performance**
   - Keep unit tests fast (<1s)
   - Use mocks appropriately
   - Parallelize when possible

## Troubleshooting

### Common Issues

#### Tests Failing on CI but Passing Locally
- Check environment differences
- Verify dependency versions
- Check for timing-dependent tests
- Ensure reproducible test data

#### Slow Test Execution
- Profile test execution times
- Identify bottlenecks
- Use appropriate test markers
- Consider test parallelization

#### Flaky Tests
- Identify non-deterministic behavior
- Add appropriate timeouts
- Use fixed random seeds
- Mock external dependencies

#### Memory Issues in Tests
- Monitor memory usage
- Clean up resources properly
- Use memory profiling tools
- Avoid memory leaks in test setup

### Debugging Tests

```bash
# Run single test with verbose output
pytest -v -s tests/unit/test_specific.py::test_function

# Debug test failures
pytest --pdb tests/unit/test_failing.py

# Run with coverage and missing line report
pytest --cov=dna_origami_ae --cov-report=term-missing

# Profile test performance
pytest --profile tests/performance/
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)
- [Mock Object Library](https://docs.python.org/3/library/unittest.mock.html)