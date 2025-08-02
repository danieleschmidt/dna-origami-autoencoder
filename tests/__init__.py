"""
DNA-Origami-AutoEncoder Test Suite

This package contains comprehensive tests for the DNA-Origami-AutoEncoder project,
organized into the following categories:

- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- performance/: Performance and benchmark tests
- fixtures/: Test data and fixtures

Test Organization:
- Each main package has corresponding unit tests
- Integration tests validate end-to-end workflows
- Performance tests ensure scalability and efficiency
- Fixtures provide reusable test data and mock objects

Running Tests:
- pytest                    # Run all tests
- pytest tests/unit/        # Run unit tests only
- pytest -m integration     # Run integration tests only
- pytest -m performance     # Run performance tests only
- pytest -k "test_encoding" # Run specific test patterns

Markers:
- @pytest.mark.unit         # Unit test
- @pytest.mark.integration  # Integration test
- @pytest.mark.performance  # Performance test  
- @pytest.mark.gpu          # Requires GPU
- @pytest.mark.slow         # Long-running test
- @pytest.mark.experimental # Experimental feature test
"""

__version__ = "0.1.0"