# Contributing to DNA-Origami-AutoEncoder

Thank you for your interest in contributing to the DNA-Origami-AutoEncoder project! This document provides guidelines and information for contributors at all levels, from code contributions to research collaborations.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Contributions](#code-contributions)
4. [Research Contributions](#research-contributions)
5. [Documentation](#documentation)
6. [Testing](#testing)
7. [Community Guidelines](#community-guidelines)
8. [Recognition](#recognition)

## Getting Started

### Prerequisites

Before contributing, ensure you have the following setup:

#### Software Requirements
- Python 3.9+ with conda/mamba for environment management
- CUDA-capable GPU (recommended for simulation work)
- Git with LFS support for large datasets
- Docker for containerized development

#### Knowledge Requirements
- **For Code Contributors**: Python, PyTorch/TensorFlow, molecular simulation basics
- **For Research Contributors**: DNA nanotechnology, machine learning, synthetic biology
- **For Documentation Contributors**: Markdown, scientific writing

### Installation for Development

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/dna-origami-autoencoder.git
   cd dna-origami-autoencoder
   git remote add upstream https://github.com/danieleschmidt/dna-origami-autoencoder.git
   ```

2. **Environment Setup**
   ```bash
   # Create development environment
   conda env create -f environment-dev.yml
   conda activate dna-origami-ae-dev
   
   # Install in development mode
   pip install -e ".[dev]"
   
   # Setup pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run tests to verify setup
   pytest tests/
   
   # Run basic pipeline test
   python -c "from dna_origami_ae import quick_test; quick_test()"
   ```

## Development Workflow

### Branch Strategy

We use a **GitFlow-inspired** branching model:

- `main`: Production-ready releases
- `develop`: Integration branch for ongoing development
- `feature/*`: New features and enhancements
- `research/*`: Experimental research branches
- `hotfix/*`: Critical bug fixes

### Making Changes

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our [style guide](#code-style)
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass locally

3. **Commit Changes**
   ```bash
   # Follow conventional commits format
   git commit -m "feat: add new DNA encoding algorithm
   
   - Implement base-4 encoding with biological constraints
   - Add GC content optimization
   - Include unit tests and benchmarks
   
   Closes #123"
   ```

4. **Submit Pull Request**
   - Push to your fork
   - Create pull request against `develop` branch
   - Fill out the pull request template completely
   - Ensure CI passes

### Code Style

We follow **PEP 8** with these specific guidelines:

#### Python Code Style
```python
# Use type hints
def encode_sequence(sequence: str, method: str = "base4") -> List[str]:
    """Encode DNA sequence using specified method.
    
    Args:
        sequence: Input DNA sequence
        method: Encoding method ("base4", "goldman", etc.)
    
    Returns:
        List of encoded segments
        
    Raises:
        ValueError: If sequence contains invalid characters
    """
    pass

# Use dataclasses for structured data
@dataclass
class OrigamiDesign:
    scaffold: str
    staples: List[str]
    shape: str
    dimensions: Tuple[float, float]
```

#### Documentation Style
- **Docstrings**: Google style for all public functions/classes
- **Comments**: Explain the "why", not the "what"
- **Type Hints**: Required for all function signatures
- **Examples**: Include usage examples in docstrings

#### File Organization
```
dna_origami_ae/
├── encoding/           # Image to DNA conversion
├── design/            # Origami structure design
├── simulation/        # Molecular dynamics
├── decoding/          # Neural network models
├── wetlab/           # Laboratory protocols
├── analysis/         # Data analysis tools
└── utils/            # Shared utilities
```

## Code Contributions

### Types of Code Contributions

#### 1. Core Algorithm Improvements
- Enhanced encoding/decoding algorithms
- New origami design patterns
- Simulation optimization
- Neural network architectures

**Example**: Implementing a new error correction algorithm
```python
# In dna_origami_ae/encoding/error_correction.py
class ReedSolomonDNA(ErrorCorrection):
    """Reed-Solomon error correction optimized for DNA storage."""
    
    def __init__(self, redundancy: float = 0.3):
        self.redundancy = redundancy
        # Implementation details...
```

#### 2. Performance Optimizations
- GPU acceleration improvements
- Memory usage optimization
- Parallel processing enhancements
- Caching strategies

#### 3. New Features
- Additional file format support
- New visualization tools
- Integration with external tools
- Web interfaces/APIs

#### 4. Bug Fixes
- Correctness issues
- Performance regressions
- Memory leaks
- Edge case handling

### Code Review Process

All code contributions go through peer review:

1. **Automated Checks**: CI runs tests, linting, and security scans
2. **Technical Review**: 2+ reviewers check code quality and correctness
3. **Research Review**: For algorithm changes, domain expert review required
4. **Documentation Review**: Ensure documentation is complete and accurate

### Testing Requirements

#### Unit Tests
```python
# tests/test_encoding.py
def test_base4_encoding():
    """Test base-4 encoding with known inputs."""
    encoder = Base4Encoder()
    result = encoder.encode("ATCG")
    expected = [0, 1, 2, 3]  # A=0, T=1, C=2, G=3
    assert result == expected

def test_biological_constraints():
    """Test that encoded sequences meet biological constraints."""
    encoder = Base4Encoder(constraints=BiologicalConstraints())
    sequence = encoder.encode_with_constraints(test_data)
    assert encoder.constraints.validate(sequence)
```

#### Integration Tests
```python
# tests/test_integration.py
def test_end_to_end_pipeline():
    """Test complete pipeline from image to DNA and back."""
    # Load test image
    image = load_test_image("mnist_7.png")
    
    # Encode to DNA
    encoder = DNAEncoder()
    dna_sequence = encoder.encode_image(image)
    
    # Design origami
    designer = OrigamiDesigner()
    origami = designer.design(dna_sequence)
    
    # Simulate folding
    simulator = OrigamiSimulator()
    structure = simulator.simulate(origami)
    
    # Decode with neural network
    decoder = TransformerDecoder()
    reconstructed = decoder.decode(structure)
    
    # Verify accuracy
    mse = calculate_mse(image, reconstructed)
    assert mse < 0.05  # Less than 5% error
```

#### Performance Tests
```python
# tests/test_performance.py
def test_encoding_speed():
    """Ensure encoding meets performance requirements."""
    encoder = DNAEncoder()
    image = generate_test_image(32, 32)
    
    start_time = time.time()
    result = encoder.encode_image(image)
    duration = time.time() - start_time
    
    assert duration < 1.0  # Less than 1 second
    assert len(result) > 0  # Valid output
```

## Research Contributions

### Types of Research Contributions

#### 1. Algorithm Development
- Novel encoding schemes
- Advanced error correction methods
- New neural network architectures
- Simulation methodology improvements

#### 2. Experimental Validation
- Wet-lab protocol development
- AFM/TEM imaging studies
- Folding kinetics analysis
- Error rate characterization

#### 3. Theoretical Analysis
- Information theory applications
- Thermodynamic modeling
- Complexity analysis
- Performance bounds

#### 4. Application Studies
- Use case demonstrations
- Benchmark comparisons
- Scalability analysis
- Cost-benefit studies

### Research Workflow

1. **Proposal**: Submit research proposal via GitHub issue
2. **Discussion**: Community discussion and feedback
3. **Implementation**: Develop experimental code/protocols
4. **Validation**: Conduct experiments and analysis
5. **Documentation**: Write up results and methodology
6. **Review**: Peer review within community
7. **Integration**: Merge validated improvements into main codebase

### Research Branch Guidelines

Research branches should follow this structure:
```
research/feature-name/
├── README.md                 # Research objectives and methodology
├── notebooks/               # Jupyter notebooks for analysis
├── experiments/            # Experimental scripts
├── data/                   # Datasets (use Git LFS)
├── results/               # Generated results and figures
└── docs/                  # Research documentation
```

### Experimental Data Management

#### Data Organization
```
data/
├── raw/                   # Original, unprocessed data
├── processed/            # Cleaned, processed datasets
├── external/             # Third-party datasets
└── synthetic/           # Simulated/generated data
```

#### Metadata Requirements
Each dataset should include:
- `README.md` with data description
- `metadata.yml` with structured information
- Data provenance and collection methods
- Usage rights and restrictions

### Publication and Dissemination

#### Internal Documentation
- Research findings documented in `docs/research/`
- Methodology shared via code and notebooks
- Results presented at team meetings

#### External Publication
- Conference presentations encouraged
- Peer-reviewed publications supported
- Preprints welcomed on appropriate servers
- Attribution follows academic standards

## Documentation

### Types of Documentation

#### 1. API Documentation
- Automatically generated from docstrings
- Include parameter descriptions and examples
- Cover all public interfaces

#### 2. User Guides
- Step-by-step tutorials
- Common use cases
- Troubleshooting guides
- Best practices

#### 3. Research Documentation
- Algorithm explanations
- Theoretical background
- Experimental methodologies
- Performance analysis

#### 4. Developer Documentation
- Architecture overviews
- Contribution guidelines
- Setup instructions
- Testing procedures

### Documentation Standards

#### Writing Style
- Clear, concise language
- Active voice preferred
- Jargon explained or linked to glossary
- International audience considerations

#### Structure
```markdown
# Title

## Overview
Brief description of the topic

## Prerequisites
What users need to know/have installed

## Step-by-Step Instructions
1. First step with code example
2. Second step with expected output
3. Continue...

## Advanced Usage
More complex scenarios

## Troubleshooting
Common issues and solutions

## Related Topics
Links to related documentation
```

#### Code Examples
```python
# Always include working code examples
from dna_origami_ae import DNAEncoder

# Initialize encoder with biological constraints
encoder = DNAEncoder(
    method="base4",
    constraints=BiologicalConstraints(
        gc_content=(0.4, 0.6),
        max_homopolymer=4
    )
)

# Encode sample image
image = load_example_image()
dna_sequence = encoder.encode_image(image)
print(f"Generated {len(dna_sequence)} base sequence")
```

## Testing

### Testing Philosophy

We follow **Test-Driven Development (TDD)** principles:
1. Write tests before implementing features
2. Ensure comprehensive test coverage
3. Test both happy paths and edge cases
4. Include performance and integration tests

### Test Categories

#### 1. Unit Tests (`tests/unit/`)
- Test individual functions/classes
- Fast execution (<1s per test)
- No external dependencies
- High code coverage (>90%)

#### 2. Integration Tests (`tests/integration/`)
- Test component interactions
- End-to-end workflow validation
- Medium execution time (<30s per test)
- May use mock external services

#### 3. Performance Tests (`tests/performance/`)
- Benchmark critical operations
- Memory usage validation
- Scalability testing
- Regression detection

#### 4. Research Tests (`tests/research/`)
- Validate scientific correctness
- Reproduce published results
- Long-running experiments
- May require special hardware

### Test Data Management

#### Synthetic Data
```python
# Generate reproducible test data
def generate_test_image(width: int, height: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic test image for reproducible testing."""
    np.random.seed(seed)
    return np.random.randint(0, 256, (height, width), dtype=np.uint8)
```

#### Real Experimental Data
- Use Git LFS for large files
- Anonymize sensitive data
- Include data usage agreements
- Provide data description and provenance

### Continuous Integration

Our CI pipeline includes:
1. **Code Quality**: Linting, formatting, type checking
2. **Security**: SAST scanning, dependency vulnerability checks
3. **Testing**: Unit, integration, and performance tests
4. **Documentation**: Build documentation and check links
5. **Packaging**: Build and test package installation

## Community Guidelines

### Communication Channels

#### GitHub
- **Issues**: Bug reports, feature requests, research proposals
- **Discussions**: General questions, ideas, announcements
- **Pull Requests**: Code and documentation contributions

#### Discord
- **General**: Casual conversation and quick questions
- **Research**: In-depth scientific discussions
- **Development**: Technical implementation discussions
- **Wet-Lab**: Experimental protocols and results

#### Mailing List
- **Announcements**: Release notifications, major updates
- **Research**: Paper discussions, collaboration opportunities
- **Events**: Conference presentations, workshops

### Getting Help

#### For Users
1. Check documentation and FAQ
2. Search existing GitHub issues
3. Ask in Discord #general channel
4. Create GitHub issue with detailed description

#### For Contributors
1. Review this contributing guide
2. Ask in Discord #development channel
3. Tag maintainers in GitHub issues
4. Schedule video call for complex discussions

#### For Researchers
1. Post in Discord #research channel
2. Create GitHub discussion for proposals
3. Reach out to domain experts directly
4. Join monthly research meetings

### Mentorship Program

We offer mentorship for new contributors:

#### For Students
- Paired with experienced researcher
- Structured learning path
- Regular check-ins and feedback
- Conference presentation opportunities

#### For Professionals
- Technical mentorship on specific areas
- Career development guidance
- Industry partnership facilitation
- Publication collaboration

### Conflict Resolution

If conflicts arise:
1. **Direct Communication**: Try to resolve directly with involved parties
2. **Mediation**: Request mediation from community leaders
3. **Formal Process**: Follow Code of Conduct enforcement procedures
4. **Appeal**: Formal appeal process available for all decisions

## Recognition

### Contribution Recognition

#### Code Contributions
- GitHub contributor listing
- Changelog acknowledgments
- Conference presentation co-authorship
- Software paper co-authorship

#### Research Contributions
- Academic paper co-authorship
- Conference presentation opportunities
- Research grant co-investigator roles
- Community award nominations

#### Documentation Contributions
- Documentation credits
- Tutorial video acknowledgments
- Workshop co-facilitation
- Educational material co-authorship

### Awards and Honors

Annual community awards for:
- **Outstanding Code Contribution**
- **Best Research Innovation**
- **Exceptional Documentation**
- **Community Leadership**
- **Mentorship Excellence**

### Professional Development

#### Career Support
- Reference letters for academic/industry positions
- Professional network introductions
- Conference travel support (when funding available)
- Skills development workshops

#### Academic Credit
- Research credit for student contributors
- Thesis committee participation
- Internship facilitation
- Publication opportunities

## Getting Started Checklist

Before making your first contribution:

- [ ] Read and understand the Code of Conduct
- [ ] Set up development environment
- [ ] Run tests to verify setup
- [ ] Read relevant documentation
- [ ] Join Discord community
- [ ] Introduce yourself in #general
- [ ] Find a "good first issue" to work on
- [ ] Ask questions if anything is unclear

## Questions?

If you have questions about contributing, please:

1. Check the FAQ section
2. Search existing GitHub issues and discussions
3. Ask in the Discord #development channel
4. Contact maintainers directly: [email]

We're excited to have you contribute to the DNA-Origami-AutoEncoder project and help advance the field of biological information systems!

---

*Last Updated: 2025-08-02*  
*Next Review: 2025-11-02*