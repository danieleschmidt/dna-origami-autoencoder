# 🔌 API Documentation

## DNA Origami AutoEncoder API Reference

### Core Classes

#### `DNAEncoder`

Main class for encoding images into DNA sequences.

```python
class DNAEncoder:
    def __init__(self, bits_per_base: int = 2, 
                 error_correction_method: str = 'reed_solomon',
                 biological_constraints: Optional[BiologicalConstraints] = None)
```

**Methods:**

- `encode_image(image: ImageData, parameters: Optional[EncodingParameters] = None) -> List[DNASequence]`
- `decode_image(sequences: List[DNASequence], width: int, height: int) -> ImageData`
- `get_encoding_efficiency(original_size: int, sequences: List[DNASequence]) -> Dict[str, float]`
- `validate_encoding(image: ImageData, sequences: List[DNASequence]) -> Dict[str, Any]`

#### `OrigamiDesigner`

Automated DNA origami structure design.

```python
class OrigamiDesigner:
    def __init__(self, design_params: Optional[DesignParameters] = None)
```

**Methods:**

- `design_origami(dna_sequence: DNASequence, target_shape: str, dimensions: Tuple[float, float]) -> OrigamiStructure`
- `design_from_image_data(sequences: List[DNASequence], target_shape: str) -> OrigamiStructure`
- `estimate_assembly_conditions(structure: OrigamiStructure) -> Dict[str, Any]`
- `export_design_files(structure: OrigamiStructure, output_dir: str, formats: List[str]) -> Dict[str, str]`

#### `OrigamiSimulator`

Molecular dynamics simulation of DNA origami folding.

```python
class OrigamiSimulator:
    def __init__(self, force_field: str = "oxDNA2", 
                 temperature: float = 300.0,
                 salt_concentration: float = 0.5)
```

**Methods:**

- `simulate_folding(origami: OrigamiStructure, simulation_params: Optional[SimulationParameters] = None) -> SimulationResult`
- `simulate_batch(structures: List[OrigamiStructure]) -> List[SimulationResult]`
- `analyze_folding_kinetics(trajectory: TrajectoryData) -> Dict[str, Any]`
- `estimate_computational_requirements(origami: OrigamiStructure, params: SimulationParameters) -> Dict[str, Any]`

#### `TransformerDecoder`

Neural network decoder for structure-to-image reconstruction.

```python
class TransformerDecoder:
    def __init__(self, config: Optional[DecoderConfig] = None)
```

**Methods:**

- `decode_structure(structure: StructureCoordinates) -> ImageData`
- `train(training_data: List[Tuple[StructureCoordinates, ImageData]], epochs: int = 100) -> Dict[str, List[float]]`
- `evaluate(test_data: List[Tuple[StructureCoordinates, ImageData]]) -> Dict[str, float]`
- `save_model(model_path: str) -> None`
- `from_pretrained(model_path: str) -> 'TransformerDecoder'`

### Data Models

#### `DNASequence`

Represents a DNA sequence with biological constraints.

```python
@dataclass
class DNASequence:
    sequence: str
    name: str
    description: str = ""
    constraints: Optional[DNAConstraints] = None
```

**Properties:**
- `length: int` - Length of sequence
- `gc_content: float` - GC content (0.0-1.0)
- `melting_temperature: float` - Estimated melting temperature (°C)

**Methods:**
- `reverse_complement() -> str`
- `analyze_sequence() -> Dict[str, Any]`
- `validate() -> Tuple[bool, List[str]]`

#### `ImageData`

Container for image data with metadata.

```python
@dataclass
class ImageData:
    data: np.ndarray
    metadata: ImageMetadata
    name: str
```

**Class Methods:**
- `from_array(array: np.ndarray, name: str) -> 'ImageData'`
- `from_file(file_path: str) -> 'ImageData'`

**Methods:**
- `resize(size: Tuple[int, int]) -> 'ImageData'`
- `crop(bbox: Tuple[int, int, int, int]) -> 'ImageData'`
- `normalize() -> 'ImageData'`
- `calculate_mse(other: 'ImageData') -> float`
- `calculate_psnr(other: 'ImageData') -> float`
- `calculate_ssim(other: 'ImageData') -> float`

#### `OrigamiStructure`

Represents a complete DNA origami design.

```python
@dataclass
class OrigamiStructure:
    name: str
    scaffold: ScaffoldPath
    staples: List[StapleStrand]
    target_shape: str
    dimensions: Tuple[float, float, float]
```

**Methods:**
- `validate_design() -> Tuple[bool, List[str]]`
- `get_design_statistics() -> Dict[str, Any]`
- `optimize_staple_lengths(target_length: int, tolerance: int) -> None`
- `export_sequences(filename: str, format: str) -> None`
- `get_all_sequences() -> List[DNASequence]`

### Utility Functions

#### Performance & Caching

```python
from dna_origami_ae.utils.cache import cached, cache_manager
from dna_origami_ae.utils.parallel import parallel_map, ParallelConfig
from dna_origami_ae.utils.memory_optimizer import memory_profile, monitor_memory

# Caching decorator
@cached(ttl=3600, cache_type='smart')
def expensive_computation(data):
    pass

# Parallel processing
config = ParallelConfig(max_workers=8, use_threads=False)
results = parallel_map(process_function, data_list, config)

# Memory monitoring
@memory_profile
def memory_intensive_function():
    pass

# Get memory stats
stats = monitor_memory()
```

#### Validation & Compliance

```python
from dna_origami_ae.utils.validators import validate_dna_sequence, ValidationError
from dna_origami_ae.utils.compliance import ensure_compliance, compliance_manager

# Input validation
try:
    is_valid, errors = validate_dna_sequence("ATGCATGC")
    if not is_valid:
        raise ValidationError("Invalid sequence", errors)
except ValidationError as e:
    print(f"Validation failed: {e}")

# Compliance checking
compliant = ensure_compliance(
    data_type="genetic_sequence",
    purpose="research",
    user_id="user123"
)
```

#### Internationalization

```python
from dna_origami_ae.utils.i18n import _, set_locale, get_available_locales

# Set language
set_locale('fr_FR')

# Translate messages
message = _("encoding_complete")  # "Encodage terminé"

# Available locales
locales = get_available_locales()
```

#### Cross-Platform Support

```python
from dna_origami_ae.utils.platform import platform_info, path_manager, get_platform_summary

# Platform detection
if platform_info.is_windows():
    # Windows-specific code
    pass

# Cross-platform paths
app_data = path_manager.get_app_data_dir()
cache_dir = path_manager.get_cache_dir()

# Platform summary
summary = get_platform_summary()
```

### Configuration

#### Environment Variables

```bash
# Performance
DNA_ORIGAMI_MAX_WORKERS=8
DNA_ORIGAMI_CACHE_SIZE_MB=1000
DNA_ORIGAMI_USE_GPU=true

# Compliance
DNA_ORIGAMI_REGULATIONS=gdpr,ccpa,pdpa
DNA_ORIGAMI_DATA_RETENTION_DAYS=730
DNA_ORIGAMI_ANONYMIZE_DATA=true

# Localization
DNA_ORIGAMI_LOCALE=en_US
DNA_ORIGAMI_TIMEZONE=UTC

# Logging
DNA_ORIGAMI_LOG_LEVEL=INFO
DNA_ORIGAMI_LOG_FILE=/var/log/dna_origami_ae.log
```

#### Configuration File

```json
{
  "general": {
    "locale": "en_US",
    "log_level": "INFO"
  },
  "performance": {
    "max_workers": 8,
    "cache_size_mb": 1000,
    "use_gpu": false
  },
  "compliance": {
    "enabled_regulations": ["gdpr", "ccpa"],
    "data_retention_days": 730,
    "anonymize_sensitive_data": true
  },
  "encoding": {
    "default_error_correction": "reed_solomon",
    "default_redundancy": 0.3,
    "chunk_size": 200
  },
  "simulation": {
    "default_force_field": "oxDNA2",
    "default_temperature": 300.0,
    "default_time_steps": 1000000
  }
}
```

### Error Handling

#### Exception Hierarchy

```python
DNAOrigarmiAEError
├── ValidationError
│   ├── DNASequenceError
│   ├── ImageDataError
│   └── StructureValidationError
├── EncodingError
│   ├── CompressionError
│   └── ErrorCorrectionError
├── SimulationError
│   ├── ForceFieldError
│   └── ConvergenceError
├── DecodingError
│   ├── ModelLoadError
│   └── ReconstructionError
└── ComplianceError
    ├── ConsentError
    └── DataRetentionError
```

#### Example Error Handling

```python
from dna_origami_ae import DNAEncoder, ValidationError, EncodingError

try:
    encoder = DNAEncoder()
    sequences = encoder.encode_image(image)
except ValidationError as e:
    print(f"Input validation failed: {e.errors}")
except EncodingError as e:
    print(f"Encoding failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Guidelines

#### Batch Processing

```python
# Efficient batch processing
from dna_origami_ae.utils.parallel import parallel_batch_process

def process_image_batch(images):
    encoder = DNAEncoder()
    return [encoder.encode_image(img) for img in images]

# Process in parallel batches
results = parallel_batch_process(
    process_image_batch,
    image_list,
    batch_size=10,
    config=ParallelConfig(max_workers=4)
)
```

#### Memory Management

```python
from dna_origami_ae.utils.memory_optimizer import ChunkedArrayProcessor

# Process large arrays in chunks
processor = ChunkedArrayProcessor(chunk_size_mb=100)

def process_large_array(array):
    return processor.process_array_chunked(
        array,
        process_func=your_processing_function,
        combine_func=np.concatenate
    )
```

#### Caching Strategies

```python
from dna_origami_ae.utils.cache import cache_sequence_analysis, cache_image_processing

# Cache expensive sequence analysis
@cache_sequence_analysis(ttl=3600)
def analyze_complex_sequence(sequence):
    # Expensive computation
    return analysis_result

# Cache image preprocessing
@cache_image_processing(ttl=1800)
def preprocess_image(image):
    # Image processing pipeline
    return processed_image
```

### Testing & Validation

#### Unit Testing

```python
import pytest
from dna_origami_ae import DNASequence, ValidationError

def test_dna_sequence_validation():
    # Valid sequence
    seq = DNASequence("ATGCATGC", "test")
    assert len(seq) == 8
    
    # Invalid sequence
    with pytest.raises(ValidationError):
        DNASequence("ATGCX", "invalid")

def test_encoding_round_trip():
    from dna_origami_ae import DNAEncoder
    import numpy as np
    
    # Create test image
    image_array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    image = ImageData.from_array(image_array, "test")
    
    # Encode and decode
    encoder = DNAEncoder()
    sequences = encoder.encode_image(image)
    decoded = encoder.decode_image(sequences, 32, 32)
    
    # Check reconstruction quality
    mse = image.calculate_mse(decoded)
    assert mse < 100  # Reasonable reconstruction
```

#### Integration Testing

```python
def test_full_pipeline():
    """Test complete image -> DNA -> origami -> simulation -> decode pipeline."""
    
    # Setup
    image = create_test_image()
    encoder = DNAEncoder()
    designer = OrigamiDesigner()
    simulator = OrigamiSimulator()
    decoder = TransformerDecoder()
    
    # Execute pipeline
    sequences = encoder.encode_image(image)
    structure = designer.design_from_image_data(sequences)
    result = simulator.simulate_folding(structure)
    decoded_image = decoder.decode_structure(result.final_structure)
    
    # Validate results
    assert len(sequences) > 0
    assert structure.validate_design()[0]
    assert result.success
    assert decoded_image.metadata.width == image.metadata.width
```

### Deployment

#### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Set environment variables
ENV DNA_ORIGAMI_MAX_WORKERS=4
ENV DNA_ORIGAMI_CACHE_SIZE_MB=500
ENV DNA_ORIGAMI_LOG_LEVEL=INFO

# Expose port for API
EXPOSE 8000

# Run the application
CMD ["python", "-m", "dna_origami_ae.api.server"]
```

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dna-origami-ae
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dna-origami-ae
  template:
    metadata:
      labels:
        app: dna-origami-ae
    spec:
      containers:
      - name: dna-origami-ae
        image: dna-origami-ae:latest
        ports:
        - containerPort: 8000
        env:
        - name: DNA_ORIGAMI_MAX_WORKERS
          value: "4"
        - name: DNA_ORIGAMI_CACHE_SIZE_MB
          value: "1000"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### API Endpoints (REST)

#### Encoding Endpoints

```
POST /api/v1/encode/image
Content-Type: multipart/form-data

Parameters:
- image: File (required)
- error_correction: string (optional, default: "reed_solomon")
- chunk_size: integer (optional, default: 200)

Response:
{
  "sequences": ["ATGCATGC...", "CGTATGC..."],
  "metadata": {
    "original_size": 1024,
    "encoded_size": 2048,
    "compression_ratio": 0.5
  }
}
```

#### Design Endpoints

```
POST /api/v1/design/origami
Content-Type: application/json

Body:
{
  "sequences": ["ATGCATGC...", "CGTATGC..."],
  "target_shape": "square",
  "dimensions": [100, 100]
}

Response:
{
  "structure_id": "struct_123",
  "name": "square_origami_0",
  "staple_count": 42,
  "estimated_yield": 0.85
}
```

#### Simulation Endpoints

```
POST /api/v1/simulate/folding
Content-Type: application/json

Body:
{
  "structure_id": "struct_123",
  "force_field": "oxDNA2",
  "temperature": 300.0,
  "time_steps": 1000000
}

Response:
{
  "simulation_id": "sim_456",
  "status": "completed",
  "computation_time": 120.5,
  "quality_score": 0.92
}
```

### WebSocket API

#### Real-time Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/simulation/sim_456');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'progress':
            console.log(`Progress: ${data.progress}%`);
            break;
        case 'completed':
            console.log('Simulation completed');
            break;
        case 'error':
            console.error('Simulation failed:', data.error);
            break;
    }
};
```

---

This API documentation provides comprehensive coverage of the DNA Origami AutoEncoder system. For more detailed examples and tutorials, see the [full documentation](https://dna-origami-ae.readthedocs.io).