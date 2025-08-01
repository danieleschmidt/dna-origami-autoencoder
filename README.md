# DNA-Origami-AutoEncoder

A groundbreaking "wet-lab ML" framework that encodes 8-bit images into self-assembling DNA origami structures and decodes them using transformer models trained on simulated base-pair kinetics. This project bridges synthetic biology and machine learning to demonstrate information storage and retrieval through biological self-assembly.

## Overview

DNA-Origami-AutoEncoder represents digital information as DNA sequences that fold into specific origami patterns. These patterns can be "read" by a neural network trained on molecular dynamics simulations, creating a biological autoencoder system. This opens new possibilities for ultra-dense data storage, biocomputing, and molecular information processing.

## Key Features

- **Image to DNA Encoding**: Convert 8-bit grayscale images to DNA sequences
- **Origami Design**: Automated scaffold/staple design for target shapes
- **Molecular Simulation**: GPU-accelerated DNA folding dynamics
- **Neural Decoding**: Transformer-based pattern recognition from AFM-like data
- **Wet-Lab Protocols**: Ready-to-use protocols for DNA synthesis and imaging
- **Error Correction**: Bio-compatible error correction codes

## Installation

```bash
# Basic installation
pip install dna-origami-autoencoder

# With molecular dynamics support
pip install dna-origami-autoencoder[md]

# With wet-lab protocol generation
pip install dna-origami-autoencoder[wetlab]

# Full installation
pip install dna-origami-autoencoder[full]

# Development
git clone https://github.com/yourusername/dna-origami-autoencoder
cd dna-origami-autoencoder
pip install -e ".[dev]"
```

## Quick Start

### Encode an Image to DNA

```python
from dna_origami_ae import DNAEncoder, OrigamiDesigner
import numpy as np
from PIL import Image

# Load 8-bit grayscale image
image = Image.open('mnist_digit.png').convert('L')
image_array = np.array(image.resize((32, 32)))

# Initialize encoder
encoder = DNAEncoder(
    bits_per_base=2,  # 00=A, 01=T, 10=G, 11=C
    error_correction='reed_solomon'
)

# Encode image to DNA sequence
dna_sequence = encoder.encode_image(image_array)
print(f"DNA sequence length: {len(dna_sequence)} bases")

# Design origami structure
designer = OrigamiDesigner(
    scaffold_length=7249,  # M13mp18 standard
    staple_length=32
)

origami = designer.design_origami(
    dna_sequence,
    target_shape='square',
    dimensions=(100, 100)  # nm
)

# Export staple sequences
origami.export_sequences('origami_design.csv')
```

### Simulate Folding and Decode

```python
from dna_origami_ae import OrigamiSimulator, TransformerDecoder

# Simulate DNA origami folding
simulator = OrigamiSimulator(
    force_field='oxDNA2',
    temperature=300,  # Kelvin
    salt_concentration=0.5  # Molar
)

# Run molecular dynamics
trajectory = simulator.simulate_folding(
    origami,
    time_steps=1000000,
    gpu_acceleration=True
)

# Extract final structure
final_structure = trajectory.get_final_structure()

# Decode with neural network
decoder = TransformerDecoder.from_pretrained('dna-origami-decoder-v1')
reconstructed_image = decoder.decode_structure(final_structure)

# Calculate reconstruction accuracy
mse = np.mean((image_array - reconstructed_image) ** 2)
print(f"Reconstruction MSE: {mse:.4f}")
```

## Architecture

```
dna-origami-autoencoder/
├── dna_origami_ae/
│   ├── encoding/
│   │   ├── image_encoder.py     # Image to DNA conversion
│   │   ├── error_correction.py  # Reed-Solomon, Hamming codes
│   │   └── compression.py        # DNA-compatible compression
│   ├── design/
│   │   ├── origami_designer.py   # Scaffold/staple design
│   │   ├── shape_library.py      # Pre-designed shapes
│   │   └── sequence_optimizer.py # Minimize secondary structures
│   ├── simulation/
│   │   ├── md_simulator.py       # Molecular dynamics
│   │   ├── oxdna_interface.py    # oxDNA integration
│   │   └── coarse_grain.py       # Fast approximations
│   ├── decoding/
│   │   ├── transformer_model.py  # Neural decoder
│   │   ├── attention_layers.py   # Custom attention for 3D
│   │   └── training.py           # Self-supervised training
│   ├── wetlab/
│   │   ├── protocol_generator.py # Lab protocols
│   │   ├── plate_designer.py     # 96-well plate layouts
│   │   └── imaging_processor.py  # AFM image processing
│   └── analysis/
│       ├── structure_metrics.py   # Folding quality metrics
│       ├── decoder_analysis.py    # Model interpretability
│       └── error_analysis.py      # Error propagation study
├── experiments/
├── protocols/
└── models/
```

## DNA Encoding Schemes

### Efficient Base-4 Encoding

```python
from dna_origami_ae.encoding import Base4Encoder, BiologicalConstraints

# Configure biological constraints
constraints = BiologicalConstraints(
    gc_content=(0.4, 0.6),  # 40-60% GC content
    max_homopolymer=4,       # No more than 4 identical bases
    avoid_sequences=['GGGG', 'CCCC'],  # Avoid G-quadruplexes
    melting_temp_range=(55, 65)  # Celsius
)

# Create constrained encoder
encoder = Base4Encoder(constraints)

# Encode with biological compatibility
data = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
dna_sequences = encoder.encode_with_constraints(data)

# Verify constraints
for seq in dna_sequences:
    assert constraints.validate(seq), "Constraint violation!"
```

### Error Correction

```python
from dna_origami_ae.encoding import DNAErrorCorrection

# DNA-optimized error correction
error_corrector = DNAErrorCorrection(
    method='dna_reed_solomon',
    redundancy=0.3,  # 30% redundancy
    burst_error_capability=10  # Handle 10-base deletions
)

# Add error correction
protected_dna = error_corrector.encode(dna_sequence)

# Simulate sequencing errors
noisy_dna = simulate_sequencing_errors(
    protected_dna,
    substitution_rate=0.01,
    deletion_rate=0.005,
    insertion_rate=0.005
)

# Recover original
recovered_dna = error_corrector.decode(noisy_dna)
accuracy = compare_sequences(dna_sequence, recovered_dna)
print(f"Recovery accuracy: {accuracy:.2%}")
```

## Origami Design

### Custom Shape Design

```python
from dna_origami_ae.design import ShapeDesigner, RoutingAlgorithm

# Design custom origami shape
shape_designer = ShapeDesigner()

# Define target shape as 2D pixels
target_shape = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]) * 20  # 20nm per pixel

# Route scaffold through shape
routing = RoutingAlgorithm(
    method='honeycomb_lattice',
    crossover_spacing=21  # bases
)

scaffold_path = routing.route_scaffold(target_shape)

# Place staples
staple_designer = shape_designer.design_staples(
    scaffold_path,
    staple_length=32,
    overlap=16
)

# Optimize for stability
staple_designer.optimize_sequences(
    minimize_secondary_structure=True,
    maximize_binding_strength=True
)
```

### 3D Origami Structures

```python
from dna_origami_ae.design import Origami3D

# Create 3D box structure
box_designer = Origami3D()

box = box_designer.create_box(
    dimensions=(50, 50, 50),  # nm
    wall_thickness=2.5,       # nm (one helix)
    lid_type='hinged'
)

# Embed data in 3D structure
data_embedding = box.embed_data(
    dna_sequence,
    embedding_method='surface_pattern'
)

# Generate assembly instructions
assembly = box.generate_assembly_protocol(
    annealing_curve='standard',
    magnesium_concentration=12.5  # mM
)
```

## Molecular Dynamics Simulation

### GPU-Accelerated Folding

```python
from dna_origami_ae.simulation import GPUSimulator, ForceField

# Setup GPU simulation
gpu_sim = GPUSimulator(
    device='cuda:0',
    precision='mixed',  # FP16 for speed, FP32 for forces
    batch_size=10       # Simulate 10 structures in parallel
)

# Configure force field
force_field = ForceField.oxDNA2(
    temperature=300,
    salt_concentration=0.5,
    external_forces={'stretching': 0.1}  # pN
)

# Run parallel simulations
trajectories = gpu_sim.run_batch(
    origami_designs,
    force_field=force_field,
    time_steps=1e7,
    save_interval=1e4
)

# Analyze folding kinetics
folding_curves = gpu_sim.analyze_folding_kinetics(trajectories)
```

### Coarse-Grained Simulation

```python
from dna_origami_ae.simulation import CoarseGrainedModel

# Fast approximate simulation
cg_model = CoarseGrainedModel(
    resolution='nucleotide',  # vs 'base-pair' or 'domain'
    implicit_solvent=True
)

# Rapid structure prediction
predicted_structure = cg_model.predict_structure(
    origami,
    simulation_time=1e-3  # seconds (vs hours for all-atom)
)

# Validate against all-atom
validation_score = cg_model.validate_against_all_atom(
    predicted_structure,
    reference_trajectory
)
```

## Neural Network Decoder

### Transformer Architecture

```python
from dna_origami_ae.decoding import OrigamiTransformer

# Custom transformer for 3D structures
model = OrigamiTransformer(
    input_dim=3,           # xyz coordinates
    hidden_dim=512,
    num_heads=8,
    num_layers=12,
    position_encoding='3d_sinusoidal',
    attention_type='sparse_3d'
)

# Train on simulated data
from dna_origami_ae.training import SelfSupervisedTrainer

trainer = SelfSupervisedTrainer(
    model,
    learning_rate=1e-4,
    batch_size=32
)

# Self-supervised pretraining
trainer.pretrain(
    trajectory_dataset,
    tasks=['coordinate_prediction', 'topology_matching'],
    epochs=100
)

# Fine-tune for decoding
trainer.finetune(
    paired_structure_image_data,
    epochs=50
)
```

### Attention Visualization

```python
from dna_origami_ae.analysis import AttentionAnalyzer

analyzer = AttentionAnalyzer(model)

# Visualize what the model "sees"
attention_maps = analyzer.get_attention_maps(
    structure,
    layer=8,
    head=3
)

# 3D visualization
analyzer.visualize_3d_attention(
    structure,
    attention_maps,
    output_file='attention_3d.html'
)

# Identify important structural motifs
motifs = analyzer.extract_attended_motifs(
    threshold=0.8,
    min_size=10
)
```

## Wet-Lab Protocols

### Automated Protocol Generation

```python
from dna_origami_ae.wetlab import ProtocolGenerator, LabEquipment

# Generate complete lab protocol
protocol_gen = ProtocolGenerator(
    equipment=LabEquipment.STANDARD_BIO_LAB,
    safety_level='BSL1'
)

protocol = protocol_gen.generate_protocol(
    origami_design,
    scale='96_well_plate',
    replicates=3
)

# Export to various formats
protocol.to_pdf('origami_protocol.pdf')
protocol.to_opentrons('origami_protocol.py')  # For robot
protocol.to_benchling()  # Direct integration

# Generate shopping list
materials = protocol.generate_materials_list(
    include_prices=True,
    vendors=['IDT', 'NEB', 'Sigma']
)
```

### AFM Image Processing

```python
from dna_origami_ae.wetlab import AFMProcessor

# Process experimental AFM images
afm_processor = AFMProcessor(
    pixel_size=2,  # nm/pixel
    flatten_method='polynomial',
    denoise_method='bilateral'
)

# Load and process AFM image
afm_image = afm_processor.load_image('origami_afm.tif')
processed = afm_processor.process(
    afm_image,
    remove_drift=True,
    enhance_edges=True
)

# Extract origami structures
structures = afm_processor.segment_origami(
    processed,
    expected_size=(100, 100),  # nm
    confidence_threshold=0.8
)

# Convert to model input format
model_input = afm_processor.to_model_input(structures[0])
decoded_image = decoder.decode_structure(model_input)
```

## Advanced Features

### Multi-Image Encoding

```python
from dna_origami_ae.encoding import MultiImageEncoder

# Encode multiple images in one origami
multi_encoder = MultiImageEncoder(
    arrangement='tiled',  # or 'layered', 'sequential'
    images_per_origami=4
)

# Encode image batch
images = [Image.open(f'image_{i}.png') for i in range(4)]
multi_origami = multi_encoder.encode_batch(
    images,
    origami_size=(200, 200),  # nm
    spacing=10  # nm between images
)

# Decode with position awareness
decoded_images = decoder.decode_multi(
    multi_origami,
    num_images=4,
    use_positional_encoding=True
)
```

### DNA Computing Integration

```python
from dna_origami_ae.computing import DNALogicGates

# Implement computation in DNA
logic_gates = DNALogicGates()

# Create XOR gate with origami
xor_gate = logic_gates.create_xor(
    input_strands=['INPUT_A', 'INPUT_B'],
    output_shape='fluorescent_pattern'
)

# Integrate with image encoding
computed_origami = encoder.encode_with_computation(
    image_array,
    computation=xor_gate,
    inputs={'INPUT_A': True, 'INPUT_B': False}
)
```

### Error Analysis

```python
from dna_origami_ae.analysis import ErrorPropagation

# Analyze error propagation
error_analyzer = ErrorPropagation()

# Simulate various error sources
error_sources = {
    'synthesis_errors': 0.001,      # 0.1% per base
    'folding_defects': 0.05,        # 5% misfolded
    'imaging_noise': 0.1,           # SNR = 10
    'model_uncertainty': 0.02       # 2% classification error
}

# Monte Carlo error analysis
error_results = error_analyzer.monte_carlo_analysis(
    pipeline=complete_pipeline,
    error_sources=error_sources,
    num_simulations=1000
)

# Plot error propagation
error_analyzer.plot_error_cascade(
    error_results,
    output_file='error_propagation.png'
)
```

## Performance Benchmarks

### Encoding Efficiency

```python
from dna_origami_ae.benchmarks import EncodingBenchmark

benchmark = EncodingBenchmark()

# Compare encoding schemes
results = benchmark.compare_encodings(
    test_images=mnist_subset,
    encoding_schemes=['base4', 'base3_balanced', 'goldman'],
    metrics=['compression_ratio', 'error_resilience', 'gc_content']
)

benchmark.plot_results(results, 'encoding_comparison.png')
```

### Decoder Accuracy

```python
from dna_origami_ae.benchmarks import DecoderBenchmark

# Evaluate decoder performance
decoder_bench = DecoderBenchmark(decoder_model)

accuracy_results = decoder_bench.evaluate(
    test_set=origami_test_set,
    noise_levels=[0, 0.1, 0.2, 0.5],
    metrics=['mse', 'ssim', 'perceptual_loss']
)

# Generate performance report
decoder_bench.generate_report(
    accuracy_results,
    include_failure_analysis=True
)
```

## Examples

### MNIST Digit Storage

```python
# Complete example: Store MNIST digit in DNA
from dna_origami_ae.examples import MNISTExample

example = MNISTExample()

# Load MNIST digit
digit_image = example.load_digit(label=7)

# Full pipeline
result = example.run_full_pipeline(
    digit_image,
    simulate_folding=True,
    add_noise=True,
    decode=True
)

print(f"Original label: 7")
print(f"Decoded label: {result.predicted_label}")
print(f"Confidence: {result.confidence:.2%}")
print(f"DNA length: {len(result.dna_sequence)} bases")
```

### QR Code in DNA

```python
from dna_origami_ae.examples import QRCodeExample

# Encode QR code in DNA origami
qr_example = QRCodeExample()

qr_origami = qr_example.create_qr_origami(
    data="https://github.com/yourusername/dna-origami-ae",
    error_correction_level='H',
    origami_size=(150, 150)  # nm
)

# Generate lab protocol
protocol = qr_example.generate_protocol(qr_origami)
```

## Future Directions

### In Vivo Folding

```python
from dna_origami_ae.research import InVivoFolder

# Design for cellular conditions
in_vivo = InVivoFolder(
    cell_type='E.coli',
    expression_system='T7',
    cellular_conditions={
        'temperature': 37,
        'pH': 7.4,
        'crowding_agents': True
    }
)

# Adapt design for in vivo
cellular_origami = in_vivo.adapt_design(
    origami_design,
    add_nuclear_localization=True,
    protect_from_nucleases=True
)
```

## Citation

```bibtex
@article{dna_origami_autoencoder,
  title={DNA-Origami-AutoEncoder: Self-Assembling Biological Information Storage},
  author={Your Name},
  journal={Nature Biotechnology},
  year={2025},
  doi={10.1038/nbt.2025.example}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Safety and Ethics

This project is for research purposes. Please follow all local regulations regarding synthetic biology and genetic engineering. See [SAFETY.md](SAFETY.md) for guidelines.

## Acknowledgments

- DNA origami pioneers including Paul Rothemund and William Shih
- The oxDNA team for simulation tools
- The synthetic biology community

## Resources

- [Documentation](https://dna-origami-ae.readthedocs.io)
- [Tutorials](https://github.com/yourusername/dna-origami-ae/tutorials)
- [Discord Community](https://discord.gg/dna-origami)
- [Wetlab Protocols Database](https://protocols.dna-origami-ae.org)
