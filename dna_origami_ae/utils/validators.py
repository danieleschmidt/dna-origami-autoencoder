"""Validation utilities for DNA origami autoencoder."""

import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from ..models.dna_sequence import DNASequence
from ..models.origami_structure import OrigamiStructure
from ..models.image_data import ImageData
from ..models.simulation_data import StructureCoordinates


def validate_dna_sequence(sequence: Union[str, DNASequence]) -> Tuple[bool, List[str]]:
    """Validate DNA sequence format and content."""
    errors = []
    
    # Convert to string if DNASequence object
    seq_str = sequence.sequence if isinstance(sequence, DNASequence) else sequence
    
    if not seq_str:
        errors.append("Empty sequence")
        return False, errors
    
    # Check for valid DNA bases
    if not re.match(r'^[ATGC]+$', seq_str.upper()):
        errors.append("Sequence contains invalid characters (only A, T, G, C allowed)")
    
    # Check length
    if len(seq_str) < 10:
        errors.append("Sequence too short (minimum 10 bases)")
    
    if len(seq_str) > 10000:
        errors.append("Sequence too long (maximum 10000 bases)")
    
    return len(errors) == 0, errors


def validate_image_data(image: ImageData) -> Tuple[bool, List[str]]:
    """Validate image data format and properties."""
    errors = []
    
    # Check dimensions
    if image.metadata.width <= 0 or image.metadata.height <= 0:
        errors.append("Invalid image dimensions")
    
    # Check data type
    if image.data.dtype not in [np.uint8, np.uint16]:
        errors.append(f"Unsupported image data type: {image.data.dtype}")
    
    # Check data range
    if image.metadata.bit_depth == 8:
        if np.any(image.data < 0) or np.any(image.data > 255):
            errors.append("8-bit image values outside valid range [0, 255]")
    elif image.metadata.bit_depth == 16:
        if np.any(image.data < 0) or np.any(image.data > 65535):
            errors.append("16-bit image values outside valid range [0, 65535]")
    
    # Check consistency
    expected_shape = (image.metadata.height, image.metadata.width)
    if image.metadata.channels > 1:
        expected_shape = (*expected_shape, image.metadata.channels)
    
    if image.data.shape != expected_shape:
        errors.append(f"Image shape {image.data.shape} doesn't match metadata {expected_shape}")
    
    return len(errors) == 0, errors


def validate_origami_structure(structure: OrigamiStructure) -> Tuple[bool, List[str]]:
    """Validate origami structure design."""
    errors = []
    
    # Validate scaffold
    scaffold_valid, scaffold_errors = validate_dna_sequence(structure.scaffold.sequence)
    if not scaffold_valid:
        errors.extend([f"Scaffold: {error}" for error in scaffold_errors])
    
    # Validate staples
    for i, staple in enumerate(structure.staples):
        staple_valid, staple_errors = validate_dna_sequence(staple.sequence)
        if not staple_valid:
            errors.extend([f"Staple {i}: {error}" for error in staple_errors])
        
        # Check staple properties
        if staple.length < 10 or staple.length > 100:
            errors.append(f"Staple {i}: Length {staple.length} outside recommended range")
    
    # Check for overlapping staples (simplified)
    for i in range(len(structure.staples)):
        for j in range(i + 1, len(structure.staples)):
            if structure.staples[i].overlaps_with(structure.staples[j]):
                errors.append(f"Staples {i} and {j} overlap")
    
    return len(errors) == 0, errors


def validate_structure_coordinates(coords: StructureCoordinates) -> Tuple[bool, List[str]]:
    """Validate 3D structure coordinates."""
    errors = []
    
    # Check array dimensions
    if coords.positions.ndim != 2 or coords.positions.shape[1] != 3:
        errors.append("Positions must be Nx3 array")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(coords.positions)):
        errors.append("Positions contain NaN values")
    
    if np.any(np.isinf(coords.positions)):
        errors.append("Positions contain infinite values")
    
    # Check atom types
    if len(coords.atom_types) != coords.positions.shape[0]:
        errors.append("Number of atom types doesn't match number of positions")
    
    # Check connectivity if present
    if coords.connectivity is not None:
        if coords.connectivity.ndim != 2 or coords.connectivity.shape[1] != 2:
            errors.append("Connectivity must be Nx2 array")
        
        max_atom_index = coords.positions.shape[0] - 1
        if np.any(coords.connectivity < 0) or np.any(coords.connectivity > max_atom_index):
            errors.append("Connectivity contains invalid atom indices")
    
    return len(errors) == 0, errors


def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = True,
                      extensions: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """Validate file path."""
    errors = []
    path = Path(file_path)
    
    if must_exist and not path.exists():
        errors.append(f"File does not exist: {file_path}")
    
    if extensions:
        if path.suffix.lower() not in [ext.lower() for ext in extensions]:
            errors.append(f"Invalid file extension. Expected: {extensions}")
    
    # Check parent directory exists (if creating new file)
    if not must_exist and not path.parent.exists():
        errors.append(f"Parent directory does not exist: {path.parent}")
    
    return len(errors) == 0, errors


def validate_parameters(params: Dict[str, Any], 
                       param_specs: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validate parameter dictionary against specifications."""
    errors = []
    
    for param_name, spec in param_specs.items():
        if 'required' in spec and spec['required'] and param_name not in params:
            errors.append(f"Required parameter missing: {param_name}")
            continue
        
        if param_name not in params:
            continue  # Optional parameter
        
        value = params[param_name]
        
        # Type validation
        if 'type' in spec:
            expected_type = spec['type']
            if not isinstance(value, expected_type):
                errors.append(f"Parameter {param_name}: expected {expected_type.__name__}, got {type(value).__name__}")
                continue
        
        # Range validation
        if 'min' in spec and value < spec['min']:
            errors.append(f"Parameter {param_name}: value {value} below minimum {spec['min']}")
        
        if 'max' in spec and value > spec['max']:
            errors.append(f"Parameter {param_name}: value {value} above maximum {spec['max']}")
        
        # Choices validation
        if 'choices' in spec and value not in spec['choices']:
            errors.append(f"Parameter {param_name}: invalid choice {value}. Options: {spec['choices']}")
    
    return len(errors) == 0, errors


def validate_encoding_parameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate DNA encoding parameters."""
    spec = {
        'bits_per_base': {'type': int, 'min': 1, 'max': 4, 'required': True},
        'error_correction': {'type': str, 'choices': ['none', 'parity', 'reed_solomon'], 'required': False},
        'chunk_size': {'type': int, 'min': 10, 'max': 1000, 'required': False},
        'redundancy': {'type': float, 'min': 0.0, 'max': 1.0, 'required': False}
    }
    
    return validate_parameters(params, spec)


def validate_simulation_parameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate molecular dynamics simulation parameters."""
    spec = {
        'temperature': {'type': float, 'min': 200.0, 'max': 400.0, 'required': True},
        'time_steps': {'type': int, 'min': 1000, 'max': 10000000, 'required': True},
        'time_step_size': {'type': float, 'min': 0.001, 'max': 0.01, 'required': False},
        'save_interval': {'type': int, 'min': 1, 'max': 100000, 'required': False},
        'force_field': {'type': str, 'choices': ['oxDNA', 'oxDNA2'], 'required': False}
    }
    
    return validate_parameters(params, spec)


def validate_decoder_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate transformer decoder configuration."""
    spec = {
        'input_dim': {'type': int, 'min': 1, 'max': 10, 'required': True},
        'hidden_dim': {'type': int, 'min': 64, 'max': 2048, 'required': True},
        'num_heads': {'type': int, 'min': 1, 'max': 32, 'required': True},
        'num_layers': {'type': int, 'min': 1, 'max': 24, 'required': True},
        'dropout': {'type': float, 'min': 0.0, 'max': 0.9, 'required': False},
        'max_sequence_length': {'type': int, 'min': 100, 'max': 50000, 'required': False}
    }
    
    return validate_parameters(config, spec)


def validate_array_dimensions(array: np.ndarray, 
                            expected_shape: Optional[Tuple[int, ...]] = None,
                            min_dims: Optional[int] = None,
                            max_dims: Optional[int] = None) -> Tuple[bool, List[str]]:
    """Validate numpy array dimensions."""
    errors = []
    
    if expected_shape and array.shape != expected_shape:
        errors.append(f"Array shape {array.shape} doesn't match expected {expected_shape}")
    
    if min_dims and array.ndim < min_dims:
        errors.append(f"Array has {array.ndim} dimensions, minimum {min_dims} required")
    
    if max_dims and array.ndim > max_dims:
        errors.append(f"Array has {array.ndim} dimensions, maximum {max_dims} allowed")
    
    return len(errors) == 0, errors


def validate_model_input(data: Any, expected_type: type) -> Tuple[bool, List[str]]:
    """Validate input data for machine learning models."""
    errors = []
    
    if not isinstance(data, expected_type):
        errors.append(f"Expected {expected_type.__name__}, got {type(data).__name__}")
        return False, errors
    
    if isinstance(data, np.ndarray):
        # Check for NaN or infinite values
        if np.any(np.isnan(data)):
            errors.append("Input contains NaN values")
        
        if np.any(np.isinf(data)):
            errors.append("Input contains infinite values")
        
        # Check data range for image data
        if data.dtype == np.uint8 and (np.any(data < 0) or np.any(data > 255)):
            errors.append("uint8 data outside valid range [0, 255]")
    
    return len(errors) == 0, errors


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, errors: List[str]):
        self.errors = errors
        super().__init__(f"{message}: {'; '.join(errors)}")


def assert_valid_dna_sequence(sequence: Union[str, DNASequence]) -> None:
    """Assert that DNA sequence is valid, raise ValidationError if not."""
    is_valid, errors = validate_dna_sequence(sequence)
    if not is_valid:
        raise ValidationError("Invalid DNA sequence", errors)


def assert_valid_image_data(image: ImageData) -> None:
    """Assert that image data is valid, raise ValidationError if not."""
    is_valid, errors = validate_image_data(image)
    if not is_valid:
        raise ValidationError("Invalid image data", errors)


def assert_valid_origami_structure(structure: OrigamiStructure) -> None:
    """Assert that origami structure is valid, raise ValidationError if not."""
    is_valid, errors = validate_origami_structure(structure)
    if not is_valid:
        raise ValidationError("Invalid origami structure", errors)


def get_validation_summary(validation_results: List[Tuple[str, bool, List[str]]]) -> Dict[str, Any]:
    """Get summary of multiple validation results."""
    total_validations = len(validation_results)
    successful_validations = sum(1 for _, is_valid, _ in validation_results if is_valid)
    
    all_errors = []
    failed_items = []
    
    for item_name, is_valid, errors in validation_results:
        if not is_valid:
            failed_items.append(item_name)
            all_errors.extend([f"{item_name}: {error}" for error in errors])
    
    return {
        'total_items': total_validations,
        'successful_items': successful_validations,
        'failed_items': len(failed_items),
        'success_rate': successful_validations / total_validations if total_validations > 0 else 0.0,
        'failed_item_names': failed_items,
        'all_errors': all_errors
    }