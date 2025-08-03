"""Helper utilities for DNA origami autoencoder."""

import time
import hashlib
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from functools import wraps

from ..models.dna_sequence import DNASequence
from ..models.origami_structure import OrigamiStructure
from ..models.image_data import ImageData


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper


def progress_bar(iterable, description: str = "", total: Optional[int] = None):
    """Simple text-based progress bar."""
    if total is None:
        total = len(iterable) if hasattr(iterable, '__len__') else None
    
    start_time = time.time()
    
    for i, item in enumerate(iterable):
        yield item
        
        if total:
            percent = (i + 1) / total * 100
            elapsed = time.time() - start_time
            eta = elapsed * (total - i - 1) / (i + 1) if i > 0 else 0
            
            bar_length = 30
            filled_length = int(bar_length * (i + 1) // total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print(f'\r{description} |{bar}| {percent:.1f}% ({i+1}/{total}) ETA: {eta:.1f}s', end='')
        else:
            print(f'\r{description} Processing item {i+1}...', end='')
    
    print()  # New line when complete


def calculate_checksum(data: Union[str, bytes, np.ndarray]) -> str:
    """Calculate SHA-256 checksum of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    
    return hashlib.sha256(data).hexdigest()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default value if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def normalize_array(array: np.ndarray, 
                   method: str = "min_max",
                   target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Normalize array using specified method."""
    if method == "min_max":
        min_val = np.min(array)
        max_val = np.max(array)
        
        if max_val > min_val:
            normalized = (array - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(array)
    
    elif method == "z_score":
        mean_val = np.mean(array)
        std_val = np.std(array)
        
        if std_val > 0:
            normalized = (array - mean_val) / std_val
            # Clip to reasonable range and normalize to [0, 1]
            normalized = np.clip(normalized, -3, 3)
            normalized = (normalized + 3) / 6
        else:
            normalized = np.ones_like(array) * 0.5
    
    elif method == "unit_vector":
        norm = np.linalg.norm(array)
        normalized = array / norm if norm > 0 else array
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Scale to target range
    if target_range != (0.0, 1.0):
        min_target, max_target = target_range
        normalized = normalized * (max_target - min_target) + min_target
    
    return normalized


def interpolate_coordinates(start_coords: np.ndarray, 
                          end_coords: np.ndarray,
                          num_steps: int) -> np.ndarray:
    """Interpolate between two sets of coordinates."""
    if start_coords.shape != end_coords.shape:
        raise ValueError("Start and end coordinates must have same shape")
    
    interpolated = np.zeros((num_steps, *start_coords.shape))
    
    for i in range(num_steps):
        alpha = i / (num_steps - 1) if num_steps > 1 else 0
        interpolated[i] = (1 - alpha) * start_coords + alpha * end_coords
    
    return interpolated


def create_grid_coordinates(width: int, height: int, 
                          spacing: float = 1.0,
                          center: bool = True) -> np.ndarray:
    """Create 2D grid coordinates."""
    x = np.arange(width) * spacing
    y = np.arange(height) * spacing
    
    if center:
        x -= np.mean(x)
        y -= np.mean(y)
    
    xx, yy = np.meshgrid(x, y)
    coordinates = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    return coordinates


def estimate_memory_usage(array_shape: Tuple[int, ...], 
                         dtype: np.dtype = np.float64) -> Dict[str, float]:
    """Estimate memory usage for numpy array."""
    num_elements = np.prod(array_shape)
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = num_elements * bytes_per_element
    
    return {
        'bytes': total_bytes,
        'kb': total_bytes / 1024,
        'mb': total_bytes / (1024 ** 2),
        'gb': total_bytes / (1024 ** 3)
    }


def batch_process(items: List[Any], 
                 process_func: Callable,
                 batch_size: int = 32,
                 description: str = "Processing") -> List[Any]:
    """Process items in batches."""
    results = []
    
    for i in progress_bar(range(0, len(items), batch_size), description):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
    
    return results


def create_backup(file_path: Union[str, Path], 
                 backup_suffix: str = ".bak") -> Path:
    """Create backup of file."""
    original_path = Path(file_path)
    backup_path = original_path.with_suffix(original_path.suffix + backup_suffix)
    
    if original_path.exists():
        import shutil
        shutil.copy2(original_path, backup_path)
        return backup_path
    else:
        raise FileNotFoundError(f"Original file not found: {original_path}")


def merge_dictionaries(*dicts: Dict[str, Any], 
                      strategy: str = "update") -> Dict[str, Any]:
    """Merge multiple dictionaries with specified strategy."""
    if not dicts:
        return {}
    
    if strategy == "update":
        result = {}
        for d in dicts:
            result.update(d)
        return result
    
    elif strategy == "deep_merge":
        result = {}
        for d in dicts:
            for key, value in d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dictionaries(result[key], value, strategy="deep_merge")
                else:
                    result[key] = value
        return result
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def flatten_dict(nested_dict: Dict[str, Any], 
                separator: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionary."""
    def _flatten(obj, parent_key=''):
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                
                if isinstance(value, dict):
                    items.extend(_flatten(value, new_key).items())
                else:
                    items.append((new_key, value))
        else:
            items.append((parent_key, obj))
        
        return dict(items)
    
    return _flatten(nested_dict)


def unflatten_dict(flat_dict: Dict[str, Any], 
                  separator: str = ".") -> Dict[str, Any]:
    """Unflatten dictionary."""
    result = {}
    
    for key, value in flat_dict.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_json(data: Any, file_path: Union[str, Path], 
             indent: int = 2, ensure_ascii: bool = False) -> None:
    """Save data to JSON file with numpy type conversion."""
    converted_data = convert_numpy_types(data)
    
    with open(file_path, 'w') as f:
        json.dump(converted_data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_json(file_path: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_directory_structure(base_path: Union[str, Path], 
                             structure: Dict[str, Any]) -> None:
    """Create directory structure from nested dictionary."""
    base_path = Path(base_path)
    
    def _create_structure(current_path: Path, struct: Dict[str, Any]):
        for name, content in struct.items():
            new_path = current_path / name
            
            if isinstance(content, dict):
                new_path.mkdir(parents=True, exist_ok=True)
                _create_structure(new_path, content)
            else:
                # Create file with content
                new_path.parent.mkdir(parents=True, exist_ok=True)
                if content is not None:
                    with open(new_path, 'w') as f:
                        f.write(str(content))
                else:
                    new_path.touch()
    
    _create_structure(base_path, structure)


def calculate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """Calculate pairwise distance matrix for coordinates."""
    n_points = coordinates.shape[0]
    distances = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def find_nearest_neighbors(coordinates: np.ndarray, 
                         k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors for each point."""
    distances = calculate_distance_matrix(coordinates)
    
    # Sort indices by distance (excluding self)
    neighbor_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    neighbor_distances = np.sort(distances, axis=1)[:, 1:k+1]
    
    return neighbor_indices, neighbor_distances


def calculate_structure_statistics(coordinates: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics for 3D structure."""
    center = np.mean(coordinates, axis=0)
    distances_from_center = np.linalg.norm(coordinates - center, axis=1)
    
    return {
        'center_x': float(center[0]),
        'center_y': float(center[1]),
        'center_z': float(center[2]) if coordinates.shape[1] > 2 else 0.0,
        'radius_of_gyration': float(np.sqrt(np.mean(distances_from_center ** 2))),
        'max_distance_from_center': float(np.max(distances_from_center)),
        'min_distance_from_center': float(np.min(distances_from_center)),
        'bounding_box_volume': float(np.prod(np.max(coordinates, axis=0) - np.min(coordinates, axis=0)))
    }


def format_time_duration(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"


def format_memory_size(bytes_size: float) -> str:
    """Format memory size in human-readable format."""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    
    size = float(bytes_size)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"


def chunked_iterator(iterable, chunk_size: int):
    """Iterate over data in chunks."""
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


# Import itertools for chunked_iterator
import itertools


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.description} completed in {format_time_duration(duration)}")
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time if timer has finished."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class MemoryTracker:
    """Track memory usage during execution."""
    
    def __init__(self):
        self.start_memory = None
        self.peak_memory = None
        
    def start(self):
        """Start memory tracking."""
        import psutil
        process = psutil.Process()
        self.start_memory = process.memory_info().rss
        self.peak_memory = self.start_memory
    
    def update(self):
        """Update peak memory usage."""
        import psutil
        process = psutil.Process()
        current_memory = process.memory_info().rss
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def get_usage(self) -> Dict[str, str]:
        """Get memory usage statistics."""
        if self.start_memory is None:
            return {'error': 'Memory tracking not started'}
        
        self.update()
        
        return {
            'start_memory': format_memory_size(self.start_memory),
            'peak_memory': format_memory_size(self.peak_memory),
            'memory_increase': format_memory_size(self.peak_memory - self.start_memory)
        }