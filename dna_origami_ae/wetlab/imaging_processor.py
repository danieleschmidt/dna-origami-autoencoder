"""AFM and microscopy image processing for DNA origami analysis."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from ..models.image_data import ImageData
from ..models.origami_structure import OrigamiStructure
from ..models.simulation_data import StructureCoordinates


class ImagingMode(Enum):
    """Microscopy imaging modes."""
    AFM_CONTACT = "afm_contact"
    AFM_TAPPING = "afm_tapping"
    AFM_NON_CONTACT = "afm_non_contact"
    FLUORESCENCE = "fluorescence"
    TEM = "transmission_em"
    SEM = "scanning_em"


@dataclass
class ImagingParameters:
    """Parameters for microscopy imaging."""
    mode: ImagingMode
    scan_size: Tuple[float, float] = (500.0, 500.0)  # nm
    resolution: Tuple[int, int] = (512, 512)  # pixels
    scan_rate: float = 1.0  # Hz
    setpoint: Optional[float] = None  # AFM setpoint
    z_range: float = 10.0  # nm
    pixel_size: float = 1.0  # nm/pixel


@dataclass
class DetectedStructure:
    """Information about a detected origami structure."""
    structure_id: int
    center_position: Tuple[float, float]  # nm
    dimensions: Tuple[float, float]  # width, height in nm
    area: float  # nm²
    perimeter: float  # nm
    shape_score: float  # 0-1, how well it matches expected shape
    brightness: float  # relative brightness
    defects_detected: List[str] = field(default_factory=list)
    folding_quality: float = 0.0  # 0-1 score
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # pixel coordinates


class AFMProcessor:
    """Process AFM images to extract origami structures."""
    
    def __init__(self, 
                 pixel_size: float = 2.0,  # nm/pixel
                 flatten_method: str = "polynomial",
                 denoise_method: str = "bilateral"):
        """Initialize AFM image processor."""
        self.pixel_size = pixel_size
        self.flatten_method = flatten_method
        self.denoise_method = denoise_method
        
        # Processing statistics
        self.processing_stats = {
            'images_processed': 0,
            'structures_detected': 0,
            'average_processing_time': 0.0,
            'detection_success_rate': 0.0
        }
    
    def load_image(self, image_path: str, format_hint: str = "auto") -> ImageData:
        """Load AFM image from file."""
        # In a real implementation, would handle various AFM file formats
        # (.spm, .nanoscope, .ibw, etc.)
        
        if format_hint == "auto":
            # Detect format from extension
            if image_path.endswith(('.spm', '.nanoscope')):
                return self._load_nanoscope_image(image_path)
            elif image_path.endswith('.ibw'):
                return self._load_igor_image(image_path)
            else:
                # Assume standard image format
                return self._load_standard_image(image_path)
        
        return self._load_standard_image(image_path)
    
    def _load_nanoscope_image(self, image_path: str) -> ImageData:
        """Load Nanoscope AFM image format."""
        # Simplified implementation
        # Real implementation would parse Nanoscope header and data
        
        # For demonstration, create synthetic AFM-like data
        data = np.random.randn(512, 512) * 2.0 + 5.0  # Height data in nm
        data = np.maximum(0, data)  # Ensure non-negative heights
        
        metadata = {
            'scan_size': (1000.0, 1000.0),  # nm
            'pixel_size': self.pixel_size,
            'z_range': 10.0,
            'scan_rate': 1.0,
            'imaging_mode': 'tapping_mode',
            'cantilever_freq': 300000.0,  # Hz
            'force_setpoint': 0.5  # nN
        }
        
        return ImageData.from_array(
            data,
            name=f"afm_{image_path.split('/')[-1]}",
            metadata_dict=metadata
        )
    
    def _load_igor_image(self, image_path: str) -> ImageData:
        """Load Igor Binary Wave (.ibw) format."""
        # Simplified implementation
        data = np.random.randn(256, 256) * 1.5 + 3.0
        data = np.maximum(0, data)
        
        return ImageData.from_array(
            data,
            name=f"igor_{image_path.split('/')[-1]}"
        )
    
    def _load_standard_image(self, image_path: str) -> ImageData:
        """Load standard image formats (TIFF, PNG, etc.)."""
        # Would use PIL or similar to load standard formats
        data = np.random.randn(512, 512) * 2.0 + 4.0
        data = np.maximum(0, data)
        
        return ImageData.from_array(
            data,
            name=f"std_{image_path.split('/')[-1]}"
        )
    
    def process(self, 
                afm_image: ImageData,
                remove_drift: bool = True,
                enhance_edges: bool = True,
                remove_noise: bool = True) -> ImageData:
        """Process AFM image with standard corrections."""
        import time
        start_time = time.time()
        
        try:
            processed_data = afm_image.data.copy()
            
            # Step 1: Remove drift and tilt
            if remove_drift:
                processed_data = self._remove_drift(processed_data)
            
            # Step 2: Flatten surface
            processed_data = self._flatten_surface(processed_data)
            
            # Step 3: Remove noise
            if remove_noise:
                processed_data = self._remove_noise(processed_data)
            
            # Step 4: Enhance edges if requested
            if enhance_edges:
                processed_data = self._enhance_edges(processed_data)
            
            # Create processed image
            processed_image = ImageData.from_array(
                processed_data,
                name=f"{afm_image.name}_processed"
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, success=True)
            
            return processed_image
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, success=False)
            raise ValueError(f"AFM processing failed: {e}")
    
    def _remove_drift(self, data: np.ndarray) -> np.ndarray:
        """Remove scanner drift from AFM data."""
        # Simple linear drift correction
        # Real implementation would use more sophisticated methods
        
        height, width = data.shape
        
        # Create coordinate arrays
        y, x = np.mgrid[0:height, 0:width]
        
        # Fit plane to data
        A = np.column_stack([x.flatten(), y.flatten(), np.ones(x.size)])
        coeffs, _, _, _ = np.linalg.lstsq(A, data.flatten(), rcond=None)
        
        # Create drift surface
        drift_surface = (coeffs[0] * x + coeffs[1] * y + coeffs[2])
        
        # Remove drift
        corrected_data = data - drift_surface
        
        return corrected_data
    
    def _flatten_surface(self, data: np.ndarray) -> np.ndarray:
        """Flatten surface using selected method."""
        if self.flatten_method == "polynomial":
            return self._polynomial_flatten(data)
        elif self.flatten_method == "line_by_line":
            return self._line_by_line_flatten(data)
        else:
            return data
    
    def _polynomial_flatten(self, data: np.ndarray, degree: int = 2) -> np.ndarray:
        """Polynomial surface flattening."""
        height, width = data.shape
        y, x = np.mgrid[0:height, 0:width]
        
        # Create polynomial basis functions
        basis = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                basis.append((x**i) * (y**j))
        
        # Stack basis functions
        A = np.column_stack([b.flatten() for b in basis])
        
        # Fit polynomial
        coeffs, _, _, _ = np.linalg.lstsq(A, data.flatten(), rcond=None)
        
        # Create polynomial surface
        poly_surface = np.zeros_like(data)
        for k, coeff in enumerate(coeffs):
            poly_surface += coeff * basis[k]
        
        return data - poly_surface
    
    def _line_by_line_flatten(self, data: np.ndarray) -> np.ndarray:
        """Line-by-line flattening."""
        flattened = data.copy()
        
        for i in range(data.shape[0]):
            line = data[i, :]
            # Fit line
            x = np.arange(len(line))
            coeffs = np.polyfit(x, line, 1)
            trend = np.polyval(coeffs, x)
            flattened[i, :] = line - trend
        
        return flattened
    
    def _remove_noise(self, data: np.ndarray) -> np.ndarray:
        """Remove noise from AFM data."""
        if self.denoise_method == "bilateral":
            return self._bilateral_filter(data)
        elif self.denoise_method == "gaussian":
            return self._gaussian_filter(data)
        else:
            return data
    
    def _bilateral_filter(self, data: np.ndarray, d: int = 5, 
                         sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """Bilateral filtering for edge-preserving noise reduction."""
        # Simplified bilateral filter implementation
        # Real implementation would use optimized algorithms
        
        height, width = data.shape
        filtered = np.zeros_like(data)
        
        pad = d // 2
        padded = np.pad(data, pad, mode='reflect')
        
        for i in range(height):
            for j in range(width):
                # Extract neighborhood
                neighborhood = padded[i:i+d, j:j+d]
                center_value = data[i, j]
                
                # Calculate spatial weights
                y, x = np.ogrid[-pad:pad+1, -pad:pad+1]
                spatial_weights = np.exp(-(x*x + y*y) / (2 * sigma_space**2))
                
                # Calculate intensity weights  
                intensity_diffs = neighborhood - center_value
                intensity_weights = np.exp(-(intensity_diffs**2) / (2 * sigma_color**2))
                
                # Combine weights
                weights = spatial_weights * intensity_weights
                
                # Apply weighted average
                filtered[i, j] = np.sum(weights * neighborhood) / np.sum(weights)
        
        return filtered
    
    def _gaussian_filter(self, data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian filtering."""
        # Simple Gaussian filter implementation
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(data, sigma=sigma)
    
    def _enhance_edges(self, data: np.ndarray) -> np.ndarray:
        """Enhance edges in AFM data."""
        # Calculate gradient magnitude
        grad_x = np.gradient(data, axis=1)
        grad_y = np.gradient(data, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Enhance edges by adding gradient information
        enhanced = data + 0.3 * grad_magnitude
        
        return enhanced
    
    def segment_origami(self,
                       processed_image: ImageData,
                       expected_size: Tuple[float, float] = (100.0, 100.0),
                       confidence_threshold: float = 0.8) -> List[DetectedStructure]:
        """Segment and detect origami structures in processed AFM image."""
        
        data = processed_image.data
        height_thresh = np.mean(data) + 2 * np.std(data)
        
        # Simple thresholding-based segmentation
        binary_mask = data > height_thresh
        
        # Find connected components
        structures = self._find_connected_components(binary_mask, data)
        
        # Filter by size and shape
        filtered_structures = []
        for struct in structures:
            if self._validate_structure_size(struct, expected_size):
                if struct.shape_score >= confidence_threshold:
                    filtered_structures.append(struct)
        
        self.processing_stats['structures_detected'] += len(filtered_structures)
        
        return filtered_structures
    
    def _find_connected_components(self, binary_mask: np.ndarray, 
                                 height_data: np.ndarray) -> List[DetectedStructure]:
        """Find connected components in binary mask."""
        # Simplified connected components analysis
        # Real implementation would use scipy.ndimage.label or similar
        
        structures = []
        labeled_mask = self._label_components(binary_mask)
        
        for label in range(1, np.max(labeled_mask) + 1):
            component_mask = (labeled_mask == label)
            
            # Calculate properties
            y_coords, x_coords = np.where(component_mask)
            
            if len(y_coords) < 50:  # Too small
                continue
            
            # Calculate center
            center_y = np.mean(y_coords) * self.pixel_size
            center_x = np.mean(x_coords) * self.pixel_size
            
            # Calculate dimensions
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            
            width = (max_x - min_x) * self.pixel_size
            height = (max_y - min_y) * self.pixel_size
            
            # Calculate other properties
            area = np.sum(component_mask) * (self.pixel_size ** 2)
            perimeter = self._calculate_perimeter(component_mask) * self.pixel_size
            brightness = np.mean(height_data[component_mask])
            
            # Shape analysis
            shape_score = self._analyze_shape(component_mask)
            folding_quality = self._assess_folding_quality(component_mask, height_data)
            
            structure = DetectedStructure(
                structure_id=label,
                center_position=(center_x, center_y),
                dimensions=(width, height),
                area=area,
                perimeter=perimeter,
                shape_score=shape_score,
                brightness=brightness,
                folding_quality=folding_quality,
                bounding_box=(min_x, min_y, max_x, max_y)
            )
            
            structures.append(structure)
        
        return structures
    
    def _label_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """Label connected components in binary mask."""
        # Simplified flood-fill labeling
        labeled = np.zeros_like(binary_mask, dtype=int)
        current_label = 1
        
        height, width = binary_mask.shape
        
        for i in range(height):
            for j in range(width):
                if binary_mask[i, j] and labeled[i, j] == 0:
                    # Start flood fill
                    self._flood_fill(binary_mask, labeled, i, j, current_label)
                    current_label += 1
        
        return labeled
    
    def _flood_fill(self, binary_mask: np.ndarray, labeled: np.ndarray,
                   start_i: int, start_j: int, label: int) -> None:
        """Flood fill algorithm for connected components."""
        stack = [(start_i, start_j)]
        height, width = binary_mask.shape
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= height or j < 0 or j >= width or
                not binary_mask[i, j] or labeled[i, j] != 0):
                continue
            
            labeled[i, j] = label
            
            # Add neighbors to stack
            stack.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
    
    def _calculate_perimeter(self, mask: np.ndarray) -> float:
        """Calculate perimeter of binary mask."""
        # Simple edge detection for perimeter
        edges = 0
        height, width = mask.shape
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                if mask[i, j]:
                    # Check if on boundary
                    neighbors = [
                        mask[i-1, j], mask[i+1, j],
                        mask[i, j-1], mask[i, j+1]
                    ]
                    if not all(neighbors):
                        edges += 1
        
        return float(edges)
    
    def _analyze_shape(self, mask: np.ndarray) -> float:
        """Analyze shape and return shape score (0-1)."""
        # Simple shape analysis based on aspect ratio and compactness
        y_coords, x_coords = np.where(mask)
        
        if len(y_coords) == 0:
            return 0.0
        
        # Calculate aspect ratio
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        aspect_ratio = min(width, height) / max(width, height)
        
        # Calculate compactness (area vs perimeter)
        area = np.sum(mask)
        perimeter = self._calculate_perimeter(mask)
        
        if perimeter == 0:
            compactness = 0
        else:
            compactness = 4 * np.pi * area / (perimeter ** 2)
        
        # Combine metrics
        shape_score = (aspect_ratio + min(1.0, compactness)) / 2.0
        
        return shape_score
    
    def _assess_folding_quality(self, mask: np.ndarray, height_data: np.ndarray) -> float:
        """Assess folding quality based on height uniformity."""
        heights = height_data[mask]
        
        if len(heights) == 0:
            return 0.0
        
        # Calculate height uniformity
        height_std = np.std(heights)
        height_mean = np.mean(heights)
        
        if height_mean == 0:
            return 0.0
        
        # Quality score based on relative height variation
        uniformity = 1.0 / (1.0 + height_std / height_mean)
        
        return min(1.0, uniformity)
    
    def _validate_structure_size(self, structure: DetectedStructure,
                               expected_size: Tuple[float, float]) -> bool:
        """Validate if structure size is reasonable."""
        expected_width, expected_height = expected_size
        actual_width, actual_height = structure.dimensions
        
        # Allow 50% variation from expected size
        width_ratio = actual_width / expected_width
        height_ratio = actual_height / expected_height
        
        return (0.5 <= width_ratio <= 2.0 and 0.5 <= height_ratio <= 2.0)
    
    def to_model_input(self, structure: DetectedStructure) -> StructureCoordinates:
        """Convert detected structure to model input format."""
        # Create synthetic 3D coordinates from 2D AFM data
        # In practice, would reconstruct 3D from height information
        
        # Generate points based on detected structure
        center_x, center_y = structure.center_position
        width, height = structure.dimensions
        
        # Create grid of points
        n_points = max(100, int(structure.area / 10))  # Point density
        
        # Random points within bounding box (simplified)
        np.random.seed(structure.structure_id)
        x_coords = np.random.uniform(center_x - width/2, center_x + width/2, n_points)
        y_coords = np.random.uniform(center_y - height/2, center_y + height/2, n_points)
        z_coords = np.random.uniform(0, 5.0, n_points)  # Height variation
        
        positions = np.column_stack([x_coords, y_coords, z_coords])
        
        return StructureCoordinates(
            positions=positions,
            structure_type="detected_origami",
            coordinate_system="cartesian",
            units="nanometers"
        )
    
    def _update_processing_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics."""
        self.processing_stats['images_processed'] += 1
        
        # Update average processing time
        prev_avg = self.processing_stats['average_processing_time']
        count = self.processing_stats['images_processed']
        self.processing_stats['average_processing_time'] = (prev_avg * (count - 1) + processing_time) / count
        
        # Update success rate
        if success:
            self.processing_stats['detection_success_rate'] = (
                (self.processing_stats['detection_success_rate'] * (count - 1) + 1.0) / count
            )
        else:
            self.processing_stats['detection_success_rate'] = (
                self.processing_stats['detection_success_rate'] * (count - 1) / count
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()


class ImageProcessor:
    """General image processor for various microscopy techniques."""
    
    def __init__(self):
        """Initialize image processor."""
        self.processors = {
            ImagingMode.AFM_TAPPING: AFMProcessor(),
            ImagingMode.AFM_CONTACT: AFMProcessor(),
            ImagingMode.AFM_NON_CONTACT: AFMProcessor()
        }
    
    def process_image(self, image: ImageData, mode: ImagingMode,
                     processing_params: Dict[str, Any] = None) -> ImageData:
        """Process image based on imaging mode."""
        if mode in self.processors:
            processor = self.processors[mode]
            
            if hasattr(processor, 'process'):
                return processor.process(image, **(processing_params or {}))
        
        # Fallback to basic processing
        return self._basic_processing(image)
    
    def _basic_processing(self, image: ImageData) -> ImageData:
        """Basic image processing for unsupported modes."""
        # Simple noise reduction and contrast enhancement
        data = image.data.copy()
        
        # Gaussian smoothing
        if data.ndim == 2:
            # Simple 3x3 smoothing kernel
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
            smoothed = np.zeros_like(data)
            
            for i in range(1, data.shape[0] - 1):
                for j in range(1, data.shape[1] - 1):
                    region = data[i-1:i+2, j-1:j+2]
                    smoothed[i, j] = np.sum(region * kernel)
            
            data = smoothed
        
        return ImageData.from_array(data, name=f"{image.name}_processed")
    
    def batch_process_images(self, images: List[ImageData], mode: ImagingMode,
                           processing_params: Dict[str, Any] = None) -> List[ImageData]:
        """Process multiple images in batch."""
        processed_images = []
        
        for image in images:
            try:
                processed_image = self.process_image(image, mode, processing_params)
                processed_images.append(processed_image)
            except Exception as e:
                print(f"Warning: Failed to process image {image.name}: {e}")
                continue
        
        return processed_images
    
    def compare_structures(self, structures1: List[DetectedStructure],
                         structures2: List[DetectedStructure]) -> Dict[str, float]:
        """Compare two sets of detected structures."""
        if not structures1 or not structures2:
            return {"similarity": 0.0, "count_difference": abs(len(structures1) - len(structures2))}
        
        # Simple comparison based on size and position
        similarities = []
        
        for struct1 in structures1:
            best_match_score = 0.0
            
            for struct2 in structures2:
                # Position similarity
                pos1 = np.array(struct1.center_position)
                pos2 = np.array(struct2.center_position)
                pos_dist = np.linalg.norm(pos1 - pos2)
                pos_similarity = 1.0 / (1.0 + pos_dist / 100.0)  # Normalize by 100nm
                
                # Size similarity
                size1 = np.array(struct1.dimensions)
                size2 = np.array(struct2.dimensions)
                size_ratio = np.min(size1) / np.max(size1) if np.max(size1) > 0 else 0
                size2_ratio = np.min(size2) / np.max(size2) if np.max(size2) > 0 else 0
                size_similarity = 1.0 - abs(size_ratio - size2_ratio)
                
                # Combined similarity
                combined_similarity = (pos_similarity + size_similarity) / 2.0
                best_match_score = max(best_match_score, combined_similarity)
            
            similarities.append(best_match_score)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        count_difference = abs(len(structures1) - len(structures2))
        
        return {
            "similarity": avg_similarity,
            "count_difference": count_difference,
            "structure_count_1": len(structures1),
            "structure_count_2": len(structures2)
        }