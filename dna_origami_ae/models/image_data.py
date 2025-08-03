"""Image data models for DNA origami encoding."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import hashlib


@dataclass
class ImageMetadata:
    """Metadata for image data."""
    
    width: int
    height: int
    channels: int = 1
    bit_depth: int = 8
    format: str = "grayscale"
    compression: Optional[str] = None
    source_file: Optional[str] = None
    preprocessing_steps: list = field(default_factory=list)
    encoding_parameters: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_pixels(self) -> int:
        """Return total number of pixels."""
        return self.width * self.height
    
    @property
    def size_bytes(self) -> int:
        """Estimate size in bytes."""
        return self.total_pixels * self.channels * (self.bit_depth // 8)
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0


@dataclass
class ImageData:
    """Container for image data with DNA encoding capabilities."""
    
    data: np.ndarray
    metadata: ImageMetadata
    name: Optional[str] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Validate and process image data on creation."""
        self._validate_data()
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _validate_data(self) -> None:
        """Validate image data consistency."""
        if self.data.ndim < 2:
            raise ValueError("Image data must be at least 2D")
        
        if self.data.ndim == 2:
            height, width = self.data.shape
            channels = 1
        elif self.data.ndim == 3:
            height, width, channels = self.data.shape
        else:
            raise ValueError("Image data must be 2D or 3D")
        
        if (height != self.metadata.height or 
            width != self.metadata.width or 
            channels != self.metadata.channels):
            raise ValueError("Image data shape doesn't match metadata")
        
        # Validate data type
        if self.metadata.bit_depth == 8:
            if self.data.dtype != np.uint8:
                raise ValueError("8-bit images must use uint8 data type")
            if np.any(self.data < 0) or np.any(self.data > 255):
                raise ValueError("8-bit image values must be in range [0, 255]")
        elif self.metadata.bit_depth == 16:
            if self.data.dtype != np.uint16:
                raise ValueError("16-bit images must use uint16 data type")
        else:
            raise ValueError(f"Unsupported bit depth: {self.metadata.bit_depth}")
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of image data."""
        return hashlib.sha256(self.data.tobytes()).hexdigest()
    
    @classmethod
    def from_file(cls, file_path: str, target_size: Optional[Tuple[int, int]] = None,
                  grayscale: bool = True) -> 'ImageData':
        """Load image from file."""
        try:
            pil_image = Image.open(file_path)
            
            # Convert to grayscale if requested
            if grayscale and pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            
            # Resize if target size specified
            if target_size:
                pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            data = np.array(pil_image)
            
            # Determine metadata
            if data.ndim == 2:
                height, width = data.shape
                channels = 1
                format_str = "grayscale"
            elif data.ndim == 3:
                height, width, channels = data.shape
                format_str = "rgb" if channels == 3 else "rgba"
            else:
                raise ValueError("Unsupported image format")
            
            metadata = ImageMetadata(
                width=width,
                height=height,
                channels=channels,
                format=format_str,
                source_file=file_path,
                preprocessing_steps=["loaded_from_file"]
            )
            
            if target_size:
                metadata.preprocessing_steps.append(f"resized_to_{target_size}")
            if grayscale and pil_image.mode != 'L':
                metadata.preprocessing_steps.append("converted_to_grayscale")
            
            return cls(
                data=data,
                metadata=metadata,
                name=file_path.split('/')[-1].split('.')[0]
            )
            
        except Exception as e:
            raise ValueError(f"Failed to load image from {file_path}: {e}")
    
    @classmethod
    def from_array(cls, array: np.ndarray, name: Optional[str] = None) -> 'ImageData':
        """Create ImageData from numpy array."""
        if array.ndim == 2:
            height, width = array.shape
            channels = 1
            format_str = "grayscale"
        elif array.ndim == 3:
            height, width, channels = array.shape
            format_str = "rgb" if channels == 3 else "rgba"
        else:
            raise ValueError("Array must be 2D or 3D")
        
        # Determine bit depth
        if array.dtype == np.uint8:
            bit_depth = 8
        elif array.dtype == np.uint16:
            bit_depth = 16
        else:
            raise ValueError("Array must be uint8 or uint16")
        
        metadata = ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            bit_depth=bit_depth,
            format=format_str
        )
        
        return cls(data=array, metadata=metadata, name=name)
    
    def to_pil(self) -> Image.Image:
        """Convert to PIL Image."""
        if self.metadata.channels == 1:
            mode = 'L'
            data = self.data
        elif self.metadata.channels == 3:
            mode = 'RGB'
            data = self.data
        elif self.metadata.channels == 4:
            mode = 'RGBA'
            data = self.data
        else:
            raise ValueError(f"Cannot convert {self.metadata.channels} channels to PIL")
        
        return Image.fromarray(data, mode=mode)
    
    def save(self, file_path: str) -> None:
        """Save image to file."""
        pil_image = self.to_pil()
        pil_image.save(file_path)
    
    def normalize(self, method: str = "min_max") -> 'ImageData':
        """Normalize image data."""
        if method == "min_max":
            # Scale to [0, 255] for uint8
            min_val = np.min(self.data)
            max_val = np.max(self.data)
            
            if max_val > min_val:
                normalized = ((self.data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(self.data, dtype=np.uint8)
                
        elif method == "z_score":
            # Z-score normalization, then clip and scale
            mean_val = np.mean(self.data)
            std_val = np.std(self.data)
            
            if std_val > 0:
                z_normalized = (self.data - mean_val) / std_val
                # Clip to [-3, 3] and scale to [0, 255]
                clipped = np.clip(z_normalized, -3, 3)
                normalized = ((clipped + 3) / 6 * 255).astype(np.uint8)
            else:
                normalized = np.full_like(self.data, 128, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        new_metadata = ImageMetadata(
            width=self.metadata.width,
            height=self.metadata.height,
            channels=self.metadata.channels,
            bit_depth=8,
            format=self.metadata.format,
            preprocessing_steps=self.metadata.preprocessing_steps + [f"normalized_{method}"]
        )
        
        return ImageData(
            data=normalized,
            metadata=new_metadata,
            name=f"{self.name}_normalized" if self.name else "normalized"
        )
    
    def resize(self, target_size: Tuple[int, int], method: str = "lanczos") -> 'ImageData':
        """Resize image to target dimensions."""
        pil_image = self.to_pil()
        
        # Select resampling method
        if method == "lanczos":
            resample = Image.Resampling.LANCZOS
        elif method == "bilinear":
            resample = Image.Resampling.BILINEAR
        elif method == "nearest":
            resample = Image.Resampling.NEAREST
        else:
            raise ValueError(f"Unknown resize method: {method}")
        
        resized_pil = pil_image.resize(target_size, resample)
        resized_array = np.array(resized_pil)
        
        new_metadata = ImageMetadata(
            width=target_size[0],
            height=target_size[1],
            channels=self.metadata.channels,
            bit_depth=self.metadata.bit_depth,
            format=self.metadata.format,
            preprocessing_steps=self.metadata.preprocessing_steps + [f"resized_to_{target_size}_{method}"]
        )
        
        return ImageData(
            data=resized_array,
            metadata=new_metadata,
            name=f"{self.name}_resized" if self.name else "resized"
        )
    
    def to_binary(self, encoding_bits: int = 8) -> np.ndarray:
        """Convert image to binary representation."""
        if encoding_bits not in [1, 2, 4, 8]:
            raise ValueError("Encoding bits must be 1, 2, 4, or 8")
        
        # Flatten image data
        flat_data = self.data.flatten()
        
        if encoding_bits == 8:
            # Direct binary representation
            return np.unpackbits(flat_data)
        else:
            # Quantize to fewer bits
            max_val = 2 ** encoding_bits - 1
            quantized = (flat_data * max_val // 255).astype(np.uint8)
            
            # Convert to binary with specified bit width
            binary_data = []
            for value in quantized:
                binary_str = format(value, f'0{encoding_bits}b')
                binary_data.extend([int(bit) for bit in binary_str])
            
            return np.array(binary_data, dtype=np.uint8)
    
    @classmethod
    def from_binary(cls, binary_data: np.ndarray, width: int, height: int,
                    channels: int = 1, encoding_bits: int = 8,
                    name: Optional[str] = None) -> 'ImageData':
        """Reconstruct image from binary data."""
        expected_bits = width * height * channels * encoding_bits
        
        if len(binary_data) != expected_bits:
            raise ValueError(f"Binary data length {len(binary_data)} doesn't match "
                           f"expected {expected_bits} for {width}x{height}x{channels} "
                           f"with {encoding_bits} bits")
        
        if encoding_bits == 8:
            # Direct reconstruction from 8-bit
            byte_data = np.packbits(binary_data)
            pixel_data = byte_data.reshape((height, width, channels) if channels > 1 else (height, width))
        else:
            # Reconstruct from quantized bits
            values_per_pixel = encoding_bits
            num_pixels = width * height * channels
            
            pixel_values = []
            for i in range(0, len(binary_data), values_per_pixel):
                bit_chunk = binary_data[i:i+values_per_pixel]
                value = sum(bit * (2 ** (values_per_pixel - 1 - j)) for j, bit in enumerate(bit_chunk))
                # Scale back to [0, 255]
                scaled_value = int(value * 255 / (2 ** encoding_bits - 1))
                pixel_values.append(scaled_value)
            
            pixel_array = np.array(pixel_values, dtype=np.uint8)
            pixel_data = pixel_array.reshape((height, width, channels) if channels > 1 else (height, width))
        
        metadata = ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            bit_depth=8,
            format="grayscale" if channels == 1 else "rgb",
            preprocessing_steps=[f"reconstructed_from_{encoding_bits}bit_binary"]
        )
        
        return cls(
            data=pixel_data,
            metadata=metadata,
            name=name or "reconstructed"
        )
    
    def calculate_mse(self, other: 'ImageData') -> float:
        """Calculate Mean Squared Error with another image."""
        if self.data.shape != other.data.shape:
            raise ValueError("Images must have same dimensions for MSE calculation")
        
        return np.mean((self.data.astype(float) - other.data.astype(float)) ** 2)
    
    def calculate_psnr(self, other: 'ImageData') -> float:
        """Calculate Peak Signal-to-Noise Ratio with another image."""
        mse = self.calculate_mse(other)
        if mse == 0:
            return float('inf')
        
        max_pixel_value = 255.0 if self.metadata.bit_depth == 8 else 65535.0
        return 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    def calculate_ssim(self, other: 'ImageData', window_size: int = 11) -> float:
        """Calculate Structural Similarity Index (simplified version)."""
        if self.data.shape != other.data.shape:
            raise ValueError("Images must have same dimensions for SSIM calculation")
        
        # Convert to float
        img1 = self.data.astype(float)
        img2 = other.data.astype(float)
        
        # Calculate local means
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        # Calculate variances and covariance
        var1 = np.var(img1)
        var2 = np.var(img2)
        cov = np.mean((img1 - mu1) * (img2 - mu2))
        
        # SSIM constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        # Calculate SSIM
        ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        
        return ssim
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive image statistics."""
        return {
            'shape': self.data.shape,
            'dtype': str(self.data.dtype),
            'min_value': int(np.min(self.data)),
            'max_value': int(np.max(self.data)),
            'mean_value': float(np.mean(self.data)),
            'std_value': float(np.std(self.data)),
            'total_pixels': self.metadata.total_pixels,
            'size_bytes': self.metadata.size_bytes,
            'aspect_ratio': self.metadata.aspect_ratio,
            'checksum': self.checksum,
            'preprocessing_steps': self.metadata.preprocessing_steps
        }
    
    def __str__(self) -> str:
        """String representation of ImageData."""
        name_part = f" ({self.name})" if self.name else ""
        return (f"ImageData{name_part}: {self.metadata.width}×{self.metadata.height}×"
                f"{self.metadata.channels}, {self.metadata.bit_depth}-bit {self.metadata.format}")
    
    def __eq__(self, other) -> bool:
        """Check equality based on data and metadata."""
        if not isinstance(other, ImageData):
            return False
        return (np.array_equal(self.data, other.data) and 
                self.metadata == other.metadata)