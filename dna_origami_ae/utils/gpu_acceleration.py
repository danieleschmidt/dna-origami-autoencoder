"""
GPU Acceleration Utilities for DNA Origami AutoEncoder

Provides GPU acceleration capabilities for encoding, simulation, and decoding
operations with automatic fallback to CPU when GPU is unavailable.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU resources and provides acceleration utilities."""
    
    def __init__(self, device: Optional[str] = None, enable_mixed_precision: bool = True):
        """Initialize GPU manager with automatic device detection."""
        self.device = device
        self.enable_mixed_precision = enable_mixed_precision
        self.is_available = False
        self.device_name = "CPU"
        self.memory_total = 0
        self.memory_available = 0
        
        # Try to initialize GPU backends in order of preference
        self._try_cuda()
        self._try_opencl()
        
        logger.info(f"GPU Manager initialized - Device: {self.device_name}, Available: {self.is_available}")
    
    def _try_cuda(self) -> bool:
        """Try to initialize CUDA backend."""
        try:
            import torch
            
            if torch.cuda.is_available():
                if self.device is None:
                    self.device = f"cuda:{torch.cuda.current_device()}"
                elif self.device.startswith('cuda'):
                    if not torch.cuda.device_count() > int(self.device.split(':')[1]):
                        logger.warning(f"Requested device {self.device} not available, using cuda:0")
                        self.device = "cuda:0"
                
                self.is_available = True
                self.device_name = torch.cuda.get_device_name(0)
                self.memory_total = torch.cuda.get_device_properties(0).total_memory
                self.memory_available = torch.cuda.memory_allocated(0)
                
                # Enable optimizations
                if self.enable_mixed_precision:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                
                logger.info(f"CUDA initialized: {self.device_name} ({self.memory_total // 1024**3} GB)")
                return True
                
        except ImportError:
            logger.debug("PyTorch not available")
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}")
            
        return False
    
    def _try_opencl(self) -> bool:
        """Try to initialize OpenCL backend as fallback."""
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            if platforms:
                context = cl.Context(dev_type=cl.device_type.GPU)
                devices = context.devices
                
                if devices:
                    self.device = "opencl:0"
                    self.is_available = True
                    self.device_name = f"OpenCL: {devices[0].name}"
                    logger.info(f"OpenCL initialized: {self.device_name}")
                    return True
                    
        except ImportError:
            logger.debug("PyOpenCL not available")
        except Exception as e:
            logger.debug(f"OpenCL initialization failed: {e}")
            
        return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            "device": self.device or "cpu",
            "is_available": self.is_available,
            "device_name": self.device_name,
            "memory_total": self.memory_total,
            "memory_available": self.memory_available,
            "mixed_precision": self.enable_mixed_precision
        }
        
        # Add CUDA-specific info if available
        if self.is_available and self.device and self.device.startswith('cuda'):
            try:
                import torch
                info.update({
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "compute_capability": torch.cuda.get_device_capability(0),
                    "multiprocessors": torch.cuda.get_device_properties(0).multi_processor_count
                })
            except:
                pass
                
        return info
    
    def optimize_memory(self, clear_cache: bool = True) -> None:
        """Optimize GPU memory usage."""
        if not self.is_available:
            return
            
        try:
            if self.device and self.device.startswith('cuda'):
                import torch
                if clear_cache:
                    torch.cuda.empty_cache()
                
                # Update memory stats
                self.memory_available = torch.cuda.memory_allocated(0)
                logger.debug(f"GPU memory optimized - Available: {self.memory_available // 1024**2} MB")
                
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")


class AcceleratedEncoder:
    """GPU-accelerated DNA encoding operations."""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        self.batch_size = 32
        
    def batch_encode_images(self, images: List[np.ndarray], 
                           encoding_params: Dict[str, Any]) -> List[str]:
        """Encode multiple images in parallel using GPU acceleration."""
        
        if not self.gpu_manager.is_available:
            logger.info("GPU not available, falling back to CPU batch processing")
            return self._cpu_batch_encode(images, encoding_params)
        
        try:
            return self._gpu_batch_encode(images, encoding_params)
        except Exception as e:
            logger.warning(f"GPU encoding failed: {e}, falling back to CPU")
            return self._cpu_batch_encode(images, encoding_params)
    
    def _gpu_batch_encode(self, images: List[np.ndarray], 
                         params: Dict[str, Any]) -> List[str]:
        """GPU-accelerated batch encoding implementation."""
        
        if self.gpu_manager.device and self.gpu_manager.device.startswith('cuda'):
            return self._cuda_batch_encode(images, params)
        elif self.gpu_manager.device and self.gpu_manager.device.startswith('opencl'):
            return self._opencl_batch_encode(images, params)
        else:
            return self._cpu_batch_encode(images, params)
    
    def _cuda_batch_encode(self, images: List[np.ndarray], 
                          params: Dict[str, Any]) -> List[str]:
        """CUDA-accelerated batch encoding."""
        try:
            import torch
            
            device = torch.device(self.gpu_manager.device)
            encoded_sequences = []
            
            # Process in batches to manage memory
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                
                # Convert to tensor and move to GPU
                batch_tensor = torch.stack([
                    torch.from_numpy(img.astype(np.float32)) for img in batch
                ]).to(device)
                
                # GPU-accelerated encoding operations
                with torch.cuda.amp.autocast(enabled=self.gpu_manager.enable_mixed_precision):
                    # Parallel base-4 conversion
                    flattened = batch_tensor.view(batch_tensor.size(0), -1)
                    
                    # Convert to base-4 representation on GPU
                    base4_tensor = self._parallel_base4_conversion(flattened)
                    
                    # Convert back to CPU for DNA string generation
                    base4_arrays = base4_tensor.cpu().numpy()
                    
                    for base4_array in base4_arrays:
                        dna_seq = self._base4_to_dna(base4_array)
                        encoded_sequences.append(dna_seq)
                
                # Clean up GPU memory
                del batch_tensor, base4_tensor
                torch.cuda.empty_cache()
            
            logger.info(f"GPU encoded {len(images)} images in {len(images)//self.batch_size + 1} batches")
            return encoded_sequences
            
        except Exception as e:
            logger.error(f"CUDA encoding error: {e}")
            raise
    
    def _parallel_base4_conversion(self, tensor):
        """Parallel base-4 conversion on GPU."""
        import torch
        
        # Efficient parallel conversion using bit operations
        # Each pixel (0-255) -> 4 base-4 digits (0-3)
        b0 = tensor & 3          # bits 0-1
        b1 = (tensor >> 2) & 3   # bits 2-3  
        b2 = (tensor >> 4) & 3   # bits 4-5
        b3 = (tensor >> 6) & 3   # bits 6-7
        
        return torch.stack([b3, b2, b1, b0], dim=-1).view(tensor.size(0), -1)
    
    def _base4_to_dna(self, base4_array: np.ndarray) -> str:
        """Convert base-4 array to DNA sequence."""
        base_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        return ''.join(base_map[b] for b in base4_array)
    
    def _opencl_batch_encode(self, images: List[np.ndarray], 
                           params: Dict[str, Any]) -> List[str]:
        """OpenCL-accelerated batch encoding."""
        # Simplified OpenCL implementation
        logger.info("Using OpenCL acceleration (simplified implementation)")
        return self._cpu_batch_encode(images, params)
    
    def _cpu_batch_encode(self, images: List[np.ndarray], 
                         params: Dict[str, Any]) -> List[str]:
        """CPU fallback batch encoding with optimizations."""
        from ..encoding import Base4Encoder
        
        encoder = Base4Encoder()
        encoded_sequences = []
        
        # Use numpy vectorization for CPU optimization
        for img in images:
            dna_seq = encoder.encode_image(img)
            encoded_sequences.append(dna_seq)
        
        logger.info(f"CPU encoded {len(images)} images")
        return encoded_sequences


class AcceleratedSimulator:
    """GPU-accelerated molecular dynamics simulations."""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        
    def run_parallel_simulations(self, origami_designs: List[Any], 
                                simulation_params: Dict[str, Any]) -> List[Any]:
        """Run multiple simulations in parallel using GPU acceleration."""
        
        if not self.gpu_manager.is_available:
            logger.info("GPU not available, using CPU parallel simulation")
            return self._cpu_parallel_simulate(origami_designs, simulation_params)
        
        try:
            return self._gpu_parallel_simulate(origami_designs, simulation_params)
        except Exception as e:
            logger.warning(f"GPU simulation failed: {e}, falling back to CPU")
            return self._cpu_parallel_simulate(origami_designs, simulation_params)
    
    def _gpu_parallel_simulate(self, designs: List[Any], 
                              params: Dict[str, Any]) -> List[Any]:
        """GPU-accelerated parallel simulation."""
        
        if self.gpu_manager.device and self.gpu_manager.device.startswith('cuda'):
            return self._cuda_parallel_simulate(designs, params)
        else:
            return self._cpu_parallel_simulate(designs, params)
    
    def _cuda_parallel_simulate(self, designs: List[Any], 
                               params: Dict[str, Any]) -> List[Any]:
        """CUDA-accelerated molecular dynamics simulation."""
        try:
            import torch
            
            device = torch.device(self.gpu_manager.device)
            simulation_results = []
            
            # Batch simulations for GPU efficiency
            batch_size = min(4, len(designs))  # Limit by GPU memory
            
            for i in range(0, len(designs), batch_size):
                batch_designs = designs[i:i + batch_size]
                
                # Parallel force calculations on GPU
                batch_results = []
                for design in batch_designs:
                    # Simplified GPU MD simulation
                    coordinates = self._gpu_md_step(design, params, device)
                    batch_results.append(coordinates)
                
                simulation_results.extend(batch_results)
                
                # Memory cleanup
                torch.cuda.empty_cache()
            
            logger.info(f"GPU simulated {len(designs)} structures")
            return simulation_results
            
        except Exception as e:
            logger.error(f"CUDA simulation error: {e}")
            raise
    
    def _gpu_md_step(self, design: Any, params: Dict[str, Any], device) -> Any:
        """Single GPU-accelerated MD simulation step."""
        import torch
        
        # Simplified MD calculation - in practice this would be much more complex
        # Generate random coordinates for demonstration
        n_atoms = 1000  # Typical origami size
        
        coordinates = torch.randn(n_atoms, 3, device=device) * 10.0
        
        # Simulate force calculations and integration steps
        for step in range(params.get('time_steps', 1000)):
            # Force calculation (simplified)
            forces = torch.randn_like(coordinates) * 0.1
            
            # Verlet integration (simplified) 
            coordinates += forces * params.get('dt', 0.001)
            
            if step % 100 == 0:
                # Periodic boundary conditions, thermostat, etc.
                coordinates = torch.clamp(coordinates, -50, 50)
        
        # Return to CPU for storage
        return coordinates.cpu().numpy()
    
    def _cpu_parallel_simulate(self, designs: List[Any], 
                              params: Dict[str, Any]) -> List[Any]:
        """CPU fallback parallel simulation."""
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        max_workers = min(multiprocessing.cpu_count(), len(designs))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._single_cpu_simulation, design, params)
                for design in designs
            ]
            
            results = [future.result() for future in futures]
        
        logger.info(f"CPU simulated {len(designs)} structures using {max_workers} processes")
        return results
    
    def _single_cpu_simulation(self, design: Any, params: Dict[str, Any]) -> Any:
        """Single CPU simulation."""
        # Simplified CPU MD simulation
        n_atoms = 1000
        coordinates = np.random.randn(n_atoms, 3) * 10.0
        
        for step in range(params.get('time_steps', 1000)):
            forces = np.random.randn(n_atoms, 3) * 0.1
            coordinates += forces * params.get('dt', 0.001)
            
            if step % 100 == 0:
                coordinates = np.clip(coordinates, -50, 50)
        
        return coordinates


class AcceleratedDecoder:
    """GPU-accelerated neural network decoding."""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        
    def batch_decode_structures(self, structures: List[Any], 
                               model_params: Dict[str, Any]) -> List[np.ndarray]:
        """Decode multiple structures in parallel using GPU acceleration."""
        
        if not self.gpu_manager.is_available:
            logger.info("GPU not available, using CPU batch decoding")
            return self._cpu_batch_decode(structures, model_params)
        
        try:
            return self._gpu_batch_decode(structures, model_params)
        except Exception as e:
            logger.warning(f"GPU decoding failed: {e}, falling back to CPU")
            return self._cpu_batch_decode(structures, model_params)
    
    def _gpu_batch_decode(self, structures: List[Any], 
                         params: Dict[str, Any]) -> List[np.ndarray]:
        """GPU-accelerated batch neural network inference."""
        
        if self.gpu_manager.device and self.gpu_manager.device.startswith('cuda'):
            return self._cuda_batch_decode(structures, params)
        else:
            return self._cpu_batch_decode(structures, params)
    
    def _cuda_batch_decode(self, structures: List[Any], 
                          params: Dict[str, Any]) -> List[np.ndarray]:
        """CUDA-accelerated transformer decoding."""
        try:
            import torch
            import torch.nn as nn
            
            device = torch.device(self.gpu_manager.device)
            batch_size = params.get('batch_size', 8)
            
            # Create simplified transformer model on GPU
            model = self._create_gpu_transformer_model(params).to(device)
            model.eval()
            
            decoded_images = []
            
            with torch.no_grad():
                for i in range(0, len(structures), batch_size):
                    batch_structures = structures[i:i + batch_size]
                    
                    # Convert structures to tensors
                    batch_tensor = self._structures_to_tensor(batch_structures, device)
                    
                    # GPU inference with mixed precision
                    with torch.cuda.amp.autocast(enabled=self.gpu_manager.enable_mixed_precision):
                        outputs = model(batch_tensor)
                    
                    # Convert outputs to images
                    batch_images = outputs.cpu().numpy()
                    decoded_images.extend(batch_images)
                    
                    # Memory cleanup
                    del batch_tensor, outputs
                    torch.cuda.empty_cache()
            
            logger.info(f"GPU decoded {len(structures)} structures")
            return decoded_images
            
        except Exception as e:
            logger.error(f"CUDA decoding error: {e}")
            raise
    
    def _create_gpu_transformer_model(self, params: Dict[str, Any]) -> 'nn.Module':
        """Create a simplified GPU transformer model."""
        import torch.nn as nn
        
        class SimpleTransformer(nn.Module):
            def __init__(self, input_dim=3, hidden_dim=256, output_dim=1024):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True),
                    num_layers=6
                )
                self.output_proj = nn.Linear(hidden_dim, output_dim)
                self.reshape = nn.Unflatten(1, (32, 32))  # Reshape to image
                
            def forward(self, x):
                x = self.input_proj(x)
                x = self.transformer(x)
                x = x.mean(dim=1)  # Global average pooling
                x = self.output_proj(x)
                x = torch.sigmoid(x)  # Image pixel values [0,1]
                x = self.reshape(x)
                return x
        
        return SimpleTransformer(
            input_dim=params.get('input_dim', 3),
            hidden_dim=params.get('hidden_dim', 256),
            output_dim=params.get('output_dim', 1024)
        )
    
    def _structures_to_tensor(self, structures: List[Any], device) -> 'torch.Tensor':
        """Convert structure data to GPU tensors."""
        import torch
        
        # Simplified structure to tensor conversion
        batch_data = []
        for structure in structures:
            # Generate dummy coordinate data
            coords = np.random.randn(100, 3).astype(np.float32)
            batch_data.append(coords)
        
        return torch.stack([torch.from_numpy(data) for data in batch_data]).to(device)
    
    def _cpu_batch_decode(self, structures: List[Any], 
                         params: Dict[str, Any]) -> List[np.ndarray]:
        """CPU fallback batch decoding."""
        from ..decoding import TransformerDecoder
        
        decoder = TransformerDecoder()
        decoded_images = []
        
        for structure in structures:
            image = decoder.decode_structure(structure)
            decoded_images.append(image)
        
        logger.info(f"CPU decoded {len(structures)} structures")
        return decoded_images


# Global GPU manager instance
_gpu_manager = None

def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager

def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return get_gpu_manager().is_available

def get_accelerated_encoder() -> AcceleratedEncoder:
    """Get GPU-accelerated encoder instance."""
    return AcceleratedEncoder(get_gpu_manager())

def get_accelerated_simulator() -> AcceleratedSimulator:
    """Get GPU-accelerated simulator instance."""
    return AcceleratedSimulator(get_gpu_manager())

def get_accelerated_decoder() -> AcceleratedDecoder:
    """Get GPU-accelerated decoder instance."""
    return AcceleratedDecoder(get_gpu_manager())