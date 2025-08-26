"""
DNA-Origami-AutoEncoder: A groundbreaking wet-lab ML framework that encodes 
images into self-assembling DNA origami structures and decodes them using 
transformer models trained on simulated base-pair kinetics.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

# Core API exports - Generation 2: Add robustness
try:
    from .models.dna_sequence import DNASequence
    from .models.image_data import ImageData
    from .models.origami_structure import OrigamiStructure
    
    # Generation 2: Robust modules
    from .utils.error_handling import DNAEncodingError, ValidationError
    from .utils.health_monitoring import HealthMonitor, performance_monitor
    from .reliability.circuit_breaker import CircuitBreaker
    from .reliability.fault_tolerance import FaultTolerantProcessor
    
    __all__ = [
        # Models
        "DNASequence",
        "ImageData", 
        "OrigamiStructure",
        # Robustness
        "DNAEncodingError",
        "ValidationError", 
        "HealthMonitor",
        "performance_monitor",
        "CircuitBreaker",
        "FaultTolerantProcessor",
    ]
except ImportError as e:
    print(f"⚠️  Generation 2 dependencies missing: {e}")
    # Fallback to Generation 1
    try:
        from .models.dna_sequence import DNASequence
        from .models.image_data import ImageData
        from .models.origami_structure import OrigamiStructure
        __all__ = ["DNASequence", "ImageData", "OrigamiStructure"]
    except ImportError:
        __all__ = []