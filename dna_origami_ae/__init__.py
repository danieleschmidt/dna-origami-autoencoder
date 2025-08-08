"""
DNA-Origami-AutoEncoder: A groundbreaking wet-lab ML framework that encodes 
images into self-assembling DNA origami structures and decodes them using 
transformer models trained on simulated base-pair kinetics.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

# Core API exports - only working modules for now
from .models.dna_sequence import DNASequence
from .models.image_data import ImageData
from .models.origami_structure import OrigamiStructure
from .encoding.image_encoder import DNAEncoder, Base4Encoder
from .encoding.biological_constraints import BiologicalConstraints
from .encoding.error_correction import DNAErrorCorrection

__all__ = [
    # Models
    "DNASequence",
    "ImageData", 
    "OrigamiStructure",
    # Encoding
    "DNAEncoder",
    "Base4Encoder", 
    "BiologicalConstraints",
    "DNAErrorCorrection",
]