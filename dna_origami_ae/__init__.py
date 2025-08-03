"""
DNA-Origami-AutoEncoder: A groundbreaking wet-lab ML framework that encodes 
images into self-assembling DNA origami structures and decodes them using 
transformer models trained on simulated base-pair kinetics.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

# Core API exports
from .encoding import DNAEncoder, Base4Encoder, BiologicalConstraints
from .design import OrigamiDesigner, ShapeDesigner, Origami3D
from .simulation import OrigamiSimulator, GPUSimulator, CoarseGrainedModel
from .decoding import TransformerDecoder, OrigamiTransformer
from .models import DNASequence, OrigamiStructure, ImageData
from .utils import validators, helpers

__all__ = [
    # Encoding
    "DNAEncoder",
    "Base4Encoder", 
    "BiologicalConstraints",
    # Design
    "OrigamiDesigner",
    "ShapeDesigner",
    "Origami3D",
    # Simulation
    "OrigamiSimulator",
    "GPUSimulator", 
    "CoarseGrainedModel",
    # Decoding
    "TransformerDecoder",
    "OrigamiTransformer",
    # Models
    "DNASequence",
    "OrigamiStructure", 
    "ImageData",
    # Utils
    "validators",
    "helpers",
]