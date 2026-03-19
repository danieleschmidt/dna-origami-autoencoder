"""
DNA Origami Autoencoder
=======================

Encode digital data as DNA nucleotide sequences, simulate the noise introduced
by DNA synthesis / sequencing / self-assembly, and decode back to the original
data using a transformer-based neural network.

Inspired by real DNA storage research:
  - Church et al. (2012) Science – first practical DNA data storage
  - Goldman et al. (2013) Nature – 739 kB in synthetic DNA
  - Organick et al. (2018) Nature Biotechnology – random access in DNA pools
  - Microsoft / Twist Bioscience – automated DNA storage pipeline

Components
----------
DNAEncoder         : binary data → nucleotide sequences (with Hamming ECC)
OrigamiSimulator   : introduce realistic sequencing / assembly errors
DNADecoder         : transformer that reconstructs data from noisy sequences
"""

from .encoder import DNAEncoder
from .simulator import OrigamiSimulator
from .decoder import DNADecoder

__all__ = ["DNAEncoder", "OrigamiSimulator", "DNADecoder"]
__version__ = "1.0.0"
