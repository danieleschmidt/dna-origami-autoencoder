"""DNA sequence models with biological constraints and validation."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class DNAConstraints:
    """Biological constraints for DNA sequence design."""
    
    gc_content_range: Tuple[float, float] = (0.4, 0.6)
    max_homopolymer_length: int = 4
    forbidden_sequences: List[str] = field(default_factory=lambda: ['GGGG', 'CCCC', 'AAAA', 'TTTT'])
    melting_temp_range: Tuple[float, float] = (55.0, 65.0)
    max_hairpin_length: int = 6
    min_loop_size: int = 3
    
    def validate_sequence(self, sequence: str) -> Tuple[bool, List[str]]:
        """Validate DNA sequence against biological constraints."""
        errors = []
        
        if not self._validate_bases(sequence):
            errors.append("Invalid DNA bases found")
            
        if not self._validate_gc_content(sequence):
            gc_content = self._calculate_gc_content(sequence)
            errors.append(f"GC content {gc_content:.2%} outside range {self.gc_content_range}")
            
        if not self._validate_homopolymers(sequence):
            errors.append(f"Homopolymer runs longer than {self.max_homopolymer_length}")
            
        if not self._validate_forbidden_sequences(sequence):
            errors.append("Contains forbidden sequences")
            
        return len(errors) == 0, errors
    
    def _validate_bases(self, sequence: str) -> bool:
        """Check if sequence contains only valid DNA bases."""
        return bool(re.match(r'^[ATGC]+$', sequence.upper()))
    
    def _validate_gc_content(self, sequence: str) -> bool:
        """Check GC content within allowed range."""
        gc_content = self._calculate_gc_content(sequence)
        return self.gc_content_range[0] <= gc_content <= self.gc_content_range[1]
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content percentage."""
        if not sequence:
            return 0.0
        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        return gc_count / len(sequence)
    
    def _validate_homopolymers(self, sequence: str) -> bool:
        """Check for excessive homopolymer runs."""
        for base in 'ATGC':
            pattern = base * (self.max_homopolymer_length + 1)
            if pattern in sequence.upper():
                return False
        return True
    
    def _validate_forbidden_sequences(self, sequence: str) -> bool:
        """Check for forbidden sequence motifs."""
        upper_seq = sequence.upper()
        return not any(forbidden in upper_seq for forbidden in self.forbidden_sequences)


@dataclass 
class DNASequence:
    """DNA sequence with metadata and validation."""
    
    sequence: str
    name: Optional[str] = None
    description: Optional[str] = None
    constraints: DNAConstraints = field(default_factory=DNAConstraints)
    metadata: Dict = field(default_factory=dict)
    skip_validation: bool = False
    
    def __post_init__(self):
        """Validate sequence on creation."""
        self.sequence = self.sequence.upper()
        if not self.skip_validation:
            is_valid, errors = self.constraints.validate_sequence(self.sequence)
            if not is_valid:
                raise ValueError(f"Invalid DNA sequence: {'; '.join(errors)}")
    
    @property
    def length(self) -> int:
        """Return sequence length."""
        return len(self.sequence)
    
    @property
    def gc_content(self) -> float:
        """Calculate GC content."""
        return self.constraints._calculate_gc_content(self.sequence)
    
    @property
    def melting_temperature(self) -> float:
        """Estimate melting temperature using nearest-neighbor method."""
        # Simplified Tm calculation for demonstration
        # In practice, would use more sophisticated methods
        gc_count = self.sequence.count('G') + self.sequence.count('C')
        at_count = self.sequence.count('A') + self.sequence.count('T')
        
        if self.length < 14:
            # For short oligonucleotides
            return 4 * gc_count + 2 * at_count
        else:
            # For longer sequences  
            return 81.5 + 0.41 * (self.gc_content * 100) - 675/self.length
    
    def reverse_complement(self) -> 'DNASequence':
        """Return reverse complement of the sequence."""
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        rev_comp = ''.join(complement_map[base] for base in reversed(self.sequence))
        
        return DNASequence(
            sequence=rev_comp,
            name=f"{self.name}_rev_comp" if self.name else None,
            description=f"Reverse complement of {self.description}" if self.description else None,
            constraints=self.constraints
        )
    
    def to_binary(self, encoding_scheme: str = 'base4') -> np.ndarray:
        """Convert DNA sequence to binary representation."""
        if encoding_scheme == 'base4':
            # A=00, T=01, G=10, C=11
            mapping = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
            binary_str = ''.join(mapping[base] for base in self.sequence)
            return np.array([int(bit) for bit in binary_str], dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported encoding scheme: {encoding_scheme}")
    
    @classmethod
    def from_binary(cls, binary_data: np.ndarray, encoding_scheme: str = 'base4', 
                    constraints: Optional[DNAConstraints] = None) -> 'DNASequence':
        """Create DNA sequence from binary data."""
        if encoding_scheme == 'base4':
            if len(binary_data) % 2 != 0:
                raise ValueError("Binary data length must be even for base4 encoding")
            
            mapping = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
            sequence = ''
            
            for i in range(0, len(binary_data), 2):
                bit_pair = ''.join(str(binary_data[i:i+2]))
                if bit_pair not in mapping:
                    raise ValueError(f"Invalid bit pair: {bit_pair}")
                sequence += mapping[bit_pair]
            
            return cls(
                sequence=sequence,
                constraints=constraints or DNAConstraints()
            )
        else:
            raise ValueError(f"Unsupported encoding scheme: {encoding_scheme}")
    
    def get_secondary_structure_score(self) -> float:
        """Calculate secondary structure propensity score."""
        # Simplified scoring based on palindromic sequences
        score = 0.0
        seq_len = len(self.sequence)
        
        # Check for hairpin potential
        for i in range(seq_len - 6):
            for j in range(i + 6, min(i + 20, seq_len)):
                substr = self.sequence[i:j]
                rev_comp = DNASequence(substr).reverse_complement().sequence
                
                # Look for reverse complement matches (potential hairpins)
                for k in range(j, seq_len - len(rev_comp) + 1):
                    if self.sequence[k:k+len(rev_comp)] == rev_comp:
                        score += len(rev_comp) ** 2
        
        return score / (seq_len ** 2)  # Normalize by sequence length
    
    def split_into_staples(self, staple_length: int = 32, overlap: int = 16) -> List['DNASequence']:
        """Split long sequence into overlapping staple sequences."""
        staples = []
        step_size = staple_length - overlap
        
        for i in range(0, len(self.sequence), step_size):
            end_pos = min(i + staple_length, len(self.sequence))
            staple_seq = self.sequence[i:end_pos]
            
            if len(staple_seq) >= overlap:  # Only include meaningful staples
                staples.append(DNASequence(
                    sequence=staple_seq,
                    name=f"{self.name}_staple_{len(staples)}" if self.name else f"staple_{len(staples)}",
                    constraints=self.constraints
                ))
        
        return staples
    
    def __str__(self) -> str:
        """String representation of DNA sequence."""
        name_part = f" ({self.name})" if self.name else ""
        return f"DNASequence[{self.length} bp]{name_part}: {self.sequence[:50]}{'...' if len(self.sequence) > 50 else ''}"
    
    def __len__(self) -> int:
        """Return sequence length."""
        return len(self.sequence)
    
    def __eq__(self, other) -> bool:
        """Check equality based on sequence."""
        if not isinstance(other, DNASequence):
            return False
        return self.sequence == other.sequence