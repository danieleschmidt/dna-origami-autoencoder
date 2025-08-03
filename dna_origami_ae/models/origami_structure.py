"""Origami structure models for DNA origami design and simulation."""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from .dna_sequence import DNASequence


@dataclass
class StapleStrand:
    """Individual staple strand in DNA origami structure."""
    
    sequence: DNASequence
    start_helix: int
    start_position: int
    end_helix: int
    end_position: int
    crossovers: List[Tuple[int, int]] = field(default_factory=list)
    color: Optional[str] = None
    binding_domains: List[Tuple[int, int]] = field(default_factory=list)
    
    @property
    def length(self) -> int:
        """Return staple strand length."""
        return len(self.sequence)
    
    @property
    def span(self) -> int:
        """Return number of helices spanned."""
        return abs(self.end_helix - self.start_helix) + 1
    
    def get_binding_strength(self, temperature: float = 37.0) -> float:
        """Calculate binding strength at given temperature."""
        # Simplified binding strength calculation
        gc_bonus = self.sequence.gc_content * 2.0
        length_factor = min(self.length / 20.0, 2.0)  # Saturates at 20 bases
        temp_penalty = max(0, (temperature - 25) * 0.02)
        
        return (gc_bonus + length_factor - temp_penalty) * 10
    
    def overlaps_with(self, other: 'StapleStrand') -> bool:
        """Check if this staple overlaps with another."""
        # Check helix overlap
        helix_overlap = not (self.end_helix < other.start_helix or 
                           other.end_helix < self.start_helix)
        
        if not helix_overlap:
            return False
            
        # Check position overlap on overlapping helices
        for helix in range(max(self.start_helix, other.start_helix),
                          min(self.end_helix, other.end_helix) + 1):
            # Simplified - assumes linear positioning
            # In reality would need more complex 3D overlap detection
            return True
        
        return False


@dataclass
class ScaffoldPath:
    """Scaffold routing path through origami structure."""
    
    sequence: DNASequence
    path_coordinates: List[Tuple[int, int, int]]  # (helix, position, base_index)
    routing_method: str = "honeycomb"
    crossover_spacing: int = 21
    
    @property
    def total_length(self) -> int:
        """Return total scaffold length."""
        return len(self.sequence)
    
    @property
    def helix_count(self) -> int:
        """Return number of helices in the path."""
        if not self.path_coordinates:
            return 0
        helices = set(coord[0] for coord in self.path_coordinates)
        return len(helices)
    
    def get_crossover_positions(self) -> List[Tuple[int, int]]:
        """Get positions where scaffold crosses between helices."""
        crossovers = []
        
        for i in range(1, len(self.path_coordinates)):
            prev_helix = self.path_coordinates[i-1][0]
            curr_helix = self.path_coordinates[i][0]
            
            if prev_helix != curr_helix:
                crossovers.append((prev_helix, curr_helix))
        
        return crossovers
    
    def validate_routing(self) -> Tuple[bool, List[str]]:
        """Validate scaffold routing for design rules."""
        errors = []
        
        # Check crossover spacing
        crossovers = self.get_crossover_positions()
        if len(crossovers) > 0:
            crossover_indices = [i for i, coord in enumerate(self.path_coordinates)
                               if i > 0 and coord[0] != self.path_coordinates[i-1][0]]
            
            for i in range(1, len(crossover_indices)):
                spacing = crossover_indices[i] - crossover_indices[i-1]
                if spacing < self.crossover_spacing - 2 or spacing > self.crossover_spacing + 2:
                    errors.append(f"Crossover spacing {spacing} outside recommended range")
        
        # Check path continuity
        for i in range(1, len(self.path_coordinates)):
            prev_pos = self.path_coordinates[i-1][1]
            curr_pos = self.path_coordinates[i][1]
            prev_helix = self.path_coordinates[i-1][0]
            curr_helix = self.path_coordinates[i][0]
            
            if prev_helix == curr_helix and abs(curr_pos - prev_pos) > 1:
                errors.append(f"Discontinuous path at position {i}")
        
        return len(errors) == 0, errors


@dataclass
class OrigamiStructure:
    """Complete DNA origami structure with scaffold and staples."""
    
    name: str
    scaffold: ScaffoldPath
    staples: List[StapleStrand] = field(default_factory=list)
    target_shape: str = "custom"
    dimensions: Tuple[float, float, float] = (100.0, 100.0, 10.0)  # nm
    design_method: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_bases(self) -> int:
        """Return total number of DNA bases in structure."""
        scaffold_bases = len(self.scaffold.sequence)
        staple_bases = sum(len(staple.sequence) for staple in self.staples)
        return scaffold_bases + staple_bases
    
    @property
    def staple_count(self) -> int:
        """Return number of staple strands."""
        return len(self.staples)
    
    @property
    def estimated_mass(self) -> float:
        """Estimate molecular mass in daltons."""
        # Average molecular weight per nucleotide ≈ 330 Da
        return self.total_bases * 330.0
    
    def add_staple(self, staple: StapleStrand) -> None:
        """Add staple strand to structure."""
        # Check for overlaps with existing staples
        for existing_staple in self.staples:
            if staple.overlaps_with(existing_staple):
                raise ValueError(f"Staple overlaps with existing staple")
        
        self.staples.append(staple)
    
    def remove_staple(self, index: int) -> Optional[StapleStrand]:
        """Remove staple strand by index."""
        if 0 <= index < len(self.staples):
            return self.staples.pop(index)
        return None
    
    def validate_design(self) -> Tuple[bool, List[str]]:
        """Validate complete origami design."""
        errors = []
        
        # Validate scaffold
        scaffold_valid, scaffold_errors = self.scaffold.validate_routing()
        if not scaffold_valid:
            errors.extend([f"Scaffold: {error}" for error in scaffold_errors])
        
        # Validate staples
        for i, staple in enumerate(self.staples):
            # Check staple length
            if staple.length < 10 or staple.length > 60:
                errors.append(f"Staple {i}: Length {staple.length} outside recommended range (10-60)")
            
            # Check binding strength
            binding_strength = staple.get_binding_strength()
            if binding_strength < 10:
                errors.append(f"Staple {i}: Low binding strength {binding_strength:.1f}")
        
        # Check staple overlaps
        for i in range(len(self.staples)):
            for j in range(i+1, len(self.staples)):
                if self.staples[i].overlaps_with(self.staples[j]):
                    errors.append(f"Staples {i} and {j} overlap")
        
        return len(errors) == 0, errors
    
    def optimize_staple_lengths(self, target_length: int = 32, tolerance: int = 8) -> None:
        """Optimize staple lengths to target value."""
        optimized_staples = []
        
        for staple in self.staples:
            if abs(staple.length - target_length) > tolerance:
                # Split or merge staples as needed
                if staple.length > target_length + tolerance:
                    # Split long staple
                    split_staples = self._split_staple(staple, target_length)
                    optimized_staples.extend(split_staples)
                else:
                    # Keep short staples for now (merging would require complex logic)
                    optimized_staples.append(staple)
            else:
                optimized_staples.append(staple)
        
        self.staples = optimized_staples
    
    def _split_staple(self, staple: StapleStrand, target_length: int) -> List[StapleStrand]:
        """Split a long staple into shorter ones."""
        splits = []
        remaining_sequence = staple.sequence.sequence
        position_offset = 0
        
        while len(remaining_sequence) > target_length:
            split_seq = remaining_sequence[:target_length]
            remaining_sequence = remaining_sequence[target_length:]
            
            split_staple = StapleStrand(
                sequence=DNASequence(split_seq, constraints=staple.sequence.constraints),
                start_helix=staple.start_helix,
                start_position=staple.start_position + position_offset,
                end_helix=staple.start_helix,  # Simplified - single helix
                end_position=staple.start_position + position_offset + target_length - 1,
                color=staple.color
            )
            splits.append(split_staple)
            position_offset += target_length
        
        if remaining_sequence:
            # Add final short piece
            final_staple = StapleStrand(
                sequence=DNASequence(remaining_sequence, constraints=staple.sequence.constraints),
                start_helix=staple.start_helix,
                start_position=staple.start_position + position_offset,
                end_helix=staple.end_helix,
                end_position=staple.end_position,
                color=staple.color
            )
            splits.append(final_staple)
        
        return splits
    
    def export_sequences(self, filename: str, format: str = "csv") -> None:
        """Export all sequences to file."""
        if format == "csv":
            self._export_csv(filename)
        elif format == "json":
            self._export_json(filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_csv(self, filename: str) -> None:
        """Export sequences to CSV format."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Type', 'Name', 'Sequence', 'Length', 'GC_Content', 'Tm'])
            
            # Scaffold
            writer.writerow([
                'Scaffold',
                'M13mp18' if 'M13' in str(self.scaffold.sequence) else 'Custom',
                self.scaffold.sequence.sequence,
                len(self.scaffold.sequence),
                f"{self.scaffold.sequence.gc_content:.2%}",
                f"{self.scaffold.sequence.melting_temperature:.1f}°C"
            ])
            
            # Staples
            for i, staple in enumerate(self.staples):
                writer.writerow([
                    'Staple',
                    f"Staple_{i+1}",
                    staple.sequence.sequence,
                    len(staple.sequence),
                    f"{staple.sequence.gc_content:.2%}",
                    f"{staple.sequence.melting_temperature:.1f}°C"
                ])
    
    def _export_json(self, filename: str) -> None:
        """Export structure to JSON format."""
        data = {
            'name': self.name,
            'target_shape': self.target_shape,
            'dimensions': self.dimensions,
            'design_method': self.design_method,
            'scaffold': {
                'sequence': self.scaffold.sequence.sequence,
                'routing_method': self.scaffold.routing_method,
                'crossover_spacing': self.scaffold.crossover_spacing,
                'path_length': len(self.scaffold.path_coordinates)
            },
            'staples': [
                {
                    'sequence': staple.sequence.sequence,
                    'start_helix': staple.start_helix,
                    'start_position': staple.start_position,
                    'end_helix': staple.end_helix,
                    'end_position': staple.end_position,
                    'length': staple.length,
                    'color': staple.color
                }
                for staple in self.staples
            ],
            'metadata': self.metadata
        }
        
        with open(filename, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=2)
    
    def get_structure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the structure."""
        staple_lengths = [len(staple.sequence) for staple in self.staples]
        staple_gcs = [staple.sequence.gc_content for staple in self.staples]
        
        return {
            'total_bases': self.total_bases,
            'staple_count': self.staple_count,
            'scaffold_length': len(self.scaffold.sequence),
            'estimated_mass_kDa': self.estimated_mass / 1000,
            'dimensions_nm': self.dimensions,
            'staple_stats': {
                'mean_length': np.mean(staple_lengths) if staple_lengths else 0,
                'std_length': np.std(staple_lengths) if staple_lengths else 0,
                'min_length': min(staple_lengths) if staple_lengths else 0,
                'max_length': max(staple_lengths) if staple_lengths else 0,
                'mean_gc_content': np.mean(staple_gcs) if staple_gcs else 0,
            },
            'scaffold_stats': {
                'gc_content': self.scaffold.sequence.gc_content,
                'melting_temp': self.scaffold.sequence.melting_temperature,
                'helix_count': self.scaffold.helix_count,
                'crossover_count': len(self.scaffold.get_crossover_positions())
            }
        }
    
    def __str__(self) -> str:
        """String representation of origami structure."""
        return (f"OrigamiStructure({self.name}): "
                f"{self.staple_count} staples, "
                f"{self.total_bases} total bases, "
                f"{self.dimensions[0]:.1f}×{self.dimensions[1]:.1f} nm")