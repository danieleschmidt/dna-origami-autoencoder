"""Main origami designer for creating DNA origami structures."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..models.dna_sequence import DNASequence, DNAConstraints
from ..models.origami_structure import OrigamiStructure, StapleStrand, ScaffoldPath
from .routing_algorithm import RoutingAlgorithm, HoneycombRouter
from .shape_library import ShapeLibrary


@dataclass
class DesignParameters:
    """Parameters for origami design."""
    
    scaffold_length: int = 7249  # M13mp18 standard
    staple_length: int = 32
    staple_overlap: int = 16
    crossover_spacing: int = 21
    routing_method: str = "honeycomb"
    target_shape: str = "square"
    dimensions: Tuple[float, float] = (100.0, 100.0)  # nm
    enforce_constraints: bool = True
    optimize_stability: bool = True


class OrigamiDesigner:
    """Main class for designing DNA origami structures."""
    
    def __init__(self, design_params: Optional[DesignParameters] = None):
        """Initialize origami designer."""
        self.params = design_params or DesignParameters()
        self.shape_library = ShapeLibrary()
        self.routing_algorithm = self._get_routing_algorithm()
        
        # Design statistics
        self.design_stats = {
            'structures_designed': 0,
            'total_staples_created': 0,
            'average_design_time': 0.0,
            'successful_designs': 0
        }
    
    def _get_routing_algorithm(self) -> RoutingAlgorithm:
        """Get routing algorithm based on parameters."""
        if self.params.routing_method == "honeycomb":
            return HoneycombRouter(crossover_spacing=self.params.crossover_spacing)
        else:
            raise ValueError(f"Unknown routing method: {self.params.routing_method}")
    
    def design_origami(self, dna_sequence: DNASequence, 
                      target_shape: Optional[str] = None,
                      dimensions: Optional[Tuple[float, float]] = None) -> OrigamiStructure:
        """Design complete origami structure from DNA sequence."""
        import time
        start_time = time.time()
        
        try:
            # Use provided parameters or defaults
            shape = target_shape or self.params.target_shape
            dims = dimensions or self.params.dimensions
            
            # Get shape template
            shape_template = self.shape_library.get_shape(shape, dims)
            
            # Design scaffold path
            scaffold_path = self._design_scaffold_path(dna_sequence, shape_template)
            
            # Design staples
            staples = self._design_staples(scaffold_path, shape_template)
            
            # Create origami structure
            structure = OrigamiStructure(
                name=f"{shape}_origami_{self.design_stats['structures_designed']}",
                scaffold=scaffold_path,
                staples=staples,
                target_shape=shape,
                dimensions=(dims[0], dims[1], 10.0),  # Add default height
                design_method=self.params.routing_method,
                metadata={
                    'design_parameters': self.params.__dict__.copy(),
                    'sequence_length': len(dna_sequence),
                    'design_timestamp': time.time()
                }
            )
            
            # Validate and optimize if requested
            if self.params.enforce_constraints:
                self._validate_design(structure)
            
            if self.params.optimize_stability:
                self._optimize_stability(structure)
            
            # Update statistics
            design_time = time.time() - start_time
            self._update_stats(structure, design_time, success=True)
            
            return structure
            
        except Exception as e:
            design_time = time.time() - start_time
            self._update_stats(None, design_time, success=False)
            raise ValueError(f"Origami design failed: {e}")
    
    def _design_scaffold_path(self, dna_sequence: DNASequence, 
                            shape_template: Dict[str, Any]) -> ScaffoldPath:
        """Design scaffold routing path."""
        # Get shape grid
        grid = shape_template['grid']
        
        # Route scaffold through the grid
        path_coordinates = self.routing_algorithm.route_scaffold(
            grid, 
            total_length=len(dna_sequence)
        )
        
        # Create scaffold path
        scaffold_path = ScaffoldPath(
            sequence=dna_sequence,
            path_coordinates=path_coordinates,
            routing_method=self.params.routing_method,
            crossover_spacing=self.params.crossover_spacing
        )
        
        return scaffold_path
    
    def _design_staples(self, scaffold_path: ScaffoldPath, 
                       shape_template: Dict[str, Any]) -> List[StapleStrand]:
        """Design staple strands for the scaffold."""
        staples = []
        
        # Get binding regions from scaffold path
        binding_regions = self._identify_binding_regions(scaffold_path)
        
        # Create staples for each binding region
        for i, region in enumerate(binding_regions):
            try:
                staple = self._create_staple_for_region(region, i)
                if staple:
                    staples.append(staple)
            except Exception as e:
                print(f"Warning: Failed to create staple {i}: {e}")
                continue
        
        return staples
    
    def _identify_binding_regions(self, scaffold_path: ScaffoldPath) -> List[Dict[str, Any]]:
        """Identify regions where staples should bind to scaffold."""
        regions = []
        
        # Simple approach: create binding regions based on staple length
        coordinates = scaffold_path.path_coordinates
        
        i = 0
        while i < len(coordinates) - self.params.staple_length:
            # Determine region boundaries
            start_idx = i
            end_idx = min(i + self.params.staple_length, len(coordinates) - 1)
            
            # Extract region coordinates
            region_coords = coordinates[start_idx:end_idx + 1]
            
            if len(region_coords) >= 10:  # Minimum staple length
                region = {
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'coordinates': region_coords,
                    'start_helix': region_coords[0][0],
                    'end_helix': region_coords[-1][0],
                    'start_position': region_coords[0][1],
                    'end_position': region_coords[-1][1]
                }
                regions.append(region)
            
            # Advance by staple length minus overlap
            i += max(1, self.params.staple_length - self.params.staple_overlap)
        
        return regions
    
    def _create_staple_for_region(self, region: Dict[str, Any], staple_id: int) -> Optional[StapleStrand]:
        """Create a staple strand for a binding region."""
        # Extract scaffold sequence for this region
        scaffold_seq = self.scaffold_path.sequence.sequence if hasattr(self, 'scaffold_path') else 'A' * 32
        
        start_idx = region['start_index']
        end_idx = region['end_index']
        
        # Get reverse complement of scaffold region
        region_length = end_idx - start_idx + 1
        if region_length > len(scaffold_seq):
            region_length = len(scaffold_seq)
        
        # Create simple staple sequence (reverse complement of scaffold region)
        scaffold_region = scaffold_seq[start_idx:start_idx + region_length]
        staple_sequence = self._reverse_complement(scaffold_region)
        
        # Create DNASequence with constraints
        try:
            dna_seq = DNASequence(
                sequence=staple_sequence,
                name=f"staple_{staple_id}",
                description=f"Staple for region {start_idx}-{end_idx}",
                constraints=DNAConstraints()
            )
        except ValueError:
            # If sequence violates constraints, modify it
            dna_seq = self._fix_staple_sequence(staple_sequence, staple_id)
        
        # Create staple strand
        staple = StapleStrand(
            sequence=dna_seq,
            start_helix=region['start_helix'],
            start_position=region['start_position'],
            end_helix=region['end_helix'],
            end_position=region['end_position'],
            crossovers=self._find_crossovers_in_region(region),
            color=self._assign_staple_color(staple_id)
        )
        
        return staple
    
    def _reverse_complement(self, sequence: str) -> str:
        """Calculate reverse complement of DNA sequence."""
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        return ''.join(complement_map.get(base, 'A') for base in reversed(sequence.upper()))
    
    def _fix_staple_sequence(self, sequence: str, staple_id: int) -> DNASequence:
        """Fix staple sequence to satisfy biological constraints."""
        constraints = DNAConstraints()
        
        # Try to optimize the sequence
        optimized_seq, success = constraints.optimize_sequence(sequence)
        
        if not success:
            # If optimization failed, create a simple valid sequence
            length = len(sequence)
            # Create alternating pattern with good GC content
            pattern = "ATGC" * (length // 4 + 1)
            optimized_seq = pattern[:length]
        
        return DNASequence(
            sequence=optimized_seq,
            name=f"staple_{staple_id}_fixed",
            constraints=constraints
        )
    
    def _find_crossovers_in_region(self, region: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Find crossover positions in a region."""
        crossovers = []
        coordinates = region['coordinates']
        
        for i in range(1, len(coordinates)):
            prev_helix = coordinates[i-1][0]
            curr_helix = coordinates[i][0]
            
            if prev_helix != curr_helix:
                crossovers.append((prev_helix, curr_helix))
        
        return crossovers
    
    def _assign_staple_color(self, staple_id: int) -> str:
        """Assign color to staple for visualization."""
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
            "#FFA500", "#800080", "#008000", "#FFC0CB", "#A52A2A", "#808080"
        ]
        return colors[staple_id % len(colors)]
    
    def _validate_design(self, structure: OrigamiStructure) -> None:
        """Validate the designed structure."""
        is_valid, errors = structure.validate_design()
        
        if not is_valid:
            print(f"Design validation warnings: {'; '.join(errors)}")
            
        # Additional validation checks
        self._check_staple_coverage(structure)
        self._check_binding_stability(structure)
    
    def _check_staple_coverage(self, structure: OrigamiStructure) -> None:
        """Check that staples provide adequate scaffold coverage."""
        scaffold_length = len(structure.scaffold.sequence)
        
        # Calculate total staple coverage
        total_coverage = sum(len(staple.sequence) for staple in structure.staples)
        coverage_ratio = total_coverage / scaffold_length
        
        if coverage_ratio < 0.8:  # Less than 80% coverage
            print(f"Warning: Low staple coverage ({coverage_ratio:.1%})")
    
    def _check_binding_stability(self, structure: OrigamiStructure) -> None:
        """Check binding stability of staples."""
        weak_staples = []
        
        for i, staple in enumerate(structure.staples):
            binding_strength = staple.get_binding_strength()
            if binding_strength < 15.0:  # Arbitrary threshold
                weak_staples.append(i)
        
        if weak_staples:
            print(f"Warning: {len(weak_staples)} staples have low binding strength")
    
    def _optimize_stability(self, structure: OrigamiStructure) -> None:
        """Optimize structure for stability."""
        # Optimize staple lengths
        structure.optimize_staple_lengths(
            target_length=self.params.staple_length,
            tolerance=8
        )
        
        # Could add more optimization strategies here
        # - Adjust crossover positions
        # - Optimize sequence composition
        # - Balance binding strengths
    
    def _update_stats(self, structure: Optional[OrigamiStructure], 
                     design_time: float, success: bool) -> None:
        """Update design statistics."""
        self.design_stats['structures_designed'] += 1
        
        if success and structure:
            self.design_stats['successful_designs'] += 1
            self.design_stats['total_staples_created'] += len(structure.staples)
        
        # Update average design time
        prev_avg = self.design_stats['average_design_time']
        count = self.design_stats['structures_designed']
        self.design_stats['average_design_time'] = (prev_avg * (count - 1) + design_time) / count
    
    def design_from_image_data(self, dna_sequences: List[DNASequence],
                              target_shape: str = "square",
                              dimensions: Tuple[float, float] = (100.0, 100.0)) -> OrigamiStructure:
        """Design origami structure from multiple DNA sequences (e.g., from image encoding)."""
        # Combine sequences into single scaffold
        combined_sequence = ""
        for seq in dna_sequences:
            combined_sequence += seq.sequence
        
        # Create combined DNA sequence
        scaffold_sequence = DNASequence(
            sequence=combined_sequence,
            name="combined_scaffold",
            description="Combined sequence from image encoding"
        )
        
        # Design origami
        return self.design_origami(scaffold_sequence, target_shape, dimensions)
    
    def estimate_assembly_conditions(self, structure: OrigamiStructure) -> Dict[str, Any]:
        """Estimate optimal assembly conditions."""
        # Analyze structure to recommend assembly conditions
        avg_staple_tm = np.mean([
            staple.sequence.melting_temperature for staple in structure.staples
        ])
        
        # Recommend annealing temperature (typically 5-10°C below lowest Tm)
        min_staple_tm = min([
            staple.sequence.melting_temperature for staple in structure.staples
        ]) if structure.staples else 60.0
        
        annealing_temp = max(45.0, min_staple_tm - 10.0)
        
        return {
            'annealing_temperature': annealing_temp,
            'suggested_mg_concentration': 12.5,  # mM
            'suggested_nacl_concentration': 50.0,  # mM
            'annealing_time_hours': 24,
            'cooling_rate_per_hour': 1.0,  # °C/hour
            'final_temperature': 20.0,
            'staple_concentration_ratio': 10,  # 10:1 staple:scaffold
            'estimated_yield': self._estimate_yield(structure)
        }
    
    def _estimate_yield(self, structure: OrigamiStructure) -> float:
        """Estimate assembly yield based on structure complexity."""
        # Simple heuristic based on number of staples and their properties
        base_yield = 0.8
        
        # Penalty for large number of staples
        staple_penalty = min(0.3, len(structure.staples) * 0.01)
        
        # Penalty for weak binding staples
        weak_staples = sum(1 for staple in structure.staples 
                          if staple.get_binding_strength() < 15.0)
        weak_penalty = weak_staples * 0.05
        
        estimated_yield = max(0.1, base_yield - staple_penalty - weak_penalty)
        return estimated_yield
    
    def export_design_files(self, structure: OrigamiStructure, 
                           output_dir: str = ".", formats: List[str] = None) -> Dict[str, str]:
        """Export design files in various formats."""
        if formats is None:
            formats = ["csv", "json"]
        
        exported_files = {}
        
        for format_type in formats:
            if format_type == "csv":
                filename = f"{output_dir}/{structure.name}_sequences.csv"
                structure.export_sequences(filename, "csv")
                exported_files["csv"] = filename
            
            elif format_type == "json":
                filename = f"{output_dir}/{structure.name}_design.json"
                structure.export_sequences(filename, "json")
                exported_files["json"] = filename
            
            elif format_type == "cadnano":
                filename = f"{output_dir}/{structure.name}_cadnano.json"
                self._export_cadnano_format(structure, filename)
                exported_files["cadnano"] = filename
        
        return exported_files
    
    def _export_cadnano_format(self, structure: OrigamiStructure, filename: str) -> None:
        """Export in caDNAno-compatible format."""
        # Simplified caDNAno export
        cadnano_data = {
            "name": structure.name,
            "vstrands": [],
            "vhelix": []
        }
        
        # This would need to be implemented based on caDNAno specification
        # For now, just save basic structure info
        import json
        with open(filename, 'w') as f:
            json.dump(cadnano_data, f, indent=2)
    
    def get_design_statistics(self) -> Dict[str, Any]:
        """Get comprehensive design statistics."""
        return self.design_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset design statistics."""
        self.design_stats = {
            'structures_designed': 0,
            'total_staples_created': 0,
            'average_design_time': 0.0,
            'successful_designs': 0
        }