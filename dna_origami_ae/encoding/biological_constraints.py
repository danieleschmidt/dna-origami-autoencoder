"""Biological constraints for DNA sequence design."""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class BiologicalConstraints:
    """Comprehensive biological constraints for DNA design."""
    
    # Basic composition constraints
    gc_content_range: Tuple[float, float] = (0.4, 0.6)
    max_homopolymer_length: int = 4
    min_sequence_length: int = 10
    max_sequence_length: int = 200
    
    # Forbidden sequences (problematic for synthesis/assembly)
    forbidden_sequences: List[str] = field(default_factory=lambda: [
        'GGGG', 'CCCC', 'AAAA', 'TTTT',  # Homopolymer runs
        'GGAGG', 'CCTCC',  # Potential G-quadruplex forming
        'TTTGGG', 'CCCAAA',  # Transcription terminators
        'GCTGCA', 'TGCAGC',  # Restriction sites (BstBI)
        'GAATTC', 'CTTAAG',  # EcoRI restriction site
        'GGATCC', 'CCTAGG',  # BamHI restriction site
    ])
    
    # Thermodynamic constraints
    melting_temp_range: Tuple[float, float] = (55.0, 75.0)
    max_hairpin_stability: float = -3.0  # kcal/mol
    max_homodimer_stability: float = -5.0  # kcal/mol
    
    # Secondary structure constraints
    max_hairpin_length: int = 6
    min_loop_size: int = 3
    max_internal_loop_size: int = 30
    
    # Synthesis constraints
    avoid_synthesis_problems: bool = True
    max_gc_run: int = 6
    max_at_run: int = 8
    
    def validate_sequence(self, sequence: str) -> Tuple[bool, List[str]]:
        """Comprehensive validation of DNA sequence."""
        errors = []
        seq_upper = sequence.upper()
        
        # Basic sequence validation
        if not self._validate_bases(seq_upper):
            errors.append("Contains invalid DNA bases")
            return False, errors
        
        # Length constraints
        if len(sequence) < self.min_sequence_length:
            errors.append(f"Sequence too short: {len(sequence)} < {self.min_sequence_length}")
        
        if len(sequence) > self.max_sequence_length:
            errors.append(f"Sequence too long: {len(sequence)} > {self.max_sequence_length}")
        
        # Composition constraints
        if not self._validate_gc_content(seq_upper):
            gc_content = self._calculate_gc_content(seq_upper)
            errors.append(f"GC content {gc_content:.1%} outside range {self.gc_content_range}")
        
        # Homopolymer constraints
        violations = self._check_homopolymers(seq_upper)
        if violations:
            errors.extend(violations)
        
        # Forbidden sequences
        forbidden_found = self._check_forbidden_sequences(seq_upper)
        if forbidden_found:
            errors.append(f"Contains forbidden sequences: {', '.join(forbidden_found)}")
        
        # GC/AT run constraints
        if self.avoid_synthesis_problems:
            gc_runs = self._check_gc_runs(seq_upper)
            at_runs = self._check_at_runs(seq_upper)
            
            if gc_runs:
                errors.append(f"Excessive GC runs found: {gc_runs}")
            if at_runs:
                errors.append(f"Excessive AT runs found: {at_runs}")
        
        # Thermodynamic constraints
        tm = self._estimate_melting_temperature(seq_upper)
        if not (self.melting_temp_range[0] <= tm <= self.melting_temp_range[1]):
            errors.append(f"Melting temperature {tm:.1f}°C outside range {self.melting_temp_range}")
        
        # Secondary structure constraints
        hairpin_issues = self._check_hairpin_structures(seq_upper)
        if hairpin_issues:
            errors.extend(hairpin_issues)
        
        return len(errors) == 0, errors
    
    def _validate_bases(self, sequence: str) -> bool:
        """Check if sequence contains only valid DNA bases."""
        return bool(re.match(r'^[ATGC]+$', sequence))
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content."""
        if not sequence:
            return 0.0
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def _validate_gc_content(self, sequence: str) -> bool:
        """Validate GC content within range."""
        gc_content = self._calculate_gc_content(sequence)
        return self.gc_content_range[0] <= gc_content <= self.gc_content_range[1]
    
    def _check_homopolymers(self, sequence: str) -> List[str]:
        """Check for excessive homopolymer runs."""
        violations = []
        
        for base in 'ATGC':
            pattern = f'{base}{{{self.max_homopolymer_length + 1},}}'
            matches = re.finditer(pattern, sequence)
            
            for match in matches:
                violations.append(
                    f"Homopolymer {base}×{len(match.group())} at position {match.start()}"
                )
        
        return violations
    
    def _check_forbidden_sequences(self, sequence: str) -> List[str]:
        """Check for forbidden sequence motifs."""
        found_sequences = []
        
        for forbidden in self.forbidden_sequences:
            if forbidden.upper() in sequence:
                found_sequences.append(forbidden)
        
        return found_sequences
    
    def _check_gc_runs(self, sequence: str) -> List[str]:
        """Check for excessive GC runs."""
        violations = []
        pattern = f'[GC]{{{self.max_gc_run + 1},}}'
        
        matches = re.finditer(pattern, sequence)
        for match in matches:
            violations.append(f"GC run of {len(match.group())} at position {match.start()}")
        
        return violations
    
    def _check_at_runs(self, sequence: str) -> List[str]:
        """Check for excessive AT runs."""
        violations = []
        pattern = f'[AT]{{{self.max_at_run + 1},}}'
        
        matches = re.finditer(pattern, sequence)
        for match in matches:
            violations.append(f"AT run of {len(match.group())} at position {match.start()}")
        
        return violations
    
    def _estimate_melting_temperature(self, sequence: str) -> float:
        """Estimate melting temperature using nearest-neighbor method."""
        if len(sequence) < 2:
            return 0.0
        
        # Simplified nearest-neighbor calculation
        # Real implementation would use full thermodynamic tables
        
        gc_count = sequence.count('G') + sequence.count('C')
        at_count = sequence.count('A') + sequence.count('T')
        
        if len(sequence) <= 13:
            # Wallace rule for short oligonucleotides
            return 4 * gc_count + 2 * at_count
        else:
            # Modified formula for longer sequences
            gc_fraction = gc_count / len(sequence)
            return 81.5 + 0.41 * (gc_fraction * 100) - 675 / len(sequence)
    
    def _check_hairpin_structures(self, sequence: str) -> List[str]:
        """Check for potential hairpin structures."""
        violations = []
        
        # Look for potential hairpins (palindromic sequences with small loops)
        for i in range(len(sequence) - self.max_hairpin_length):
            for stem_length in range(3, self.max_hairpin_length + 1):
                for loop_size in range(self.min_loop_size, 10):
                    end_pos = i + 2 * stem_length + loop_size
                    
                    if end_pos <= len(sequence):
                        left_stem = sequence[i:i + stem_length]
                        loop = sequence[i + stem_length:i + stem_length + loop_size]
                        right_stem = sequence[i + stem_length + loop_size:end_pos]
                        
                        # Check if stems can form base pairs
                        if self._can_form_base_pairs(left_stem, right_stem[::-1]):
                            stability = self._estimate_hairpin_stability(left_stem, loop, right_stem)
                            
                            if stability < self.max_hairpin_stability:
                                violations.append(
                                    f"Stable hairpin (ΔG≈{stability:.1f} kcal/mol) at position {i}"
                                )
        
        return violations
    
    def _can_form_base_pairs(self, seq1: str, seq2: str) -> bool:
        """Check if two sequences can form Watson-Crick base pairs."""
        if len(seq1) != len(seq2):
            return False
        
        base_pairs = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        pairs = 0
        
        for b1, b2 in zip(seq1, seq2):
            if base_pairs.get(b1) == b2:
                pairs += 1
        
        # Require at least 70% complementarity
        return pairs / len(seq1) >= 0.7
    
    def _estimate_hairpin_stability(self, left_stem: str, loop: str, right_stem: str) -> float:
        """Estimate hairpin stability (simplified)."""
        # Simplified calculation - real version would use thermodynamic parameters
        
        # Stem stability (more GC = more stable)
        stem_gc = (left_stem.count('G') + left_stem.count('C') + 
                  right_stem.count('G') + right_stem.count('C'))
        stem_stability = -stem_gc * 1.5  # ~1.5 kcal/mol per GC pair
        
        # Loop penalty (larger loops less stable)
        loop_penalty = len(loop) * 0.5
        
        return stem_stability + loop_penalty
    
    def optimize_sequence(self, sequence: str, max_attempts: int = 100) -> Tuple[str, bool]:
        """Attempt to optimize sequence to satisfy constraints."""
        current_seq = sequence.upper()
        
        for attempt in range(max_attempts):
            is_valid, errors = self.validate_sequence(current_seq)
            
            if is_valid:
                return current_seq, True
            
            # Try to fix specific issues
            current_seq = self._apply_fixes(current_seq, errors)
        
        return current_seq, False
    
    def _apply_fixes(self, sequence: str, errors: List[str]) -> str:
        """Apply heuristic fixes to address constraint violations."""
        seq_list = list(sequence)
        
        # Fix homopolymer runs
        for error in errors:
            if "Homopolymer" in error:
                seq_list = self._fix_homopolymers(seq_list)
        
        # Fix forbidden sequences
        for error in errors:
            if "forbidden sequences" in error:
                seq_list = self._fix_forbidden_sequences(seq_list)
        
        # Adjust GC content
        for error in errors:
            if "GC content" in error:
                seq_list = self._adjust_gc_content(seq_list)
        
        return ''.join(seq_list)
    
    def _fix_homopolymers(self, seq_list: List[str]) -> List[str]:
        """Break up homopolymer runs."""
        bases = ['A', 'T', 'G', 'C']
        
        i = 0
        while i < len(seq_list) - self.max_homopolymer_length:
            # Check for homopolymer run
            current_base = seq_list[i]
            run_length = 1
            
            j = i + 1
            while j < len(seq_list) and seq_list[j] == current_base:
                run_length += 1
                j += 1
            
            # If run is too long, substitute some bases
            if run_length > self.max_homopolymer_length:
                # Replace every (max_length + 1)th base with a different base
                replacement_bases = [b for b in bases if b != current_base]
                
                for k in range(i + self.max_homopolymer_length, j, self.max_homopolymer_length + 1):
                    if k < len(seq_list):
                        seq_list[k] = np.random.choice(replacement_bases)
            
            i = j
        
        return seq_list
    
    def _fix_forbidden_sequences(self, seq_list: List[str]) -> List[str]:
        """Remove forbidden sequence motifs."""
        sequence = ''.join(seq_list)
        
        for forbidden in self.forbidden_sequences:
            while forbidden in sequence:
                # Find and replace forbidden sequence
                pos = sequence.find(forbidden)
                if pos != -1:
                    # Replace middle base with different base
                    middle_pos = pos + len(forbidden) // 2
                    original_base = seq_list[middle_pos]
                    
                    alternatives = [b for b in 'ATGC' if b != original_base]
                    seq_list[middle_pos] = np.random.choice(alternatives)
                    
                    sequence = ''.join(seq_list)
        
        return seq_list
    
    def _adjust_gc_content(self, seq_list: List[str]) -> List[str]:
        """Adjust GC content to target range."""
        current_gc = self._calculate_gc_content(''.join(seq_list))
        target_gc = sum(self.gc_content_range) / 2
        
        if current_gc < self.gc_content_range[0]:
            # Need more GC - replace some A/T with G/C
            at_positions = [i for i, base in enumerate(seq_list) if base in 'AT']
            n_to_replace = int((target_gc - current_gc) * len(seq_list))
            
            if at_positions and n_to_replace > 0:
                positions_to_change = np.random.choice(
                    at_positions, 
                    size=min(n_to_replace, len(at_positions)), 
                    replace=False
                )
                
                for pos in positions_to_change:
                    seq_list[pos] = np.random.choice(['G', 'C'])
        
        elif current_gc > self.gc_content_range[1]:
            # Need less GC - replace some G/C with A/T
            gc_positions = [i for i, base in enumerate(seq_list) if base in 'GC']
            n_to_replace = int((current_gc - target_gc) * len(seq_list))
            
            if gc_positions and n_to_replace > 0:
                positions_to_change = np.random.choice(
                    gc_positions,
                    size=min(n_to_replace, len(gc_positions)),
                    replace=False
                )
                
                for pos in positions_to_change:
                    seq_list[pos] = np.random.choice(['A', 'T'])
        
        return seq_list
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of all constraints."""
        return {
            'gc_content_range': self.gc_content_range,
            'max_homopolymer_length': self.max_homopolymer_length,
            'sequence_length_range': (self.min_sequence_length, self.max_sequence_length),
            'melting_temp_range': self.melting_temp_range,
            'forbidden_sequences_count': len(self.forbidden_sequences),
            'secondary_structure_limits': {
                'max_hairpin_length': self.max_hairpin_length,
                'min_loop_size': self.min_loop_size,
                'max_hairpin_stability': self.max_hairpin_stability
            },
            'synthesis_constraints': {
                'avoid_synthesis_problems': self.avoid_synthesis_problems,
                'max_gc_run': self.max_gc_run,
                'max_at_run': self.max_at_run
            }
        }