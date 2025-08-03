"""Force field implementations for DNA origami simulation."""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..models.simulation_data import StructureCoordinates


@dataclass
class ForceFieldParameters:
    """Parameters for molecular force field."""
    
    temperature: float = 300.0  # Kelvin
    salt_concentration: float = 0.5  # Molar
    dielectric_constant: float = 78.4  # Water
    cutoff_distance: float = 15.0  # Angstroms
    electrostatic_damping: float = 0.1


class ForceField(ABC):
    """Abstract base class for molecular force fields."""
    
    @abstractmethod
    def calculate_energy(self, coords: StructureCoordinates) -> float:
        """Calculate total system energy."""
        pass
    
    @abstractmethod
    def calculate_forces(self, coords: StructureCoordinates) -> np.ndarray:
        """Calculate forces on all atoms."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get force field parameters."""
        pass


class oxDNAForceField(ForceField):
    """oxDNA force field for DNA simulation."""
    
    def __init__(self, 
                 temperature: float = 300.0,
                 salt_concentration: float = 0.5):
        """Initialize oxDNA force field."""
        self.params = ForceFieldParameters(
            temperature=temperature,
            salt_concentration=salt_concentration
        )
        
        # oxDNA-specific parameters
        self.bond_length = 6.0  # Angstroms
        self.bond_strength = 100.0  # kcal/mol/A^2
        self.angle_strength = 50.0  # kcal/mol/rad^2
        self.stacking_strength = 5.0  # kcal/mol
        self.hydrogen_bond_strength = 8.0  # kcal/mol
        
        # Debye screening length for electrostatics
        self.debye_length = self._calculate_debye_length()
    
    def _calculate_debye_length(self) -> float:
        """Calculate Debye screening length."""
        # Simplified calculation
        # Real implementation would consider ionic strength more carefully
        ionic_strength = self.params.salt_concentration
        debye_length = 3.04 / np.sqrt(ionic_strength)  # Angstroms
        return debye_length
    
    def calculate_energy(self, coords: StructureCoordinates) -> float:
        """Calculate total system energy using oxDNA model."""
        total_energy = 0.0
        
        # Bond energy
        bond_energy = self._calculate_bond_energy(coords)
        total_energy += bond_energy
        
        # Angle energy
        angle_energy = self._calculate_angle_energy(coords)
        total_energy += angle_energy
        
        # Stacking energy
        stacking_energy = self._calculate_stacking_energy(coords)
        total_energy += stacking_energy
        
        # Hydrogen bonding energy
        hbond_energy = self._calculate_hydrogen_bond_energy(coords)
        total_energy += hbond_energy
        
        # Electrostatic energy
        electrostatic_energy = self._calculate_electrostatic_energy(coords)
        total_energy += electrostatic_energy
        
        return total_energy
    
    def _calculate_bond_energy(self, coords: StructureCoordinates) -> float:
        """Calculate bonded interaction energy."""
        if coords.connectivity is None:
            return 0.0
        
        energy = 0.0
        
        for bond in coords.connectivity:
            i, j = bond
            if i < len(coords.positions) and j < len(coords.positions):
                r_ij = np.linalg.norm(coords.positions[i] - coords.positions[j])
                deviation = r_ij - self.bond_length
                energy += 0.5 * self.bond_strength * deviation**2
        
        return energy
    
    def _calculate_angle_energy(self, coords: StructureCoordinates) -> float:
        """Calculate angle bending energy."""
        # Simplified - would need proper angle topology
        return 0.0
    
    def _calculate_stacking_energy(self, coords: StructureCoordinates) -> float:
        """Calculate base stacking energy."""
        energy = 0.0
        positions = coords.positions
        
        # Look for stacking interactions between adjacent bases
        for i in range(len(positions) - 1):
            r_ij = np.linalg.norm(positions[i] - positions[i+1])
            
            if 3.0 < r_ij < 4.5:  # Typical stacking distance
                # Morse potential for stacking
                energy += -self.stacking_strength * np.exp(-2.0 * (r_ij - 3.5))
        
        return energy
    
    def _calculate_hydrogen_bond_energy(self, coords: StructureCoordinates) -> float:
        """Calculate hydrogen bonding energy."""
        energy = 0.0
        positions = coords.positions
        
        # Look for hydrogen bonding pairs
        for i in range(len(positions)):
            for j in range(i + 2, len(positions)):  # Skip adjacent atoms
                r_ij = np.linalg.norm(positions[i] - positions[j])
                
                if 2.5 < r_ij < 3.5:  # Hydrogen bond distance range
                    # Check if bases can form hydrogen bonds (simplified)
                    if self._can_hydrogen_bond(coords.atom_types[i], coords.atom_types[j]):
                        # Morse potential for hydrogen bonding
                        energy += -self.hydrogen_bond_strength * np.exp(-2.0 * (r_ij - 3.0))
        
        return energy
    
    def _can_hydrogen_bond(self, atom_type1: str, atom_type2: str) -> bool:
        """Check if two atom types can form hydrogen bonds."""
        # Simplified - in reality would check specific base pairing rules
        hbond_pairs = [
            ('A_atom', 'T_atom'), ('T_atom', 'A_atom'),
            ('G_atom', 'C_atom'), ('C_atom', 'G_atom')
        ]
        return (atom_type1, atom_type2) in hbond_pairs
    
    def _calculate_electrostatic_energy(self, coords: StructureCoordinates) -> float:
        """Calculate electrostatic interaction energy."""
        energy = 0.0
        positions = coords.positions
        
        # Assign charges (simplified)
        charges = self._assign_charges(coords.atom_types)
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                
                if r_ij < self.params.cutoff_distance:
                    # Coulomb interaction with Debye screening
                    coulomb = (charges[i] * charges[j]) / (self.params.dielectric_constant * r_ij)
                    screening = np.exp(-r_ij / self.debye_length)
                    energy += coulomb * screening
        
        return energy
    
    def _assign_charges(self, atom_types: list) -> np.ndarray:
        """Assign partial charges to atoms."""
        charges = np.zeros(len(atom_types))
        
        # Simplified charge assignment
        for i, atom_type in enumerate(atom_types):
            if 'P' in atom_type:  # Phosphate
                charges[i] = -0.8
            elif any(base in atom_type for base in ['A', 'T', 'G', 'C']):
                charges[i] = 0.1  # Slight positive charge on bases
        
        return charges
    
    def calculate_forces(self, coords: StructureCoordinates) -> np.ndarray:
        """Calculate forces using finite differences."""
        forces = np.zeros_like(coords.positions)
        delta = 0.001  # Small displacement for finite differences
        
        for i in range(len(coords.positions)):
            for j in range(3):  # x, y, z components
                # Positive displacement
                coords_plus = StructureCoordinates(
                    positions=coords.positions.copy(),
                    atom_types=coords.atom_types.copy(),
                    connectivity=coords.connectivity
                )
                coords_plus.positions[i, j] += delta
                energy_plus = self.calculate_energy(coords_plus)
                
                # Negative displacement
                coords_minus = StructureCoordinates(
                    positions=coords.positions.copy(),
                    atom_types=coords.atom_types.copy(),
                    connectivity=coords.connectivity
                )
                coords_minus.positions[i, j] -= delta
                energy_minus = self.calculate_energy(coords_minus)
                
                # Force is negative gradient
                forces[i, j] = -(energy_plus - energy_minus) / (2 * delta)
        
        return forces
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get force field parameters."""
        return {
            'force_field': 'oxDNA',
            'temperature': self.params.temperature,
            'salt_concentration': self.params.salt_concentration,
            'bond_length': self.bond_length,
            'bond_strength': self.bond_strength,
            'stacking_strength': self.stacking_strength,
            'hydrogen_bond_strength': self.hydrogen_bond_strength,
            'debye_length': self.debye_length,
            'cutoff_distance': self.params.cutoff_distance
        }