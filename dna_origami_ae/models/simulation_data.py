"""Simulation data models for molecular dynamics and structure analysis."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from enum import Enum


class SimulationStatus(Enum):
    """Simulation status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StructureCoordinates:
    """3D coordinates for molecular structure."""
    
    positions: np.ndarray  # Shape: (n_atoms, 3)
    atom_types: Optional[List[str]] = None
    connectivity: Optional[np.ndarray] = None  # Bonds between atoms
    box_vectors: Optional[np.ndarray] = None  # Simulation box dimensions
    structure_type: str = "dna_origami"
    coordinate_system: str = "cartesian" 
    units: str = "nanometers"
    
    def __post_init__(self):
        """Validate coordinate data."""
        if self.positions.shape[1] != 3:
            raise ValueError("Positions must have shape (n_atoms, 3)")
        
        if self.atom_types is not None and len(self.atom_types) != self.positions.shape[0]:
            raise ValueError("Number of atom types must match number of positions")
        
        # Create default atom types if not provided
        if self.atom_types is None:
            self.atom_types = ['C'] * self.positions.shape[0]
        
        if self.connectivity is not None:
            if self.connectivity.shape[1] != 2:
                raise ValueError("Connectivity must have shape (n_bonds, 2)")
    
    @property
    def n_atoms(self) -> int:
        """Return number of atoms."""
        return self.positions.shape[0]
    
    @property
    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass (assuming equal masses)."""
        return np.mean(self.positions, axis=0)
    
    @property
    def geometric_center(self) -> np.ndarray:
        """Calculate geometric center."""
        return np.mean(self.positions, axis=0)
    
    @property
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bounding box (min, max) coordinates."""
        min_coords = np.min(self.positions, axis=0)
        max_coords = np.max(self.positions, axis=0)
        return min_coords, max_coords
    
    @property
    def radius_of_gyration(self) -> float:
        """Calculate radius of gyration."""
        center = self.center_of_mass
        distances_squared = np.sum((self.positions - center) ** 2, axis=1)
        return np.sqrt(np.mean(distances_squared))
    
    def translate(self, vector: np.ndarray) -> 'StructureCoordinates':
        """Translate structure by vector."""
        new_positions = self.positions + vector
        return StructureCoordinates(
            positions=new_positions,
            atom_types=self.atom_types.copy(),
            connectivity=self.connectivity.copy() if self.connectivity is not None else None,
            box_vectors=self.box_vectors.copy() if self.box_vectors is not None else None
        )
    
    def center_at_origin(self) -> 'StructureCoordinates':
        """Center structure at origin."""
        center = self.geometric_center
        return self.translate(-center)
    
    def get_distance_matrix(self) -> np.ndarray:
        """Calculate pairwise distance matrix."""
        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    def get_dna_backbone_atoms(self) -> np.ndarray:
        """Get indices of DNA backbone atoms (P, O5', C5', C4', C3', O3')."""
        backbone_atoms = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
        indices = []
        
        for i, atom_type in enumerate(self.atom_types):
            if atom_type in backbone_atoms:
                indices.append(i)
        
        return np.array(indices)
    
    def calculate_helix_parameters(self) -> Dict[str, float]:
        """Calculate DNA helix parameters (simplified)."""
        backbone_indices = self.get_dna_backbone_atoms()
        
        if len(backbone_indices) < 10:
            return {'rise_per_bp': 0.0, 'twist_per_bp': 0.0, 'major_groove_width': 0.0}
        
        backbone_positions = self.positions[backbone_indices]
        
        # Simplified helix parameter calculation
        # In practice, would use more sophisticated algorithms
        
        # Estimate rise per base pair
        backbone_length = np.sum(np.linalg.norm(
            backbone_positions[1:] - backbone_positions[:-1], axis=1))
        estimated_bp_count = len(backbone_indices) / 6  # 6 backbone atoms per nucleotide
        rise_per_bp = backbone_length / estimated_bp_count if estimated_bp_count > 0 else 0.0
        
        # Estimate twist (very simplified)
        twist_per_bp = 36.0  # Standard B-form DNA
        
        # Estimate major groove width (simplified)
        distances = self.get_distance_matrix()[backbone_indices][:, backbone_indices]
        major_groove_width = np.mean(distances[distances > 10.0])  # Arbitrary threshold
        
        return {
            'rise_per_bp': rise_per_bp,
            'twist_per_bp': twist_per_bp,
            'major_groove_width': major_groove_width,
            'estimated_bp_count': estimated_bp_count
        }


@dataclass 
class TrajectoryData:
    """Time series of molecular coordinates."""
    
    frames: List[StructureCoordinates]
    timestamps: np.ndarray
    temperature: float = 300.0
    pressure: float = 1.0
    energy_data: Optional[Dict[str, np.ndarray]] = None
    
    def __post_init__(self):
        """Validate trajectory data."""
        if len(self.frames) != len(self.timestamps):
            raise ValueError("Number of frames must match number of timestamps")
        
        if len(self.frames) == 0:
            raise ValueError("Trajectory must contain at least one frame")
        
        # Validate frame consistency
        n_atoms = self.frames[0].n_atoms
        for i, frame in enumerate(self.frames):
            if frame.n_atoms != n_atoms:
                raise ValueError(f"Frame {i} has different number of atoms")
    
    @property
    def n_frames(self) -> int:
        """Return number of frames."""
        return len(self.frames)
    
    @property
    def simulation_time(self) -> float:
        """Return total simulation time."""
        return self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0
    
    @property
    def time_step(self) -> float:
        """Return average time step."""
        if len(self.timestamps) < 2:
            return 0.0
        diffs = np.diff(self.timestamps)
        return np.mean(diffs)
    
    def get_final_structure(self) -> StructureCoordinates:
        """Get final structure from trajectory."""
        return self.frames[-1]
    
    def get_frame(self, index: int) -> StructureCoordinates:
        """Get specific frame by index."""
        return self.frames[index]
    
    def calculate_rmsd(self, reference_frame: int = 0) -> np.ndarray:
        """Calculate RMSD relative to reference frame."""
        reference = self.frames[reference_frame].positions
        rmsds = []
        
        for frame in self.frames:
            # Align structures (simplified - no rotation alignment)
            diff = frame.positions - reference
            rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
            rmsds.append(rmsd)
        
        return np.array(rmsds)
    
    def calculate_radius_of_gyration_series(self) -> np.ndarray:
        """Calculate radius of gyration for each frame."""
        return np.array([frame.radius_of_gyration for frame in self.frames])
    
    def get_average_structure(self) -> StructureCoordinates:
        """Calculate average structure over trajectory."""
        all_positions = np.array([frame.positions for frame in self.frames])
        avg_positions = np.mean(all_positions, axis=0)
        
        return StructureCoordinates(
            positions=avg_positions,
            atom_types=self.frames[0].atom_types.copy(),
            connectivity=self.frames[0].connectivity.copy() if self.frames[0].connectivity is not None else None
        )
    
    def analyze_folding_kinetics(self) -> Dict[str, Any]:
        """Analyze folding kinetics from trajectory."""
        rmsd_series = self.calculate_rmsd()
        rg_series = self.calculate_radius_of_gyration_series()
        
        # Detect folding completion (when RMSD stabilizes)
        rmsd_window = 50  # frames
        rmsd_threshold = 2.0  # Angstroms
        
        folding_complete_frame = None
        if len(rmsd_series) > rmsd_window:
            for i in range(rmsd_window, len(rmsd_series)):
                window_std = np.std(rmsd_series[i-rmsd_window:i])
                if window_std < rmsd_threshold:
                    folding_complete_frame = i
                    break
        
        # Calculate folding rate
        folding_time = None
        if folding_complete_frame is not None:
            folding_time = self.timestamps[folding_complete_frame]
        
        return {
            'rmsd_series': rmsd_series,
            'radius_of_gyration_series': rg_series,
            'folding_complete_frame': folding_complete_frame,
            'folding_time': folding_time,
            'final_rmsd': rmsd_series[-1],
            'final_radius_of_gyration': rg_series[-1],
            'rmsd_std': np.std(rmsd_series[-50:]) if len(rmsd_series) >= 50 else np.std(rmsd_series)
        }


@dataclass
class SimulationResult:
    """Complete simulation result with metadata."""
    
    trajectory: TrajectoryData
    status: SimulationStatus
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    computation_time: Optional[float] = None
    
    @property
    def success(self) -> bool:
        """Check if simulation completed successfully."""
        return self.status == SimulationStatus.COMPLETED
    
    @property
    def final_structure(self) -> Optional[StructureCoordinates]:
        """Get final structure if simulation completed."""
        return self.trajectory.get_final_structure() if self.success else None
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """Calculate simulation quality metrics."""
        if not self.success:
            return {'quality_score': 0.0}
        
        folding_analysis = self.trajectory.analyze_folding_kinetics()
        
        # Quality score based on multiple factors
        rmsd_stability = 1.0 / (1.0 + folding_analysis['rmsd_std'])  # Lower variance = higher quality
        folding_completion = 1.0 if folding_analysis['folding_complete_frame'] is not None else 0.5
        
        # Structure reasonableness checks
        final_structure = self.final_structure
        rg = final_structure.radius_of_gyration
        structure_reasonableness = min(1.0, 50.0 / rg) if rg > 0 else 0.0  # Penalize overly extended structures
        
        quality_score = (rmsd_stability + folding_completion + structure_reasonableness) / 3.0
        
        return {
            'quality_score': quality_score,
            'rmsd_stability': rmsd_stability,
            'folding_completion': folding_completion,
            'structure_reasonableness': structure_reasonableness,
            'final_rmsd': folding_analysis['final_rmsd'],
            'final_rg': folding_analysis['final_radius_of_gyration']
        }
    
    def export_pdb(self, filename: str, frame: int = -1) -> None:
        """Export specific frame to PDB format."""
        if not self.success:
            raise ValueError("Cannot export PDB from failed simulation")
        
        structure = self.trajectory.get_frame(frame)
        
        with open(filename, 'w') as f:
            f.write("HEADER    DNA ORIGAMI SIMULATION\n")
            f.write(f"TITLE     SIMULATION RESULT - FRAME {frame}\n")
            f.write("REMARK    Generated by DNA-Origami-AutoEncoder\n")
            
            for i, (pos, atom_type) in enumerate(zip(structure.positions, structure.atom_types)):
                # Simplified PDB format
                f.write(f"ATOM  {i+1:5d} {atom_type:>4s} DNA A{i+1:4d}    "
                       f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00           {atom_type[0]}\n")
            
            f.write("END\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive simulation summary."""
        summary = {
            'status': self.status.value,
            'success': self.success,
            'n_frames': self.trajectory.n_frames if self.trajectory else 0,
            'simulation_time': self.trajectory.simulation_time if self.trajectory else 0.0,
            'computation_time': self.computation_time,
            'parameters': self.parameters.copy(),
        }
        
        if self.success:
            quality_metrics = self.get_quality_metrics()
            summary.update(quality_metrics)
            
            final_structure = self.final_structure
            if final_structure:
                summary.update({
                    'final_n_atoms': final_structure.n_atoms,
                    'final_radius_of_gyration': final_structure.radius_of_gyration,
                    'bounding_box_size': np.linalg.norm(
                        final_structure.bounding_box[1] - final_structure.bounding_box[0]
                    )
                })
        else:
            summary['error_message'] = self.error_message
        
        return summary
    
    def __str__(self) -> str:
        """String representation of simulation result."""
        status_str = self.status.value.upper()
        if self.success:
            return f"SimulationResult({status_str}): {self.trajectory.n_frames} frames, {self.trajectory.simulation_time:.1f}ns"
        else:
            return f"SimulationResult({status_str}): {self.error_message or 'No details'}"