"""Main origami simulation class for molecular dynamics."""

import numpy as np
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..models.origami_structure import OrigamiStructure
from ..models.simulation_data import SimulationResult, TrajectoryData, StructureCoordinates, SimulationStatus
from .force_field import ForceField, oxDNAForceField
from .md_simulator import MDSimulator


@dataclass
class SimulationParameters:
    """Parameters for origami simulation."""
    
    temperature: float = 300.0  # Kelvin
    salt_concentration: float = 0.5  # Molar
    time_steps: int = 1000000
    time_step_size: float = 0.002  # ps
    save_interval: int = 1000
    force_field: str = "oxDNA2"
    thermostat: str = "langevin"
    pressure: float = 1.0  # atm
    gpu_acceleration: bool = True
    output_frequency: int = 10000


class OrigamiSimulator:
    """High-level interface for DNA origami molecular dynamics simulation."""
    
    def __init__(self, 
                 force_field: str = "oxDNA2",
                 temperature: float = 300.0,
                 salt_concentration: float = 0.5):
        """Initialize origami simulator."""
        self.force_field_name = force_field
        self.temperature = temperature
        self.salt_concentration = salt_concentration
        
        # Initialize force field
        self.force_field = self._get_force_field()
        
        # Initialize MD simulator
        self.md_simulator = MDSimulator(self.force_field)
        
        # Simulation statistics
        self.simulation_stats = {
            'total_simulations': 0,
            'successful_simulations': 0,
            'total_simulation_time': 0.0,
            'average_folding_time': 0.0
        }
    
    def _get_force_field(self) -> ForceField:
        """Get force field instance."""
        if self.force_field_name.lower() in ["oxdna", "oxdna2"]:
            return oxDNAForceField(
                temperature=self.temperature,
                salt_concentration=self.salt_concentration
            )
        else:
            raise ValueError(f"Unknown force field: {self.force_field_name}")
    
    def simulate_folding(self, 
                        origami: OrigamiStructure,
                        simulation_params: Optional[SimulationParameters] = None) -> SimulationResult:
        """Simulate DNA origami folding."""
        params = simulation_params or SimulationParameters()
        start_time = time.time()
        
        try:
            # Generate initial structure
            initial_coords = self._generate_initial_structure(origami)
            
            # Run MD simulation
            trajectory = self.md_simulator.run_simulation(
                initial_coords,
                time_steps=params.time_steps,
                time_step_size=params.time_step_size,
                temperature=params.temperature,
                save_interval=params.save_interval
            )
            
            # Create simulation result
            computation_time = time.time() - start_time
            
            result = SimulationResult(
                trajectory=trajectory,
                status=SimulationStatus.COMPLETED,
                parameters=params.__dict__.copy(),
                computation_time=computation_time,
                metadata={
                    'origami_name': origami.name,
                    'scaffold_length': len(origami.scaffold.sequence),
                    'num_staples': len(origami.staples),
                    'force_field': self.force_field_name,
                    'simulation_timestamp': time.time()
                }
            )
            
            # Update statistics
            self._update_stats(result, success=True)
            
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            
            # Return failed simulation result
            result = SimulationResult(
                trajectory=TrajectoryData(
                    frames=[],
                    timestamps=np.array([]),
                    temperature=params.temperature
                ),
                status=SimulationStatus.FAILED,
                error_message=str(e),
                computation_time=computation_time,
                parameters=params.__dict__.copy()
            )
            
            self._update_stats(result, success=False)
            return result
    
    def _generate_initial_structure(self, origami: OrigamiStructure) -> StructureCoordinates:
        """Generate initial 3D coordinates for origami structure."""
        # Simplified structure generation
        # In practice, this would use sophisticated molecular modeling
        
        total_bases = len(origami.scaffold.sequence) + sum(len(s.sequence) for s in origami.staples)
        
        # Generate random initial positions (will be optimized during simulation)
        positions = np.random.normal(0, 10.0, size=(total_bases, 3))
        
        # Center the structure
        positions -= np.mean(positions, axis=0)
        
        # Create atom types (simplified)
        atom_types = []
        
        # Scaffold atoms
        for base in origami.scaffold.sequence.sequence:
            atom_types.extend(self._get_atoms_for_base(base))
        
        # Staple atoms
        for staple in origami.staples:
            for base in staple.sequence.sequence:
                atom_types.extend(self._get_atoms_for_base(base))
        
        # Ensure positions and atom_types have same length
        if len(positions) != len(atom_types):
            # Adjust to match
            min_length = min(len(positions), len(atom_types))
            positions = positions[:min_length]
            atom_types = atom_types[:min_length]
        
        return StructureCoordinates(
            positions=positions,
            atom_types=atom_types
        )
    
    def _get_atoms_for_base(self, base: str) -> List[str]:
        """Get atom types for DNA base (simplified)."""
        # Simplified representation - each base as one "atom"
        return [f"{base}_atom"]
    
    def simulate_batch(self, 
                      origami_structures: List[OrigamiStructure],
                      simulation_params: Optional[SimulationParameters] = None) -> List[SimulationResult]:
        """Simulate multiple origami structures."""
        results = []
        
        for i, origami in enumerate(origami_structures):
            print(f"Simulating structure {i+1}/{len(origami_structures)}: {origami.name}")
            
            result = self.simulate_folding(origami, simulation_params)
            results.append(result)
            
            # Print progress
            if result.success:
                print(f"  Completed successfully in {result.computation_time:.1f}s")
            else:
                print(f"  Failed: {result.error_message}")
        
        return results
    
    def analyze_folding_kinetics(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Analyze folding kinetics from simulation trajectory."""
        if trajectory.n_frames < 2:
            return {'error': 'Insufficient trajectory data'}
        
        # Use the trajectory's built-in analysis
        kinetics = trajectory.analyze_folding_kinetics()
        
        # Add additional analysis
        rg_series = kinetics['radius_of_gyration_series']
        
        # Detect folding phases
        phases = self._detect_folding_phases(rg_series)
        
        # Calculate folding rate
        folding_rate = self._calculate_folding_rate(kinetics)
        
        return {
            **kinetics,
            'folding_phases': phases,
            'folding_rate': folding_rate,
            'simulation_quality': self._assess_simulation_quality(trajectory)
        }
    
    def _detect_folding_phases(self, rg_series: np.ndarray) -> Dict[str, Any]:
        """Detect different phases of folding."""
        if len(rg_series) < 10:
            return {'phases': []}
        
        # Simple phase detection based on radius of gyration changes
        phases = []
        
        # Initial expansion phase
        if rg_series[0] < np.max(rg_series):
            expansion_end = np.argmax(rg_series)
            phases.append({
                'name': 'expansion',
                'start_frame': 0,
                'end_frame': expansion_end,
                'description': 'Initial structure expansion'
            })
        
        # Compaction phase
        if len(rg_series) > 50:
            # Look for sustained decrease in RG
            window_size = min(50, len(rg_series) // 4)
            for i in range(window_size, len(rg_series) - window_size):
                early_avg = np.mean(rg_series[i-window_size:i])
                late_avg = np.mean(rg_series[i:i+window_size])
                
                if early_avg > late_avg * 1.1:  # 10% decrease
                    phases.append({
                        'name': 'compaction',
                        'start_frame': i - window_size,
                        'end_frame': i + window_size,
                        'description': 'Structure compaction phase'
                    })
                    break
        
        return {'phases': phases, 'total_phases': len(phases)}
    
    def _calculate_folding_rate(self, kinetics: Dict[str, Any]) -> Optional[float]:
        """Calculate folding rate (1/folding_time)."""
        folding_time = kinetics.get('folding_time')
        if folding_time and folding_time > 0:
            return 1.0 / folding_time
        return None
    
    def _assess_simulation_quality(self, trajectory: TrajectoryData) -> Dict[str, float]:
        """Assess quality of simulation."""
        quality_metrics = {}
        
        # Trajectory length adequacy
        min_frames = 1000
        length_score = min(1.0, trajectory.n_frames / min_frames)
        quality_metrics['trajectory_length_score'] = length_score
        
        # Stability assessment (low RMSD variance in final portion)
        if trajectory.n_frames > 100:
            final_portion = min(100, trajectory.n_frames // 4)
            rmsd_series = trajectory.calculate_rmsd()
            
            if len(rmsd_series) >= final_portion:
                final_rmsd_std = np.std(rmsd_series[-final_portion:])
                stability_score = 1.0 / (1.0 + final_rmsd_std)
                quality_metrics['stability_score'] = stability_score
        
        # Overall quality score
        scores = list(quality_metrics.values())
        quality_metrics['overall_quality'] = np.mean(scores) if scores else 0.0
        
        return quality_metrics
    
    def estimate_computational_requirements(self, 
                                          origami: OrigamiStructure,
                                          simulation_params: SimulationParameters) -> Dict[str, Any]:
        """Estimate computational requirements for simulation."""
        # Estimate based on system size and simulation length
        total_atoms = len(origami.scaffold.sequence) + sum(len(s.sequence) for s in origami.staples)
        
        # Rough estimates (would be calibrated from actual runs)
        atoms_per_base = 30  # Approximate for all-atom model
        total_atoms_full = total_atoms * atoms_per_base
        
        # Time estimates (very rough)
        time_per_step_per_atom = 1e-6  # seconds (depends on hardware)
        estimated_time = (simulation_params.time_steps * 
                         total_atoms_full * 
                         time_per_step_per_atom)
        
        # Memory estimates
        bytes_per_atom_per_frame = 100  # coordinates, velocities, forces
        frames_stored = simulation_params.time_steps // simulation_params.save_interval
        estimated_memory_gb = (total_atoms_full * 
                              frames_stored * 
                              bytes_per_atom_per_frame) / (1024**3)
        
        return {
            'total_atoms': total_atoms_full,
            'estimated_time_hours': estimated_time / 3600,
            'estimated_memory_gb': estimated_memory_gb,
            'recommended_gpu_memory_gb': max(4, estimated_memory_gb * 2),
            'frames_to_save': frames_stored,
            'trajectory_size_gb': estimated_memory_gb
        }
    
    def _update_stats(self, result: SimulationResult, success: bool) -> None:
        """Update simulation statistics."""
        self.simulation_stats['total_simulations'] += 1
        
        if success:
            self.simulation_stats['successful_simulations'] += 1
            
            if result.computation_time:
                total_time = self.simulation_stats['total_simulation_time']
                count = self.simulation_stats['successful_simulations']
                
                new_total = total_time + result.computation_time
                self.simulation_stats['total_simulation_time'] = new_total
                self.simulation_stats['average_folding_time'] = new_total / count
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        stats = self.simulation_stats.copy()
        
        if stats['total_simulations'] > 0:
            stats['success_rate'] = (stats['successful_simulations'] / 
                                   stats['total_simulations'])
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset simulation statistics."""
        self.simulation_stats = {
            'total_simulations': 0,
            'successful_simulations': 0,
            'total_simulation_time': 0.0,
            'average_folding_time': 0.0
        }