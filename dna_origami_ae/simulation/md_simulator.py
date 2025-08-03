"""Molecular dynamics simulator for DNA origami."""

import numpy as np
import time
from typing import Dict, Any, Optional

from ..models.simulation_data import StructureCoordinates, TrajectoryData
from .force_field import ForceField


class MDSimulator:
    """Molecular dynamics simulator using Verlet integration."""
    
    def __init__(self, force_field: ForceField):
        """Initialize MD simulator with force field."""
        self.force_field = force_field
        
        # Physical constants
        self.kb = 0.001987  # Boltzmann constant in kcal/(mol·K)
        self.mass = 1.0  # Unified atomic mass unit (simplified)
    
    def run_simulation(self, 
                      initial_coords: StructureCoordinates,
                      time_steps: int = 1000000,
                      time_step_size: float = 0.002,  # ps
                      temperature: float = 300.0,
                      save_interval: int = 1000) -> TrajectoryData:
        """Run molecular dynamics simulation."""
        
        # Initialize system
        positions = initial_coords.positions.copy()
        velocities = self._initialize_velocities(positions.shape, temperature)
        forces = self.force_field.calculate_forces(initial_coords)
        
        # Storage for trajectory
        saved_frames = []
        timestamps = []
        
        print(f"Starting MD simulation: {time_steps} steps, {time_step_size} ps timestep")
        
        for step in range(time_steps):
            # Velocity Verlet integration
            positions, velocities, forces = self._verlet_step(
                positions, velocities, forces, time_step_size
            )
            
            # Apply thermostat
            velocities = self._apply_thermostat(velocities, temperature)
            
            # Save frame if needed
            if step % save_interval == 0:
                frame_coords = StructureCoordinates(
                    positions=positions.copy(),
                    atom_types=initial_coords.atom_types.copy(),
                    connectivity=initial_coords.connectivity
                )
                saved_frames.append(frame_coords)
                timestamps.append(step * time_step_size)
                
                # Progress update
                if step % (save_interval * 10) == 0:
                    progress = (step / time_steps) * 100
                    print(f"Progress: {progress:.1f}% (step {step}/{time_steps})")
        
        # Create trajectory data
        trajectory = TrajectoryData(
            frames=saved_frames,
            timestamps=np.array(timestamps),
            temperature=temperature
        )
        
        print(f"Simulation completed: {len(saved_frames)} frames saved")
        return trajectory
    
    def _initialize_velocities(self, shape: tuple, temperature: float) -> np.ndarray:
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        n_atoms, n_dims = shape
        
        # Standard deviation for Maxwell-Boltzmann distribution
        sigma = np.sqrt(self.kb * temperature / self.mass)
        
        # Random velocities
        velocities = np.random.normal(0, sigma, (n_atoms, n_dims))
        
        # Remove center of mass motion
        velocities -= np.mean(velocities, axis=0)
        
        return velocities
    
    def _verlet_step(self, positions: np.ndarray, velocities: np.ndarray, 
                    forces: np.ndarray, dt: float) -> tuple:
        """Perform one Velocity Verlet integration step."""
        
        # Update positions
        new_positions = positions + velocities * dt + 0.5 * forces * dt**2 / self.mass
        
        # Calculate new forces
        coords = StructureCoordinates(
            positions=new_positions,
            atom_types=['C'] * len(new_positions),  # Simplified
            connectivity=None
        )
        
        new_forces = self.force_field.calculate_forces(coords)
        
        # Update velocities
        new_velocities = velocities + 0.5 * (forces + new_forces) * dt / self.mass
        
        return new_positions, new_velocities, new_forces
    
    def _apply_thermostat(self, velocities: np.ndarray, target_temp: float) -> np.ndarray:
        """Apply Berendsen thermostat for temperature control."""
        
        # Calculate current temperature
        kinetic_energy = 0.5 * self.mass * np.sum(velocities**2)
        n_dof = velocities.size - 3  # Degrees of freedom (minus COM motion)
        current_temp = 2 * kinetic_energy / (n_dof * self.kb)
        
        # Scaling factor for Berendsen thermostat
        tau = 0.1  # ps (coupling time constant)
        dt = 0.002  # ps (timestep)
        
        if current_temp > 0:
            scaling = np.sqrt(1 + (dt / tau) * (target_temp / current_temp - 1))
            velocities *= scaling
        
        return velocities