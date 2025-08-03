"""Routing algorithms for DNA origami scaffold design."""

import numpy as np
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod


class RoutingAlgorithm(ABC):
    """Abstract base class for scaffold routing algorithms."""
    
    @abstractmethod
    def route_scaffold(self, grid: np.ndarray, total_length: int) -> List[Tuple[int, int, int]]:
        """Route scaffold through grid."""
        pass


class HoneycombRouter(RoutingAlgorithm):
    """Honeycomb lattice routing algorithm."""
    
    def __init__(self, crossover_spacing: int = 21):
        """Initialize honeycomb router."""
        self.crossover_spacing = crossover_spacing
    
    def route_scaffold(self, grid: np.ndarray, total_length: int) -> List[Tuple[int, int, int]]:
        """Route scaffold through honeycomb lattice."""
        path = []
        
        # Get grid dimensions
        height, width = grid.shape
        
        # Find all valid positions
        valid_positions = np.where(grid)
        valid_coords = list(zip(valid_positions[0], valid_positions[1]))
        
        if not valid_coords:
            return path
        
        # Start at top-left valid position
        current_row, current_col = valid_coords[0]
        current_helix = 0
        current_position = 0
        base_index = 0
        
        # Direction for serpentine routing
        direction = 1  # 1 for right, -1 for left
        
        while base_index < total_length and current_row < height:
            # Add current position to path
            if current_row < height and current_col < width and grid[current_row, current_col]:
                path.append((current_helix, current_position, base_index))
                base_index += 1
                current_position += 1
            
            # Move to next position
            if direction == 1:  # Moving right
                if current_col + 1 < width and grid[current_row, current_col + 1]:
                    current_col += 1
                else:
                    # End of row, move to next row and reverse direction
                    current_row += 1
                    direction = -1
                    current_helix += 1
                    current_position = 0
                    
                    # Add crossover
                    if current_row < height and len(path) > 0:
                        path.append((current_helix, current_position, base_index))
                        base_index += 1
            
            else:  # Moving left
                if current_col - 1 >= 0 and grid[current_row, current_col - 1]:
                    current_col -= 1
                else:
                    # End of row, move to next row and reverse direction
                    current_row += 1
                    direction = 1
                    current_helix += 1
                    current_position = 0
                    
                    # Add crossover
                    if current_row < height and len(path) > 0:
                        path.append((current_helix, current_position, base_index))
                        base_index += 1
        
        return path


class SquareLatticeRouter(RoutingAlgorithm):
    """Square lattice routing algorithm."""
    
    def __init__(self, crossover_spacing: int = 21):
        """Initialize square lattice router."""
        self.crossover_spacing = crossover_spacing
    
    def route_scaffold(self, grid: np.ndarray, total_length: int) -> List[Tuple[int, int, int]]:
        """Route scaffold through square lattice."""
        path = []
        
        # Simple row-by-row routing
        height, width = grid.shape
        base_index = 0
        helix = 0
        
        for row in range(height):
            position = 0
            
            if row % 2 == 0:  # Even rows: left to right
                for col in range(width):
                    if grid[row, col] and base_index < total_length:
                        path.append((helix, position, base_index))
                        base_index += 1
                        position += 1
            else:  # Odd rows: right to left
                for col in range(width - 1, -1, -1):
                    if grid[row, col] and base_index < total_length:
                        path.append((helix, position, base_index))
                        base_index += 1
                        position += 1
            
            helix += 1
        
        return path