"""Shape library for DNA origami design."""

import numpy as np
from typing import Dict, Any, Tuple, List


class ShapeLibrary:
    """Library of predefined shapes for DNA origami design."""
    
    def __init__(self):
        """Initialize shape library with basic shapes."""
        self.shapes = {}
        self._initialize_basic_shapes()
    
    def _initialize_basic_shapes(self):
        """Initialize basic geometric shapes."""
        self.shapes = {
            'square': self._create_square,
            'rectangle': self._create_rectangle,
            'circle': self._create_circle,
            'triangle': self._create_triangle,
            'cross': self._create_cross,
            'star': self._create_star
        }
    
    def get_shape(self, shape_name: str, dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Get shape template with specified dimensions."""
        if shape_name not in self.shapes:
            raise ValueError(f"Unknown shape: {shape_name}")
        
        return self.shapes[shape_name](dimensions)
    
    def _create_square(self, dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create square shape template."""
        width, height = dimensions
        grid_size = min(int(width // 2.5), int(height // 2.5))  # ~2.5nm per grid point
        
        grid = np.ones((grid_size, grid_size), dtype=bool)
        
        return {
            'name': 'square',
            'grid': grid,
            'dimensions': dimensions,
            'grid_spacing': 2.5,  # nm
            'total_points': np.sum(grid)
        }
    
    def _create_rectangle(self, dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create rectangle shape template."""
        width, height = dimensions
        grid_width = int(width // 2.5)
        grid_height = int(height // 2.5)
        
        grid = np.ones((grid_height, grid_width), dtype=bool)
        
        return {
            'name': 'rectangle',
            'grid': grid,
            'dimensions': dimensions,
            'grid_spacing': 2.5,
            'total_points': np.sum(grid)
        }
    
    def _create_circle(self, dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create circle shape template."""
        width, height = dimensions
        radius = min(width, height) / 2
        grid_size = int(radius * 2 // 2.5)
        
        center = grid_size // 2
        y, x = np.ogrid[:grid_size, :grid_size]
        mask = (x - center)**2 + (y - center)**2 <= (grid_size//2)**2
        
        grid = mask.astype(bool)
        
        return {
            'name': 'circle',
            'grid': grid,
            'dimensions': dimensions,
            'grid_spacing': 2.5,
            'total_points': np.sum(grid)
        }
    
    def _create_triangle(self, dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create triangle shape template."""
        width, height = dimensions
        grid_width = int(width // 2.5)
        grid_height = int(height // 2.5)
        
        grid = np.zeros((grid_height, grid_width), dtype=bool)
        
        for i in range(grid_height):
            start_col = int((grid_width / 2) * (grid_height - i) / grid_height)
            end_col = grid_width - start_col
            grid[i, start_col:end_col] = True
        
        return {
            'name': 'triangle',
            'grid': grid,
            'dimensions': dimensions,
            'grid_spacing': 2.5,
            'total_points': np.sum(grid)
        }
    
    def _create_cross(self, dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create cross shape template."""
        width, height = dimensions
        grid_width = int(width // 2.5)
        grid_height = int(height // 2.5)
        
        grid = np.zeros((grid_height, grid_width), dtype=bool)
        
        # Vertical bar
        center_col = grid_width // 2
        bar_width = max(1, grid_width // 8)
        grid[:, center_col-bar_width:center_col+bar_width] = True
        
        # Horizontal bar
        center_row = grid_height // 2
        bar_height = max(1, grid_height // 8)
        grid[center_row-bar_height:center_row+bar_height, :] = True
        
        return {
            'name': 'cross',
            'grid': grid,
            'dimensions': dimensions,
            'grid_spacing': 2.5,
            'total_points': np.sum(grid)
        }
    
    def _create_star(self, dimensions: Tuple[float, float]) -> Dict[str, Any]:
        """Create star shape template."""
        width, height = dimensions
        radius = min(width, height) / 2
        grid_size = int(radius * 2 // 2.5)
        
        center = grid_size // 2
        grid = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Create 5-pointed star
        angles = np.linspace(0, 2*np.pi, 11)  # 5 points + 5 valleys + start
        outer_radius = grid_size // 2
        inner_radius = outer_radius // 2
        
        for i in range(len(angles) - 1):
            if i % 2 == 0:  # Outer point
                r = outer_radius
            else:  # Inner point
                r = inner_radius
            
            x = int(center + r * np.cos(angles[i]))
            y = int(center + r * np.sin(angles[i]))
            
            # Draw line from center to point (simplified)
            for j in range(min(r, outer_radius)):
                px = int(center + j * np.cos(angles[i]))
                py = int(center + j * np.sin(angles[i]))
                if 0 <= px < grid_size and 0 <= py < grid_size:
                    grid[py, px] = True
        
        return {
            'name': 'star',
            'grid': grid,
            'dimensions': dimensions,
            'grid_spacing': 2.5,
            'total_points': np.sum(grid)
        }
    
    def list_available_shapes(self) -> List[str]:
        """Get list of available shape names."""
        return list(self.shapes.keys())
    
    def add_custom_shape(self, name: str, shape_func) -> None:
        """Add custom shape to library."""
        self.shapes[name] = shape_func