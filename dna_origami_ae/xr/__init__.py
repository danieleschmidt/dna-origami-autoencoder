"""
XR/VR Integration Module for DNA Origami AutoEncoder

This module provides extended reality interfaces for visualizing and interacting
with DNA origami structures in virtual and augmented reality environments.
"""

from .mesh_visualizer import MeshVisualizer
from .agent_xr_interface import AgentXRInterface
from .immersive_designer import ImmersiveDesigner
from .spatial_optimizer import SpatialOptimizer

__all__ = [
    'MeshVisualizer',
    'AgentXRInterface', 
    'ImmersiveDesigner',
    'SpatialOptimizer'
]