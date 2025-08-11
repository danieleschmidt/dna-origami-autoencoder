"""
Optimization module for DNA Origami AutoEncoder.

Provides comprehensive optimization capabilities including:
- Performance optimization
- Memory optimization
- GPU acceleration
- Distributed processing
- Adaptive algorithms
- Caching strategies
"""

from .performance_optimizer import (
    PerformanceOptimizer,
    OptimizationStrategy,
    PerformanceBenchmark,
    AdaptiveOptimizer
)

from .integration_optimizer import (
    IntegratedOptimizer,
    OptimizationConfig,
    SystemOptimizer,
    AutoTuner
)

__all__ = [
    'PerformanceOptimizer',
    'OptimizationStrategy', 
    'PerformanceBenchmark',
    'AdaptiveOptimizer',
    'IntegratedOptimizer',
    'OptimizationConfig',
    'SystemOptimizer',
    'AutoTuner'
]