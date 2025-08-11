"""Research module for novel algorithmic contributions and experimental validation."""

from .benchmark_suite import (
    ComparativeBenchmark,
    PerformanceProfiler,
    StatisticalValidator,
    ResultsAnalyzer
)
from .experimental_validation import (
    ExperimentalFramework,
    BaselineComparator,
    NovelAlgorithmValidator,
    PublicationDataGenerator
)
from .algorithmic_contributions import (
    NovelEncodingAlgorithms,
    HybridConstraintSolver,
    AdaptiveLearningSystem,
    BiologicalOptimizer
)

__all__ = [
    'ComparativeBenchmark',
    'PerformanceProfiler', 
    'StatisticalValidator',
    'ResultsAnalyzer',
    'ExperimentalFramework',
    'BaselineComparator',
    'NovelAlgorithmValidator',
    'PublicationDataGenerator',
    'NovelEncodingAlgorithms',
    'HybridConstraintSolver',
    'AdaptiveLearningSystem',
    'BiologicalOptimizer'
]