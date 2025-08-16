"""Research module for novel algorithmic contributions and experimental validation."""

# Conditional imports for maximum compatibility
__all__ = []

# Existing research components
try:
    from .benchmark_suite import (
        ComparativeBenchmark,
        PerformanceProfiler,
        StatisticalValidator,
        ResultsAnalyzer
    )
    __all__.extend(['ComparativeBenchmark', 'PerformanceProfiler', 'StatisticalValidator', 'ResultsAnalyzer'])
except ImportError:
    pass

# Enhanced research components
try:
    from .novel_algorithms import (
        NovelAlgorithmConfig,
        QuantumInspiredEncoder,
        AdaptiveFoldingPredictor,
        BiomimeticOptimizer,
        apply_novel_algorithms
    )
    __all__.extend(['NovelAlgorithmConfig', 'QuantumInspiredEncoder', 'AdaptiveFoldingPredictor', 
                    'BiomimeticOptimizer', 'apply_novel_algorithms'])
except ImportError:
    pass

try:
    from .adaptive_learning import (
        LearningConfig,
        ContinualLearningSystem,
        create_adaptive_learning_system
    )
    __all__.extend(['LearningConfig', 'ContinualLearningSystem', 'create_adaptive_learning_system'])
except ImportError:
    pass