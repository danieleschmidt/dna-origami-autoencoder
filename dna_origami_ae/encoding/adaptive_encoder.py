"""Advanced adaptive DNA encoder with machine learning optimization."""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..models.image_data import ImageData
from ..models.dna_sequence import DNASequence
from .biological_constraints import BiologicalConstraints
from .image_encoder import Base4Encoder, EncodingParameters
from ..utils.performance import adaptive_cache, performance_monitor


@dataclass
class AdaptiveEncodingConfig:
    """Configuration for adaptive encoding algorithms."""
    
    # ML-based optimization
    enable_ml_optimization: bool = True
    constraint_learning_rate: float = 0.01
    optimization_window: int = 1000
    
    # Parallel processing
    max_workers: int = 4
    use_process_pool: bool = False
    chunk_parallel_threshold: int = 100
    
    # Adaptive algorithms
    enable_genetic_optimization: bool = True
    enable_clustering_optimization: bool = True
    enable_dynamic_constraint_adaptation: bool = True
    
    # Performance tracking
    enable_performance_profiling: bool = True
    cache_encoding_patterns: bool = True
    adaptive_chunk_sizing: bool = True


class GeneticConstraintOptimizer:
    """Genetic algorithm for optimizing DNA constraint satisfaction."""
    
    def __init__(self, constraints: BiologicalConstraints):
        self.constraints = constraints
        self.population_size = 50
        self.generations = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
    def optimize_encoding(self, binary_data: np.ndarray, 
                         base_encoder: Base4Encoder) -> Tuple[str, float]:
        """Use genetic algorithm to find optimal encoding strategy."""
        
        def fitness_function(params):
            """Evaluate fitness of encoding parameters."""
            try:
                # Modify encoding based on parameters
                padding_size = int(params[0] * 10)  # 0-10 bases padding
                permutation_seed = int(params[1] * 1000)  # Random permutation
                
                # Apply permutation to binary data
                np.random.seed(permutation_seed)
                if len(binary_data) > padding_size:
                    permuted_data = np.concatenate([
                        binary_data,
                        np.random.choice([0, 1], size=padding_size)
                    ])
                else:
                    permuted_data = binary_data
                    
                # Encode with modified data
                dna_sequence = base_encoder.encode_binary_to_dna(permuted_data)
                
                # Evaluate constraints
                is_valid, violations = self.constraints.validate_sequence(dna_sequence)
                
                # Fitness = constraint satisfaction + sequence quality
                constraint_score = 1.0 if is_valid else max(0.1, 1.0 - len(violations) * 0.1)
                length_penalty = min(1.0, 200.0 / len(dna_sequence))  # Prefer shorter sequences
                gc_balance = self._evaluate_gc_content(dna_sequence)
                
                return constraint_score * 0.6 + length_penalty * 0.2 + gc_balance * 0.2
                
            except Exception:
                return 0.01  # Very low fitness for invalid solutions
        
        # Run differential evolution optimization
        bounds = [(0, 1), (0, 1)]  # Normalized parameter bounds
        result = differential_evolution(
            lambda x: -fitness_function(x),  # Minimize negative fitness
            bounds,
            maxiter=self.generations,
            popsize=15,
            seed=42
        )
        
        # Generate optimized sequence
        if result.success:
            best_params = result.x
            padding_size = int(best_params[0] * 10)
            permutation_seed = int(best_params[1] * 1000)
            
            np.random.seed(permutation_seed)
            optimized_data = np.concatenate([
                binary_data,
                np.random.choice([0, 1], size=padding_size)
            ]) if len(binary_data) > padding_size else binary_data
            
            dna_sequence = base_encoder.encode_binary_to_dna(optimized_data)
            return dna_sequence, -result.fun
        else:
            # Fallback to standard encoding
            return base_encoder.encode_binary_to_dna(binary_data), 0.5
    
    def _evaluate_gc_content(self, sequence: str) -> float:
        """Evaluate GC content balance."""
        gc_count = sequence.count('G') + sequence.count('C')
        gc_ratio = gc_count / len(sequence) if len(sequence) > 0 else 0
        
        # Optimal GC content is around 50%
        return 1.0 - abs(gc_ratio - 0.5) * 2


class ClusteringBasedEncoder:
    """Clustering-based encoding optimization for similar image patterns."""
    
    def __init__(self, config: AdaptiveEncodingConfig):
        self.config = config
        self.pattern_clusters = {}
        self.cluster_models = {}
        self.encoding_cache = {}
        
    @adaptive_cache(max_size=10000, ttl=3600)
    def encode_with_clustering(self, image_data: ImageData, 
                              base_encoder: Base4Encoder) -> Tuple[str, Dict]:
        """Encode using clustering-based pattern recognition."""
        
        # Extract image features for clustering
        features = self._extract_image_features(image_data)
        
        # Find or create cluster
        cluster_id = self._find_cluster(features)
        
        if cluster_id in self.encoding_cache:
            # Use cached encoding strategy for this cluster
            strategy = self.encoding_cache[cluster_id]
            return self._apply_encoding_strategy(image_data, strategy, base_encoder)
        else:
            # Develop new encoding strategy for this cluster
            strategy = self._develop_encoding_strategy(image_data, features)
            self.encoding_cache[cluster_id] = strategy
            return self._apply_encoding_strategy(image_data, strategy, base_encoder)
    
    def _extract_image_features(self, image_data: ImageData) -> np.ndarray:
        """Extract statistical features from image for clustering."""
        img_array = image_data.data
        
        features = [
            np.mean(img_array),           # Mean intensity
            np.std(img_array),            # Standard deviation
            np.max(img_array) - np.min(img_array),  # Dynamic range
            len(np.unique(img_array)),    # Unique values
            np.sum(np.diff(img_array.flatten()) != 0),  # Transitions
            np.mean(np.abs(np.gradient(img_array))),    # Edge content
            np.var(img_array),            # Variance
            np.sum(img_array > np.mean(img_array)) / img_array.size,  # High value ratio
        ]
        
        return np.array(features)
    
    def _find_cluster(self, features: np.ndarray) -> int:
        """Find appropriate cluster for image features."""
        feature_key = tuple(np.round(features, 3))
        
        if not self.pattern_clusters:
            # Initialize first cluster
            self.pattern_clusters[0] = [features]
            return 0
        
        # Find closest cluster
        min_distance = float('inf')
        closest_cluster = 0
        
        for cluster_id, cluster_features in self.pattern_clusters.items():
            centroid = np.mean(cluster_features, axis=0)
            distance = np.linalg.norm(features - centroid)
            
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        
        # Add to closest cluster if distance is reasonable
        threshold = np.std([np.linalg.norm(f) for cluster_features in self.pattern_clusters.values() 
                           for f in cluster_features]) if len(self.pattern_clusters) > 1 else 1.0
        
        if min_distance < threshold:
            self.pattern_clusters[closest_cluster].append(features)
            return closest_cluster
        else:
            # Create new cluster
            new_cluster_id = max(self.pattern_clusters.keys()) + 1
            self.pattern_clusters[new_cluster_id] = [features]
            return new_cluster_id
    
    def _develop_encoding_strategy(self, image_data: ImageData, 
                                  features: np.ndarray) -> Dict:
        """Develop encoding strategy based on image characteristics."""
        
        # Analyze image characteristics
        entropy = -np.sum(np.histogram(image_data.data, bins=256)[0] / image_data.data.size * 
                         np.log2(np.histogram(image_data.data, bins=256)[0] / image_data.data.size + 1e-10))
        
        sparsity = np.sum(image_data.data == 0) / image_data.data.size
        edge_density = np.mean(np.abs(np.gradient(image_data.data)))
        
        # Choose strategy based on characteristics
        if entropy > 7.0:  # High entropy - use compression
            strategy = {
                'compression': True,
                'chunk_size': 150,
                'error_correction_strength': 0.25,
                'constraint_strictness': 0.8
            }
        elif sparsity > 0.7:  # Sparse image - use run-length encoding
            strategy = {
                'compression': True,
                'chunk_size': 300,
                'error_correction_strength': 0.2,
                'constraint_strictness': 0.6
            }
        elif edge_density > 50:  # High edge density - prioritize accuracy
            strategy = {
                'compression': False,
                'chunk_size': 100,
                'error_correction_strength': 0.4,
                'constraint_strictness': 0.9
            }
        else:  # Balanced strategy
            strategy = {
                'compression': False,
                'chunk_size': 200,
                'error_correction_strength': 0.3,
                'constraint_strictness': 0.7
            }
        
        return strategy
    
    def _apply_encoding_strategy(self, image_data: ImageData, strategy: Dict,
                               base_encoder: Base4Encoder) -> Tuple[str, Dict]:
        """Apply encoding strategy to image data."""
        
        # Convert image to binary with strategy parameters
        binary_data = image_data.to_binary()
        
        # Apply compression if specified
        if strategy['compression']:
            # Simple run-length encoding for demonstration
            binary_data = self._compress_binary_rle(binary_data)
        
        # Encode with appropriate chunk size
        chunk_size = strategy['chunk_size']
        sequences = []
        
        for i in range(0, len(binary_data), chunk_size * 2):
            chunk = binary_data[i:i + chunk_size * 2]
            if len(chunk) % 2 != 0:
                chunk = np.append(chunk, 0)
            
            dna_chunk = base_encoder.encode_binary_to_dna(chunk)
            sequences.append(dna_chunk)
        
        # Combine sequences
        final_sequence = ''.join(sequences)
        
        strategy_metadata = {
            'strategy': strategy,
            'cluster_based': True,
            'chunks_count': len(sequences),
            'compression_used': strategy['compression']
        }
        
        return final_sequence, strategy_metadata
    
    def _compress_binary_rle(self, binary_data: np.ndarray) -> np.ndarray:
        """Simple run-length encoding for binary data."""
        if len(binary_data) == 0:
            return binary_data
            
        compressed = []
        current_bit = binary_data[0]
        count = 1
        
        for i in range(1, len(binary_data)):
            if binary_data[i] == current_bit and count < 255:
                count += 1
            else:
                # Encode run as: bit + count (8 bits each)
                compressed.extend([current_bit] + [int(b) for b in format(count, '08b')])
                current_bit = binary_data[i]
                count = 1
        
        # Add final run
        compressed.extend([current_bit] + [int(b) for b in format(count, '08b')])
        
        return np.array(compressed, dtype=np.uint8)


class AdaptiveDNAEncoder:
    """Advanced adaptive encoder with ML optimization."""
    
    def __init__(self, config: Optional[AdaptiveEncodingConfig] = None,
                 constraints: Optional[BiologicalConstraints] = None):
        self.config = config or AdaptiveEncodingConfig()
        self.constraints = constraints or BiologicalConstraints()
        
        # Initialize components
        self.base_encoder = Base4Encoder(self.constraints)
        self.genetic_optimizer = GeneticConstraintOptimizer(self.constraints)
        self.clustering_encoder = ClusteringBasedEncoder(self.config)
        
        # Performance tracking
        self.encoding_history = []
        self.performance_metrics = {
            'total_encodings': 0,
            'successful_encodings': 0,
            'average_constraint_satisfaction': 0.0,
            'average_encoding_time': 0.0,
            'optimization_improvements': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
    
    @performance_monitor
    def encode_image_adaptive(self, image_data: ImageData, 
                            params: Optional[EncodingParameters] = None) -> Tuple[List[DNASequence], Dict]:
        """Encode image using adaptive algorithms."""
        
        start_time = time.time()
        params = params or EncodingParameters()
        
        try:
            # Choose encoding strategy based on image and configuration
            if self.config.enable_clustering_optimization:
                # Use clustering-based encoding
                dna_sequence, metadata = self.clustering_encoder.encode_with_clustering(
                    image_data, self.base_encoder
                )
                
                # Apply genetic optimization if enabled
                if self.config.enable_genetic_optimization and len(dna_sequence) > 100:
                    binary_data = image_data.to_binary()
                    optimized_sequence, fitness = self.genetic_optimizer.optimize_encoding(
                        binary_data, self.base_encoder
                    )
                    
                    # Use optimized sequence if significantly better
                    original_valid, _ = self.constraints.validate_sequence(dna_sequence)
                    optimized_valid, _ = self.constraints.validate_sequence(optimized_sequence)
                    
                    if optimized_valid and (not original_valid or fitness > 0.8):
                        dna_sequence = optimized_sequence
                        metadata['genetic_optimization'] = True
                        metadata['fitness_score'] = fitness
                        
                        with self._lock:
                            self.performance_metrics['optimization_improvements'] += 1
            
            else:
                # Use standard encoding with optional genetic optimization
                binary_data = image_data.to_binary()
                
                if self.config.enable_genetic_optimization:
                    dna_sequence, fitness = self.genetic_optimizer.optimize_encoding(
                        binary_data, self.base_encoder
                    )
                    metadata = {'genetic_optimization': True, 'fitness_score': fitness}
                else:
                    dna_sequence = self.base_encoder.encode_binary_to_dna(binary_data)
                    metadata = {'standard_encoding': True}
            
            # Create DNASequence objects
            sequences = [DNASequence(
                sequence=dna_sequence,
                metadata={
                    'image_id': getattr(image_data, 'id', 'unknown'),
                    'encoding_method': 'adaptive',
                    'timestamp': time.time(),
                    **metadata
                }
            )]
            
            # Update performance metrics
            encoding_time = time.time() - start_time
            is_valid, _ = self.constraints.validate_sequence(dna_sequence)
            
            with self._lock:
                self.performance_metrics['total_encodings'] += 1
                if is_valid:
                    self.performance_metrics['successful_encodings'] += 1
                
                # Update running averages
                n = self.performance_metrics['total_encodings']
                self.performance_metrics['average_encoding_time'] = (
                    (self.performance_metrics['average_encoding_time'] * (n-1) + encoding_time) / n
                )
                
                constraint_score = 1.0 if is_valid else 0.5
                self.performance_metrics['average_constraint_satisfaction'] = (
                    (self.performance_metrics['average_constraint_satisfaction'] * (n-1) + constraint_score) / n
                )
            
            # Store encoding history for learning
            if len(self.encoding_history) > self.config.optimization_window:
                self.encoding_history.pop(0)
            
            self.encoding_history.append({
                'image_features': self.clustering_encoder._extract_image_features(image_data),
                'sequence_length': len(dna_sequence),
                'constraint_valid': is_valid,
                'encoding_time': encoding_time,
                'metadata': metadata
            })
            
            return sequences, {
                'encoding_time': encoding_time,
                'constraint_valid': is_valid,
                'sequence_length': len(dna_sequence),
                'optimization_applied': metadata.get('genetic_optimization', False),
                'performance_metrics': self.performance_metrics.copy()
            }
            
        except Exception as e:
            # Fallback to basic encoding
            binary_data = image_data.to_binary()
            dna_sequence = self.base_encoder.encode_binary_to_dna(binary_data)
            
            sequences = [DNASequence(
                sequence=dna_sequence,
                metadata={
                    'image_id': getattr(image_data, 'id', 'unknown'),
                    'encoding_method': 'fallback',
                    'error': str(e),
                    'timestamp': time.time()
                }
            )]
            
            return sequences, {
                'encoding_time': time.time() - start_time,
                'error': str(e),
                'fallback_used': True
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        with self._lock:
            success_rate = (self.performance_metrics['successful_encodings'] / 
                          max(1, self.performance_metrics['total_encodings']))
            
            report = {
                'total_encodings': self.performance_metrics['total_encodings'],
                'success_rate': success_rate,
                'average_encoding_time': self.performance_metrics['average_encoding_time'],
                'average_constraint_satisfaction': self.performance_metrics['average_constraint_satisfaction'],
                'optimization_improvements': self.performance_metrics['optimization_improvements'],
                'clustering_enabled': self.config.enable_clustering_optimization,
                'genetic_optimization_enabled': self.config.enable_genetic_optimization,
                'cluster_count': len(self.clustering_encoder.pattern_clusters),
                'cache_size': len(self.clustering_encoder.encoding_cache)
            }
        
        # Add historical analysis if available
        if len(self.encoding_history) > 10:
            encoding_times = [h['encoding_time'] for h in self.encoding_history[-50:]]
            sequence_lengths = [h['sequence_length'] for h in self.encoding_history[-50:]]
            
            report.update({
                'recent_avg_time': np.mean(encoding_times),
                'recent_time_std': np.std(encoding_times),
                'recent_avg_length': np.mean(sequence_lengths),
                'recent_length_std': np.std(sequence_lengths),
                'improvement_trend': self._calculate_improvement_trend()
            })
        
        return report
    
    def _calculate_improvement_trend(self) -> float:
        """Calculate performance improvement trend over recent encodings."""
        if len(self.encoding_history) < 20:
            return 0.0
        
        recent_scores = []
        for entry in self.encoding_history[-20:]:
            score = (1.0 if entry['constraint_valid'] else 0.5) / max(0.1, entry['encoding_time'])
            recent_scores.append(score)
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_scores))
        coefficients = np.polyfit(x, recent_scores, 1)
        return coefficients[0]  # Slope indicates trend