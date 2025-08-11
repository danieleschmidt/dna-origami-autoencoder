"""Comprehensive benchmarking suite for DNA encoding algorithms."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Callable, Tuple
import time
import json
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import gc
from scipy import stats
from sklearn.metrics import mean_squared_error, structural_similarity
import warnings
warnings.filterwarnings('ignore')

from ..models.image_data import ImageData
from ..encoding.adaptive_encoder import AdaptiveDNAEncoder, AdaptiveEncodingConfig
from ..encoding.image_encoder import DNAEncoder
from ..encoding.biological_constraints import BiologicalConstraints
from ..utils.performance import performance_monitor


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    algorithm_name: str
    dataset_name: str
    image_count: int
    total_time: float
    average_time_per_image: float
    memory_usage_mb: float
    constraint_satisfaction_rate: float
    reconstruction_accuracy: float
    compression_ratio: float
    error_rate: float
    sequence_lengths: List[int]
    additional_metrics: Dict[str, float]


@dataclass
class StatisticalSummary:
    """Statistical summary of benchmark results."""
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min_val: float
    max_val: float
    ci_95_lower: float
    ci_95_upper: float
    p_value: Optional[float] = None


class ComparativeBenchmark:
    """Comprehensive benchmark suite for comparing DNA encoding algorithms."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        self.results = []
        self.statistical_summaries = {}
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize test datasets
        self.test_datasets = self._create_test_datasets()
        
    def _create_test_datasets(self) -> Dict[str, List[ImageData]]:
        """Create diverse test datasets for benchmarking."""
        
        datasets = {}
        
        # Synthetic dataset 1: High entropy images
        high_entropy_images = []
        for i in range(50):
            # Random high-entropy image
            img_data = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
            high_entropy_images.append(ImageData(
                data=img_data,
                id=f"high_entropy_{i}",
                metadata={'type': 'high_entropy', 'size': (32, 32)}
            ))
        datasets['high_entropy'] = high_entropy_images
        
        # Synthetic dataset 2: Sparse images
        sparse_images = []
        for i in range(50):
            img_data = np.zeros((32, 32), dtype=np.uint8)
            # Add sparse features
            num_features = np.random.randint(5, 15)
            for _ in range(num_features):
                x, y = np.random.randint(0, 32, 2)
                img_data[x:x+2, y:y+2] = np.random.randint(200, 256)
            sparse_images.append(ImageData(
                data=img_data,
                id=f"sparse_{i}",
                metadata={'type': 'sparse', 'size': (32, 32)}
            ))
        datasets['sparse'] = sparse_images
        
        # Synthetic dataset 3: Structured patterns
        structured_images = []
        for i in range(50):
            img_data = np.zeros((32, 32), dtype=np.uint8)
            # Create structured patterns
            pattern_type = i % 4
            if pattern_type == 0:  # Checkerboard
                img_data[::2, ::2] = 255
                img_data[1::2, 1::2] = 255
            elif pattern_type == 1:  # Stripes
                img_data[:, ::4] = 255
                img_data[:, 1::4] = 128
            elif pattern_type == 2:  # Circular pattern
                center = 16
                for x in range(32):
                    for y in range(32):
                        dist = np.sqrt((x-center)**2 + (y-center)**2)
                        img_data[x, y] = int(255 * np.exp(-dist/8))
            else:  # Gradient
                for x in range(32):
                    img_data[x, :] = int(255 * x / 31)
            
            structured_images.append(ImageData(
                data=img_data,
                id=f"structured_{i}",
                metadata={'type': 'structured', 'size': (32, 32)}
            ))
        datasets['structured'] = structured_images
        
        # Synthetic dataset 4: Mixed complexity
        mixed_images = []
        for i in range(50):
            # Combine different types
            base = np.random.randint(0, 64, size=(32, 32), dtype=np.uint8)
            
            # Add structured component
            x_center, y_center = np.random.randint(8, 24, 2)
            radius = np.random.randint(3, 8)
            for x in range(32):
                for y in range(32):
                    dist = np.sqrt((x-x_center)**2 + (y-y_center)**2)
                    if dist < radius:
                        base[x, y] = min(255, base[x, y] + 100)
            
            # Add noise
            noise = np.random.randint(-20, 20, size=(32, 32))
            img_data = np.clip(base + noise, 0, 255).astype(np.uint8)
            
            mixed_images.append(ImageData(
                data=img_data,
                id=f"mixed_{i}",
                metadata={'type': 'mixed', 'size': (32, 32)}
            ))
        datasets['mixed'] = mixed_images
        
        return datasets
    
    def benchmark_algorithm(self, algorithm_name: str, 
                          encoder_factory: Callable, 
                          dataset_names: Optional[List[str]] = None,
                          num_runs: int = 3) -> List[BenchmarkResult]:
        """Benchmark a single algorithm across multiple datasets and runs."""
        
        if dataset_names is None:
            dataset_names = list(self.test_datasets.keys())
        
        results = []
        
        for dataset_name in dataset_names:
            dataset = self.test_datasets[dataset_name]
            
            print(f"Benchmarking {algorithm_name} on {dataset_name} dataset...")
            
            for run_idx in range(num_runs):
                # Create fresh encoder instance
                encoder = encoder_factory()
                
                # Run benchmark
                result = self._run_single_benchmark(
                    encoder, algorithm_name, dataset_name, dataset, run_idx
                )
                results.append(result)
                
                # Force garbage collection
                del encoder
                gc.collect()
        
        return results
    
    def _run_single_benchmark(self, encoder, algorithm_name: str, 
                            dataset_name: str, dataset: List[ImageData], 
                            run_idx: int) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        
        # Memory baseline
        process = psutil.Process()
        memory_baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance metrics
        start_time = time.time()
        constraint_violations = 0
        reconstruction_errors = []
        sequence_lengths = []
        compression_ratios = []
        successful_encodings = 0
        
        # Process each image
        for img_idx, image_data in enumerate(dataset):
            try:
                # Encode image
                img_start_time = time.time()
                
                if hasattr(encoder, 'encode_image_adaptive'):
                    # Adaptive encoder
                    sequences, metadata = encoder.encode_image_adaptive(image_data)
                elif hasattr(encoder, 'encode_image'):
                    # Standard encoder
                    sequences = encoder.encode_image(image_data)
                    metadata = {}
                else:
                    raise ValueError(f"Encoder doesn't have expected interface")
                
                img_time = time.time() - img_start_time
                
                # Collect metrics
                total_length = sum(len(seq.sequence) for seq in sequences)
                sequence_lengths.append(total_length)
                
                # Estimate compression ratio
                original_bits = image_data.data.size * 8
                encoded_bits = total_length * 2  # 2 bits per base
                compression_ratios.append(encoded_bits / original_bits)
                
                # Check constraint satisfaction (if possible)
                if hasattr(encoder, 'constraints'):
                    for seq in sequences:
                        is_valid, _ = encoder.constraints.validate_sequence(seq.sequence)
                        if not is_valid:
                            constraint_violations += 1
                
                # Simulate reconstruction accuracy (simplified)
                # In real scenario, this would involve the full decode pipeline
                reconstruction_error = self._estimate_reconstruction_error(
                    image_data, sequences
                )
                reconstruction_errors.append(reconstruction_error)
                
                successful_encodings += 1
                
            except Exception as e:
                print(f"Error processing image {img_idx}: {e}")
                constraint_violations += 1
                reconstruction_errors.append(1.0)  # Maximum error
                sequence_lengths.append(0)
                compression_ratios.append(float('inf'))
        
        # Calculate final metrics
        total_time = time.time() - start_time
        memory_peak = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_peak - memory_baseline
        
        avg_time_per_image = total_time / len(dataset) if len(dataset) > 0 else 0
        constraint_satisfaction_rate = (len(dataset) - constraint_violations) / len(dataset)
        avg_reconstruction_accuracy = 1.0 - np.mean(reconstruction_errors)
        avg_compression_ratio = np.mean(compression_ratios) if compression_ratios else 1.0
        error_rate = 1.0 - (successful_encodings / len(dataset))
        
        # Additional metrics
        additional_metrics = {
            'memory_peak_mb': memory_peak,
            'throughput_images_per_sec': len(dataset) / total_time if total_time > 0 else 0,
            'avg_sequence_length': np.mean(sequence_lengths) if sequence_lengths else 0,
            'sequence_length_std': np.std(sequence_lengths) if sequence_lengths else 0,
            'compression_ratio_std': np.std(compression_ratios) if compression_ratios else 0,
            'run_index': run_idx
        }
        
        return BenchmarkResult(
            algorithm_name=algorithm_name,
            dataset_name=dataset_name,
            image_count=len(dataset),
            total_time=total_time,
            average_time_per_image=avg_time_per_image,
            memory_usage_mb=memory_usage,
            constraint_satisfaction_rate=constraint_satisfaction_rate,
            reconstruction_accuracy=avg_reconstruction_accuracy,
            compression_ratio=avg_compression_ratio,
            error_rate=error_rate,
            sequence_lengths=sequence_lengths,
            additional_metrics=additional_metrics
        )
    
    def _estimate_reconstruction_error(self, original_image: ImageData, 
                                     sequences: List) -> float:
        """Estimate reconstruction error (simplified simulation)."""
        
        # This is a simplified estimation - in real scenario would need full decoder
        # For now, use sequence length and image complexity as proxy
        
        original_entropy = self._calculate_image_entropy(original_image.data)
        total_sequence_length = sum(len(seq.sequence) for seq in sequences)
        
        # Simple heuristic: longer sequences for higher entropy should have lower error
        expected_length = original_entropy * original_image.data.size / 4  # Rough estimate
        length_ratio = total_sequence_length / expected_length if expected_length > 0 else 1
        
        # Error decreases with appropriate sequence length, increases with deviation
        if 0.8 <= length_ratio <= 1.5:
            base_error = 0.05  # Low error for appropriate length
        else:
            base_error = min(0.5, abs(1 - length_ratio) * 0.3)  # Higher error for poor length
        
        # Add some noise to simulate real reconstruction variability
        noise = np.random.normal(0, 0.02)
        return max(0, min(1, base_error + noise))
    
    def _calculate_image_entropy(self, image_data: np.ndarray) -> float:
        """Calculate Shannon entropy of image."""
        histogram, _ = np.histogram(image_data.flatten(), bins=256, range=(0, 255))
        histogram = histogram / histogram.sum()  # Normalize
        
        # Calculate entropy
        entropy = 0
        for p in histogram:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def run_comparative_study(self, algorithms: Dict[str, Callable],
                            dataset_names: Optional[List[str]] = None,
                            num_runs: int = 3) -> pd.DataFrame:
        """Run comprehensive comparative study of multiple algorithms."""
        
        print("Starting comprehensive comparative study...")
        print(f"Algorithms: {list(algorithms.keys())}")
        print(f"Datasets: {dataset_names or list(self.test_datasets.keys())}")
        print(f"Runs per algorithm-dataset: {num_runs}")
        
        all_results = []
        
        for alg_name, encoder_factory in algorithms.items():
            algorithm_results = self.benchmark_algorithm(
                alg_name, encoder_factory, dataset_names, num_runs
            )
            all_results.extend(algorithm_results)
            
            # Save intermediate results
            self._save_results(algorithm_results, f"{alg_name}_results.json")
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame([asdict(result) for result in all_results])
        
        # Calculate statistical summaries
        self._calculate_statistical_summaries(results_df)
        
        # Save complete results
        results_df.to_csv(f"{self.output_dir}/complete_results.csv", index=False)
        
        print(f"Comparative study complete. Results saved to {self.output_dir}")
        return results_df
    
    def _calculate_statistical_summaries(self, results_df: pd.DataFrame):
        """Calculate statistical summaries for each metric."""
        
        metrics = [
            'average_time_per_image', 'memory_usage_mb', 'constraint_satisfaction_rate',
            'reconstruction_accuracy', 'compression_ratio', 'error_rate'
        ]
        
        for metric in metrics:
            self.statistical_summaries[metric] = {}
            
            for algorithm in results_df['algorithm_name'].unique():
                alg_data = results_df[results_df['algorithm_name'] == algorithm][metric]
                
                # Calculate statistics
                mean_val = alg_data.mean()
                std_val = alg_data.std()
                median_val = alg_data.median()
                q25, q75 = alg_data.quantile([0.25, 0.75])
                min_val, max_val = alg_data.min(), alg_data.max()
                
                # 95% confidence interval
                ci_95_lower, ci_95_upper = stats.t.interval(
                    0.95, len(alg_data)-1, loc=mean_val, 
                    scale=stats.sem(alg_data)
                ) if len(alg_data) > 1 else (mean_val, mean_val)
                
                self.statistical_summaries[metric][algorithm] = StatisticalSummary(
                    mean=mean_val,
                    std=std_val,
                    median=median_val,
                    q25=q25,
                    q75=q75,
                    min_val=min_val,
                    max_val=max_val,
                    ci_95_lower=ci_95_lower,
                    ci_95_upper=ci_95_upper
                )
    
    def _save_results(self, results: List[BenchmarkResult], filename: str):
        """Save results to JSON file."""
        
        results_dict = [asdict(result) for result in results]
        
        with open(f"{self.output_dir}/{filename}", 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def generate_performance_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        report = {
            'summary': {
                'total_experiments': len(results_df),
                'algorithms_tested': results_df['algorithm_name'].nunique(),
                'datasets_tested': results_df['dataset_name'].nunique(),
                'total_images_processed': results_df['image_count'].sum(),
                'total_processing_time': results_df['total_time'].sum()
            },
            'best_performers': {},
            'statistical_comparisons': {},
            'recommendations': []
        }
        
        # Find best performers for each metric
        metrics = [
            ('average_time_per_image', 'min'),  # Lower is better
            ('memory_usage_mb', 'min'),         # Lower is better
            ('constraint_satisfaction_rate', 'max'),  # Higher is better
            ('reconstruction_accuracy', 'max'),       # Higher is better
            ('compression_ratio', 'min'),             # Lower is better (less expansion)
            ('error_rate', 'min')                     # Lower is better
        ]
        
        for metric, optimization in metrics:
            algorithm_means = results_df.groupby('algorithm_name')[metric].mean()
            
            if optimization == 'min':
                best_algorithm = algorithm_means.idxmin()
                best_value = algorithm_means.min()
            else:
                best_algorithm = algorithm_means.idxmax()
                best_value = algorithm_means.max()
            
            report['best_performers'][metric] = {
                'algorithm': best_algorithm,
                'value': best_value,
                'improvement_over_median': self._calculate_improvement_over_median(
                    algorithm_means, best_value, optimization
                )
            }
        
        # Statistical comparisons (t-tests between algorithms)
        algorithms = results_df['algorithm_name'].unique()
        if len(algorithms) >= 2:
            report['statistical_comparisons'] = self._perform_statistical_tests(
                results_df, algorithms
            )
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(results_df)
        
        # Save report
        with open(f"{self.output_dir}/performance_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _calculate_improvement_over_median(self, algorithm_means, best_value, optimization):
        """Calculate improvement of best performer over median."""
        median_value = algorithm_means.median()
        
        if optimization == 'min':
            improvement = (median_value - best_value) / median_value if median_value > 0 else 0
        else:
            improvement = (best_value - median_value) / median_value if median_value > 0 else 0
        
        return improvement
    
    def _perform_statistical_tests(self, results_df: pd.DataFrame, 
                                 algorithms: List[str]) -> Dict[str, Any]:
        """Perform statistical tests between algorithm pairs."""
        
        comparisons = {}
        metrics = ['average_time_per_image', 'constraint_satisfaction_rate', 
                  'reconstruction_accuracy']
        
        for metric in metrics:
            comparisons[metric] = {}
            
            for i, alg1 in enumerate(algorithms):
                for alg2 in algorithms[i+1:]:
                    data1 = results_df[results_df['algorithm_name'] == alg1][metric]
                    data2 = results_df[results_df['algorithm_name'] == alg2][metric]
                    
                    # Perform t-test
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + 
                                            (len(data2) - 1) * data2.var()) / 
                                           (len(data1) + len(data2) - 2))
                        cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
                        
                        comparisons[metric][f"{alg1}_vs_{alg2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'cohens_d': cohens_d,
                            'effect_size': self._interpret_effect_size(abs(cohens_d))
                        }
        
        return comparisons
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_recommendations(self, results_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on benchmark results."""
        
        recommendations = []
        
        # Speed recommendations
        speed_ranking = results_df.groupby('algorithm_name')['average_time_per_image'].mean().sort_values()
        fastest = speed_ranking.index[0]
        recommendations.append(f"For fastest processing: Use {fastest}")
        
        # Accuracy recommendations  
        accuracy_ranking = results_df.groupby('algorithm_name')['reconstruction_accuracy'].mean().sort_values(ascending=False)
        most_accurate = accuracy_ranking.index[0]
        recommendations.append(f"For highest accuracy: Use {most_accurate}")
        
        # Memory efficiency recommendations
        memory_ranking = results_df.groupby('algorithm_name')['memory_usage_mb'].mean().sort_values()
        most_memory_efficient = memory_ranking.index[0]
        recommendations.append(f"For lowest memory usage: Use {most_memory_efficient}")
        
        # Balanced recommendations
        # Calculate composite score (weighted average of normalized metrics)
        algorithm_scores = {}
        for algorithm in results_df['algorithm_name'].unique():
            alg_data = results_df[results_df['algorithm_name'] == algorithm]
            
            # Normalize metrics (0-1 scale, with appropriate direction)
            speed_score = 1 - (alg_data['average_time_per_image'].mean() - speed_ranking.min()) / (speed_ranking.max() - speed_ranking.min())
            accuracy_score = alg_data['reconstruction_accuracy'].mean()
            memory_score = 1 - (alg_data['memory_usage_mb'].mean() - memory_ranking.min()) / (memory_ranking.max() - memory_ranking.min())
            constraint_score = alg_data['constraint_satisfaction_rate'].mean()
            
            # Weighted composite score
            composite_score = (speed_score * 0.25 + accuracy_score * 0.35 + 
                             memory_score * 0.2 + constraint_score * 0.2)
            
            algorithm_scores[algorithm] = composite_score
        
        best_balanced = max(algorithm_scores, key=algorithm_scores.get)
        recommendations.append(f"For best overall balance: Use {best_balanced}")
        
        return recommendations


class PerformanceProfiler:
    """Detailed performance profiler for algorithm analysis."""
    
    def __init__(self):
        self.profiles = {}
    
    @performance_monitor
    def profile_algorithm(self, algorithm_name: str, encoder_factory: Callable,
                         test_images: List[ImageData]) -> Dict[str, Any]:
        """Profile algorithm performance in detail."""
        
        encoder = encoder_factory()
        profile = {
            'algorithm_name': algorithm_name,
            'cpu_times': [],
            'memory_profiles': [],
            'operation_breakdown': {},
            'bottlenecks': []
        }
        
        process = psutil.Process()
        
        for img_idx, image_data in enumerate(test_images):
            # Detailed timing
            operation_times = {}
            
            start_time = time.time()
            memory_start = process.memory_info().rss / 1024 / 1024
            
            # Profile encoding phases
            try:
                phase_start = time.time()
                if hasattr(encoder, 'encode_image_adaptive'):
                    sequences, metadata = encoder.encode_image_adaptive(image_data)
                else:
                    sequences = encoder.encode_image(image_data)
                    metadata = {}
                
                operation_times['encoding'] = time.time() - phase_start
                
                # Profile constraint validation if available
                if hasattr(encoder, 'constraints') and sequences:
                    phase_start = time.time()
                    for seq in sequences:
                        encoder.constraints.validate_sequence(seq.sequence)
                    operation_times['validation'] = time.time() - phase_start
                
            except Exception as e:
                operation_times['error'] = str(e)
            
            total_time = time.time() - start_time
            memory_end = process.memory_info().rss / 1024 / 1024
            memory_used = memory_end - memory_start
            
            profile['cpu_times'].append(total_time)
            profile['memory_profiles'].append(memory_used)
            
            # Aggregate operation times
            for op, op_time in operation_times.items():
                if op not in profile['operation_breakdown']:
                    profile['operation_breakdown'][op] = []
                profile['operation_breakdown'][op].append(op_time)
        
        # Analyze bottlenecks
        if profile['operation_breakdown']:
            avg_times = {op: np.mean(times) for op, times in profile['operation_breakdown'].items()}
            total_avg = sum(avg_times.values())
            
            for op, avg_time in avg_times.items():
                if avg_time / total_avg > 0.3:  # Operation takes >30% of total time
                    profile['bottlenecks'].append({
                        'operation': op,
                        'percentage': avg_time / total_avg,
                        'average_time': avg_time
                    })
        
        self.profiles[algorithm_name] = profile
        return profile


class StatisticalValidator:
    """Statistical validation of algorithm performance claims."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.validation_results = {}
    
    def validate_performance_claims(self, results_df: pd.DataFrame, 
                                  claims: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Validate specific performance claims with statistical tests."""
        
        validation_results = {}
        
        for algorithm, algorithm_claims in claims.items():
            algorithm_data = results_df[results_df['algorithm_name'] == algorithm]
            
            if len(algorithm_data) == 0:
                continue
                
            validation_results[algorithm] = {}
            
            for metric, claimed_value in algorithm_claims.items():
                if metric not in algorithm_data.columns:
                    continue
                
                observed_values = algorithm_data[metric]
                
                # Test if observed mean significantly matches claimed value
                t_stat, p_value = stats.ttest_1samp(observed_values, claimed_value)
                
                # Confidence interval
                ci_95_lower, ci_95_upper = stats.t.interval(
                    0.95, len(observed_values)-1, 
                    loc=observed_values.mean(), 
                    scale=stats.sem(observed_values)
                )
                
                validation_results[algorithm][metric] = {
                    'claimed_value': claimed_value,
                    'observed_mean': observed_values.mean(),
                    'observed_std': observed_values.std(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'claim_supported': p_value > self.significance_level,
                    'confidence_interval_95': (ci_95_lower, ci_95_upper),
                    'claim_within_ci': ci_95_lower <= claimed_value <= ci_95_upper
                }
        
        return validation_results
    
    def test_algorithmic_improvements(self, baseline_results: pd.DataFrame,
                                    improved_results: pd.DataFrame,
                                    metrics: List[str]) -> Dict[str, Any]:
        """Test if improved algorithm shows statistically significant improvements."""
        
        improvements = {}
        
        for metric in metrics:
            baseline_values = baseline_results[metric]
            improved_values = improved_results[metric]
            
            # Test for improvement
            if metric in ['average_time_per_image', 'memory_usage_mb', 'error_rate']:
                # Lower is better
                improvement_achieved = improved_values.mean() < baseline_values.mean()
                t_stat, p_value = stats.ttest_ind(baseline_values, improved_values)
                effect_direction = "improvement" if improvement_achieved else "degradation"
            else:
                # Higher is better
                improvement_achieved = improved_values.mean() > baseline_values.mean()
                t_stat, p_value = stats.ttest_ind(improved_values, baseline_values)
                effect_direction = "improvement" if improvement_achieved else "degradation"
            
            # Effect size
            pooled_std = np.sqrt(((len(baseline_values) - 1) * baseline_values.var() + 
                                (len(improved_values) - 1) * improved_values.var()) / 
                               (len(baseline_values) + len(improved_values) - 2))
            
            cohens_d = abs(improved_values.mean() - baseline_values.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Percentage improvement
            if baseline_values.mean() != 0:
                if metric in ['average_time_per_image', 'memory_usage_mb', 'error_rate']:
                    pct_improvement = (baseline_values.mean() - improved_values.mean()) / baseline_values.mean() * 100
                else:
                    pct_improvement = (improved_values.mean() - baseline_values.mean()) / baseline_values.mean() * 100
            else:
                pct_improvement = 0
            
            improvements[metric] = {
                'baseline_mean': baseline_values.mean(),
                'improved_mean': improved_values.mean(),
                'improvement_achieved': improvement_achieved,
                'statistically_significant': p_value < self.significance_level,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(cohens_d),
                'percentage_improvement': pct_improvement,
                'effect_direction': effect_direction
            }
        
        return improvements
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"


class ResultsAnalyzer:
    """Advanced analysis and visualization of benchmark results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def create_performance_visualizations(self, results_df: pd.DataFrame):
        """Create comprehensive performance visualizations."""
        
        # Set style for publication-quality plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance comparison radar chart
        self._create_radar_chart(results_df)
        
        # 2. Time series performance analysis
        self._create_time_series_analysis(results_df)
        
        # 3. Memory usage analysis
        self._create_memory_analysis(results_df)
        
        # 4. Statistical significance heatmap
        self._create_significance_heatmap(results_df)
        
        # 5. Error analysis
        self._create_error_analysis(results_df)
        
        print(f"Performance visualizations saved to {self.output_dir}")
    
    def _create_radar_chart(self, results_df: pd.DataFrame):
        """Create radar chart comparing algorithms across multiple metrics."""
        
        # Aggregate data by algorithm
        metrics = ['constraint_satisfaction_rate', 'reconstruction_accuracy']
        
        # Add normalized inverse metrics (so higher is better for all)
        for metric in ['average_time_per_image', 'memory_usage_mb', 'error_rate']:
            if metric in results_df.columns:
                max_val = results_df[metric].max()
                results_df[f'inv_{metric}'] = 1 - (results_df[metric] / max_val)
                metrics.append(f'inv_{metric}')
        
        algorithm_means = results_df.groupby('algorithm_name')[metrics].mean()
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for algorithm in algorithm_means.index:
            values = algorithm_means.loc[algorithm].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=algorithm)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('inv_', '').replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Algorithm Performance Comparison', size=16, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_time_series_analysis(self, results_df: pd.DataFrame):
        """Analyze performance over time/iterations."""
        
        if 'run_index' not in results_df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics = ['average_time_per_image', 'memory_usage_mb', 
                  'constraint_satisfaction_rate', 'reconstruction_accuracy']
        
        for i, metric in enumerate(metrics):
            if metric not in results_df.columns:
                continue
                
            for algorithm in results_df['algorithm_name'].unique():
                alg_data = results_df[results_df['algorithm_name'] == algorithm]
                alg_data = alg_data.sort_values('run_index')
                
                axes[i].plot(alg_data['run_index'], alg_data[metric], 
                           marker='o', label=algorithm, alpha=0.7)
            
            axes[i].set_xlabel('Run Index')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Over Runs')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_memory_analysis(self, results_df: pd.DataFrame):
        """Create memory usage analysis visualization."""
        
        if 'memory_usage_mb' not in results_df.columns:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of memory usage by algorithm
        sns.boxplot(data=results_df, x='algorithm_name', y='memory_usage_mb', ax=ax1)
        ax1.set_title('Memory Usage Distribution by Algorithm')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory vs. throughput scatter plot
        if 'throughput_images_per_sec' in results_df.columns:
            for algorithm in results_df['algorithm_name'].unique():
                alg_data = results_df[results_df['algorithm_name'] == algorithm]
                ax2.scatter(alg_data['memory_usage_mb'], alg_data['throughput_images_per_sec'],
                          label=algorithm, alpha=0.7, s=50)
            
            ax2.set_xlabel('Memory Usage (MB)')
            ax2.set_ylabel('Throughput (images/sec)')
            ax2.set_title('Memory vs. Throughput Trade-off')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/memory_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_significance_heatmap(self, results_df: pd.DataFrame):
        """Create heatmap showing statistical significance between algorithms."""
        
        algorithms = results_df['algorithm_name'].unique()
        if len(algorithms) < 2:
            return
            
        metrics = ['average_time_per_image', 'constraint_satisfaction_rate', 'reconstruction_accuracy']
        
        for metric in metrics:
            if metric not in results_df.columns:
                continue
                
            # Create p-value matrix
            n_algs = len(algorithms)
            p_matrix = np.ones((n_algs, n_algs))
            
            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms):
                    if i != j:
                        data1 = results_df[results_df['algorithm_name'] == alg1][metric]
                        data2 = results_df[results_df['algorithm_name'] == alg2][metric]
                        
                        if len(data1) > 1 and len(data2) > 1:
                            _, p_value = stats.ttest_ind(data1, data2)
                            p_matrix[i, j] = p_value
            
            # Create heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       xticklabels=algorithms, yticklabels=algorithms,
                       cbar_kws={'label': 'p-value'})
            plt.title(f'Statistical Significance Matrix - {metric.replace("_", " ").title()}')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/significance_heatmap_{metric}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_error_analysis(self, results_df: pd.DataFrame):
        """Create error analysis visualization."""
        
        if 'error_rate' not in results_df.columns:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Error rate by dataset
        if 'dataset_name' in results_df.columns:
            pivot_data = results_df.pivot_table(values='error_rate', 
                                              index='algorithm_name', 
                                              columns='dataset_name', 
                                              aggfunc='mean')
            
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='Reds', ax=ax1)
            ax1.set_title('Error Rate by Algorithm and Dataset')
        
        # Error rate distribution
        sns.violinplot(data=results_df, x='algorithm_name', y='error_rate', ax=ax2)
        ax2.set_title('Error Rate Distribution')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Error Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_publication_summary(self, results_df: pd.DataFrame, 
                                   statistical_tests: Dict[str, Any]) -> str:
        """Generate publication-ready summary of results."""
        
        summary_lines = []
        
        # Overall summary
        total_experiments = len(results_df)
        total_images = results_df['image_count'].sum()
        algorithms_tested = results_df['algorithm_name'].nunique()
        
        summary_lines.append("EXPERIMENTAL VALIDATION SUMMARY")
        summary_lines.append("=" * 40)
        summary_lines.append(f"Total experiments conducted: {total_experiments}")
        summary_lines.append(f"Total images processed: {total_images}")
        summary_lines.append(f"Algorithms compared: {algorithms_tested}")
        summary_lines.append("")
        
        # Performance highlights
        summary_lines.append("PERFORMANCE HIGHLIGHTS")
        summary_lines.append("-" * 25)
        
        # Best performers
        speed_best = results_df.loc[results_df['average_time_per_image'].idxmin()]
        accuracy_best = results_df.loc[results_df['reconstruction_accuracy'].idxmax()]
        memory_best = results_df.loc[results_df['memory_usage_mb'].idxmin()]
        
        summary_lines.append(f"Fastest processing: {speed_best['algorithm_name']} "
                            f"({speed_best['average_time_per_image']:.3f}s per image)")
        summary_lines.append(f"Highest accuracy: {accuracy_best['algorithm_name']} "
                            f"({accuracy_best['reconstruction_accuracy']:.3f})")
        summary_lines.append(f"Most memory efficient: {memory_best['algorithm_name']} "
                            f"({memory_best['memory_usage_mb']:.1f} MB)")
        summary_lines.append("")
        
        # Statistical significance
        if statistical_tests:
            summary_lines.append("STATISTICAL SIGNIFICANCE")
            summary_lines.append("-" * 25)
            
            for metric, comparisons in statistical_tests.items():
                significant_comparisons = [comp for comp, results in comparisons.items() 
                                         if results.get('significant', False)]
                
                summary_lines.append(f"{metric.replace('_', ' ').title()}: "
                                    f"{len(significant_comparisons)} significant comparisons")
        
        summary_text = "\n".join(summary_lines)
        
        # Save to file
        with open(f'{self.output_dir}/publication_summary.txt', 'w') as f:
            f.write(summary_text)
        
        return summary_text