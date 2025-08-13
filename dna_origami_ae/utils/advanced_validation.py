"""Advanced validation and quality assurance for DNA origami encoding."""

import numpy as np
import time
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime

from ..models.image_data import ImageData
from ..models.dna_sequence import DNASequence
from ..encoding.biological_constraints import BiologicalConstraints
from ..encoding.error_correction import DNAErrorCorrection


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    
    is_valid: bool
    validation_score: float  # 0-1 score
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0


@dataclass
class QualityMetrics:
    """Quality metrics for encoded sequences."""
    
    # Biological quality
    avg_gc_content: float
    gc_content_std: float
    homopolymer_violations: int
    constraint_violations: int
    synthesis_complexity_score: float
    
    # Encoding quality
    compression_ratio: float
    error_correction_overhead: float
    sequence_redundancy: float
    information_density: float
    
    # Performance metrics
    encoding_time_ms: float
    decoding_time_ms: float
    memory_usage_mb: float
    validation_time_ms: float


class ComprehensiveValidator:
    """Advanced validation system for DNA origami encoding."""
    
    def __init__(self, 
                 constraints: Optional[BiologicalConstraints] = None,
                 enable_logging: bool = True,
                 log_level: str = "INFO"):
        """Initialize validator with constraints and logging."""
        self.constraints = constraints or BiologicalConstraints()
        self.enable_logging = enable_logging
        
        if enable_logging:
            self.logger = self._setup_logger(log_level)
        else:
            self.logger = None
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
        self.quality_trends: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_validations': 0,
            'avg_validation_time_ms': 0.0,
            'success_rate': 0.0,
            'critical_failures': 0
        }
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger(f"dna_origami_validator_{id(self)}")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler for detailed logs
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"dna_validation_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def validate_complete_pipeline(self, 
                                 original_image: ImageData,
                                 dna_sequences: List[DNASequence],
                                 decoded_image: Optional[ImageData] = None,
                                 encoding_params: Optional[Dict] = None) -> ValidationResult:
        """Comprehensive validation of entire encoding pipeline."""
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Starting comprehensive validation for image: {original_image.name}")
        
        errors = []
        warnings = []
        metrics = {}
        validation_score = 0.0
        
        try:
            # 1. Input validation
            input_score, input_errors, input_warnings = self._validate_input_image(original_image)
            errors.extend(input_errors)
            warnings.extend(input_warnings)
            metrics['input_validation_score'] = input_score
            
            # 2. DNA sequence validation
            seq_score, seq_errors, seq_warnings, seq_metrics = self._validate_dna_sequences(dna_sequences)
            errors.extend(seq_errors)
            warnings.extend(seq_warnings)
            metrics.update(seq_metrics)
            metrics['sequence_validation_score'] = seq_score
            
            # 3. Biological constraints validation
            bio_score, bio_errors, bio_warnings = self._validate_biological_constraints(dna_sequences)
            errors.extend(bio_errors)
            warnings.extend(bio_warnings)
            metrics['biological_validation_score'] = bio_score
            
            # 4. Round-trip validation
            if decoded_image:
                rt_score, rt_errors, rt_warnings, rt_metrics = self._validate_round_trip(
                    original_image, decoded_image
                )
                errors.extend(rt_errors)
                warnings.extend(rt_warnings)
                metrics.update(rt_metrics)
                metrics['round_trip_score'] = rt_score
            else:
                rt_score = 0.5  # Partial score if no round-trip test
                warnings.append("No decoded image provided for round-trip validation")
            
            # 5. Performance validation
            perf_score, perf_metrics = self._validate_performance(dna_sequences, encoding_params)
            metrics.update(perf_metrics)
            metrics['performance_score'] = perf_score
            
            # Calculate overall validation score
            weights = {
                'input': 0.15,
                'sequence': 0.25,
                'biological': 0.25,
                'round_trip': 0.25,
                'performance': 0.10
            }
            
            validation_score = (
                weights['input'] * input_score +
                weights['sequence'] * seq_score +
                weights['biological'] * bio_score +
                weights['round_trip'] * rt_score +
                weights['performance'] * perf_score
            )
            
            # Determine overall validity
            is_valid = len(errors) == 0 and validation_score >= 0.7
            
            # Add quality metrics
            quality_metrics = self._calculate_quality_metrics(
                original_image, dna_sequences, decoded_image
            )
            metrics['quality'] = quality_metrics.__dict__
            
        except Exception as e:
            errors.append(f"Validation failed with exception: {str(e)}")
            is_valid = False
            validation_score = 0.0
            
            if self.logger:
                self.logger.error(f"Validation exception: {e}", exc_info=True)
        
        execution_time = (time.time() - start_time) * 1000
        
        result = ValidationResult(
            is_valid=is_valid,
            validation_score=validation_score,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            execution_time_ms=execution_time
        )
        
        # Update statistics
        self._update_performance_stats(result)
        self.validation_history.append(result)
        
        if self.logger:
            self.logger.info(
                f"Validation complete: score={validation_score:.3f}, "
                f"valid={is_valid}, time={execution_time:.1f}ms"
            )
        
        return result
    
    def _validate_input_image(self, image: ImageData) -> Tuple[float, List[str], List[str]]:
        """Validate input image quality and compatibility."""
        errors = []
        warnings = []
        score = 1.0
        
        # Check image dimensions
        if image.metadata.width * image.metadata.height > 1000000:  # 1MP limit
            warnings.append(f"Large image ({image.metadata.width}x{image.metadata.height}) may be slow to process")
            score -= 0.1
        
        if image.metadata.width < 8 or image.metadata.height < 8:
            errors.append("Image too small for reliable encoding")
            score -= 0.5
        
        # Check data quality
        unique_values = len(np.unique(image.data))
        if unique_values < 4:
            warnings.append(f"Low color diversity ({unique_values} unique values)")
            score -= 0.2
        
        # Check for extreme values
        if np.any(image.data == 0) and np.any(image.data == 255):
            if np.mean(image.data) < 50 or np.mean(image.data) > 200:
                warnings.append("Image has extreme brightness values")
                score -= 0.1
        
        # Validate checksum integrity
        expected_checksum = hashlib.sha256(image.data.tobytes()).hexdigest()
        if image.checksum and image.checksum != expected_checksum:
            errors.append("Image data corruption detected (checksum mismatch)")
            score -= 0.8
        
        return max(0.0, score), errors, warnings
    
    def _validate_dna_sequences(self, sequences: List[DNASequence]) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        """Validate DNA sequences for completeness and consistency."""
        errors = []
        warnings = []
        score = 1.0
        metrics = {}
        
        if not sequences:
            errors.append("No DNA sequences provided")
            return 0.0, errors, warnings, metrics
        
        # Check sequence count
        metrics['sequence_count'] = len(sequences)
        if len(sequences) > 1000:
            warnings.append(f"Large number of sequences ({len(sequences)}) may be difficult to synthesize")
            score -= 0.1
        
        # Validate individual sequences
        total_bases = 0
        valid_sequences = 0
        
        for i, seq in enumerate(sequences):
            try:
                # Basic validation
                if not seq.sequence:
                    errors.append(f"Empty sequence at index {i}")
                    continue
                
                if not all(base in 'ATGC' for base in seq.sequence.upper()):
                    errors.append(f"Invalid bases in sequence {i}")
                    continue
                
                valid_sequences += 1
                total_bases += len(seq.sequence)
                
                # Length validation
                if len(seq.sequence) < 10:
                    warnings.append(f"Very short sequence {i} (length: {len(seq.sequence)})")
                    score -= 0.05
                
                if len(seq.sequence) > 300:
                    warnings.append(f"Very long sequence {i} (length: {len(seq.sequence)})")
                    score -= 0.05
                
            except Exception as e:
                errors.append(f"Validation error for sequence {i}: {str(e)}")
        
        # Overall sequence statistics
        metrics['valid_sequences'] = valid_sequences
        metrics['total_bases'] = total_bases
        metrics['avg_sequence_length'] = total_bases / len(sequences) if sequences else 0
        
        if valid_sequences < len(sequences) * 0.95:
            errors.append(f"Too many invalid sequences ({len(sequences) - valid_sequences} out of {len(sequences)})")
            score -= 0.5
        
        return max(0.0, score), errors, warnings, metrics
    
    def _validate_biological_constraints(self, sequences: List[DNASequence]) -> Tuple[float, List[str], List[str]]:
        """Validate biological feasibility of sequences."""
        errors = []
        warnings = []
        score = 1.0
        
        total_violations = 0
        severe_violations = 0
        
        for i, seq in enumerate(sequences):
            is_valid, seq_errors = self.constraints.validate_sequence(seq.sequence)
            
            if not is_valid:
                total_violations += len(seq_errors)
                
                # Categorize severity
                for error in seq_errors:
                    if any(keyword in error.lower() for keyword in ['forbidden', 'homopolymer', 'gc content']):
                        severe_violations += 1
                        errors.append(f"Sequence {i}: {error}")
                    else:
                        warnings.append(f"Sequence {i}: {error}")
        
        # Calculate biological score
        if severe_violations > 0:
            score -= min(0.8, severe_violations * 0.1)
        
        if total_violations > len(sequences) * 0.1:  # More than 10% violation rate
            score -= 0.3
        
        return max(0.0, score), errors, warnings
    
    def _validate_round_trip(self, original: ImageData, decoded: ImageData) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        """Validate encoding/decoding round-trip accuracy."""
        errors = []
        warnings = []
        metrics = {}
        score = 1.0
        
        try:
            # Dimension validation
            if original.data.shape != decoded.data.shape:
                errors.append(f"Shape mismatch: original {original.data.shape} vs decoded {decoded.data.shape}")
                return 0.0, errors, warnings, metrics
            
            # Calculate quality metrics
            mse = original.calculate_mse(decoded)
            psnr = original.calculate_psnr(decoded)
            ssim = original.calculate_ssim(decoded)
            
            metrics['mse'] = mse
            metrics['psnr'] = psnr
            metrics['ssim'] = ssim
            
            # Evaluate quality thresholds
            if mse > 10000:
                errors.append(f"Very high reconstruction error (MSE: {mse:.1f})")
                score -= 0.5
            elif mse > 5000:
                warnings.append(f"High reconstruction error (MSE: {mse:.1f})")
                score -= 0.2
            
            if psnr < 10:
                errors.append(f"Very low PSNR ({psnr:.1f} dB)")
                score -= 0.3
            elif psnr < 15:
                warnings.append(f"Low PSNR ({psnr:.1f} dB)")
                score -= 0.1
            
            if ssim < 0.5:
                errors.append(f"Very low structural similarity (SSIM: {ssim:.3f})")
                score -= 0.3
            elif ssim < 0.7:
                warnings.append(f"Low structural similarity (SSIM: {ssim:.3f})")
                score -= 0.1
            
            # Pixel value range validation
            if np.max(decoded.data) > 255 or np.min(decoded.data) < 0:
                errors.append("Decoded image has out-of-range pixel values")
                score -= 0.2
            
            # Statistical comparison
            orig_stats = {
                'mean': np.mean(original.data),
                'std': np.std(original.data),
                'min': np.min(original.data),
                'max': np.max(original.data)
            }
            
            decoded_stats = {
                'mean': np.mean(decoded.data),
                'std': np.std(decoded.data),
                'min': np.min(decoded.data),
                'max': np.max(decoded.data)
            }
            
            metrics['original_stats'] = orig_stats
            metrics['decoded_stats'] = decoded_stats
            
            # Check for statistical drift
            mean_diff = abs(orig_stats['mean'] - decoded_stats['mean'])
            if mean_diff > 20:
                warnings.append(f"Significant mean shift: {mean_diff:.1f}")
                score -= 0.1
            
        except Exception as e:
            errors.append(f"Round-trip validation failed: {str(e)}")
            score = 0.0
        
        return max(0.0, score), errors, warnings, metrics
    
    def _validate_performance(self, sequences: List[DNASequence], 
                            encoding_params: Optional[Dict] = None) -> Tuple[float, Dict[str, Any]]:
        """Validate performance characteristics."""
        metrics = {}
        score = 1.0
        
        # Sequence efficiency
        total_bases = sum(len(seq.sequence) for seq in sequences)
        metrics['total_bases'] = total_bases
        metrics['sequences_per_kb'] = len(sequences) / (total_bases / 1000) if total_bases > 0 else 0
        
        # Storage efficiency
        if encoding_params and 'original_size_bytes' in encoding_params:
            original_size = encoding_params['original_size_bytes']
            # Estimate DNA storage cost (330 Da per base, ~650 g/mol per base pair)
            dna_mass_ng = total_bases * 330 * 1e-9  # nanograms
            metrics['storage_density_mb_per_gram'] = (original_size / 1024**2) / (dna_mass_ng * 1e-9)
        
        # Complexity scoring
        complexity_scores = []
        for seq in sequences:
            # Simple complexity: unique k-mers / total k-mers
            k = 3
            kmers = [seq.sequence[i:i+k] for i in range(len(seq.sequence) - k + 1)]
            if kmers:
                complexity = len(set(kmers)) / len(kmers)
                complexity_scores.append(complexity)
        
        if complexity_scores:
            avg_complexity = np.mean(complexity_scores)
            metrics['average_sequence_complexity'] = avg_complexity
            
            if avg_complexity < 0.3:
                score -= 0.2  # Very repetitive sequences
        
        return score, metrics
    
    def _calculate_quality_metrics(self, 
                                 original_image: ImageData,
                                 dna_sequences: List[DNASequence],
                                 decoded_image: Optional[ImageData] = None) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        
        # Biological quality
        gc_contents = [seq.gc_content for seq in dna_sequences]
        avg_gc = np.mean(gc_contents) if gc_contents else 0.0
        gc_std = np.std(gc_contents) if len(gc_contents) > 1 else 0.0
        
        homopolymer_violations = 0
        constraint_violations = 0
        
        for seq in dna_sequences:
            is_valid, errors = self.constraints.validate_sequence(seq.sequence)
            if not is_valid:
                constraint_violations += len(errors)
                for error in errors:
                    if 'homopolymer' in error.lower():
                        homopolymer_violations += 1
        
        # Synthesis complexity (simplified)
        total_bases = sum(len(seq.sequence) for seq in dna_sequences)
        synthesis_complexity = min(1.0, len(dna_sequences) / 100 + total_bases / 50000)
        
        # Encoding quality
        original_size = original_image.metadata.size_bytes
        dna_size = total_bases  # 1 byte per base for synthesis
        compression_ratio = original_size / dna_size if dna_size > 0 else 0.0
        
        # Placeholder values for timing (would be measured in real implementation)
        encoding_time = 0.0
        decoding_time = 0.0
        memory_usage = total_bases * 4 / (1024 * 1024)  # Rough estimate in MB
        validation_time = 0.0
        
        return QualityMetrics(
            avg_gc_content=avg_gc,
            gc_content_std=gc_std,
            homopolymer_violations=homopolymer_violations,
            constraint_violations=constraint_violations,
            synthesis_complexity_score=synthesis_complexity,
            compression_ratio=compression_ratio,
            error_correction_overhead=1.0,  # Placeholder
            sequence_redundancy=0.0,  # Placeholder
            information_density=original_size / total_bases if total_bases > 0 else 0.0,
            encoding_time_ms=encoding_time,
            decoding_time_ms=decoding_time,
            memory_usage_mb=memory_usage,
            validation_time_ms=validation_time
        )
    
    def _update_performance_stats(self, result: ValidationResult) -> None:
        """Update internal performance statistics."""
        self.performance_stats['total_validations'] += 1
        
        # Update average validation time
        prev_avg = self.performance_stats['avg_validation_time_ms']
        n = self.performance_stats['total_validations']
        new_avg = (prev_avg * (n - 1) + result.execution_time_ms) / n
        self.performance_stats['avg_validation_time_ms'] = new_avg
        
        # Update success rate
        successes = sum(1 for r in self.validation_history if r.is_valid)
        self.performance_stats['success_rate'] = successes / len(self.validation_history)
        
        # Track critical failures
        if len(result.errors) > 5:
            self.performance_stats['critical_failures'] += 1
    
    def get_validation_report(self, include_history: bool = False) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'performance_stats': self.performance_stats.copy(),
            'constraint_summary': self.constraints.get_constraint_summary(),
            'last_validation': None
        }
        
        if self.validation_history:
            last_result = self.validation_history[-1]
            report['last_validation'] = {
                'timestamp': last_result.timestamp.isoformat(),
                'is_valid': last_result.is_valid,
                'score': last_result.validation_score,
                'execution_time_ms': last_result.execution_time_ms,
                'error_count': len(last_result.errors),
                'warning_count': len(last_result.warnings)
            }
        
        if include_history:
            report['validation_history'] = [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'is_valid': r.is_valid,
                    'score': r.validation_score,
                    'execution_time_ms': r.execution_time_ms
                }
                for r in self.validation_history[-50:]  # Last 50 validations
            ]
        
        return report
    
    def export_validation_data(self, file_path: str) -> None:
        """Export validation data to file."""
        report = self.get_validation_report(include_history=True)
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.logger:
            self.logger.info(f"Validation data exported to {file_path}")