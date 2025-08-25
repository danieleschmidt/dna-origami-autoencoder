"""
Advanced Error Handling and Recovery System for DNA Origami AutoEncoder

Provides comprehensive error handling, recovery mechanisms, validation,
and monitoring capabilities with detailed logging and alerting.
"""

import functools
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from enum import Enum
import warnings

import numpy as np


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"           # Minor issues, continue processing
    MEDIUM = "medium"     # Significant issues, may need intervention  
    HIGH = "high"         # Critical issues, likely to cause failures
    CRITICAL = "critical" # System-threatening issues, immediate action required


class ErrorCategory(Enum):
    """Error category classification for better error handling."""
    VALIDATION = "validation"         # Input validation errors
    COMPUTATION = "computation"       # Computational errors
    MEMORY = "memory"                # Memory-related issues
    GPU = "gpu"                      # GPU/hardware issues  
    BIOLOGICAL = "biological"        # Biological constraint violations
    NETWORK = "network"              # Network/IO issues
    CONFIGURATION = "configuration"   # Configuration issues
    SYSTEM = "system"                # System-level issues


class DNAOrigarniError(Exception):
    """Base exception class for DNA Origami AutoEncoder."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 details: Optional[Dict[str, Any]] = None,
                 recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = time.time()


class ValidationError(DNAOrigarniError):
    """Input validation and constraint violation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 expected: Any = None, actual: Any = None, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, **kwargs)
        self.field = field
        self.expected = expected
        self.actual = actual
        
        if field:
            self.details.update({
                "field": field,
                "expected": expected, 
                "actual": actual
            })


class ComputationError(DNAOrigarniError):
    """Computational and algorithmic errors."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 input_data: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, ErrorCategory.COMPUTATION, **kwargs)
        self.operation = operation
        self.input_data = input_data
        
        if operation:
            self.details["operation"] = operation
        if input_data:
            self.details["input_data"] = input_data


class MemoryError(DNAOrigarniError):
    """Memory allocation and management errors."""
    
    def __init__(self, message: str, requested_size: Optional[int] = None,
                 available_size: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorCategory.MEMORY, 
                        ErrorSeverity.HIGH, **kwargs)
        self.requested_size = requested_size
        self.available_size = available_size
        
        self.details.update({
            "requested_size": requested_size,
            "available_size": available_size
        })


class GPUError(DNAOrigarniError):
    """GPU and hardware acceleration errors."""
    
    def __init__(self, message: str, device: Optional[str] = None,
                 cuda_error: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.GPU, 
                        ErrorSeverity.HIGH, **kwargs)
        self.device = device
        self.cuda_error = cuda_error
        
        self.details.update({
            "device": device,
            "cuda_error": cuda_error
        })


class BiologicalConstraintError(DNAOrigarniError):
    """Biological constraint and validation errors."""
    
    def __init__(self, message: str, constraint_type: Optional[str] = None,
                 sequence: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.BIOLOGICAL,
                        ErrorSeverity.MEDIUM, **kwargs)
        self.constraint_type = constraint_type
        self.sequence = sequence
        
        self.details.update({
            "constraint_type": constraint_type,
            "sequence": sequence[:100] if sequence else None  # Truncate for logging
        })


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_stats = {category.value: 0 for category in ErrorCategory}
        self.recovery_strategies = {}
        self.error_callbacks = []
        self.max_retry_attempts = 3
        self.retry_delay = 1.0  # seconds
        
        # Security: Sensitive data patterns to redact from logs
        import re
        self.sensitive_patterns = [
            re.compile(r'password\w*[:\s=]*[\w\d]+', re.IGNORECASE),
            re.compile(r'token[:\s=]*[\w\d]+', re.IGNORECASE),
            re.compile(r'key[:\s=]*[\w\d]+', re.IGNORECASE),
            re.compile(r'secret[:\s=]*[\w\d]+', re.IGNORECASE),
            re.compile(r'api[_\-\s]*key[:\s=]*[\w\d]+', re.IGNORECASE),
            re.compile(r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'),  # Credit cards
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')  # Emails
        ]
        
    def register_recovery_strategy(self, error_type: Type[Exception], 
                                 strategy: Callable[[Exception], Any]) -> None:
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type.__name__}")
        
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback function to be called when errors occur."""
        self.error_callbacks.append(callback)
        
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle an error with appropriate logging, recovery, and callbacks."""
        
        # Update error statistics
        if isinstance(error, DNAOrigarniError):
            self.error_stats[error.category.value] += 1
        else:
            self.error_stats[ErrorCategory.SYSTEM.value] += 1
            
        # Log error with context
        self._log_error(error, context)
        
        # Execute error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as callback_error:
                self.logger.error(f"Error callback failed: {callback_error}")
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error, context)
        if recovery_result is not None:
            return recovery_result
            
        # Re-raise if no recovery possible
        raise error
        
    def _log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error with comprehensive information."""
        
        if isinstance(error, DNAOrigarniError):
            log_level = self._severity_to_log_level(error.severity)
            error_info = {
                "category": error.category.value,
                "severity": error.severity.value,
                "recoverable": error.recoverable,
                "details": error.details,
                "timestamp": error.timestamp
            }
        else:
            log_level = logging.ERROR
            error_info = {"type": type(error).__name__}
            
        if context:
            error_info["context"] = context
            
        # Log with sanitized error message for security
        sanitized_error = self._sanitize_error_message(str(error))
        self.logger.log(
            log_level,
            f"Error occurred: {sanitized_error}",
            extra={
                "error_info": error_info,
                "traceback": traceback.format_exc()
            }
        )
        
    def _sanitize_error_message(self, message: str) -> str:
        """Remove sensitive data from error messages."""
        sanitized = message
        for pattern in self.sensitive_patterns:
            sanitized = pattern.sub('[REDACTED]', sanitized)
        return sanitized
        
    def _severity_to_log_level(self, severity: ErrorSeverity) -> int:
        """Convert error severity to logging level."""
        severity_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return severity_map[severity]
        
    def _attempt_recovery(self, error: Exception, 
                         context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Attempt to recover from error using registered strategies."""
        
        error_type = type(error)
        
        # Check for exact type match first
        if error_type in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {error_type.__name__}")
                return self.recovery_strategies[error_type](error)
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
                
        # Check for parent class matches
        for registered_type, strategy in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                try:
                    self.logger.info(f"Attempting recovery using {registered_type.__name__} strategy")
                    return strategy(error)
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")
                    
        return None
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            "error_counts": self.error_stats.copy(),
            "total_errors": sum(self.error_stats.values()),
            "recovery_strategies": len(self.recovery_strategies),
            "callbacks": len(self.error_callbacks)
        }
        
    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.error_stats = {category.value: 0 for category in ErrorCategory}
        self.logger.info("Error statistics reset")


class ValidationFramework:
    """Comprehensive validation framework for inputs and constraints."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
        self.validation_rules = {}
        
    def add_validation_rule(self, name: str, rule: Callable[[Any], bool],
                          error_message: str = None) -> None:
        """Add a validation rule."""
        self.validation_rules[name] = {
            "rule": rule,
            "error_message": error_message or f"Validation failed for rule: {name}"
        }
        
    def validate_image_data(self, image: np.ndarray) -> None:
        """Validate image data for encoding."""
        
        if not isinstance(image, np.ndarray):
            raise ValidationError(
                "Image must be a numpy array",
                field="image",
                expected="numpy.ndarray",
                actual=type(image).__name__
            )
            
        if image.ndim not in [2, 3]:
            raise ValidationError(
                "Image must be 2D or 3D array",
                field="image.ndim", 
                expected="2 or 3",
                actual=image.ndim
            )
            
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValidationError(
                "Image must have uint8, float32, or float64 dtype",
                field="image.dtype",
                expected="uint8, float32, or float64",
                actual=str(image.dtype)
            )
            
        if image.size == 0:
            raise ValidationError(
                "Image cannot be empty",
                field="image.size",
                expected="> 0",
                actual=image.size
            )
            
        # Check for reasonable dimensions
        max_dimension = 10000
        if any(dim > max_dimension for dim in image.shape):
            raise ValidationError(
                f"Image dimensions too large (max: {max_dimension})",
                field="image.shape",
                expected=f"all dimensions <= {max_dimension}",
                actual=image.shape
            )
            
    def validate_dna_sequence(self, sequence: str) -> None:
        """Validate DNA sequence format and content."""
        
        if not isinstance(sequence, str):
            raise ValidationError(
                "DNA sequence must be a string",
                field="sequence",
                expected="str",
                actual=type(sequence).__name__
            )
            
        if not sequence:
            raise ValidationError(
                "DNA sequence cannot be empty",
                field="sequence",
                expected="non-empty string",
                actual="empty string"
            )
            
        # Check for valid DNA bases
        valid_bases = set('ATGC')
        invalid_bases = set(sequence.upper()) - valid_bases
        if invalid_bases:
            raise ValidationError(
                f"Invalid DNA bases found: {invalid_bases}",
                field="sequence",
                expected="only A, T, G, C",
                actual=f"contains {invalid_bases}"
            )
            
        # Check sequence length constraints
        min_length, max_length = 10, 1000000
        if not (min_length <= len(sequence) <= max_length):
            raise ValidationError(
                f"DNA sequence length must be between {min_length} and {max_length}",
                field="sequence.length",
                expected=f"{min_length} - {max_length}",
                actual=len(sequence)
            )
            
    def validate_biological_constraints(self, sequence: str, 
                                      constraints: Dict[str, Any]) -> None:
        """Validate biological constraints on DNA sequence."""
        
        self.validate_dna_sequence(sequence)
        
        # GC content validation
        if "gc_content" in constraints:
            gc_min, gc_max = constraints["gc_content"]
            gc_count = sequence.upper().count('G') + sequence.upper().count('C')
            gc_content = gc_count / len(sequence)
            
            if not (gc_min <= gc_content <= gc_max):
                raise BiologicalConstraintError(
                    f"GC content {gc_content:.2%} outside range [{gc_min:.2%}, {gc_max:.2%}]",
                    constraint_type="gc_content",
                    sequence=sequence
                )
                
        # Homopolymer run validation  
        if "max_homopolymer" in constraints:
            max_run = constraints["max_homopolymer"]
            current_run = 1
            current_base = sequence[0] if sequence else ''
            
            for base in sequence[1:]:
                if base == current_base:
                    current_run += 1
                    if current_run > max_run:
                        raise BiologicalConstraintError(
                            f"Homopolymer run of {current_run} {current_base}s exceeds limit {max_run}",
                            constraint_type="max_homopolymer",
                            sequence=sequence
                        )
                else:
                    current_run = 1
                    current_base = base
                    
        # Avoid sequences validation
        if "avoid_sequences" in constraints:
            avoid_list = constraints["avoid_sequences"]
            for avoid_seq in avoid_list:
                if avoid_seq in sequence.upper():
                    raise BiologicalConstraintError(
                        f"Forbidden sequence '{avoid_seq}' found in DNA sequence",
                        constraint_type="avoid_sequences", 
                        sequence=sequence
                    )
                    
    def validate_origami_design(self, design: Dict[str, Any]) -> None:
        """Validate origami design parameters."""
        
        required_fields = ["scaffold_length", "staple_length", "dimensions"]
        for field in required_fields:
            if field not in design:
                raise ValidationError(
                    f"Missing required field: {field}",
                    field=field,
                    expected="present",
                    actual="missing"
                )
                
        # Validate scaffold length
        scaffold_length = design["scaffold_length"]
        if not isinstance(scaffold_length, int) or scaffold_length <= 0:
            raise ValidationError(
                "Scaffold length must be positive integer",
                field="scaffold_length",
                expected="positive integer",
                actual=scaffold_length
            )
            
        # Validate staple length
        staple_length = design["staple_length"]
        if not isinstance(staple_length, int) or not (10 <= staple_length <= 100):
            raise ValidationError(
                "Staple length must be between 10 and 100",
                field="staple_length",
                expected="10 <= staple_length <= 100",
                actual=staple_length
            )
            
        # Validate dimensions
        dimensions = design["dimensions"]
        if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2:
            raise ValidationError(
                "Dimensions must be a 2-element list/tuple",
                field="dimensions",
                expected="2-element list/tuple",
                actual=dimensions
            )
            
        if not all(isinstance(d, (int, float)) and d > 0 for d in dimensions):
            raise ValidationError(
                "Dimensions must be positive numbers",
                field="dimensions",
                expected="positive numbers",
                actual=dimensions
            )


def retry_on_error(max_attempts: int = 3, delay: float = 1.0, 
                  backoff: float = 2.0, exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Decorator to retry function calls on specific exceptions."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:  # Don't delay on last attempt
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(f"All {max_attempts} attempts failed")
                        
            raise last_exception
        return wrapper
    return decorator


@contextmanager
def error_context(operation: str, **context):
    """Context manager for enhanced error handling with operation context."""
    
    error_handler = ErrorHandler()
    start_time = time.time()
    
    try:
        logging.info(f"Starting operation: {operation}")
        yield error_handler
        
    except Exception as e:
        duration = time.time() - start_time
        enhanced_context = {
            "operation": operation,
            "duration": duration,
            **context
        }
        
        error_handler.handle_error(e, enhanced_context)
        
    else:
        duration = time.time() - start_time
        logging.info(f"Operation '{operation}' completed successfully in {duration:.2f}s")


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection."""
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.timeout:
                raise ComputationError(
                    "Circuit breaker is OPEN - too many recent failures",
                    operation=func.__name__,
                    severity=ErrorSeverity.HIGH
                )
            else:
                self.state = "HALF_OPEN"
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
            
    def _on_success(self):
        """Handle successful function call."""
        self.failure_count = 0
        self.state = "CLOSED"
        
    def _on_failure(self):
        """Handle failed function call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


# Global error handler instance
_global_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def setup_error_recovery():
    """Setup standard error recovery strategies."""
    handler = get_error_handler()
    
    # Memory error recovery
    def memory_error_recovery(error):
        import gc
        gc.collect()  # Force garbage collection
        logging.warning("Memory error recovery: performed garbage collection")
        return None
        
    handler.register_recovery_strategy(MemoryError, memory_error_recovery)
    
    # GPU error recovery
    def gpu_error_recovery(error):
        try:
            import torch
            torch.cuda.empty_cache()
            logging.warning("GPU error recovery: cleared CUDA cache")
        except:
            pass
        return None
        
    handler.register_recovery_strategy(GPUError, gpu_error_recovery)
    
    logging.info("Error recovery strategies initialized")