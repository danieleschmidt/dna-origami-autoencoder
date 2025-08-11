"""Advanced error handling and recovery systems for DNA encoding."""

import functools
import time
import traceback
import threading
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import inspect
from concurrent.futures import TimeoutError
import psutil

from .logger import get_logger


class ErrorSeverity(Enum):
    """Classification of error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Available recovery actions for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    DEGRADE_PERFORMANCE = "degrade_performance"


@dataclass
class ErrorContext:
    """Context information for error analysis."""
    function_name: str
    parameters: Dict[str, Any]
    timestamp: float
    thread_id: str
    memory_usage_mb: float
    cpu_percent: float
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    recovery_action: Optional[RecoveryAction] = None


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = threading.Lock()
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == 'open':
                    if time.time() - self.last_failure_time > self.timeout:
                        self.state = 'half-open'
                    else:
                        raise Exception("Circuit breaker is open")
                
                try:
                    result = func(*args, **kwargs)
                    if self.state == 'half-open':
                        self.state = 'closed'
                        self.failure_count = 0
                    return result
                    
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = 'open'
                    
                    raise e
        
        return wrapper


class RetryMechanism:
    """Advanced retry mechanism with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0,
                 retriable_exceptions: Tuple = None):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retriable_exceptions = retriable_exceptions or (Exception,)
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except self.retriable_exceptions as e:
                    last_exception = e
                    
                    if attempt < self.max_attempts - 1:
                        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                        time.sleep(delay)
                    else:
                        break
                        
                except Exception as e:
                    # Non-retriable exception, fail immediately
                    raise e
            
            # All attempts failed
            raise last_exception
        
        return wrapper


class AdvancedErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("error_handler")
        self.error_history = []
        self.recovery_strategies = {}
        self.error_patterns = {}
        self.performance_degradation_active = False
        self._lock = threading.Lock()
        
        # Default recovery strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default error recovery strategies."""
        
        # Memory-related errors
        self.recovery_strategies['MemoryError'] = {
            'actions': [RecoveryAction.DEGRADE_PERFORMANCE, RecoveryAction.RETRY],
            'fallback': self._memory_fallback
        }
        
        # Timeout errors
        self.recovery_strategies['TimeoutError'] = {
            'actions': [RecoveryAction.RETRY, RecoveryAction.DEGRADE_PERFORMANCE],
            'fallback': self._timeout_fallback
        }
        
        # Validation errors
        self.recovery_strategies['ValueError'] = {
            'actions': [RecoveryAction.FALLBACK],
            'fallback': self._validation_fallback
        }
        
        # IO errors
        self.recovery_strategies['IOError'] = {
            'actions': [RecoveryAction.RETRY, RecoveryAction.SKIP],
            'fallback': self._io_fallback
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any], 
                    fallback_func: Optional[Callable] = None) -> Tuple[bool, Any]:
        """
        Handle error with automatic recovery attempts.
        
        Returns:
            Tuple[bool, Any]: (recovery_successful, result_or_error)
        """
        
        # Collect system state
        process = psutil.Process()
        
        error_context = ErrorContext(
            function_name=context.get('function_name', 'unknown'),
            parameters=context.get('parameters', {}),
            timestamp=time.time(),
            thread_id=str(threading.current_thread().ident),
            memory_usage_mb=process.memory_info().rss / 1024 / 1024,
            cpu_percent=process.cpu_percent(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            severity=self._classify_error_severity(error, context)
        )
        
        # Log error
        self.logger.error(f"Error in {error_context.function_name}: {error_context.error_message}")
        
        # Store error history
        with self._lock:
            self.error_history.append(error_context)
            if len(self.error_history) > 1000:  # Limit history size
                self.error_history = self.error_history[-1000:]
        
        # Attempt recovery
        recovery_successful, result = self._attempt_recovery(error, error_context, fallback_func)
        
        if recovery_successful:
            self.logger.info(f"Recovered from error in {error_context.function_name}")
        else:
            self.logger.error(f"Failed to recover from error in {error_context.function_name}")
        
        return recovery_successful, result
    
    def _classify_error_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Classify error severity based on type and context."""
        
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (TimeoutError, ConnectionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _attempt_recovery(self, error: Exception, error_context: ErrorContext, 
                         fallback_func: Optional[Callable]) -> Tuple[bool, Any]:
        """Attempt error recovery using configured strategies."""
        
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            
            for action in strategy['actions']:
                if action == RecoveryAction.RETRY:
                    # Already handled by retry decorator, skip
                    continue
                    
                elif action == RecoveryAction.DEGRADE_PERFORMANCE:
                    self._activate_performance_degradation()
                    if fallback_func:
                        try:
                            result = fallback_func()
                            return True, result
                        except Exception as e:
                            continue
                            
                elif action == RecoveryAction.FALLBACK:
                    fallback = strategy.get('fallback')
                    if fallback:
                        try:
                            result = fallback(error, error_context)
                            return True, result
                        except Exception:
                            continue
                
                elif action == RecoveryAction.SKIP:
                    self.logger.warning(f"Skipping operation due to error: {error}")
                    return True, None
        
        # No recovery possible
        return False, error
    
    def _activate_performance_degradation(self):
        """Activate performance degradation mode to recover from resource issues."""
        
        if not self.performance_degradation_active:
            self.performance_degradation_active = True
            self.logger.warning("Activating performance degradation mode")
            
            # Schedule deactivation after some time
            threading.Timer(300, self._deactivate_performance_degradation).start()  # 5 minutes
    
    def _deactivate_performance_degradation(self):
        """Deactivate performance degradation mode."""
        self.performance_degradation_active = False
        self.logger.info("Performance degradation mode deactivated")
    
    def _memory_fallback(self, error: Exception, context: ErrorContext) -> Any:
        """Fallback strategy for memory errors."""
        import gc
        gc.collect()  # Force garbage collection
        
        # Reduce batch size or processing complexity
        params = context.parameters
        if 'batch_size' in params and params['batch_size'] > 1:
            params['batch_size'] = max(1, params['batch_size'] // 2)
            self.logger.info(f"Reduced batch size to {params['batch_size']}")
        
        return None  # Indicate fallback applied
    
    def _timeout_fallback(self, error: Exception, context: ErrorContext) -> Any:
        """Fallback strategy for timeout errors."""
        # Increase timeout or reduce work complexity
        params = context.parameters
        if 'timeout' in params:
            params['timeout'] = params['timeout'] * 1.5
            self.logger.info(f"Increased timeout to {params['timeout']}")
        
        return None
    
    def _validation_fallback(self, error: Exception, context: ErrorContext) -> Any:
        """Fallback strategy for validation errors."""
        # Use default or simplified parameters
        self.logger.info("Using fallback validation parameters")
        return None
    
    def _io_fallback(self, error: Exception, context: ErrorContext) -> Any:
        """Fallback strategy for IO errors."""
        # Skip or use cached data
        self.logger.info("Using IO fallback strategy")
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        
        with self._lock:
            if not self.error_history:
                return {"message": "No errors recorded"}
            
            # Count by error type
            error_counts = {}
            severity_counts = {}
            
            for error in self.error_history:
                error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
                severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            
            # Calculate error rate over time
            recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
            error_rate = len(recent_errors) / 60  # Errors per minute
            
            return {
                'total_errors': len(self.error_history),
                'recent_error_rate_per_minute': error_rate,
                'error_types': error_counts,
                'severity_distribution': severity_counts,
                'most_common_error': max(error_counts, key=error_counts.get) if error_counts else None,
                'performance_degradation_active': self.performance_degradation_active
            }


def robust_execution(max_attempts: int = 3, fallback_func: Optional[Callable] = None,
                    circuit_breaker: bool = False, timeout: Optional[int] = None):
    """
    Decorator for robust function execution with error handling and recovery.
    
    Args:
        max_attempts: Maximum retry attempts
        fallback_func: Fallback function to call on failure
        circuit_breaker: Enable circuit breaker pattern
        timeout: Execution timeout in seconds
    """
    
    def decorator(func):
        error_handler = AdvancedErrorHandler()
        
        # Apply circuit breaker if requested
        if circuit_breaker:
            func = CircuitBreaker()(func)
        
        # Apply retry mechanism
        if max_attempts > 1:
            func = RetryMechanism(max_attempts)(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Apply timeout if specified
            if timeout:
                def timeout_handler():
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
                
                timer = threading.Timer(timeout, timeout_handler)
                timer.start()
            
            try:
                # Collect context information
                context = {
                    'function_name': func.__name__,
                    'parameters': {
                        'args': args,
                        'kwargs': kwargs
                    }
                }
                
                result = func(*args, **kwargs)
                
                if timeout:
                    timer.cancel()
                
                return result
                
            except Exception as e:
                if timeout:
                    timer.cancel()
                
                # Attempt error recovery
                context = {
                    'function_name': func.__name__,
                    'parameters': {'args': args, 'kwargs': kwargs}
                }
                
                recovery_successful, result = error_handler.handle_error(e, context, fallback_func)
                
                if recovery_successful:
                    return result
                else:
                    # Re-raise original error if recovery failed
                    raise e
        
        # Add error statistics method to wrapped function
        wrapper.get_error_stats = error_handler.get_error_statistics
        
        return wrapper
    
    return decorator


class ResourceMonitor:
    """Monitor system resources and trigger defensive actions."""
    
    def __init__(self, memory_threshold: float = 0.85, cpu_threshold: float = 0.9):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.monitoring_active = False
        self.alert_callbacks = []
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        self.logger = get_logger("resource_monitor")
    
    def add_alert_callback(self, callback: Callable[[str, float], None]):
        """Add callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self, interval: int = 5):
        """Start resource monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self._monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join()
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Main monitoring loop."""
        while not self._stop_event.wait(interval):
            try:
                # Check memory usage
                memory_percent = psutil.virtual_memory().percent / 100.0
                if memory_percent > self.memory_threshold:
                    self._trigger_alert('memory', memory_percent)
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1) / 100.0
                if cpu_percent > self.cpu_threshold:
                    self._trigger_alert('cpu', cpu_percent)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
    
    def _trigger_alert(self, resource_type: str, usage: float):
        """Trigger resource usage alert."""
        self.logger.warning(f"High {resource_type} usage: {usage:.1%}")
        
        for callback in self.alert_callbacks:
            try:
                callback(resource_type, usage)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")


# Global resource monitor instance
resource_monitor = ResourceMonitor()


class GracefulDegradation:
    """Graceful degradation system for maintaining functionality under stress."""
    
    def __init__(self):
        self.degradation_levels = {
            'normal': {'batch_size_multiplier': 1.0, 'quality_reduction': 0.0},
            'light': {'batch_size_multiplier': 0.75, 'quality_reduction': 0.1},
            'moderate': {'batch_size_multiplier': 0.5, 'quality_reduction': 0.2},
            'heavy': {'batch_size_multiplier': 0.25, 'quality_reduction': 0.3}
        }
        
        self.current_level = 'normal'
        self.degradation_history = []
        self.logger = get_logger("graceful_degradation")
    
    def activate_degradation(self, level: str, reason: str = ""):
        """Activate specific degradation level."""
        if level in self.degradation_levels:
            self.current_level = level
            self.degradation_history.append({
                'level': level,
                'reason': reason,
                'timestamp': time.time()
            })
            
            self.logger.warning(f"Activated degradation level: {level}. Reason: {reason}")
        else:
            self.logger.error(f"Invalid degradation level: {level}")
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current degradation parameters."""
        return self.degradation_levels[self.current_level].copy()
    
    def reset_to_normal(self):
        """Reset to normal operation level."""
        if self.current_level != 'normal':
            self.current_level = 'normal'
            self.logger.info("Reset to normal operation level")


# Global graceful degradation instance
graceful_degradation = GracefulDegradation()


def setup_resource_monitoring():
    """Setup global resource monitoring with automated degradation."""
    
    def resource_alert_handler(resource_type: str, usage: float):
        """Handle resource alerts with automated degradation."""
        
        if resource_type == 'memory' and usage > 0.9:
            graceful_degradation.activate_degradation('heavy', f'High memory usage: {usage:.1%}')
        elif resource_type == 'memory' and usage > 0.85:
            graceful_degradation.activate_degradation('moderate', f'Memory usage: {usage:.1%}')
        elif resource_type == 'cpu' and usage > 0.95:
            graceful_degradation.activate_degradation('moderate', f'High CPU usage: {usage:.1%}')
    
    resource_monitor.add_alert_callback(resource_alert_handler)
    resource_monitor.start_monitoring()