"""Comprehensive logging system for DNA origami autoencoder."""

import logging
import sys
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import threading
from contextlib import contextmanager


class DNAOriGAEFormatter(logging.Formatter):
    """Custom formatter for DNA origami autoencoder logging."""
    
    def __init__(self):
        """Initialize formatter with structured format."""
        super().__init__()
        self.hostname = "dna-origami-ae"
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data."""
        # Add custom fields
        record.hostname = self.hostname
        record.component = getattr(record, 'component', 'unknown')
        record.operation = getattr(record, 'operation', 'unknown')
        record.session_id = getattr(record, 'session_id', 'none')
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'hostname': record.hostname,
            'component': record.component,
            'operation': record.operation,
            'session_id': record.session_id,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom metrics if present
        if hasattr(record, 'metrics'):
            log_entry['metrics'] = record.metrics
            
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize performance logger."""
        self.logger = logger
        self._timers = {}
        self._lock = threading.Lock()
    
    def start_timer(self, operation: str, session_id: str = None) -> None:
        """Start timing an operation."""
        key = f"{operation}:{session_id or 'default'}"
        with self._lock:
            self._timers[key] = time.time()
    
    def end_timer(self, operation: str, session_id: str = None, 
                  additional_metrics: Dict[str, Any] = None) -> float:
        """End timing and log performance metrics."""
        key = f"{operation}:{session_id or 'default'}"
        end_time = time.time()
        
        with self._lock:
            start_time = self._timers.get(key)
            if start_time:
                duration = end_time - start_time
                del self._timers[key]
            else:
                duration = 0.0
        
        # Prepare metrics
        metrics = {
            'duration_seconds': duration,
            'operation': operation
        }
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log performance
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                'component': 'performance',
                'operation': operation,
                'session_id': session_id or 'default',
                'metrics': metrics
            }
        )
        
        return duration
    
    @contextmanager
    def time_operation(self, operation: str, session_id: str = None):
        """Context manager for timing operations."""
        self.start_timer(operation, session_id)
        try:
            yield
        finally:
            self.end_timer(operation, session_id)


class DNAOriGAELogger:
    """Central logging system for DNA origami autoencoder."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize logger if not already done."""
        if not self._initialized:
            self._setup_logging()
            self._initialized = True
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration."""
        # Create main logger
        self.logger = logging.getLogger('dna_origami_ae')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with structured logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(DNAOriGAEFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler for all logs (optional - disable if no permissions)
        try:
            log_dir = Path('/var/log/dna-origami-ae')
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / 'application.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(DNAOriGAEFormatter())
            self.logger.addHandler(file_handler)
            
            # Error file handler
            error_handler = logging.FileHandler(log_dir / 'errors.log')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(DNAOriGAEFormatter())
            self.logger.addHandler(error_handler)
        except (PermissionError, OSError):
            # Log to local directory if /var/log is not writable
            log_dir = Path('./logs')
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / 'application.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(DNAOriGAEFormatter())
            self.logger.addHandler(file_handler)
        
        # Performance logger
        self.performance = PerformanceLogger(self.logger)
        
        # Component-specific loggers
        self.component_loggers = {}
        
        self.logger.info(
            "DNA Origami AutoEncoder logging system initialized",
            extra={'component': 'logging', 'operation': 'init'}
        )
    
    def get_component_logger(self, component: str) -> logging.Logger:
        """Get logger for specific component."""
        if component not in self.component_loggers:
            logger = logging.getLogger(f'dna_origami_ae.{component}')
            logger.setLevel(logging.DEBUG)
            
            # Add component name to all records
            class ComponentFilter(logging.Filter):
                def filter(self, record):
                    record.component = component
                    return True
            
            logger.addFilter(ComponentFilter())
            self.component_loggers[component] = logger
        
        return self.component_loggers[component]
    
    def log_pipeline_start(self, session_id: str, parameters: Dict[str, Any]):
        """Log start of pipeline execution."""
        self.logger.info(
            f"Pipeline execution started",
            extra={
                'component': 'pipeline',
                'operation': 'start',
                'session_id': session_id,
                'metrics': {'parameters': parameters}
            }
        )
        self.performance.start_timer('full_pipeline', session_id)
    
    def log_pipeline_end(self, session_id: str, success: bool, 
                        metrics: Dict[str, Any] = None):
        """Log end of pipeline execution."""
        duration = self.performance.end_timer('full_pipeline', session_id, metrics)
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Pipeline execution {'completed' if success else 'failed'}",
            extra={
                'component': 'pipeline',
                'operation': 'end',
                'session_id': session_id,
                'metrics': {
                    'success': success,
                    'duration': duration,
                    **(metrics or {})
                }
            }
        )
    
    def log_error(self, component: str, operation: str, error: Exception, 
                 session_id: str = None, context: Dict[str, Any] = None):
        """Log error with full context."""
        self.logger.error(
            f"Error in {component}.{operation}: {str(error)}",
            extra={
                'component': component,
                'operation': operation,
                'session_id': session_id or 'none',
                'metrics': {
                    'error_type': type(error).__name__,
                    'context': context or {}
                }
            },
            exc_info=True
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any],
                          severity: str = 'INFO'):
        """Log security-related events."""
        level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(
            level,
            f"Security event: {event_type}",
            extra={
                'component': 'security',
                'operation': event_type,
                'metrics': {
                    'event_type': event_type,
                    'severity': severity,
                    'details': details
                }
            }
        )
    
    def log_health_check(self, component: str, status: str, 
                        metrics: Dict[str, Any] = None):
        """Log health check results."""
        level = logging.INFO if status == 'healthy' else logging.WARN
        self.logger.log(
            level,
            f"Health check: {component} is {status}",
            extra={
                'component': 'health',
                'operation': 'check',
                'metrics': {
                    'component': component,
                    'status': status,
                    'metrics': metrics or {}
                }
            }
        )


# Global logger instance
dna_logger = DNAOriGAELogger()


def get_logger(component: str = 'main') -> logging.Logger:
    """Get component-specific logger."""
    return dna_logger.get_component_logger(component)


def log_performance(operation: str, session_id: str = None):
    """Decorator for performance logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            dna_logger.performance.start_timer(operation, session_id)
            try:
                result = func(*args, **kwargs)
                dna_logger.performance.end_timer(operation, session_id, {
                    'success': True,
                    'function': func.__name__
                })
                return result
            except Exception as e:
                dna_logger.performance.end_timer(operation, session_id, {
                    'success': False,
                    'error': str(e),
                    'function': func.__name__
                })
                raise
        return wrapper
    return decorator


@contextmanager
def log_context(component: str, operation: str, session_id: str = None):
    """Context manager for automatic error logging."""
    logger = get_logger(component)
    logger.info(f"Starting {operation}", extra={
        'operation': operation,
        'session_id': session_id
    })
    
    try:
        yield logger
        logger.info(f"Completed {operation}", extra={
            'operation': operation,
            'session_id': session_id
        })
    except Exception as e:
        dna_logger.log_error(component, operation, e, session_id)
        raise


class HealthMonitor:
    """Monitor system health and component status."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.logger = get_logger('health')
        self.component_status = {}
        self._lock = threading.Lock()
    
    def register_component(self, component: str, health_check_func):
        """Register component for health monitoring."""
        with self._lock:
            self.component_status[component] = {
                'health_check': health_check_func,
                'status': 'unknown',
                'last_check': None,
                'consecutive_failures': 0
            }
    
    def check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of specific component."""
        if component not in self.component_status:
            return {'status': 'unknown', 'error': 'Component not registered'}
        
        comp_info = self.component_status[component]
        
        try:
            health_result = comp_info['health_check']()
            status = 'healthy' if health_result.get('healthy', False) else 'unhealthy'
            
            with self._lock:
                comp_info['status'] = status
                comp_info['last_check'] = datetime.now()
                if status == 'healthy':
                    comp_info['consecutive_failures'] = 0
                else:
                    comp_info['consecutive_failures'] += 1
            
            dna_logger.log_health_check(component, status, health_result)
            return health_result
            
        except Exception as e:
            with self._lock:
                comp_info['status'] = 'error'
                comp_info['consecutive_failures'] += 1
                comp_info['last_check'] = datetime.now()
            
            error_result = {'healthy': False, 'error': str(e)}
            dna_logger.log_health_check(component, 'error', error_result)
            return error_result
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        overall_status = 'healthy'
        component_results = {}
        
        for component in self.component_status.keys():
            result = self.check_component_health(component)
            component_results[component] = result
            
            if not result.get('healthy', False):
                overall_status = 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'components': component_results,
            'timestamp': datetime.now().isoformat()
        }


# Global health monitor
health_monitor = HealthMonitor()