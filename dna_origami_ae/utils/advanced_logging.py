"""Advanced logging system with structured logging, performance tracking, and analytics."""

import logging
import json
import time
import threading
import functools
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import uuid
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import gzip
from pathlib import Path

from .performance import performance_monitor


@dataclass
class LogEntry:
    """Structured log entry with comprehensive metadata."""
    
    timestamp: float
    level: str
    message: str
    logger_name: str
    thread_id: str
    function_name: str
    line_number: int
    module_name: str
    
    # Performance metrics
    memory_usage_mb: float
    cpu_percent: float
    
    # Context information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    # Custom fields
    extra_data: Optional[Dict[str, Any]] = None
    
    # Processing metrics
    processing_time_ms: Optional[float] = None
    items_processed: Optional[int] = None
    throughput_items_per_sec: Optional[float] = None
    
    # Error context
    error_type: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class PerformanceTracker:
    """Track performance metrics for logging."""
    
    def __init__(self):
        self.operation_metrics = defaultdict(list)
        self.current_operations = {}
        self._lock = threading.Lock()
    
    def start_operation(self, operation_id: str, operation_type: str) -> str:
        """Start tracking an operation."""
        with self._lock:
            self.current_operations[operation_id] = {
                'type': operation_type,
                'start_time': time.time(),
                'start_memory': psutil.Process().memory_info().rss / 1024 / 1024
            }
        return operation_id
    
    def end_operation(self, operation_id: str, items_processed: int = 0) -> Dict[str, float]:
        """End operation tracking and return metrics."""
        with self._lock:
            if operation_id not in self.current_operations:
                return {}
            
            operation = self.current_operations.pop(operation_id)
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            duration = end_time - operation['start_time']
            memory_delta = end_memory - operation['start_memory']
            throughput = items_processed / duration if duration > 0 else 0
            
            metrics = {
                'duration_seconds': duration,
                'memory_delta_mb': memory_delta,
                'items_processed': items_processed,
                'throughput_items_per_sec': throughput
            }
            
            # Store for analytics
            self.operation_metrics[operation['type']].append(metrics)
            
            # Limit stored metrics
            if len(self.operation_metrics[operation['type']]) > 1000:
                self.operation_metrics[operation['type']] = self.operation_metrics[operation['type']][-1000:]
            
            return metrics


class LogBuffer:
    """Thread-safe circular buffer for log entries."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add(self, entry: LogEntry):
        """Add log entry to buffer."""
        with self._lock:
            self.buffer.append(entry)
    
    def get_recent(self, count: int = 100) -> List[LogEntry]:
        """Get recent log entries."""
        with self._lock:
            return list(self.buffer)[-count:]
    
    def clear(self):
        """Clear buffer."""
        with self._lock:
            self.buffer.clear()


class LogAnalytics:
    """Real-time log analytics and pattern detection."""
    
    def __init__(self):
        self.error_patterns = defaultdict(int)
        self.performance_trends = defaultdict(list)
        self.alert_thresholds = {
            'error_rate_per_minute': 10,
            'avg_response_time_ms': 5000,
            'memory_usage_mb': 1000
        }
        self.alert_callbacks = []
        self._lock = threading.Lock()
    
    def analyze_entry(self, entry: LogEntry):
        """Analyze log entry for patterns and trends."""
        with self._lock:
            # Track error patterns
            if entry.level in ['ERROR', 'CRITICAL']:
                pattern_key = f"{entry.logger_name}:{entry.error_type}" if entry.error_type else entry.logger_name
                self.error_patterns[pattern_key] += 1
            
            # Track performance trends
            if entry.processing_time_ms:
                trend_key = f"{entry.logger_name}:processing_time"
                self.performance_trends[trend_key].append({
                    'timestamp': entry.timestamp,
                    'value': entry.processing_time_ms
                })
                
                # Limit trend data
                if len(self.performance_trends[trend_key]) > 1000:
                    self.performance_trends[trend_key] = self.performance_trends[trend_key][-1000:]
            
            # Check for alerts
            self._check_alerts(entry)
    
    def _check_alerts(self, entry: LogEntry):
        """Check for alert conditions."""
        current_time = time.time()
        
        # Error rate alert
        recent_errors = sum(1 for pattern, count in self.error_patterns.items()
                           if any('ERROR' in k or 'CRITICAL' in k for k in [pattern]))
        
        if recent_errors > self.alert_thresholds['error_rate_per_minute']:
            self._trigger_alert('high_error_rate', {
                'error_count': recent_errors,
                'threshold': self.alert_thresholds['error_rate_per_minute']
            })
        
        # Performance alert
        if entry.processing_time_ms and entry.processing_time_ms > self.alert_thresholds['avg_response_time_ms']:
            self._trigger_alert('slow_response', {
                'processing_time_ms': entry.processing_time_ms,
                'threshold': self.alert_thresholds['avg_response_time_ms'],
                'function': entry.function_name
            })
        
        # Memory alert
        if entry.memory_usage_mb > self.alert_thresholds['memory_usage_mb']:
            self._trigger_alert('high_memory', {
                'memory_usage_mb': entry.memory_usage_mb,
                'threshold': self.alert_thresholds['memory_usage_mb']
            })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger alert to registered callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception:
                pass  # Don't let alert callback errors crash logging
    
    def add_alert_callback(self, callback):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary."""
        with self._lock:
            # Top error patterns
            top_errors = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Performance statistics
            perf_stats = {}
            for trend_key, values in self.performance_trends.items():
                if values:
                    times = [v['value'] for v in values]
                    perf_stats[trend_key] = {
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                        'count': len(times)
                    }
            
            return {
                'top_error_patterns': top_errors,
                'performance_statistics': perf_stats,
                'total_error_count': sum(self.error_patterns.values()),
                'monitored_trends': len(self.performance_trends)
            }


class AdvancedJSONFormatter(logging.Formatter):
    """Advanced JSON formatter with structured logging."""
    
    def __init__(self, include_performance: bool = True, include_system: bool = True):
        super().__init__()
        self.include_performance = include_performance
        self.include_system = include_system
        self.performance_tracker = PerformanceTracker()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Get system metrics
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Create structured log entry
        log_entry = LogEntry(
            timestamp=record.created,
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            thread_id=str(threading.current_thread().ident),
            function_name=record.funcName,
            line_number=record.lineno,
            module_name=record.module,
            memory_usage_mb=memory_mb,
            cpu_percent=cpu_percent
        )
        
        # Add exception information if present
        if record.exc_info:
            log_entry.error_type = record.exc_info[0].__name__
            log_entry.error_details = {
                'exception_text': self.formatException(record.exc_info)
            }
        
        # Add extra fields
        extra_data = {}
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'exc_info', 'exc_text', 'stack_info', 'getMessage'):
                extra_data[key] = value
        
        if extra_data:
            log_entry.extra_data = extra_data
        
        # Convert to JSON
        return json.dumps(asdict(log_entry), default=str)


class AdvancedLogger:
    """Advanced logging system with enhanced capabilities."""
    
    def __init__(self, name: str, level: int = logging.INFO, 
                 enable_analytics: bool = True, enable_buffering: bool = True):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Components
        self.performance_tracker = PerformanceTracker()
        self.log_buffer = LogBuffer() if enable_buffering else None
        self.analytics = LogAnalytics() if enable_analytics else None
        
        # Thread pool for async logging
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="log_writer")
        
        # Setup handlers
        self._setup_handlers()
        
        # Context storage
        self._context = threading.local()
        
        # Metrics
        self.log_counts = defaultdict(int)
        self._metrics_lock = threading.Lock()
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = AdvancedJSONFormatter()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logs
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / f"{self.name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        self.logger.addHandler(file_handler)
        
        # Rotating file handler for long-term storage
        from logging.handlers import RotatingFileHandler
        rotating_handler = RotatingFileHandler(
            log_dir / f"{self.name}_rotating.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        rotating_handler.setLevel(logging.DEBUG)
        rotating_handler.setFormatter(console_formatter)
        self.logger.addHandler(rotating_handler)
    
    def set_context(self, **kwargs):
        """Set logging context for current thread."""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        self._context.data.update(kwargs)
    
    def clear_context(self):
        """Clear logging context for current thread."""
        if hasattr(self._context, 'data'):
            self._context.data = {}
    
    def log_with_performance(self, level: int, message: str, 
                           operation_id: Optional[str] = None,
                           items_processed: int = 0, **kwargs):
        """Log with performance tracking."""
        
        # Add context data
        if hasattr(self._context, 'data'):
            kwargs.update(self._context.data)
        
        # Add performance metrics if operation_id provided
        if operation_id and operation_id in self.performance_tracker.current_operations:
            metrics = self.performance_tracker.end_operation(operation_id, items_processed)
            kwargs.update({
                'processing_time_ms': metrics.get('duration_seconds', 0) * 1000,
                'items_processed': items_processed,
                'throughput_items_per_sec': metrics.get('throughput_items_per_sec', 0),
                'memory_delta_mb': metrics.get('memory_delta_mb', 0)
            })
        
        # Update metrics
        with self._metrics_lock:
            self.log_counts[logging.getLevelName(level)] += 1
        
        # Log the message
        self.logger.log(level, message, extra=kwargs)
        
        # Add to buffer and analytics if enabled
        if self.log_buffer or self.analytics:
            # Create log entry for buffer/analytics
            process = psutil.Process()
            entry = LogEntry(
                timestamp=time.time(),
                level=logging.getLevelName(level),
                message=message,
                logger_name=self.name,
                thread_id=str(threading.current_thread().ident),
                function_name=kwargs.get('function_name', ''),
                line_number=kwargs.get('line_number', 0),
                module_name=kwargs.get('module_name', ''),
                memory_usage_mb=process.memory_info().rss / 1024 / 1024,
                cpu_percent=process.cpu_percent(),
                extra_data=kwargs,
                processing_time_ms=kwargs.get('processing_time_ms'),
                items_processed=items_processed,
                throughput_items_per_sec=kwargs.get('throughput_items_per_sec')
            )
            
            if self.log_buffer:
                self.log_buffer.add(entry)
            
            if self.analytics:
                # Analyze asynchronously
                self.executor.submit(self.analytics.analyze_entry, entry)
    
    def start_operation(self, operation_type: str, operation_id: Optional[str] = None) -> str:
        """Start tracking an operation."""
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        
        self.performance_tracker.start_operation(operation_id, operation_type)
        self.debug(f"Started operation: {operation_type}", 
                  extra={'operation_id': operation_id, 'operation_type': operation_type})
        
        return operation_id
    
    def end_operation(self, operation_id: str, message: str = "", items_processed: int = 0):
        """End operation tracking with logging."""
        metrics = self.performance_tracker.end_operation(operation_id, items_processed)
        
        self.info(message or f"Completed operation {operation_id}", 
                 extra={'operation_id': operation_id, 'metrics': metrics})
    
    # Standard logging methods with enhancements
    def debug(self, message: str, **kwargs):
        self.log_with_performance(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.log_with_performance(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log_with_performance(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log_with_performance(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.log_with_performance(logging.CRITICAL, message, **kwargs)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._metrics_lock:
            base_metrics = {
                'log_counts': dict(self.log_counts),
                'total_logs': sum(self.log_counts.values())
            }
        
        # Add analytics if available
        if self.analytics:
            analytics_summary = self.analytics.get_analytics_summary()
            base_metrics.update(analytics_summary)
        
        # Add performance metrics
        perf_summary = {}
        for op_type, metrics_list in self.performance_tracker.operation_metrics.items():
            if metrics_list:
                durations = [m['duration_seconds'] for m in metrics_list]
                throughputs = [m['throughput_items_per_sec'] for m in metrics_list if m['throughput_items_per_sec'] > 0]
                
                perf_summary[op_type] = {
                    'count': len(metrics_list),
                    'avg_duration_seconds': sum(durations) / len(durations),
                    'min_duration_seconds': min(durations),
                    'max_duration_seconds': max(durations),
                    'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0
                }
        
        base_metrics['performance_summary'] = perf_summary
        
        return base_metrics
    
    def export_logs(self, format_type: str = 'json', 
                   count: int = 1000) -> Union[str, Dict[str, Any]]:
        """Export recent logs in specified format."""
        
        if not self.log_buffer:
            return {"error": "Log buffering not enabled"}
        
        recent_logs = self.log_buffer.get_recent(count)
        
        if format_type.lower() == 'json':
            return {
                'logs': [asdict(entry) for entry in recent_logs],
                'total_count': len(recent_logs),
                'export_timestamp': time.time()
            }
        elif format_type.lower() == 'csv':
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            if recent_logs:
                writer = csv.DictWriter(output, fieldnames=asdict(recent_logs[0]).keys())
                writer.writeheader()
                for entry in recent_logs:
                    writer.writerow(asdict(entry))
            
            return output.getvalue()
        else:
            return {"error": f"Unsupported format: {format_type}"}


# Decorator for automatic operation logging
def log_operation(operation_type: str, logger: Optional[AdvancedLogger] = None,
                 log_args: bool = False, log_result: bool = False):
    """Decorator for automatic operation logging with performance tracking."""
    
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_advanced_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_id = logger.start_operation(operation_type)
            
            # Set function context
            logger.set_context(
                function_name=func.__name__,
                module_name=func.__module__
            )
            
            start_time = time.time()
            
            try:
                # Log input if requested
                if log_args:
                    logger.debug(f"Starting {operation_type} with args",
                               extra={'args': str(args), 'kwargs': str(kwargs)})
                
                result = func(*args, **kwargs)
                
                # Calculate items processed (heuristic)
                items_processed = 0
                if isinstance(result, (list, tuple)):
                    items_processed = len(result)
                elif hasattr(result, '__len__'):
                    try:
                        items_processed = len(result)
                    except:
                        items_processed = 1
                else:
                    items_processed = 1
                
                # Log success
                duration = time.time() - start_time
                logger.end_operation(
                    operation_id, 
                    f"Successfully completed {operation_type} in {duration:.3f}s",
                    items_processed
                )
                
                # Log result if requested
                if log_result:
                    logger.debug(f"Operation {operation_type} result",
                               extra={'result_type': type(result).__name__,
                                     'result_size': items_processed})
                
                return result
                
            except Exception as e:
                # Log error
                duration = time.time() - start_time
                logger.error(f"Failed {operation_type} after {duration:.3f}s: {e}",
                           extra={'error_type': type(e).__name__, 
                                 'error_details': str(e),
                                 'operation_id': operation_id})
                raise
            
            finally:
                logger.clear_context()
        
        return wrapper
    
    return decorator


# Global logger registry
_logger_registry = {}
_registry_lock = threading.Lock()


def get_advanced_logger(name: str, **kwargs) -> AdvancedLogger:
    """Get or create advanced logger instance."""
    
    with _registry_lock:
        if name not in _logger_registry:
            _logger_registry[name] = AdvancedLogger(name, **kwargs)
        
        return _logger_registry[name]


def setup_global_logging(log_level: int = logging.INFO,
                        enable_analytics: bool = True,
                        enable_performance_tracking: bool = True):
    """Setup global logging configuration."""
    
    # Set root logger level
    logging.root.setLevel(log_level)
    
    # Create main application logger
    main_logger = get_advanced_logger("dna_origami_ae", 
                                     level=log_level,
                                     enable_analytics=enable_analytics)
    
    # Setup analytics alerts if enabled
    if enable_analytics and main_logger.analytics:
        def alert_handler(alert_type: str, data: Dict[str, Any]):
            main_logger.warning(f"Analytics alert: {alert_type}", extra=data)
        
        main_logger.analytics.add_alert_callback(alert_handler)
    
    return main_logger


# Performance logging decorator shorthand
performance_logged = lambda operation_type: log_operation(
    operation_type, log_args=False, log_result=True
)