"""Health monitoring and system diagnostics for DNA origami encoding."""

import time
import psutil
import threading
import json
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
from queue import Queue, Empty
import warnings


@dataclass
class HealthMetrics:
    """System health metrics."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_threads: int
    gpu_available: bool = False
    gpu_memory_percent: float = 0.0
    encoding_queue_size: int = 0
    error_rate: float = 0.0
    avg_response_time_ms: float = 0.0


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    
    timestamp: datetime
    severity: str  # 'warning', 'error', 'critical'
    metric: str
    value: float
    threshold: float
    message: str
    resolved: bool = False


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for fault tolerance."""
    
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    failure_threshold: int = 5
    timeout_seconds: int = 60


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, 
                 monitoring_interval: float = 5.0,
                 history_size: int = 1000,
                 enable_alerts: bool = True):
        """Initialize health monitor."""
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts
        
        # Monitoring data
        self.metrics_history: List[HealthMetrics] = []
        self.alerts: List[PerformanceAlert] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Alert thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate': 0.05,  # 5%
            'avg_response_time_ms': 5000.0  # 5 seconds
        }
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread = None
        self._stats_lock = threading.Lock()
        
        # Performance tracking
        self.operation_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_response_time_ms': 0.0
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # GPU detection
        self.gpu_available = self._detect_gpu()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup health monitoring logging."""
        logger = logging.getLogger(f"health_monitor_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - HealthMonitor - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _detect_gpu(self) -> bool:
        """Detect GPU availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                return False
    
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring_active:
            self.logger.warning("Health monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = self._collect_metrics()
                
                with self._stats_lock:
                    self.metrics_history.append(metrics)
                    
                    # Maintain history size
                    if len(self.metrics_history) > self.history_size:
                        self.metrics_history.pop(0)
                
                # Check for alerts
                if self.enable_alerts:
                    self._check_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_memory_percent = 0.0
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_stats()
                    allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                    reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                    if reserved > 0:
                        gpu_memory_percent = (allocated / reserved) * 100
            except Exception:
                pass
        
        # Calculate error rate and response time
        with self._stats_lock:
            total_ops = self.operation_stats['total_operations']
            failed_ops = self.operation_stats['failed_operations']
            total_time = self.operation_stats['total_response_time_ms']
            
            error_rate = failed_ops / total_ops if total_ops > 0 else 0.0
            avg_response_time = total_time / total_ops if total_ops > 0 else 0.0
        
        return HealthMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_usage_percent=disk.percent,
            active_threads=threading.active_count(),
            gpu_available=self.gpu_available,
            gpu_memory_percent=gpu_memory_percent,
            error_rate=error_rate,
            avg_response_time_ms=avg_response_time
        )
    
    def _check_alerts(self, metrics: HealthMetrics) -> None:
        """Check for alert conditions."""
        alert_checks = [
            ('cpu_percent', metrics.cpu_percent, 'High CPU usage'),
            ('memory_percent', metrics.memory_percent, 'High memory usage'),
            ('disk_usage_percent', metrics.disk_usage_percent, 'High disk usage'),
            ('error_rate', metrics.error_rate, 'High error rate'),
            ('avg_response_time_ms', metrics.avg_response_time_ms, 'High response time')
        ]
        
        for metric_name, value, description in alert_checks:
            threshold = self.thresholds.get(metric_name, float('inf'))
            
            if value > threshold:
                severity = self._determine_severity(metric_name, value, threshold)
                
                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    severity=severity,
                    metric=metric_name,
                    value=value,
                    threshold=threshold,
                    message=f"{description}: {value:.1f} exceeds threshold {threshold:.1f}"
                )
                
                self.alerts.append(alert)
                
                # Log alert
                log_method = getattr(self.logger, severity.lower(), self.logger.info)
                log_method(alert.message)
                
                # Maintain alert history
                if len(self.alerts) > 100:
                    self.alerts.pop(0)
    
    def _determine_severity(self, metric: str, value: float, threshold: float) -> str:
        """Determine alert severity based on how much threshold is exceeded."""
        ratio = value / threshold
        
        if ratio > 2.0:
            return 'critical'
        elif ratio > 1.5:
            return 'error'
        else:
            return 'warning'
    
    def record_operation(self, operation_name: str, 
                        execution_time_ms: float, 
                        success: bool) -> None:
        """Record operation performance metrics."""
        with self._stats_lock:
            self.operation_stats['total_operations'] += 1
            self.operation_stats['total_response_time_ms'] += execution_time_ms
            
            if success:
                self.operation_stats['successful_operations'] += 1
            else:
                self.operation_stats['failed_operations'] += 1
        
        # Update circuit breaker
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreakerState()
        
        circuit = self.circuit_breakers[operation_name]
        
        if success:
            circuit.success_count += 1
            if circuit.state == "HALF_OPEN" and circuit.success_count >= 3:
                circuit.state = "CLOSED"
                circuit.failure_count = 0
                self.logger.info(f"Circuit breaker for {operation_name} closed")
        else:
            circuit.failure_count += 1
            circuit.last_failure_time = datetime.now()
            
            if (circuit.state == "CLOSED" and 
                circuit.failure_count >= circuit.failure_threshold):
                circuit.state = "OPEN"
                self.logger.warning(f"Circuit breaker for {operation_name} opened")
    
    def check_circuit_breaker(self, operation_name: str) -> bool:
        """Check if operation should be allowed (circuit breaker pattern)."""
        if operation_name not in self.circuit_breakers:
            return True
        
        circuit = self.circuit_breakers[operation_name]
        
        if circuit.state == "CLOSED":
            return True
        elif circuit.state == "OPEN":
            # Check if timeout has passed
            if (circuit.last_failure_time and 
                datetime.now() - circuit.last_failure_time > 
                timedelta(seconds=circuit.timeout_seconds)):
                circuit.state = "HALF_OPEN"
                circuit.success_count = 0
                self.logger.info(f"Circuit breaker for {operation_name} half-opened")
                return True
            else:
                return False
        elif circuit.state == "HALF_OPEN":
            return True
        
        return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status summary."""
        if not self.metrics_history:
            return {
                'status': 'initializing', 
                'health_score': 100.0,
                'message': 'Health monitoring starting...',
                'issues': [],
                'latest_metrics': {},
                'active_alerts': 0,
                'circuit_breakers': {}
            }
        
        latest_metrics = self.metrics_history[-1]
        
        # Determine overall health
        health_score = 100.0
        issues = []
        
        if latest_metrics.cpu_percent > self.thresholds['cpu_percent']:
            health_score -= 20
            issues.append(f"High CPU usage: {latest_metrics.cpu_percent:.1f}%")
        
        if latest_metrics.memory_percent > self.thresholds['memory_percent']:
            health_score -= 25
            issues.append(f"High memory usage: {latest_metrics.memory_percent:.1f}%")
        
        if latest_metrics.error_rate > self.thresholds['error_rate']:
            health_score -= 30
            issues.append(f"High error rate: {latest_metrics.error_rate:.1%}")
        
        if latest_metrics.avg_response_time_ms > self.thresholds['avg_response_time_ms']:
            health_score -= 15
            issues.append(f"Slow response time: {latest_metrics.avg_response_time_ms:.1f}ms")
        
        # Determine status
        if health_score >= 90:
            status = 'healthy'
        elif health_score >= 70:
            status = 'degraded'
        elif health_score >= 50:
            status = 'unhealthy'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'health_score': health_score,
            'issues': issues,
            'latest_metrics': latest_metrics.__dict__,
            'active_alerts': len([a for a in self.alerts if not a.resolved]),
            'circuit_breakers': {name: cb.state for name, cb in self.circuit_breakers.items()}
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get performance trends over specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            'timestamps': [m.timestamp.timestamp() for m in recent_metrics],
            'cpu_percent': [m.cpu_percent for m in recent_metrics],
            'memory_percent': [m.memory_percent for m in recent_metrics],
            'error_rate': [m.error_rate for m in recent_metrics],
            'response_time_ms': [m.avg_response_time_ms for m in recent_metrics]
        }
    
    def export_health_report(self, file_path: str) -> None:
        """Export comprehensive health report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'monitoring_config': {
                'interval_seconds': self.monitoring_interval,
                'history_size': self.history_size,
                'thresholds': self.thresholds
            },
            'current_status': self.get_health_status(),
            'performance_trends': self.get_performance_trends(),
            'recent_alerts': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'severity': a.severity,
                    'metric': a.metric,
                    'value': a.value,
                    'threshold': a.threshold,
                    'message': a.message,
                    'resolved': a.resolved
                }
                for a in self.alerts[-50:]  # Last 50 alerts
            ],
            'operation_statistics': self.operation_stats.copy(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'gpu_available': self.gpu_available,
                'python_version': f"{psutil.PROCFS_PATH}"  # placeholder
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Health report exported to {file_path}")


class PerformanceProfiler:
    """Performance profiling utilities."""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = {}
        self.active_timers: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.active_timers[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and record duration."""
        if operation not in self.active_timers:
            return 0.0
        
        duration = (time.time() - self.active_timers[operation]) * 1000  # ms
        del self.active_timers[operation]
        
        if operation not in self.profiles:
            self.profiles[operation] = []
        
        self.profiles[operation].append(duration)
        
        # Keep only recent measurements
        if len(self.profiles[operation]) > 1000:
            self.profiles[operation] = self.profiles[operation][-500:]
        
        return duration
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.profiles or not self.profiles[operation]:
            return {}
        
        times = self.profiles[operation]
        return {
            'count': len(times),
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.profiles.keys()}


def performance_monitor(operation_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            profiler = getattr(wrapper, '_profiler', None)
            if profiler is None:
                profiler = PerformanceProfiler()
                wrapper._profiler = profiler
            
            profiler.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                duration = profiler.end_timer(operation_name)
                return result
            except Exception as e:
                profiler.end_timer(operation_name)
                raise
        
        return wrapper
    return decorator