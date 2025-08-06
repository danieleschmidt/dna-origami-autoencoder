"""
Comprehensive Monitoring and Health Checks for DNA Origami AutoEncoder

Provides real-time monitoring, health checks, performance metrics,
and alerting capabilities for the DNA origami framework.
"""

import logging
import os
import psutil
import sys
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
import json
import warnings

import numpy as np


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    COUNTER = "counter"         # Monotonically increasing
    GAUGE = "gauge"            # Current value that can go up/down
    HISTOGRAM = "histogram"     # Distribution of values
    TIMER = "timer"            # Duration measurements
    RATE = "rate"              # Rate over time


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass 
class HealthCheck:
    """Health check configuration and result."""
    name: str
    check_function: Callable[[], Tuple[HealthStatus, str]]
    interval: float = 60.0  # seconds
    timeout: float = 10.0   # seconds
    last_run: float = 0.0
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_message: str = ""
    failure_count: int = 0
    max_failures: int = 3


class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        with self.lock:
            self.metrics[metric.name].append(metric)
            
    def record_counter(self, name: str, value: float = 1.0, **tags) -> None:
        """Record a counter metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags
        )
        self.record_metric(metric)
        
    def record_gauge(self, name: str, value: float, **tags) -> None:
        """Record a gauge metric."""
        metric = PerformanceMetric(
            name=name, 
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags
        )
        self.record_metric(metric)
        
    def record_timer(self, name: str, duration: float, **tags) -> None:
        """Record a timing metric."""
        metric = PerformanceMetric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            unit="seconds",
            tags=tags
        )
        self.record_metric(metric)
        
    def record_histogram(self, name: str, value: float, **tags) -> None:
        """Record a histogram metric."""
        metric = PerformanceMetric(
            name=name,
            value=value, 
            metric_type=MetricType.HISTOGRAM,
            tags=tags
        )
        self.record_metric(metric)
        
    def get_latest_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get the most recent metric value."""
        with self.lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1]
        return None
        
    def get_metric_history(self, name: str, limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Get historical metric values."""
        with self.lock:
            if name in self.metrics:
                history = list(self.metrics[name])
                return history[-limit:] if limit else history
        return []
        
    def get_metric_statistics(self, name: str, window: int = 100) -> Dict[str, float]:
        """Get statistical summary of metric over recent window."""
        history = self.get_metric_history(name, window)
        
        if not history:
            return {}
            
        values = [m.value for m in history]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }
        
    def get_all_metrics(self) -> Dict[str, List[PerformanceMetric]]:
        """Get all collected metrics."""
        with self.lock:
            return {name: list(metrics) for name, metrics in self.metrics.items()}


class HealthMonitor:
    """Monitors system health and performs periodic health checks."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks = {}
        self.overall_status = HealthStatus.UNKNOWN
        self.monitoring_thread = None
        self.running = False
        self.check_interval = 30.0  # seconds
        self.logger = logging.getLogger(__name__)
        
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Registered health check: {health_check.name}")
        
    def start_monitoring(self) -> None:
        """Start the health monitoring thread."""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Health monitoring started")
            
    def stop_monitoring(self) -> None:
        """Stop the health monitoring thread."""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._run_health_checks()
                self._collect_system_metrics()
                self._update_overall_status()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
                
    def _run_health_checks(self) -> None:
        """Run all registered health checks."""
        current_time = time.time()
        
        for name, check in self.health_checks.items():
            # Skip if not time to run this check
            if current_time - check.last_run < check.interval:
                continue
                
            try:
                # Run health check with timeout
                status, message = self._run_single_check(check)
                
                check.last_run = current_time
                check.last_status = status
                check.last_message = message
                
                if status == HealthStatus.HEALTHY:
                    check.failure_count = 0
                else:
                    check.failure_count += 1
                    
                # Record health check metric
                self.metrics_collector.record_gauge(
                    f"health_check.{name}",
                    1.0 if status == HealthStatus.HEALTHY else 0.0,
                    status=status.value
                )
                
                self.logger.debug(f"Health check '{name}': {status.value} - {message}")
                
            except Exception as e:
                check.last_status = HealthStatus.CRITICAL
                check.last_message = f"Health check failed: {e}"
                check.failure_count += 1
                self.logger.error(f"Health check '{name}' failed: {e}")
                
    def _run_single_check(self, check: HealthCheck) -> Tuple[HealthStatus, str]:
        """Run a single health check with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Health check '{check.name}' timed out")
            
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(check.timeout))
        
        try:
            result = check.check_function()
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            return HealthStatus.CRITICAL, f"Check timed out after {check.timeout}s"
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            return HealthStatus.CRITICAL, f"Check failed: {e}"
            
    def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_gauge("system.cpu.percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record_gauge("system.memory.percent", memory.percent)
            self.metrics_collector.record_gauge("system.memory.available", memory.available)
            self.metrics_collector.record_gauge("system.memory.used", memory.used)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics_collector.record_gauge("system.disk.percent", disk.percent)
            self.metrics_collector.record_gauge("system.disk.free", disk.free)
            
            # Process metrics
            process = psutil.Process()
            self.metrics_collector.record_gauge("process.memory.rss", process.memory_info().rss)
            self.metrics_collector.record_gauge("process.memory.vms", process.memory_info().vms)
            self.metrics_collector.record_gauge("process.cpu.percent", process.cpu_percent())
            
            # GPU metrics (if available)
            self._collect_gpu_metrics()
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            
    def _collect_gpu_metrics(self) -> None:
        """Collect GPU metrics if available."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # GPU memory
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_cached = torch.cuda.memory_reserved(device)
                memory_total = torch.cuda.get_device_properties(device).total_memory
                
                self.metrics_collector.record_gauge("gpu.memory.allocated", memory_allocated)
                self.metrics_collector.record_gauge("gpu.memory.cached", memory_cached) 
                self.metrics_collector.record_gauge("gpu.memory.total", memory_total)
                self.metrics_collector.record_gauge("gpu.memory.percent", 
                                                  (memory_allocated / memory_total) * 100)
                
                # GPU utilization (if nvidia-ml-py available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.metrics_collector.record_gauge("gpu.utilization", gpu_util.gpu)
                except ImportError:
                    pass
                    
        except ImportError:
            pass  # PyTorch not available
        except Exception as e:
            self.logger.debug(f"GPU metrics collection failed: {e}")
            
    def _update_overall_status(self) -> None:
        """Update overall system health status."""
        if not self.health_checks:
            self.overall_status = HealthStatus.UNKNOWN
            return
            
        statuses = [check.last_status for check in self.health_checks.values()]
        
        # Determine overall status based on individual checks
        if any(status == HealthStatus.CRITICAL for status in statuses):
            self.overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            self.overall_status = HealthStatus.DEGRADED
        elif any(status == HealthStatus.WARNING for status in statuses):
            self.overall_status = HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            self.overall_status = HealthStatus.HEALTHY
        else:
            self.overall_status = HealthStatus.UNKNOWN
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status summary."""
        return {
            "overall_status": self.overall_status.value,
            "checks": {
                name: {
                    "status": check.last_status.value,
                    "message": check.last_message,
                    "last_run": check.last_run,
                    "failure_count": check.failure_count
                }
                for name, check in self.health_checks.items()
            },
            "timestamp": time.time()
        }


class PerformanceProfiler:
    """Performance profiling utilities for DNA origami operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_profiles = {}
        self.lock = threading.Lock()
        
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        return ProfileContext(operation_name, self)
        
    def start_profile(self, profile_id: str, operation_name: str) -> None:
        """Start profiling an operation."""
        with self.lock:
            self.active_profiles[profile_id] = {
                "operation": operation_name,
                "start_time": time.time(),
                "start_memory": self._get_memory_usage()
            }
            
    def end_profile(self, profile_id: str, **tags) -> None:
        """End profiling and record metrics."""
        with self.lock:
            if profile_id not in self.active_profiles:
                return
                
            profile_data = self.active_profiles.pop(profile_id)
            
            # Calculate metrics
            duration = time.time() - profile_data["start_time"]
            memory_delta = self._get_memory_usage() - profile_data["start_memory"]
            
            # Record metrics
            operation = profile_data["operation"]
            self.metrics_collector.record_timer(f"operation.duration.{operation}", duration, **tags)
            self.metrics_collector.record_gauge(f"operation.memory_delta.{operation}", memory_delta, **tags)
            self.metrics_collector.record_counter(f"operation.count.{operation}", 1.0, **tags)
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0.0


class ProfileContext:
    """Context manager for operation profiling."""
    
    def __init__(self, operation_name: str, profiler: PerformanceProfiler):
        self.operation_name = operation_name
        self.profiler = profiler
        self.profile_id = f"{operation_name}_{time.time()}_{threading.get_ident()}"
        
    def __enter__(self):
        self.profiler.start_profile(self.profile_id, self.operation_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        tags = {}
        if exc_type:
            tags["error"] = exc_type.__name__
            tags["success"] = "false"
        else:
            tags["success"] = "true"
            
        self.profiler.end_profile(self.profile_id, **tags)


def create_standard_health_checks(metrics_collector: MetricsCollector) -> List[HealthCheck]:
    """Create standard health checks for DNA origami system."""
    
    checks = []
    
    # System resource checks
    def check_memory():
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return HealthStatus.CRITICAL, f"Memory usage critical: {memory.percent:.1f}%"
        elif memory.percent > 80:
            return HealthStatus.WARNING, f"Memory usage high: {memory.percent:.1f}%"
        else:
            return HealthStatus.HEALTHY, f"Memory usage normal: {memory.percent:.1f}%"
            
    def check_disk_space():
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            return HealthStatus.CRITICAL, f"Disk usage critical: {disk.percent:.1f}%"
        elif disk.percent > 85:
            return HealthStatus.WARNING, f"Disk usage high: {disk.percent:.1f}%"
        else:
            return HealthStatus.HEALTHY, f"Disk usage normal: {disk.percent:.1f}%"
            
    def check_cpu():
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            return HealthStatus.CRITICAL, f"CPU usage critical: {cpu_percent:.1f}%"
        elif cpu_percent > 85:
            return HealthStatus.WARNING, f"CPU usage high: {cpu_percent:.1f}%"
        else:
            return HealthStatus.HEALTHY, f"CPU usage normal: {cpu_percent:.1f}%"
            
    def check_gpu():
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_used = torch.cuda.memory_allocated(device)
                memory_total = torch.cuda.get_device_properties(device).total_memory
                usage_percent = (memory_used / memory_total) * 100
                
                if usage_percent > 95:
                    return HealthStatus.CRITICAL, f"GPU memory critical: {usage_percent:.1f}%"
                elif usage_percent > 85:
                    return HealthStatus.WARNING, f"GPU memory high: {usage_percent:.1f}%"
                else:
                    return HealthStatus.HEALTHY, f"GPU memory normal: {usage_percent:.1f}%"
            else:
                return HealthStatus.HEALTHY, "GPU not available (expected)"
        except ImportError:
            return HealthStatus.HEALTHY, "PyTorch not available (expected)"
        except Exception as e:
            return HealthStatus.WARNING, f"GPU check failed: {e}"
            
    # DNA-specific checks
    def check_encoding_performance():
        # Check recent encoding performance
        recent_metrics = metrics_collector.get_metric_history("operation.duration.encode_image", 10)
        if recent_metrics:
            recent_durations = [m.value for m in recent_metrics]
            avg_duration = np.mean(recent_durations)
            
            if avg_duration > 30.0:  # 30 seconds threshold
                return HealthStatus.WARNING, f"Encoding performance degraded: {avg_duration:.1f}s avg"
            else:
                return HealthStatus.HEALTHY, f"Encoding performance good: {avg_duration:.1f}s avg"
        else:
            return HealthStatus.HEALTHY, "No recent encoding operations"
            
    # Create health check objects
    checks.extend([
        HealthCheck("memory", check_memory, interval=30.0),
        HealthCheck("disk_space", check_disk_space, interval=60.0),
        HealthCheck("cpu", check_cpu, interval=30.0),
        HealthCheck("gpu", check_gpu, interval=30.0),
        HealthCheck("encoding_performance", check_encoding_performance, interval=120.0)
    ])
    
    return checks


class MonitoringDashboard:
    """Simple text-based monitoring dashboard."""
    
    def __init__(self, health_monitor: HealthMonitor, metrics_collector: MetricsCollector):
        self.health_monitor = health_monitor
        self.metrics_collector = metrics_collector
        
    def generate_status_report(self) -> str:
        """Generate a comprehensive status report."""
        lines = []
        lines.append("=" * 60)
        lines.append("DNA ORIGAMI AUTOENCODER - SYSTEM STATUS")
        lines.append("=" * 60)
        
        # Overall health
        health_status = self.health_monitor.get_health_status()
        status_symbol = {
            "healthy": "✓",
            "warning": "⚠", 
            "degraded": "⚠",
            "critical": "✗",
            "unknown": "?"
        }
        
        symbol = status_symbol.get(health_status["overall_status"], "?")
        lines.append(f"Overall Status: {symbol} {health_status['overall_status'].upper()}")
        lines.append("")
        
        # Health checks
        lines.append("HEALTH CHECKS:")
        lines.append("-" * 40)
        for name, check in health_status["checks"].items():
            symbol = status_symbol.get(check["status"], "?")
            lines.append(f"{symbol} {name:<20} {check['status']:<10} {check['message']}")
        lines.append("")
        
        # Key metrics
        lines.append("KEY METRICS:")
        lines.append("-" * 40)
        
        key_metrics = [
            "system.cpu.percent",
            "system.memory.percent", 
            "system.disk.percent",
            "gpu.memory.percent"
        ]
        
        for metric_name in key_metrics:
            latest = self.metrics_collector.get_latest_metric(metric_name)
            if latest:
                lines.append(f"{metric_name:<25} {latest.value:>8.1f} {latest.unit}")
            
        lines.append("")
        
        # Recent operations
        lines.append("RECENT OPERATIONS:")
        lines.append("-" * 40)
        
        operation_metrics = [name for name in self.metrics_collector.get_all_metrics().keys() 
                           if name.startswith("operation.duration")]
        
        for metric_name in operation_metrics[-5:]:  # Last 5 operations
            stats = self.metrics_collector.get_metric_statistics(metric_name, 10)
            if stats:
                op_name = metric_name.replace("operation.duration.", "")
                lines.append(f"{op_name:<20} {stats['mean']:>6.2f}s avg, {stats['count']:>3d} calls")
                
        lines.append("")
        lines.append(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Global monitoring instances
_metrics_collector = None
_health_monitor = None
_profiler = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance.""" 
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor(get_metrics_collector())
        # Register standard health checks
        for check in create_standard_health_checks(get_metrics_collector()):
            _health_monitor.register_health_check(check)
    return _health_monitor

def get_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler(get_metrics_collector())
    return _profiler

def start_monitoring():
    """Start the global monitoring system."""
    monitor = get_health_monitor()
    monitor.start_monitoring()
    logging.info("Global monitoring system started")

def stop_monitoring():
    """Stop the global monitoring system."""
    monitor = get_health_monitor()
    monitor.stop_monitoring()
    logging.info("Global monitoring system stopped")

def get_status_report() -> str:
    """Get a comprehensive system status report."""
    dashboard = MonitoringDashboard(get_health_monitor(), get_metrics_collector())
    return dashboard.generate_status_report()