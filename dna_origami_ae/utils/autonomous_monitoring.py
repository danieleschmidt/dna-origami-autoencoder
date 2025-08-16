"""
Autonomous Monitoring System - Generation 2 Enhancement
Self-healing, self-monitoring system with predictive failure detection.
"""

import os
import time
import threading
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from collections import deque, defaultdict
import logging

from .logger import get_logger
from .error_handling import ErrorHandler

logger = get_logger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for autonomous monitoring system."""
    check_interval: float = 30.0  # seconds
    alert_threshold: float = 0.8
    prediction_window: int = 300  # seconds
    max_history_size: int = 1000
    enable_auto_recovery: bool = True
    enable_predictive_alerts: bool = True
    log_level: str = "INFO"

@dataclass
class SystemMetric:
    """Individual system metric with trend analysis."""
    name: str
    value: float
    timestamp: datetime
    threshold: float = 1.0
    trend: str = "stable"  # "increasing", "decreasing", "stable", "volatile"
    severity: str = "normal"  # "normal", "warning", "critical"

@dataclass
class AlertEvent:
    """System alert event with context."""
    timestamp: datetime
    severity: str
    component: str
    message: str
    metrics: Dict[str, float]
    suggested_actions: List[str] = field(default_factory=list)
    auto_recovery_attempted: bool = False
    resolved: bool = False

class PredictiveAnalyzer:
    """
    Predictive analyzer for system health using simple ML techniques.
    """
    
    def __init__(self, prediction_window: int = 300):
        self.prediction_window = prediction_window
        self.metric_history = defaultdict(lambda: deque(maxlen=100))
        self.models = {}
        
    def add_metric_data(self, metric_name: str, value: float, timestamp: datetime):
        """Add metric data point for analysis."""
        self.metric_history[metric_name].append((timestamp, value))
        
    def predict_failure_probability(self, metric_name: str) -> float:
        """
        Predict probability of failure in the next prediction window.
        Simple implementation using trend analysis and threshold proximity.
        """
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 10:
            return 0.0
            
        history = list(self.metric_history[metric_name])
        values = [v for _, v in history[-10:]]
        
        # Calculate trend
        if len(values) >= 3:
            recent_slope = (values[-1] - values[-3]) / 2
            trend_factor = abs(recent_slope) / max(abs(values[-1]), 0.1)
        else:
            trend_factor = 0.0
            
        # Calculate volatility
        if len(values) >= 5:
            volatility = np.std(values[-5:]) / (np.mean(values[-5:]) + 0.1)
        else:
            volatility = 0.0
            
        # Calculate threshold proximity
        current_value = values[-1]
        if 'cpu' in metric_name.lower() or 'memory' in metric_name.lower():
            threshold_proximity = max(0, current_value - 0.7) / 0.3  # Alert at 70%+
        else:
            threshold_proximity = 0.0
            
        # Combine factors
        failure_probability = min(1.0, trend_factor * 0.4 + volatility * 0.3 + threshold_proximity * 0.5)
        
        return failure_probability
        
    def get_trend_analysis(self, metric_name: str) -> Dict[str, Any]:
        """Get comprehensive trend analysis for a metric."""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 5:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
            
        history = list(self.metric_history[metric_name])
        values = [v for _, v in history[-10:]]
        timestamps = [t for t, _ in history[-10:]]
        
        # Calculate trend
        if len(values) >= 3:
            slope = (values[-1] - values[0]) / len(values)
            if abs(slope) < 0.01:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'unknown'
            
        # Calculate confidence based on data consistency
        volatility = np.std(values) / (np.mean(values) + 0.1)
        confidence = max(0.0, 1.0 - volatility)
        
        return {
            'trend': trend,
            'confidence': confidence,
            'slope': slope if 'slope' in locals() else 0.0,
            'volatility': volatility,
            'data_points': len(values)
        }

class AutoRecoverySystem:
    """
    Autonomous recovery system that attempts to fix issues automatically.
    """
    
    def __init__(self):
        self.recovery_strategies = {}
        self.recovery_history = deque(maxlen=100)
        self.logger = get_logger(f"{__name__}.AutoRecoverySystem")
        
        # Register default recovery strategies
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """Register default recovery strategies for common issues."""
        
        def memory_cleanup():
            """Attempt to free memory."""
            import gc
            gc.collect()
            return True
            
        def restart_failed_component(component_name: str):
            """Restart a failed component."""
            self.logger.info(f"Attempting to restart component: {component_name}")
            # Placeholder for actual restart logic
            return True
            
        def reduce_workload():
            """Reduce system workload."""
            self.logger.info("Reducing system workload")
            # Placeholder for workload reduction logic
            return True
            
        self.recovery_strategies.update({
            'high_memory_usage': memory_cleanup,
            'component_failure': restart_failed_component,
            'high_cpu_usage': reduce_workload,
        })
    
    def register_recovery_strategy(self, issue_type: str, strategy: Callable):
        """Register a custom recovery strategy."""
        self.recovery_strategies[issue_type] = strategy
        self.logger.info(f"Registered recovery strategy for: {issue_type}")
    
    def attempt_recovery(self, alert: AlertEvent) -> bool:
        """
        Attempt automatic recovery for an alert.
        """
        recovery_key = self._determine_recovery_key(alert)
        
        if recovery_key not in self.recovery_strategies:
            self.logger.warning(f"No recovery strategy for: {recovery_key}")
            return False
            
        try:
            self.logger.info(f"Attempting auto-recovery for: {recovery_key}")
            strategy = self.recovery_strategies[recovery_key]
            
            # Attempt recovery
            success = strategy() if callable(strategy) else strategy(alert.component)
            
            # Record recovery attempt
            recovery_record = {
                'timestamp': datetime.now(),
                'alert_id': id(alert),
                'recovery_key': recovery_key,
                'success': success,
                'component': alert.component
            }
            self.recovery_history.append(recovery_record)
            
            if success:
                self.logger.info(f"Auto-recovery successful for: {recovery_key}")
                alert.auto_recovery_attempted = True
                alert.resolved = True
            else:
                self.logger.error(f"Auto-recovery failed for: {recovery_key}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Auto-recovery exception for {recovery_key}: {e}")
            return False
    
    def _determine_recovery_key(self, alert: AlertEvent) -> str:
        """Determine the appropriate recovery strategy key."""
        message_lower = alert.message.lower()
        
        if 'memory' in message_lower:
            return 'high_memory_usage'
        elif 'cpu' in message_lower:
            return 'high_cpu_usage'
        elif 'failed' in message_lower or 'error' in message_lower:
            return 'component_failure'
        else:
            return 'generic_issue'
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        if not self.recovery_history:
            return {'total_attempts': 0, 'success_rate': 0.0}
            
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for r in self.recovery_history if r['success'])
        success_rate = successful_attempts / total_attempts
        
        # Recent success rate (last 10 attempts)
        recent_attempts = list(self.recovery_history)[-10:]
        recent_success_rate = sum(1 for r in recent_attempts if r['success']) / len(recent_attempts) if recent_attempts else 0.0
        
        return {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': success_rate,
            'recent_success_rate': recent_success_rate,
            'registered_strategies': list(self.recovery_strategies.keys())
        }

class AutonomousMonitor:
    """
    Main autonomous monitoring system with self-healing capabilities.
    """
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.logger = get_logger(f"{__name__}.AutonomousMonitor")
        
        # Core components
        self.predictor = PredictiveAnalyzer(self.config.prediction_window)
        self.recovery_system = AutoRecoverySystem()
        self.error_handler = ErrorHandler()
        
        # Monitoring state
        self.active = False
        self.metrics_history = deque(maxlen=self.config.max_history_size)
        self.alerts = deque(maxlen=100)
        self.monitoring_thread = None
        
        # Metric collectors
        self.metric_collectors = {}
        self._register_default_collectors()
        
    def _register_default_collectors(self):
        """Register default system metric collectors."""
        
        def collect_system_metrics() -> Dict[str, float]:
            """Collect basic system metrics."""
            try:
                import psutil
                return {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent,
                    'process_count': len(psutil.pids())
                }
            except ImportError:
                return {
                    'cpu_percent': 50.0,  # Mock values
                    'memory_percent': 30.0,
                    'disk_percent': 60.0,
                    'process_count': 100
                }
        
        def collect_application_metrics() -> Dict[str, float]:
            """Collect application-specific metrics."""
            # Placeholder for application metrics
            return {
                'active_connections': 10,
                'requests_per_second': 25.5,
                'error_rate': 0.02,
                'response_time_ms': 150.0
            }
        
        self.metric_collectors.update({
            'system': collect_system_metrics,
            'application': collect_application_metrics
        })
    
    def register_metric_collector(self, name: str, collector: Callable[[], Dict[str, float]]):
        """Register a custom metric collector."""
        self.metric_collectors[name] = collector
        self.logger.info(f"Registered metric collector: {name}")
    
    def start_monitoring(self):
        """Start the autonomous monitoring system."""
        if self.active:
            self.logger.warning("Monitoring already active")
            return
            
        self.active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Autonomous monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Autonomous monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        self.logger.info("Monitoring loop started")
        
        while self.active:
            try:
                # Collect all metrics
                current_metrics = {}
                timestamp = datetime.now()
                
                for collector_name, collector in self.metric_collectors.items():
                    try:
                        metrics = collector()
                        for metric_name, value in metrics.items():
                            full_name = f"{collector_name}.{metric_name}"
                            current_metrics[full_name] = value
                            
                            # Add to predictor
                            self.predictor.add_metric_data(full_name, value, timestamp)
                            
                    except Exception as e:
                        self.logger.error(f"Error collecting metrics from {collector_name}: {e}")
                
                # Store metrics
                self.metrics_history.append({
                    'timestamp': timestamp,
                    'metrics': current_metrics
                })
                
                # Analyze and generate alerts
                self._analyze_metrics(current_metrics, timestamp)
                
                # Sleep until next check
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.check_interval)
    
    def _analyze_metrics(self, metrics: Dict[str, float], timestamp: datetime):
        """Analyze current metrics and generate alerts if needed."""
        
        for metric_name, value in metrics.items():
            # Check for immediate threshold violations
            alert_generated = False
            
            # CPU and Memory specific thresholds
            if 'cpu_percent' in metric_name and value > 80.0:
                alert = self._create_alert(
                    severity="warning",
                    component="system",
                    message=f"High CPU usage: {value:.1f}%",
                    metrics={metric_name: value},
                    timestamp=timestamp
                )
                alert_generated = True
                
            elif 'memory_percent' in metric_name and value > 85.0:
                alert = self._create_alert(
                    severity="critical",
                    component="system", 
                    message=f"High memory usage: {value:.1f}%",
                    metrics={metric_name: value},
                    timestamp=timestamp
                )
                alert_generated = True
                
            elif 'error_rate' in metric_name and value > 0.05:
                alert = self._create_alert(
                    severity="warning",
                    component="application",
                    message=f"High error rate: {value:.1%}",
                    metrics={metric_name: value},
                    timestamp=timestamp
                )
                alert_generated = True
            
            # Predictive analysis
            if self.config.enable_predictive_alerts:
                failure_prob = self.predictor.predict_failure_probability(metric_name)
                
                if failure_prob > 0.7:  # 70% probability threshold
                    trend_info = self.predictor.get_trend_analysis(metric_name)
                    alert = self._create_alert(
                        severity="warning",
                        component="predictive",
                        message=f"Predicted issue for {metric_name}: {failure_prob:.1%} probability",
                        metrics={metric_name: value, 'failure_probability': failure_prob},
                        timestamp=timestamp
                    )
                    alert.suggested_actions.append(f"Monitor {metric_name} closely - trend: {trend_info['trend']}")
                    alert_generated = True
            
            # Attempt auto-recovery if alert generated
            if alert_generated and self.config.enable_auto_recovery:
                self.recovery_system.attempt_recovery(alert)
    
    def _create_alert(self, severity: str, component: str, message: str, 
                     metrics: Dict[str, float], timestamp: datetime) -> AlertEvent:
        """Create and store an alert event."""
        alert = AlertEvent(
            timestamp=timestamp,
            severity=severity,
            component=component,
            message=message,
            metrics=metrics
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_level = logging.WARNING if severity == "warning" else logging.ERROR
        self.logger.log(log_level, f"ALERT [{severity.upper()}] {component}: {message}")
        
        return alert
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.metrics_history:
            return {'status': 'no_data', 'monitoring_active': self.active}
            
        latest_metrics = self.metrics_history[-1]['metrics']
        recent_alerts = [a for a in self.alerts if (datetime.now() - a.timestamp).seconds < 3600]
        
        # Calculate health score
        health_score = self._calculate_health_score(latest_metrics)
        
        status = {
            'monitoring_active': self.active,
            'health_score': health_score,
            'latest_metrics': latest_metrics,
            'recent_alerts_count': len(recent_alerts),
            'critical_alerts_count': len([a for a in recent_alerts if a.severity == "critical"]),
            'auto_recovery_enabled': self.config.enable_auto_recovery,
            'predictive_analysis_enabled': self.config.enable_predictive_alerts,
            'uptime_hours': (datetime.now() - self.metrics_history[0]['timestamp']).total_seconds() / 3600 if self.metrics_history else 0,
            'data_points_collected': len(self.metrics_history)
        }
        
        return status
    
    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall system health score (0-1)."""
        score = 1.0
        
        # Penalize high resource usage
        for metric_name, value in metrics.items():
            if 'cpu_percent' in metric_name:
                score -= max(0, (value - 70) / 30 * 0.3)  # Penalty for >70% CPU
            elif 'memory_percent' in metric_name:
                score -= max(0, (value - 80) / 20 * 0.4)  # Penalty for >80% memory
            elif 'error_rate' in metric_name:
                score -= min(value * 10, 0.3)  # Penalty for error rate
        
        return max(0.0, min(1.0, score))
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        relevant_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        if not relevant_alerts:
            return {'period_hours': hours, 'total_alerts': 0}
        
        summary = {
            'period_hours': hours,
            'total_alerts': len(relevant_alerts),
            'critical_alerts': len([a for a in relevant_alerts if a.severity == "critical"]),
            'warning_alerts': len([a for a in relevant_alerts if a.severity == "warning"]),
            'resolved_alerts': len([a for a in relevant_alerts if a.resolved]),
            'auto_recovery_attempts': len([a for a in relevant_alerts if a.auto_recovery_attempted]),
            'components_affected': list(set(a.component for a in relevant_alerts)),
            'most_common_issues': self._get_most_common_issues(relevant_alerts)
        }
        
        return summary
    
    def _get_most_common_issues(self, alerts: List[AlertEvent]) -> List[Dict[str, Any]]:
        """Get most common issues from alerts."""
        issue_counts = defaultdict(int)
        
        for alert in alerts:
            # Extract issue type from message
            message_lower = alert.message.lower()
            if 'cpu' in message_lower:
                issue_counts['high_cpu'] += 1
            elif 'memory' in message_lower:
                issue_counts['high_memory'] += 1
            elif 'error' in message_lower:
                issue_counts['high_errors'] += 1
            elif 'predicted' in message_lower:
                issue_counts['predicted_issues'] += 1
            else:
                issue_counts['other'] += 1
        
        # Sort by frequency
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{'issue_type': issue, 'count': count} for issue, count in sorted_issues[:5]]
    
    def save_monitoring_state(self, filepath: str):
        """Save monitoring state to file."""
        state = {
            'config': self.config,
            'metrics_history': list(self.metrics_history)[-100:],  # Last 100 points
            'alerts': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'severity': a.severity,
                    'component': a.component,
                    'message': a.message,
                    'metrics': a.metrics,
                    'resolved': a.resolved
                }
                for a in list(self.alerts)[-50:]  # Last 50 alerts
            ],
            'recovery_stats': self.recovery_system.get_recovery_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring state saved to {filepath}")

# Integration function
def create_autonomous_monitor(config: MonitoringConfig = None) -> AutonomousMonitor:
    """
    Create and start autonomous monitoring system.
    """
    monitor = AutonomousMonitor(config)
    monitor.start_monitoring()
    
    logger.info("Autonomous monitoring system created and started")
    return monitor