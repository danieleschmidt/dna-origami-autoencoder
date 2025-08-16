"""
Adaptive Auto-Scaling System - Generation 3 Enhancement
Intelligent resource scaling based on load patterns and performance metrics.
"""

import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import json

from .logger import get_logger
from .autonomous_monitoring import AutonomousMonitor
from .quantum_acceleration import QuantumLoadBalancer, QuantumConfig

logger = get_logger(__name__)

@dataclass
class ScalingConfig:
    """Configuration for adaptive scaling system."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0  # Percentage
    target_memory_utilization: float = 80.0  # Percentage
    scale_up_threshold: float = 85.0  # Percentage
    scale_down_threshold: float = 50.0  # Percentage
    scale_up_cooldown: int = 300  # Seconds
    scale_down_cooldown: int = 600  # Seconds
    prediction_window: int = 900  # Seconds (15 minutes)
    enable_predictive_scaling: bool = True
    enable_quantum_optimization: bool = True

@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: datetime
    event_type: str  # "scale_up", "scale_down", "prediction", "optimization"
    current_instances: int
    target_instances: int
    trigger_metric: str
    metric_value: float
    reason: str

@dataclass
class ResourceInstance:
    """Resource instance representation."""
    instance_id: str
    created_at: datetime
    cpu_cores: int
    memory_gb: float
    status: str  # "initializing", "running", "terminating", "terminated"
    load_percentage: float = 0.0
    tasks_processed: int = 0
    last_activity: datetime = field(default_factory=datetime.now)

class LoadPredictor:
    """
    Machine learning-based load prediction for proactive scaling.
    """
    
    def __init__(self, prediction_window: int = 900):
        self.prediction_window = prediction_window
        self.historical_metrics = deque(maxlen=2000)  # Store 2000 data points
        self.pattern_cache = {}
        self.seasonal_patterns = defaultdict(list)
        
    def add_metric_point(self, timestamp: datetime, cpu_percent: float, 
                        memory_percent: float, request_rate: float):
        """Add a metric data point for learning."""
        metric_point = {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'request_rate': request_rate,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'minute_of_hour': timestamp.minute
        }
        
        self.historical_metrics.append(metric_point)
        
        # Update seasonal patterns
        time_key = f"{timestamp.weekday()}_{timestamp.hour}"
        self.seasonal_patterns[time_key].append({
            'cpu': cpu_percent,
            'memory': memory_percent,
            'requests': request_rate
        })
        
        # Limit seasonal pattern history
        if len(self.seasonal_patterns[time_key]) > 50:
            self.seasonal_patterns[time_key] = self.seasonal_patterns[time_key][-30:]
    
    def predict_load(self, horizon_minutes: int = 15) -> Dict[str, float]:
        """
        Predict load for the next horizon_minutes.
        Returns predicted CPU, memory, and request rate.
        """
        if len(self.historical_metrics) < 10:
            return self._get_default_prediction()
        
        current_time = datetime.now()
        future_time = current_time + timedelta(minutes=horizon_minutes)
        
        # Get recent trend
        recent_data = list(self.historical_metrics)[-20:]  # Last 20 points
        trend_prediction = self._calculate_trend_prediction(recent_data, horizon_minutes)
        
        # Get seasonal prediction
        seasonal_prediction = self._get_seasonal_prediction(future_time)
        
        # Combine predictions (weighted average)
        combined_prediction = {}
        for metric in ['cpu_percent', 'memory_percent', 'request_rate']:
            trend_value = trend_prediction.get(metric, 50.0)
            seasonal_value = seasonal_prediction.get(metric, 50.0)
            
            # Weight recent trend more heavily for short-term predictions
            weight_trend = 0.7 if horizon_minutes <= 30 else 0.4
            weight_seasonal = 1.0 - weight_trend
            
            combined_value = (trend_value * weight_trend + seasonal_value * weight_seasonal)
            combined_prediction[metric] = max(0.0, min(100.0, combined_value))
        
        return combined_prediction
    
    def _calculate_trend_prediction(self, recent_data: List[Dict], horizon_minutes: int) -> Dict[str, float]:
        """Calculate prediction based on recent trends."""
        if len(recent_data) < 3:
            return self._get_default_prediction()
        
        predictions = {}
        
        for metric in ['cpu_percent', 'memory_percent', 'request_rate']:
            values = [point[metric] for point in recent_data]
            
            # Simple linear trend calculation
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)  # Linear fit
            
            # Predict future value
            future_x = len(values) + (horizon_minutes / 5)  # Assuming 5-minute intervals
            predicted_value = coeffs[0] * future_x + coeffs[1]
            
            predictions[metric] = predicted_value
        
        return predictions
    
    def _get_seasonal_prediction(self, future_time: datetime) -> Dict[str, float]:
        """Get prediction based on seasonal/daily patterns."""
        time_key = f"{future_time.weekday()}_{future_time.hour}"
        
        if time_key not in self.seasonal_patterns or not self.seasonal_patterns[time_key]:
            return self._get_default_prediction()
        
        pattern_data = self.seasonal_patterns[time_key]
        
        return {
            'cpu_percent': np.mean([p['cpu'] for p in pattern_data]),
            'memory_percent': np.mean([p['memory'] for p in pattern_data]),
            'request_rate': np.mean([p['requests'] for p in pattern_data])
        }
    
    def _get_default_prediction(self) -> Dict[str, float]:
        """Get default prediction when insufficient data."""
        return {
            'cpu_percent': 50.0,
            'memory_percent': 60.0,
            'request_rate': 10.0
        }
    
    def get_prediction_confidence(self, prediction: Dict[str, float]) -> float:
        """Calculate confidence level for prediction."""
        if len(self.historical_metrics) < 20:
            return 0.3  # Low confidence with little data
        
        # Calculate variance in recent data
        recent_data = list(self.historical_metrics)[-20:]
        variances = []
        
        for metric in ['cpu_percent', 'memory_percent', 'request_rate']:
            values = [point[metric] for point in recent_data]
            variance = np.var(values)
            variances.append(variance)
        
        # Higher variance = lower confidence
        avg_variance = np.mean(variances)
        confidence = max(0.1, 1.0 - (avg_variance / 1000.0))  # Normalize variance
        
        return min(1.0, confidence)

class AdaptiveScaler:
    """
    Main adaptive scaling system with predictive capabilities.
    """
    
    def __init__(self, config: ScalingConfig = None, monitor: AutonomousMonitor = None):
        self.config = config or ScalingConfig()
        self.monitor = monitor
        self.logger = get_logger(f"{__name__}.AdaptiveScaler")
        
        # Scaling state
        self.instances = {}
        self.current_instance_count = self.config.min_instances
        self.scaling_events = deque(maxlen=1000)
        self.last_scale_up = datetime.min
        self.last_scale_down = datetime.min
        
        # Load prediction
        self.load_predictor = LoadPredictor(self.config.prediction_window)
        
        # Quantum optimization
        self.quantum_balancer = None
        if self.config.enable_quantum_optimization:
            self.quantum_balancer = QuantumLoadBalancer(self.current_instance_count)
        
        # Monitoring and control
        self.active = True
        self.scaling_thread = None
        
        # Initialize instances
        self._initialize_instances()
        
        # Start scaling loop
        self._start_scaling_loop()
    
    def _initialize_instances(self):
        """Initialize minimum number of instances."""
        for i in range(self.config.min_instances):
            instance = self._create_instance(f"instance_{i}")
            self.instances[instance.instance_id] = instance
        
        self.logger.info(f"Initialized {self.config.min_instances} instances")
    
    def _create_instance(self, instance_id: str) -> ResourceInstance:
        """Create a new resource instance."""
        return ResourceInstance(
            instance_id=instance_id,
            created_at=datetime.now(),
            cpu_cores=4,  # Default configuration
            memory_gb=8.0,
            status="running"
        )
    
    def _start_scaling_loop(self):
        """Start the adaptive scaling monitoring loop."""
        def scaling_loop():
            while self.active:
                try:
                    self._scaling_iteration()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Scaling loop error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self.scaling_thread = threading.Thread(target=scaling_loop, daemon=True)
        self.scaling_thread.start()
        self.logger.info("Adaptive scaling loop started")
    
    def _scaling_iteration(self):
        """Single iteration of scaling analysis and decision."""
        current_time = datetime.now()
        
        # Collect current metrics
        current_metrics = self._collect_current_metrics()
        
        # Add to predictor
        self.load_predictor.add_metric_point(
            current_time,
            current_metrics['cpu_percent'],
            current_metrics['memory_percent'],
            current_metrics['request_rate']
        )
        
        # Get predictions if enabled
        if self.config.enable_predictive_scaling:
            predictions = self.load_predictor.predict_load(15)  # 15-minute horizon
            confidence = self.load_predictor.get_prediction_confidence(predictions)
        else:
            predictions = current_metrics
            confidence = 1.0
        
        # Make scaling decision
        scaling_decision = self._make_scaling_decision(current_metrics, predictions, confidence)
        
        # Execute scaling if needed
        if scaling_decision['action'] != 'none':
            self._execute_scaling(scaling_decision)
        
        # Update instance loads
        self._update_instance_loads(current_metrics)
        
        # Optimize quantum balancer if enabled
        if self.quantum_balancer and self.config.enable_quantum_optimization:
            self.quantum_balancer.optimize_all_processors()
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        if self.monitor:
            status = self.monitor.get_current_status()
            latest_metrics = status.get('latest_metrics', {})
            
            cpu_percent = latest_metrics.get('system.cpu_percent', 50.0)
            memory_percent = latest_metrics.get('system.memory_percent', 60.0)
            request_rate = latest_metrics.get('application.requests_per_second', 10.0)
        else:
            # Simulate metrics if no monitor available
            cpu_percent = 50.0 + np.random.normal(0, 10)
            memory_percent = 60.0 + np.random.normal(0, 15)
            request_rate = 10.0 + np.random.normal(0, 5)
        
        return {
            'cpu_percent': max(0.0, min(100.0, cpu_percent)),
            'memory_percent': max(0.0, min(100.0, memory_percent)),
            'request_rate': max(0.0, request_rate)
        }
    
    def _make_scaling_decision(self, current_metrics: Dict[str, float], 
                             predictions: Dict[str, float], confidence: float) -> Dict[str, Any]:
        """Make intelligent scaling decision based on current and predicted metrics."""
        current_time = datetime.now()
        
        # Check cooldown periods
        scale_up_cooldown_ok = (current_time - self.last_scale_up).total_seconds() > self.config.scale_up_cooldown
        scale_down_cooldown_ok = (current_time - self.last_scale_down).total_seconds() > self.config.scale_down_cooldown
        
        # Determine primary metrics to consider
        metrics_to_use = predictions if confidence > 0.6 else current_metrics
        
        cpu_load = metrics_to_use['cpu_percent']
        memory_load = metrics_to_use['memory_percent']
        
        # Scale up conditions
        scale_up_needed = False
        scale_up_reason = ""
        
        if cpu_load > self.config.scale_up_threshold and scale_up_cooldown_ok:
            scale_up_needed = True
            scale_up_reason = f"High CPU load: {cpu_load:.1f}%"
        elif memory_load > self.config.scale_up_threshold and scale_up_cooldown_ok:
            scale_up_needed = True
            scale_up_reason = f"High memory load: {memory_load:.1f}%"
        elif (cpu_load > self.config.target_cpu_utilization and 
              memory_load > self.config.target_memory_utilization and
              confidence > 0.8 and scale_up_cooldown_ok):
            scale_up_needed = True
            scale_up_reason = f"Predicted high load (confidence: {confidence:.1%})"
        
        # Scale down conditions
        scale_down_needed = False
        scale_down_reason = ""
        
        if (cpu_load < self.config.scale_down_threshold and 
            memory_load < self.config.scale_down_threshold and 
            scale_down_cooldown_ok and
            self.current_instance_count > self.config.min_instances):
            scale_down_needed = True
            scale_down_reason = f"Low resource utilization: CPU {cpu_load:.1f}%, Memory {memory_load:.1f}%"
        
        # Determine action
        if scale_up_needed and self.current_instance_count < self.config.max_instances:
            target_instances = min(self.config.max_instances, self.current_instance_count + 1)
            return {
                'action': 'scale_up',
                'target_instances': target_instances,
                'reason': scale_up_reason,
                'trigger_metric': 'cpu_percent' if 'CPU' in scale_up_reason else 'memory_percent',
                'metric_value': cpu_load if 'CPU' in scale_up_reason else memory_load,
                'confidence': confidence
            }
        elif scale_down_needed:
            target_instances = max(self.config.min_instances, self.current_instance_count - 1)
            return {
                'action': 'scale_down',
                'target_instances': target_instances,
                'reason': scale_down_reason,
                'trigger_metric': 'combined',
                'metric_value': (cpu_load + memory_load) / 2,
                'confidence': confidence
            }
        else:
            return {
                'action': 'none',
                'target_instances': self.current_instance_count,
                'reason': 'No scaling needed',
                'trigger_metric': 'none',
                'metric_value': 0.0,
                'confidence': confidence
            }
    
    def _execute_scaling(self, decision: Dict[str, Any]):
        """Execute the scaling decision."""
        current_instances = self.current_instance_count
        target_instances = decision['target_instances']
        
        if decision['action'] == 'scale_up':
            self._scale_up(target_instances - current_instances)
            self.last_scale_up = datetime.now()
        elif decision['action'] == 'scale_down':
            self._scale_down(current_instances - target_instances)
            self.last_scale_down = datetime.now()
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=datetime.now(),
            event_type=decision['action'],
            current_instances=current_instances,
            target_instances=target_instances,
            trigger_metric=decision['trigger_metric'],
            metric_value=decision['metric_value'],
            reason=decision['reason']
        )
        
        self.scaling_events.append(event)
        
        self.logger.info(
            f"Scaling {decision['action']}: {current_instances} -> {target_instances} "
            f"({decision['reason']})"
        )
    
    def _scale_up(self, count: int):
        """Scale up by adding instances."""
        for i in range(count):
            instance_id = f"instance_{len(self.instances)}"
            instance = self._create_instance(instance_id)
            self.instances[instance_id] = instance
            self.current_instance_count += 1
            
            # Update quantum balancer if enabled
            if self.quantum_balancer:
                # Add processor to quantum balancer
                self.quantum_balancer.num_processors += 1
                new_config = QuantumConfig(
                    max_workers=max(2, self.quantum_balancer.processors[0].config.max_workers),
                    quantum_parallelism=max(2, self.quantum_balancer.processors[0].config.quantum_parallelism)
                )
                from .quantum_acceleration import QuantumProcessor
                new_processor = QuantumProcessor(new_config)
                self.quantum_balancer.processors.append(new_processor)
        
        self.logger.info(f"Scaled up: added {count} instances")
    
    def _scale_down(self, count: int):
        """Scale down by removing instances."""
        instances_to_remove = list(self.instances.keys())[-count:]  # Remove newest instances
        
        for instance_id in instances_to_remove:
            if instance_id in self.instances:
                self.instances[instance_id].status = "terminating"
                del self.instances[instance_id]
                self.current_instance_count -= 1
        
        # Update quantum balancer if enabled
        if self.quantum_balancer and count > 0:
            processors_to_remove = self.quantum_balancer.processors[-count:]
            for processor in processors_to_remove:
                processor.shutdown()
            self.quantum_balancer.processors = self.quantum_balancer.processors[:-count]
            self.quantum_balancer.num_processors -= count
        
        self.logger.info(f"Scaled down: removed {count} instances")
    
    def _update_instance_loads(self, metrics: Dict[str, float]):
        """Update load information for instances."""
        avg_cpu = metrics['cpu_percent']
        avg_memory = metrics['memory_percent']
        
        # Distribute load across instances (simplified)
        base_load = (avg_cpu + avg_memory) / 2
        
        for instance in self.instances.values():
            # Add some variance to simulate different instance loads
            variance = np.random.normal(0, 5)
            instance.load_percentage = max(0.0, min(100.0, base_load + variance))
            instance.last_activity = datetime.now()
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling system metrics."""
        current_time = datetime.now()
        
        # Recent scaling events (last 24 hours)
        recent_events = [
            event for event in self.scaling_events
            if (current_time - event.timestamp).total_seconds() < 86400
        ]
        
        scale_up_events = [e for e in recent_events if e.event_type == 'scale_up']
        scale_down_events = [e for e in recent_events if e.event_type == 'scale_down']
        
        # Instance statistics
        running_instances = [i for i in self.instances.values() if i.status == 'running']
        avg_instance_load = np.mean([i.load_percentage for i in running_instances]) if running_instances else 0.0
        
        # Prediction confidence
        recent_prediction = self.load_predictor.predict_load(15)
        prediction_confidence = self.load_predictor.get_prediction_confidence(recent_prediction)
        
        metrics = {
            'current_state': {
                'instance_count': self.current_instance_count,
                'running_instances': len(running_instances),
                'average_instance_load': avg_instance_load,
                'quantum_optimization_enabled': self.config.enable_quantum_optimization
            },
            'scaling_activity': {
                'scale_up_events_24h': len(scale_up_events),
                'scale_down_events_24h': len(scale_down_events),
                'last_scale_up': self.last_scale_up.isoformat() if self.last_scale_up != datetime.min else None,
                'last_scale_down': self.last_scale_down.isoformat() if self.last_scale_down != datetime.min else None
            },
            'prediction': {
                'next_15min_cpu': recent_prediction['cpu_percent'],
                'next_15min_memory': recent_prediction['memory_percent'],
                'next_15min_requests': recent_prediction['request_rate'],
                'prediction_confidence': prediction_confidence,
                'historical_data_points': len(self.load_predictor.historical_metrics)
            },
            'configuration': {
                'min_instances': self.config.min_instances,
                'max_instances': self.config.max_instances,
                'target_cpu_utilization': self.config.target_cpu_utilization,
                'target_memory_utilization': self.config.target_memory_utilization,
                'predictive_scaling_enabled': self.config.enable_predictive_scaling
            }
        }
        
        # Add quantum metrics if available
        if self.quantum_balancer:
            quantum_metrics = self.quantum_balancer.get_load_balancer_metrics()
            metrics['quantum_optimization'] = quantum_metrics
        
        return metrics
    
    def get_instance_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all instances."""
        instance_details = []
        
        for instance in self.instances.values():
            details = {
                'instance_id': instance.instance_id,
                'status': instance.status,
                'created_at': instance.created_at.isoformat(),
                'uptime_hours': (datetime.now() - instance.created_at).total_seconds() / 3600,
                'cpu_cores': instance.cpu_cores,
                'memory_gb': instance.memory_gb,
                'load_percentage': instance.load_percentage,
                'tasks_processed': instance.tasks_processed,
                'last_activity': instance.last_activity.isoformat()
            }
            instance_details.append(details)
        
        return instance_details
    
    def force_scale(self, target_instances: int, reason: str = "Manual scaling"):
        """Force scaling to a specific number of instances."""
        if target_instances < self.config.min_instances:
            target_instances = self.config.min_instances
        elif target_instances > self.config.max_instances:
            target_instances = self.config.max_instances
        
        current_instances = self.current_instance_count
        
        if target_instances > current_instances:
            self._scale_up(target_instances - current_instances)
        elif target_instances < current_instances:
            self._scale_down(current_instances - target_instances)
        
        # Record manual scaling event
        event = ScalingEvent(
            timestamp=datetime.now(),
            event_type="manual_scaling",
            current_instances=current_instances,
            target_instances=target_instances,
            trigger_metric="manual",
            metric_value=0.0,
            reason=reason
        )
        self.scaling_events.append(event)
        
        self.logger.info(f"Manual scaling: {current_instances} -> {target_instances} ({reason})")
    
    def shutdown(self):
        """Shutdown the adaptive scaling system."""
        self.active = False
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
        
        if self.quantum_balancer:
            self.quantum_balancer.shutdown()
        
        self.logger.info("Adaptive scaling system shutdown complete")

# Integration function
def create_adaptive_scaler(config: ScalingConfig = None, 
                          monitor: AutonomousMonitor = None) -> AdaptiveScaler:
    """
    Create and start adaptive scaling system.
    """
    scaler = AdaptiveScaler(config, monitor)
    
    logger.info("Adaptive scaling system created and started")
    return scaler