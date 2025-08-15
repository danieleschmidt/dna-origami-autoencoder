"""
Auto-Scaling System for DNA Origami AutoEncoder

Provides intelligent auto-scaling capabilities for DNA origami design and
fabrication workloads, including predictive scaling, resource optimization,
and cost-aware scaling decisions.
"""

import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque
import statistics
import math

from ..utils.logger import get_logger
from ..utils.performance import PerformanceOptimizer
from ..mesh.mesh_coordinator import MeshCoordinator, NodeType

logger = get_logger(__name__)


class ScalingMetric(Enum):
    """Metrics used for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization" 
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"           # React to current metrics
    PREDICTIVE = "predictive"       # Predict future demand
    SCHEDULE_BASED = "schedule"     # Scale based on schedule
    HYBRID = "hybrid"              # Combination of policies
    COST_OPTIMIZED = "cost_optimized"  # Optimize for cost


@dataclass
class ScalingRule:
    """Defines a scaling rule."""
    name: str
    metric: ScalingMetric
    threshold_up: float
    threshold_down: float
    scale_up_amount: int
    scale_down_amount: int
    cooldown_period: float  # Seconds to wait between scaling actions
    evaluation_window: int  # Number of data points to consider
    enabled: bool = True


@dataclass
class ResourceInstance:
    """Represents a scalable resource instance."""
    instance_id: str
    instance_type: str
    node_type: NodeType
    status: str  # pending, running, terminating, terminated
    created_at: float
    terminated_at: Optional[float] = None
    cost_per_hour: float = 0.0
    performance_rating: float = 1.0
    utilization: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingEvent:
    """Records a scaling event."""
    event_id: str
    timestamp: float
    action: str  # scale_up, scale_down
    trigger_metric: ScalingMetric
    trigger_value: float
    threshold: float
    instances_added: int = 0
    instances_removed: int = 0
    rule_name: str = ""
    cost_impact: float = 0.0


class AutoScaler:
    """
    Intelligent auto-scaling system for DNA origami workloads.
    
    Features:
    - Multi-metric scaling decisions
    - Predictive scaling based on historical patterns
    - Cost-aware scaling optimization
    - Integration with mesh networking
    - Custom scaling policies
    - Real-time performance monitoring
    - Resource efficiency optimization
    """
    
    def __init__(self,
                 mesh_coordinator: Optional[MeshCoordinator] = None,
                 min_instances: int = 1,
                 max_instances: int = 100,
                 scaling_policy: ScalingPolicy = ScalingPolicy.HYBRID,
                 prediction_window: int = 300,  # 5 minutes
                 cost_optimization: bool = True):
        
        self.mesh_coordinator = mesh_coordinator
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scaling_policy = scaling_policy
        self.prediction_window = prediction_window
        self.cost_optimization = cost_optimization
        
        # Resource management
        self.instances: Dict[str, ResourceInstance] = {}
        self.scaling_rules: List[ScalingRule] = []
        self.scaling_events: List[ScalingEvent] = []
        
        # Metrics and monitoring
        self.metrics_history: Dict[ScalingMetric, deque] = {
            metric: deque(maxlen=1000) for metric in ScalingMetric
        }
        self.custom_metrics: Dict[str, deque] = {}
        
        # Predictive modeling
        self.demand_patterns: Dict[str, List[float]] = {}
        self.prediction_models: Dict[ScalingMetric, Any] = {}
        self.seasonal_patterns: Dict[str, Dict[str, float]] = {}
        
        # Cost tracking
        self.cost_history: deque = deque(maxlen=1000)
        self.instance_type_costs: Dict[str, float] = {
            "small": 0.05,   # $0.05/hour
            "medium": 0.10,  # $0.10/hour  
            "large": 0.20,   # $0.20/hour
            "xlarge": 0.40,  # $0.40/hour
            "gpu": 0.80      # $0.80/hour
        }
        
        # Threading and state
        self.running = False
        self.last_scaling_action: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Performance optimization
        self.performance_optimizer = PerformanceOptimizer()
        
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        # CPU-based scaling
        self.scaling_rules.append(ScalingRule(
            name="cpu_scale_up",
            metric=ScalingMetric.CPU_UTILIZATION,
            threshold_up=80.0,
            threshold_down=30.0,
            scale_up_amount=2,
            scale_down_amount=1,
            cooldown_period=300.0,  # 5 minutes
            evaluation_window=3
        ))
        
        # Queue length scaling
        self.scaling_rules.append(ScalingRule(
            name="queue_scale_up",
            metric=ScalingMetric.QUEUE_LENGTH,
            threshold_up=10.0,
            threshold_down=2.0,
            scale_up_amount=3,
            scale_down_amount=2,
            cooldown_period=120.0,  # 2 minutes
            evaluation_window=2
        ))
        
        # Response time scaling
        self.scaling_rules.append(ScalingRule(
            name="response_time_scale",
            metric=ScalingMetric.RESPONSE_TIME,
            threshold_up=5000.0,  # 5 seconds
            threshold_down=1000.0,  # 1 second
            scale_up_amount=2,
            scale_down_amount=1,
            cooldown_period=180.0,  # 3 minutes
            evaluation_window=3
        ))
        
    async def start(self):
        """Start the auto-scaler."""
        self.running = True
        logger.info("Starting auto-scaler")
        
        # Start scaling loop
        scaling_task = asyncio.create_task(self._scaling_loop())
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        prediction_task = asyncio.create_task(self._prediction_loop())
        cost_task = asyncio.create_task(self._cost_optimization_loop())
        
        await asyncio.gather(
            scaling_task,
            metrics_task, 
            prediction_task,
            cost_task
        )
        
    async def stop(self):
        """Stop the auto-scaler."""
        self.running = False
        logger.info("Stopping auto-scaler")
        
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
        
    def remove_scaling_rule(self, rule_name: str) -> bool:
        """Remove a scaling rule by name."""
        for i, rule in enumerate(self.scaling_rules):
            if rule.name == rule_name:
                del self.scaling_rules[i]
                logger.info(f"Removed scaling rule: {rule_name}")
                return True
        return False
        
    def add_custom_metric(self, metric_name: str, metric_value: float):
        """Add a custom metric value."""
        if metric_name not in self.custom_metrics:
            self.custom_metrics[metric_name] = deque(maxlen=1000)
        
        self.custom_metrics[metric_name].append({
            "value": metric_value,
            "timestamp": time.time()
        })
        
    async def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.running:
            try:
                await self._evaluate_scaling_decisions()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(60)
                
    async def _evaluate_scaling_decisions(self):
        """Evaluate all scaling rules and make decisions."""
        with self._lock:
            for rule in self.scaling_rules:
                if not rule.enabled:
                    continue
                    
                # Check cooldown period
                last_action = self.last_scaling_action.get(rule.name, 0)
                if time.time() - last_action < rule.cooldown_period:
                    continue
                    
                await self._evaluate_rule(rule)
                
    async def _evaluate_rule(self, rule: ScalingRule):
        """Evaluate a specific scaling rule."""
        try:
            # Get recent metric values
            metric_values = self._get_recent_metric_values(
                rule.metric, rule.evaluation_window
            )
            
            if not metric_values:
                return
                
            # Calculate average value
            avg_value = statistics.mean(metric_values)
            
            # Apply scaling policy modifications
            if self.scaling_policy == ScalingPolicy.PREDICTIVE:
                avg_value = await self._apply_predictive_adjustment(rule.metric, avg_value)
            elif self.scaling_policy == ScalingPolicy.COST_OPTIMIZED:
                await self._apply_cost_optimization(rule, avg_value)
                return
                
            # Make scaling decision
            if avg_value > rule.threshold_up:
                await self._scale_up(rule, avg_value)
            elif avg_value < rule.threshold_down:
                await self._scale_down(rule, avg_value)
                
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
            
    def _get_recent_metric_values(self, 
                                 metric: ScalingMetric, 
                                 window_size: int) -> List[float]:
        """Get recent values for a metric."""
        if metric == ScalingMetric.CUSTOM_METRIC:
            # Handle custom metrics separately
            return []
            
        if metric not in self.metrics_history:
            return []
            
        recent_data = list(self.metrics_history[metric])[-window_size:]
        return [data["value"] for data in recent_data if isinstance(data, dict)]
        
    async def _apply_predictive_adjustment(self, 
                                         metric: ScalingMetric, 
                                         current_value: float) -> float:
        """Apply predictive adjustments to metric value."""
        try:
            # Get prediction for next time window
            predicted_value = await self._predict_metric_value(metric)
            
            if predicted_value is not None:
                # Weight current and predicted values
                adjusted_value = 0.7 * current_value + 0.3 * predicted_value
                
                logger.debug(
                    f"Predictive adjustment for {metric.value}: "
                    f"current={current_value:.2f}, predicted={predicted_value:.2f}, "
                    f"adjusted={adjusted_value:.2f}"
                )
                
                return adjusted_value
                
        except Exception as e:
            logger.error(f"Predictive adjustment error: {e}")
            
        return current_value
        
    async def _predict_metric_value(self, metric: ScalingMetric) -> Optional[float]:
        """Predict future metric value using historical patterns."""
        if metric not in self.metrics_history:
            return None
            
        # Get historical data
        historical_data = list(self.metrics_history[metric])
        if len(historical_data) < 10:
            return None
            
        values = [data["value"] for data in historical_data if isinstance(data, dict)]
        timestamps = [data["timestamp"] for data in historical_data if isinstance(data, dict)]
        
        if len(values) < 10:
            return None
            
        try:
            # Simple linear regression prediction
            # In production, would use more sophisticated ML models
            x = np.array(range(len(values)))
            y = np.array(values)
            
            # Linear regression: y = mx + b
            m, b = np.polyfit(x, y, 1)
            
            # Predict next value
            next_x = len(values)
            predicted_value = m * next_x + b
            
            # Apply bounds checking
            predicted_value = max(0, predicted_value)  # No negative values
            
            return float(predicted_value)
            
        except Exception as e:
            logger.error(f"Prediction error for {metric.value}: {e}")
            return None
            
    async def _scale_up(self, rule: ScalingRule, trigger_value: float):
        """Scale up resources based on rule."""
        current_instances = len([i for i in self.instances.values() if i.status == "running"])
        
        if current_instances >= self.max_instances:
            logger.warning("Cannot scale up: at maximum instance limit")
            return
            
        instances_to_add = min(
            rule.scale_up_amount,
            self.max_instances - current_instances
        )
        
        logger.info(
            f"Scaling up: adding {instances_to_add} instances "
            f"(rule: {rule.name}, trigger: {trigger_value:.2f})"
        )
        
        # Launch new instances
        new_instances = await self._launch_instances(instances_to_add, rule)
        
        # Record scaling event
        event = ScalingEvent(
            event_id=f"scale_up_{int(time.time())}",
            timestamp=time.time(),
            action="scale_up",
            trigger_metric=rule.metric,
            trigger_value=trigger_value,
            threshold=rule.threshold_up,
            instances_added=instances_to_add,
            rule_name=rule.name,
            cost_impact=sum(i.cost_per_hour for i in new_instances)
        )
        
        self.scaling_events.append(event)
        self.last_scaling_action[rule.name] = time.time()
        
    async def _scale_down(self, rule: ScalingRule, trigger_value: float):
        """Scale down resources based on rule."""
        running_instances = [i for i in self.instances.values() if i.status == "running"]
        
        if len(running_instances) <= self.min_instances:
            logger.info("Cannot scale down: at minimum instance limit")
            return
            
        instances_to_remove = min(
            rule.scale_down_amount,
            len(running_instances) - self.min_instances
        )
        
        logger.info(
            f"Scaling down: removing {instances_to_remove} instances "
            f"(rule: {rule.name}, trigger: {trigger_value:.2f})"
        )
        
        # Select instances to terminate (least utilized first)
        instances_to_terminate = self._select_instances_for_termination(instances_to_remove)
        
        # Terminate instances
        await self._terminate_instances(instances_to_terminate)
        
        # Record scaling event
        event = ScalingEvent(
            event_id=f"scale_down_{int(time.time())}",
            timestamp=time.time(),
            action="scale_down",
            trigger_metric=rule.metric,
            trigger_value=trigger_value,
            threshold=rule.threshold_down,
            instances_removed=instances_to_remove,
            rule_name=rule.name,
            cost_impact=-sum(i.cost_per_hour for i in instances_to_terminate)
        )
        
        self.scaling_events.append(event)
        self.last_scaling_action[rule.name] = time.time()
        
    async def _launch_instances(self, 
                              count: int, 
                              rule: ScalingRule) -> List[ResourceInstance]:
        """Launch new resource instances."""
        new_instances = []
        
        for i in range(count):
            # Determine instance type based on workload
            instance_type = self._select_optimal_instance_type(rule)
            
            instance = ResourceInstance(
                instance_id=f"instance_{int(time.time())}_{i}",
                instance_type=instance_type,
                node_type=NodeType.DESIGN_NODE,  # Default for now
                status="pending",
                created_at=time.time(),
                cost_per_hour=self.instance_type_costs.get(instance_type, 0.10)
            )
            
            # Add to instances
            self.instances[instance.instance_id] = instance
            new_instances.append(instance)
            
            # If mesh coordinator is available, register the new node
            if self.mesh_coordinator:
                try:
                    # This would integrate with actual mesh node creation
                    logger.info(f"Registering instance {instance.instance_id} with mesh")
                    # await self.mesh_coordinator.register_node(instance)
                except Exception as e:
                    logger.error(f"Failed to register mesh node: {e}")
                    
            # Mark as running (simplified)
            instance.status = "running"
            
        return new_instances
        
    def _select_optimal_instance_type(self, rule: ScalingRule) -> str:
        """Select optimal instance type based on scaling rule and workload."""
        # Simple heuristic - would be more sophisticated in production
        if rule.metric == ScalingMetric.CPU_UTILIZATION:
            return "large" if rule.threshold_up > 70 else "medium"
        elif rule.metric == ScalingMetric.QUEUE_LENGTH:
            return "xlarge" if rule.threshold_up > 20 else "large"
        elif rule.metric == ScalingMetric.RESPONSE_TIME:
            return "gpu" if rule.threshold_up > 10000 else "large"
        else:
            return "medium"  # Default
            
    def _select_instances_for_termination(self, count: int) -> List[ResourceInstance]:
        """Select instances for termination based on utilization."""
        running_instances = [i for i in self.instances.values() if i.status == "running"]
        
        # Sort by utilization (terminate least utilized first)
        def utilization_score(instance):
            cpu_util = instance.utilization.get("cpu", 0.0)
            memory_util = instance.utilization.get("memory", 0.0)
            return (cpu_util + memory_util) / 2
            
        running_instances.sort(key=utilization_score)
        
        return running_instances[:count]
        
    async def _terminate_instances(self, instances: List[ResourceInstance]):
        """Terminate specified instances."""
        for instance in instances:
            logger.info(f"Terminating instance {instance.instance_id}")
            
            instance.status = "terminating"
            instance.terminated_at = time.time()
            
            # If mesh coordinator is available, unregister the node
            if self.mesh_coordinator:
                try:
                    # This would integrate with actual mesh node removal
                    logger.info(f"Unregistering instance {instance.instance_id} from mesh")
                    # await self.mesh_coordinator.unregister_node(instance.instance_id)
                except Exception as e:
                    logger.error(f"Failed to unregister mesh node: {e}")
                    
            # Mark as terminated
            instance.status = "terminated"
            
    async def _metrics_collection_loop(self):
        """Collect metrics for scaling decisions."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
                
    async def _collect_system_metrics(self):
        """Collect system metrics from various sources."""
        current_time = time.time()
        
        # Simulate metric collection (would integrate with actual monitoring)
        metrics = {
            ScalingMetric.CPU_UTILIZATION: np.random.uniform(20, 90),
            ScalingMetric.MEMORY_UTILIZATION: np.random.uniform(30, 85),
            ScalingMetric.QUEUE_LENGTH: np.random.uniform(0, 25),
            ScalingMetric.RESPONSE_TIME: np.random.uniform(500, 8000),
            ScalingMetric.THROUGHPUT: np.random.uniform(10, 100),
            ScalingMetric.ERROR_RATE: np.random.uniform(0, 5)
        }
        
        # Add to history
        for metric, value in metrics.items():
            self.metrics_history[metric].append({
                "value": value,
                "timestamp": current_time
            })
            
    async def _prediction_loop(self):
        """Update prediction models periodically."""
        while self.running:
            try:
                if self.scaling_policy in [ScalingPolicy.PREDICTIVE, ScalingPolicy.HYBRID]:
                    await self._update_prediction_models()
                    
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(600)
                
    async def _update_prediction_models(self):
        """Update prediction models with latest data."""
        for metric in ScalingMetric:
            if metric == ScalingMetric.CUSTOM_METRIC:
                continue
                
            historical_data = list(self.metrics_history[metric])
            
            if len(historical_data) >= 50:  # Minimum data for modeling
                try:
                    model = await self._train_prediction_model(metric, historical_data)
                    self.prediction_models[metric] = model
                    
                except Exception as e:
                    logger.error(f"Failed to update prediction model for {metric.value}: {e}")
                    
    async def _train_prediction_model(self, 
                                    metric: ScalingMetric, 
                                    data: List[Dict]) -> Dict[str, Any]:
        """Train a prediction model for a specific metric."""
        # Simple moving average model (would use more sophisticated models in production)
        values = [d["value"] for d in data[-100:]]  # Last 100 data points
        
        # Calculate moving averages of different windows
        ma_5 = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
        ma_15 = np.mean(values[-15:]) if len(values) >= 15 else np.mean(values)
        ma_30 = np.mean(values[-30:]) if len(values) >= 30 else np.mean(values)
        
        # Simple trend calculation
        if len(values) >= 10:
            recent_values = values[-10:]
            old_values = values[-20:-10] if len(values) >= 20 else values[:-10]
            trend = np.mean(recent_values) - np.mean(old_values) if old_values else 0
        else:
            trend = 0
            
        return {
            "type": "moving_average",
            "ma_5": ma_5,
            "ma_15": ma_15,
            "ma_30": ma_30,
            "trend": trend,
            "last_updated": time.time()
        }
        
    async def _cost_optimization_loop(self):
        """Optimize costs through intelligent scaling decisions."""
        while self.running:
            try:
                if self.cost_optimization:
                    await self._optimize_costs()
                    
                await asyncio.sleep(600)  # Every 10 minutes
                
            except Exception as e:
                logger.error(f"Cost optimization error: {e}")
                await asyncio.sleep(1800)
                
    async def _optimize_costs(self):
        """Optimize resource costs while maintaining performance."""
        running_instances = [i for i in self.instances.values() if i.status == "running"]
        
        if not running_instances:
            return
            
        # Calculate current cost
        current_cost = sum(i.cost_per_hour for i in running_instances)
        
        # Analyze utilization patterns
        underutilized_instances = []
        for instance in running_instances:
            cpu_util = instance.utilization.get("cpu", 0.0)
            memory_util = instance.utilization.get("memory", 0.0)
            avg_util = (cpu_util + memory_util) / 2
            
            if avg_util < 20.0:  # Less than 20% utilized
                underutilized_instances.append(instance)
                
        # Consider instance type optimization
        if underutilized_instances:
            logger.info(
                f"Found {len(underutilized_instances)} underutilized instances "
                f"for cost optimization"
            )
            
            # Could implement instance type switching or consolidation here
            
        # Record cost information
        self.cost_history.append({
            "timestamp": time.time(),
            "hourly_cost": current_cost,
            "instance_count": len(running_instances),
            "cost_per_instance": current_cost / len(running_instances)
        })
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status and metrics."""
        running_instances = len([i for i in self.instances.values() if i.status == "running"])
        total_instances = len(self.instances)
        
        # Calculate recent costs
        recent_costs = list(self.cost_history)[-24:]  # Last 24 data points
        avg_hourly_cost = statistics.mean([c["hourly_cost"] for c in recent_costs]) if recent_costs else 0
        
        # Recent scaling events
        recent_events = [e for e in self.scaling_events if time.time() - e.timestamp < 3600]  # Last hour
        
        return {
            "running_instances": running_instances,
            "total_instances": total_instances,
            "instance_limits": {
                "min": self.min_instances,
                "max": self.max_instances
            },
            "scaling_policy": self.scaling_policy.value,
            "active_rules": len([r for r in self.scaling_rules if r.enabled]),
            "total_rules": len(self.scaling_rules),
            "recent_events": len(recent_events),
            "cost_optimization": self.cost_optimization,
            "average_hourly_cost": avg_hourly_cost,
            "current_metrics": {
                metric.value: list(self.metrics_history[metric])[-1] if self.metrics_history[metric] else None
                for metric in ScalingMetric if metric != ScalingMetric.CUSTOM_METRIC
            }
        }
        
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get detailed cost analysis."""
        if not self.cost_history:
            return {"error": "No cost data available"}
            
        recent_costs = list(self.cost_history)
        
        total_cost_24h = sum(c["hourly_cost"] for c in recent_costs[-24:])
        avg_cost_per_hour = statistics.mean([c["hourly_cost"] for c in recent_costs])
        
        # Cost by instance type
        instance_type_costs = {}
        for instance in self.instances.values():
            if instance.status == "running":
                instance_type = instance.instance_type
                if instance_type not in instance_type_costs:
                    instance_type_costs[instance_type] = {"count": 0, "cost": 0.0}
                instance_type_costs[instance_type]["count"] += 1
                instance_type_costs[instance_type]["cost"] += instance.cost_per_hour
                
        return {
            "total_cost_24h": total_cost_24h,
            "average_hourly_cost": avg_cost_per_hour,
            "cost_trend": "increasing" if len(recent_costs) > 1 and recent_costs[-1]["hourly_cost"] > recent_costs[-2]["hourly_cost"] else "stable",
            "instance_type_breakdown": instance_type_costs,
            "scaling_events_impact": sum(e.cost_impact for e in self.scaling_events[-10:]),  # Last 10 events
            "optimization_enabled": self.cost_optimization
        }
        
    # Additional methods would be implemented for:
    # - Advanced prediction models (ARIMA, LSTM, etc.)
    # - Seasonal pattern detection
    # - Multi-objective optimization
    # - Integration with cloud provider APIs
    # - Advanced cost optimization strategies