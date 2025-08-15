"""
Advanced Load Balancer for DNA Origami AutoEncoder

Provides intelligent load balancing across distributed DNA origami design
and fabrication nodes with advanced algorithms, health monitoring,
and performance optimization.
"""

import asyncio
import time
import json
import hashlib
import random
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict
import statistics
import heapq

from ..utils.logger import get_logger
from ..utils.performance import PerformanceOptimizer
from ..mesh.mesh_coordinator import MeshNode, NodeType

logger = get_logger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_AWARE = "resource_aware"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"
    AI_OPTIMIZED = "ai_optimized"


class HealthStatus(Enum):
    """Health status of backend nodes."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


@dataclass
class BackendNode:
    """Represents a backend node for load balancing."""
    node_id: str
    address: str
    port: int
    weight: float = 1.0
    
    # Health and performance metrics
    health_status: HealthStatus = HealthStatus.HEALTHY
    last_health_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    total_requests: int = 0
    active_connections: int = 0
    
    # Performance metrics
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_rate: float = 1.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Load balancing state
    last_selected: float = 0.0
    selection_count: int = 0
    
    def average_response_time(self) -> float:
        """Get average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
        
    def current_load_score(self) -> float:
        """Calculate current load score (lower is better)."""
        # Combine multiple factors into a single load score
        connection_factor = self.active_connections * 0.3
        cpu_factor = self.cpu_utilization * 0.3
        memory_factor = self.memory_utilization * 0.2
        response_factor = min(self.average_response_time() / 1000.0, 10.0) * 0.2
        
        return connection_factor + cpu_factor + memory_factor + response_factor


@dataclass
class LoadBalancingRequest:
    """Represents a load balancing request."""
    request_id: str
    client_id: str
    request_type: str
    payload_size: int
    priority: int = 1
    session_id: Optional[str] = None
    requirements: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass 
class RequestResult:
    """Result of a load balanced request."""
    request_id: str
    node_id: str
    success: bool
    response_time: float
    error: Optional[str] = None
    payload_size: int = 0


class LoadBalancer:
    """
    Advanced load balancer for DNA origami distributed systems.
    
    Features:
    - Multiple load balancing algorithms
    - Intelligent health monitoring
    - Session affinity support
    - Circuit breaker integration  
    - Performance-based routing
    - Adaptive algorithm selection
    - Real-time metrics and monitoring
    """
    
    def __init__(self,
                 algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ADAPTIVE,
                 health_check_interval: float = 30.0,
                 max_failures: int = 3,
                 enable_session_affinity: bool = True,
                 adaptive_learning: bool = True):
        
        self.algorithm = algorithm
        self.health_check_interval = health_check_interval
        self.max_failures = max_failures
        self.enable_session_affinity = enable_session_affinity
        self.adaptive_learning = adaptive_learning
        
        # Backend nodes
        self.backend_nodes: Dict[str, BackendNode] = {}
        self.healthy_nodes: List[BackendNode] = []
        
        # Load balancing state
        self.round_robin_index = 0
        self.session_affinity: Dict[str, str] = {}  # session_id -> node_id
        self.consistent_hash_ring: List[Tuple[int, str]] = []
        
        # Request tracking
        self.active_requests: Dict[str, LoadBalancingRequest] = {}
        self.request_history: deque = deque(maxlen=10000)
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Algorithm performance tracking for adaptive selection
        self.algorithm_performance: Dict[LoadBalancingAlgorithm, Dict[str, float]] = {}
        self.current_algorithm_start_time = time.time()
        
        # Threading and locks
        self.running = False
        self._lock = threading.Lock()
        
        # Performance optimization
        self.performance_optimizer = PerformanceOptimizer()
        
        self._initialize_algorithm_tracking()
        
    def _initialize_algorithm_tracking(self):
        """Initialize algorithm performance tracking."""
        for algo in LoadBalancingAlgorithm:
            self.algorithm_performance[algo] = {
                "avg_response_time": 0.0,
                "success_rate": 1.0,
                "throughput": 0.0,
                "last_evaluation": time.time()
            }
            
    async def start(self):
        """Start the load balancer."""
        self.running = True
        logger.info("Starting load balancer")
        
        # Start health monitoring
        health_task = asyncio.create_task(self._health_monitoring_loop())
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        adaptive_task = asyncio.create_task(self._adaptive_algorithm_loop())
        
        await asyncio.gather(health_task, metrics_task, adaptive_task)
        
    async def stop(self):
        """Stop the load balancer."""
        self.running = False
        logger.info("Stopping load balancer")
        
    def add_backend_node(self, 
                        node_id: str,
                        address: str, 
                        port: int,
                        weight: float = 1.0,
                        mesh_node: Optional[MeshNode] = None):
        """Add a backend node to the load balancer."""
        node = BackendNode(
            node_id=node_id,
            address=address,
            port=port,
            weight=weight
        )
        
        # If mesh node is provided, copy relevant metrics
        if mesh_node:
            node.cpu_utilization = mesh_node.cpu_usage
            node.memory_utilization = mesh_node.memory_usage
            
        with self._lock:
            self.backend_nodes[node_id] = node
            self._update_healthy_nodes()
            self._update_consistent_hash_ring()
            
        logger.info(f"Added backend node: {node_id} at {address}:{port}")
        
    def remove_backend_node(self, node_id: str) -> bool:
        """Remove a backend node from the load balancer."""
        with self._lock:
            if node_id in self.backend_nodes:
                del self.backend_nodes[node_id]
                self._update_healthy_nodes()
                self._update_consistent_hash_ring()
                
                # Clear session affinity for this node
                sessions_to_clear = [
                    session for session, nid in self.session_affinity.items()
                    if nid == node_id
                ]
                for session in sessions_to_clear:
                    del self.session_affinity[session]
                    
                logger.info(f"Removed backend node: {node_id}")
                return True
                
        return False
        
    def _update_healthy_nodes(self):
        """Update the list of healthy nodes."""
        self.healthy_nodes = [
            node for node in self.backend_nodes.values()
            if node.health_status == HealthStatus.HEALTHY
        ]
        
    def _update_consistent_hash_ring(self):
        """Update the consistent hash ring for consistent hashing algorithm."""
        self.consistent_hash_ring.clear()
        
        for node in self.backend_nodes.values():
            # Add multiple points on the ring for better distribution
            for i in range(int(node.weight * 100)):
                hash_key = hashlib.md5(f"{node.node_id}:{i}".encode()).hexdigest()
                hash_value = int(hash_key, 16)
                self.consistent_hash_ring.append((hash_value, node.node_id))
                
        self.consistent_hash_ring.sort()
        
    async def route_request(self, request: LoadBalancingRequest) -> Optional[BackendNode]:
        """Route a request to an appropriate backend node."""
        with self._lock:
            if not self.healthy_nodes:
                logger.error("No healthy backend nodes available")
                return None
                
            # Check session affinity first
            if (self.enable_session_affinity and 
                request.session_id and 
                request.session_id in self.session_affinity):
                
                node_id = self.session_affinity[request.session_id]
                if node_id in self.backend_nodes:
                    node = self.backend_nodes[node_id]
                    if node.health_status == HealthStatus.HEALTHY:
                        return node
                    else:
                        # Remove stale session affinity
                        del self.session_affinity[request.session_id]
                        
            # Select node based on algorithm
            if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                node = self._round_robin_selection()
            elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                node = self._weighted_round_robin_selection()
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                node = self._least_connections_selection()
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                node = self._least_response_time_selection()
            elif self.algorithm == LoadBalancingAlgorithm.RESOURCE_AWARE:
                node = self._resource_aware_selection(request)
            elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
                node = self._consistent_hash_selection(request)
            elif self.algorithm == LoadBalancingAlgorithm.ADAPTIVE:
                node = self._adaptive_selection(request)
            elif self.algorithm == LoadBalancingAlgorithm.AI_OPTIMIZED:
                node = await self._ai_optimized_selection(request)
            else:
                node = self._round_robin_selection()  # Fallback
                
            if node:
                # Update selection tracking
                node.last_selected = time.time()
                node.selection_count += 1
                node.active_connections += 1
                
                # Set session affinity if enabled
                if self.enable_session_affinity and request.session_id:
                    self.session_affinity[request.session_id] = node.node_id
                    
                # Track active request
                self.active_requests[request.request_id] = request
                
            return node
            
    def _round_robin_selection(self) -> Optional[BackendNode]:
        """Round robin node selection."""
        if not self.healthy_nodes:
            return None
            
        node = self.healthy_nodes[self.round_robin_index]
        self.round_robin_index = (self.round_robin_index + 1) % len(self.healthy_nodes)
        return node
        
    def _weighted_round_robin_selection(self) -> Optional[BackendNode]:
        """Weighted round robin node selection."""
        if not self.healthy_nodes:
            return None
            
        # Calculate total weight
        total_weight = sum(node.weight for node in self.healthy_nodes)
        
        # Select based on weights
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for node in self.healthy_nodes:
            current_weight += node.weight
            if rand_val <= current_weight:
                return node
                
        return self.healthy_nodes[-1]  # Fallback
        
    def _least_connections_selection(self) -> Optional[BackendNode]:
        """Least connections node selection."""
        if not self.healthy_nodes:
            return None
            
        return min(self.healthy_nodes, key=lambda n: n.active_connections)
        
    def _least_response_time_selection(self) -> Optional[BackendNode]:
        """Least response time node selection."""
        if not self.healthy_nodes:
            return None
            
        return min(self.healthy_nodes, key=lambda n: n.average_response_time())
        
    def _resource_aware_selection(self, request: LoadBalancingRequest) -> Optional[BackendNode]:
        """Resource-aware node selection based on current load."""
        if not self.healthy_nodes:
            return None
            
        # Calculate load scores for all healthy nodes
        scored_nodes = []
        for node in self.healthy_nodes:
            load_score = node.current_load_score()
            
            # Adjust score based on request requirements
            if request.requirements:
                if request.requirements.get("cpu_intensive", False):
                    load_score += node.cpu_utilization * 0.5
                if request.requirements.get("memory_intensive", False):
                    load_score += node.memory_utilization * 0.5
                    
            scored_nodes.append((load_score, node))
            
        # Select node with lowest load score
        scored_nodes.sort(key=lambda x: x[0])
        return scored_nodes[0][1]
        
    def _consistent_hash_selection(self, request: LoadBalancingRequest) -> Optional[BackendNode]:
        """Consistent hash node selection."""
        if not self.consistent_hash_ring:
            return self._round_robin_selection()
            
        # Create hash of request identifier
        hash_key = request.client_id + request.request_type
        hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
        
        # Find the first node on the ring with hash >= request hash
        for ring_hash, node_id in self.consistent_hash_ring:
            if ring_hash >= hash_value:
                if node_id in self.backend_nodes:
                    node = self.backend_nodes[node_id]
                    if node.health_status == HealthStatus.HEALTHY:
                        return node
                        
        # Wrap around to first node
        if self.consistent_hash_ring:
            node_id = self.consistent_hash_ring[0][1]
            if node_id in self.backend_nodes:
                node = self.backend_nodes[node_id]
                if node.health_status == HealthStatus.HEALTHY:
                    return node
                    
        return self._round_robin_selection()  # Fallback
        
    def _adaptive_selection(self, request: LoadBalancingRequest) -> Optional[BackendNode]:
        """Adaptive node selection that learns from performance."""
        if not self.healthy_nodes:
            return None
            
        # Use different strategies based on request type and current performance
        current_time = time.time()
        
        # For high-priority requests, use least response time
        if request.priority >= 3:
            return self._least_response_time_selection()
            
        # For CPU-intensive requests, use resource-aware
        if request.requirements.get("cpu_intensive", False):
            return self._resource_aware_selection(request)
            
        # Default to least connections for balanced load
        return self._least_connections_selection()
        
    async def _ai_optimized_selection(self, request: LoadBalancingRequest) -> Optional[BackendNode]:
        """AI-optimized node selection using machine learning."""
        if not self.healthy_nodes:
            return None
            
        # This would implement actual ML-based selection
        # For now, use a combination of multiple factors
        
        scored_nodes = []
        for node in self.healthy_nodes:
            # Feature vector for ML model
            features = [
                node.active_connections / 100.0,  # Normalized connections
                node.cpu_utilization,
                node.memory_utilization, 
                node.average_response_time() / 1000.0,  # Normalized response time
                node.success_rate,
                1.0 / (time.time() - node.last_selected + 1),  # Recency factor
            ]
            
            # Simple scoring function (would be replaced by trained model)
            score = sum(f * w for f, w in zip(features, [0.2, 0.2, 0.2, 0.3, 0.1, 0.0]))
            scored_nodes.append((score, node))
            
        # Select node with lowest score (best predicted performance)
        scored_nodes.sort(key=lambda x: x[0])
        return scored_nodes[0][1]
        
    async def record_request_result(self, result: RequestResult):
        """Record the result of a load balanced request."""
        with self._lock:
            # Remove from active requests
            if result.request_id in self.active_requests:
                request = self.active_requests[result.request_id]
                del self.active_requests[result.request_id]
                
                # Update node metrics
                if result.node_id in self.backend_nodes:
                    node = self.backend_nodes[result.node_id]
                    node.active_connections = max(0, node.active_connections - 1)
                    node.total_requests += 1
                    node.response_times.append(result.response_time)
                    
                    # Update success rate
                    if result.success:
                        node.success_rate = 0.95 * node.success_rate + 0.05 * 1.0
                        node.consecutive_failures = 0
                    else:
                        node.success_rate = 0.95 * node.success_rate + 0.05 * 0.0
                        node.consecutive_failures += 1
                        
                        # Check if node should be marked unhealthy
                        if node.consecutive_failures >= self.max_failures:
                            node.health_status = HealthStatus.UNHEALTHY
                            self._update_healthy_nodes()
                            logger.warning(f"Node {result.node_id} marked as unhealthy")
                            
                # Add to request history
                self.request_history.append({
                    "request_id": result.request_id,
                    "node_id": result.node_id,
                    "success": result.success,
                    "response_time": result.response_time,
                    "algorithm": self.algorithm.value,
                    "timestamp": time.time()
                })
                
                # Update performance metrics
                self.performance_metrics["response_time"].append(result.response_time)
                self.performance_metrics["success_rate"].append(1.0 if result.success else 0.0)
                
    async def _health_monitoring_loop(self):
        """Monitor health of backend nodes."""
        while self.running:
            try:
                await self._check_node_health()
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _check_node_health(self):
        """Check health of all backend nodes."""
        for node in self.backend_nodes.values():
            try:
                # Perform health check (simplified implementation)
                is_healthy = await self._perform_health_check(node)
                
                if is_healthy:
                    if node.health_status == HealthStatus.UNHEALTHY:
                        logger.info(f"Node {node.node_id} recovered")
                        node.health_status = HealthStatus.HEALTHY
                        node.consecutive_failures = 0
                        self._update_healthy_nodes()
                        
                    node.last_health_check = time.time()
                    
                else:
                    node.consecutive_failures += 1
                    
                    if node.consecutive_failures >= self.max_failures:
                        if node.health_status != HealthStatus.UNHEALTHY:
                            logger.warning(f"Node {node.node_id} failed health check")
                            node.health_status = HealthStatus.UNHEALTHY
                            self._update_healthy_nodes()
                            
            except Exception as e:
                logger.error(f"Health check failed for node {node.node_id}: {e}")
                node.consecutive_failures += 1
                
    async def _perform_health_check(self, node: BackendNode) -> bool:
        """Perform health check on a specific node."""
        # Simplified health check - would implement actual HTTP/TCP checks
        # For now, simulate based on node metrics
        
        # Consider node unhealthy if overloaded
        if (node.cpu_utilization > 95 or 
            node.memory_utilization > 95 or
            node.average_response_time() > 30000):  # 30 seconds
            return False
            
        return True
        
    async def _adaptive_algorithm_loop(self):
        """Adaptive algorithm selection loop."""
        if not self.adaptive_learning:
            return
            
        while self.running:
            try:
                await self._evaluate_algorithm_performance()
                await asyncio.sleep(300)  # Evaluate every 5 minutes
                
            except Exception as e:
                logger.error(f"Adaptive algorithm error: {e}")
                await asyncio.sleep(600)
                
    async def _evaluate_algorithm_performance(self):
        """Evaluate current algorithm performance and switch if needed."""
        if self.algorithm != LoadBalancingAlgorithm.ADAPTIVE:
            return
            
        current_time = time.time()
        evaluation_window = 300  # 5 minutes
        
        # Get recent performance metrics
        recent_requests = [
            req for req in self.request_history
            if current_time - req["timestamp"] < evaluation_window
        ]
        
        if len(recent_requests) < 10:  # Need minimum requests for evaluation
            return
            
        # Calculate current performance
        avg_response_time = statistics.mean([req["response_time"] for req in recent_requests])
        success_rate = statistics.mean([req["success"] for req in recent_requests])
        throughput = len(recent_requests) / evaluation_window * 60  # requests per minute
        
        # Update current algorithm performance
        current_algo = LoadBalancingAlgorithm(recent_requests[0]["algorithm"])
        self.algorithm_performance[current_algo] = {
            "avg_response_time": avg_response_time,
            "success_rate": success_rate,
            "throughput": throughput,
            "last_evaluation": current_time
        }
        
        # Find best performing algorithm
        best_algorithm = None
        best_score = -1
        
        for algo, perf in self.algorithm_performance.items():
            if algo == LoadBalancingAlgorithm.ADAPTIVE:
                continue
                
            # Calculate composite performance score
            score = (
                perf["success_rate"] * 0.4 +
                (1.0 - min(perf["avg_response_time"] / 10000.0, 1.0)) * 0.3 +
                min(perf["throughput"] / 100.0, 1.0) * 0.3
            )
            
            if score > best_score:
                best_score = score
                best_algorithm = algo
                
        # Switch to best algorithm if significantly better
        if (best_algorithm and 
            best_algorithm != current_algo and
            best_score > self.algorithm_performance[current_algo]["success_rate"] * 1.1):
            
            logger.info(f"Switching from {current_algo.value} to {best_algorithm.value}")
            self.algorithm = best_algorithm
            
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get current load balancer status."""
        with self._lock:
            total_nodes = len(self.backend_nodes)
            healthy_nodes = len(self.healthy_nodes)
            active_requests = len(self.active_requests)
            
            # Calculate aggregate metrics
            total_connections = sum(node.active_connections for node in self.backend_nodes.values())
            avg_response_time = statistics.mean([
                node.average_response_time() for node in self.healthy_nodes
            ]) if self.healthy_nodes else 0
            
            return {
                "algorithm": self.algorithm.value,
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "unhealthy_nodes": total_nodes - healthy_nodes,
                "active_requests": active_requests,
                "total_connections": total_connections,
                "session_affinity_enabled": self.enable_session_affinity,
                "active_sessions": len(self.session_affinity),
                "average_response_time": avg_response_time,
                "recent_requests": len(self.request_history),
                "adaptive_learning": self.adaptive_learning
            }
            
    def get_node_metrics(self) -> Dict[str, Any]:
        """Get metrics for all backend nodes."""
        with self._lock:
            node_metrics = {}
            
            for node_id, node in self.backend_nodes.items():
                node_metrics[node_id] = {
                    "health_status": node.health_status.value,
                    "active_connections": node.active_connections,
                    "total_requests": node.total_requests,
                    "selection_count": node.selection_count,
                    "success_rate": node.success_rate,
                    "average_response_time": node.average_response_time(),
                    "cpu_utilization": node.cpu_utilization,
                    "memory_utilization": node.memory_utilization,
                    "weight": node.weight,
                    "last_selected": node.last_selected,
                    "consecutive_failures": node.consecutive_failures
                }
                
            return node_metrics
            
    # Additional methods would be implemented for:
    # - Advanced ML-based node selection
    # - Dynamic weight adjustment
    # - Predictive load balancing
    # - Integration with service discovery
    # - Advanced health check strategies