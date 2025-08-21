"""
Edge Computing Framework for DNA Origami Processing
Implements distributed edge processing for real-time DNA origami computation.
"""

import asyncio
import aiohttp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
import uuid
from pathlib import Path
import socket
import psutil

from ..models.dna_sequence import DNASequence
from ..models.image_data import ImageData
from ..utils.logger import get_logger
from ..utils.performance_optimized import PerformanceTracker
from .quantum_enhanced_encoder import quantum_enhanced_pipeline, QuantumEncodingConfig

logger = get_logger(__name__)

@dataclass
class EdgeNode:
    """Represents an edge computing node in the network."""
    node_id: str
    ip_address: str
    port: int
    capabilities: Dict[str, Any]
    current_load: float = 0.0
    last_heartbeat: Optional[datetime] = None
    is_active: bool = True
    
    # Performance metrics
    average_latency: float = 0.0
    success_rate: float = 1.0
    processing_capacity: int = 10  # concurrent tasks
    
    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()
    
    @property
    def endpoint(self) -> str:
        return f"http://{self.ip_address}:{self.port}"
    
    def is_healthy(self, timeout_seconds: int = 30) -> bool:
        """Check if node is healthy based on last heartbeat."""
        if not self.is_active:
            return False
        
        time_since_heartbeat = datetime.now() - self.last_heartbeat
        return time_since_heartbeat.total_seconds() < timeout_seconds

@dataclass
class ComputationTask:
    """Represents a computation task for edge processing."""
    task_id: str
    task_type: str
    input_data: Any
    priority: int = 1  # 1 = highest priority
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    estimated_compute_time: float = 1.0  # seconds
    required_capabilities: List[str] = field(default_factory=list)
    
    # Task state
    status: str = "pending"  # pending, assigned, running, completed, failed
    assigned_node: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if task has exceeded its deadline."""
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline

class EdgeComputingOrchestrator:
    """
    Orchestrates distributed edge computing for DNA origami processing.
    Manages load balancing, fault tolerance, and optimization.
    """
    
    def __init__(self, coordinator_port: int = 8080):
        self.coordinator_port = coordinator_port
        self.logger = get_logger(f"{__name__}.EdgeComputingOrchestrator")
        self.performance_tracker = PerformanceTracker()
        
        # Node management
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.node_health_check_interval = 10  # seconds
        
        # Task management
        self.pending_tasks: List[ComputationTask] = []
        self.active_tasks: Dict[str, ComputationTask] = {}
        self.completed_tasks: Dict[str, ComputationTask] = {}
        
        # Load balancing
        self.load_balancing_strategy = "least_loaded"  # least_loaded, round_robin, performance_based
        self.max_retries = 3
        
        # Performance monitoring
        self.performance_metrics = {
            'total_tasks_processed': 0,
            'average_processing_time': 0.0,
            'success_rate': 1.0,
            'active_nodes': 0
        }
        
        self.logger.info("Edge computing orchestrator initialized")
    
    async def start_coordinator(self):
        """Start the edge computing coordinator service."""
        self.logger.info(f"Starting edge coordinator on port {self.coordinator_port}")
        
        # Start background tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._task_scheduler_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        
        # Start HTTP server for node communication
        app = self._create_web_app()
        from aiohttp import web
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.coordinator_port)
        await site.start()
        
        self.logger.info("Edge coordinator started successfully")
    
    def _create_web_app(self):
        """Create web application for edge node communication."""
        from aiohttp import web
        
        app = web.Application()
        
        # Node registration and health endpoints
        app.router.add_post('/register', self._handle_node_registration)
        app.router.add_post('/heartbeat', self._handle_heartbeat)
        app.router.add_post('/task_complete', self._handle_task_completion)
        
        # Task management endpoints
        app.router.add_post('/submit_task', self._handle_task_submission)
        app.router.add_get('/task_status/{task_id}', self._handle_task_status)
        
        # Monitoring endpoints
        app.router.add_get('/metrics', self._handle_metrics_request)
        app.router.add_get('/nodes', self._handle_nodes_status)
        
        return app
    
    async def _handle_node_registration(self, request):
        """Handle edge node registration."""
        data = await request.json()
        
        node = EdgeNode(
            node_id=data['node_id'],
            ip_address=data['ip_address'],
            port=data['port'],
            capabilities=data.get('capabilities', {}),
            processing_capacity=data.get('processing_capacity', 10)
        )
        
        self.edge_nodes[node.node_id] = node
        self.logger.info(f"Registered edge node: {node.node_id} at {node.endpoint}")
        
        return web.json_response({'status': 'registered', 'node_id': node.node_id})
    
    async def _handle_heartbeat(self, request):
        """Handle heartbeat from edge nodes."""
        data = await request.json()
        node_id = data['node_id']
        
        if node_id in self.edge_nodes:
            node = self.edge_nodes[node_id]
            node.last_heartbeat = datetime.now()
            node.current_load = data.get('current_load', 0.0)
            node.is_active = True
            
            return web.json_response({'status': 'acknowledged'})
        else:
            return web.json_response({'status': 'unknown_node'}, status=404)
    
    async def _handle_task_completion(self, request):
        """Handle task completion notification from edge nodes."""
        data = await request.json()
        task_id = data['task_id']
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            if data.get('success', False):
                task.status = "completed"
                task.result = data.get('result')
            else:
                task.status = "failed"
                task.error = data.get('error', 'Unknown error')
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            
            # Update node metrics
            if task.assigned_node in self.edge_nodes:
                node = self.edge_nodes[task.assigned_node]
                processing_time = (datetime.now() - task.created_at).total_seconds()
                self._update_node_metrics(node, processing_time, data.get('success', False))
            
            self.logger.info(f"Task {task_id} completed with status: {task.status}")
            return web.json_response({'status': 'acknowledged'})
        else:
            return web.json_response({'status': 'unknown_task'}, status=404)
    
    async def _handle_task_submission(self, request):
        """Handle new task submission."""
        data = await request.json()
        
        task = ComputationTask(
            task_id=str(uuid.uuid4()),
            task_type=data['task_type'],
            input_data=data['input_data'],
            priority=data.get('priority', 1),
            estimated_compute_time=data.get('estimated_compute_time', 1.0),
            required_capabilities=data.get('required_capabilities', [])
        )
        
        # Set deadline if provided
        if 'deadline_seconds' in data:
            task.deadline = datetime.now() + timedelta(seconds=data['deadline_seconds'])
        
        self.pending_tasks.append(task)
        self.logger.info(f"Task submitted: {task.task_id} (type: {task.task_type})")
        
        return web.json_response({'task_id': task.task_id, 'status': 'submitted'})
    
    async def _handle_task_status(self, request):
        """Handle task status query."""
        task_id = request.match_info['task_id']
        
        # Check all task collections
        for task_dict in [self.pending_tasks, self.active_tasks, self.completed_tasks]:
            if isinstance(task_dict, list):
                # pending_tasks is a list
                for task in task_dict:
                    if task.task_id == task_id:
                        return web.json_response(self._task_to_dict(task))
            else:
                # active_tasks and completed_tasks are dicts
                if task_id in task_dict:
                    return web.json_response(self._task_to_dict(task_dict[task_id]))
        
        return web.json_response({'error': 'Task not found'}, status=404)
    
    async def _handle_metrics_request(self, request):
        """Handle metrics request."""
        return web.json_response(self.performance_metrics)
    
    async def _handle_nodes_status(self, request):
        """Handle nodes status request."""
        nodes_status = {}
        for node_id, node in self.edge_nodes.items():
            nodes_status[node_id] = {
                'endpoint': node.endpoint,
                'is_healthy': node.is_healthy(),
                'current_load': node.current_load,
                'capabilities': node.capabilities,
                'average_latency': node.average_latency,
                'success_rate': node.success_rate
            }
        
        return web.json_response(nodes_status)
    
    def _task_to_dict(self, task: ComputationTask) -> Dict[str, Any]:
        """Convert task to dictionary for JSON response."""
        return {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'status': task.status,
            'priority': task.priority,
            'created_at': task.created_at.isoformat(),
            'assigned_node': task.assigned_node,
            'error': task.error
        }
    
    async def _health_check_loop(self):
        """Background loop for health checking edge nodes."""
        while True:
            try:
                await self._check_node_health()
                await asyncio.sleep(self.node_health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    async def _check_node_health(self):
        """Check health of all edge nodes."""
        unhealthy_nodes = []
        
        for node_id, node in self.edge_nodes.items():
            if not node.is_healthy():
                unhealthy_nodes.append(node_id)
                node.is_active = False
                
                # Reassign tasks from unhealthy nodes
                await self._reassign_tasks_from_node(node_id)
        
        if unhealthy_nodes:
            self.logger.warning(f"Unhealthy nodes detected: {unhealthy_nodes}")
        
        # Update active nodes count
        self.performance_metrics['active_nodes'] = sum(
            1 for node in self.edge_nodes.values() if node.is_active
        )
    
    async def _reassign_tasks_from_node(self, failed_node_id: str):
        """Reassign tasks from a failed node."""
        tasks_to_reassign = []
        
        for task_id, task in self.active_tasks.items():
            if task.assigned_node == failed_node_id:
                tasks_to_reassign.append(task)
        
        for task in tasks_to_reassign:
            # Move back to pending queue
            task.status = "pending"
            task.assigned_node = None
            self.pending_tasks.append(task)
            del self.active_tasks[task.task_id]
            
            self.logger.info(f"Reassigned task {task.task_id} due to node failure")
    
    async def _task_scheduler_loop(self):
        """Background loop for task scheduling."""
        while True:
            try:
                await self._schedule_pending_tasks()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                self.logger.error(f"Task scheduling error: {e}")
                await asyncio.sleep(5)
    
    async def _schedule_pending_tasks(self):
        """Schedule pending tasks to available edge nodes."""
        if not self.pending_tasks:
            return
        
        # Sort tasks by priority and deadline
        self.pending_tasks.sort(key=lambda t: (t.priority, t.created_at))
        
        # Remove expired tasks
        self.pending_tasks = [t for t in self.pending_tasks if not t.is_expired()]
        
        # Find available nodes
        available_nodes = [
            node for node in self.edge_nodes.values()
            if node.is_active and node.current_load < 0.8
        ]
        
        if not available_nodes:
            return
        
        # Schedule tasks
        tasks_to_schedule = []
        for task in self.pending_tasks[:]:
            if not task.required_capabilities or self._node_has_capabilities(available_nodes, task.required_capabilities):
                # Find best node for this task
                best_node = self._select_best_node(available_nodes, task)
                if best_node:
                    task.assigned_node = best_node.node_id
                    task.status = "assigned"
                    tasks_to_schedule.append(task)
                    
                    # Remove from pending
                    self.pending_tasks.remove(task)
                    
                    # Add to active
                    self.active_tasks[task.task_id] = task
                    
                    # Update node load
                    best_node.current_load += 0.1  # Estimate load increase
        
        # Send tasks to nodes
        for task in tasks_to_schedule:
            await self._send_task_to_node(task)
    
    def _node_has_capabilities(self, nodes: List[EdgeNode], required_capabilities: List[str]) -> bool:
        """Check if any node has the required capabilities."""
        for node in nodes:
            if all(cap in node.capabilities for cap in required_capabilities):
                return True
        return False
    
    def _select_best_node(self, available_nodes: List[EdgeNode], task: ComputationTask) -> Optional[EdgeNode]:
        """Select the best node for a task based on load balancing strategy."""
        # Filter nodes by capabilities
        suitable_nodes = [
            node for node in available_nodes
            if not task.required_capabilities or 
            all(cap in node.capabilities for cap in task.required_capabilities)
        ]
        
        if not suitable_nodes:
            return None
        
        if self.load_balancing_strategy == "least_loaded":
            return min(suitable_nodes, key=lambda n: n.current_load)
        elif self.load_balancing_strategy == "performance_based":
            # Select based on combined performance metrics
            return min(suitable_nodes, key=lambda n: n.average_latency * (1 - n.success_rate))
        else:  # round_robin
            return suitable_nodes[len(self.active_tasks) % len(suitable_nodes)]
    
    async def _send_task_to_node(self, task: ComputationTask):
        """Send task to assigned edge node."""
        node = self.edge_nodes[task.assigned_node]
        
        payload = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'input_data': task.input_data,
            'estimated_compute_time': task.estimated_compute_time
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{node.endpoint}/execute_task",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        task.status = "running"
                        self.logger.info(f"Task {task.task_id} sent to node {node.node_id}")
                    else:
                        # Task assignment failed
                        task.status = "pending"
                        task.assigned_node = None
                        self.pending_tasks.append(task)
                        del self.active_tasks[task.task_id]
                        
        except Exception as e:
            self.logger.error(f"Failed to send task {task.task_id} to node {node.node_id}: {e}")
            
            # Return task to pending queue
            task.status = "pending"
            task.assigned_node = None
            self.pending_tasks.append(task)
            del self.active_tasks[task.task_id]
    
    def _update_node_metrics(self, node: EdgeNode, processing_time: float, success: bool):
        """Update performance metrics for a node."""
        # Update average latency with exponential moving average
        alpha = 0.1
        node.average_latency = alpha * processing_time + (1 - alpha) * node.average_latency
        
        # Update success rate
        node.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * node.success_rate
        
        # Reduce current load
        node.current_load = max(0.0, node.current_load - 0.1)
    
    async def _performance_monitoring_loop(self):
        """Background loop for performance monitoring."""
        while True:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _update_performance_metrics(self):
        """Update overall system performance metrics."""
        total_completed = len(self.completed_tasks)
        successful_tasks = sum(
            1 for task in self.completed_tasks.values() 
            if task.status == "completed"
        )
        
        self.performance_metrics.update({
            'total_tasks_processed': total_completed,
            'success_rate': successful_tasks / total_completed if total_completed > 0 else 1.0,
            'pending_tasks': len(self.pending_tasks),
            'active_tasks': len(self.active_tasks),
            'active_nodes': sum(1 for node in self.edge_nodes.values() if node.is_active)
        })
        
        # Calculate average processing time
        if self.completed_tasks:
            processing_times = [
                (task.created_at - datetime.now()).total_seconds()
                for task in self.completed_tasks.values()
                if task.status == "completed"
            ]
            if processing_times:
                self.performance_metrics['average_processing_time'] = np.mean(processing_times)

class EdgeWorkerNode:
    """
    Edge worker node that processes DNA origami computation tasks.
    """
    
    def __init__(self, node_id: str, coordinator_url: str, worker_port: int = 8081):
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.worker_port = worker_port
        self.logger = get_logger(f"{__name__}.EdgeWorkerNode.{node_id}")
        
        # Node capabilities
        self.capabilities = {
            'quantum_encoding': True,
            'dna_folding_simulation': True,
            'error_correction': True,
            'gpu_acceleration': self._detect_gpu(),
            'max_image_size': 512  # pixels
        }
        
        # Performance tracking
        self.current_load = 0.0
        self.active_tasks = {}
        self.performance_tracker = PerformanceTracker()
        
        self.logger.info(f"Edge worker node {node_id} initialized")
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU acceleration is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def start_worker(self):
        """Start the edge worker node."""
        self.logger.info(f"Starting edge worker on port {self.worker_port}")
        
        # Register with coordinator
        await self._register_with_coordinator()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._resource_monitoring_loop())
        
        # Start HTTP server
        app = self._create_worker_app()
        from aiohttp import web
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.worker_port)
        await site.start()
        
        self.logger.info("Edge worker started successfully")
    
    def _create_worker_app(self):
        """Create web application for worker node."""
        from aiohttp import web
        
        app = web.Application()
        
        # Task execution endpoint
        app.router.add_post('/execute_task', self._handle_task_execution)
        
        # Health and status endpoints
        app.router.add_get('/health', self._handle_health_check)
        app.router.add_get('/status', self._handle_status_request)
        
        return app
    
    async def _register_with_coordinator(self):
        """Register this worker node with the coordinator."""
        registration_data = {
            'node_id': self.node_id,
            'ip_address': self._get_local_ip(),
            'port': self.worker_port,
            'capabilities': self.capabilities,
            'processing_capacity': psutil.cpu_count()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.coordinator_url}/register",
                    json=registration_data
                ) as response:
                    if response.status == 200:
                        self.logger.info("Successfully registered with coordinator")
                    else:
                        self.logger.error(f"Registration failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to register with coordinator: {e}")
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to a remote server to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to coordinator."""
        while True:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _send_heartbeat(self):
        """Send heartbeat to coordinator."""
        heartbeat_data = {
            'node_id': self.node_id,
            'current_load': self.current_load,
            'active_tasks': len(self.active_tasks),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.coordinator_url}/heartbeat",
                    json=heartbeat_data,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"Heartbeat failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat: {e}")
    
    async def _resource_monitoring_loop(self):
        """Monitor resource usage and update current load."""
        while True:
            try:
                # Calculate current load based on CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Combine metrics (CPU has more weight)
                self.current_load = (0.7 * cpu_percent + 0.3 * memory_percent) / 100.0
                
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _handle_task_execution(self, request):
        """Handle task execution request."""
        try:
            task_data = await request.json()
            task_id = task_data['task_id']
            
            self.logger.info(f"Executing task: {task_id}")
            
            # Add to active tasks
            self.active_tasks[task_id] = {
                'start_time': datetime.now(),
                'task_type': task_data['task_type']
            }
            
            # Execute task based on type
            result = await self._execute_task(task_data)
            
            # Remove from active tasks
            del self.active_tasks[task_id]
            
            # Notify coordinator of completion
            await self._notify_task_completion(task_id, result, success=True)
            
            return web.json_response({'status': 'accepted'})
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            
            # Clean up and notify failure
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            await self._notify_task_completion(task_id, None, success=False, error=str(e))
            
            return web.json_response({'status': 'failed', 'error': str(e)}, status=500)
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Any:
        """Execute a computation task."""
        task_type = task_data['task_type']
        input_data = task_data['input_data']
        
        if task_type == 'quantum_dna_encoding':
            return await self._execute_quantum_encoding_task(input_data)
        elif task_type == 'dna_folding_simulation':
            return await self._execute_folding_simulation_task(input_data)
        elif task_type == 'error_correction':
            return await self._execute_error_correction_task(input_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _execute_quantum_encoding_task(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantum DNA encoding task."""
        # Create image data from input
        image_array = np.array(input_data['image_data'])
        image_data = ImageData.from_array(image_array, name="edge_processing_image")
        
        # Configure quantum encoding
        config = QuantumEncodingConfig()
        if 'config' in input_data:
            for key, value in input_data['config'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Execute quantum encoding pipeline
        result = quantum_enhanced_pipeline(image_data, config)
        
        # Convert result to serializable format
        serializable_result = {
            'quantum_metrics': result['quantum_encoding']['quantum_metrics'],
            'num_sequences': len(result['quantum_encoding']['dna_sequences']),
            'folding_metrics': result['quantum_folding']['global_metrics'],
            'processing_node': self.node_id
        }
        
        return serializable_result
    
    async def _execute_folding_simulation_task(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DNA folding simulation task."""
        # Placeholder for folding simulation
        # In a real implementation, this would use molecular dynamics
        
        dna_sequence = input_data['dna_sequence']
        
        # Simulate processing time
        await asyncio.sleep(1)
        
        # Mock folding results
        return {
            'folding_energy': -1.5 * len(dna_sequence),
            'stability_score': 0.85,
            'folding_time': 1.0,
            'processing_node': self.node_id
        }
    
    async def _execute_error_correction_task(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute error correction task."""
        # Placeholder for error correction
        # In a real implementation, this would apply sophisticated error correction
        
        data = input_data['data']
        
        # Simulate error correction
        await asyncio.sleep(0.5)
        
        return {
            'corrected_data': data,  # Mock - same data
            'errors_detected': 0,
            'errors_corrected': 0,
            'processing_node': self.node_id
        }
    
    async def _notify_task_completion(
        self, 
        task_id: str, 
        result: Any, 
        success: bool, 
        error: Optional[str] = None
    ):
        """Notify coordinator of task completion."""
        completion_data = {
            'task_id': task_id,
            'success': success,
            'result': result,
            'error': error,
            'node_id': self.node_id,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.coordinator_url}/task_complete",
                    json=completion_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Task completion notification sent: {task_id}")
                    else:
                        self.logger.warning(f"Completion notification failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to notify task completion: {e}")
    
    async def _handle_health_check(self, request):
        """Handle health check request."""
        return web.json_response({
            'status': 'healthy',
            'node_id': self.node_id,
            'current_load': self.current_load,
            'active_tasks': len(self.active_tasks),
            'capabilities': self.capabilities
        })
    
    async def _handle_status_request(self, request):
        """Handle status request."""
        return web.json_response({
            'node_id': self.node_id,
            'current_load': self.current_load,
            'active_tasks': len(self.active_tasks),
            'capabilities': self.capabilities,
            'uptime_seconds': (datetime.now() - datetime.now()).total_seconds()  # Placeholder
        })

# High-level functions for edge computing integration
async def submit_edge_computation_task(
    coordinator_url: str,
    task_type: str,
    input_data: Any,
    priority: int = 1,
    deadline_seconds: Optional[int] = None
) -> str:
    """
    Submit a computation task to the edge computing network.
    Returns task ID for tracking.
    """
    submission_data = {
        'task_type': task_type,
        'input_data': input_data,
        'priority': priority
    }
    
    if deadline_seconds:
        submission_data['deadline_seconds'] = deadline_seconds
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{coordinator_url}/submit_task",
            json=submission_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result['task_id']
            else:
                raise Exception(f"Task submission failed: {response.status}")

async def get_task_result(coordinator_url: str, task_id: str) -> Dict[str, Any]:
    """
    Get the result of a computation task.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{coordinator_url}/task_status/{task_id}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Task status query failed: {response.status}")

async def distributed_quantum_encoding(
    coordinator_url: str,
    image_data: ImageData,
    config: Optional[QuantumEncodingConfig] = None
) -> Dict[str, Any]:
    """
    Perform distributed quantum encoding using edge computing network.
    """
    logger.info("Starting distributed quantum encoding")
    
    # Prepare input data
    input_data = {
        'image_data': image_data.data.tolist(),
        'config': config.__dict__ if config else {}
    }
    
    # Submit task
    task_id = await submit_edge_computation_task(
        coordinator_url,
        'quantum_dna_encoding',
        input_data,
        priority=1,
        deadline_seconds=300  # 5 minutes
    )
    
    # Poll for completion
    max_wait_time = 300  # 5 minutes
    start_time = datetime.now()
    
    while (datetime.now() - start_time).total_seconds() < max_wait_time:
        task_status = await get_task_result(coordinator_url, task_id)
        
        if task_status['status'] == 'completed':
            logger.info("Distributed quantum encoding completed successfully")
            return task_status
        elif task_status['status'] == 'failed':
            raise Exception(f"Distributed encoding failed: {task_status.get('error')}")
        
        # Wait before next poll
        await asyncio.sleep(2)
    
    raise Exception("Distributed encoding timed out")