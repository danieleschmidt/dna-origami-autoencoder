"""
AI Agent Swarm Intelligence for DNA Origami AutoEncoder

Provides intelligent agent swarm coordination for autonomous DNA origami
design, optimization, and fabrication tasks.
"""

import asyncio
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import networkx as nx

from .logger import get_logger
from .performance import PerformanceOptimizer

logger = get_logger(__name__)


class AgentCapability(Enum):
    """Defines capabilities that agents can possess."""
    DESIGN = "design"
    OPTIMIZE = "optimize"
    ANALYZE = "analyze"
    SIMULATE = "simulate"
    FABRICATE = "fabricate"
    COORDINATE = "coordinate"
    LEARN = "learn"
    COMMUNICATE = "communicate"


@dataclass
class AgentTask:
    """Represents a task that can be assigned to an agent."""
    task_id: str
    task_type: str
    priority: int
    data: Dict[str, Any]
    requirements: List[AgentCapability]
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    assigned_agent: Optional[str] = None
    status: str = "pending"  # pending, assigned, running, completed, failed
    result: Optional[Any] = None


@dataclass
class AgentProfile:
    """Represents an AI agent's profile and state."""
    agent_id: str
    capabilities: List[AgentCapability]
    specialization_score: Dict[AgentCapability, float]
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    current_tasks: List[str] = field(default_factory=list)
    workload: float = 0.0
    reputation: float = 1.0
    learning_rate: float = 0.01
    collaboration_preference: float = 0.5  # 0 = solo, 1 = team-oriented
    

class SwarmBehavior(Enum):
    """Different swarm intelligence behaviors."""
    EXPLORATION = "exploration"      # Explore new design spaces
    EXPLOITATION = "exploitation"    # Optimize known good solutions
    COLLABORATION = "collaboration"  # Work together on complex tasks
    COMPETITION = "competition"      # Compete for better solutions
    CONSENSUS = "consensus"          # Reach agreement on decisions
    ADAPTATION = "adaptation"        # Learn and adapt strategies


class AgentSwarm:
    """
    Implements swarm intelligence for coordinating multiple AI agents
    in DNA origami design and optimization tasks.
    
    Features:
    - Emergent swarm behaviors
    - Distributed task allocation
    - Collective learning and memory
    - Self-organization and adaptation
    - Multi-objective optimization
    - Fault tolerance and recovery
    """
    
    def __init__(self,
                 max_agents: int = 50,
                 swarm_behaviors: List[SwarmBehavior] = None,
                 learning_enabled: bool = True,
                 communication_topology: str = "small_world"):
        
        self.max_agents = max_agents
        self.swarm_behaviors = swarm_behaviors or list(SwarmBehavior)
        self.learning_enabled = learning_enabled
        self.communication_topology = communication_topology
        
        # Agent management
        self.agents: Dict[str, AgentProfile] = {}
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        
        # Swarm intelligence components
        self.collective_memory: Dict[str, Any] = {}
        self.swarm_knowledge_base: Dict[str, Any] = {}
        self.consensus_mechanisms: Dict[str, Callable] = {}
        
        # Communication network
        self.communication_graph = nx.Graph()
        self.message_routing: Dict[str, List[str]] = {}
        
        # Performance and adaptation
        self.performance_optimizer = PerformanceOptimizer()
        self.adaptation_history: List[Dict[str, Any]] = []
        self.swarm_metrics: Dict[str, float] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_agents)
        self.running = False
        
        # Initialize swarm behaviors
        self._initialize_swarm_behaviors()
        self._initialize_consensus_mechanisms()
        
    def _initialize_swarm_behaviors(self):
        """Initialize different swarm behavior patterns."""
        self.behavior_strategies = {
            SwarmBehavior.EXPLORATION: self._exploration_strategy,
            SwarmBehavior.EXPLOITATION: self._exploitation_strategy,
            SwarmBehavior.COLLABORATION: self._collaboration_strategy,
            SwarmBehavior.COMPETITION: self._competition_strategy,
            SwarmBehavior.CONSENSUS: self._consensus_strategy,
            SwarmBehavior.ADAPTATION: self._adaptation_strategy
        }
        
    def _initialize_consensus_mechanisms(self):
        """Initialize consensus mechanisms for swarm decision making."""
        self.consensus_mechanisms = {
            "voting": self._voting_consensus,
            "reputation_weighted": self._reputation_weighted_consensus,
            "expertise_based": self._expertise_based_consensus,
            "convergence": self._convergence_consensus
        }
        
    async def register_agent(self, 
                           agent_id: str, 
                           agent_type: str,
                           capabilities: List[str]) -> AgentProfile:
        """Register a new agent with the swarm."""
        if len(self.agents) >= self.max_agents:
            raise ValueError("Maximum number of agents reached")
            
        # Convert string capabilities to enum
        agent_capabilities = []
        for cap_str in capabilities:
            try:
                capability = AgentCapability(cap_str)
                agent_capabilities.append(capability)
            except ValueError:
                logger.warning(f"Unknown capability: {cap_str}")
                
        # Create agent profile
        profile = AgentProfile(
            agent_id=agent_id,
            capabilities=agent_capabilities,
            specialization_score={
                cap: np.random.uniform(0.5, 1.0) 
                for cap in agent_capabilities
            }
        )
        
        self.agents[agent_id] = profile
        
        # Add to communication network
        self.communication_graph.add_node(agent_id)
        self._update_communication_topology()
        
        logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")
        
        return profile
        
    async def unregister_agent(self, agent_id: str):
        """Remove an agent from the swarm."""
        if agent_id not in self.agents:
            return
            
        # Reassign current tasks
        agent = self.agents[agent_id]
        for task_id in agent.current_tasks:
            await self._reassign_task(task_id)
            
        # Remove from communication network
        self.communication_graph.remove_node(agent_id)
        self._update_communication_topology()
        
        # Remove from agents
        del self.agents[agent_id]
        
        logger.info(f"Unregistered agent: {agent_id}")
        
    def _update_communication_topology(self):
        """Update the communication network topology."""
        if self.communication_topology == "fully_connected":
            # Every agent can communicate with every other agent
            for agent1 in self.agents:
                for agent2 in self.agents:
                    if agent1 != agent2:
                        self.communication_graph.add_edge(agent1, agent2)
                        
        elif self.communication_topology == "small_world":
            # Small world network with some random connections
            agents = list(self.agents.keys())
            n = len(agents)
            
            if n > 2:
                # Create ring lattice
                for i in range(n):
                    self.communication_graph.add_edge(agents[i], agents[(i+1) % n])
                    if n > 3:
                        self.communication_graph.add_edge(agents[i], agents[(i+2) % n])
                        
                # Add some random connections
                for _ in range(n // 4):
                    agent1, agent2 = np.random.choice(agents, 2, replace=False)
                    self.communication_graph.add_edge(agent1, agent2)
                    
        elif self.communication_topology == "hierarchical":
            # Hierarchical structure based on agent capabilities
            agents = list(self.agents.keys())
            
            # Create hierarchy based on number of capabilities
            agents_by_capability = sorted(
                agents,
                key=lambda a: len(self.agents[a].capabilities),
                reverse=True
            )
            
            # Connect in tree structure
            for i, agent in enumerate(agents_by_capability[1:], 1):
                parent = agents_by_capability[i // 2]
                self.communication_graph.add_edge(agent, parent)
                
    async def submit_task(self, 
                         task_type: str,
                         task_data: Dict[str, Any],
                         requirements: List[str],
                         priority: int = 1,
                         deadline: Optional[float] = None) -> str:
        """Submit a task to the swarm for execution."""
        task_id = f"swarm_task_{len(self.task_queue):06d}"
        
        # Convert string requirements to capabilities
        capability_requirements = []
        for req in requirements:
            try:
                capability = AgentCapability(req)
                capability_requirements.append(capability)
            except ValueError:
                logger.warning(f"Unknown capability requirement: {req}")
                
        task = AgentTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=task_data,
            requirements=capability_requirements,
            deadline=deadline
        )
        
        self.task_queue.append(task)
        
        # Trigger task allocation
        await self._allocate_tasks()
        
        logger.info(f"Submitted task {task_id} to swarm")
        
        return task_id
        
    async def _allocate_tasks(self):
        """Allocate pending tasks to suitable agents using swarm intelligence."""
        if not self.task_queue:
            return
            
        # Sort tasks by priority and deadline
        self.task_queue.sort(key=lambda t: (-t.priority, t.deadline or float('inf')))
        
        allocated_tasks = []
        
        for task in self.task_queue:
            if task.status != "pending":
                continue
                
            # Find suitable agents
            suitable_agents = self._find_suitable_agents(task)
            
            if not suitable_agents:
                logger.warning(f"No suitable agents found for task {task.task_id}")
                continue
                
            # Use swarm behavior to select agent
            selected_agent = await self._swarm_agent_selection(task, suitable_agents)
            
            if selected_agent:
                await self._assign_task_to_agent(task, selected_agent)
                allocated_tasks.append(task)
                
        # Remove allocated tasks from queue
        self.task_queue = [t for t in self.task_queue if t not in allocated_tasks]
        
    def _find_suitable_agents(self, task: AgentTask) -> List[str]:
        """Find agents suitable for executing a task."""
        suitable_agents = []
        
        for agent_id, agent in self.agents.items():
            # Check capability requirements
            if not all(cap in agent.capabilities for cap in task.requirements):
                continue
                
            # Check workload capacity
            if agent.workload >= 1.0:  # Agent at maximum capacity
                continue
                
            # Check specialization scores
            avg_specialization = np.mean([
                agent.specialization_score.get(cap, 0.0) 
                for cap in task.requirements
            ])
            
            if avg_specialization < 0.3:  # Minimum competency threshold
                continue
                
            suitable_agents.append(agent_id)
            
        return suitable_agents
        
    async def _swarm_agent_selection(self, 
                                   task: AgentTask, 
                                   suitable_agents: List[str]) -> Optional[str]:
        """Use swarm intelligence to select the best agent for a task."""
        if not suitable_agents:
            return None
            
        # Calculate selection probabilities using multiple swarm behaviors
        selection_scores = {}
        
        for agent_id in suitable_agents:
            agent = self.agents[agent_id]
            score = 0.0
            
            # Competency score
            competency = np.mean([
                agent.specialization_score.get(cap, 0.0) 
                for cap in task.requirements
            ])
            score += competency * 0.4
            
            # Workload score (prefer less loaded agents)
            workload_score = 1.0 - agent.workload
            score += workload_score * 0.3
            
            # Reputation score
            score += agent.reputation * 0.2
            
            # Collaboration potential (for collaborative tasks)
            if task.task_type in ["design_optimization", "multi_objective"]:
                collaboration_score = agent.collaboration_preference
                score += collaboration_score * 0.1
                
            selection_scores[agent_id] = score
            
        # Apply swarm behavior modifications
        for behavior in self.swarm_behaviors:
            selection_scores = await self.behavior_strategies[behavior](
                task, selection_scores, suitable_agents
            )
            
        # Select agent probabilistically based on scores
        agents = list(selection_scores.keys())
        scores = list(selection_scores.values())
        
        # Normalize scores to probabilities
        total_score = sum(scores)
        if total_score > 0:
            probabilities = [s / total_score for s in scores]
            selected_agent = np.random.choice(agents, p=probabilities)
            return selected_agent
            
        return None
        
    async def _assign_task_to_agent(self, task: AgentTask, agent_id: str):
        """Assign a task to a specific agent."""
        agent = self.agents[agent_id]
        
        task.assigned_agent = agent_id
        task.status = "assigned"
        
        # Update agent state
        agent.current_tasks.append(task.task_id)
        agent.workload += self._estimate_task_workload(task)
        
        logger.info(f"Assigned task {task.task_id} to agent {agent_id}")
        
        # Start task execution
        asyncio.create_task(self._execute_task(task))
        
    def _estimate_task_workload(self, task: AgentTask) -> float:
        """Estimate the workload of a task (0.0 to 1.0)."""
        # Simple estimation based on task complexity
        base_workload = {
            "simple_design": 0.2,
            "optimization": 0.4,
            "simulation": 0.6,
            "analysis": 0.3,
            "fabrication": 0.5,
            "coordination": 0.3
        }
        
        return base_workload.get(task.task_type, 0.4)
        
    async def _execute_task(self, task: AgentTask):
        """Execute a task assigned to an agent."""
        agent_id = task.assigned_agent
        agent = self.agents[agent_id]
        
        try:
            task.status = "running"
            start_time = time.time()
            
            # Simulate task execution
            execution_time = np.random.uniform(1.0, 5.0)  # 1-5 seconds
            await asyncio.sleep(execution_time)
            
            # Simulate task result
            success = np.random.random() > 0.1  # 90% success rate
            
            if success:
                task.status = "completed"
                task.result = {"success": True, "execution_time": execution_time}
                
                # Update agent reputation positively
                agent.reputation = min(1.0, agent.reputation + 0.01)
                
            else:
                task.status = "failed"
                task.result = {"success": False, "error": "Simulated failure"}
                
                # Update agent reputation negatively
                agent.reputation = max(0.0, agent.reputation - 0.02)
                
            # Record performance
            performance_record = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "success": success,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            agent.performance_history.append(performance_record)
            
            # Update collective memory
            if self.learning_enabled:
                await self._update_collective_memory(task, agent_id, success)
                
        except Exception as e:
            task.status = "failed"
            task.result = {"success": False, "error": str(e)}
            logger.error(f"Task execution failed: {e}")
            
        finally:
            # Clean up agent state
            agent.current_tasks.remove(task.task_id)
            agent.workload -= self._estimate_task_workload(task)
            agent.workload = max(0.0, agent.workload)
            
            # Move task to completed list
            self.completed_tasks.append(task)
            
    async def _update_collective_memory(self, task: AgentTask, agent_id: str, success: bool):
        """Update the swarm's collective memory with task outcomes."""
        memory_key = f"{task.task_type}_{hash(str(sorted(task.requirements)))}"
        
        if memory_key not in self.collective_memory:
            self.collective_memory[memory_key] = {
                "successful_agents": [],
                "failed_agents": [],
                "success_patterns": {},
                "failure_patterns": {}
            }
            
        memory = self.collective_memory[memory_key]
        
        if success:
            memory["successful_agents"].append(agent_id)
            # Record successful pattern
            agent = self.agents[agent_id]
            pattern = {
                "capabilities": list(agent.capabilities),
                "specialization_scores": agent.specialization_score.copy(),
                "workload_at_assignment": agent.workload
            }
            memory["success_patterns"][agent_id] = pattern
        else:
            memory["failed_agents"].append(agent_id)
            # Record failure pattern
            agent = self.agents[agent_id]
            pattern = {
                "capabilities": list(agent.capabilities),
                "specialization_scores": agent.specialization_score.copy(),
                "workload_at_assignment": agent.workload
            }
            memory["failure_patterns"][agent_id] = pattern
            
    # Swarm behavior strategies
    async def _exploration_strategy(self, task, selection_scores, agents):
        """Exploration behavior - prefer agents with less experience."""
        for agent_id in agents:
            agent = self.agents[agent_id]
            # Boost score for agents with less experience in this task type
            experience = len([t for t in agent.performance_history 
                            if t["task_type"] == task.task_type])
            exploration_boost = 1.0 / (1.0 + experience * 0.1)
            selection_scores[agent_id] *= exploration_boost
            
        return selection_scores
        
    async def _exploitation_strategy(self, task, selection_scores, agents):
        """Exploitation behavior - prefer agents with proven success."""
        for agent_id in agents:
            agent = self.agents[agent_id]
            # Boost score for agents with high success rate
            recent_tasks = agent.performance_history[-10:]  # Last 10 tasks
            if recent_tasks:
                success_rate = sum(t["success"] for t in recent_tasks) / len(recent_tasks)
                selection_scores[agent_id] *= (1.0 + success_rate)
                
        return selection_scores
        
    async def _collaboration_strategy(self, task, selection_scores, agents):
        """Collaboration behavior - consider team formation opportunities."""
        # This is simplified - full implementation would consider
        # forming teams of complementary agents
        for agent_id in agents:
            agent = self.agents[agent_id]
            collaboration_boost = agent.collaboration_preference
            selection_scores[agent_id] *= (1.0 + collaboration_boost * 0.2)
            
        return selection_scores
        
    async def _competition_strategy(self, task, selection_scores, agents):
        """Competition behavior - multiple agents compete for better solutions."""
        # For high-priority tasks, assign to multiple agents
        if task.priority >= 3:
            # Boost all capable agents to encourage competition
            for agent_id in agents:
                selection_scores[agent_id] *= 1.1
                
        return selection_scores
        
    async def _consensus_strategy(self, task, selection_scores, agents):
        """Consensus behavior - use collective decision making."""
        # Use consensus mechanism to make selection
        consensus_result = await self.consensus_mechanisms["reputation_weighted"](
            agents, task
        )
        
        if consensus_result:
            # Boost consensus choice
            selection_scores[consensus_result] *= 2.0
            
        return selection_scores
        
    async def _adaptation_strategy(self, task, selection_scores, agents):
        """Adaptation behavior - learn from collective memory."""
        memory_key = f"{task.task_type}_{hash(str(sorted(task.requirements)))}"
        
        if memory_key in self.collective_memory:
            memory = self.collective_memory[memory_key]
            
            # Boost agents with historical success
            for agent_id in agents:
                if agent_id in memory.get("successful_agents", []):
                    selection_scores[agent_id] *= 1.5
                elif agent_id in memory.get("failed_agents", []):
                    selection_scores[agent_id] *= 0.7
                    
        return selection_scores
        
    # Consensus mechanisms
    async def _voting_consensus(self, agents: List[str], task: AgentTask) -> Optional[str]:
        """Simple voting consensus mechanism."""
        # Simplified implementation
        if agents:
            return np.random.choice(agents)
        return None
        
    async def _reputation_weighted_consensus(self, agents: List[str], task: AgentTask) -> Optional[str]:
        """Reputation-weighted consensus mechanism."""
        if not agents:
            return None
            
        weights = [self.agents[agent_id].reputation for agent_id in agents]
        total_weight = sum(weights)
        
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
            return np.random.choice(agents, p=probabilities)
            
        return np.random.choice(agents)
        
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status and metrics."""
        total_tasks = len(self.completed_tasks)
        successful_tasks = sum(1 for t in self.completed_tasks if t.result and t.result.get("success"))
        
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.current_tasks]),
            "pending_tasks": len(self.task_queue),
            "running_tasks": len([a.current_tasks for a in self.agents.values()]),
            "completed_tasks": total_tasks,
            "success_rate": successful_tasks / max(total_tasks, 1),
            "average_reputation": np.mean([a.reputation for a in self.agents.values()]) if self.agents else 0,
            "swarm_behaviors": [b.value for b in self.swarm_behaviors],
            "collective_memory_size": len(self.collective_memory),
            "communication_topology": self.communication_topology,
            "network_density": nx.density(self.communication_graph)
        }
        
    async def start(self):
        """Start the agent swarm."""
        self.running = True
        logger.info("Started agent swarm")
        
    async def stop(self):
        """Stop the agent swarm."""
        self.running = False
        logger.info("Stopped agent swarm")
        
    # Additional methods would be implemented for:
    # - _reassign_task()
    # - _expertise_based_consensus()
    # - _convergence_consensus()
    # - Advanced swarm behaviors
    # - Learning and adaptation mechanisms
    # - Multi-objective optimization
    # - Fault tolerance and recovery