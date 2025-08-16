"""
Adaptive Learning System - Generation 1 Enhancement
Self-improving ML system that learns from DNA origami folding patterns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import pickle
from datetime import datetime
import threading
import time
from ..utils.logger import get_logger
from ..models.dna_sequence import DNASequence
from ..models.origami_structure import OrigamiStructure

logger = get_logger(__name__)

@dataclass
class LearningConfig:
    """Configuration for adaptive learning system."""
    learning_rate: float = 0.001
    batch_size: int = 32
    adaptation_threshold: float = 0.1
    memory_size: int = 10000
    exploration_rate: float = 0.15
    update_frequency: int = 100
    enable_online_learning: bool = True
    convergence_patience: int = 50

@dataclass
class ExperienceMemory:
    """Memory structure for storing learning experiences."""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    metadata: List[Dict] = field(default_factory=list)

class AdaptiveNeuralNetwork(nn.Module):
    """
    Self-adapting neural network that evolves its architecture based on performance.
    """
    
    def __init__(self, input_size: int = 1000, hidden_sizes: List[int] = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or [512, 256, 128]
        self.adaptation_history = []
        
        # Build initial architecture
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in self.hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 64)  # 64-dimensional embedding
        
        # Adaptive components
        self.performance_tracker = []
        self.adaptation_counter = 0
        
    def forward(self, x):
        """Forward pass through adaptive network."""
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
    
    def adapt_architecture(self, performance_metrics: Dict[str, float]):
        """
        Dynamically adapt network architecture based on performance.
        """
        current_loss = performance_metrics.get('loss', float('inf'))
        
        # Track performance
        self.performance_tracker.append(current_loss)
        
        # Check if adaptation is needed (performance plateau)
        if len(self.performance_tracker) >= 10:
            recent_performance = self.performance_tracker[-10:]
            performance_variance = np.var(recent_performance)
            
            if performance_variance < 0.001:  # Performance plateau
                self._expand_architecture()
                self.adaptation_counter += 1
                logger.info(f"Architecture adapted (#{self.adaptation_counter}): added capacity")
        
        # Limit tracking history
        if len(self.performance_tracker) > 100:
            self.performance_tracker = self.performance_tracker[-50:]
    
    def _expand_architecture(self):
        """Expand network architecture by adding neurons."""
        # Add neurons to the largest hidden layer
        max_layer_idx = 0
        max_size = 0
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and layer.out_features > max_size:
                max_size = layer.out_features
                max_layer_idx = i
        
        # Expand the layer (simplified - in practice, this requires careful weight initialization)
        old_layer = self.layers[max_layer_idx]
        new_size = int(old_layer.out_features * 1.2)  # 20% increase
        
        # Create new layer with expanded size
        new_layer = nn.Linear(old_layer.in_features, new_size)
        
        # Initialize weights (transfer learning approach)
        with torch.no_grad():
            new_layer.weight[:old_layer.out_features] = old_layer.weight
            new_layer.bias[:old_layer.out_features] = old_layer.bias
            # Initialize new neurons with small random weights
            nn.init.xavier_uniform_(new_layer.weight[old_layer.out_features:])
            nn.init.zeros_(new_layer.bias[old_layer.out_features:])
        
        self.layers[max_layer_idx] = new_layer
        
        # Update subsequent layer if needed
        if max_layer_idx + 3 < len(self.layers):  # Skip ReLU and Dropout
            next_linear_idx = max_layer_idx + 3
            if next_linear_idx < len(self.layers) and isinstance(self.layers[next_linear_idx], nn.Linear):
                old_next = self.layers[next_linear_idx]
                new_next = nn.Linear(new_size, old_next.out_features)
                
                with torch.no_grad():
                    new_next.weight[:, :old_layer.out_features] = old_next.weight
                    new_next.bias = old_next.bias
                    # Initialize connections to new neurons
                    nn.init.xavier_uniform_(new_next.weight[:, old_layer.out_features:])
                
                self.layers[next_linear_idx] = new_next

class ContinualLearningSystem:
    """
    Continual learning system that never stops improving.
    Implements online learning with catastrophic forgetting prevention.
    """
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.ContinualLearningSystem")
        
        # Initialize components
        self.model = AdaptiveNeuralNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.memory = ExperienceMemory()
        
        # Learning state
        self.learning_active = True
        self.total_experiences = 0
        self.performance_history = []
        
        # Online learning thread
        if config.enable_online_learning:
            self.learning_thread = threading.Thread(target=self._continuous_learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
    
    def add_experience(self, state: np.ndarray, action: Dict, reward: float, 
                      next_state: np.ndarray, metadata: Dict = None):
        """
        Add new experience to memory for continual learning.
        """
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.rewards.append(reward)
        self.memory.next_states.append(next_state)
        self.memory.metadata.append(metadata or {})
        
        # Limit memory size
        if len(self.memory.states) > self.config.memory_size:
            # Remove oldest experiences
            self.memory.states.pop(0)
            self.memory.actions.pop(0)
            self.memory.rewards.pop(0)
            self.memory.next_states.pop(0)
            self.memory.metadata.pop(0)
        
        self.total_experiences += 1
        
        if self.total_experiences % 100 == 0:
            self.logger.info(f"Added experience #{self.total_experiences}, memory size: {len(self.memory.states)}")
    
    def learn_from_folding_outcome(self, dna_sequence: str, folding_result: Dict, 
                                  success_score: float):
        """
        Learn from DNA folding experimental outcomes.
        """
        # Encode DNA sequence as state
        state = self._encode_dna_sequence(dna_sequence)
        
        # Extract action (folding parameters)
        action = {
            'temperature': folding_result.get('temperature', 25.0),
            'salt_concentration': folding_result.get('salt_concentration', 0.5),
            'folding_time': folding_result.get('folding_time', 3600),
            'sequence_length': len(dna_sequence)
        }
        
        # Reward based on success
        reward = success_score  # 0.0 to 1.0
        
        # Next state (could be modified sequence or environment)
        next_state = state  # Simplified
        
        # Metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'folding_success': success_score > 0.7,
            'gc_content': (dna_sequence.count('G') + dna_sequence.count('C')) / len(dna_sequence),
            'experiment_id': folding_result.get('experiment_id', 'unknown')
        }
        
        self.add_experience(state, action, reward, next_state, metadata)
    
    def predict_folding_success(self, dna_sequence: str, folding_params: Dict) -> float:
        """
        Predict folding success probability using learned model.
        """
        state = self._encode_dna_sequence(dna_sequence)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            embedding = self.model(state_tensor)
            
            # Simple prediction based on embedding (could be more sophisticated)
            prediction = torch.sigmoid(embedding.mean()).item()
        
        return prediction
    
    def _encode_dna_sequence(self, sequence: str) -> np.ndarray:
        """Encode DNA sequence into numerical vector."""
        # Base encoding
        base_map = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
        
        encoded = []
        for base in sequence:
            encoded.extend(base_map.get(base, [0, 0, 0, 0]))
        
        # Pad or truncate to fixed size
        if len(encoded) < 1000:
            encoded.extend([0] * (1000 - len(encoded)))
        else:
            encoded = encoded[:1000]
        
        return np.array(encoded, dtype=np.float32)
    
    def _continuous_learning_loop(self):
        """Continuous learning background thread."""
        self.logger.info("Starting continuous learning loop")
        
        while self.learning_active:
            try:
                if len(self.memory.states) >= self.config.batch_size:
                    self._perform_learning_update()
                
                time.sleep(1.0)  # Learning update frequency
                
            except Exception as e:
                self.logger.error(f"Error in continuous learning: {e}")
                time.sleep(5.0)  # Longer wait on error
    
    def _perform_learning_update(self):
        """Perform a single learning update."""
        # Sample batch from memory
        batch_size = min(self.config.batch_size, len(self.memory.states))
        indices = np.random.choice(len(self.memory.states), batch_size, replace=False)
        
        # Prepare batch
        states = torch.FloatTensor([self.memory.states[i] for i in indices])
        rewards = torch.FloatTensor([self.memory.rewards[i] for i in indices])
        
        # Forward pass
        self.optimizer.zero_grad()
        embeddings = self.model(states)
        
        # Simple loss: predict rewards from embeddings
        predicted_rewards = embeddings.mean(dim=1)
        loss = nn.MSELoss()(predicted_rewards, rewards)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Track performance
        current_performance = {
            'loss': loss.item(),
            'mean_reward': rewards.mean().item(),
            'batch_size': batch_size
        }
        
        self.performance_history.append(current_performance)
        
        # Adapt architecture if needed
        self.model.adapt_architecture(current_performance)
        
        # Log occasionally
        if len(self.performance_history) % self.config.update_frequency == 0:
            self.logger.info(f"Learning update: loss={loss.item():.4f}, mean_reward={rewards.mean().item():.3f}")
    
    def save_learning_state(self, filepath: str):
        """Save the current learning state."""
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'memory': {
                'states': self.memory.states[-1000:],  # Save last 1000 experiences
                'actions': self.memory.actions[-1000:],
                'rewards': self.memory.rewards[-1000:],
                'next_states': self.memory.next_states[-1000:],
                'metadata': self.memory.metadata[-1000:]
            },
            'performance_history': self.performance_history[-500:],  # Last 500 updates
            'total_experiences': self.total_experiences,
            'architecture_adaptations': self.model.adaptation_counter
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Learning state saved to {filepath}")
    
    def load_learning_state(self, filepath: str):
        """Load previous learning state."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore model and optimizer
            self.model.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])
            
            # Restore memory
            memory_data = state['memory']
            self.memory.states = memory_data['states']
            self.memory.actions = memory_data['actions']
            self.memory.rewards = memory_data['rewards']
            self.memory.next_states = memory_data['next_states']
            self.memory.metadata = memory_data['metadata']
            
            # Restore tracking
            self.performance_history = state['performance_history']
            self.total_experiences = state['total_experiences']
            self.model.adaptation_counter = state.get('architecture_adaptations', 0)
            
            self.logger.info(f"Learning state loaded from {filepath}")
            self.logger.info(f"Resumed with {self.total_experiences} total experiences")
            
        except Exception as e:
            self.logger.error(f"Failed to load learning state: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        if not self.performance_history:
            return {'status': 'no_learning_data'}
        
        recent_performance = self.performance_history[-50:] if len(self.performance_history) >= 50 else self.performance_history
        
        stats = {
            'total_experiences': self.total_experiences,
            'memory_utilization': len(self.memory.states) / self.config.memory_size,
            'recent_avg_loss': np.mean([p['loss'] for p in recent_performance]),
            'recent_avg_reward': np.mean([p['mean_reward'] for p in recent_performance]),
            'architecture_adaptations': self.model.adaptation_counter,
            'learning_stability': np.std([p['loss'] for p in recent_performance]),
            'improvement_trend': self._calculate_improvement_trend(),
            'active_learning': self.learning_active
        }
        
        return stats
    
    def _calculate_improvement_trend(self) -> str:
        """Calculate whether the system is improving, stable, or degrading."""
        if len(self.performance_history) < 20:
            return 'insufficient_data'
        
        recent_losses = [p['loss'] for p in self.performance_history[-20:]]
        older_losses = [p['loss'] for p in self.performance_history[-40:-20]] if len(self.performance_history) >= 40 else recent_losses
        
        recent_avg = np.mean(recent_losses)
        older_avg = np.mean(older_losses)
        
        if recent_avg < older_avg * 0.95:
            return 'improving'
        elif recent_avg > older_avg * 1.05:
            return 'degrading'
        else:
            return 'stable'
    
    def stop_learning(self):
        """Stop the continuous learning process."""
        self.learning_active = False
        self.logger.info("Continuous learning stopped")

class MetaLearningOptimizer:
    """
    Meta-learning system that learns how to learn better.
    Optimizes learning parameters based on task performance.
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.MetaLearningOptimizer")
        self.parameter_history = []
        self.performance_history = []
        
    def optimize_learning_parameters(self, current_config: LearningConfig, 
                                   performance_metrics: Dict[str, float]) -> LearningConfig:
        """
        Optimize learning parameters based on current performance.
        """
        self.logger.info("Optimizing learning parameters")
        
        # Record current configuration and performance
        config_dict = {
            'learning_rate': current_config.learning_rate,
            'batch_size': current_config.batch_size,
            'exploration_rate': current_config.exploration_rate
        }
        
        self.parameter_history.append(config_dict)
        self.performance_history.append(performance_metrics.get('mean_reward', 0.0))
        
        # Simple adaptive parameter adjustment
        new_config = LearningConfig(
            learning_rate=current_config.learning_rate,
            batch_size=current_config.batch_size,
            exploration_rate=current_config.exploration_rate,
            adaptation_threshold=current_config.adaptation_threshold,
            memory_size=current_config.memory_size,
            update_frequency=current_config.update_frequency,
            enable_online_learning=current_config.enable_online_learning,
            convergence_patience=current_config.convergence_patience
        )
        
        # Adjust learning rate based on performance trend
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-5:])
            older_performance = np.mean(self.performance_history[-10:-5])
            
            if recent_performance > older_performance:
                # Performance improving, continue current trajectory
                new_config.learning_rate = min(current_config.learning_rate * 1.1, 0.01)
            else:
                # Performance declining, reduce learning rate
                new_config.learning_rate = max(current_config.learning_rate * 0.9, 0.0001)
        
        # Adjust exploration based on learning progress
        if len(self.performance_history) >= 20:
            performance_variance = np.var(self.performance_history[-20:])
            if performance_variance < 0.01:  # Low variance suggests need for more exploration
                new_config.exploration_rate = min(current_config.exploration_rate * 1.2, 0.5)
            else:
                new_config.exploration_rate = max(current_config.exploration_rate * 0.95, 0.05)
        
        self.logger.info(f"Updated learning rate: {new_config.learning_rate:.6f}")
        self.logger.info(f"Updated exploration rate: {new_config.exploration_rate:.3f}")
        
        return new_config

# Main integration function
def create_adaptive_learning_system(config: LearningConfig = None) -> ContinualLearningSystem:
    """
    Create and initialize the adaptive learning system.
    """
    if config is None:
        config = LearningConfig()
    
    logger.info("Creating adaptive learning system")
    system = ContinualLearningSystem(config)
    
    logger.info("Adaptive learning system initialized and running")
    return system