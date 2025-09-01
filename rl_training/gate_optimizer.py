# rl_training/gate_optimizer.py
# core/context_assembler.py
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
import json

from models.rl_experience import RLExperience, RLAction, RLState, ExperienceBatch
from core.gate_networks import LSTMGateNetwork
from rl_training.experience_buffer import PrioritizedExperienceBuffer
from rl_training.reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """
    Policy network for gate threshold optimization
    Maps states to actions (threshold adjustments)
    """
    
    def __init__(self, state_size: int = 130, action_size: int = 4, hidden_size: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size)
        )
        
        # Action bounds for threshold adjustments
        self.action_bounds = {
            'forget_adj': [-0.2, 0.2],
            'input_adj': [-0.2, 0.2], 
            'output_adj': [-0.2, 0.2],
            'confidence': [-0.1, 0.1]
        }
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass - outputs raw action values"""
        return self.network(state)
    
    def get_action(self, state: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """Get bounded action with optional exploration noise"""
        raw_action = self.forward(state)
        
        # Apply bounds using tanh activation
        bounded_action = torch.tanh(raw_action)
        
        # Scale to action bounds
        scaled_action = torch.zeros_like(bounded_action)
        bounds = list(self.action_bounds.values())
        
        for i, (low, high) in enumerate(bounds):
            scale = (high - low) / 2.0
            offset = (high + low) / 2.0
            scaled_action[:, i] = bounded_action[:, i] * scale + offset
        
        # Add exploration noise if requested
        if add_noise:
            noise = torch.randn_like(scaled_action) * 0.01  # Small noise
            scaled_action = scaled_action + noise
            
            # Re-clip to bounds
            for i, (low, high) in enumerate(bounds):
                scaled_action[:, i] = torch.clamp(scaled_action[:, i], low, high)
        
        return scaled_action

class ValueNetwork(nn.Module):
    """
    Value network for estimating state values
    Used in Actor-Critic methods
    """
    
    def __init__(self, state_size: int = 130, hidden_size: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass - outputs state value"""
        return self.network(state)

class GateOptimizer:
    """
    Main optimizer for LSTM gate networks using reinforcement learning
    Implements Soft Actor-Critic (SAC) algorithm
    """
    
    def __init__(self,
                 gate_network: LSTMGateNetwork,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 buffer_size: int = 100000,
                 batch_size: int = 64):
        
        self.gate_network = gate_network
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update rate
        self.alpha = alpha  # Entropy regularization
        self.batch_size = batch_size
        
        # State and action dimensions
        self.state_dim = 130  # From RLState.get_vector_size()
        self.action_dim = 4   # From RLAction.to_vector()
        
        # Networks
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        self.value_net = ValueNetwork(self.state_dim)
        self.target_value_net = ValueNetwork(self.state_dim)
        self.q_net1 = self._create_q_network()
        self.q_net2 = self._create_q_network()
        
        # Initialize target network
        self._hard_update(self.target_value_net, self.value_net)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=learning_rate)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = PrioritizedExperienceBuffer(buffer_size)
        self.reward_calculator = RewardCalculator()
        
        # Training statistics
        self.training_step = 0
        self.last_losses = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'q1_loss': 0.0,
            'q2_loss': 0.0
        }
        
        # Performance tracking
        self.performance_history = []
        self.threshold_history = []
        
    def _create_q_network(self) -> nn.Module:
        """Create Q-network (state-action value function)"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def add_experience(self, experience: RLExperience):
        """Add experience to buffer"""
        # Calculate initial TD error estimate
        with torch.no_grad():
            state_tensor = torch.tensor(experience.state.to_vector(), dtype=torch.float32).unsqueeze(0)
            current_value = self.value_net(state_tensor)
            
            if experience.next_state:
                next_state_tensor = torch.tensor(experience.next_state.to_vector(), dtype=torch.float32).unsqueeze(0)
                next_value = self.target_value_net(next_state_tensor)
                target_value = experience.get_total_reward() + self.gamma * next_value
            else:
                target_value = torch.tensor([[experience.get_total_reward()]], dtype=torch.float32)
            
            td_error = abs(target_value.item() - current_value.item())
        
        self.experience_buffer.add(experience, td_error)
    
    def get_action(self, state: RLState, explore: bool = True) -> RLAction:
        """Get action from policy network"""
        with torch.no_grad():
            state_tensor = torch.tensor(state.to_vector(), dtype=torch.float32).unsqueeze(0)
            action_tensor = self.policy_net.get_action(state_tensor, add_noise=explore)
            action_values = action_tensor.squeeze(0).numpy()
        
        return RLAction.from_vector(action_values.tolist())
    
    async def train_step(self) -> Dict[str, float]:
        """Perform one training step"""
        if not self.experience_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch from experience buffer
        batch, indices, weights = self.experience_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.tensor(batch.get_states(), dtype=torch.float32)
        actions = torch.tensor(batch.get_actions(), dtype=torch.float32)
        rewards = torch.tensor(batch.get_rewards(), dtype=torch.float32).unsqueeze(1)
        next_states_list = batch.get_next_states()
        dones = torch.tensor(batch.get_dones(), dtype=torch.float32).unsqueeze(1)
        is_weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        
        # Handle next states (some may be None)
        next_states = []
        for next_state in next_states_list:
            if next_state is not None:
                next_states.append(next_state)
            else:
                next_states.append([0.0] * self.state_dim)  # Dummy state for terminal
        next_states = torch.tensor(next_states, dtype=torch.float32)
        
        # Update Q-networks
        q1_loss, q2_loss = self._update_q_networks(states, actions, rewards, next_states, dones, is_weights)
        
        # Update value network
        value_loss = self._update_value_network(states, is_weights)
        
        # Update policy network
        policy_loss = self._update_policy_network(states, is_weights)
        
        # Soft update target networks
        self._soft_update(self.target_value_net, self.value_net)
        
        # Update priorities in experience buffer
        with torch.no_grad():
            current_values = self.value_net(states)
            target_values = rewards + self.gamma * self.target_value_net(next_states) * (1 - dones)
            td_errors = abs(target_values - current_values).squeeze().cpu().numpy()
        
        self.experience_buffer.update_priorities(indices, td_errors.tolist())
        
        # Update statistics
        self.training_step += 1
        losses = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'buffer_size': self.experience_buffer.size()
        }
        self.last_losses = losses
        
        return losses
    
    def _update_q_networks(self, states, actions, rewards, next_states, dones, is_weights):
        """Update Q-networks (critics)"""
        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.policy_net.get_action(next_states, add_noise=False)
            next_q1 = self.q_net1(torch.cat([next_states, next_actions], dim=1))
            next_q2 = self.q_net2(torch.cat([next_states, next_actions], dim=1))
            next_q = torch.min(next_q1, next_q2)
            
            # Add entropy term
            next_values = self.target_value_net(next_states)
            target_q = rewards + self.gamma * (next_values) * (1 - dones)
        
        # Current Q-values
        current_q1 = self.q_net1(torch.cat([states, actions], dim=1))
        current_q2 = self.q_net2(torch.cat([states, actions], dim=1))
        
        # Q-losses with importance sampling weights
        q1_loss = (is_weights * F.mse_loss(current_q1, target_q, reduction='none')).mean()
        q2_loss = (is_weights * F.mse_loss(current_q2, target_q, reduction='none')).mean()
        
        # Update Q-networks
        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()
        
        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()
        
        return q1_loss.item(), q2_loss.item()
    
    def _update_value_network(self, states, is_weights):
        """Update value network"""
        with torch.no_grad():
            actions = self.policy_net.get_action(states, add_noise=False)
            q1_values = self.q_net1(torch.cat([states, actions], dim=1))
            q2_values = self.q_net2(torch.cat([states, actions], dim=1))
            min_q_values = torch.min(q1_values, q2_values)
            
            # Entropy term (approximated)
            entropy_estimate = -self.alpha * torch.log(torch.tensor(0.5))  # Simplified
            target_values = min_q_values + entropy_estimate
        
        current_values = self.value_net(states)
        value_loss = (is_weights * F.mse_loss(current_values, target_values, reduction='none')).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def _update_policy_network(self, states, is_weights):
        """Update policy network (actor)"""
        actions = self.policy_net.get_action(states, add_noise=False)
        q1_values = self.q_net1(torch.cat([states, actions], dim=1))
        q2_values = self.q_net2(torch.cat([states, actions], dim=1))
        min_q_values = torch.min(q1_values, q2_values)
        
        # Policy loss = -Q(s,a) (we want to maximize Q-value)
        policy_loss = (is_weights * (-min_q_values)).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def _soft_update(self, target_net, source_net):
        """Soft update target network"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
    
    def _hard_update(self, target_net, source_net):
        """Hard update target network"""
        target_net.load_state_dict(source_net.state_dict())
    
    async def optimize_gates_for_user(self, 
                                     user_id: str,
                                     user_context: Dict[str, Any],
                                     performance_feedback: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize gate thresholds for a specific user based on their context and feedback
        """
        # Create current state representation
        current_state = self._create_state_from_context(user_id, user_context)
        
        # Get current gate thresholds
        current_thresholds = self.gate_network.get_thresholds()
        
        # Get action from policy
        action = self.get_action(current_state, explore=False)
        
        # Apply action to get new thresholds
        new_thresholds = action.apply_to_thresholds(current_thresholds)
        
        # Update gate network thresholds
        self.gate_network.update_thresholds(new_thresholds)
        
        # Calculate reward from performance feedback
        reward = self.reward_calculator.calculate_comprehensive_reward(performance_feedback)
        
        # Store the optimization result
        optimization_result = {
            'user_id': user_id,
            'old_thresholds': current_thresholds,
            'new_thresholds': new_thresholds,
            'action_taken': action.to_vector(),
            'reward_received': reward,
            'timestamp': datetime.now()
        }
        
        self.threshold_history.append(optimization_result)
        
        logger.info(f"Optimized gates for user {user_id}: {current_thresholds} -> {new_thresholds}")
        
        return new_thresholds
    
    def _create_state_from_context(self, user_id: str, context: Dict[str, Any]) -> RLState:
        """Create RLState from user context"""
        return RLState(
            user_id=user_id,
            feature_vector=context.get('feature_vector', [0.5] * 90),
            context_vector=context.get('context_vector', []),
            user_emotional_state=context.get('emotions', {}),
            time_features=context.get('time_features', {}),
            memory_stats=context.get('memory_stats', {})
        )
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        buffer_stats = self.experience_buffer.get_priority_stats()
        
        return {
            'training_step': self.training_step,
            'buffer_stats': buffer_stats,
            'last_losses': self.last_losses,
            'threshold_optimizations': len(self.threshold_history),
            'recent_performance': self.performance_history[-10:] if self.performance_history else [],
            'network_info': {
                'policy_params': sum(p.numel() for p in self.policy_net.parameters()),
                'value_params': sum(p.numel() for p in self.value_net.parameters()),
                'q1_params': sum(p.numel() for p in self.q_net1.parameters()),
                'q2_params': sum(p.numel() for p in self.q_net2.parameters())
            }
        }
    
    def save_model(self, filepath: str):
        """Save optimizer state"""
        torch.save({
            'policy_net_state': self.policy_net.state_dict(),
            'value_net_state': self.value_net.state_dict(),
            'target_value_net_state': self.target_value_net.state_dict(),
            'q_net1_state': self.q_net1.state_dict(),
            'q_net2_state': self.q_net2.state_dict(),
            'policy_optimizer_state': self.policy_optimizer.state_dict(),
            'value_optimizer_state': self.value_optimizer.state_dict(),
            'q_optimizer1_state': self.q_optimizer1.state_dict(),
            'q_optimizer2_state': self.q_optimizer2.state_dict(),
            'training_step': self.training_step,
            'hyperparameters': {
                'gamma': self.gamma,
                'tau': self.tau,
                'alpha': self.alpha,
                'batch_size': self.batch_size
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load optimizer state"""
        checkpoint = torch.load(filepath)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.value_net.load_state_dict(checkpoint['value_net_state'])
        self.target_value_net.load_state_dict(checkpoint['target_value_net_state'])
        self.q_net1.load_state_dict(checkpoint['q_net1_state'])
        self.q_net2.load_state_dict(checkpoint['q_net2_state'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])
        self.q_optimizer1.load_state_dict(checkpoint['q_optimizer1_state'])
        self.q_optimizer2.load_state_dict(checkpoint['q_optimizer2_state'])
        
        self.training_step = checkpoint['training_step']
        
        # Update hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.gamma = hyperparams['gamma']
        self.tau = hyperparams['tau']
        self.alpha = hyperparams['alpha']
        self.batch_size = hyperparams['batch_size']
        
        logger.info(f"Loaded optimizer model from {filepath}")

class TrainingScheduler:
    """
    Scheduler for managing training frequency and intensity
    """
    
    def __init__(self, 
                 min_experiences: int = 1000,
                 training_frequency: int = 100,  # Train every N experiences
                 batch_size: int = 64,
                 max_training_steps_per_session: int = 10):
        
        self.min_experiences = min_experiences
        self.training_frequency = training_frequency
        self.batch_size = batch_size
        self.max_training_steps = max_training_steps_per_session
        
        self.experiences_since_training = 0
        self.last_training = datetime.now()
    
    def should_train(self, buffer_size: int) -> bool:
        """Check if it's time to train"""
        if buffer_size < self.min_experiences:
            return False
        
        self.experiences_since_training += 1
        
        return self.experiences_since_training >= self.training_frequency
    
    async def run_training_session(self, optimizer: GateOptimizer) -> Dict[str, Any]:
        """Run a training session"""
        session_stats = {
            'start_time': datetime.now(),
            'steps_completed': 0,
            'losses': [],
            'buffer_size': optimizer.experience_buffer.size()
        }
        
        for step in range(self.max_training_steps):
            if not optimizer.experience_buffer.is_ready(self.batch_size):
                break
            
            losses = await optimizer.train_step()
            if losses:
                session_stats['losses'].append(losses)
                session_stats['steps_completed'] += 1
        
        session_stats['end_time'] = datetime.now()
        session_stats['duration'] = (session_stats['end_time'] - session_stats['start_time']).total_seconds()
        
        # Reset training counter
        self.experiences_since_training = 0
        self.last_training = datetime.now()
        
        return session_stats