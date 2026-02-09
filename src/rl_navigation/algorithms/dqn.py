"""DQN algorithm implementation for RL Navigation."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..buffers import ReplayBuffer
from ..models import QNetwork
from ..utils import epsilon_schedule, get_device


class DQN:
    """Deep Q-Network agent for discrete action spaces.
    
    Implements the DQN algorithm with experience replay and target network
    for stable learning in autonomous navigation tasks.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[torch.device] = None,
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ):
        """Initialize the DQN agent.
        
        Args:
            state_size: Size of state space.
            action_size: Size of action space.
            learning_rate: Learning rate for optimizer.
            gamma: Discount factor.
            epsilon_start: Starting epsilon for exploration.
            epsilon_end: Final epsilon for exploration.
            epsilon_decay: Episodes over which to decay epsilon.
            buffer_size: Size of experience replay buffer.
            batch_size: Batch size for training.
            target_update_freq: Frequency to update target network.
            device: Device to run on.
            hidden_sizes: Hidden layer sizes for Q-network.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = device or get_device()
        
        # Initialize networks
        self.q_network = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size, (state_size,), self.device)
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state.
            training: Whether in training mode (affects exploration).
            
        Returns:
            Selected action.
        """
        if training:
            epsilon = epsilon_schedule(
                self.episode_count, 
                self.epsilon_start, 
                self.epsilon_end, 
                self.epsilon_decay
            )
            
            if np.random.random() < epsilon:
                return np.random.choice(self.action_size)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> Optional[float]:
        """Store experience and train if enough samples available.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.
            
        Returns:
            Loss value if training occurred, None otherwise.
        """
        # Store experience
        self.memory.add(state, action, reward, next_state, done)
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            return self.train()
        
        return None
    
    def train(self) -> float:
        """Train the Q-network on a batch of experiences.
        
        Returns:
            Training loss.
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def update_episode(self) -> None:
        """Update episode count for epsilon scheduling."""
        self.episode_count += 1
    
    def save(self, filepath: str) -> None:
        """Save the agent's state.
        
        Args:
            filepath: Path to save the model.
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's state.
        
        Args:
            filepath: Path to load the model from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
