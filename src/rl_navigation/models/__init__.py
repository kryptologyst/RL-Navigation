"""Neural network models for RL Navigation."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Deep Q-Network for discrete action spaces.
    
    A simple feedforward neural network that maps observations to Q-values
    for each possible action.
    """
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: Tuple[int, ...] = (64, 64)):
        """Initialize the Q-network.
        
        Args:
            input_size: Size of input observation.
            output_size: Number of actions.
            hidden_sizes: Sizes of hidden layers.
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            
        Returns:
            Q-values tensor of shape (batch_size, output_size).
        """
        return self.network(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture.
    
    Separates value and advantage streams to better estimate Q-values,
    especially when actions don't significantly affect the environment.
    """
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: Tuple[int, ...] = (64, 64)):
        """Initialize the dueling Q-network.
        
        Args:
            input_size: Size of input observation.
            output_size: Number of actions.
            hidden_sizes: Sizes of hidden layers.
        """
        super().__init__()
        
        # Shared feature extraction
        shared_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes[:-1]:
            shared_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dueling network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            
        Returns:
            Q-values tensor of shape (batch_size, output_size).
        """
        features = self.shared(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class ActorCritic(nn.Module):
    """Actor-Critic network for continuous control.
    
    Combines actor (policy) and critic (value function) networks
    for policy gradient methods.
    """
    
    def __init__(
        self, 
        input_size: int, 
        action_size: int, 
        hidden_sizes: Tuple[int, ...] = (64, 64)
    ):
        """Initialize the actor-critic network.
        
        Args:
            input_size: Size of input observation.
            action_size: Number of action dimensions.
            hidden_sizes: Sizes of hidden layers.
        """
        super().__init__()
        
        # Shared feature extraction
        shared_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes[:-1]:
            shared_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], action_size),
            nn.Tanh()  # Output actions in [-1, 1]
        )
        
        # Critic (value function) head
        self.critic = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the actor-critic network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            
        Returns:
            Tuple of (actions, values).
        """
        features = self.shared(x)
        
        actions = self.actor(features)
        values = self.critic(features)
        
        return actions, values
    
    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Get action from the actor network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            
        Returns:
            Actions tensor of shape (batch_size, action_size).
        """
        return self.actor(self.shared(x))
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value from the critic network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            
        Returns:
            Values tensor of shape (batch_size, 1).
        """
        return self.critic(self.shared(x))
