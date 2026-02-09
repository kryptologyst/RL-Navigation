"""Experience replay buffer for RL Navigation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions.
    
    Stores (state, action, reward, next_state, done) transitions
    and provides random sampling for training.
    """
    
    def __init__(self, capacity: int, state_shape: Tuple[int, ...], device: torch.device):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
            state_shape: Shape of state observations.
            device: Device to store tensors on.
        """
        self.capacity = capacity
        self.device = device
        self.state_shape = state_shape
        
        # Initialize storage arrays
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.size = 0
        self.ptr = 0
    
    def add(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Add a transition to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.BoolTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer.
    
    Samples transitions with probability proportional to their TD error,
    giving higher priority to more "surprising" transitions.
    """
    
    def __init__(
        self, 
        capacity: int, 
        state_shape: Tuple[int, ...], 
        device: torch.device,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        """Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
            state_shape: Shape of state observations.
            device: Device to store tensors on.
            alpha: Prioritization exponent (0 = uniform sampling).
            beta: Importance sampling exponent (1 = full correction).
            beta_increment: Beta increment per sample.
        """
        super().__init__(capacity, state_shape, device)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Add a transition with maximum priority.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.
        """
        super().add(state, action, reward, next_state, done)
        
        # Set priority to maximum
        self.priorities[self.ptr - 1] = self.max_priority
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch using prioritized sampling.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights).
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get transitions
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.BoolTensor(self.dones[indices]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update.
            priorities: New priority values.
        """
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
