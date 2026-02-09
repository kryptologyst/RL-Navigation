"""Utility functions for RL Navigation project."""

import random
from typing import Optional, Union

import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Set gymnasium seed
    gym.utils.seeding.np_random(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def normalize_rewards(rewards: np.ndarray, running_mean: Optional[float] = None, 
                     running_var: Optional[float] = None) -> tuple[np.ndarray, float, float]:
    """Normalize rewards using running statistics.
    
    Args:
        rewards: Array of rewards to normalize.
        running_mean: Current running mean (optional).
        running_var: Current running variance (optional).
        
    Returns:
        Tuple of (normalized_rewards, new_mean, new_var).
    """
    if running_mean is None:
        running_mean = np.mean(rewards)
    if running_var is None:
        running_var = np.var(rewards)
    
    # Update running statistics
    new_mean = 0.99 * running_mean + 0.01 * np.mean(rewards)
    new_var = 0.99 * running_var + 0.01 * np.var(rewards)
    
    # Normalize
    normalized = (rewards - new_mean) / (np.sqrt(new_var) + 1e-8)
    return normalized, new_mean, new_var


def epsilon_schedule(episode: int, start_eps: float = 1.0, end_eps: float = 0.01, 
                    decay_episodes: int = 1000) -> float:
    """Linear epsilon decay schedule.
    
    Args:
        episode: Current episode number.
        start_eps: Starting epsilon value.
        end_eps: Final epsilon value.
        decay_episodes: Number of episodes to decay over.
        
    Returns:
        Current epsilon value.
    """
    if episode >= decay_episodes:
        return end_eps
    return start_eps - (start_eps - end_eps) * episode / decay_episodes


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval for data.
    
    Args:
        data: Array of values.
        confidence: Confidence level (default 0.95 for 95% CI).
        
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(n)
    
    # Use t-distribution for small samples
    if n < 30:
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_val * std_err
    else:
        # Use normal distribution for large samples
        z_val = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z_val * std_err
    
    return mean - margin, mean + margin
