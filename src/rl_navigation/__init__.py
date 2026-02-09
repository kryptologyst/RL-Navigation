"""RL Navigation: Modern Reinforcement Learning for Autonomous Navigation.

This package provides a comprehensive framework for autonomous navigation using
reinforcement learning, featuring multiple algorithms, custom environments,
and evaluation tools.

WARNING: This is a research/educational project. NOT FOR PRODUCTION CONTROL
OF REAL AUTONOMOUS SYSTEMS.
"""

__version__ = "0.1.0"
__author__ = "RL Research Team"

from .envs import NavigationEnv
from .algorithms import DQN
from .utils import set_seed, get_device

__all__ = [
    "NavigationEnv",
    "DQN", 
    "set_seed",
    "get_device",
]
