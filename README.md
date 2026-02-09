# RL Navigation: Modern Reinforcement Learning for Autonomous Navigation

A comprehensive framework for autonomous navigation using reinforcement learning, featuring multiple algorithms, custom environments, and evaluation tools.

**WARNING**: This is a research/educational project. NOT FOR PRODUCTION CONTROL OF REAL AUTONOMOUS SYSTEMS.

## Features

- **Custom Navigation Environment**: 2D grid world with obstacles and goals
- **Modern RL Algorithms**: DQN with experience replay and target networks
- **Comprehensive Evaluation**: Metrics, confidence intervals, and ablation studies
- **Interactive Demo**: Streamlit-based visualization and policy evaluation
- **Production-Ready Structure**: Clean code, type hints, tests, and documentation

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon acceleration)

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install with pip
pip install -e .

# For development
pip install -e ".[dev]"

# For advanced features
pip install -e ".[advanced]"
```

## Quick Start

### 1. Train an Agent

```bash
# Basic training
python scripts/train.py

# Custom configuration
python scripts/train.py --episodes 2000 --learning-rate 0.0005 --grid-size 25

# Fast training for testing
python scripts/train.py --config configs/fast.yaml
```

### 2. Evaluate the Agent

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/DQN_NavigationEnv_final.pt

# Run ablation study
python scripts/evaluate.py --checkpoint checkpoints/DQN_NavigationEnv_final.pt --ablation

# Save trajectory data
python scripts/evaluate.py --checkpoint checkpoints/DQN_NavigationEnv_final.pt --save-trajectories
```

### 3. Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Environment Description

The `NavigationEnv` is a custom 2D grid world environment where an agent must navigate from a starting position to a goal position while avoiding obstacles.

### Observation Space

- **Shape**: `(4 + grid_size²,)`
- **Components**:
  - Agent position (normalized): `[agent_x, agent_y]`
  - Goal position (normalized): `[goal_x, goal_y]`
  - Obstacle map (flattened): `grid_size × grid_size` binary values

### Action Space

- **Type**: Discrete
- **Size**: 4 actions
- **Actions**: 0=Up, 1=Down, 2=Left, 3=Right

### Reward Function

- **Goal reached**: +100.0
- **Distance-based reward**: Encourages moving towards goal
- **Step penalty**: -0.1 per step (encourages efficiency)

### Environment Parameters

- `grid_size`: Size of the square grid (default: 20)
- `max_steps`: Maximum steps per episode (default: 200)
- `obstacle_density`: Fraction of cells that are obstacles (default: 0.1)

## Algorithms

### Deep Q-Network (DQN)

Implementation of the DQN algorithm with:

- **Experience Replay**: Stores and samples past experiences
- **Target Network**: Stabilizes learning with delayed updates
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Double DQN**: Reduces overestimation bias (optional)

#### Key Parameters

- `learning_rate`: Learning rate for optimizer (default: 1e-3)
- `gamma`: Discount factor (default: 0.99)
- `epsilon_start`: Starting exploration rate (default: 1.0)
- `epsilon_end`: Final exploration rate (default: 0.01)
- `epsilon_decay`: Episodes over which to decay epsilon (default: 500)
- `buffer_size`: Size of experience replay buffer (default: 100000)
- `batch_size`: Batch size for training (default: 64)
- `target_update_freq`: Frequency to update target network (default: 100)

## Evaluation Metrics

### Learning Metrics

- **Average Return**: Mean episode reward ± 95% confidence interval
- **Success Rate**: Fraction of episodes reaching the goal
- **Sample Efficiency**: Steps to reach performance threshold
- **Episode Length**: Average steps per episode

### Stability Metrics

- **Reward Variance**: Consistency across episodes
- **Learning Curve**: Performance over training episodes
- **Convergence**: Stability of final performance

### Ablation Studies

- **Obstacle Density**: Performance across different obstacle densities
- **Grid Size**: Scalability to different environment sizes
- **Algorithm Components**: Impact of individual components

## Project Structure

```
rl-navigation/
├── src/rl_navigation/          # Main package
│   ├── algorithms/             # RL algorithms
│   ├── buffers/                # Experience replay buffers
│   ├── envs/                   # Custom environments
│   ├── models/                 # Neural network models
│   ├── utils/                  # Utility functions
│   └── __init__.py
├── configs/                    # Configuration files
├── scripts/                    # Training and evaluation scripts
├── tests/                      # Test suite
├── demo/                       # Streamlit demo
├── assets/                     # Generated plots and videos
├── checkpoints/               # Saved models
├── logs/                      # Training logs
└── README.md
```

## Configuration

### YAML Configuration

Use YAML files for reproducible experiments:

```yaml
# configs/default.yaml
env:
  name: "NavigationEnv"
  grid_size: 20
  max_steps: 200
  obstacle_density: 0.1

algorithm:
  name: "DQN"
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01

training:
  num_episodes: 1000
  eval_freq: 100
  save_freq: 500
```

### Command Line Arguments

All scripts support extensive command-line configuration:

```bash
python scripts/train.py --help
python scripts/evaluate.py --help
```

## Development

### Code Quality

```bash
# Format code
black src/ scripts/ tests/

# Lint code
ruff check src/ scripts/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Results

### Expected Performance

With default configuration on NavigationEnv (20×20 grid, 10% obstacles):

- **Training Episodes**: 1000
- **Final Success Rate**: >80%
- **Average Return**: >50
- **Convergence**: ~500 episodes

### Performance Variations

- **Grid Size**: Larger grids require more training
- **Obstacle Density**: Higher density reduces success rate
- **Algorithm**: DQN provides good baseline performance

## Safety and Limitations

### Safety Disclaimers

- **NOT FOR PRODUCTION**: This is a research/educational project
- **NO REAL-WORLD CONTROL**: Do not use for actual autonomous systems
- **SIMULATION ONLY**: All environments are simulated
- **RESEARCH PURPOSE**: Intended for learning and experimentation

### Known Limitations

- **Discrete Actions**: Only supports discrete action spaces
- **Grid World**: Limited to 2D grid-based navigation
- **Static Environment**: Obstacles and goals are fixed per episode
- **Simple Reward**: Basic reward function may not capture all objectives

### Future Improvements

- **Continuous Control**: Support for continuous action spaces
- **Dynamic Environments**: Moving obstacles and changing goals
- **Multi-Agent**: Support for multiple agents
- **Advanced Algorithms**: PPO, SAC, and other modern algorithms
- **Real-World Integration**: Interface with real robotics platforms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{rl_navigation,
  title={RL Navigation: Modern Reinforcement Learning for Autonomous Navigation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/RL-Navigation}
}
```

## Acknowledgments

- OpenAI Gym/Gymnasium for the environment interface
- PyTorch for the deep learning framework
- Streamlit for the interactive demo
- The RL research community for algorithms and insights
# RL-Navigation
