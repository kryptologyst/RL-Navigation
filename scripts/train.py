"""Training script for RL Navigation agents."""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from rl_navigation import DQN, NavigationEnv, set_seed, get_device


def train_agent(
    env_name: str = "NavigationEnv",
    algorithm: str = "DQN",
    num_episodes: int = 1000,
    max_steps: int = 200,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: int = 500,
    buffer_size: int = 100000,
    batch_size: int = 64,
    target_update_freq: int = 100,
    eval_freq: int = 100,
    save_freq: int = 500,
    seed: int = 42,
    device: Optional[str] = None,
    save_dir: str = "checkpoints",
    log_dir: str = "logs",
) -> Dict[str, List[float]]:
    """Train a RL agent on the navigation environment.
    
    Args:
        env_name: Name of the environment.
        algorithm: RL algorithm to use.
        num_episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        learning_rate: Learning rate for the agent.
        gamma: Discount factor.
        epsilon_start: Starting epsilon for exploration.
        epsilon_end: Final epsilon for exploration.
        epsilon_decay: Episodes over which to decay epsilon.
        buffer_size: Size of experience replay buffer.
        batch_size: Batch size for training.
        target_update_freq: Frequency to update target network.
        eval_freq: Frequency to evaluate the agent.
        save_freq: Frequency to save checkpoints.
        seed: Random seed for reproducibility.
        device: Device to run on ('cuda', 'mps', 'cpu').
        save_dir: Directory to save checkpoints.
        log_dir: Directory to save logs.
        
    Returns:
        Dictionary containing training metrics.
    """
    # Set up device and seed
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    
    set_seed(seed)
    
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create environment
    if env_name == "NavigationEnv":
        env = NavigationEnv(max_steps=max_steps, seed=seed)
    else:
        env = gym.make(env_name)
    
    # Create agent
    if algorithm == "DQN":
        agent = DQN(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    eval_rewards = []
    
    # Training loop
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience and train
            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Update episode count for epsilon scheduling
        agent.update_episode()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if loss_count > 0:
            episode_losses.append(episode_loss / loss_count)
        else:
            episode_losses.append(0)
        
        # Update progress bar
        avg_reward = np.mean(episode_rewards[-100:])
        avg_length = np.mean(episode_lengths[-100:])
        avg_loss = np.mean(episode_losses[-100:]) if episode_losses else 0
        
        pbar.set_postfix({
            'Reward': f'{avg_reward:.2f}',
            'Length': f'{avg_length:.1f}',
            'Loss': f'{avg_loss:.4f}',
        })
        
        # Evaluation
        if episode % eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=10)
            eval_rewards.append(eval_reward)
            tqdm.write(f"Episode {episode}: Eval Reward = {eval_reward:.2f}")
        
        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_path = Path(save_dir) / f"{algorithm}_{env_name}_ep{episode}.pt"
            agent.save(str(checkpoint_path))
    
    # Final evaluation
    final_eval_reward = evaluate_agent(agent, env, num_episodes=100)
    eval_rewards.append(final_eval_reward)
    
    # Save final model
    final_path = Path(save_dir) / f"{algorithm}_{env_name}_final.pt"
    agent.save(str(final_path))
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_losses': episode_losses,
        'eval_rewards': eval_rewards,
    }


def evaluate_agent(agent, env, num_episodes: int = 10) -> float:
    """Evaluate the agent's performance.
    
    Args:
        agent: Trained RL agent.
        env: Environment to evaluate on.
        num_episodes: Number of episodes to evaluate.
        
    Returns:
        Average reward over evaluation episodes.
    """
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL Navigation Agent")
    
    # Environment arguments
    parser.add_argument("--env", type=str, default="NavigationEnv", 
                       help="Environment name")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode")
    
    # Algorithm arguments
    parser.add_argument("--algorithm", type=str, default="DQN",
                       help="RL algorithm to use")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                       help="Starting epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                       help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=int, default=500,
                       help="Epsilon decay episodes")
    
    # Training arguments
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--buffer-size", type=int, default=100000,
                       help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--target-update-freq", type=int, default=100,
                       help="Target network update frequency")
    
    # Evaluation and saving
    parser.add_argument("--eval-freq", type=int, default=100,
                       help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=500,
                       help="Save frequency")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory to save logs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Train agent
    metrics = train_agent(
        env_name=args.env,
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
    
    print(f"\nTraining completed!")
    print(f"Final evaluation reward: {metrics['eval_rewards'][-1]:.2f}")
    print(f"Average training reward (last 100 episodes): {np.mean(metrics['episode_rewards'][-100:]):.2f}")


if __name__ == "__main__":
    main()
