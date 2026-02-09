"""Evaluation script for RL Navigation agents."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from rl_navigation import DQN, NavigationEnv, set_seed, get_device, compute_confidence_interval


def evaluate_agent(
    checkpoint_path: str,
    env_name: str = "NavigationEnv",
    algorithm: str = "DQN",
    num_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 42,
    device: Optional[str] = None,
    render: bool = False,
    save_trajectories: bool = False,
) -> Dict[str, float]:
    """Evaluate a trained RL agent.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint.
        env_name: Name of the environment.
        algorithm: RL algorithm used.
        num_episodes: Number of episodes to evaluate.
        max_steps: Maximum steps per episode.
        seed: Random seed for reproducibility.
        device: Device to run on.
        render: Whether to render episodes.
        save_trajectories: Whether to save trajectory data.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    # Set up device and seed
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    
    set_seed(seed)
    
    # Create environment
    if env_name == "NavigationEnv":
        env = NavigationEnv(max_steps=max_steps, seed=seed, render_mode="rgb_array" if render else None)
    else:
        env = gym.make(env_name, render_mode="rgb_array" if render else None)
    
    # Create agent
    if algorithm == "DQN":
        agent = DQN(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Load checkpoint
    agent.load(checkpoint_path)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = 0
    trajectories = [] if save_trajectories else None
    
    # Evaluation loop
    pbar = tqdm(range(num_episodes), desc="Evaluating")
    
    for episode in pbar:
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        trajectory = [] if save_trajectories else None
        
        while True:
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if save_trajectories:
                trajectory.append({
                    'state': state.copy(),
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done,
                    'info': step_info,
                })
            
            state = next_state
            
            if done:
                # Check if goal was reached
                if 'distance_to_goal' in step_info and step_info['distance_to_goal'] < 1.0:
                    success_rate += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if save_trajectories:
            trajectories.append(trajectory)
        
        # Update progress bar
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        current_success_rate = success_rate / (episode + 1)
        
        pbar.set_postfix({
            'Reward': f'{avg_reward:.2f}',
            'Length': f'{avg_length:.1f}',
            'Success': f'{current_success_rate:.2f}',
        })
    
    env.close()
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    ci_lower, ci_upper = compute_confidence_interval(episode_rewards)
    
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    success_rate = success_rate / num_episodes
    
    metrics = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'reward_ci_lower': ci_lower,
        'reward_ci_upper': ci_upper,
        'mean_length': mean_length,
        'std_length': std_length,
        'success_rate': success_rate,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }
    
    if save_trajectories:
        metrics['trajectories'] = trajectories
    
    return metrics


def run_ablation_study(
    checkpoint_path: str,
    env_name: str = "NavigationEnv",
    algorithm: str = "DQN",
    num_episodes: int = 50,
    max_steps: int = 200,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run ablation study with different environment configurations.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint.
        env_name: Name of the environment.
        algorithm: RL algorithm used.
        num_episodes: Number of episodes per configuration.
        max_steps: Maximum steps per episode.
        seed: Random seed for reproducibility.
        device: Device to run on.
        
    Returns:
        Dictionary containing results for each configuration.
    """
    results = {}
    
    # Different obstacle densities
    obstacle_densities = [0.05, 0.1, 0.15, 0.2]
    
    for density in obstacle_densities:
        print(f"\nEvaluating with obstacle density: {density}")
        
        # Create environment with specific density
        if env_name == "NavigationEnv":
            env = NavigationEnv(
                max_steps=max_steps, 
                obstacle_density=density,
                seed=seed
            )
        else:
            env = gym.make(env_name)
        
        # Evaluate
        metrics = evaluate_agent(
            checkpoint_path=checkpoint_path,
            env_name=env_name,
            algorithm=algorithm,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
            device=device,
        )
        
        results[f'density_{density}'] = {
            'mean_reward': metrics['mean_reward'],
            'std_reward': metrics['std_reward'],
            'success_rate': metrics['success_rate'],
            'mean_length': metrics['mean_length'],
        }
        
        env.close()
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RL Navigation Agent")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--env", type=str, default="NavigationEnv",
                       help="Environment name")
    parser.add_argument("--algorithm", type=str, default="DQN",
                       help="RL algorithm used")
    
    # Evaluation arguments
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run on")
    
    # Output arguments
    parser.add_argument("--render", action="store_true",
                       help="Render episodes")
    parser.add_argument("--save-trajectories", action="store_true",
                       help="Save trajectory data")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study")
    
    args = parser.parse_args()
    
    if args.ablation:
        # Run ablation study
        results = run_ablation_study(
            checkpoint_path=args.checkpoint,
            env_name=args.env,
            algorithm=args.algorithm,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            device=args.device,
        )
        
        print("\nAblation Study Results:")
        print("=" * 50)
        for config, metrics in results.items():
            print(f"{config}:")
            print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"  Success Rate: {metrics['success_rate']:.2f}")
            print(f"  Mean Length: {metrics['mean_length']:.1f}")
            print()
    
    else:
        # Standard evaluation
        metrics = evaluate_agent(
            checkpoint_path=args.checkpoint,
            env_name=args.env,
            algorithm=args.algorithm,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            device=args.device,
            render=args.render,
            save_trajectories=args.save_trajectories,
        )
        
        print("\nEvaluation Results:")
        print("=" * 30)
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"95% CI: [{metrics['reward_ci_lower']:.2f}, {metrics['reward_ci_upper']:.2f}]")
        print(f"Success Rate: {metrics['success_rate']:.2f}")
        print(f"Mean Episode Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
        
        # Save results
        if args.output:
            # Remove trajectories from saved results (too large)
            save_metrics = {k: v for k, v in metrics.items() if k != 'trajectories'}
            with open(args.output, 'w') as f:
                json.dump(save_metrics, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
