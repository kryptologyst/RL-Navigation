#!/usr/bin/env python3
"""Example script demonstrating the refactored RL Navigation framework.

This script shows how to use the modernized RL Navigation package for
autonomous navigation tasks.

WARNING: This is a research/educational project. NOT FOR PRODUCTION CONTROL
OF REAL AUTONOMOUS SYSTEMS.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from rl_navigation import NavigationEnv, DQN, set_seed, get_device


def main():
    """Main example function."""
    parser = argparse.ArgumentParser(description="RL Navigation Example")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--grid-size", type=int, default=15, help="Grid size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    args = parser.parse_args()
    
    print("🤖 RL Navigation Example")
    print("=" * 50)
    print(f"Episodes: {args.episodes}")
    print(f"Grid size: {args.grid_size}×{args.grid_size}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device or 'auto-detect'}")
    print()
    
    # Set up device and seed
    device = torch.device(args.device) if args.device else get_device()
    set_seed(args.seed)
    
    print(f"Using device: {device}")
    
    # Create environment
    env = NavigationEnv(
        grid_size=args.grid_size,
        max_steps=args.grid_size * 2,
        obstacle_density=0.1,
        render_mode="rgb_array" if args.render else None,
        seed=args.seed,
    )
    
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    print()
    
    # Create agent
    agent = DQN(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=args.episodes // 2,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=50,
        device=device,
    )
    
    print("DQN agent created and ready for training!")
    print()
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()
    
    print("Starting training...")
    for episode in range(args.episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            loss = agent.step(state, action, reward, next_state, done)
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        agent.update_episode()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if episode % 20 == 0 or episode == args.episodes - 1:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_length = np.mean(episode_lengths[-20:])
            print(f"Episode {episode:3d}: Reward = {episode_reward:6.2f}, "
                  f"Avg Reward = {avg_reward:6.2f}, Length = {episode_length:3d}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # Final evaluation
    print("\nEvaluating trained agent...")
    eval_rewards = []
    eval_lengths = []
    
    for _ in range(10):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
    
    # Print results
    print("\nResults:")
    print("=" * 30)
    print(f"Training episodes: {args.episodes}")
    print(f"Final training reward: {episode_rewards[-1]:.2f}")
    print(f"Average training reward (last 20): {np.mean(episode_rewards[-20:]):.2f}")
    print(f"Average evaluation reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Average evaluation length: {np.mean(eval_lengths):.1f}")
    print(f"Success rate: {sum(r > 50 for r in eval_rewards) / len(eval_rewards):.1%}")
    
    # Save model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"dqn_example_{args.grid_size}x{args.grid_size}.pt"
    agent.save(str(checkpoint_path))
    print(f"\nModel saved to: {checkpoint_path}")
    
    # Demo episode
    if args.render:
        print("\nRunning demo episode...")
        state, info = env.reset()
        print(f"Start: {info['agent_pos']}, Goal: {info['goal_pos']}")
        
        step = 0
        while step < env.max_steps:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, "
                  f"Pos={step_info['agent_pos']}")
            
            if done:
                print(f"Episode completed! Total reward: {reward:.2f}")
                break
            
            state = next_state
            step += 1
    
    env.close()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
