"""Streamlit demo for RL Navigation."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rl_navigation import DQN, NavigationEnv, set_seed, get_device


def load_agent(checkpoint_path: str, env: NavigationEnv) -> DQN:
    """Load a trained agent from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        env: Environment instance.
        
    Returns:
        Loaded DQN agent.
    """
    device = get_device()
    agent = DQN(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        device=device,
    )
    agent.load(checkpoint_path)
    return agent


def run_episode(agent: DQN, env: NavigationEnv, render: bool = False) -> Dict:
    """Run a single episode and collect data.
    
    Args:
        agent: Trained RL agent.
        env: Environment instance.
        render: Whether to render the episode.
        
    Returns:
        Episode data dictionary.
    """
    state, info = env.reset()
    episode_data = {
        'states': [state.copy()],
        'actions': [],
        'rewards': [],
        'positions': [info['agent_pos'].copy()],
        'goal_pos': info['goal_pos'].copy(),
        'total_reward': 0,
        'length': 0,
        'success': False,
    }
    
    while True:
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        episode_data['states'].append(next_state.copy())
        episode_data['positions'].append(step_info['agent_pos'].copy())
        episode_data['total_reward'] += reward
        episode_data['length'] += 1
        
        if done:
            episode_data['success'] = step_info.get('distance_to_goal', 1.0) < 1.0
            break
        
        state = next_state
    
    return episode_data


def create_trajectory_plot(episode_data: Dict, grid_size: int = 20) -> go.Figure:
    """Create a plot showing the agent's trajectory.
    
    Args:
        episode_data: Episode data dictionary.
        grid_size: Size of the grid.
        
    Returns:
        Plotly figure.
    """
    positions = np.array(episode_data['positions'])
    goal_pos = episode_data['goal_pos']
    
    fig = go.Figure()
    
    # Add trajectory
    fig.add_trace(go.Scatter(
        x=positions[:, 1],
        y=positions[:, 0],
        mode='lines+markers',
        name='Trajectory',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
    ))
    
    # Add start position
    fig.add_trace(go.Scatter(
        x=[positions[0, 1]],
        y=[positions[0, 0]],
        mode='markers',
        name='Start',
        marker=dict(color='green', size=12, symbol='circle'),
    ))
    
    # Add goal position
    fig.add_trace(go.Scatter(
        x=[goal_pos[1]],
        y=[goal_pos[0]],
        mode='markers',
        name='Goal',
        marker=dict(color='red', size=12, symbol='star'),
    ))
    
    fig.update_layout(
        title='Agent Trajectory',
        xaxis_title='X Position',
        yaxis_title='Y Position',
        xaxis=dict(range=[-0.5, grid_size - 0.5]),
        yaxis=dict(range=[-0.5, grid_size - 0.5]),
        width=500,
        height=500,
    )
    
    return fig


def create_reward_plot(episode_data: Dict) -> go.Figure:
    """Create a plot showing rewards over time.
    
    Args:
        episode_data: Episode data dictionary.
        
    Returns:
        Plotly figure.
    """
    rewards = episode_data['rewards']
    cumulative_rewards = np.cumsum(rewards)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Reward per Step', 'Cumulative Reward'),
        vertical_spacing=0.1,
    )
    
    # Reward per step
    fig.add_trace(
        go.Scatter(
            y=rewards,
            mode='lines+markers',
            name='Step Reward',
            line=dict(color='blue'),
        ),
        row=1, col=1,
    )
    
    # Cumulative reward
    fig.add_trace(
        go.Scatter(
            y=cumulative_rewards,
            mode='lines',
            name='Cumulative Reward',
            line=dict(color='green'),
        ),
        row=2, col=1,
    )
    
    fig.update_layout(
        title='Reward Analysis',
        height=400,
        showlegend=False,
    )
    
    fig.update_xaxes(title_text='Step', row=2, col=1)
    fig.update_yaxes(title_text='Reward', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Reward', row=2, col=1)
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RL Navigation Demo",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("🤖 RL Navigation Demo")
    st.markdown("""
    This demo showcases a reinforcement learning agent trained for autonomous navigation.
    The agent learns to navigate from a starting position to a goal while avoiding obstacles.
    
    **WARNING**: This is a research/educational demo. NOT FOR PRODUCTION CONTROL OF REAL AUTONOMOUS SYSTEMS.
    """)
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Environment parameters
    grid_size = st.sidebar.slider("Grid Size", 10, 30, 20)
    obstacle_density = st.sidebar.slider("Obstacle Density", 0.05, 0.3, 0.1)
    max_steps = st.sidebar.slider("Max Steps", 50, 500, 200)
    
    # Evaluation parameters
    num_episodes = st.sidebar.slider("Evaluation Episodes", 1, 50, 10)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
    
    # Model selection
    st.sidebar.header("Model")
    checkpoint_dir = Path("checkpoints")
    checkpoint_files = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
    
    if checkpoint_files:
        checkpoint_names = [f.name for f in checkpoint_files]
        selected_checkpoint = st.sidebar.selectbox(
            "Select Checkpoint",
            checkpoint_names,
            index=0,
        )
        checkpoint_path = checkpoint_dir / selected_checkpoint
    else:
        st.sidebar.error("No checkpoints found. Please train a model first.")
        st.stop()
    
    # Create environment
    set_seed(seed)
    env = NavigationEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        obstacle_density=obstacle_density,
        seed=seed,
    )
    
    # Load agent
    try:
        agent = load_agent(str(checkpoint_path), env)
        st.sidebar.success(f"Loaded model: {selected_checkpoint}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.stop()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Live Evaluation")
        
        if st.button("Run Episode", type="primary"):
            with st.spinner("Running episode..."):
                episode_data = run_episode(agent, env)
                
                # Display episode results
                col1_1, col1_2, col1_3 = st.columns(3)
                with col1_1:
                    st.metric("Total Reward", f"{episode_data['total_reward']:.2f}")
                with col1_2:
                    st.metric("Episode Length", episode_data['length'])
                with col1_3:
                    st.metric("Success", "✅" if episode_data['success'] else "❌")
                
                # Create plots
                trajectory_fig = create_trajectory_plot(episode_data, grid_size)
                reward_fig = create_reward_plot(episode_data)
                
                st.plotly_chart(trajectory_fig, use_container_width=True)
                st.plotly_chart(reward_fig, use_container_width=True)
    
    with col2:
        st.header("Batch Evaluation")
        
        if st.button("Run Batch Evaluation"):
            with st.spinner(f"Running {num_episodes} episodes..."):
                episode_rewards = []
                episode_lengths = []
                success_count = 0
                
                for _ in range(num_episodes):
                    episode_data = run_episode(agent, env)
                    episode_rewards.append(episode_data['total_reward'])
                    episode_lengths.append(episode_data['length'])
                    if episode_data['success']:
                        success_count += 1
                
                # Display batch results
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_length = np.mean(episode_lengths)
                success_rate = success_count / num_episodes
                
                st.metric("Mean Reward", f"{mean_reward:.2f} ± {std_reward:.2f}")
                st.metric("Mean Length", f"{mean_length:.1f}")
                st.metric("Success Rate", f"{success_rate:.2f}")
                
                # Create distribution plot
                fig = px.histogram(
                    x=episode_rewards,
                    nbins=20,
                    title="Reward Distribution",
                    labels={'x': 'Total Reward', 'y': 'Count'},
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Environment info
    st.header("Environment Information")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Observation Space")
        st.write(f"Shape: {env.observation_space.shape}")
        st.write(f"Type: {env.observation_space.dtype}")
        
        st.subheader("Action Space")
        st.write(f"Size: {env.action_space.n}")
        st.write("Actions: 0=Up, 1=Down, 2=Left, 3=Right")
    
    with col4:
        st.subheader("Environment Parameters")
        st.write(f"Grid Size: {grid_size}×{grid_size}")
        st.write(f"Obstacle Density: {obstacle_density:.1%}")
        st.write(f"Max Steps: {max_steps}")
        st.write(f"Random Seed: {seed}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Demo:**
    - Built with Streamlit and PyTorch
    - Uses a custom navigation environment
    - Implements Deep Q-Network (DQN) algorithm
    - Features experience replay and target networks
    """)
    
    env.close()


if __name__ == "__main__":
    main()
