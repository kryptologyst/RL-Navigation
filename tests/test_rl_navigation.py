"""Tests for RL Navigation package."""

import pytest
import numpy as np
import torch

from rl_navigation import NavigationEnv, DQN, set_seed, get_device


class TestNavigationEnv:
    """Test cases for NavigationEnv."""
    
    def test_env_creation(self):
        """Test environment creation."""
        env = NavigationEnv(grid_size=10, max_steps=50)
        assert env.grid_size == 10
        assert env.max_steps == 50
        assert env.action_space.n == 4
        assert env.observation_space.shape[0] == 4 + 10 * 10
    
    def test_reset(self):
        """Test environment reset."""
        env = NavigationEnv(seed=42)
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert 'agent_pos' in info
        assert 'goal_pos' in info
        assert 'step_count' in info
    
    def test_step(self):
        """Test environment step."""
        env = NavigationEnv(seed=42)
        obs, info = env.reset()
        
        action = 0  # Up
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    
    def test_episode_completion(self):
        """Test episode completion."""
        env = NavigationEnv(max_steps=5, seed=42)
        obs, info = env.reset()
        
        for _ in range(10):  # More than max_steps
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        assert terminated or truncated
    
    def test_render(self):
        """Test environment rendering."""
        env = NavigationEnv(render_mode="rgb_array", seed=42)
        obs, info = env.reset()
        
        # Test RGB array rendering
        img = env.render()
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3  # RGB channels
        
        env.close()


class TestDQN:
    """Test cases for DQN agent."""
    
    def test_agent_creation(self):
        """Test DQN agent creation."""
        agent = DQN(state_size=10, action_size=4)
        assert agent.state_size == 10
        assert agent.action_size == 4
        assert agent.device is not None
    
    def test_action_selection(self):
        """Test action selection."""
        agent = DQN(state_size=10, action_size=4)
        state = np.random.random(10)
        
        action = agent.select_action(state, training=True)
        assert isinstance(action, int)
        assert 0 <= action < 4
        
        action = agent.select_action(state, training=False)
        assert isinstance(action, int)
        assert 0 <= action < 4
    
    def test_experience_storage(self):
        """Test experience storage and training."""
        agent = DQN(state_size=10, action_size=4, batch_size=32)
        
        # Add experiences
        for _ in range(100):
            state = np.random.random(10)
            action = np.random.randint(4)
            reward = np.random.random()
            next_state = np.random.random(10)
            done = np.random.random() < 0.1
            
            loss = agent.step(state, action, reward, next_state, done)
            
            if loss is not None:
                assert isinstance(loss, float)
                assert loss >= 0
    
    def test_save_load(self):
        """Test model saving and loading."""
        agent = DQN(state_size=10, action_size=4)
        
        # Train a bit
        for _ in range(100):
            state = np.random.random(10)
            action = np.random.randint(4)
            reward = np.random.random()
            next_state = np.random.random(10)
            done = np.random.random() < 0.1
            
            agent.step(state, action, reward, next_state, done)
        
        # Save and load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            agent.save(f.name)
            
            # Create new agent and load
            new_agent = DQN(state_size=10, action_size=4)
            new_agent.load(f.name)
            
            # Test that they produce same actions
            state = np.random.random(10)
            action1 = agent.select_action(state, training=False)
            action2 = new_agent.select_action(state, training=False)
            
            assert action1 == action2


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy
        val1 = np.random.random()
        set_seed(42)
        val2 = np.random.random()
        assert val1 == val2
        
        # Test torch
        val1 = torch.rand(1).item()
        set_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_epsilon_schedule(self):
        """Test epsilon scheduling."""
        from rl_navigation.utils import epsilon_schedule
        
        # Test start value
        eps = epsilon_schedule(0, start_eps=1.0, end_eps=0.01, decay_episodes=100)
        assert eps == 1.0
        
        # Test end value
        eps = epsilon_schedule(100, start_eps=1.0, end_eps=0.01, decay_episodes=100)
        assert eps == 0.01
        
        # Test middle value
        eps = epsilon_schedule(50, start_eps=1.0, end_eps=0.01, decay_episodes=100)
        assert 0.01 < eps < 1.0
    
    def test_confidence_interval(self):
        """Test confidence interval computation."""
        from rl_navigation.utils import compute_confidence_interval
        
        data = np.random.normal(0, 1, 100)
        lower, upper = compute_confidence_interval(data)
        
        assert lower < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)


class TestIntegration:
    """Integration tests."""
    
    def test_training_loop(self):
        """Test complete training loop."""
        env = NavigationEnv(grid_size=10, max_steps=50, seed=42)
        agent = DQN(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            batch_size=32,
        )
        
        # Run a few episodes
        for episode in range(5):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(50):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                loss = agent.step(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            agent.update_episode()
            assert episode_reward is not None
        
        env.close()
    
    def test_evaluation(self):
        """Test evaluation without exploration."""
        env = NavigationEnv(grid_size=10, max_steps=50, seed=42)
        agent = DQN(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
        )
        
        # Train a bit
        for _ in range(100):
            state = np.random.random(env.observation_space.shape[0])
            action = np.random.randint(env.action_space.n)
            reward = np.random.random()
            next_state = np.random.random(env.observation_space.shape[0])
            done = np.random.random() < 0.1
            
            agent.step(state, action, reward, next_state, done)
        
        # Evaluate
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(50):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        assert episode_reward is not None
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
