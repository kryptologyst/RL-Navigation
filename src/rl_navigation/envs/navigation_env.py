"""Custom navigation environment for autonomous navigation tasks."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NavigationEnv(gym.Env):
    """Custom 2D navigation environment with obstacles and goals.
    
    The agent must navigate from a starting position to a goal position
    while avoiding obstacles in a 2D grid world.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        grid_size: int = 20,
        max_steps: int = 200,
        obstacle_density: float = 0.1,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the navigation environment.
        
        Args:
            grid_size: Size of the square grid world.
            max_steps: Maximum number of steps per episode.
            obstacle_density: Fraction of grid cells that are obstacles.
            render_mode: Rendering mode ('human' or 'rgb_array').
            seed: Random seed for reproducibility.
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density
        self.render_mode = render_mode
        
        # Action space: 4 discrete actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [agent_x, agent_y, goal_x, goal_y, obstacle_map]
        # obstacle_map is flattened grid
        obs_size = 4 + grid_size * grid_size
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = None
        self.step_count = 0
        
        # For rendering
        self.screen = None
        self.clock = None
        
        self.reset(seed=seed)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset.
            
        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        
        # Generate obstacles
        self.obstacles = self._generate_obstacles()
        
        # Place agent and goal
        self.agent_pos = self._get_random_free_position()
        self.goal_pos = self._get_random_free_position()
        
        # Ensure agent and goal are different
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = self._get_random_free_position()
        
        self.step_count = 0
        
        obs = self._get_observation()
        info = {
            "agent_pos": self.agent_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "step_count": self.step_count,
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right).
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Move agent
        new_pos = self._move_agent(action)
        
        # Check if move is valid
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_observation()
        info = {
            "agent_pos": self.agent_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "step_count": self.step_count,
            "distance_to_goal": self._get_distance_to_goal(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            self._render_human()
        return None
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
    
    def _generate_obstacles(self) -> np.ndarray:
        """Generate random obstacles in the grid.
        
        Returns:
            Boolean array indicating obstacle positions.
        """
        obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density)
        
        # Randomly place obstacles
        positions = self.np_random.choice(
            self.grid_size * self.grid_size, 
            size=num_obstacles, 
            replace=False
        )
        
        for pos in positions:
            row, col = divmod(pos, self.grid_size)
            obstacles[row, col] = True
        
        return obstacles
    
    def _get_random_free_position(self) -> np.ndarray:
        """Get a random position that is not an obstacle.
        
        Returns:
            Array of [row, col] coordinates.
        """
        while True:
            row = self.np_random.integers(0, self.grid_size)
            col = self.np_random.integers(0, self.grid_size)
            if not self.obstacles[row, col]:
                return np.array([row, col])
    
    def _move_agent(self, action: int) -> np.ndarray:
        """Calculate new agent position after taking action.
        
        Args:
            action: Action to take.
            
        Returns:
            New agent position.
        """
        new_pos = self.agent_pos.copy()
        
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Down
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 2:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # Right
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        
        return new_pos
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is valid (not an obstacle).
        
        Args:
            pos: Position to check.
            
        Returns:
            True if position is valid.
        """
        row, col = pos
        return (0 <= row < self.grid_size and 
                0 <= col < self.grid_size and 
                not self.obstacles[row, col])
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state.
        
        Returns:
            Reward value.
        """
        # Check if reached goal
        if np.array_equal(self.agent_pos, self.goal_pos):
            return 100.0
        
        # Distance-based reward (encourage moving towards goal)
        distance = self._get_distance_to_goal()
        max_distance = math.sqrt(2) * self.grid_size
        distance_reward = (max_distance - distance) / max_distance
        
        # Small penalty for each step to encourage efficiency
        step_penalty = -0.1
        
        return distance_reward + step_penalty
    
    def _get_distance_to_goal(self) -> float:
        """Calculate Euclidean distance to goal.
        
        Returns:
            Distance to goal.
        """
        return np.linalg.norm(self.agent_pos - self.goal_pos)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation vector.
        """
        # Normalize positions to [0, 1]
        agent_norm = self.agent_pos / self.grid_size
        goal_norm = self.goal_pos / self.grid_size
        
        # Flatten obstacle map
        obstacle_flat = self.obstacles.flatten().astype(np.float32)
        
        # Combine all observations
        obs = np.concatenate([agent_norm, goal_norm, obstacle_flat])
        return obs.astype(np.float32)
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array.
        
        Returns:
            RGB array representation.
        """
        # Create RGB image
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Set obstacles (black)
        img[self.obstacles] = [0, 0, 0]
        
        # Set agent position (blue)
        img[self.agent_pos[0], self.agent_pos[1]] = [0, 0, 255]
        
        # Set goal position (green)
        img[self.goal_pos[0], self.goal_pos[1]] = [0, 255, 0]
        
        # Upscale for better visibility
        scale = 20
        img_upscaled = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        
        return img_upscaled
    
    def _render_human(self) -> None:
        """Render environment for human viewing."""
        try:
            import pygame
            
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((400, 400))
                self.clock = pygame.time.Clock()
            
            # Clear screen
            self.screen.fill((255, 255, 255))
            
            # Draw grid
            cell_size = 400 // self.grid_size
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    rect = pygame.Rect(col * cell_size, row * cell_size, 
                                     cell_size, cell_size)
                    
                    if self.obstacles[row, col]:
                        pygame.draw.rect(self.screen, (0, 0, 0), rect)
                    elif row == self.agent_pos[0] and col == self.agent_pos[1]:
                        pygame.draw.rect(self.screen, (0, 0, 255), rect)
                    elif row == self.goal_pos[0] and col == self.goal_pos[1]:
                        pygame.draw.rect(self.screen, (0, 255, 0), rect)
                    else:
                        pygame.draw.rect(self.screen, (255, 255, 255), rect)
                    
                    pygame.draw.rect(self.screen, (128, 128, 128), rect, 1)
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
        except ImportError:
            # Fallback to ASCII rendering
            self._render_ascii()
    
    def _render_ascii(self) -> None:
        """ASCII rendering fallback."""
        print("\n" + "=" * (self.grid_size + 2))
        for row in range(self.grid_size):
            line = "|"
            for col in range(self.grid_size):
                if self.obstacles[row, col]:
                    line += "#"
                elif row == self.agent_pos[0] and col == self.agent_pos[1]:
                    line += "A"
                elif row == self.goal_pos[0] and col == self.goal_pos[1]:
                    line += "G"
                else:
                    line += " "
            line += "|"
            print(line)
        print("=" * (self.grid_size + 2))
