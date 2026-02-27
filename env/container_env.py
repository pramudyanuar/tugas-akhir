import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from height_map import HeightMap
from lbcp import is_stable
from action_mask import ActionMask

# Import dari parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset.random_generator import RandomGenerator


class ContainerEnv:
    """
    3D Container Loading Environment dengan action masking dan LBCP validation.
    """
    
    def __init__(self, container_length=59, container_width=23, container_height=23,
                 max_items=50, seed=None):
        """
        Initialize container environment.
        
        Args:
            container_length (int): Panjang container
            container_width (int): Lebar container
            container_height (int): Tinggi container
            max_items (int): Maksimum jumlah items per episode
            seed (int): Random seed untuk reproducibility
        """
        self.L = container_length
        self.W = container_width
        self.H = container_height
        self.max_items = max_items
        
        self.height_map = HeightMap(self.L, self.W, self.H)
        self.action_mask_calculator = ActionMask(self.L, self.W, self.H)
        self.dataset_generator = RandomGenerator(seed=seed)
        
        self.items = []
        self.current_index = 0
        self.container_volume = self.L * self.W * self.H
        
        # Tracking untuk metrics
        self.episode_reward = 0.0
        self.episode_length = 0
        self.placed_items = []
        self.placed_positions = []
        
        # State size: height_map (L*W) + item_dims (3)
        self.state_size = self.L * self.W + 3
        # Action size: positions (L*W) + skip action
        self.action_size = self.L * self.W + 1
    
    def reset(self, seed=None):
        """
        Reset environment dan generate new episode.
        
        Args:
            seed (int, optional): Seed untuk random generation
            
        Returns:
            tuple: (state, action_mask)
        """
        self.height_map.reset()
        
        if seed is not None:
            self.dataset_generator.set_seed(seed)
        
        # Generate episode items
        self.items = self.dataset_generator.generate_episode(num_items=self.max_items)
        self.current_index = 0
        
        # Reset metrics
        self.episode_reward = 0.0
        self.episode_length = 0
        self.placed_items = []
        self.placed_positions = []
        
        return self._get_state_and_mask()
    
    def _get_state_and_mask(self):
        """
        Get current state dan action mask.
        
        State format: [height_map.flatten(), item_length, item_width, item_height]
        
        Returns:
            tuple: (state, action_mask)
        """
        # Check if episode done
        if self.current_index >= len(self.items):
            # Return dummy state dan mask
            state = np.zeros(self.state_size, dtype=np.float32)
            action_mask = np.zeros(self.action_size, dtype=np.float32)
            return state, action_mask
        
        # Get current item
        item_l, item_w, item_h = self.items[self.current_index]
        
        # Create state: normalized height_map + item dims
        normalized_height = self.height_map.normalize().flatten()
        item_dims = np.array([item_l / self.L, item_w / self.W, item_h / self.H], 
                            dtype=np.float32)
        state = np.concatenate([normalized_height, item_dims])
        
        # Get action mask
        masking_result = self.action_mask_calculator.combine_masks(
            item_l, item_w, item_h, self.height_map
        )
        action_mask = self.action_mask_calculator.get_action_vector(
            item_l, item_w, item_h, self.height_map
        )
        
        return state, action_mask
    
    def step(self, action):
        """
        Execute action dalam environment.
        
        Args:
            action (int): Action index (0 to L*W-1 untuk placement, L*W untuk skip)
            
        Returns:
            tuple: (next_state, reward, done, info)
                - next_state: Next state dan mask
                - reward: Reward dari action
                - done: Episode done flag
                - info: Additional info (success, utilization, etc)
        """
        self.episode_length += 1
        
        # Check if episode done
        if self.current_index >= len(self.items):
            return self._get_state_and_mask(), 0.0, True, {'success': False}
        
        item_l, item_w, item_h = self.items[self.current_index]
        
        # Skip action (index == L*W)
        if action == self.L * self.W:
            # Skip to next item
            reward = 0.0
            self.current_index += 1
            done = self.current_index >= len(self.items)
            
            next_state, next_mask = self._get_state_and_mask()
            info = {'success': False, 'action_type': 'skip'}
            
            return (next_state, next_mask), reward, done, info
        
        # Position action
        x = action % self.L
        y = action // self.L
        
        # Check valid position
        if not self._is_valid_position(x, y, item_l, item_w, item_h):
            # Invalid placement, skip to next item
            reward = -0.1  # Penalty untuk invalid placement
            self.current_index += 1
            done = self.current_index >= len(self.items)
            
            next_state, next_mask = self._get_state_and_mask()
            info = {'success': False, 'action_type': 'invalid'}
            
            return (next_state, next_mask), reward, done, info
        
        # Place item
        base_height = self.height_map.max_height_in_region(x, y, item_l, item_w)
        new_height = base_height + item_h
        
        self.height_map.update_region(x, y, item_l, item_w, new_height)
        
        # Store placement info
        self.placed_items.append((item_l, item_w, item_h))
        self.placed_positions.append((x, y, base_height))
        
        # Calculate reward based on volume utilization
        item_volume = item_l * item_w * item_h
        reward = (item_volume / self.container_volume) * 10.0  # Scale rewards
        
        self.episode_reward += reward
        
        # Move to next item
        self.current_index += 1
        done = self.current_index >= len(self.items)
        
        next_state, next_mask = self._get_state_and_mask()
        info = {
            'success': True,
            'action_type': 'placement',
            'item_volume': item_volume,
            'position': (x, y, base_height)
        }
        
        return (next_state, next_mask), reward, done, info
    
    def _is_valid_position(self, x, y, item_l, item_w, item_h):
        """
        Check if position is valid (boundary + overflow + stability).
        
        Args:
            x, y: Position
            item_l, item_w, item_h: Item dimensions
            
        Returns:
            bool: True jika valid
        """
        # Check boundary
        if x + item_l > self.L or y + item_w > self.W:
            return False
        
        # Check overflow
        base_height = self.height_map.max_height_in_region(x, y, item_l, item_w)
        if base_height + item_h > self.H:
            return False
        
        # Check stability with LBCP
        try:
            if not is_stable(self.height_map.map, x, y, item_l, item_w, item_h, self.H):
                return False
        except Exception:
            return False
        
        return True
    
    def get_utilization(self):
        """
        Get container utilization percentage.
        
        Returns:
            float: Utilization percentage (0-100)
        """
        if not self.placed_items:
            return 0.0
        
        total_placed_volume = sum(item[0] * item[1] * item[2] 
                                 for item in self.placed_items)
        utilization = (total_placed_volume / self.container_volume) * 100.0
        
        return min(utilization, 100.0)
    
    def get_max_height(self):
        """Get maximum height in current container."""
        return int(np.max(self.height_map.map))
    
    def render(self):
        """Print container state."""
        print(f"Episode Length: {self.episode_length}")
        print(f"Items Placed: {len(self.placed_items)}/{len(self.items)}")
        print(f"Utilization: {self.get_utilization():.2f}%")
        print(f"Max Height: {self.get_max_height()}/{self.H}")
        print(f"Episode Reward: {self.episode_reward:.2f}")


if __name__ == "__main__":
    """Test cases untuk ContainerEnv"""
    
    print("=" * 70)
    print("Test Case 1: Environment Initialization")
    print("=" * 70)
    
    env = ContainerEnv(seed=42)
    state, action_mask = env.reset()
    
    print(f"State shape: {state.shape}")
    print(f"Action mask shape: {action_mask.shape}")
    print(f"Expected state size: {env.state_size}")
    print(f"Expected action size: {env.action_size}")
    print(f"Valid actions: {np.sum(action_mask > 0)}")
    
    assert state.shape == (env.state_size,), f"Wrong state shape! {state.shape}"
    assert action_mask.shape == (env.action_size,), f"Wrong action mask shape!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 2: Step with Valid Placement")
    print("=" * 70)
    
    env = ContainerEnv(seed=123)
    state, action_mask = env.reset()
    
    # Find first valid action
    valid_positions = np.where(action_mask > 0)[0]
    if len(valid_positions) > 0:
        action = valid_positions[0]
        (next_state, next_mask), reward, done, info = env.step(action)
        
        print(f"Action taken: {action}")
        print(f"Reward: {reward:.4f}")
        print(f"Done: {done}")
        print(f"Success: {info['success']}")
        print(f"Next state shape: {next_state.shape}")
        
        assert next_state.shape == (env.state_size,), "Wrong next state shape!"
        assert reward > 0, "Reward should be positive for valid placement!"
        print("✓ PASSED\n")
    else:
        print("No valid positions found\n")
    
    print("=" * 70)
    print("Test Case 3: Skip Action")
    print("=" * 70)
    
    env = ContainerEnv(seed=456)
    state, action_mask = env.reset()
    
    skip_action = env.L * env.W
    (next_state, next_mask), reward, done, info = env.step(skip_action)
    
    print(f"Skip action: {skip_action}")
    print(f"Reward: {reward:.4f}")
    print(f"Action type: {info['action_type']}")
    
    assert info['action_type'] == 'skip', "Should be skip action!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 4: Utilization Calculation")
    print("=" * 70)
    
    env = ContainerEnv(seed=789)
    state, action_mask = env.reset()
    
    # Place a few items
    for _ in range(3):
        valid_positions = np.where(action_mask > 0)[0]
        if len(valid_positions) > 0:
            action = valid_positions[0]
            (state, action_mask), reward, done, info = env.step(action)
            if done:
                break
        else:
            break
    
    utilization = env.get_utilization()
    max_height = env.get_max_height()
    
    print(f"Items placed: {len(env.placed_items)}")
    print(f"Utilization: {utilization:.2f}%")
    print(f"Max height: {max_height}")
    print(f"Episode reward: {env.episode_reward:.4f}")
    
    assert 0 <= utilization <= 100, "Utilization out of range!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 5: Full Episode")
    print("=" * 70)
    
    env = ContainerEnv(max_items=10, seed=999)
    state, action_mask = env.reset()
    
    step_count = 0
    while step_count < 100:  # Max 100 steps
        valid_positions = np.where(action_mask > 0)[0]
        
        if len(valid_positions) == 0:
            # No valid positions, must skip
            action = env.L * env.W
        else:
            # Take first valid action
            action = valid_positions[0]
        
        (state, action_mask), reward, done, info = env.step(action)
        step_count += 1
        
        if done:
            break
    
    print(f"Episode finished!")
    print(f"Steps taken: {step_count}")
    print(f"Items placed: {len(env.placed_items)}")
    print(f"Final utilization: {env.get_utilization():.2f}%")
    print(f"Final episode reward: {env.episode_reward:.4f}")
    
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("All ContainerEnv tests passed!")
    print("=" * 70)