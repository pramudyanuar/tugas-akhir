import numpy as np
import sys
import os

# Use relative imports for clean module structure
from .height_map import HeightMap
from .lbcp import is_stable
from .action_mask import ActionMask
from .feasibility_map import FeasibilityMap

# Import dari parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.random_generator import RandomGenerator
from src.data.cutting_stock import CuttingStockGenerator
from src.data.perfect_pack_generator import PerfectPackGenerator


class ContainerEnv:
    """
    3D Container Loading Environment dengan action masking dan LBCP validation.
    """
    
    def __init__(self, container_length=60, container_width=24, container_height=26,
                 max_items=50, seed=None, dataset_type='random'):
        """
        Initialize container environment.
        
        Args:
            container_length (int): Panjang container (default: 60 = 6m / 20ft)
            container_width (int): Lebar container (default: 24 = 2.4m / 8ft)
            container_height (int): Tinggi container (default: 26 = 2.6m / 8.5ft)
            max_items (int): Maksimum jumlah items per episode
            seed (int): Random seed untuk reproducibility
            dataset_type (str): 'random', 'cutting_stock', atau 'perfect_pack'
        """
        self.L = container_length
        self.W = container_width
        self.H = container_height
        self.max_items = max_items
        self.dataset_type = dataset_type
        
        self.height_map = HeightMap(self.L, self.W, self.H)
        self.feasibility_map = FeasibilityMap(self.L, self.W)
        self.action_mask_calculator = ActionMask(self.L, self.W, self.H)
        
        # Initialize dataset generator based pada type
        if dataset_type == 'cutting_stock':
            self.dataset_generator = CuttingStockGenerator(seed=seed)
        elif dataset_type == 'perfect_pack':
            self.dataset_generator = PerfectPackGenerator(
                bin_width=self.W,
                bin_height=self.H,
                sigma=2,
                seed=seed
            )
        else:
            self.dataset_generator = RandomGenerator(seed=seed)
        
        self.items = []
        self.current_index = 0
        self.container_volume = self.L * self.W * self.H
        
        # Tracking untuk metrics
        self.episode_reward = 0.0
        self.episode_length = 0
        self.placed_items = []
        self.placed_positions = []
        
        # State size: height_map (L*W) + item_dims (3) + min_height_info (1)
        self.state_size = self.L * self.W + 3 + 1
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
        self.feasibility_map.reset()
        
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
        
        State format: [height_map.flatten(), item_length, item_width, item_height, min_available_height]
        
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
        
        # Create state: normalized height_map + item dims + min height info
        normalized_height = self.height_map.normalize().flatten()
        item_dims = np.array([item_l / self.L, item_w / self.W, item_h / self.H], 
                            dtype=np.float32)
        
        # Option 3: Add min_height_info to encourage bottom-up filling
        # This helps the network learn to prefer lower positions
        min_height = np.min(self.height_map.map) / self.H  # Normalized
        min_height_info = np.array([min_height], dtype=np.float32)
        
        state = np.concatenate([normalized_height, item_dims, min_height_info])
        
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
        
        # Option 1: Enhanced reward function for efficient packing
        # Encourages both volume utilization AND bottom-up filling AND back-to-front access
        item_volume = item_l * item_w * item_h
        
        # Current container utilization
        total_placed_volume = sum(item[0] * item[1] * item[2] 
                                 for item in self.placed_items)
        current_utilization = total_placed_volume / self.container_volume
        
        # Height efficiency penalty: penalize stacking too high
        max_height_now = np.max(self.height_map.map)
        height_efficiency = 1.0 - (max_height_now / self.H)  # 1.0 = low, 0.0 = high
        
        # Placement quality: prefer placing at lower heights (bottom-up)
        placement_height_ratio = base_height / self.H
        height_penalty = placement_height_ratio * 0.1  # Slight penalty for high placement
        
        # Back-to-front preference: reward placing items deeper (higher y = further from door)
        # This keeps front area accessible for unloading
        back_placement_ratio = y / self.W  # 0.0 = front, 1.0 = back
        back_bonus = back_placement_ratio * 0.5  # Bonus for back placement
        
        # Combined reward
        volume_reward = (item_volume / self.container_volume) * 8.0  # 80% weight
        utilization_bonus = current_utilization * 2.0  # Encourage filling efficiently
        height_bonus = height_efficiency * 1.0  # Encourage spreading vertically
        
        reward = volume_reward + utilization_bonus + height_bonus + back_bonus - height_penalty
        
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
        Check if position is valid (boundary + overflow + stability + passway accessibility).
        
        Args:
            x, y: Position
            item_l, item_w, item_h: Item dimensions
            
        Returns:
            bool: True jika valid
        """
        # Door constraint: pintu ada di depan (y=0), min clearance
        # Tapi yang penting: jangan block akses ke belakang
        # Strategy: prefer placements di BELAKANG (tinggi y) untuk menjaga akses depan
        
        PASSWAY_MINIMUM = 2  # Minimum width passway untuk akses
        
        # Check boundary
        if x + item_l > self.L or y + item_w > self.W:
            return False
        
        # Check overflow
        base_height = self.height_map.max_height_in_region(x, y, item_l, item_w)
        if base_height + item_h > self.H:
            return False
        
        # Passway accessibility: jika item di depan (y < W-PASSWAY_MINIMUM)
        # pastikan ada minimal 1 clear path untuk akses belakang
        if y < (self.W - PASSWAY_MINIMUM):
            # Item di area depan - check jangan sepenuhnya block passway
            # Simplication: allow if item doesn't span entire width
            # atau ada corridor yang tetap terbuka
            if item_w >= (self.W - PASSWAY_MINIMUM):
                # Item ini akan block passway sepenuhnya - REJECT
                return False
        
        # Check stability with LBCP
        # Lantai (base_height=0) selalu support, items di atas harus penuhi LBCP strict
        try:
            if base_height == 0:
                # Item di atas lantai - always stable (lantai adalah support sempurna)
                pass
            else:
                # Item berlapis di atas items lain - check dengan strict LBCP
                if not is_stable(self.height_map.map, x, y, item_l, item_w, item_h, self.H, strict_mode=True):
                    return False
        except Exception:
            # Jika ada error di strict check, reject untuk safety
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

    print("=" * 70)