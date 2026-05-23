import numpy as np
import sys
import os

# Use relative imports for clean module structure
from .height_map import HeightMap
from .lbcp import is_stable, validate_structural_stability, update_feasibility_map
from .action_mask import ActionMask

# Import dari parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.random_generator import RandomGenerator
from src.data.cutting_stock import CuttingStockGenerator
from src.data.perfect_pack_generator import PerfectPackGenerator
from src.planning.repack_trial import RepackTrial
from src.utils.item_utils import get_item_dims, get_item_stacking, make_item


class ContainerEnv:
    """
    3D Container Loading Environment dengan action masking dan LBCP validation.
    """
    
    def __init__(self, container_length=60, container_width=24, container_height=26,
                 max_items=50, seed=None, dataset_type='random',
                 use_structural_validation=True, cog_tolerance=0.15,
                 layered_min_height=2, layered_max_height=6,
                 perfect_pack_sigma=4, perfect_pack_size_bias=3.0,
                 perfect_pack_mean_ratio=0.25, fast_stability_mask=False,
                 max_episode_length=500):
        """
        Initialize container environment.
        
        Args:
            container_length (int): Panjang container (default: 60 = 6m / 20ft)
            container_width (int): Lebar container (default: 24 = 2.4m / 8ft)
            container_height (int): Tinggi container (default: 26 = 2.6m / 8.5ft)
            max_items (int): Maksimum jumlah items per episode
            seed (int): Random seed untuk reproducibility
            dataset_type (str): 'random', 'cutting_stock', 'perfect_pack',
                                atau 'perfect_pack_layered'
        """
        self.L = container_length
        self.W = container_width
        self.H = container_height
        self.max_items = max_items
        self.max_episode_length = max(100, int(max_episode_length))  # Minimum 100 steps per episode
        self.dataset_type = dataset_type
        self.use_structural_validation = use_structural_validation
        self.cog_tolerance = cog_tolerance
        self.fast_stability_mask = bool(fast_stability_mask)
        self.layered_min_height = max(1, int(layered_min_height))
        self.layered_max_height = max(self.layered_min_height, int(layered_max_height))
        self.perfect_pack_sigma = perfect_pack_sigma
        self.perfect_pack_size_bias = perfect_pack_size_bias
        self.perfect_pack_mean_ratio = perfect_pack_mean_ratio
        self.invalid_penalty = -0.2
        self.skip_penalty = -0.05
        self.step_penalty = 0.01
        
        self.height_map = HeightMap(self.L, self.W, self.H)
        self.action_mask_calculator = ActionMask(self.L, self.W, self.H)
        self.feasibility_map = np.ones((self.L, self.W), dtype=bool)
        self.top_item_map = np.full((self.L, self.W), -1, dtype=np.int32)
        self.debug_mask_stats = False
        if dataset_type == 'cutting_stock':
            self.dataset_generator = CuttingStockGenerator(
                seed=seed,
                container_dims=(self.L, self.W, self.H),
                target_utilization=1.0,
            )
        elif dataset_type == 'perfect_pack':
            self.dataset_generator = PerfectPackGenerator(
                bin_width=self.L,
                bin_height=self.W,
                seed=seed,
                sigma=self.perfect_pack_sigma,
                size_bias=self.perfect_pack_size_bias,
                mean_ratio=self.perfect_pack_mean_ratio,
            )
        elif dataset_type == 'perfect_pack_layered':
            self.dataset_generator = PerfectPackGenerator(
                bin_width=self.L,
                bin_height=self.W,
                seed=seed,
                sigma=self.perfect_pack_sigma,
                size_bias=self.perfect_pack_size_bias,
                mean_ratio=self.perfect_pack_mean_ratio,
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
        self.ground_truth_positions = []
        
        # State size: height_map (L*W) + item_dims (3) + min_height_info (1)
        self.state_size = self.L * self.W + 3 + 1
        # Action size: positions (L*W) + skip action
        self.action_size = self.L * self.W + 1
        
        # Pre-allocate state buffer for faster state creation (avoid repeated allocation/concat)
        self._state_buffer = np.zeros(self.state_size, dtype=np.float32)
        self._normalized_height_buffer = np.zeros((self.L, self.W), dtype=np.float32)
    
    def reset(self, seed=None):
        """
        Reset environment dan generate new episode.
        
        Args:
            seed (int, optional): Seed untuk random generation
            
        Returns:
            tuple: (state, action_mask)
        """
        self.height_map.reset()
        self.feasibility_map.fill(True)
        self.top_item_map.fill(-1)
        self.top_item_map.fill(-1)
        
        if seed is not None:
            self.dataset_generator.set_seed(seed)
        
        # Generate episode items
        if self.dataset_type == 'perfect_pack':
            self.items = self.dataset_generator.generate_perfect_pack()
            self.ground_truth_positions = []
        elif self.dataset_type == 'perfect_pack_layered':
            items, positions = self.dataset_generator.generate_layered_perfect_pack_with_positions(
                container_height=self.H,
                min_layer_height=self.layered_min_height,
                max_layer_height=self.layered_max_height,
            )
            self.items = items
            self.ground_truth_positions = positions
        else:
            self.items = self.dataset_generator.generate_episode(num_items=self.max_items)
            self.ground_truth_positions = []
        self.current_index = 0
        
        # Reset metrics
        self.episode_reward = 0.0
        self.episode_length = 0
        self.placed_items = []
        self.placed_positions = []
        
        return self._get_state_and_mask()
    
    def _get_state_and_mask(self, item_dims=None, orientation=None):
        """
        Get current state dan action mask (optimized with pre-allocated buffers).
        
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
        if item_dims is None:
            current_item = self.items[self.current_index]
            item_l, item_w, item_h = get_item_dims(current_item)
            item_stacking = get_item_stacking(current_item)
        else:
            item_l, item_w, item_h = item_dims
            item_stacking = 'stackable'

        if orientation is not None:
            item_l, item_w, item_h = self._rotate_item_dims(
                item_l, item_w, item_h, orientation
            )
        
        # Optimized state creation using pre-allocated buffers (avoid repeated concat)
        # Normalize height_map in-place
        hm_max = np.max(self.height_map.map)
        if hm_max > 0:
            self._normalized_height_buffer[:] = self.height_map.map / self.H
        else:
            self._normalized_height_buffer[:] = 0.0
        
        # Fill state buffer
        hm_size = self.L * self.W
        self._state_buffer[:hm_size] = self._normalized_height_buffer.ravel()
        self._state_buffer[hm_size] = item_l / self.L
        self._state_buffer[hm_size + 1] = item_w / self.W
        self._state_buffer[hm_size + 2] = item_h / self.H
        
        min_height = np.min(self.height_map.map) / self.H  # Normalized
        self._state_buffer[hm_size + 3] = min_height
        
        state = self._state_buffer.copy()  # Return copy to avoid external mutations
        
        # Get action mask
        masking_result = self.action_mask_calculator.combine_masks(
            item_l,
            item_w,
            item_h,
            self.height_map,
            top_item_map=self.top_item_map,
            placed_items=self.placed_items,
            item_stacking=item_stacking,
            feasibility_map=self.feasibility_map,
            use_structural_validation=self.use_structural_validation,
            cog_tolerance=self.cog_tolerance,
            fast_stability_mask=self.fast_stability_mask,
        )
        action_mask = self.action_mask_calculator.get_action_vector(
            item_l,
            item_w,
            item_h,
            self.height_map,
            top_item_map=self.top_item_map,
            placed_items=self.placed_items,
            item_stacking=item_stacking,
            feasibility_map=self.feasibility_map,
            use_structural_validation=self.use_structural_validation,
            cog_tolerance=self.cog_tolerance,
            fast_stability_mask=self.fast_stability_mask,
        )

        if self.debug_mask_stats:
            num_valid = masking_result.get('num_valid', 0)
            can_skip = masking_result.get('can_skip', False)
            mask_bound = masking_result.get('mask_bound')
            mask_overflow = masking_result.get('mask_overflow')
            mask_unstable = masking_result.get('mask_unstable')
            if mask_bound is not None and mask_overflow is not None and mask_unstable is not None:
                total_cells = mask_bound.size
                bound_valid = int(np.sum(mask_bound))
                overflow_valid = int(np.sum(mask_bound & mask_overflow))
                unstable_valid = int(np.sum(mask_bound & mask_overflow & mask_unstable))
                removed_bound = total_cells - bound_valid
                removed_overflow = bound_valid - overflow_valid
                removed_unstable = overflow_valid - unstable_valid
                print(
                    f"Mask breakdown | total={total_cells} "
                    f"removed_bound={removed_bound} "
                    f"removed_overflow={removed_overflow} "
                    f"removed_unstable={removed_unstable}"
                )
            print(
                f"Mask debug | item={self.current_index} dims=({item_l},{item_w},{item_h}) "
                f"valid={num_valid} can_skip={can_skip}"
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
        
        # Force episode done if max length reached (for large item sets)
        if self.episode_length > self.max_episode_length:
            return self._get_state_and_mask(), 0.0, True, {'success': False, 'reason': 'max_episode_length'}
        
        # Check if episode done
        if self.current_index >= len(self.items):
            return self._get_state_and_mask(), 0.0, True, {'success': False}
        
        current_item = self.items[self.current_index]
        item_l, item_w, item_h = get_item_dims(current_item)
        item_stacking = get_item_stacking(current_item)
        orientation = None

        if isinstance(action, (tuple, list)) and len(action) == 2:
            action, orientation = action
        elif isinstance(action, dict):
            orientation = action.get('orientation')
            action = action.get('action')

        if orientation is not None:
            item_l, item_w, item_h = self._rotate_item_dims(
                item_l, item_w, item_h, orientation
            )
        
        # Skip action (index == L*W)
        if action == self.L * self.W:
            # Skip to next item
            reward = self.skip_penalty
            self.current_index += 1
            done = self.current_index >= len(self.items)
            
            self.episode_reward += reward  # Accumulate penalty
            
            next_state, next_mask = self._get_state_and_mask()
            info = {'success': False, 'action_type': 'skip'}
            
            return (next_state, next_mask), reward, done, info
        
        # Position action
        x = action % self.L
        y = action // self.L
        
        # Check valid position
        if not self._is_valid_position(x, y, item_l, item_w, item_h, item_stacking):
            # Invalid placement, skip to next item
            reward = self.invalid_penalty  # Penalty untuk invalid placement
            self.current_index += 1
            done = self.current_index >= len(self.items)
            
            self.episode_reward += reward  # Accumulate penalty
            
            next_state, next_mask = self._get_state_and_mask()
            info = {'success': False, 'action_type': 'invalid'}
            
            return (next_state, next_mask), reward, done, info

        support_polygon = None
        if self.use_structural_validation:
            obj_payload = {'x': x, 'y': y, 'w': item_l, 'd': item_w}
            valid, support_polygon, _ = validate_structural_stability(
                obj_payload,
                None,
                self.height_map.map,
                self.feasibility_map,
                self.cog_tolerance,
            )
            if not valid:
                reward = self.invalid_penalty
                self.current_index += 1
                done = self.current_index >= len(self.items)
                self.episode_reward += reward  # Accumulate penalty
                next_state, next_mask = self._get_state_and_mask()
                info = {'success': False, 'action_type': 'invalid'}
                return (next_state, next_mask), reward, done, info
        
        # Place item
        base_height = self.height_map.max_height_in_region(x, y, item_l, item_w)
        new_height = base_height + item_h
        
        self.height_map.update_region(x, y, item_l, item_w, new_height)

        if self.use_structural_validation and support_polygon is not None:
            self.feasibility_map = update_feasibility_map(self.feasibility_map, support_polygon)
        
        # Store placement info
        self.placed_items.append(make_item(item_l, item_w, item_h, item_stacking))
        self.placed_positions.append((x, y, base_height))
        placed_index = len(self.placed_items) - 1
        self.top_item_map[x:x + item_l, y:y + item_w] = placed_index
        
        # Option 1: Enhanced reward function for efficient packing
        # Encourages both volume utilization AND bottom-up filling
        item_volume = item_l * item_w * item_h
        
        # Current container utilization
        total_placed_volume = sum(
            get_item_dims(item)[0] * get_item_dims(item)[1] * get_item_dims(item)[2]
            for item in self.placed_items
        )
        current_utilization = total_placed_volume / self.container_volume
        
        # Height efficiency penalty: penalize stacking too high
        max_height_now = np.max(self.height_map.map)
        height_efficiency = 1.0 - (max_height_now / self.H)  # 1.0 = low, 0.0 = high
        
        # Placement quality: prefer placing at lower heights (bottom-up)
        placement_height_ratio = base_height / self.H
        height_penalty = placement_height_ratio * 0.1  # Slight penalty for high placement
        
        # Combined reward
        volume_reward = (item_volume / self.container_volume) * 8.0  # 80% weight
        utilization_bonus = current_utilization * 2.0  # Encourage filling efficiently
        height_bonus = height_efficiency * 1.0  # Encourage spreading vertically
        
        reward = volume_reward + utilization_bonus + height_bonus - height_penalty - self.step_penalty
        
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
    
    def _is_valid_position(self, x, y, item_l, item_w, item_h, item_stacking=None):
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
        
        # Check stacking policy
        if not self._stacking_allows_placement(x, y, item_l, item_w, item_stacking):
            return False

        # Check stability with LBCP
        try:
            if self.use_structural_validation:
                obj_payload = {'x': x, 'y': y, 'w': item_l, 'd': item_w}
                valid, _, _ = validate_structural_stability(
                    obj_payload,
                    None,
                    self.height_map.map,
                    self.feasibility_map,
                    self.cog_tolerance,
                )
                if not valid:
                    return False

            if not is_stable(self.height_map.map, x, y, item_l, item_w, item_h, self.H):
                return False
        except Exception:
            return False
        
        return True

    def _rotate_item_dims(self, item_l, item_w, item_h, orientation):
        """Rotate item dimensions based on orientation index."""
        if orientation is None:
            return item_l, item_w, item_h

        if int(orientation) % 2 == 1:
            return item_w, item_l, item_h

        return item_l, item_w, item_h
    
    def get_utilization(self):
        """
        Get container utilization percentage.
        
        Returns:
            float: Utilization percentage (0-100)
        """
        if not self.placed_items:
            return 0.0
        
        total_placed_volume = sum(
            get_item_dims(item)[0] * get_item_dims(item)[1] * get_item_dims(item)[2]
            for item in self.placed_items
        )
        utilization = (total_placed_volume / self.container_volume) * 100.0
        
        return min(utilization, 100.0)
    
    def get_max_height(self):
        """Get maximum height in current container."""
        return int(np.max(self.height_map.map))
    
    def perform_repack(self, strategy='load_balanced'):
        """
        Perform repacking operation untuk reorganisasi items.
        
        Args:
            strategy (str): Repacking strategy ('blf', 'load_balanced', 'min_height', 'auto')
            
        Returns:
            dict: Repacking result dengan:
                - 'success': bool
                - 'reward': float (reward untuk repacking action)
                - 'old_utilization': float
                - 'new_utilization': float
                - 'improvement': float
                - 'description': str
        """
        if len(self.placed_items) == 0:
            return {
                'success': False,
                'reward': 0.0,
                'old_utilization': 0.0,
                'new_utilization': 0.0,
                'improvement': 0.0,
                'description': 'No items to repack'
            }
        
        # Calculate old metrics
        old_util = self.get_utilization()
        old_max_height = self.get_max_height()
        
        # Attempt repacking (planning-only search)
        repack_trial = RepackTrial(container_dims=(self.L, self.W, self.H), time_limit=5.0, env=self)
        env_state = {
            'items': self.items,
            'current_index': self.current_index,
            'height_map': self.height_map,
            'placed_items': self.placed_items,
            'placed_positions': self.placed_positions,
            'feasibility_map': self.feasibility_map,
            'top_item_map': self.top_item_map,
        }
        repack_result = repack_trial.attempt_repack(env_state, require_full_pack=False)
        
        if not repack_result['success']:
            return {
                'success': False,
                'reward': -0.1,  # Penalty untuk repack gagal
                'old_utilization': old_util,
                'new_utilization': old_util,
                'improvement': 0.0,
                'description': 'Repacking failed'
            }

        new_positions = repack_result.get('positions', [])
        if len(new_positions) != len(self.placed_items) or any(pos is None for pos in new_positions):
            return {
                'success': False,
                'reward': -0.1,
                'old_utilization': old_util,
                'new_utilization': old_util,
                'improvement': 0.0,
                'description': 'Repacking positions incomplete'
            }

        # Apply new placement plan to environment
        self.height_map.reset()
        self.feasibility_map.fill(True)

        for idx, (item, (x, y, base_height)) in enumerate(zip(self.placed_items, new_positions)):
            item_l, item_w, item_h = get_item_dims(item)
            if self.use_structural_validation:
                obj_payload = {'x': x, 'y': y, 'w': item_l, 'd': item_w}
                valid, support_polygon, _ = validate_structural_stability(
                    obj_payload,
                    None,
                    self.height_map.map,
                    self.feasibility_map,
                    self.cog_tolerance,
                )
                if not valid:
                    return {
                        'success': False,
                        'reward': -0.1,
                        'old_utilization': old_util,
                        'new_utilization': old_util,
                        'improvement': 0.0,
                        'description': 'Repacking produced unstable placement'
                    }

            new_height = base_height + item_h
            self.height_map.update_region(x, y, item_l, item_w, new_height)
            self.top_item_map[x:x + item_l, y:y + item_w] = idx

            if self.use_structural_validation:
                self.feasibility_map = update_feasibility_map(
                    self.feasibility_map,
                    support_polygon
                )

        self.placed_positions = new_positions
        
        # Calculate new metrics
        new_util = self.get_utilization()
        new_max_height = self.get_max_height()
        
        # Calculate improvement
        height_improvement = old_max_height / max(new_max_height, 1.0)
        util_improvement = new_util / max(old_util, 0.1)
        
        # Reward untuk successful repacking
        reward = 0.1 + 0.05 * (height_improvement - 1.0)
        
        return {
            'success': True,
            'reward': reward,
            'old_utilization': old_util,
            'new_utilization': new_util,
            'improvement': height_improvement,
            'description': 'Repacking applied',
            'strategy': 'repack_trial'
        }

    def _stacking_allows_placement(self, x, y, item_l, item_w, item_stacking):
        if item_stacking is None:
            item_stacking = 'stackable'

        base_height = self.height_map.max_height_in_region(x, y, item_l, item_w)
        if base_height <= 0:
            return True

        region = self.height_map.map[x:x + item_l, y:y + item_w]
        support_mask = region == base_height
        if not np.any(support_mask):
            return True

        support_indices = set(self.top_item_map[x:x + item_l, y:y + item_w][support_mask].tolist())
        for idx in support_indices:
            if idx < 0 or idx >= len(self.placed_items):
                continue
            support_item = self.placed_items[idx]
            support_stack = get_item_stacking(support_item)
            if support_stack == 'no_stack':
                return False
            if support_stack == 'fragile' and item_stacking != 'fragile':
                return False

        return True
    
    def render(self):
        """Print container state."""
        print(f"Episode Length: {self.episode_length}")
        print(f"Items Placed: {len(self.placed_items)}/{len(self.items)}")
        print(f"Utilization: {self.get_utilization():.2f}%")
        print(f"Max Height: {self.get_max_height()}/{self.H}")
        print(f"Episode Reward: {self.episode_reward:.2f}")

    print("=" * 70)