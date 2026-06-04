import numpy as np
import sys
import os

# Use relative imports for clean module structure
from .height_map import HeightMap
from .lbcp import is_stable, validate_structural_stability, update_feasibility_map
from .action_mask import ActionMask

# Import dari parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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
                 max_episode_length=500, dataset_path=None,
                 buffer_capacity=3, max_waiting_steps=5,
                 defer_penalty=-0.02, overflow_penalty=-0.5):
        """
        Initialize container environment.
        """
        self.L = container_length
        self.W = container_width
        self.H = container_height
        self.max_items = max_items
        self.max_episode_length = max(100, int(max_episode_length))
        self.dataset_type = dataset_type
        
        # Buffer configuration
        self.buffer_capacity = buffer_capacity
        self.max_waiting_steps = max_waiting_steps
        self.defer_penalty = defer_penalty
        self.overflow_penalty = overflow_penalty
        self.deferred_buffer = []

        # Buffer metric tracking
        self.num_deferred_items = 0
        self.num_deferred_success = 0
        self.num_rejected_items = 0
        self.num_fragile_violations = 0
        self.num_weight_violations = 0
        self.num_stability_violations = 0
        self.total_waiting_time = 0
        self.deferred_attempts = 0
        
        if self.dataset_type == 'rs':
            import torch
            rs_dataset_path = 'dataset/rs.pt'
            if not os.path.exists(rs_dataset_path):
                rs_dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset', 'rs.pt')
            self.rs_data = torch.load(rs_dataset_path, map_location='cpu')
            # Override to 10x10x10 if using default 20ft container dims
            if self.L == 60 and self.W == 24 and self.H == 26:
                self.L = 10
                self.W = 10
                self.H = 10
        elif self.dataset_type == 'perfect_pack_pt' or dataset_path is not None:
            import torch
            if dataset_path is None:
                dataset_path = 'dataset/perfect_pack.pt'
            if not os.path.exists(dataset_path):
                dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset', 'perfect_pack.pt')
            self.perfect_pack_data = torch.load(dataset_path, map_location='cpu')
            # Override to 10x10x10 if using default 20ft container dims
            if self.L == 60 and self.W == 24 and self.H == 26:
                self.L = 10
                self.W = 10
                self.H = 10

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
        self.debug_invalid_placement = False
        # Import generators locally to avoid circular dependencies
        from src.data.random_generator import RandomGenerator
        from src.data.cutting_stock import CuttingStockGenerator
        from src.data.perfect_pack_generator import PerfectPackGenerator

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
        elif dataset_type == 'rs':
            self.dataset_generator = None
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
        
        # Reset holding buffer and metrics
        self.deferred_buffer = []
        self.num_deferred_items = 0
        self.num_deferred_success = 0
        self.num_rejected_items = 0
        self.num_fragile_violations = 0
        self.num_weight_violations = 0
        self.num_stability_violations = 0
        self.total_waiting_time = 0
        self.deferred_attempts = 0

        if seed is not None and self.dataset_generator is not None:
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
                enforce_stability=True,
                cog_tolerance=self.cog_tolerance,
                max_stability_checks=128,
            )
            self.items = items
            self.ground_truth_positions = positions
        elif self.dataset_type == 'perfect_pack_pt':
            if seed is not None:
                ep_idx = int(seed) % len(self.perfect_pack_data)
            else:
                if not hasattr(self, 'pp_rng'):
                    self.pp_rng = np.random.RandomState(seed)
                ep_idx = self.pp_rng.randint(0, len(self.perfect_pack_data))
            # Deep copy to allow modification of item properties if needed
            import copy
            self.items = copy.deepcopy(self.perfect_pack_data[ep_idx])
            self.ground_truth_positions = []
        elif self.dataset_type == 'rs':
            if seed is not None:
                ep_idx = int(seed) % len(self.rs_data)
            else:
                if not hasattr(self, 'rs_rng'):
                    self.rs_rng = np.random.RandomState(seed)
                ep_idx = self.rs_rng.randint(0, len(self.rs_data))
            
            raw_items = self.rs_data[ep_idx]
            self.items = []
            for item in raw_items:
                self.items.append({
                    'l': int(item[0]),
                    'w': int(item[1]),
                    'h': int(item[2]),
                    'stacking': 'stackable'
                })
            self.ground_truth_positions = []
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
        
        item_weight = current_item.get('weight', float(item_l * item_w * item_h))

        # Skip action (index == L*W)
        if action == self.L * self.W:
            # Check if buffer is not full
            if len(self.deferred_buffer) < self.buffer_capacity:
                deferred_item = dict(current_item)
                deferred_item['waiting_time'] = 0
                deferred_item['arrival_step'] = self.episode_length
                self.deferred_buffer.append(deferred_item)
                self.num_deferred_items += 1
                reward = self.defer_penalty
                action_type = 'defer'
            else:
                # Buffer is full, try to place oldest/highest priority buffer item to free space
                self.deferred_buffer.sort(key=lambda x: (
                    0 if x.get('waiting_time', 0) >= self.max_waiting_steps else 1,
                    0 if x.get('stacking') == 'fragile' else 1,
                    -x.get('waiting_time', 0)
                ))
                best_buf_idx = 0
                buf_item = self.deferred_buffer[best_buf_idx]
                buf_pos = self._find_dblf_placement(buf_item)
                if buf_pos is not None:
                    # Place the buffer item
                    buf_x, buf_y = buf_pos
                    buf_l, buf_w, buf_h = get_item_dims(buf_item)
                    buf_stacking = get_item_stacking(buf_item)
                    buf_base_height = self.height_map.max_height_in_region(buf_x, buf_y, buf_l, buf_w)
                    buf_new_height = buf_base_height + buf_h
                    self.height_map.update_region(buf_x, buf_y, buf_l, buf_w, buf_new_height)
                    
                    if self.use_structural_validation:
                        _, buf_support_polygon = self._validate_stability(buf_x, buf_y, buf_l, buf_w, buf_h)
                        if buf_support_polygon is not None:
                            self.feasibility_map = update_feasibility_map(self.feasibility_map, buf_support_polygon)
                            
                    placed_buf_dict = make_item(buf_l, buf_w, buf_h, buf_stacking)
                    placed_buf_dict['weight'] = buf_item.get('weight', buf_l * buf_w * buf_h)
                    placed_buf_dict['load_bearing'] = buf_item.get('load_bearing', float('inf'))
                    self.placed_items.append(placed_buf_dict)
                    self.placed_positions.append((buf_x, buf_y, buf_base_height))
                    buf_placed_idx = len(self.placed_items) - 1
                    self.top_item_map[buf_x:buf_x + buf_l, buf_y:buf_y + buf_w] = buf_placed_idx
                    
                    # Remove from buffer
                    self.deferred_buffer.pop(best_buf_idx)
                    self.num_deferred_success += 1
                    
                    # Defer current incoming item
                    deferred_item = dict(current_item)
                    deferred_item['waiting_time'] = 0
                    deferred_item['arrival_step'] = self.episode_length
                    self.deferred_buffer.append(deferred_item)
                    self.num_deferred_items += 1
                    
                    # Calculate reward: defer penalty + buf placed reward
                    buf_vol = buf_l * buf_w * buf_h
                    buf_volume_reward = (buf_vol / self.container_volume) * 8.0
                    buf_height_penalty = (buf_base_height / self.H) * 0.1
                    reward = self.defer_penalty + buf_volume_reward - buf_height_penalty + 0.2
                    action_type = 'place_buf_and_defer'
                else:
                    # Buffer item could not be placed, reject current incoming item
                    reward = self.overflow_penalty
                    self.num_rejected_items += 1
                    action_type = 'reject_overflow'
                    
            self.current_index += 1
            done = self.current_index >= len(self.items)
            self.episode_reward += reward
            
            # Increment waiting time for buffer items
            for item in self.deferred_buffer:
                item['waiting_time'] += 1
                
            next_state, next_mask = self._get_state_and_mask()
            info = {
                'success': False,
                'action_type': action_type,
                'reward': reward
            }
            return (next_state, next_mask), reward, done, info
        
        # Position action
        x = action % self.L
        y = action // self.L
        
        # Check valid position (geometry boundary, stacking allowed, stability, load-bearing)
        if not self._is_valid_position(x, y, item_l, item_w, item_h, item_stacking, item_weight):
            if self.debug_invalid_placement:
                reason = self._get_invalid_reason(x, y, item_l, item_w, item_h, item_stacking)
                print(
                    f"Invalid placement | item_idx={self.current_index} "
                    f"pos=({x},{y}) dims=({item_l},{item_w},{item_h}) reason={reason}",
                    flush=True,
                )
            
            # Since position is invalid, try to defer to holding buffer
            if len(self.deferred_buffer) < self.buffer_capacity:
                deferred_item = dict(current_item)
                deferred_item['waiting_time'] = 0
                deferred_item['arrival_step'] = self.episode_length
                self.deferred_buffer.append(deferred_item)
                self.num_deferred_items += 1
                reward = self.invalid_penalty + self.defer_penalty
                action_type = 'invalid_deferred'
            else:
                # Buffer is full, try to place oldest/highest priority buffer item to free space
                self.deferred_buffer.sort(key=lambda x: (
                    0 if x.get('waiting_time', 0) >= self.max_waiting_steps else 1,
                    0 if x.get('stacking') == 'fragile' else 1,
                    -x.get('waiting_time', 0)
                ))
                best_buf_idx = 0
                buf_item = self.deferred_buffer[best_buf_idx]
                buf_pos = self._find_dblf_placement(buf_item)
                if buf_pos is not None:
                    # Place the buffer item
                    buf_x, buf_y = buf_pos
                    buf_l, buf_w, buf_h = get_item_dims(buf_item)
                    buf_stacking = get_item_stacking(buf_item)
                    buf_base_height = self.height_map.max_height_in_region(buf_x, buf_y, buf_l, buf_w)
                    buf_new_height = buf_base_height + buf_h
                    self.height_map.update_region(buf_x, buf_y, buf_l, buf_w, buf_new_height)
                    
                    if self.use_structural_validation:
                        _, buf_support_polygon = self._validate_stability(buf_x, buf_y, buf_l, buf_w, buf_h)
                        if buf_support_polygon is not None:
                            self.feasibility_map = update_feasibility_map(self.feasibility_map, buf_support_polygon)
                            
                    placed_buf_dict = make_item(buf_l, buf_w, buf_h, buf_stacking)
                    placed_buf_dict['weight'] = buf_item.get('weight', buf_l * buf_w * buf_h)
                    placed_buf_dict['load_bearing'] = buf_item.get('load_bearing', float('inf'))
                    self.placed_items.append(placed_buf_dict)
                    self.placed_positions.append((buf_x, buf_y, buf_base_height))
                    buf_placed_idx = len(self.placed_items) - 1
                    self.top_item_map[buf_x:buf_x + buf_l, buf_y:buf_y + buf_w] = buf_placed_idx
                    
                    # Remove from buffer
                    self.deferred_buffer.pop(best_buf_idx)
                    self.num_deferred_success += 1
                    
                    # Defer current incoming item
                    deferred_item = dict(current_item)
                    deferred_item['waiting_time'] = 0
                    deferred_item['arrival_step'] = self.episode_length
                    self.deferred_buffer.append(deferred_item)
                    self.num_deferred_items += 1
                    
                    # Calculate reward: invalid penalty + defer penalty + buf placed reward
                    buf_vol = buf_l * buf_w * buf_h
                    buf_volume_reward = (buf_vol / self.container_volume) * 8.0
                    buf_height_penalty = (buf_base_height / self.H) * 0.1
                    reward = self.invalid_penalty + self.defer_penalty + buf_volume_reward - buf_height_penalty + 0.2
                    action_type = 'invalid_placed_buf_and_deferred'
                else:
                    # Buffer item could not be placed, reject current incoming item
                    reward = self.invalid_penalty + self.overflow_penalty
                    self.num_rejected_items += 1
                    action_type = 'invalid_reject_overflow'
                    
            self.current_index += 1
            done = self.current_index >= len(self.items)
            self.episode_reward += reward
            
            # Increment waiting time for buffer items
            for item in self.deferred_buffer:
                item['waiting_time'] += 1
                
            next_state, next_mask = self._get_state_and_mask()
            info = {
                'success': False,
                'action_type': action_type,
                'reward': reward
            }
            return (next_state, next_mask), reward, done, info

        # Place item
        base_height = self.height_map.max_height_in_region(x, y, item_l, item_w)
        new_height = base_height + item_h
        
        self.height_map.update_region(x, y, item_l, item_w, new_height)

        if self.use_structural_validation:
            _, support_polygon = self._validate_stability(x, y, item_l, item_w, item_h)
            if support_polygon is not None:
                self.feasibility_map = update_feasibility_map(self.feasibility_map, support_polygon)
        
        # Store placement info
        placed_item_dict = make_item(item_l, item_w, item_h, item_stacking)
        placed_item_dict['weight'] = item_weight
        placed_item_dict['load_bearing'] = current_item.get('load_bearing', float('inf'))
        self.placed_items.append(placed_item_dict)
        self.placed_positions.append((x, y, base_height))
        placed_index = len(self.placed_items) - 1
        self.top_item_map[x:x + item_l, y:y + item_w] = placed_index
        
        # Option 1: Enhanced reward function for efficient packing
        item_volume = item_l * item_w * item_h
        
        # Base placement reward
        volume_reward = (item_volume / self.container_volume) * 8.0
        placement_height_ratio = base_height / self.H
        height_penalty = placement_height_ratio * 0.1
        
        reward = volume_reward - height_penalty - self.step_penalty
        
        # --- Retry items from the holding buffer ---
        placed_any = True
        while placed_any:
            placed_any = False
            # Sort the buffer by priority
            self.deferred_buffer.sort(key=lambda x: (
                0 if x.get('waiting_time', 0) >= self.max_waiting_steps else 1,
                0 if x.get('stacking') == 'fragile' else 1,
                -x.get('waiting_time', 0)
            ))
            
            # Find the first item in the buffer that can be placed
            for idx, buf_item in enumerate(self.deferred_buffer):
                buf_pos = self._find_dblf_placement(buf_item)
                if buf_pos is not None:
                    # Place it!
                    buf_x, buf_y = buf_pos
                    buf_l, buf_w, buf_h = get_item_dims(buf_item)
                    buf_stacking = get_item_stacking(buf_item)
                    buf_base_height = self.height_map.max_height_in_region(buf_x, buf_y, buf_l, buf_w)
                    buf_new_height = buf_base_height + buf_h
                    self.height_map.update_region(buf_x, buf_y, buf_l, buf_w, buf_new_height)
                    
                    if self.use_structural_validation:
                        _, buf_support_polygon = self._validate_stability(buf_x, buf_y, buf_l, buf_w, buf_h)
                        if buf_support_polygon is not None:
                            self.feasibility_map = update_feasibility_map(self.feasibility_map, buf_support_polygon)
                            
                    placed_buf_dict = make_item(buf_l, buf_w, buf_h, buf_stacking)
                    placed_buf_dict['weight'] = buf_item.get('weight', buf_l * buf_w * buf_h)
                    placed_buf_dict['load_bearing'] = buf_item.get('load_bearing', float('inf'))
                    self.placed_items.append(placed_buf_dict)
                    self.placed_positions.append((buf_x, buf_y, buf_base_height))
                    buf_placed_idx = len(self.placed_items) - 1
                    self.top_item_map[buf_x:buf_x + buf_l, buf_y:buf_y + buf_w] = buf_placed_idx
                    
                    # Accumulate reward for successful placement from buffer
                    buf_vol = buf_l * buf_w * buf_h
                    buf_volume_reward = (buf_vol / self.container_volume) * 8.0
                    buf_height_penalty = (buf_base_height / self.H) * 0.1
                    buf_success_bonus = 0.2
                    reward += buf_volume_reward - buf_height_penalty + buf_success_bonus
                    
                    # Remove from buffer and update metrics
                    self.deferred_buffer.pop(idx)
                    self.num_deferred_success += 1
                    placed_any = True
                    break

        # Calculate final metrics after all retries
        total_placed_volume = sum(
            get_item_dims(item)[0] * get_item_dims(item)[1] * get_item_dims(item)[2]
            for item in self.placed_items
        )
        current_utilization = total_placed_volume / self.container_volume
        max_height_now = np.max(self.height_map.map)
        height_efficiency = 1.0 - (max_height_now / self.H)
        
        # Add utilization & height bonuses to final step reward
        utilization_bonus = current_utilization * 2.0
        height_bonus = height_efficiency * 1.0
        reward += utilization_bonus + height_bonus
        
        self.episode_reward += reward
        
        # Increment waiting time for remaining buffer items
        for item in self.deferred_buffer:
            item['waiting_time'] += 1
        
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
    
    def _validate_stability(self, x, y, item_l, item_w, item_h):
        """
        Validate position stability.
        Uses fast 60% area support check if fast_stability_mask is True,
        otherwise uses strict LBCP validation (convex hull & CoG).
        """
        if not self.use_structural_validation:
            return True, None
            
        if self.fast_stability_mask:
            # Fast path check on height map
            hm = self.height_map.map
            window = hm[x:x+item_l, y:y+item_w]
            max_height = np.max(window)
            support_count = np.sum(window == max_height)
            valid = support_count >= (item_l * item_w * 0.6)
            return valid, None
        else:
            # Strict validation
            obj_payload = {'x': x, 'y': y, 'w': item_l, 'd': item_w}
            valid, support_polygon, _ = validate_structural_stability(
                obj_payload,
                None,
                self.height_map.map,
                self.feasibility_map,
                self.cog_tolerance,
            )
            return valid, support_polygon

    def _is_valid_position(self, x, y, item_l, item_w, item_h, item_stacking=None, item_weight=None):
        """
        Check if position is valid (boundary + overflow + stability + load bearing).
        
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

        # Check stability
        try:
            valid, _ = self._validate_stability(x, y, item_l, item_w, item_h)
            if not valid:
                return False
        except Exception:
            return False
            
        # Check load bearing capacity
        if item_weight is None:
            item_weight = float(item_l * item_w * item_h)
        if not self._check_load_bearing_after_placement(x, y, item_l, item_w, item_h, item_weight):
            return False
        
        return True

    def _get_invalid_reason(self, x, y, item_l, item_w, item_h, item_stacking=None):
        """Return a short reason string for invalid placements (debug only)."""
        if x + item_l > self.L or y + item_w > self.W:
            return 'out_of_bounds'

        base_height = self.height_map.max_height_in_region(x, y, item_l, item_w)
        if base_height + item_h > self.H:
            return 'overflow'

        if not self._stacking_allows_placement(x, y, item_l, item_w, item_stacking):
            return 'stacking_policy'

        try:
            valid, _ = self._validate_stability(x, y, item_l, item_w, item_h)
            if not valid:
                return 'stability'
        except Exception:
            return 'stability_exception'

        return 'unknown'

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
        from src.planning.repack_trial import RepackTrial
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
                valid, support_polygon = self._validate_stability(x, y, item_l, item_w, item_h)
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

            if self.use_structural_validation and support_polygon is not None:
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

    def _check_load_bearing_after_placement(self, x, y, item_l, item_w, item_h, item_weight):
        base_height = self.height_map.max_height_in_region(x, y, item_l, item_w)
        if base_height <= 0:
            return True
            
        region = self.height_map.map[x:x + item_l, y:y + item_w]
        support_mask = region == base_height
        if not np.any(support_mask):
            return True
            
        # Build simulation structures
        sim_placed_items = list(self.placed_items)
        sim_placed_positions = list(self.placed_positions)
        
        # Add the new item to simulate
        new_idx = len(sim_placed_items)
        sim_placed_items.append({
            'l': item_l, 'w': item_w, 'h': item_h,
            'weight': item_weight, 'load_bearing': 0.0
        })
        sim_placed_positions.append((x, y, base_height))
        
        # Build dynamic support mapping
        support_lists = [set() for _ in range(len(sim_placed_items))]
        for i in range(len(sim_placed_items)):
            ix, iy, iz = sim_placed_positions[i]
            item_i = sim_placed_items[i]
            il, iw, ih = item_i['l'], item_i['w'], item_i['h']
            
            for j in range(len(sim_placed_items)):
                if i == j:
                    continue
                jx, jy, jz = sim_placed_positions[j]
                item_j = sim_placed_items[j]
                jl, jw, jh = item_j['l'], item_j['w'], item_j['h']
                
                # Check vertical adjacency and overlap
                if jz + jh == iz:
                    if max(ix, jx) < min(ix + il, jx + jl) and max(iy, jy) < min(iy + iw, jy + jw):
                        support_lists[i].add(j)
                        
        # Propagate weights top to bottom
        accumulated_weights = [0.0] * len(sim_placed_items)
        sorted_indices = sorted(range(len(sim_placed_items)), key=lambda idx: sim_placed_positions[idx][2], reverse=True)
        
        for idx in sorted_indices:
            weight_idx = sim_placed_items[idx].get('weight', 0.0)
            total_weight_on_idx = weight_idx + accumulated_weights[idx]
            
            sups = support_lists[idx]
            if sups:
                distributed_weight = total_weight_on_idx / len(sups)
                for sup_idx in sups:
                    accumulated_weights[sup_idx] += distributed_weight
                    
        # Verify if any item exceeds its load capacity
        for idx in range(len(sim_placed_items) - 1): # don't check the new item itself
            max_bearing = sim_placed_items[idx].get('load_bearing', float('inf'))
            if accumulated_weights[idx] > max_bearing:
                return False
                
        return True

    def _find_dblf_placement(self, item):
        item_l, item_w, item_h = get_item_dims(item)
        item_stacking = get_item_stacking(item)
        item_weight = item.get('weight', float(item_l * item_w * item_h))
        
        best_pos = None
        best_z = float('inf')
        
        for x in range(self.L - item_l + 1):
            for y in range(self.W - item_w + 1):
                if self._is_valid_position(x, y, item_l, item_w, item_h, item_stacking, item_weight):
                    base_z = self.height_map.max_height_in_region(x, y, item_l, item_w)
                    if base_z < best_z:
                        best_z = base_z
                        best_pos = (x, y)
                    elif base_z == best_z:
                        if best_pos is None or y < best_pos[1] or (y == best_pos[1] and x < best_pos[0]):
                            best_pos = (x, y)
        return best_pos

    def get_buffer_stats(self):
        total_items = len(self.items)
        # Add remaining waiting times of items currently in the buffer
        current_waiting = sum(item.get('waiting_time', 0) for item in self.deferred_buffer)
        total_wait = self.total_waiting_time + current_waiting
        total_deferred = self.num_deferred_items
        
        return {
            'defer_rate': self.num_deferred_items / max(total_items, 1),
            'success_rate': self.num_deferred_success / max(total_deferred, 1),
            'overflow_rate': self.num_rejected_items / max(total_items, 1),
            'avg_waiting_steps': total_wait / max(total_deferred, 1),
            'remaining_in_buffer': len(self.deferred_buffer)
        }
    
    def render(self):
        """Print container state."""
        print(f"Episode Length: {self.episode_length}")
        print(f"Items Placed: {len(self.placed_items)}/{len(self.items)}")
        print(f"Utilization: {self.get_utilization():.2f}%")
        print(f"Max Height: {self.get_max_height()}/{self.H}")
        print(f"Episode Reward: {self.episode_reward:.2f}")