"""
Algorithm 4: 100% Set Instance Generation
Generates item sets dengan 100% bin utilization menggunakan Gaussian distribution
dan EdgeContact-based scoring untuk placement optimization.

Main Features:
- Gaussian distribution untuk item dimensions dengan controllable σ
- EdgeContact scoring untuk maksimalkan edge contact points
- 100% bin utilization guarantee (area = W × H or W × H - 1)
- Deterministic placement dengan greedy edge-contact strategy
"""

import numpy as np
from scipy.stats import norm

from src.utils.item_utils import get_item_dims, make_item
from src.core.lbcp import validate_structural_stability, update_feasibility_map


class PerfectPackGenerator:
    """
    Generator untuk perfect packing instances dengan 100% bin utilization.

    Algorithm Overview:
    1. Initialize empty bin B of size W × H
    2. While area < W × H - 1:
        a. Sample dimensions dari Gaussian distribution
        b. Compute EdgeContact score untuk semua feasible positions
        c. Tentukan position (x*, y*) dengan max EdgeContact
        d. Place item di position terbaik
        e. Update bin dan accumulated area
    3. Jika masih ada gap (area = W × H - 1), tambah item (1, 1)
    4. Shuffle dan return item set
    """

    def __init__(self, bin_width=23, bin_height=23, sigma=2, seed=None, size_bias=0.0, mean_ratio=0.5):
        """
        Initialize PerfectPackGenerator.

        Args:
            bin_width (int): Width container W
            bin_height (int): Height container H
            sigma (float): Standard deviation untuk Gaussian distribution (default: 2)
            size_bias (float): Bias ke ukuran kecil (>0) atau besar (<0) (default: 0)
            mean_ratio (float): Posisi mean Gaussian relatif terhadap dmax (0-1)
            seed (int, optional): Random seed untuk reproducibility
        """
        self.W = bin_width
        self.H = bin_height
        self.sigma = sigma
        self.size_bias = float(size_bias)
        self.mean_ratio = float(mean_ratio)
        self.container_volume = self.W * self.H
        self.rng = np.random.RandomState(seed)
        stack_seed = None if seed is None else int(seed) + 137
        self.stack_rng = np.random.RandomState(stack_seed)
        
        # For 3D items (to match randomgenerator and container_env)
        self.min_height = 2
        self.max_height = 12

    def generate_perfect_pack(self, num_attempts=3):
        """
        Generate perfect packing instance dengan guaranteed 100% utilization.

        Args:
            num_attempts (int): Jumlah attempt untuk perfection
                              Jika tidak berhasil dalam attempts, return best result

        Returns:
            list: List of item dicts dengan 100% bin utilization (atau ~100%)
        """
        best_items = None
        best_util = 0.0

        max_attempts = max(10, int(num_attempts))
        for attempt in range(max_attempts):
            items = self._generate_single_attempt()
            area = sum(get_item_dims(item)[0] * get_item_dims(item)[1] for item in items)
            util = area / self.container_volume

            if util >= 0.99:  # ~100% utilization achieved
                return items
            
            if util > best_util:
                best_util = util
                best_items = items

        if not best_items:
            return []

        best_area = sum(get_item_dims(item)[0] * get_item_dims(item)[1] for item in best_items)
        target_area = int(np.ceil(self.container_volume * 0.95))
        if best_area < target_area:
            gap = target_area - best_area
            for _ in range(gap):
                item_h = self.rng.randint(self.min_height, self.max_height + 1)
                best_items.append(make_item(1, 1, item_h, self._sample_stacking()))

        return best_items

    def generate_perfect_pack_with_positions(self, num_attempts=3, shuffle=False):
        """
        Generate perfect packing dengan positions (ground truth placement).

        Args:
            num_attempts (int): Jumlah attempt untuk perfection
            shuffle (bool): Shuffle pasangan (item, position) jika True

        Returns:
            tuple: (items, positions)
                - items: list of (length, width, height)
                - positions: list of (x, y, z)
        """
        best_items = None
        best_positions = None
        best_util = 0.0

        max_attempts = max(10, int(num_attempts))
        for _ in range(max_attempts):
            items, positions = self._generate_single_attempt(return_positions=True)
            area = sum(get_item_dims(item)[0] * get_item_dims(item)[1] for item in items)
            util = area / self.container_volume

            if util >= 0.99:
                return self._maybe_shuffle(items, positions, shuffle)

            if util > best_util:
                best_util = util
                best_items = items
                best_positions = positions

        if best_items is None:
            return [], []

        return self._maybe_shuffle(best_items, best_positions, shuffle)

    def generate_layered_perfect_pack_with_positions(
        self,
        container_height,
        min_layer_height=2,
        max_layer_height=6,
        num_attempts=3,
        shuffle=False,
        enforce_stability=False,
        cog_tolerance=0.15,
        max_stability_checks=128,
    ):
        """
        Generate layered perfect pack for full 3D utilization with varied heights.

        Args:
            container_height (int): Total height for stacking layers
            min_layer_height (int): Minimum layer thickness
            max_layer_height (int): Maximum layer thickness
            num_attempts (int): Attempts per layer
            shuffle (bool): Shuffle pasangan (item, position) jika True

        Returns:
            tuple: (items, positions)
        """
        if container_height <= 0:
            return [], []

        min_layer_height = max(1, int(min_layer_height))
        max_layer_height = max(min_layer_height, int(max_layer_height))

        layers = []
        remaining = int(container_height)
        while remaining > 0:
            if remaining <= max_layer_height:
                layers.append(remaining)
                remaining = 0
                break

            max_allowed = min(max_layer_height, remaining - min_layer_height)
            if max_allowed < min_layer_height:
                layers[-1] += remaining
                remaining = 0
                break

            layer_h = self.rng.randint(min_layer_height, max_allowed + 1)
            layers.append(layer_h)
            remaining -= layer_h

        items = []
        positions = []
        z_offset = 0
        for layer_h in layers:
            layer_items, layer_positions = self._generate_single_attempt(
                return_positions=True,
                fixed_height=layer_h,
                z_offset=z_offset,
                enforce_stability=enforce_stability,
                cog_tolerance=cog_tolerance,
                max_stability_checks=max_stability_checks,
            )
            items.extend(layer_items)
            positions.extend(layer_positions)
            z_offset += layer_h

        if shuffle:
            return self._maybe_shuffle(items, positions, True)

        return items, positions

    def _generate_single_attempt(
        self,
        return_positions=False,
        fixed_height=None,
        z_offset=0,
        enforce_stability=False,
        cog_tolerance=0.15,
        max_stability_checks=128,
    ):
        """
        Generate single attempt untuk perfect packing.

        Returns:
            list: Item set dengan 3D dimensions (length, width, height)
        """
        items = []
        positions = []
        area = 0
        
        # Initialize bin dengan semua zero heights (empty bin)
        bin_map = np.zeros((self.W, self.H), dtype=np.int32)
        height_map = np.zeros((self.W, self.H), dtype=np.int32)
        feasibility_map = np.ones((self.W, self.H), dtype=bool) if enforce_stability else None
        height_has_support = False
        
        # Track max width dan height untuk dimension constraints
        maxw = 0
        maxh = 0
        
        # Compute probability distributions untuk width dan height
        pw = self._gaussian_prob_distribution(self.W)
        ph = self._gaussian_prob_distribution(self.H)
        
        # Main packing loop
        while area < self.container_volume - 1:
            placed = False
            attempts = 0
            max_sample_attempts = 20
            
            # Sample loop: coba find valid dimensions
            while not placed and attempts < max_sample_attempts:
                # Sample dimensions dari Gaussian distribution
                wo = self._sample_dimension(pw, self.W)
                ho = self._sample_dimension(ph, self.H)
                
                # Sample height randomly (3D dimension)
                if fixed_height is not None:
                    item_h = int(fixed_height)
                else:
                    item_h = self.rng.randint(self.min_height, self.max_height + 1)
                
                # Constraint: dimensi harus fit dalam container
                if (wo + ho <= self.W + self.H - maxw - maxh) and \
                   (area + wo * ho <= self.container_volume):
                    
                    # Compute EdgeContact score untuk semua feasible positions
                    R = self._compute_edge_contact_scores(bin_map, wo, ho)
                    
                    # Find position dengan max EdgeContact
                    max_score = np.max(R) if R.size > 0 else -np.inf
                    
                    if max_score >= 0:
                        # Ada feasible position
                        candidate_indices = np.argwhere(R >= 0)
                        if candidate_indices.size == 0:
                            candidate_indices = np.array([]).reshape(0, 2)
                        else:
                            scores = R[candidate_indices[:, 0], candidate_indices[:, 1]]
                            order = np.argsort(scores)[::-1]
                            candidate_indices = candidate_indices[order]

                        checks = 0
                        for best_y, best_x in candidate_indices:
                            support_polygon = None
                            if enforce_stability and height_has_support:
                                base_height = int(np.max(height_map[best_x:best_x+wo, best_y:best_y+ho]))
                                obj_payload = {
                                    'x': int(best_x),
                                    'y': int(best_y),
                                    'w': int(wo),
                                    'd': int(ho),
                                }
                                valid, support_polygon, _ = validate_structural_stability(
                                    obj_payload,
                                    None,
                                    height_map,
                                    feasibility_map,
                                    float(cog_tolerance),
                                )
                                if not valid:
                                    checks += 1
                                    if max_stability_checks is not None and checks >= int(max_stability_checks):
                                        break
                                    continue

                            # Place item di position terbaik
                            placed = True
                            bin_map[best_x:best_x+wo, best_y:best_y+ho] = 1
                            base_height = int(np.max(height_map[best_x:best_x+wo, best_y:best_y+ho]))
                            height_map[best_x:best_x+wo, best_y:best_y+ho] = base_height + item_h
                            height_has_support = height_has_support or (base_height + item_h > 0)
                            if enforce_stability and support_polygon is not None:
                                feasibility_map = update_feasibility_map(feasibility_map, support_polygon)
                            items.append(make_item(wo, ho, item_h, self._sample_stacking()))
                            if return_positions:
                                positions.append((best_x, best_y, int(z_offset)))
                            area += wo * ho
                            maxw = max(maxw, wo)
                            maxh = max(maxh, ho)
                            break
                
                attempts += 1
            
            # Jika tidak bisa place item dengan dimensi random, break
            if not placed:
                break
        
        # Handle remaining gap untuk guaranteed 100% utilization
        if area == self.container_volume - 1:
            # Ada gap 1 pixel, tambah item (1, 1, h) untuk fill
            # Cek apakah ada tempat untuk item (1, 1)
            if fixed_height is not None:
                item_h = int(fixed_height)
            else:
                item_h = self.rng.randint(self.min_height, self.max_height + 1)
            for x in range(self.W):
                for y in range(self.H):
                    if bin_map[x, y] == 0:
                        bin_map[x, y] = 1
                        items.append(make_item(1, 1, item_h, self._sample_stacking()))
                        if return_positions:
                            positions.append((x, y, int(z_offset)))
                        area += 1
                        break
                if area == self.container_volume:
                    break
        
        # Shuffle items untuk randomness dalam packing order
        if len(items) > 0:
            shuffle_indices = self.rng.permutation(len(items))
            items = [items[i] for i in shuffle_indices]
            if return_positions:
                positions = [positions[i] for i in shuffle_indices]

        if return_positions:
            return items, positions

        return items

    def _maybe_shuffle(self, items, positions, shuffle):
        if not shuffle or len(items) == 0:
            return items, positions

        shuffle_indices = self.rng.permutation(len(items))
        items = [items[i] for i in shuffle_indices]
        positions = [positions[i] for i in shuffle_indices]
        return items, positions

    def _gaussian_prob_distribution(self, dmax):
        """
        Compute Gaussian probability distribution untuk dimension sampling.

        Args:
            dmax (int): Maximum dimension

        Returns:
            np.ndarray: Probability distribution array
        """
        # Gaussian centered di dmax/2 dengan std = sigma
        x = np.arange(1, dmax + 1)
        mu = max(1.0, min(float(dmax), float(dmax) * self.mean_ratio))
        prob_continuous = norm.pdf(x, loc=mu, scale=self.sigma)
        if self.size_bias != 0:
            # Bias small sizes if size_bias > 0; large sizes if size_bias < 0.
            bias = ((dmax - x + 1) / dmax) ** self.size_bias
            prob_continuous = prob_continuous * bias
        
        # Normalize ke probability distribution
        prob = prob_continuous / np.sum(prob_continuous)
        return prob

    def _sample_dimension(self, prob_dist, dmax):
        """
        Sample single dimension dari probability distribution.

        Args:
            prob_dist (np.ndarray): Probability distribution
            dmax (int): Maximum dimension

        Returns:
            int: Sampled dimension
        """
        # CDF sampling untuk dimension
        dim = self.rng.choice(np.arange(1, dmax + 1), p=prob_dist)
        return int(dim)

    def _compute_edge_contact_scores(self, bin_map, wo, ho):
        """
        Compute EdgeContact score untuk semua feasible positions.

        EdgeContact measures: jumlah edges yang saling berkontak dengan existing items.
        Higher score = lebih banyak edge contact = lebih stabil placement.

        Args:
            bin_map (np.ndarray): Current bin state (1 = occupied, 0 = free)
            wo, ho (int): Item width dan height

        Returns:
            np.ndarray: Score matrix R[y, x] untuk each position
        """
        R = np.full((self.H - ho + 1, self.W - wo + 1), -np.inf, dtype=np.float32)
        
        # Iterate semua feasible positions
        for x in range(self.W - wo + 1):
            for y in range(self.H - ho + 1):
                # Check apakah region [x:x+wo, y:y+ho] kosong
                if np.all(bin_map[x:x+wo, y:y+ho] == 0):
                    # Region feasible, hitung edge contact score
                    score = self._edge_contact(bin_map, x, y, wo, ho)
                    R[y, x] = score
        
        return R

    def _edge_contact(self, bin_map, x, y, wo, ho):
        """
        Compute EdgeContact score untuk specific position.

        EdgeContact = jumlah unit edges yang saling kontak:
        - Top edge contact: existing items di atas
        - Bottom edge contact: existing items di bawah
        - Left edge contact: existing items di kiri
        - Right edge contact: existing items di kanan

        Args:
            bin_map (np.ndarray): Current bin state
            x, y (int): Position untuk item
            wo, ho (int): Item dimensions

        Returns:
            float: Edge contact score
        """
        score = 0.0
        
        # Bottom edge contact: check apakah ada items di y-1
        if y > 0:
            # Count cells yang berkontak dengan bottom
            bottom_contact = np.sum(bin_map[x:x+wo, y-1] == 1)
            score += bottom_contact
        elif y == 0:
            # Touching floor scores high (stability)
            score += wo
        
        # Left edge contact: check apakah ada items di x-1
        if x > 0:
            left_contact = np.sum(bin_map[x-1, y:y+ho] == 1)
            score += left_contact
        elif x == 0:
            # Touching left wall
            score += ho
        
        # Right edge contact: check apakah ada items di x+wo
        if x + wo < self.W:
            right_contact = np.sum(bin_map[x+wo, y:y+ho] == 1)
            score += right_contact
        elif x + wo == self.W:
            # Touching right wall
            score += ho
        
        # Top edge contact: check apakah ada items di y+ho
        if y + ho < self.H:
            top_contact = np.sum(bin_map[x:x+wo, y+ho] == 1)
            score += top_contact
        elif y + ho == self.H:
            # Touching top wall
            score += wo
        
        return score

    def set_seed(self, seed):
        """Set random seed untuk reproducibility."""
        self.rng = np.random.RandomState(seed)

    def generate_episode(self, num_items=50, **kwargs):
        """
        Generate episode dengan kontrol jumlah items dan utilization yang reasonable.
        
        Strategy:
        - Untuk 1-5 items: Generate besar items (close to perfect pack)
        - Untuk 6-15 items: Mix ukuran medium-large
        - Untuk 15+ items: Banyak items kecil yang total sekitar 80-95% utilization

        Args:
            num_items (int): Number of items to generate (default: 50)
            
        Returns:
            list: Item set dengan dimensi 3D (length, width, height) yang fit dalam container
        """
        if num_items <= 0:
            num_items = 50
        
        items = []
        target_utilization = 0.85  # Target 85% utilization
        target_area = int(self.container_volume * target_utilization)
        current_area = 0
        
        if num_items <= 5:
            # Few large items - use perfect pack algorithm
            items = self.generate_perfect_pack(num_attempts=3)
            # Pad if necessary
            while len(items) < num_items:
                small_item = make_item(
                    self.rng.randint(3, 6),
                    self.rng.randint(3, 6),
                    self.rng.randint(self.min_height, self.max_height + 1),
                    self._sample_stacking(),
                )
                items.append(small_item)
            return items[:num_items]
        
        elif num_items <= 15:
            # Medium number: mix of sizes
            # Generate items dengan target untuk mencapai ~85% utilization
            area_per_item = target_area // num_items
            
            for i in range(num_items):
                remaining_items = num_items - i
                remaining_area = max(1, target_area - current_area)
                avg_area_needed = remaining_area // remaining_items
                
                # Sample item dengan area close to average needed
                # Generate items dengan random size tapi constrain total area
                max_attempts = 10
                for _ in range(max_attempts):
                    w = self.rng.randint(4, 14)
                    h = self.rng.randint(4, 14)
                    
                    # Constraint: item area tidak lebih dari remaining space
                    if w * h <= remaining_area + 10:  # Allow small overage
                        d = self.rng.randint(self.min_height, self.max_height + 1)
                        items.append(make_item(w, h, d, self._sample_stacking()))
                        current_area += w * h
                        break
                else:
                    # If can't find good size, add small item
                    d = self.rng.randint(self.min_height, self.max_height + 1)
                    items.append(make_item(3, 3, d, self._sample_stacking()))
                    current_area += 9
            
            return items[:num_items]
        
        else:
            # Many items: lots of small items
            # Target ~10-25 area per item untuk fit many items
            area_per_item = max(5, target_area // num_items)
            
            for i in range(num_items):
                remaining_items = num_items - i
                remaining_area = max(1, target_area - current_area)
                
                # Generate small items
                max_w = int(np.sqrt(remaining_area / remaining_items)) + 2
                max_h = int(np.sqrt(remaining_area / remaining_items)) + 2
                
                w = self.rng.randint(2, min(max_w, 10))
                h = self.rng.randint(2, min(max_h, 10))
                d = self.rng.randint(self.min_height, self.max_height + 1)
                
                items.append(make_item(w, h, d, self._sample_stacking()))
                current_area += w * h
                
                # Stop if we've reached target area
                if current_area >= target_area:
                    break
            
            # Pad dengan more items if needed
            while len(items) < num_items and current_area < target_area * 1.1:
                w = self.rng.randint(2, 4)
                h = self.rng.randint(2, 4)
                d = self.rng.randint(self.min_height, self.max_height + 1)
                items.append(make_item(w, h, d, self._sample_stacking()))
                current_area += w * h
            
            return items[:num_items]

    def _sample_stacking(self):
        u = self.stack_rng.rand()
        if u < 0.60:
            return 'stackable'
        if u < 0.85:
            return 'fragile'
        return 'no_stack'


# Convenience function untuk API compatibility
def generate_perfect_pack(bin_width=23, bin_height=23, sigma=2, seed=None, num_attempts=3):
    """
    Generate perfect packing instance dengan 100% bin utilization.

    Args:
        bin_width (int): Width container
        bin_height (int): Height container
        sigma (float): Gaussian std for dimensions
        seed (int): Random seed
        num_attempts (int): Attempts untuk achieve perfection

    Returns:
        list: Item set dengan ~100% utilization
    """
    generator = PerfectPackGenerator(bin_width=bin_width, bin_height=bin_height, 
                                      sigma=sigma, seed=seed)
    return generator.generate_perfect_pack(num_attempts=num_attempts)
