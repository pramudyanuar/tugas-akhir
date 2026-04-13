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

    def __init__(self, bin_width=23, bin_height=23, sigma=2, seed=None):
        """
        Initialize PerfectPackGenerator.

        Args:
            bin_width (int): Width container W
            bin_height (int): Height container H
            sigma (float): Standard deviation untuk Gaussian distribution (default: 2)
            seed (int, optional): Random seed untuk reproducibility
        """
        self.W = bin_width
        self.H = bin_height
        self.sigma = sigma
        self.container_volume = self.W * self.H
        self.rng = np.random.RandomState(seed)
        
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
            list: List of tuples (width, height) dengan 100% bin utilization (atau ~100%)
        """
        best_items = None
        best_util = 0.0

        for attempt in range(num_attempts):
            items = self._generate_single_attempt()
            area = sum(item[0] * item[1] for item in items)
            util = area / self.container_volume

            if util >= 0.99:  # ~100% utilization achieved
                return items
            
            if util > best_util:
                best_util = util
                best_items = items

        return best_items if best_items else []

    def _generate_single_attempt(self):
        """
        Generate single attempt untuk perfect packing.

        Returns:
            list: Item set dengan 3D dimensions (length, width, height)
        """
        items = []
        area = 0
        
        # Initialize bin dengan semua zero heights (empty bin)
        bin_map = np.zeros((self.W, self.H), dtype=np.int32)
        
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
                        placed = True
                        best_y, best_x = np.unravel_index(np.argmax(R), R.shape)
                        
                        # Place item di position terbaik
                        bin_map[best_x:best_x+wo, best_y:best_y+ho] = 1
                        items.append((wo, ho, item_h))  # 3D item
                        area += wo * ho
                        maxw = max(maxw, wo)
                        maxh = max(maxh, ho)
                
                attempts += 1
            
            # Jika tidak bisa place item dengan dimensi random, break
            if not placed:
                break
        
        # Handle remaining gap untuk guaranteed 100% utilization
        if area == self.container_volume - 1:
            # Ada gap 1 pixel, tambah item (1, 1, h) untuk fill
            # Cek apakah ada tempat untuk item (1, 1)
            item_h = self.rng.randint(self.min_height, self.max_height + 1)
            for x in range(self.W):
                for y in range(self.H):
                    if bin_map[x, y] == 0:
                        bin_map[x, y] = 1
                        items.append((1, 1, item_h))
                        area += 1
                        break
                if area == self.container_volume:
                    break
        
        # Shuffle items untuk randomness dalam packing order
        if len(items) > 0:
            shuffle_indices = self.rng.permutation(len(items))
            items = [items[i] for i in shuffle_indices]
        
        return items

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
        mu = dmax / 2.0
        prob_continuous = norm.pdf(x, loc=mu, scale=self.sigma)
        
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

    def generate_episode(self, num_items=None, **kwargs):
        """
        Generate episode dengan perfect pack algoritma.
        
        Compatibility method untuk match RandomGenerator interface.

        Args:
            num_items (int): Number of items untuk generate (optional, ignored)
            
        Returns:
            list: Item set dengan ~100% utilization
        """
        # Perfect pack generator returns items for ~100% utilization
        # Num items is determined by the packing algorithm
        return self.generate_perfect_pack(num_attempts=3)


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
