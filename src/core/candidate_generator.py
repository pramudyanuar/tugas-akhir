import numpy as np
from collections import OrderedDict
import hashlib

class CandidateGenerator:
    def __init__(self, grid_L, grid_W):
        self.L = grid_L
        self.W = grid_W
        # LRU cache for sorted candidates (key: (mask_hash, zone_priority) -> sorted_actions)
        self._sort_cache = OrderedDict()
        self._sort_cache_max = 256

    def generate_all(self):
        candidates = []
        for x in range(self.L):
            for y in range(self.W):
                candidates.append((x, y))
        return candidates
    
    def clear_cache(self):
        """Clear candidate sorting cache to free memory."""
        self._sort_cache.clear()

    def generate_from_macro(self, action_mask, macro_decision=None, top_k=128):
        """
        Generate candidate actions berdasarkan macro decision dari high-level agent.
        
        Uses LRU cache to avoid re-sorting identical masks with same zone_priority.

        Args:
            action_mask (np.ndarray): Valid action mask (L*W + 1)
            macro_decision (dict): Dict dengan keys:
                - orientation: int (0-5)
                - zone_priority: str
                - allow_repacking: bool
            top_k (int): Maksimum jumlah candidates yang dikembalikan

        Returns:
            list[int]: List action indices (tanpa skip action)
        """
        if action_mask is None or len(action_mask) < self.L * self.W:
            return []

        if macro_decision is None:
            macro_decision = {}

        zone_priority = macro_decision.get('zone_priority', 'center')

        # Exclude skip action, hanya candidate posisi placement.
        flat_mask = np.asarray(action_mask[:self.L * self.W]) > 0
        valid_actions = np.where(flat_mask)[0].tolist()

        if len(valid_actions) == 0:
            return []

        # Compute cache key from mask hash + zone priority
        try:
            mask_hash = hashlib.md5(np.asarray(action_mask[:self.L * self.W]).tobytes()).hexdigest()[:8]
        except Exception:
            mask_hash = None

        cache_key = (mask_hash, zone_priority) if mask_hash else None

        # Check cache
        if cache_key is not None and cache_key in self._sort_cache:
            sorted_actions = self._sort_cache[cache_key]
            self._sort_cache.move_to_end(cache_key)
        else:
            # Compute scores and sort
            scored_candidates = []
            for action in valid_actions:
                x = action % self.L
                y = action // self.L
                score = self._zone_score(x, y, zone_priority)
                scored_candidates.append((score, action))

            # Sort descending score, score lebih tinggi = prioritas lebih tinggi.
            scored_candidates.sort(key=lambda t: t[0], reverse=True)
            sorted_actions = [action for _, action in scored_candidates]
            
            # Cache result
            if cache_key is not None:
                self._sort_cache[cache_key] = sorted_actions
                if len(self._sort_cache) > self._sort_cache_max:
                    self._sort_cache.popitem(last=False)

        if top_k is not None and top_k > 0:
            return sorted_actions[:top_k]
        return sorted_actions

    def _zone_score(self, x, y, zone_priority):
        """Heuristic zone scoring untuk prioritas area placement."""
        center_x = (self.L - 1) / 2.0
        center_y = (self.W - 1) / 2.0

        if zone_priority == 'left_to_right':
            # Prioritize y=0 (left) to y=W (right), then smaller x (back wall)
            return -y * 100 - x
        if zone_priority == 'right_to_left':
            # Prioritize y=W (right) to y=0 (left), then smaller x (back wall)
            return y * 100 - x
        if zone_priority == 'front_to_back':
            # Prioritize x=L (front door) to x=0 (back wall), then smaller y (left wall)
            return x * 100 - y
        if zone_priority == 'back_to_front':
            # Pintu di x = L, dinding belakang di x = 0.
            # Prioritaskan x kecil (belakang) terlebih dahulu, baru isi y (lebar 0-24).
            return -x * 100 - y

        # Default: center priority
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        return -dist