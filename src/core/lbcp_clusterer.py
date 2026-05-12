"""Load-Balanced Clustering Problem (LBCP) Clusterer."""

import numpy as np

from src.utils.item_utils import get_item_dims


class LBCPClusterer:
    """
    Load-Balanced Clustering Problem (LBCP) module.
    
    Performs weight-based clustering untuk memastikan distribusi beban yang seimbang
    dalam kontainer.
    
    Features:
    - Weight-based item grouping
    - Load balance analysis
    - Center of gravity tracking
    - Cluster reordering untuk optimal distribution
    """
    
    def __init__(self, num_clusters=4, balance_threshold=0.15):
        """
        Initialize LBCP Clusterer.
        
        Args:
            num_clusters (int): Target number of clusters
            balance_threshold (float): Threshold untuk acceptable load imbalance
        """
        self.num_clusters = num_clusters
        self.balance_threshold = balance_threshold
        self.clusters = []
        self.cluster_weights = []
        self.center_of_gravity = None
    
    def cluster_by_weight(self, items):
        """
        Perform weight-based clustering menggunakan greedy algorithm.
        
        Strategy: Assign each item ke cluster dengan minimum current weight
        untuk memastikan balanced distribution.
        
        Args:
            items (list): List of item dicts atau tuples
            
        Returns:
            list: List of clusters, each cluster adalah list of items
        """
        if len(items) == 0:
            return []
        
        # Initialize clusters dengan berat nol
        self.clusters = [[] for _ in range(self.num_clusters)]
        self.cluster_weights = [0.0] * self.num_clusters
        
        # Sort items by volume (descending) untuk better distribution
        sorted_items = sorted(
            items,
            key=lambda x: get_item_dims(x)[0] * get_item_dims(x)[1] * get_item_dims(x)[2],
            reverse=True,
        )
        
        # Greedy assignment: assign each item ke cluster dengan min weight
        for item in sorted_items:
            l, w, h = get_item_dims(item)
            item_weight = l * w * h  # Use volume as proxy for weight
            
            # Find cluster dengan minimum current weight
            min_cluster_idx = np.argmin(self.cluster_weights)
            
            # Assign item ke cluster
            self.clusters[min_cluster_idx].append(item)
            self.cluster_weights[min_cluster_idx] += item_weight
        
        return self.clusters
    
    def compute_load_balance(self):
        """
        Compute load balance coefficient.
        
        LB = 1.0 jika perfectly balanced, < 1.0 jika imbalanced
        
        Returns:
            float: Load balance coefficient (0.0 to 1.0)
        """
        if len(self.cluster_weights) == 0:
            return 1.0
        
        total_weight = sum(self.cluster_weights)
        if total_weight == 0:
            return 1.0
        
        # Compute variance dari cluster weights
        avg_weight = total_weight / len(self.cluster_weights)
        variance = sum((w - avg_weight) ** 2 for w in self.cluster_weights) / len(self.cluster_weights)
        std_dev = np.sqrt(variance)
        
        # Load balance: inverse of coefficient of variation
        if avg_weight > 0:
            cv = std_dev / avg_weight
            lb = 1.0 / (1.0 + cv)
        else:
            lb = 1.0
        
        return lb
    
    def compute_center_of_gravity(self, cluster_idx):
        """
        Compute center of gravity untuk cluster.
        
        Args:
            cluster_idx (int): Index dari cluster
            
        Returns:
            float: X-coordinate dari center of gravity (normalized)
        """
        if cluster_idx >= len(self.clusters) or len(self.clusters[cluster_idx]) == 0:
            return 0.5
        
        cluster = self.clusters[cluster_idx]
        total_volume = sum(
            get_item_dims(item)[0] * get_item_dims(item)[1] * get_item_dims(item)[2]
            for item in cluster
        )
        
        if total_volume == 0:
            return 0.5
        
        weighted_sum = sum(
            (get_item_dims(item)[0] / 2.0)
            * get_item_dims(item)[0]
            * get_item_dims(item)[1]
            * get_item_dims(item)[2]
            for item in cluster
        )
        cog = weighted_sum / total_volume / 100.0  # Normalize to [0, 1]
        
        return np.clip(cog, 0.0, 1.0)
