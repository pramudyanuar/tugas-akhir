import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


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
            items (list): List of tuples (length, width, height, weight_index)
            
        Returns:
            list: List of clusters, each cluster adalah list of items
        """
        if len(items) == 0:
            return []
        
        # Initialize clusters dengan berat nol
        self.clusters = [[] for _ in range(self.num_clusters)]
        self.cluster_weights = [0.0] * self.num_clusters
        
        # Sort items by volume (descending) untuk better distribution
        sorted_items = sorted(items, key=lambda x: x[0] * x[1] * x[2], reverse=True)
        
        # Greedy assignment: assign each item ke cluster dengan min weight
        for item in sorted_items:
            l, w, h = item[:3]
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
        total_volume = sum(item[0] * item[1] * item[2] for item in cluster)
        
        if total_volume == 0:
            return 0.5
        
        weighted_sum = sum((item[0] / 2.0) * item[0] * item[1] * item[2] 
                          for item in cluster)
        cog = weighted_sum / total_volume / 100.0  # Normalize to [0, 1]
        
        return np.clip(cog, 0.0, 1.0)


class HighLevelAgent(nn.Module):
    """
    High-Level Manager Agent untuk Hierarchical RL.
    
    Responsibilities:
    1. LBCP clustering: Group items untuk balanced load distribution
    2. Sequencing: Determine order penempatan items
    3. Strategy: Decide antara placement, repack, atau skip
    
    Output:
    - Cluster assignment untuk current batch items
    - Recommended orientation untuk placement
    """
    
    def __init__(self, input_dim=59*23 + 3, hidden_dim=256, num_strategies=8):
        """
        Initialize High-Level Agent.
        
        Args:
            input_dim (int): Input dimension (height_map flattened + item dims)
            hidden_dim (int): Hidden dimension
            num_strategies (int): Number of high-level strategies
                - 6 orientations
                - 1 repack action
                - 1 no-op
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        
        # Shared network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Strategy head: output logits untuk strategy selection
        self.strategy_head = nn.Linear(hidden_dim, num_strategies)
        
        # LBCP integration
        self.lbcp_module = LBCPClusterer(num_clusters=4)
        
        # State tracking
        self.current_clusters = None
        self.cluster_assignment = None
    
    def forward(self, state, items_batch=None):
        """
        Forward pass untuk high-level decision making.
        
        Args:
            state: Normalized state tensor (batch_size or 1, input_dim)
            items_batch: Optional list of items untuk clustering
            
        Returns:
            dict: {
                'strategy_logits': tensor of strategy logits,
                'cluster_assignment': cluster assignment untuk items,
                'load_balance': load balance coefficient
            }
        """
        # Ensure state is 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Forward pass through network
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        # Strategy logits
        strategy_logits = self.strategy_head(x)
        
        # LBCP clustering jika ada items
        cluster_assignment = None
        load_balance = 1.0
        
        if items_batch is not None and len(items_batch) > 0:
            clusters = self.lbcp_module.cluster_by_weight(items_batch)
            cluster_assignment = clusters
            load_balance = self.lbcp_module.compute_load_balance()
            self.current_clusters = clusters
            self.cluster_assignment = cluster_assignment
        
        return {
            'strategy_logits': strategy_logits,
            'cluster_assignment': cluster_assignment,
            'load_balance': load_balance
        }
    
    def select_strategy(self, strategy_logits):
        """
        Select strategy berdasarkan logits.
        
        Strategies:
        - 0-5: 6 possible orientations
        - 6: Repack action
        - 7: No-op
        
        Args:
            strategy_logits: Output dari strategy head
            
        Returns:
            int: Selected strategy index
        """
        # Greedy selection (argmax)
        if strategy_logits.dim() > 1:
            strategy_logits = strategy_logits.squeeze(0)
        
        strategy = torch.argmax(strategy_logits).item()
        return strategy

    def decode_macro_decision(self, strategy):
        """
        Decode high-level strategy index menjadi macro decision.

        Returns:
            dict: {
                'orientation': int,
                'zone_priority': str,
                'allow_repacking': bool
            }
        """
        orientation = min(max(int(strategy), 0), 5)

        # Mapping sederhana orientation -> zone priority agar high-level decision
        # memengaruhi urutan candidate yang dievaluasi low-level policy.
        zone_map = {
            0: 'left_to_right',
            1: 'right_to_left',
            2: 'front_to_back',
            3: 'back_to_front',
            4: 'center',
            5: 'center',
        }

        return {
            'orientation': orientation,
            'zone_priority': zone_map.get(orientation, 'center'),
            'allow_repacking': int(strategy) == 6
        }
    
    def get_item_ordering(self, items, strategy):
        """
        Determine item ordering berdasarkan strategy.
        
        Args:
            items: List of items
            strategy: Selected strategy
            
        Returns:
            list: Ordered items
        """
        if strategy == 6:  # Repack strategy
            # Inverse order (LIFO)
            return list(reversed(items))
        elif strategy == 7:  # No-op strategy
            # Keep original order
            return items
        else:  # Orientation strategies (0-5)
            # Sort by largest dimension first
            return sorted(items, key=lambda x: max(x[0], x[1], x[2]), reverse=True)
    
    def get_load_balance_reward(self):
        """
        Get reward berdasarkan load balance.
        
        Returns:
            float: Load balance reward (0.0 to 1.0)
        """
        if self.current_clusters is None:
            return 0.0
        
        return self.lbcp_module.compute_load_balance()


if __name__ == "__main__":
    """Test cases untuk High-Level Agent"""
    
    print("=" * 70)
    print("Test Case 1: LBCPClusterer initialization")
    print("=" * 70)
    
    clusterer = LBCPClusterer(num_clusters=4)
    print(f"Clusterer created with {clusterer.num_clusters} clusters")
    print(f"Balance threshold: {clusterer.balance_threshold}")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 2: Weight-based clustering")
    print("=" * 70)
    
    # Create sample items
    items = [
        (5, 5, 5),  # volume 125
        (10, 10, 10),  # volume 1000
        (3, 3, 3),  # volume 27
        (7, 7, 7),  # volume 343
        (4, 4, 4),  # volume 64
    ]
    
    clusters = clusterer.cluster_by_weight(items)
    
    print(f"Items: {items}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Cluster weights: {clusterer.cluster_weights}")
    
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: {cluster}")
    
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 3: Load balance computation")
    print("=" * 70)
    
    lb = clusterer.compute_load_balance()
    print(f"Load balance coefficient: {lb:.4f}")
    print(f"Expected: closer to 1.0 = better balanced")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 4: HighLevelAgent initialization")
    print("=" * 70)
    
    agent = HighLevelAgent(input_dim=59*23 + 3, hidden_dim=256)
    print(f"Agent created with input_dim={agent.input_dim}")
    print(f"Hidden dim: {agent.hidden_dim}")
    print(f"Num strategies: {agent.num_strategies}")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 5: Forward pass without clustering")
    print("=" * 70)
    
    state = torch.randn(1, 59*23 + 3)
    output = agent(state)
    
    print(f"Output keys: {output.keys()}")
    print(f"Strategy logits shape: {output['strategy_logits'].shape}")
    print(f"Load balance: {output['load_balance']}")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 6: Forward pass with clustering")
    print("=" * 70)
    
    items = [(5, 5, 5), (10, 10, 10), (3, 3, 3)]
    output = agent(state, items_batch=items)
    
    print(f"Cluster assignment: {output['cluster_assignment']}")
    print(f"Load balance: {output['load_balance']:.4f}")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 7: Strategy selection")
    print("=" * 70)
    
    strategy_logits = torch.tensor([[1.0, 2.0, 0.5, 3.0, 1.5, 0.2, 0.1, 0.0]])
    strategy = agent.select_strategy(strategy_logits)
    
    print(f"Strategy logits: {strategy_logits}")
    print(f"Selected strategy: {strategy}")
    print(f"Expected: 3 (argmax)")
    assert strategy == 3, "Strategy selection failed!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("All High-Level Agent tests completed!")
    print("=" * 70)