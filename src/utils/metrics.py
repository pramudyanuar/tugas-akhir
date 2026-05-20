import numpy as np
from typing import List, Tuple

from src.utils.item_utils import get_item_dims

class Metrics:
    """
    Metrics calculator untuk mengukur performa 3D bin packing.
    
    Features:
    - Hitung utilization % : persentase ruang yang digunakan
    - Hitung success rate : persentase item yang berhasil ditempatkan
    - Hitung average height distribution : distribusi tinggi dari items
    """
    
    @staticmethod
    def calculate_utilization(placed_items: List[Tuple[int, int, int]], 
                             container_dims: Tuple[int, int, int]) -> float:
        """
        Hitung utilization percentage dari container.
        
        Utilization = (total volume dari placed items / total volume container) * 100%
        
        Args:
            placed_items: List of item dicts atau tuples
            container_dims: Tuple (L, W, H) dimensi container
            
        Returns:
            float: Utilization percentage (0-100)
        """
        if not placed_items or len(placed_items) == 0:
            return 0.0
        
        container_volume = container_dims[0] * container_dims[1] * container_dims[2]
        if container_volume == 0:
            raise ValueError("Container dimensions must be positive")
        
        # Hitung total volume dari placed items
        total_placed_volume = sum(
            get_item_dims(item)[0] * get_item_dims(item)[1] * get_item_dims(item)[2]
            for item in placed_items
        )
        
        utilization = (total_placed_volume / container_volume) * 100.0
        
        return min(utilization, 100.0)  # Cap at 100%
    
    @staticmethod
    def calculate_success_rate(num_placed: int, total_items: int) -> float:
        """
        Hitung success rate dari item placement.
        
        Success Rate = (num_placed / total_items) * 100%
        
        Args:
            num_placed (int): Jumlah items yang berhasil ditempatkan
            total_items (int): Total jumlah items yang ingin ditempatkan
            
        Returns:
            float: Success rate percentage (0-100)
        """
        if total_items == 0:
            raise ValueError("total_items harus positif")
        
        if num_placed > total_items:
            raise ValueError(f"num_placed ({num_placed}) tidak bisa lebih dari total_items ({total_items})")
        
        success_rate = (num_placed / total_items) * 100.0
        
        return success_rate
    
    @staticmethod
    def calculate_average_height_distribution(placed_items: List[Tuple[int, int, int]],
                                             positions: List[Tuple[int, int, int]] = None) -> dict:
        """
        Hitung average height distribution dari placed items.
        
        Menghitung:
        - Average height dari semua items
        - Max height yang dicapai
        - Min height
        - Standard deviation dari heights
        - Height distribution histogram
        
        Args:
            placed_items: List of item dicts atau tuples
            positions: Optional list of tuples (x, y, z) posisi items
            
        Returns:
            dict: Dictionary berisi:
                - 'average': rata-rata tinggi
                - 'max': tinggi maksimum
                - 'min': tinggi minimum
                - 'std_dev': standard deviation
                - 'count': jumlah items
                - 'histogram': distribusi height dalam bins
        """
        if not placed_items or len(placed_items) == 0:
            return {
                'average': 0.0,
                'max': 0,
                'min': 0,
                'std_dev': 0.0,
                'count': 0,
                'histogram': {}
            }
        
        heights = np.array([get_item_dims(item)[2] for item in placed_items])
        
        # Jika positions diberikan, hitung actual stacking height
        if positions:
            actual_heights = np.array([
                pos[2] + get_item_dims(item)[2]
                for pos, item in zip(positions, placed_items)
            ])
        else:
            actual_heights = heights
        
        # Calculate statistics
        avg_height = float(np.mean(actual_heights))
        max_height = int(np.max(actual_heights))
        min_height = int(np.min(actual_heights))
        std_dev = float(np.std(actual_heights))
        
        # Create histogram (bins by height ranges)
        bins = np.arange(0, max_height + 2, max(1, max_height // 10 + 1))
        hist, bin_edges = np.histogram(actual_heights, bins=bins)
        
        histogram = {}
        for i in range(len(hist)):
            bin_label = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
            histogram[bin_label] = int(hist[i])
        
        return {
            'average': avg_height,
            'max': max_height,
            'min': min_height,
            'std_dev': std_dev,
            'count': len(placed_items),
            'histogram': histogram
        }
    
    @staticmethod
    def print_metrics_report(placed_items: List[Tuple[int, int, int]],
                             total_items: int,
                             container_dims: Tuple[int, int, int],
                             positions: List[Tuple[int, int, int]] = None) -> None:
        """
        Print comprehensive metrics report.
        
        Args:
            placed_items: List of placed items (length, width, height)
            total_items: Total items to place
            container_dims: Container dimensions (L, W, H)
            positions: Optional positions of items
        """
        num_placed = len(placed_items)
        utilization = Metrics.calculate_utilization(placed_items, container_dims)
        success_rate = Metrics.calculate_success_rate(num_placed, total_items)
        height_dist = Metrics.calculate_average_height_distribution(placed_items, positions)
        
        print("\n" + "=" * 70)
        print("PACKING METRICS REPORT")
        print("=" * 70)
        
        print(f"\n📦 PLACEMENT METRICS:")
        print(f"  Items Placed:        {num_placed}/{total_items}")
        print(f"  Success Rate:        {success_rate:.2f}%")
        
        print(f"\n📊 UTILIZATION METRICS:")
        print(f"  Container:           {container_dims[0]} x {container_dims[1]} x {container_dims[2]}")
        print(f"  Container Volume:    {container_dims[0] * container_dims[1] * container_dims[2]} units³")
        print(f"  Utilization:         {utilization:.2f}%")
        
        print(f"\n📈 HEIGHT DISTRIBUTION:")
        print(f"  Average Height:      {height_dist['average']:.2f}")
        print(f"  Max Height:          {height_dist['max']}")
        print(f"  Min Height:          {height_dist['min']}")
        print(f"  Std Deviation:       {height_dist['std_dev']:.2f}")
        
        if height_dist['histogram']:
            print(f"\n  Height Histogram:")
            for bin_label, count in sorted(height_dist['histogram'].items()):
                bar = "█" * count
                print(f"    {bin_label:>10}: {bar} ({count})")
        
        print("\n" + "=" * 70 + "\n")

