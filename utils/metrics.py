import numpy as np
from typing import List, Tuple

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
            placed_items: List of tuples (length, width, height) untuk items yang ditempatkan
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
        total_placed_volume = sum(item[0] * item[1] * item[2] for item in placed_items)
        
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
            placed_items: List of tuples (length, width, height)
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
        
        heights = np.array([item[2] for item in placed_items])
        
        # Jika positions diberikan, hitung actual stacking height
        if positions:
            actual_heights = np.array([pos[2] + item[2] for pos, item in zip(positions, placed_items)])
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


if __name__ == "__main__":
    """Test cases untuk Metrics"""
    
    print("=" * 70)
    print("Test Case 1: Hitung Utilization %")
    print("=" * 70)
    
    # Test case 1: Good utilization
    placed_items_1 = [
        (10, 10, 5),   # 500 volume
        (8, 8, 6),     # 384 volume
        (6, 6, 4)      # 144 volume
    ]
    container = (30, 30, 30)  # 27000 volume
    
    utilization = Metrics.calculate_utilization(placed_items_1, container)
    print(f"Placed items: {len(placed_items_1)}")
    print(f"Items: {placed_items_1}")
    print(f"Container: {container} = {container[0]*container[1]*container[2]} units³")
    print(f"Total placed volume: {sum(item[0]*item[1]*item[2] for item in placed_items_1)} units³")
    print(f"Utilization: {utilization:.2f}%")
    assert 0 <= utilization <= 100, "Utilization out of range!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 2: Hitung Success Rate")
    print("=" * 70)
    
    # Test case 2a: Perfect placement
    success_rate_100 = Metrics.calculate_success_rate(10, 10)
    print(f"Placed 10/10 items: Success Rate = {success_rate_100:.2f}%")
    assert success_rate_100 == 100.0, "Should be 100%!"
    print("✓ PASSED")
    
    # Test case 2b: Partial placement
    success_rate_50 = Metrics.calculate_success_rate(5, 10)
    print(f"Placed 5/10 items: Success Rate = {success_rate_50:.2f}%")
    assert success_rate_50 == 50.0, "Should be 50%!"
    print("✓ PASSED")
    
    # Test case 2c: No placement
    success_rate_0 = Metrics.calculate_success_rate(0, 10)
    print(f"Placed 0/10 items: Success Rate = {success_rate_0:.2f}%")
    assert success_rate_0 == 0.0, "Should be 0%!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 3: Hitung Average Height Distribution")
    print("=" * 70)
    
    # Test case 3: Height distribution
    placed_items_3 = [
        (5, 5, 2),    # height = 2
        (5, 5, 3),    # height = 3
        (5, 5, 4),    # height = 4
        (5, 5, 5),    # height = 5
    ]
    
    distribution = Metrics.calculate_average_height_distribution(placed_items_3)
    
    print(f"Placed items and heights: {placed_items_3}")
    print(f"Average Height: {distribution['average']:.2f}")
    print(f"Max Height: {distribution['max']}")
    print(f"Min Height: {distribution['min']}")
    print(f"Std Deviation: {distribution['std_dev']:.2f}")
    print(f"Item Count: {distribution['count']}")
    
    assert distribution['average'] == 3.5, "Average should be 3.5"
    assert distribution['max'] == 5, "Max should be 5"
    assert distribution['min'] == 2, "Min should be 2"
    assert distribution['count'] == 4, "Count should be 4"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 4: Height Distribution dengan Positions (Stacking)")
    print("=" * 70)
    
    placed_items_4 = [
        (5, 5, 2),    
        (5, 5, 3),    
        (5, 5, 4),    
    ]
    positions_4 = [
        (0, 0, 0),    # Item 1 di z=0, top at z=2
        (5, 0, 2),    # Item 2 di z=2, top at z=5
        (10, 0, 5),   # Item 3 di z=5, top at z=9
    ]
    
    distribution_stacked = Metrics.calculate_average_height_distribution(placed_items_4, positions_4)
    
    print(f"Items with positions (z_base, height):")
    for i, (item, pos) in enumerate(zip(placed_items_4, positions_4)):
        top_height = pos[2] + item[2]
        print(f"  Item {i+1}: z_base={pos[2]}, height={item[2]}, top={top_height}")
    
    print(f"\nAverage Stack Height: {distribution_stacked['average']:.2f}")
    print(f"Max Stack Height: {distribution_stacked['max']}")
    
    expected_avg = (2 + 5 + 9) / 3
    assert abs(distribution_stacked['average'] - expected_avg) < 0.01, f"Expected {expected_avg}, got {distribution_stacked['average']}"
    assert distribution_stacked['max'] == 9, "Max should be 9"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 5: Metrics Report")
    print("=" * 70)
    
    placed_items_report = [
        (10, 8, 5),
        (8, 6, 4),
        (6, 5, 3),
        (5, 5, 2),
    ]
    total_items_report = 10
    container_report = (30, 30, 20)
    positions_report = [
        (0, 0, 0),
        (10, 0, 5),
        (18, 0, 9),
        (24, 0, 12),
    ]
    
    Metrics.print_metrics_report(placed_items_report, total_items_report, 
                                 container_report, positions_report)
    
    print("=" * 70)
    print("All Metrics tests completed!")
    print("=" * 70)
