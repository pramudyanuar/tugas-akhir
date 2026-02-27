import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def compute_support_cells(height_map, x, y, l, w, base_height):
    """
    Support cell extraction: mengekstrak semua sel dari region yang bertumpu
    pada base_height (tinggi maksimum di region).
    
    Args:
        height_map: 2D array tinggi
        x, y: koordinat awal region
        l, w: panjang dan lebar region
        base_height: tinggi dasar (tempat benda bertumpu)
        
    Returns:
        numpy array: koordinat [x, y] dari setiap support cell
    """
    region = height_map[x:x+l, y:y+w]
    coords = []

    for i in range(l):
        for j in range(w):
            if region[i, j] == base_height:
                coords.append([x+i, y+j])

    return np.array(coords) if coords else np.array([]).reshape(0, 2)

def compute_convex_hull(support_cells):
    """
    Convex hull: menghitung convex hull dari support cells.
    
    Args:
        support_cells: numpy array dengan koordinat support cells
        
    Returns:
        tuple: (hull_points, success) 
               hull_points: vertices dari convex hull
               success: True jika convex hull berhasil dihitung
               
    Raises:
        Exception: jika convex hull gagal dihitung
    """
    if len(support_cells) < 3:
        raise ValueError(f"Convex hull memerlukan minimal 3 points, got {len(support_cells)}")
    
    if len(support_cells) == 3:
        # Jika tepat 3 points, return semua sebagai hull
        return support_cells, True
    
    try:
        hull = ConvexHull(support_cells)
        hull_points = support_cells[hull.vertices]
        return hull_points, True
    except Exception as e:
        # Collinear points, tidak bisa membentuk convex hull
        raise Exception(f"Convex hull computation failed: {str(e)}")

def is_cog_inside_polygon(hull_points, cog):
    """
    CoG inside polygon check: memastikan center of gravity berada di dalam
    convex hull dari support cells.
    
    Args:
        hull_points: vertices dari convex hull
        cog: center of gravity [x, y]
        
    Returns:
        bool: True jika CoG berada di dalam polygon
    """
    if len(hull_points) < 3:
        return False
    
    # Gunakan matplotlib.path.Path untuk point-in-polygon test
    path = Path(hull_points)
    is_inside = path.contains_point(cog)
    
    return bool(is_inside)

def is_stable(height_map, x, y, l, w, h, max_height):
    """
    Stability check: memastikan benda stabil dengan memvalidasi:
    1. Tinggi maksimum tidak melebihi max_height
    2. Ada minimal 3 support cells
    3. Convex hull dapat dihitung
    4. CoG berada di dalam convex hull
    
    Args:
        height_map: 2D array tinggi dari HeightMap
        x, y: koordinat awal region
        l, w: panjang dan lebar region item
        h: tinggi item
        max_height: tinggi maksimum yang diizinkan
        
    Returns:
        bool: True jika stable, False jika unstable
    """
    # Check height overflow
    base_height = np.max(height_map[x:x+l, y:y+w])
    if base_height + h > max_height:
        return False

    # Support cell extraction
    support_cells = compute_support_cells(height_map, x, y, l, w, base_height)

    # Minimal 3 support cells untuk stable
    if len(support_cells) < 3:
        return False

    # Convex hull computation
    try:
        hull_points, _ = compute_convex_hull(support_cells)
    except Exception:
        # Jika convex hull gagal (collinear), tidak stable
        return False

    # CoG computation: center of geometry dari region
    cog = np.array([x + l/2.0, y + w/2.0])

    # CoG inside polygon check
    return is_cog_inside_polygon(hull_points, cog)


if __name__ == "__main__":
    """Test cases untuk LBCP Stability"""
    
    # Test Case 1: Flat floor → stable
    print("=" * 60)
    print("Test Case 1: Flat floor → stable")
    print("=" * 60)
    height_map_flat = np.zeros((10, 10), dtype=np.int32)
    # Flat floor di z=0, item 3x3 dengan height 2
    result_flat = is_stable(height_map_flat, 2, 2, 3, 3, 2, 10)
    print(f"Flat floor (3x3 item on z=0): {result_flat}")
    print(f"Expected: True")
    assert result_flat == True, "Flat floor test failed!"
    print("✓ PASSED\n")

    # Test Case 2: Overhang kecil → unstable
    print("=" * 60)
    print("Test Case 2: Overhang kecil → unstable")
    print("=" * 60)
    height_map_overhang = np.zeros((15, 15), dtype=np.int32)
    # Create strong overhang: support cells hanya di satu sisi saja
    # Region akan menjadi 5x5, tapi support hanya di baris pertama
    height_map_overhang[3, 3:8] = 5    # Support hanya di 1 baris (y=3:8)
    height_map_overhang[4:8, 3:8] = 0  # Tidak ada support di baris lainnya
    result_overhang = is_stable(height_map_overhang, 3, 3, 5, 5, 2, 10)
    print(f"Overhang kecil (5x5 item with single row support): {result_overhang}")
    print(f"Expected: False (CoG outside support polygon/collinear)")
    assert result_overhang == False, "Overhang test failed!"
    print("✓ PASSED\n")

    # Test Case 3: Single support cell → unstable
    print("=" * 60)
    print("Test Case 3: Single support cell → unstable")
    print("=" * 60)
    height_map_single = np.zeros((10, 10), dtype=np.int32)
    # Minimal support cells (< 3)
    height_map_single[3, 3] = 5  # Only 1 support cell
    result_single = is_stable(height_map_single, 2, 2, 3, 3, 2, 10)
    print(f"Single support cell: {result_single}")
    print(f"Expected: False (need minimal 3 support cells)")
    assert result_single == False, "Single support cell test failed!"
    print("✓ PASSED\n")

    print("=" * 60)
    print("All LBCP Stability tests completed!")
    print("=" * 60)