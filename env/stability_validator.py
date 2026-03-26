"""Load-Balanced Container Packing (LBCP) Stability Validator."""

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path


class StabilityValidator:
    """
    Load-Balanced Container Packing (LBCP) Stability Validator.
    
    Handles physical stability checking untuk items dalam container:
    - Support cell extraction
    - Convex hull computation
    - Center of gravity validation
    - Overhang detection
    """
    
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def is_stable(height_map, x, y, l, w, h, max_height, strict_mode=False):
        """
        Stability check: memastikan benda stabil dengan memvalidasi:
        1. Tinggi maksimum tidak melebihi max_height
        2. Ada minimal 3 support cells (hanya di strict_mode)
        3. Convex hull dapat dihitung (hanya di strict_mode)
        4. CoG berada di dalam convex hull (hanya di strict_mode)
        
        Args:
            height_map: 2D array tinggi dari HeightMap
            x, y: koordinat awal region
            l, w: panjang dan lebar region item
            h: tinggi item
            max_height: tinggi maksimum yang diizinkan
            strict_mode: Jika False (default), hanya check height overflow.
                        Jika True, check semua kondisi termasuk support cells & CoG
            
        Returns:
            bool: True jika stable, False jika unstable
        """
        # Check height overflow (ALWAYS)
        base_height = np.max(height_map[x:x+l, y:y+w])
        if base_height + h > max_height:
            return False

        # Strict mode: check support cells dan CoG
        if strict_mode:
            # Support cell extraction
            support_cells = StabilityValidator.compute_support_cells(height_map, x, y, l, w, base_height)

            # Minimal 3 support cells untuk stable
            if len(support_cells) < 3:
                return False

            # Convex hull computation
            try:
                hull_points, _ = StabilityValidator.compute_convex_hull(support_cells)
            except Exception:
                # Jika convex hull gagal (collinear), tidak stable
                return False

            # CoG computation: center of geometry dari region
            cog = np.array([x + l/2.0, y + w/2.0])

            # CoG inside polygon check
            return StabilityValidator.is_cog_inside_polygon(hull_points, cog)
        
        # Non-strict mode: height check is enough
        return True
