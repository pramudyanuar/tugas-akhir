"""Load-Balanced Container Packing (LBCP) Stability Validator."""

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import zlib
from collections import OrderedDict

# Optional numba JIT for geometric functions
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        # Support both @njit and @njit(...) usage when numba is unavailable.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(f):
            return f

        return decorator


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
    def clear_cache():
        """Clear validation cache to free memory (call periodically between episodes)."""
        if hasattr(StabilityValidator, '_validate_cache'):
            StabilityValidator._validate_cache.clear()
    
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
        region = height_map[x:x + l, y:y + w]
        if region.size == 0:
            return np.array([]).reshape(0, 2)

        local_coords = np.argwhere(region == base_height)
        if local_coords.size == 0:
            return np.array([]).reshape(0, 2)

        local_coords = local_coords.astype(np.int32, copy=False)
        local_coords[:, 0] += int(x)
        local_coords[:, 1] += int(y)
        return local_coords

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

    @staticmethod
    def _compute_cog_set(x, y, l, w, cog_tolerance):
        """
        Compute CoG set as a small square around the center point.

        This approximates Eq. 1 tolerance region with axis-aligned offsets.
        """
        cx = x + l / 2.0
        cy = y + w / 2.0

        if cog_tolerance is None or cog_tolerance <= 0:
            return np.array([[cx, cy]])

        delta = float(cog_tolerance)
        return np.array(
            [
                [cx - delta, cy - delta],
                [cx - delta, cy + delta],
                [cx + delta, cy - delta],
                [cx + delta, cy + delta],
                [cx, cy],
            ]
        )

    @staticmethod
    def validate(new_object, load_configuration, height_map_t, feasibility_map_t, cog_tolerance):
        """
        Algorithm 1: Structural Stability Validation.

        Args:
            new_object (dict): {'x': xi, 'y': yi, 'w': wi, 'd': di}
            load_configuration: Unused (kept for compatibility)
            height_map_t (np.ndarray): Height map at time t
            feasibility_map_t (np.ndarray): Feasibility map at time t
            cog_tolerance (float): CoG tolerance delta

        Returns:
            tuple: (valid, support_polygon, support_height)
        """
        # Simple LRU cache to avoid repeating expensive hull/point-inside checks
        if not hasattr(StabilityValidator, '_validate_cache'):
            StabilityValidator._validate_cache = OrderedDict()
            StabilityValidator._validate_cache_max = 4096

        if new_object is None:
            return False, np.array([]).reshape(0, 2), None

        xi = int(new_object.get('x', 0))
        yi = int(new_object.get('y', 0))
        wi = int(new_object.get('w', 0))
        di = int(new_object.get('d', 0))

        if wi <= 0 or di <= 0:
            return False, np.array([]).reshape(0, 2), None

        hm = height_map_t
        fm = feasibility_map_t

        region = hm[xi:xi + wi, yi:yi + di]
        if region.size == 0:
            return False, np.array([]).reshape(0, 2), None

        # Algorithm 1 line 10: support height is max height in the region.
        support_height = np.max(region)

        # Compute small fingerprints for caching (cheap checksum)
        try:
            hm_hash = zlib.adler32(region.tobytes())
        except Exception:
            hm_hash = None
        fm_hash = None
        try:
            if fm is not None:
                fm_region = fm[xi:xi + wi, yi:yi + di]
                fm_hash = zlib.adler32(fm_region.tobytes())
        except Exception:
            fm_hash = None

        cache_key = (int(xi), int(yi), int(wi), int(di), int(support_height), hm_hash, fm_hash, float(cog_tolerance))
        cache = StabilityValidator._validate_cache
        if cache_key in cache:
            # Move to end (recently used)
            cache.move_to_end(cache_key)
            cached_valid, cached_hull_list, cached_support_height = cache[cache_key]
            hull_arr = np.array(cached_hull_list) if cached_hull_list is not None and len(cached_hull_list) else np.array([]).reshape(0, 2)
            return cached_valid, hull_arr, cached_support_height

        # Algorithm 1 line 11: contact points at support height
        contact_cells = StabilityValidator.compute_support_cells(
            hm, xi, yi, wi, di, support_height
        )

        if contact_cells.size == 0:
            return False, np.array([]).reshape(0, 2), support_height

        # Algorithm 1 line 12: feasible points within region
        feasible_cells = []
        for px in range(xi, xi + wi):
            for py in range(yi, yi + di):
                if fm is None or fm[px, py]:
                    feasible_cells.append([px, py])

        feasible_cells = np.array(feasible_cells) if feasible_cells else np.array([]).reshape(0, 2)

        if feasible_cells.size == 0:
            return False, np.array([]).reshape(0, 2), support_height

        # Algorithm 1 line 13: intersection of contact and feasible sets
        contact_set = {tuple(p) for p in contact_cells}
        feasible_set = {tuple(p) for p in feasible_cells}
        intersection = contact_set & feasible_set
        if not intersection:
            return False, np.array([]).reshape(0, 2), support_height

        support_points = np.array(list(intersection))

        # Compute support polygon (convex hull)
        try:
            hull_points, _ = StabilityValidator.compute_convex_hull(support_points)
        except Exception:
            # Cache negative result to avoid repeated convex hull attempts
            cache[cache_key] = (False, None, support_height)
            # enforce max size
            if len(cache) > StabilityValidator._validate_cache_max:
                cache.popitem(last=False)
            return False, np.array([]).reshape(0, 2), support_height

        # Compute CoG set and check containment
        cog_set = StabilityValidator._compute_cog_set(xi, yi, wi, di, cog_tolerance)
        path = Path(hull_points)
        inside_mask = path.contains_points(cog_set)
        is_valid = bool(np.all(inside_mask))

        # Store into cache (store hull as list for lightness)
        hull_list = hull_points.tolist() if hull_points is not None and len(hull_points) else None
        cache[cache_key] = (is_valid, hull_list, support_height)
        if len(cache) > StabilityValidator._validate_cache_max:
            cache.popitem(last=False)

        return is_valid, hull_points, support_height

    @staticmethod
    def update_feasibility_map(feasibility_map_t, updated_support_polygon):
        """
        Algorithm 2: Structural Stability Update.

        Marks all grid points inside the support polygon as feasible.
        """
        if feasibility_map_t is None:
            return None

        if updated_support_polygon is None or len(updated_support_polygon) < 3:
            return feasibility_map_t

        fm = np.array(feasibility_map_t, copy=True)
        L, W = fm.shape

        polygon = np.asarray(updated_support_polygon)
        path = Path(polygon)

        xs, ys = np.meshgrid(np.arange(L), np.arange(W), indexing='ij')
        points = np.stack([xs.ravel(), ys.ravel()], axis=1)
        inside = path.contains_points(points, radius=1e-9)
        fm.reshape(-1)[inside] = True

        # Ensure polygon vertices are included even if on boundary
        for vx, vy in polygon:
            ix, iy = int(round(vx)), int(round(vy))
            if 0 <= ix < L and 0 <= iy < W:
                fm[ix, iy] = True

        return fm
