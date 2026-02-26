import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def compute_support_cells(height_map, x, y, l, w, base_height):
    region = height_map[x:x+l, y:y+w]
    coords = []

    for i in range(l):
        for j in range(w):
            if region[i, j] == base_height:
                coords.append([x+i, y+j])

    return np.array(coords)

def is_stable(height_map, x, y, l, w, h, max_height):
    base_height = np.max(height_map[x:x+l, y:y+w])

    if base_height + h > max_height:
        return False

    support_cells = compute_support_cells(height_map, x, y, l, w, base_height)

    if len(support_cells) < 3:
        return False

    try:
        hull = ConvexHull(support_cells)
        hull_points = support_cells[hull.vertices]
    except Exception:
        # If ConvexHull fails (collinear points), consider as stable if enough support
        return len(support_cells) >= 3

    cog = np.array([x + l/2.0, y + w/2.0])

    path = Path(hull_points)
    return path.contains_point(cog)