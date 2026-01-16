import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def compute_sdf_sampling(p_start, p_end, polygons, n_samples=1000):
    """Compute SDF along line segment."""
    t = np.linspace(0, 1, n_samples)
    points = p_start + t[:, None] * (p_end - p_start)
    
    sdf = np.full(len(points), np.inf)
    all_bounds = []
    
    for poly in polygons:
        v = np.asarray(poly)
        all_bounds.append((v.min(axis=0), v.max(axis=0)))
        
        # Compute distance to this polygon's edges
        poly_distances = np.full(len(points), np.inf)
        for i in range(len(v)):
            v1, v2 = v[i], v[(i + 1) % len(v)]
            edge_vec = v2 - v1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            
            if edge_len_sq < 1e-10:
                continue
            
            t_closest = np.clip(np.dot(points - v1, edge_vec) / edge_len_sq, 0, 1)
            closest_points = v1 + t_closest[:, None] * edge_vec
            edge_distances = np.linalg.norm(points - closest_points, axis=1)
            poly_distances = np.minimum(poly_distances, edge_distances)
        
        # Ray casting for inside/outside
        v_next = np.roll(v, -1, axis=0)
        py = points[:, 1:2]
        px = points[:, 0:1]

        v_y, v_next_y = v[:, 1], v_next[:, 1]
        v_x, v_next_x = v[:, 0], v_next[:, 0]

        # Edges that cross the horizontal ray
        crosses = ((v_y <= py) & (py < v_next_y)) | ((v_next_y <= py) & (py < v_y))

        # Safe division - mask out near-horizontal edges
        dy = v_next_y - v_y
        valid = np.abs(dy) > 1e-12
        safe_dy = np.where(valid, dy, 1.0)

        t_cross = (py - v_y) / safe_dy
        x_cross = v_x + t_cross * (v_next_x - v_x)

        # Only count valid crossings to the right
        crossings_right = crosses & valid & (x_cross > px)
        inside_poly = np.sum(crossings_right, axis=1) % 2 == 1
        
        poly_sdf = np.where(inside_poly, -poly_distances, poly_distances)
        
        # Union: take minimum SDF
        sdf = np.minimum(sdf, poly_sdf)
    
    print(f"  SDF range: [{np.min(sdf):.6f}, {np.max(sdf):.6f}]")
    print(f"  Zero crossings: {np.sum(np.diff(np.sign(sdf)) != 0)}")
    for i, (bmin, bmax) in enumerate(all_bounds):
        print(f"  Polygon {i} bounds: [{bmin}, {bmax}]")
    
    return t, sdf

def extract_all_lines(node, depth=0, path='', result=None):
    """Recursively extract all (p_start, p_end) tuples with labels."""
    if result is None:
        result = []
    if node is None:
        return result
    
    for pc in ['pc1', 'pc2']:
        if pc in node:
            label = f"D{depth}: {path}{pc}" if path else f"D{depth}: {pc}"
            result.append((node[pc], label))
    
    for quad in ['++', '+-', '--', '-+']:
        if quad in node:
            extract_all_lines(node[quad], depth + 1, f"{path}{quad}/", result)
    
    return result
