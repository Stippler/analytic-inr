import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class AxisSegment:
    """Single PCA axis line segment with metrics."""
    start: np.ndarray
    end: np.ndarray
    variance: float
    n_left: int
    n_right: int
    axis_idx: int


@dataclass
class PCANode:
    """Node in recursive PCA tree."""
    center: np.ndarray
    axes: List[AxisSegment] = field(default_factory=list)
    children: List['PCANode'] = field(default_factory=list)
    depth: int = 0
    n_points: int = 0


def compute_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA. Returns (center, axes, variances)."""
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    return center, eigenvectors[:, order].T, eigenvalues[order]


def clip_line_to_halfspaces(origin: np.ndarray, direction: np.ndarray, 
                             planes: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    """
    Clip line to region defined by halfspace planes.
    Each plane is (normal, point) where normal points inward.
    Returns (t_min, t_max) or (inf, -inf) if no valid segment.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    t_min, t_max = -np.inf, np.inf
    
    for normal, point in planes:
        d_dot_n = np.dot(direction, normal)
        if abs(d_dot_n) > 1e-12:
            t = np.dot(point - origin, normal) / d_dot_n
            if d_dot_n > 0:
                t_min = max(t_min, t)
            else:
                t_max = min(t_max, t)
        elif np.dot(origin - point, normal) < -1e-12:
            return np.inf, -np.inf
    
    return (t_min, t_max) if t_min <= t_max else (np.inf, -np.inf)


def make_box_planes(bounds: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create inward-pointing halfspace planes for bounding box."""
    n_dims = bounds.shape[0]
    planes = []
    for i in range(n_dims):
        normal_neg = np.zeros(n_dims)
        normal_neg[i] = 1.0
        point_neg = bounds[i, 0] * normal_neg
        planes.append((normal_neg, point_neg))
        
        normal_pos = np.zeros(n_dims)
        normal_pos[i] = -1.0
        point_pos = bounds[i, 1] * normal_pos
        planes.append((normal_pos, point_pos))
    
    return planes


def recursive_pca(points: np.ndarray,
                  bounds: Optional[np.ndarray] = None,
                  planes: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                  min_points: int = 3,
                  min_variance: float = 1e-4,
                  max_depth: int = 4,
                  depth: int = 0) -> Optional[PCANode]:
    """
    Recursive PCA with boundary plane clipping.
    
    bounds: (n_dims, 2) array of [min, max] for each dimension
    planes: list of (normal, point) tuples defining halfspaces
    """
    n_dims = points.shape[1]
    
    if len(points) < min_points or depth >= max_depth:
        return None
    
    # Initialize bounds and planes if root
    if bounds is None:
        bounds = np.stack([points.min(axis=0), points.max(axis=0)], axis=1)
    if planes is None:
        planes = make_box_planes(bounds)
    
    center, pca_axes, variances = compute_pca(points)
    node = PCANode(center=center, depth=depth, n_points=len(points))
    
    # Process each PCA axis
    for axis_idx in range(n_dims):
        axis = pca_axes[axis_idx]
        variance = variances[axis_idx]
        
        if variance < min_variance:
            continue
        
        # Clip axis to boundary planes
        t_min, t_max = clip_line_to_halfspaces(center, axis, planes)
        if not np.isfinite(t_min) or not np.isfinite(t_max):
            continue
        
        start = center + t_min * axis
        end = center + t_max * axis
        
        projections = (points - center) @ axis
        n_left = (projections <= 0).sum()
        n_right = (projections >= 0).sum()
        
        node.axes.append(AxisSegment(
            start=start,
            end=end,
            variance=variance,
            n_left=n_left,
            n_right=n_right,
            axis_idx=axis_idx
        ))
    
    # Recurse into 2^n_dims subspaces
    projections = (points - center) @ pca_axes.T
    n_splits = 2 ** n_dims
    
    for split_idx in range(n_splits):
        mask = np.ones(len(points), dtype=bool)
        child_planes = list(planes)
        
        for axis_idx in range(n_dims):
            sign = 1 if (split_idx >> axis_idx) & 1 else -1
            axis_mask = (projections[:, axis_idx] * sign >= 0)
            mask &= axis_mask
            
            # Add splitting plane for this child
            child_planes.append((sign * pca_axes[axis_idx], center))
        
        subspace_points = points[mask]
        
        if len(subspace_points) >= min_points:
            child = recursive_pca(
                subspace_points,
                bounds=bounds,
                planes=child_planes,
                min_points=min_points,
                min_variance=min_variance,
                max_depth=max_depth,
                depth=depth + 1
            )
            if child is not None:
                node.children.append(child)
    
    return node


def flatten_tree(node: Optional[PCANode]) -> List[AxisSegment]:
    """Flatten PCA tree into list of all axis segments."""
    if node is None:
        return []
    
    segments = list(node.axes)
    for child in node.children:
        segments.extend(flatten_tree(child))
    
    return segments


def compute_sdf_sampling_3d(p_start, p_end, mesh, n_samples=200):
    """
    Compute SDF along line segment using trimesh proximity queries.
    
    Optimized version that uses normal direction for sign instead of contains(),
    which is much faster for large meshes.
    """
    t = np.linspace(0, 1, n_samples)
    points = p_start + t[:, None] * (p_end - p_start)
    
    # Get closest points and distances (fast with rtree if installed)
    closest_points, distances, triangle_id = mesh.nearest.on_surface(points)
    
    # Determine sign using surface normals (much faster than contains())
    # If point-to-surface vector points opposite to surface normal, we're inside
    face_normals = mesh.face_normals[triangle_id]
    to_surface = closest_points - points
    
    # Normalize to_surface vectors
    to_surface_norm = np.linalg.norm(to_surface, axis=1, keepdims=True)
    to_surface_norm = np.where(to_surface_norm > 1e-10, to_surface_norm, 1.0)
    to_surface_normalized = to_surface / to_surface_norm
    
    # Dot product: positive if same direction (outside), negative if opposite (inside)
    dot_products = np.sum(to_surface_normalized * face_normals, axis=1)
    
    # Sign: negative inside, positive outside
    sdf = np.where(dot_products < 0, -distances, distances)
    
    return t, sdf


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


def robust_pca(vertices: np.ndarray):
    """
    Compute robust PCA with deterministic orientation.
    Works for any dimension (2D, 3D, etc.).
    
    Args:
        vertices: (N, D) array of points
    
    Returns:
        center: (D,) centroid
        axes: (D, D) principal axes (rows are axes, sorted by variance)
        eigenvalues: (D,) variances along each axis
    """
    center = vertices.mean(axis=0)
    centered = vertices - center
    n_dims = vertices.shape[1]
    
    # Covariance
    cov = centered.T @ centered / len(vertices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort descending by variance
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = eigenvectors[:, order].T.copy()
    
    # Near-equal variance: prefer coordinate-alignment for most dominant axis
    # (Only check first two axes if they exist)
    if n_dims >= 2 and eigenvalues[0] > 1e-10:
        ratio = eigenvalues[1] / eigenvalues[0]
        if ratio > 0.99:
            # Swap if second axis is more aligned with first coordinate
            if abs(axes[1, 0]) > abs(axes[0, 0]):
                axes[[0, 1]] = axes[[1, 0]]
                eigenvalues[[0, 1]] = eigenvalues[[1, 0]]
    
    # Deterministic sign: point towards where more mass lies
    # Fallback to positive coordinate direction if mass is nearly balanced
    for i in range(n_dims):
        # Project centered points onto this axis
        projections = centered @ axes[i]
        mass_sum = projections.sum()
        
        # If mass is nearly balanced (within 10% of total spread)
        mass_threshold = 0.1 * np.abs(projections).sum()
        
        if abs(mass_sum) < mass_threshold:
            # Fallback: prefer positive direction along first non-zero coordinate
            for dim in range(n_dims):
                if abs(axes[i, dim]) > 1e-10:
                    if axes[i, dim] < 0:
                        axes[i] *= -1
                    break
        else:
            # Use mass-based direction
            if mass_sum < 0:
                axes[i] *= -1
    
    return center, axes, eigenvalues


def clip_line(origin, direction, halfspace_normals=None, halfspace_points=None, box_min=-1, box_max=1):
    """Clip line to region, return (t_min, t_max) or (None, None)."""
    direction = direction / np.linalg.norm(direction)
    t_min, t_max = -np.inf, np.inf
    
    # Clip to box
    for i in range(len(origin)):
        if abs(direction[i]) > 1e-12:
            t1 = (box_min - origin[i]) / direction[i]
            t2 = (box_max - origin[i]) / direction[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        elif origin[i] < box_min or origin[i] > box_max:
            return None, None
    
    # Clip to half-spaces
    if halfspace_normals is not None:
        for normal, point in zip(halfspace_normals, halfspace_points):
            d_dot_n = np.dot(direction, normal)
            if abs(d_dot_n) > 1e-12:
                t_boundary = np.dot(point - origin, normal) / d_dot_n
                if d_dot_n > 0:
                    t_min = max(t_min, t_boundary)
                else:
                    t_max = min(t_max, t_boundary)
            elif np.dot(point - origin, normal) < 0:
                return None, None
    
    return (t_min, t_max) if t_min <= t_max else (None, None)


def build_hierarchical_pca(vertices, max_depth=2, box_min=-1, box_max=1, 
                           halfspace_normals=None, halfspace_points=None):
    """
    Build hierarchical PCA. Returns dict: {'pc1': (p_start, p_end), 'pc2': (p_start, p_end), '++': {...}, ...}
    
    halfspace_normals/halfspace_points: accumulated constraints from ALL parent levels
    """
    if len(vertices) < 3:
        return None
    
    center, axes, variances = robust_pca(vertices)
    
    # Create line segments for PC1 and PC2
    lines = {}
    for i, pc_key in enumerate(['pc1', 'pc2']):
        t_range = clip_line(center, axes[i], halfspace_normals, halfspace_points, box_min, box_max)
        if t_range[0] is not None:
            direction = axes[i] / np.linalg.norm(axes[i])
            p_start = center + t_range[0] * direction
            p_end = center + t_range[1] * direction
            lines[pc_key] = (p_start, p_end)
    
    node = {'center': center, 'axes': axes, 'variances': variances, **lines}
    
    # Recurse into quadrants
    if max_depth > 0:
        proj = (vertices - center) @ axes.T
        for s1, s2, key in [(1, 1, '++'), (1, -1, '+-'), (-1, -1, '--'), (-1, 1, '-+')]:
            mask = ((np.sign(proj[:, 0]) == s1) | (proj[:, 0] == 0)) & \
                   ((np.sign(proj[:, 1]) == s2) | (proj[:, 1] == 0))
            
            # Accumulate half-spaces: add current level's constraints to existing ones
            new_normals = list(halfspace_normals) if halfspace_normals else []
            new_points = list(halfspace_points) if halfspace_points else []
            new_normals.extend([s1 * axes[0], s2 * axes[1]])
            new_points.extend([center, center])
            
            child = build_hierarchical_pca(vertices[mask], max_depth - 1, box_min, box_max,
                                          new_normals, new_points)
            if child is not None:
                node[key] = child
    
    return node


def simplify_sdf_to_knots(t_values, sdf_values, tolerance=0.005):
    """Simplify SDF to knots using Douglas-Peucker algorithm."""
    
    def perpendicular_distance(point_idx, start_idx, end_idx):
        x0, y0 = t_values[point_idx], sdf_values[point_idx]
        x1, y1 = t_values[start_idx], sdf_values[start_idx]
        x2, y2 = t_values[end_idx], sdf_values[end_idx]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / np.sqrt(dx**2 + dy**2)
    
    def douglas_peucker(start_idx, end_idx):
        max_dist, max_idx = 0, start_idx
        for i in range(start_idx + 1, end_idx):
            dist = perpendicular_distance(i, start_idx, end_idx)
            if dist > max_dist:
                max_dist, max_idx = dist, i
        
        if max_dist > tolerance:
            left = douglas_peucker(start_idx, max_idx)
            right = douglas_peucker(max_idx, end_idx)
            return left + right[1:]
        return [start_idx, end_idx]
    
    # Douglas-Peucker
    keep = sorted(set(douglas_peucker(0, len(t_values) - 1)))
    
    knot_t, knot_sdf = t_values[keep], sdf_values[keep]
    
    # Compute error
    sdf_interp = np.interp(t_values, knot_t, knot_sdf)
    max_error = np.max(np.abs(sdf_values - sdf_interp))
    mean_error = np.mean(np.abs(sdf_values - sdf_interp))
    
    return knot_t, knot_sdf, max_error, mean_error


def preprocess_polygons_to_splines(polygons: List[np.ndarray], 
                                    max_depth: int = 2,
                                    n_samples: int = 1000,
                                    tolerance: float = 0.005) -> Tuple[List[Dict[str, Any]], Dict]:
    """
    Complete preprocessing pipeline: polygons -> hierarchical PCA -> line segments -> SDF -> knots.
    
    Returns:
        all_knots: List of dicts containing line segment data and knots
        hierarchy: Hierarchical PCA tree structure
    """
    from .spline import Spline
    
    # Step 1: Concatenate all polygon vertices
    vertices = np.concatenate(polygons, axis=0)
    
    # Step 2: Build hierarchical PCA
    hierarchy = build_hierarchical_pca(vertices, max_depth=max_depth)
    
    # Step 3: Extract all line segments
    all_lines_data = extract_all_lines(hierarchy)
    line_segments = [line for line, _ in all_lines_data]
    labels = [label for _, label in all_lines_data]
    
    # Step 4: Compute SDF and knots for each line segment
    all_knots = []
    for (p_start, p_end), label in zip(line_segments, labels):
        # Compute SDF along line
        t, sdf = compute_sdf_sampling(p_start, p_end, polygons, n_samples=n_samples)
        
        # Simplify to knots
        knot_t, knot_sdf, max_err, mean_err = simplify_sdf_to_knots(t, sdf, tolerance=tolerance)
        
        # Extract metadata from label
        depth = int(label.split(':')[0].replace('D', '')) if 'D' in label else 0
        pc_type = 'pc1' if 'pc1' in label else 'pc2'
        
        all_knots.append({
            't': t,
            'sdf': sdf,
            'knot_t': knot_t,
            'knot_sdf': knot_sdf,
            'max_error': max_err,
            'mean_error': mean_err,
            'label': label,
            'line': (p_start, p_end),
            'depth': depth,
            'pc_type': pc_type
        })
        
    return all_knots, hierarchy