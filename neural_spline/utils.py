from typing import List, Optional, Tuple
import numpy as np
import trimesh
import pysdf


def create_polygons2d(spec, convex=True, star_ratio=0.5, stretch=(1.0, 1.0), rotation=0.0) -> List[np.ndarray]:
    """
    Generate non-overlapping polygons in [-1, 1]^2 based on a specification string.
    
    Parameters:
    -----------
    spec : str
        Specification string in format "n_polygons x vertices_per_polygon"
        Examples: "3x4" (3 polygons with 4 vertices each)
                  "4x5" (4 polygons with 5 vertices each)
        
        Alternatively, can specify different vertices per polygon:
        "4,3,5" (3 polygons with 4, 3, and 5 vertices respectively)
    
    convex : bool
        If True, generate convex polygons (regular polygons)
        If False, generate non-convex star-like polygons
    
    star_ratio : float (0 to 1)
        For non-convex polygons, ratio of inner radius to outer radius
        Smaller values create more pronounced star shapes
        Only used when convex=False
    
    stretch : tuple, list, or float
        Stretching factor for polygons in (x, y) directions.
        - Single float: uniform scaling (e.g., 1.5)
        - Tuple (sx, sy): stretch all polygons by sx in x and sy in y
        - List of tuples: per-polygon stretch factors
        Examples: (2.0, 1.0) stretches 2x in x-direction (rectangles from squares)
                  (1.0, 0.5) compresses in y-direction (flattened shapes)
    
    rotation : float, list, or np.ndarray
        Rotation angle(s) in radians for the polygons.
        - Single float: rotate all polygons by the same angle
        - List/array: per-polygon rotation angles
        Examples: 0.5 (rotate all by 0.5 radians)
                  [0, np.pi/4, np.pi/2] (different rotation per polygon)
    
    Returns:
    --------
    list of np.ndarray
        List of polygon vertex arrays, each of shape (n_vertices, 2)
    """
    # Parse the specification
    if 'x' in spec:
        # Format: "n_polygons x vertices_per_polygon"
        parts = spec.split('x')
        n_polygons = int(parts[0])
        vertices_per_polygon = int(parts[1])
        vertices_list = [vertices_per_polygon] * n_polygons
    else:
        # Format: "v1,v2,v3,..."
        vertices_list = [int(v) for v in spec.split(',')]
        n_polygons = len(vertices_list)
    
    # Parse stretch parameter
    if isinstance(stretch, (int, float)):
        # Single float: uniform scaling
        stretch_list = [(stretch, stretch)] * n_polygons
    elif isinstance(stretch, tuple):
        # Tuple: same stretch for all polygons
        stretch_list = [stretch] * n_polygons
    elif isinstance(stretch, list):
        # List: per-polygon stretches
        stretch_list = stretch
        if len(stretch_list) < n_polygons:
            # Extend with (1.0, 1.0) if not enough specified
            stretch_list.extend([(1.0, 1.0)] * (n_polygons - len(stretch_list)))
    else:
        stretch_list = [(1.0, 1.0)] * n_polygons
    
    # Parse rotation parameter
    if isinstance(rotation, (int, float)):
        # Single value: same rotation for all polygons
        rotation_list = [float(rotation)] * n_polygons
    elif isinstance(rotation, (list, np.ndarray)):
        # List/array: per-polygon rotations
        rotation_list = list(rotation)
        if len(rotation_list) < n_polygons:
            # Extend with 0.0 if not enough specified
            rotation_list.extend([0.0] * (n_polygons - len(rotation_list)))
    else:
        rotation_list = [0.0] * n_polygons
    
    # Determine grid layout to fit polygons without overlap
    grid_cols = int(np.ceil(np.sqrt(n_polygons)))
    grid_rows = int(np.ceil(n_polygons / grid_cols))
    
    # Cell dimensions (with padding)
    padding = 0.05
    cell_width = 2.0 / grid_cols
    cell_height = 2.0 / grid_rows
    
    polygons = []
    
    for idx, n_vertices in enumerate(vertices_list):
        # Determine cell position
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Cell center in [-1, 1]^2
        cell_center_x = -1.0 + cell_width * (col + 0.5)
        cell_center_y = -1.0 + cell_height * (row + 0.5)
        
        # Get stretch factors for this polygon
        sx, sy = stretch_list[idx]
        
        # Get rotation angle for this polygon
        angle_offset = rotation_list[idx]
        
        # Polygon radius (inscribed in cell with padding, accounting for stretch)
        # Ensure stretched polygon fits within cell boundaries
        radius_x = (cell_width / 2) * 0.8 / sx if sx > 0 else cell_width * 0.4
        radius_y = (cell_height / 2) * 0.8 / sy if sy > 0 else cell_height * 0.4
        radius = min(radius_x, radius_y)
        
        if convex:
            # Generate regular convex polygon vertices (centered at origin)
            angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
            angles += idx * 0.3
            
            vertices = np.zeros((n_vertices, 2))
            vertices[:, 0] = radius * sx * np.cos(angles)
            vertices[:, 1] = radius * sy * np.sin(angles)
        else:
            # Generate non-convex star-like polygon (centered at origin)
            n_points = n_vertices * 2
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            angles += idx * 0.3
            
            # Alternate between outer and inner radii
            radii = np.zeros(n_points)
            radii[::2] = radius
            radii[1::2] = radius * star_ratio
            
            vertices = np.zeros((n_points, 2))
            vertices[:, 0] = radii * sx * np.cos(angles)
            vertices[:, 1] = radii * sy * np.sin(angles)
        
        # Apply rotation using rotation matrix (after stretch)
        if angle_offset != 0:
            cos_a = np.cos(angle_offset)
            sin_a = np.sin(angle_offset)
            rotation_matrix = np.array([[cos_a, -sin_a],
                                       [sin_a, cos_a]])
            vertices = vertices @ rotation_matrix.T
        
        # Translate to cell center
        vertices[:, 0] += cell_center_x
        vertices[:, 1] += cell_center_y
        
        polygons.append(vertices)
    
    return polygons


def load_mesh(mesh_path: str, padding: float = 0.1):
    """
    Load mesh and normalize to fit in [-1, 1]^3 with padding.
    Returns mesh and SDF function.
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # Center and scale to fit in [-1+padding, 1-padding]^3
    mesh.vertices -= mesh.centroid
    max_extent = np.max(mesh.extents)
    if max_extent > 0:
        scale = 2.0 * (1.0 - padding) / max_extent
        mesh.vertices *= scale
    
    # Create SDF function using pysdf (fast, vectorized)
    sdf_fn = pysdf.SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32))
    
    return mesh, sdf_fn


def query_sdf_along_line(sdf_fn, line_start: np.ndarray, line_end: np.ndarray, num_samples: int = 1000):
    """
    Compute signed distance along a line. Vectorized for efficiency.
    
    For very large num_samples (10k+), this is much faster than individual queries.
    pysdf handles batched queries efficiently using BVH acceleration.
    
    Returns (t_vals, sdf_vals) where t_vals are in [0, 1].
    
    Note: This samples uniformly. For adaptive sampling near surface, use
    query_sdf_along_line_adaptive() instead.
    """
    line_start = np.asarray(line_start, dtype=np.float32).reshape(3)
    line_end = np.asarray(line_end, dtype=np.float32).reshape(3)
    t_vals = np.linspace(0, 1, num_samples, dtype=np.float32)
    points = line_start[None, :] + t_vals[:, None] * (line_end - line_start)[None, :]
    sdf_vals = sdf_fn(points)
    return t_vals, sdf_vals


def query_sdf_along_line_adaptive(sdf_fn, line_start: np.ndarray, line_end: np.ndarray, 
                                   num_coarse: int = 100, num_fine: int = 1000, 
                                   surface_threshold: float = 0.1):
    """
    Adaptive sampling: dense near surface, sparse elsewhere.
    
    First samples coarsely, identifies regions near surface (|SDF| < threshold),
    then samples densely in those regions.
    
    Returns (t_vals, sdf_vals) with non-uniform t spacing.
    """
    line_start = np.asarray(line_start, dtype=np.float32).reshape(3)
    line_end = np.asarray(line_end, dtype=np.float32).reshape(3)
    
    # Coarse pass
    t_coarse = np.linspace(0, 1, num_coarse, dtype=np.float32)
    points_coarse = line_start[None, :] + t_coarse[:, None] * (line_end - line_start)[None, :]
    sdf_coarse = sdf_fn(points_coarse)
    
    # Find intervals near surface
    near_surface = np.abs(sdf_coarse) < surface_threshold
    
    if not np.any(near_surface):
        # No surface nearby, return coarse
        return t_coarse, sdf_coarse
    
    # Find contiguous regions near surface
    t_vals_list = []
    for i in range(len(t_coarse) - 1):
        if near_surface[i] or near_surface[i + 1]:
            # Dense sampling in this interval
            n_dense = max(10, num_fine // np.sum(near_surface))
            t_dense = np.linspace(t_coarse[i], t_coarse[i + 1], n_dense, endpoint=False)
            t_vals_list.append(t_dense)
        else:
            # Keep coarse sample
            t_vals_list.append(np.array([t_coarse[i]]))
    
    t_vals_list.append(np.array([t_coarse[-1]]))  # Last point
    t_vals = np.concatenate(t_vals_list)
    
    # Compute SDF at all points
    points = line_start[None, :] + t_vals[:, None] * (line_end - line_start)[None, :]
    sdf_vals = sdf_fn(points)
    
    return t_vals, sdf_vals

