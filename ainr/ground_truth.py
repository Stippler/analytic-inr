"""
Ground truth shape generation for testing and benchmarking.
"""

from typing import List
import numpy as np


def generate_polygons(spec, convex=True, star_ratio=0.5, stretch=(1.0, 1.0), rotation=0.0) -> List[np.ndarray]:
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


def generate_circle(center=(0.0, 0.0), radius=0.5, n_points=64):
    """
    Generate a circle as a polygon.
    
    Parameters:
    -----------
    center : tuple
        Center coordinates (x, y)
    radius : float
        Circle radius
    n_points : int
        Number of vertices to approximate the circle
        
    Returns:
    --------
    np.ndarray
        Vertex array of shape (n_points, 2)
    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    vertices = np.zeros((n_points, 2))
    vertices[:, 0] = center[0] + radius * np.cos(angles)
    vertices[:, 1] = center[1] + radius * np.sin(angles)
    return vertices


def generate_rectangle(center=(0.0, 0.0), width=1.0, height=0.5):
    """
    Generate a rectangle.
    
    Parameters:
    -----------
    center : tuple
        Center coordinates (x, y)
    width : float
        Rectangle width
    height : float
        Rectangle height
        
    Returns:
    --------
    np.ndarray
        Vertex array of shape (4, 2)
    """
    cx, cy = center
    hw, hh = width / 2, height / 2
    vertices = np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh]
    ])
    return vertices


def generate_star(center=(0.0, 0.0), outer_radius=0.5, inner_radius=0.25, n_points=5):
    """
    Generate a star polygon.
    
    Parameters:
    -----------
    center : tuple
        Center coordinates (x, y)
    outer_radius : float
        Radius of outer points
    inner_radius : float
        Radius of inner points
    n_points : int
        Number of star points
        
    Returns:
    --------
    np.ndarray
        Vertex array of shape (n_points * 2, 2)
    """
    n_vertices = n_points * 2
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    
    # Alternate between outer and inner radii
    radii = np.zeros(n_vertices)
    radii[::2] = outer_radius
    radii[1::2] = inner_radius
    
    vertices = np.zeros((n_vertices, 2))
    vertices[:, 0] = center[0] + radii * np.cos(angles)
    vertices[:, 1] = center[1] + radii * np.sin(angles)
    
    return vertices

