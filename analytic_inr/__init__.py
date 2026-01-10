"""
Analytic INR - A library for implicit neural representations with analytic components.
"""

__version__ = "0.1.0"

from . import voronoi
from . import ray_marching_voronoi

# Export commonly used functions directly
from .voronoi import (
    polygons_to_edges,
    compute_edge_voronoi_grid,
    extract_voronoi_boundaries,
    get_voronoi_boundary_points,
    distance_point_to_segment
)

from .ray_marching_voronoi import (
    RayMarchingVoronoi,
    polygons_to_tensor_list
)

__all__ = [
    "voronoi",
    "ray_marching_voronoi",
    "polygons_to_edges",
    "compute_edge_voronoi_grid", 
    "extract_voronoi_boundaries",
    "get_voronoi_boundary_points",
    "distance_point_to_segment",
    "RayMarchingVoronoi",
    "polygons_to_tensor_list"
]

