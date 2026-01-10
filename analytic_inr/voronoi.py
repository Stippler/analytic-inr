"""
Voronoi-based polygon analysis utilities.

This module provides functions for:
- Detecting polygon overlaps
- Computing Voronoi diagrams from polygon edges (line segments)
"""

import torch
import numpy as np
from typing import List, Tuple


def get_polygon_bounds(polygon):
    """
    Get the bounding box of a polygon.
    
    Args:
        polygon: torch.Tensor of shape (n, 2) with vertex coordinates
    
    Returns:
        tuple: (min_x, max_x, min_y, max_y) bounding box coordinates
    
    Example:
        >>> polygon = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        >>> bounds = get_polygon_bounds(polygon)
        >>> bounds
        (0.0, 1.0, 0.0, 1.0)
    """
    min_x = polygon[:, 0].min().item()
    max_x = polygon[:, 0].max().item()
    min_y = polygon[:, 1].min().item()
    max_y = polygon[:, 1].max().item()
    return min_x, max_x, min_y, max_y


def bounding_boxes_overlap(bbox1, bbox2, margin=0.5):
    """
    Check if two bounding boxes overlap with an optional margin.
    
    Args:
        bbox1: Tuple (min_x, max_x, min_y, max_y) for first bounding box
        bbox2: Tuple (min_x, max_x, min_y, max_y) for second bounding box
        margin: Additional margin to add to bbox1 for overlap detection
    
    Returns:
        bool: True if bounding boxes overlap, False otherwise
    
    Example:
        >>> bbox1 = (0, 2, 0, 2)
        >>> bbox2 = (1.5, 3, 1.5, 3)
        >>> bounding_boxes_overlap(bbox1, bbox2)
        True
    """
    min_x1, max_x1, min_y1, max_y1 = bbox1
    min_x2, max_x2, min_y2, max_y2 = bbox2
    
    min_x1 -= margin
    max_x1 += margin
    min_y1 -= margin
    max_y1 += margin
    
    return not (max_x1 < min_x2 or max_x2 < min_x1 or 
                max_y1 < min_y2 or max_y2 < min_y1)


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        point: torch.Tensor of shape (2,) with point coordinates
        polygon: torch.Tensor of shape (n, 2) with polygon vertices
    
    Returns:
        bool: True if point is inside polygon, False otherwise
    
    Example:
        >>> point = torch.tensor([0.5, 0.5])
        >>> polygon = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        >>> point_in_polygon(point, polygon)
        True
    """
    x, y = point[0].item(), point[1].item()
    vertices = polygon.cpu().numpy()
    n = len(vertices)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def polygons_overlap(poly1, poly2):
    """
    Check if two polygons overlap by testing if any vertices are inside the other polygon.
    
    Args:
        poly1: torch.Tensor of shape (n, 2) with first polygon vertices
        poly2: torch.Tensor of shape (m, 2) with second polygon vertices
    
    Returns:
        bool: True if polygons overlap, False otherwise
    
    Example:
        >>> poly1 = torch.tensor([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=torch.float32)
        >>> poly2 = torch.tensor([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=torch.float32)
        >>> polygons_overlap(poly1, poly2)
        True
    """
    for vertex in poly1:
        if point_in_polygon(vertex, poly2):
            return True
    for vertex in poly2:
        if point_in_polygon(vertex, poly1):
            return True
    return False


def distance_point_to_segment(point, start, end):
    """
    Compute the minimum distance from a point to a line segment.
    
    Args:
        point: torch.Tensor of shape (2,) or (N, 2)
        start: torch.Tensor of shape (2,) - segment start point
        end: torch.Tensor of shape (2,) - segment end point
    
    Returns:
        torch.Tensor: Distance(s) from point(s) to the line segment
    """
    # Vector from start to end
    segment = end - start
    segment_length_sq = torch.sum(segment * segment)
    
    if segment_length_sq == 0:
        # Degenerate case: start == end
        return torch.norm(point - start, dim=-1)
    
    # Vector from start to point
    to_point = point - start
    
    # Project point onto the line (parameter t in [0, 1] means on segment)
    t = torch.sum(to_point * segment, dim=-1) / segment_length_sq
    t = torch.clamp(t, 0, 1)
    
    # Find closest point on segment
    if point.dim() == 1:
        closest = start + t * segment
    else:
        closest = start + t.unsqueeze(-1) * segment
    
    # Return distance to closest point
    return torch.norm(point - closest, dim=-1)


def compute_edge_voronoi_grid(edges, grid_resolution=500, bounds=(-1, 1), device='cpu'):
    """
    Compute a high-resolution grid-based Voronoi diagram for line segments (edges).
    
    For each point in a grid, determines which edge it is closest to.
    This creates a discrete approximation of the Voronoi diagram for edges.
    
    Args:
        edges: List of tuples (start, end) where start/end are torch.Tensors of shape (2,)
        grid_resolution: Number of points along each axis (default: 500 for good quality)
        bounds: Tuple (min, max) for the grid bounds (default: (-1, 1))
        device: Device to run computation on ('cpu' or 'cuda')
    
    Returns:
        tuple: (grid_x, grid_y, voronoi_labels, distances) where:
            - grid_x: torch.Tensor of shape (resolution, resolution) with x coordinates
            - grid_y: torch.Tensor of shape (resolution, resolution) with y coordinates
            - voronoi_labels: torch.Tensor of shape (resolution, resolution) with edge indices
            - distances: torch.Tensor of shape (resolution, resolution) with distances to nearest edge
    """
    # Create grid
    x = torch.linspace(bounds[0], bounds[1], grid_resolution, device=device)
    y = torch.linspace(bounds[0], bounds[1], grid_resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    
    # Flatten grid for batch processing
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    # Compute distance from each grid point to each edge
    num_edges = len(edges)
    distances_to_edges = torch.zeros(points.shape[0], num_edges, device=device)
    
    for i, (start, end) in enumerate(edges):
        # Ensure edges are on the same device
        start = start.to(device)
        end = end.to(device)
        distances_to_edges[:, i] = distance_point_to_segment(points, start, end)
    
    # Find closest edge for each point
    min_distances, voronoi_labels = torch.min(distances_to_edges, dim=1)
    
    # Reshape back to grid
    voronoi_labels = voronoi_labels.reshape(grid_resolution, grid_resolution)
    min_distances = min_distances.reshape(grid_resolution, grid_resolution)
    
    return grid_x, grid_y, voronoi_labels, min_distances


def extract_voronoi_boundaries(voronoi_labels, epsilon=1e-6):
    """
    Extract boundaries between different Voronoi regions.
    
    Args:
        voronoi_labels: torch.Tensor of shape (H, W) with region labels
        epsilon: Threshold for boundary detection (default: 1e-6)
    
    Returns:
        torch.Tensor: Binary mask of shape (H, W) where 1 indicates boundary
    """
    # Compute gradients (changes in labels indicate boundaries)
    dx = torch.abs(voronoi_labels[1:, :] - voronoi_labels[:-1, :])
    dy = torch.abs(voronoi_labels[:, 1:] - voronoi_labels[:, :-1])
    
    # Create boundary mask (only mark actual transitions)
    boundaries = torch.zeros_like(voronoi_labels, dtype=torch.bool)
    boundaries[1:, :] |= (dx > epsilon)
    boundaries[:, 1:] |= (dy > epsilon)
    
    return boundaries


def get_voronoi_boundary_points(voronoi_labels, grid_x, grid_y):
    """
    Extract actual boundary points from Voronoi labels for plotting.
    
    Args:
        voronoi_labels: torch.Tensor of shape (H, W) with region labels
        grid_x: torch.Tensor of shape (H, W) with x coordinates
        grid_y: torch.Tensor of shape (H, W) with y coordinates
    
    Returns:
        tuple: (boundary_x, boundary_y) lists of coordinates for each boundary segment
    """
    boundaries = extract_voronoi_boundaries(voronoi_labels)
    
    # Get coordinates of boundary points
    boundary_mask = boundaries.cpu().numpy()
    boundary_x = grid_x.cpu().numpy()[boundary_mask]
    boundary_y = grid_y.cpu().numpy()[boundary_mask]
    
    return boundary_x, boundary_y


def polygons_to_edges(polygons):
    """
    Convert a list of polygons to edges.
    
    Args:
        polygons: List of torch.Tensors, each of shape (n, 2) with polygon vertices
    
    Returns:
        List of tuples (start, end) for each edge
    """
    edges = []
    
    for polygon in polygons:
        num_vertices = polygon.shape[0]
        for i in range(num_vertices):
            start = polygon[i]
            end = polygon[(i + 1) % num_vertices]
            edges.append((start, end))
    
    return edges


def polygons_to_edges_with_samples(polygons, samples_per_edge=10):
    """
    Convert a list of polygons to edges and sample points along each edge.
    
    For Voronoi diagrams based on polygon edges, we need to densely sample
    each edge with multiple points to approximate "distance to edge" regions.
    
    Args:
        polygons: List of torch.Tensors, each of shape (n, 2) with polygon vertices
        samples_per_edge: Number of points to sample along each edge (default: 10)
    
    Returns:
        tuple: (edges, sampled_points, edge_indices) where:
            - edges: List of tuples (start, end) for each edge
            - sampled_points: torch.Tensor of shape (total_samples, 2)
            - edge_indices: List mapping each sample point to its edge index
    
    Example:
        >>> polygon = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        >>> edges, samples, indices = polygons_to_edges_with_samples([polygon], samples_per_edge=5)
        >>> len(edges)
        4
        >>> samples.shape
        torch.Size([20, 2])
    """
    edges = []
    sampled_points = []
    edge_indices = []
    
    edge_idx = 0
    for polygon in polygons:
        num_vertices = polygon.shape[0]
        for i in range(num_vertices):
            start = polygon[i]
            end = polygon[(i + 1) % num_vertices]
            
            # Store edge
            edges.append((start, end))
            
            # Sample points along the edge (excluding endpoints to avoid vertex bias)
            for j in range(samples_per_edge):
                t = (j + 1) / (samples_per_edge + 1)  # Interpolation parameter
                point = start + t * (end - start)
                sampled_points.append(point)
                edge_indices.append(edge_idx)
            
            edge_idx += 1
    
    return edges, torch.stack(sampled_points), edge_indices


def polygons_to_edges_with_midpoints(polygons):
    """
    Convert a list of polygons to edges and compute their midpoints.
    
    This is a simple version that only uses one point per edge.
    For better Voronoi approximation, use polygons_to_edges_with_samples().
    
    Args:
        polygons: List of torch.Tensors, each of shape (n, 2) with polygon vertices
    
    Returns:
        tuple: (edges, edge_midpoints) where:
            - edges: List of tuples (start, end) for each edge
            - edge_midpoints: torch.Tensor of shape (total_edges, 2)
    
    Example:
        >>> polygon = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        >>> edges, midpoints = polygons_to_edges_with_midpoints([polygon])
        >>> len(edges)
        4
        >>> midpoints.shape
        torch.Size([4, 2])
    """
    edges = []
    edge_midpoints = []
    
    for polygon in polygons:
        num_vertices = polygon.shape[0]
        for i in range(num_vertices):
            start = polygon[i]
            end = polygon[(i + 1) % num_vertices]
            
            # Store edge
            edges.append((start, end))
            
            # Compute midpoint
            midpoint = (start + end) / 2
            edge_midpoints.append(midpoint)
    
    return edges, torch.stack(edge_midpoints)

