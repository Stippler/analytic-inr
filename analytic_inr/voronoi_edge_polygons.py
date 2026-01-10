"""
Enhanced Voronoi-based polygon analysis utilities with polygon extraction.

This module provides functions for computing Voronoi diagrams from polygon edges
and extracting the resulting Voronoi polygons.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
import cv2


def compute_edge_voronoi_polygons(
    edges: List[Tuple[torch.Tensor, torch.Tensor]], 
    grid_resolution: int = 500,
    bounds: Tuple[float, float] = (-1, 1),
    device: str = 'cpu',
    min_region_size: int = 50
) -> Dict:
    """
    Compute Voronoi polygons for line segments (edges) and extract the polygonal regions.
    
    Each polygon represents the region closest to a particular edge.
    
    Args:
        edges: List of tuples (start, end) where start/end are torch.Tensors of shape (2,)
        grid_resolution: Number of points along each axis (default: 500)
        bounds: Tuple (min, max) for the grid bounds (default: (-1, 1))
        device: Device to run computation on ('cpu' or 'cuda')
        min_region_size: Minimum number of pixels for a valid region
    
    Returns:
        dict: Dictionary containing:
            - 'polygons': List of torch.Tensors, each of shape (n, 2) representing a Voronoi polygon
            - 'edge_indices': List mapping each polygon to its corresponding edge index
            - 'grid_x', 'grid_y': Grid coordinates
            - 'voronoi_labels': Label grid
            - 'distances': Distance grid
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
        start = start.to(device)
        end = end.to(device)
        distances_to_edges[:, i] = distance_point_to_segment(points, start, end)
    
    # Find closest edge for each point
    min_distances, voronoi_labels = torch.min(distances_to_edges, dim=1)
    
    # Reshape back to grid
    voronoi_labels = voronoi_labels.reshape(grid_resolution, grid_resolution)
    min_distances = min_distances.reshape(grid_resolution, grid_resolution)
    
    # Extract polygons from the labeled regions
    polygons, edge_indices = extract_polygons_from_labels(
        voronoi_labels, grid_x, grid_y, min_region_size
    )
    
    return {
        'polygons': polygons,
        'edge_indices': edge_indices,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'voronoi_labels': voronoi_labels,
        'distances': min_distances
    }


def distance_point_to_segment(point, start, end):
    """
    Compute the minimum distance from point(s) to a line segment.
    """
    segment = end - start
    segment_length_sq = torch.sum(segment * segment)
    
    if segment_length_sq == 0:
        return torch.norm(point - start, dim=-1)
    
    to_point = point - start
    t = torch.sum(to_point * segment, dim=-1) / segment_length_sq
    t = torch.clamp(t, 0, 1)
    
    if point.dim() == 1:
        closest = start + t * segment
    else:
        closest = start + t.unsqueeze(-1) * segment
    
    return torch.norm(point - closest, dim=-1)


def extract_polygons_from_labels(
    voronoi_labels: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    min_region_size: int = 50
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Extract polygonal boundaries for each Voronoi region.
    
    Args:
        voronoi_labels: torch.Tensor of shape (H, W) with region labels
        grid_x: torch.Tensor of shape (H, W) with x coordinates
        grid_y: torch.Tensor of shape (H, W) with y coordinates
        min_region_size: Minimum number of pixels for a valid region
    
    Returns:
        tuple: (polygons, edge_indices) where:
            - polygons: List of torch.Tensors, each of shape (n, 2) for polygon vertices
            - edge_indices: List of edge indices corresponding to each polygon
    """
    labels_np = voronoi_labels.cpu().numpy().astype(np.int32)
    grid_x_np = grid_x.cpu().numpy()
    grid_y_np = grid_y.cpu().numpy()
    
    polygons = []
    edge_indices = []
    
    # Process each unique label (edge)
    unique_labels = np.unique(labels_np)
    
    for label in unique_labels:
        # Create binary mask for this region
        mask = (labels_np == label).astype(np.uint8)
        
        # Skip small regions
        if np.sum(mask) < min_region_size:
            continue
        
        # Find contours of the region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Get the largest contour (main region)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour to reduce number of vertices
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        
        if len(simplified_contour) < 3:
            continue
        
        # Convert pixel indices to actual coordinates
        polygon_vertices = []
        for point in simplified_contour:
            px, py = point[0]
            # Convert pixel coordinates to grid coordinates
            x_coord = grid_x_np[py, px]
            y_coord = grid_y_np[py, px]
            polygon_vertices.append([x_coord, y_coord])
        
        polygon = torch.tensor(polygon_vertices, dtype=torch.float32)
        polygons.append(polygon)
        edge_indices.append(int(label))
    
    return polygons, edge_indices


def compute_edge_voronoi_with_sampling(
    edges: List[Tuple[torch.Tensor, torch.Tensor]],
    samples_per_edge: int = 20,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    margin: float = 0.1
) -> Dict:
    """
    Compute Voronoi diagram by densely sampling points along edges.
    
    This approach samples multiple points along each edge and computes
    a point-based Voronoi diagram, which can then be merged to get
    edge-based regions.
    
    Args:
        edges: List of tuples (start, end) where start/end are torch.Tensors of shape (2,)
        samples_per_edge: Number of points to sample along each edge
        bounds: Optional (min_x, max_x, min_y, max_y) bounds for the diagram
        margin: Margin to add around bounds
    
    Returns:
        dict: Dictionary containing:
            - 'vertices': Voronoi vertices
            - 'regions': List of vertex indices for each region
            - 'edge_to_regions': Mapping from edge index to Voronoi regions
    """
    # Sample points along edges
    sampled_points = []
    point_to_edge = []
    
    for edge_idx, (start, end) in enumerate(edges):
        start_np = start.cpu().numpy()
        end_np = end.cpu().numpy()
        
        # Sample points along the edge
        for i in range(samples_per_edge):
            t = i / max(1, samples_per_edge - 1)
            point = start_np + t * (end_np - start_np)
            sampled_points.append(point)
            point_to_edge.append(edge_idx)
    
    sampled_points = np.array(sampled_points)
    
    # Compute bounds if not provided
    if bounds is None:
        min_x = sampled_points[:, 0].min() - margin
        max_x = sampled_points[:, 0].max() + margin
        min_y = sampled_points[:, 1].min() - margin
        max_y = sampled_points[:, 1].max() + margin
    else:
        min_x, max_x, min_y, max_y = bounds
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
    
    # Add boundary points to ensure bounded Voronoi regions
    boundary_points = [
        [min_x, min_y], [max_x, min_y],
        [max_x, max_y], [min_x, max_y],
        [min_x, (min_y + max_y) / 2], [max_x, (min_y + max_y) / 2],
        [(min_x + max_x) / 2, min_y], [(min_x + max_x) / 2, max_y]
    ]
    
    all_points = np.vstack([sampled_points, boundary_points])
    
    # Compute Voronoi diagram
    vor = Voronoi(all_points)
    
    # Group regions by edge
    edge_to_regions = {}
    for i in range(len(sampled_points)):
        edge_idx = point_to_edge[i]
        if edge_idx not in edge_to_regions:
            edge_to_regions[edge_idx] = []
        
        # Find the Voronoi region for this point
        point_region = vor.point_region[i]
        if point_region != -1:
            edge_to_regions[edge_idx].append(point_region)
    
    return {
        'voronoi': vor,
        'vertices': vor.vertices,
        'regions': vor.regions,
        'edge_to_regions': edge_to_regions,
        'sampled_points': sampled_points,
        'point_to_edge': point_to_edge
    }


def merge_voronoi_regions_by_edge(voronoi_result: Dict) -> List[np.ndarray]:
    """
    Merge Voronoi regions that belong to the same edge to create edge-based polygons.
    
    Args:
        voronoi_result: Result from compute_edge_voronoi_with_sampling
    
    Returns:
        List of numpy arrays, each containing vertices of a merged polygon for each edge
    """
    vor = voronoi_result['voronoi']
    edge_to_regions = voronoi_result['edge_to_regions']
    
    edge_polygons = []
    
    for edge_idx in sorted(edge_to_regions.keys()):
        region_indices = edge_to_regions[edge_idx]
        
        # Collect all vertices from all regions for this edge
        all_vertices = []
        for region_idx in region_indices:
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                vertices = vor.vertices[region]
                all_vertices.extend(vertices)
        
        if len(all_vertices) < 3:
            continue
        
        all_vertices = np.array(all_vertices)
        
        # Compute convex hull of all vertices to get the merged polygon
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(all_vertices)
            hull_vertices = all_vertices[hull.vertices]
            edge_polygons.append(hull_vertices)
        except:
            # If convex hull fails, skip this edge
            continue
    
    return edge_polygons


def visualize_edge_voronoi(
    edges: List[Tuple[torch.Tensor, torch.Tensor]],
    voronoi_polygons: List[torch.Tensor],
    edge_indices: List[int],
    figsize: Tuple[int, int] = (10, 10)
):
    """
    Visualize the edge-based Voronoi diagram.
    
    Args:
        edges: Original edges
        voronoi_polygons: List of Voronoi polygons
        edge_indices: Indices mapping polygons to edges
        figsize: Figure size for visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import LineCollection
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot original edges
    ax1.set_title("Original Edges")
    edge_lines = []
    for start, end in edges:
        edge_lines.append([start.cpu().numpy(), end.cpu().numpy()])
    
    lc = LineCollection(edge_lines, colors='blue', linewidths=2)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot Voronoi polygons
    ax2.set_title("Edge-based Voronoi Regions")
    
    # Use different colors for each region
    colors = plt.cm.tab20(np.linspace(0, 1, len(edges)))
    
    for polygon, edge_idx in zip(voronoi_polygons, edge_indices):
        if edge_idx < len(colors):
            poly_np = polygon.cpu().numpy()
            poly_patch = patches.Polygon(
                poly_np, 
                closed=True, 
                alpha=0.5, 
                facecolor=colors[edge_idx],
                edgecolor='black',
                linewidth=1
            )
            ax2.add_patch(poly_patch)
    
    # Also plot original edges on the Voronoi diagram
    lc2 = LineCollection(edge_lines, colors='red', linewidths=1, alpha=0.7)
    ax2.add_collection(lc2)
    
    ax2.autoscale()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage function
def example_edge_voronoi():
    """
    Example showing how to compute edge-based Voronoi polygons.
    """
    # Create a simple example with a square and a triangle
    square = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5]
    ], dtype=torch.float32)
    
    triangle = torch.tensor([
        [-0.5, -0.5],
        [-0.2, -0.5],
        [-0.35, -0.2]
    ], dtype=torch.float32)
    
    # Convert polygons to edges
    edges = []
    for polygon in [square, triangle]:
        n = len(polygon)
        for i in range(n):
            start = polygon[i]
            end = polygon[(i + 1) % n]
            edges.append((start, end))
    
    # Method 1: Grid-based approach with polygon extraction
    print("Computing grid-based edge Voronoi diagram...")
    result = compute_edge_voronoi_polygons(
        edges, 
        grid_resolution=200,
        bounds=(-1, 1),
        min_region_size=20
    )
    
    print(f"Found {len(result['polygons'])} Voronoi regions")
    for i, (polygon, edge_idx) in enumerate(zip(result['polygons'], result['edge_indices'])):
        print(f"  Region {i}: Edge {edge_idx}, {len(polygon)} vertices")
    
    # Method 2: Sampling-based approach
    print("\nComputing sampling-based edge Voronoi diagram...")
    voronoi_result = compute_edge_voronoi_with_sampling(
        edges,
        samples_per_edge=10
    )
    
    merged_polygons = merge_voronoi_regions_by_edge(voronoi_result)
    print(f"Found {len(merged_polygons)} merged Voronoi regions")
    
    return result, edges


if __name__ == "__main__":
    # Run example
    result, edges = example_edge_voronoi()
    
    # Optionally visualize
    if len(result['polygons']) > 0:
        visualize_edge_voronoi(
            edges,
            result['polygons'],
            result['edge_indices']
        )
