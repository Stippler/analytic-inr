"""
Exact computation of edge-based Voronoi polygons.

This module computes the exact Voronoi diagram where the sites are line segments
(edges) and extracts the resulting polygonal regions.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class EdgeVoronoiDiagram:
    """
    Compute exact Voronoi diagram for line segments (edges).
    
    Each Voronoi cell represents the region of points closest to a particular edge.
    This is also known as the "medial axis transform" when applied to polygon boundaries.
    """
    
    def __init__(self, edges: List[Tuple[torch.Tensor, torch.Tensor]], 
                 bounds: Optional[Tuple[float, float, float, float]] = None):
        """
        Initialize with a list of edges.
        
        Args:
            edges: List of tuples (start, end) where start/end are torch.Tensors of shape (2,)
            bounds: Optional (min_x, max_x, min_y, max_y) bounds for clipping
        """
        self.edges = edges
        self.bounds = bounds or self._compute_bounds()
        self.voronoi_polygons = []
        self.edge_to_polygon = {}
        
    def _compute_bounds(self, margin: float = 0.2) -> Tuple[float, float, float, float]:
        """Compute bounding box from edges with margin."""
        all_points = []
        for start, end in self.edges:
            all_points.append(start.cpu().numpy())
            all_points.append(end.cpu().numpy())
        
        all_points = np.array(all_points)
        min_x, min_y = all_points.min(axis=0) - margin
        max_x, max_y = all_points.max(axis=0) + margin
        
        return (min_x, max_x, min_y, max_y)
    
    def compute_approximation(self, samples_per_edge: int = 50) -> List[np.ndarray]:
        """
        Compute approximate edge Voronoi by sampling points along edges.
        
        This creates a point-based Voronoi diagram from densely sampled edge points,
        then merges regions belonging to the same edge.
        
        Args:
            samples_per_edge: Number of sample points per edge
            
        Returns:
            List of polygon vertices for each edge's Voronoi region
        """
        # Sample points along each edge
        sampled_points = []
        point_to_edge = []
        
        for edge_idx, (start, end) in enumerate(self.edges):
            start_np = start.cpu().numpy()
            end_np = end.cpu().numpy()
            
            # Sample points along the edge
            for i in range(samples_per_edge):
                t = i / max(1, samples_per_edge - 1)
                point = start_np + t * (end_np - start_np)
                sampled_points.append(point)
                point_to_edge.append(edge_idx)
        
        sampled_points = np.array(sampled_points)
        
        # Add boundary points for bounded regions
        min_x, max_x, min_y, max_y = self.bounds
        boundary_points = self._generate_boundary_points(min_x, max_x, min_y, max_y, n=20)
        
        all_points = np.vstack([sampled_points, boundary_points])
        
        # Compute Voronoi diagram
        vor = Voronoi(all_points)
        
        # Extract and merge regions for each edge
        self.voronoi_polygons = self._extract_edge_regions(
            vor, point_to_edge, len(sampled_points)
        )
        
        return self.voronoi_polygons
    
    def _generate_boundary_points(self, min_x: float, max_x: float, 
                                   min_y: float, max_y: float, n: int = 20) -> np.ndarray:
        """Generate points along the boundary rectangle."""
        boundary_points = []
        
        # Top edge
        for i in range(n):
            x = min_x + (max_x - min_x) * i / (n - 1)
            boundary_points.append([x, max_y])
        
        # Right edge
        for i in range(1, n):
            y = max_y - (max_y - min_y) * i / (n - 1)
            boundary_points.append([max_x, y])
        
        # Bottom edge
        for i in range(1, n):
            x = max_x - (max_x - min_x) * i / (n - 1)
            boundary_points.append([x, min_y])
        
        # Left edge
        for i in range(1, n-1):
            y = min_y + (max_y - min_y) * i / (n - 1)
            boundary_points.append([min_x, y])
        
        return np.array(boundary_points)
    
    def _extract_edge_regions(self, vor: Voronoi, point_to_edge: List[int], 
                              num_sampled: int) -> List[np.ndarray]:
        """
        Extract and merge Voronoi regions for each edge.
        
        Returns a polygon for each edge representing its Voronoi cell.
        """
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
        
        edge_polygons = {}
        
        # Collect all Voronoi vertices for each edge
        for point_idx in range(num_sampled):
            edge_idx = point_to_edge[point_idx]
            region_idx = vor.point_region[point_idx]
            
            if region_idx == -1:
                continue
                
            region = vor.regions[region_idx]
            if -1 in region or len(region) < 3:
                continue
            
            # Get vertices of this Voronoi region
            vertices = vor.vertices[region]
            
            # Clip to bounds
            vertices = self._clip_polygon_to_bounds(vertices)
            
            if vertices is not None and len(vertices) >= 3:
                if edge_idx not in edge_polygons:
                    edge_polygons[edge_idx] = []
                edge_polygons[edge_idx].append(vertices)
        
        # Merge polygons for each edge using union
        final_polygons = []
        for edge_idx in sorted(edge_polygons.keys()):
            polygons = edge_polygons[edge_idx]
            
            if len(polygons) == 0:
                continue
            
            # Convert to Shapely polygons and compute union
            shapely_polygons = []
            for poly_vertices in polygons:
                try:
                    shapely_poly = ShapelyPolygon(poly_vertices)
                    if shapely_poly.is_valid:
                        shapely_polygons.append(shapely_poly)
                except:
                    continue
            
            if len(shapely_polygons) > 0:
                # Compute union of all polygons for this edge
                merged = unary_union(shapely_polygons)
                
                # Extract coordinates
                if hasattr(merged, 'exterior'):
                    coords = np.array(merged.exterior.coords[:-1])  # Remove duplicate last point
                    final_polygons.append(coords)
                    self.edge_to_polygon[edge_idx] = len(final_polygons) - 1
        
        return final_polygons
    
    def _clip_polygon_to_bounds(self, vertices: np.ndarray) -> Optional[np.ndarray]:
        """Clip a polygon to the bounding box using Sutherland-Hodgman algorithm."""
        from shapely.geometry import Polygon as ShapelyPolygon, box
        
        min_x, max_x, min_y, max_y = self.bounds
        
        try:
            # Create polygon and bounding box
            poly = ShapelyPolygon(vertices)
            bbox = box(min_x, min_y, max_x, max_y)
            
            # Compute intersection
            clipped = poly.intersection(bbox)
            
            if clipped.is_empty:
                return None
            
            # Extract coordinates
            if hasattr(clipped, 'exterior'):
                return np.array(clipped.exterior.coords[:-1])
            
            return None
        except:
            return None
    
    def compute_exact(self, resolution: int = 500) -> List[torch.Tensor]:
        """
        Compute exact edge Voronoi using grid-based distance computation.
        
        This method computes the exact distance from each grid point to each edge,
        then extracts polygonal regions.
        
        Args:
            resolution: Grid resolution for computation
            
        Returns:
            List of torch.Tensors representing Voronoi polygons
        """
        min_x, max_x, min_y, max_y = self.bounds
        
        # Create grid
        x = torch.linspace(min_x, max_x, resolution)
        y = torch.linspace(min_y, max_y, resolution)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        
        # Flatten for computation
        points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        # Compute distances to all edges
        num_edges = len(self.edges)
        distances = torch.full((points.shape[0], num_edges), float('inf'))
        
        for i, (start, end) in enumerate(self.edges):
            distances[:, i] = self._point_to_segment_distance(points, start, end)
        
        # Find closest edge for each point
        _, labels = torch.min(distances, dim=1)
        labels = labels.reshape(resolution, resolution)
        
        # Extract polygons using contour detection
        import cv2
        
        polygons = []
        labels_np = labels.numpy().astype(np.uint8)
        
        for edge_idx in range(num_edges):
            # Create binary mask for this edge's region
            mask = (labels_np == edge_idx).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            # Get largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Simplify contour
            epsilon = 0.002 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(simplified) < 3:
                continue
            
            # Convert pixel coordinates to world coordinates
            vertices = []
            for point in simplified:
                px, py = point[0]
                x_coord = min_x + (max_x - min_x) * px / (resolution - 1)
                y_coord = min_y + (max_y - min_y) * py / (resolution - 1)
                vertices.append([x_coord, y_coord])
            
            polygon = torch.tensor(vertices, dtype=torch.float32)
            polygons.append(polygon)
            self.edge_to_polygon[edge_idx] = len(polygons) - 1
        
        self.voronoi_polygons = polygons
        return polygons
    
    def _point_to_segment_distance(self, points: torch.Tensor, 
                                   start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Compute distance from points to a line segment."""
        segment = end - start
        segment_length_sq = torch.sum(segment * segment)
        
        if segment_length_sq == 0:
            return torch.norm(points - start, dim=1)
        
        # Project points onto line
        t = torch.clamp(
            torch.sum((points - start) * segment, dim=1) / segment_length_sq,
            0, 1
        )
        
        # Find closest point on segment
        closest = start + t.unsqueeze(1) * segment
        
        return torch.norm(points - closest, dim=1)
    
    def visualize(self, show_edges: bool = True, show_labels: bool = False):
        """
        Visualize the edge Voronoi diagram.
        
        Args:
            show_edges: Whether to show original edges
            show_labels: Whether to label regions with edge indices
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Use different colors for each region
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(self.edges), 1)))
        
        # Plot Voronoi polygons
        for poly_idx, polygon in enumerate(self.voronoi_polygons):
            # Find which edge this polygon belongs to
            edge_idx = None
            for e_idx, p_idx in self.edge_to_polygon.items():
                if p_idx == poly_idx:
                    edge_idx = e_idx
                    break
            
            if edge_idx is not None:
                poly_np = polygon.numpy() if torch.is_tensor(polygon) else polygon
                patch = patches.Polygon(
                    poly_np,
                    closed=True,
                    alpha=0.5,
                    facecolor=colors[edge_idx % len(colors)],
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(patch)
                
                if show_labels:
                    # Add label at polygon centroid
                    centroid = poly_np.mean(axis=0)
                    ax.text(centroid[0], centroid[1], str(edge_idx),
                           ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Plot original edges
        if show_edges:
            for edge_idx, (start, end) in enumerate(self.edges):
                start_np = start.cpu().numpy() if torch.is_tensor(start) else start
                end_np = end.cpu().numpy() if torch.is_tensor(end) else end
                ax.plot([start_np[0], end_np[0]], [start_np[1], end_np[1]],
                       'r-', linewidth=2, alpha=0.7)
                
                # Add edge midpoint marker
                midpoint = (start_np + end_np) / 2
                ax.plot(midpoint[0], midpoint[1], 'ro', markersize=5)
        
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Edge-based Voronoi Diagram')
        
        plt.tight_layout()
        plt.show()


def example_usage():
    """Example showing how to use the EdgeVoronoiDiagram class."""
    
    # Create some example polygons
    square = torch.tensor([
        [0.2, 0.2],
        [0.8, 0.2],
        [0.8, 0.8],
        [0.2, 0.8]
    ], dtype=torch.float32)
    
    triangle = torch.tensor([
        [-0.6, -0.6],
        [-0.1, -0.6],
        [-0.35, -0.1]
    ], dtype=torch.float32)
    
    # Convert polygons to edges
    edges = []
    for polygon in [square, triangle]:
        n = len(polygon)
        for i in range(n):
            start = polygon[i]
            end = polygon[(i + 1) % n]
            edges.append((start, end))
    
    # Create EdgeVoronoiDiagram
    evd = EdgeVoronoiDiagram(edges)
    
    # Method 1: Compute exact Voronoi using grid-based approach
    print("Computing exact edge Voronoi...")
    exact_polygons = evd.compute_exact(resolution=300)
    print(f"Found {len(exact_polygons)} Voronoi regions (exact method)")
    
    # Visualize
    evd.visualize(show_edges=True, show_labels=True)
    
    # Method 2: Compute approximate Voronoi using sampling
    evd2 = EdgeVoronoiDiagram(edges)
    print("\nComputing approximate edge Voronoi...")
    approx_polygons = evd2.compute_approximation(samples_per_edge=30)
    print(f"Found {len(approx_polygons)} Voronoi regions (approximation method)")
    
    # Return polygons as torch tensors
    torch_polygons = [torch.tensor(p, dtype=torch.float32) for p in approx_polygons]
    
    return torch_polygons, edges


if __name__ == "__main__":
    polygons, edges = example_usage()
    
    # Print information about the extracted polygons
    for i, polygon in enumerate(polygons):
        print(f"Polygon {i}: {polygon.shape[0]} vertices")
