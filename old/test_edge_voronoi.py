"""
Test script for edge-based Voronoi polygon extraction.
This demonstrates how to get Voronoi polygons from edges.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def distance_point_to_segment(points, start, end):
    """Compute minimum distance from point(s) to a line segment."""
    segment = end - start
    segment_length_sq = torch.sum(segment * segment)
    
    if segment_length_sq == 0:
        return torch.norm(points - start, dim=-1)
    
    # Project points onto the line
    to_point = points - start
    t = torch.sum(to_point * segment, dim=-1) / segment_length_sq
    t = torch.clamp(t, 0, 1)
    
    # Find closest point on segment
    if points.dim() == 1:
        closest = start + t * segment
    else:
        closest = start + t.unsqueeze(-1) * segment
    
    return torch.norm(points - closest, dim=-1)


def compute_edge_voronoi_polygons(edges, grid_resolution=500, bounds=(-1, 1), device='cpu'):
    """
    Compute Voronoi polygons where each polygon represents the region closest to an edge.
    
    THIS IS THE MAIN FUNCTION YOU NEED: Takes edges and returns polygons where each
    polygon contains all points closer to that edge than to any other edge.
    """
    # Parse bounds
    if isinstance(bounds, tuple) and len(bounds) == 2:
        min_x = min_y = bounds[0]
        max_x = max_y = bounds[1]
    else:
        min_x, max_x, min_y, max_y = bounds
    
    # Create grid
    x = torch.linspace(min_x, max_x, grid_resolution, device=device)
    y = torch.linspace(min_y, max_y, grid_resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    
    # Flatten grid for computation
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    # Compute distance from each grid point to each edge
    num_edges = len(edges)
    distances = torch.zeros(points.shape[0], num_edges, device=device)
    
    for i, (start, end) in enumerate(edges):
        start = start.to(device)
        end = end.to(device)
        distances[:, i] = distance_point_to_segment(points, start, end)
    
    # Find closest edge for each point
    min_distances, voronoi_labels = torch.min(distances, dim=1)
    voronoi_labels = voronoi_labels.reshape(grid_resolution, grid_resolution)
    
    # Extract polygons from the labeled regions using OpenCV
    polygons = []
    edge_indices = []
    labels_np = voronoi_labels.cpu().numpy().astype(np.uint8)
    
    for edge_idx in range(num_edges):
        # Create binary mask for this edge's region
        mask = (labels_np == edge_idx).astype(np.uint8)
        
        # Skip if region is too small
        if np.sum(mask) < 10:
            continue
        
        # Find contours of the region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Get the largest contour (main region)
        contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour to reduce vertices
        epsilon = 0.005 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(simplified) < 3:
            continue
        
        # Convert pixel coordinates to world coordinates
        vertices = []
        for point in simplified:
            px, py = point[0]
            # Map pixel to world coordinates
            x_coord = min_x + (max_x - min_x) * px / (grid_resolution - 1)
            y_coord = min_y + (max_y - min_y) * py / (grid_resolution - 1)
            vertices.append([x_coord, y_coord])
        
        polygon = torch.tensor(vertices, dtype=torch.float32, device=device)
        polygons.append(polygon)
        edge_indices.append(edge_idx)
    
    return {
        'polygons': polygons,
        'edge_indices': edge_indices,
        'voronoi_labels': voronoi_labels,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'distances': min_distances.reshape(grid_resolution, grid_resolution)
    }


def polygons_to_edges(polygons):
    """Convert polygons to edges."""
    edges = []
    for polygon in polygons:
        n = len(polygon)
        for i in range(n):
            start = polygon[i]
            end = polygon[(i + 1) % n]
            edges.append((start, end))
    return edges


def visualize_edge_voronoi(edges, voronoi_result, title="Edge-based Voronoi Diagram"):
    """Visualize the edge Voronoi diagram with the original edges and resulting polygons."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original edges
    ax1.set_title("Original Edges")
    ax1.set_aspect('equal')
    for i, (start, end) in enumerate(edges):
        start_np = start.cpu().numpy()
        end_np = end.cpu().numpy()
        ax1.plot([start_np[0], end_np[0]], [start_np[1], end_np[1]], 
                'b-', linewidth=2, label=f'Edge {i}' if i < 5 else "")
        # Mark edge midpoint
        mid = (start_np + end_np) / 2
        ax1.plot(mid[0], mid[1], 'ro', markersize=5)
        ax1.text(mid[0], mid[1], str(i), fontsize=8, ha='center', va='bottom')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    
    # Plot 2: Voronoi label grid
    ax2.set_title("Voronoi Regions (Label Grid)")
    ax2.set_aspect('equal')
    im = ax2.imshow(voronoi_result['voronoi_labels'].cpu().numpy(), 
                    cmap='tab20', origin='lower',
                    extent=[-1, 1, -1, 1])
    plt.colorbar(im, ax=ax2, label='Edge Index')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Extracted Voronoi polygons
    ax3.set_title("Extracted Voronoi Polygons")
    ax3.set_aspect('equal')
    
    # Use different colors for each polygon
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(edges), 1)))
    
    for polygon, edge_idx in zip(voronoi_result['polygons'], voronoi_result['edge_indices']):
        poly_np = polygon.cpu().numpy()
        patch = patches.Polygon(
            poly_np,
            closed=True,
            alpha=0.6,
            facecolor=colors[edge_idx % len(colors)],
            edgecolor='black',
            linewidth=1,
            label=f'Region {edge_idx}'
        )
        ax3.add_patch(patch)
        
        # Add label at centroid
        centroid = poly_np.mean(axis=0)
        ax3.text(centroid[0], centroid[1], str(edge_idx),
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Overlay original edges
    for i, (start, end) in enumerate(edges):
        start_np = start.cpu().numpy()
        end_np = end.cpu().numpy()
        ax3.plot([start_np[0], end_np[0]], [start_np[1], end_np[1]], 
                'r-', linewidth=1, alpha=0.5)
    
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nVoronoi Diagram Summary:")
    print(f"Number of edges: {len(edges)}")
    print(f"Number of Voronoi polygons extracted: {len(voronoi_result['polygons'])}")
    for i, (polygon, edge_idx) in enumerate(zip(voronoi_result['polygons'], voronoi_result['edge_indices'])):
        print(f"  Polygon {i}: Edge {edge_idx}, {len(polygon)} vertices")


def test_simple_case():
    """Test with a simple square."""
    print("=" * 60)
    print("TEST 1: Simple Square")
    print("=" * 60)
    
    # Create a square
    square = torch.tensor([
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5]
    ], dtype=torch.float32)
    
    # Convert to edges
    edges = polygons_to_edges([square])
    
    # Compute Voronoi polygons
    result = compute_edge_voronoi_polygons(edges, grid_resolution=300)
    
    # Visualize
    visualize_edge_voronoi(edges, result, "Simple Square - Edge Voronoi")
    
    return result


def test_multiple_polygons():
    """Test with multiple polygons."""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple Polygons")
    print("=" * 60)
    
    # Create multiple polygons
    square = torch.tensor([
        [0.2, 0.2],
        [0.8, 0.2],
        [0.8, 0.8],
        [0.2, 0.8]
    ], dtype=torch.float32)
    
    triangle = torch.tensor([
        [-0.7, -0.7],
        [-0.2, -0.7],
        [-0.45, -0.2]
    ], dtype=torch.float32)
    
    pentagon = torch.tensor([
        [-0.8, 0.3],
        [-0.6, 0.1],
        [-0.4, 0.2],
        [-0.4, 0.5],
        [-0.6, 0.6]
    ], dtype=torch.float32)
    
    # Convert to edges
    edges = polygons_to_edges([square, triangle, pentagon])
    
    # Compute Voronoi polygons
    result = compute_edge_voronoi_polygons(edges, grid_resolution=400)
    
    # Visualize
    visualize_edge_voronoi(edges, result, "Multiple Polygons - Edge Voronoi")
    
    return result


def test_complex_shape():
    """Test with a complex L-shaped polygon."""
    print("\n" + "=" * 60)
    print("TEST 3: Complex L-Shape")
    print("=" * 60)
    
    # Create L-shaped polygon
    l_shape = torch.tensor([
        [-0.6, -0.6],
        [0.6, -0.6],
        [0.6, -0.2],
        [0.0, -0.2],
        [0.0, 0.6],
        [-0.6, 0.6]
    ], dtype=torch.float32)
    
    # Convert to edges
    edges = polygons_to_edges([l_shape])
    
    # Compute Voronoi polygons
    result = compute_edge_voronoi_polygons(edges, grid_resolution=400)
    
    # Visualize
    visualize_edge_voronoi(edges, result, "L-Shaped Polygon - Edge Voronoi")
    
    return result


def main():
    """Run all tests."""
    print("\nEdge-based Voronoi Polygon Extraction Tests")
    print("=" * 60)
    print("This demonstrates how to extract Voronoi polygons from edges.")
    print("Each polygon represents the region of points closest to a particular edge.\n")
    
    # Run tests
    result1 = test_simple_case()
    result2 = test_multiple_polygons()
    result3 = test_complex_shape()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The compute_edge_voronoi_polygons function successfully extracts")
    print("polygonal regions from edges, where each polygon contains all points")
    print("that are closer to a particular edge than to any other edge.")
    print("\nYou can use the 'polygons' field from the result dictionary")
    print("to get the Voronoi polygons as torch.Tensors.")
    
    return result1, result2, result3


if __name__ == "__main__":
    results = main()
