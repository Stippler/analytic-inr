#%% Imports and Setup
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.collections import LineCollection

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_polygons = 6
vertices_per_polygon = 6
bounds = (0, 10)
radius_range = (0.6, 1.2)

print(f"Device: {device}")
print(f"Generating {num_polygons} polygons with {vertices_per_polygon} vertices each")

##%% Function: Generate Single Polygon
def generate_polygon(center, num_vertices, radius_range=(0.5, 1.5), device='cpu'):
    """Generate a single 2D polygon with vertices in CCW order."""
    angles = torch.linspace(0, 2 * np.pi, num_vertices + 1, device=device)[:-1]
    radii = torch.rand(num_vertices, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    
    x = center[0] + radii * torch.cos(angles)
    y = center[1] + radii * torch.sin(angles)
    vertices = torch.stack([x, y], dim=1)
    
    return vertices

##%% Overlap Detection Functions
def get_polygon_bounds(polygon):
    """Get bounding box of a polygon."""
    min_x = polygon[:, 0].min().item()
    max_x = polygon[:, 0].max().item()
    min_y = polygon[:, 1].min().item()
    max_y = polygon[:, 1].max().item()
    return min_x, max_x, min_y, max_y

def bounding_boxes_overlap(bbox1, bbox2, margin=0.5):
    """Check if two bounding boxes overlap with margin."""
    min_x1, max_x1, min_y1, max_y1 = bbox1
    min_x2, max_x2, min_y2, max_y2 = bbox2
    
    min_x1 -= margin
    max_x1 += margin
    min_y1 -= margin
    max_y1 += margin
    
    return not (max_x1 < min_x2 or max_x2 < min_x1 or 
                max_y1 < min_y2 or max_y2 < min_y1)

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm."""
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
    """Check if two polygons overlap."""
    for vertex in poly1:
        if point_in_polygon(vertex, poly2):
            return True
    for vertex in poly2:
        if point_in_polygon(vertex, poly1):
            return True
    return False

##%% Generate Non-Overlapping Polygons
def generate_non_overlapping_polygons(num_polygons, vertices_per_polygon, 
                                      bounds=(0, 10), radius_range=(0.5, 1.0),
                                      device='cpu', max_attempts=1000):
    """Generate multiple non-overlapping 2D polygons."""
    polygons = []
    bboxes = []
    attempts = 0
    
    print(f"\nGenerating {num_polygons} non-overlapping polygons...")
    
    while len(polygons) < num_polygons and attempts < max_attempts * num_polygons:
        attempts += 1
        
        center_x = torch.rand(1, device=device) * (bounds[1] - bounds[0]) + bounds[0]
        center_y = torch.rand(1, device=device) * (bounds[1] - bounds[0]) + bounds[0]
        center = (center_x.item(), center_y.item())
        
        candidate = generate_polygon(center, vertices_per_polygon, radius_range, device)
        candidate_bbox = get_polygon_bounds(candidate)
        
        overlaps = False
        for bbox in bboxes:
            if bounding_boxes_overlap(candidate_bbox, bbox):
                for existing_poly in polygons:
                    if polygons_overlap(candidate, existing_poly):
                        overlaps = True
                        break
                if overlaps:
                    break
        
        if not overlaps:
            polygons.append(candidate)
            bboxes.append(candidate_bbox)
            print(f"  ✓ Polygon {len(polygons)}/{num_polygons} placed")
    
    return polygons

# Generate polygons
polygons = generate_non_overlapping_polygons(
    num_polygons, vertices_per_polygon, bounds, radius_range, device
)

print(f"\nSuccessfully generated {len(polygons)} polygons")

##%% Visualize Step 1: Non-Overlapping Polygons
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_title('Step 1: Generated Non-Overlapping Polygons', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(bounds[0] - 1, bounds[1] + 1)
ax.set_ylim(bounds[0] - 1, bounds[1] + 1)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)

colors = plt.cm.Set3(np.linspace(0, 1, len(polygons)))

for i, polygon in enumerate(polygons):
    poly_np = polygon.cpu().numpy()
    patch = MPLPolygon(poly_np, facecolor=colors[i], edgecolor='black', 
                      linewidth=2, alpha=0.6, label=f'Polygon {i+1}')
    ax.add_patch(patch)
    ax.plot(poly_np[:, 0], poly_np[:, 1], 'ko', markersize=8, zorder=5)

ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

print("✓ Step 1: Polygons visualized")

##%% Convert Polygons to Edges with Midpoints
def polygons_to_edges_with_midpoints(polygons):
    """Convert polygons to edges and compute their midpoints."""
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

edges, edge_midpoints = polygons_to_edges_with_midpoints(polygons)
print(f"\nTotal edges: {len(edges)}")
print(f"Edge midpoints computed: {edge_midpoints.shape}")

##%% Visualize Step 2: Polygon Edges with Midpoints
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_title('Step 2: Polygon Edges with Midpoints', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(bounds[0] - 1, bounds[1] + 1)
ax.set_ylim(bounds[0] - 1, bounds[1] + 1)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)

colors = plt.cm.Set3(np.linspace(0, 1, len(polygons)))

edge_idx = 0
for poly_idx, polygon in enumerate(polygons):
    num_vertices = polygon.shape[0]
    
    for i in range(num_vertices):
        start, end = edges[edge_idx]
        start_np = start.cpu().numpy()
        end_np = end.cpu().numpy()
        midpoint_np = edge_midpoints[edge_idx].cpu().numpy()
        
        # Draw edge
        ax.plot([start_np[0], end_np[0]], [start_np[1], end_np[1]], 
               '-', color=colors[poly_idx], linewidth=3, alpha=0.6)
        
        # Draw midpoint
        ax.plot(midpoint_np[0], midpoint_np[1], 'ro', markersize=10, zorder=5)
        
        # Label edge and midpoint
        ax.text(midpoint_np[0], midpoint_np[1], f'E{edge_idx}', fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
               ha='center', va='bottom')
        
        edge_idx += 1

# Legend
for i in range(len(polygons)):
    ax.plot([], [], '-', color=colors[i], linewidth=3, 
           label=f'Polygon {i+1}')

ax.plot([], [], 'ro', markersize=10, label='Edge midpoints')
ax.legend(loc='upper right', fontsize=10)

ax.text(0.02, 0.98, f'Total edges: {len(edges)}\nRed dots = edge midpoints', 
       transform=ax.transAxes, fontsize=12, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

print("✓ Step 2: Edges with midpoints visualized")

##%% Compute Voronoi Diagram on Edge Midpoints
edge_midpoints_np = edge_midpoints.cpu().numpy()
vor = Voronoi(edge_midpoints_np)

print(f"\nVoronoi computation complete:")
print(f"  - Input points (edge midpoints): {len(edge_midpoints_np)}")
print(f"  - Voronoi vertices: {len(vor.vertices)}")
print(f"  - Voronoi regions: {len(vor.regions)}")

##%% Visualize Step 3: Voronoi Diagram of Edges
fig, ax = plt.subplots(figsize=(12, 12))

ax.set_title('Step 3: Voronoi Diagram of Edge Midpoints', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(bounds[0] - 1, bounds[1] + 1)
ax.set_ylim(bounds[0] - 1, bounds[1] + 1)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)

# Plot Voronoi diagram
voronoi_plot_2d(vor, ax=ax, show_vertices=True, line_colors='blue',
                line_width=2, point_size=10)

# Highlight edge midpoints (Voronoi generators)
ax.plot(edge_midpoints_np[:, 0], edge_midpoints_np[:, 1], 'ro', 
       markersize=8, label='Edge midpoints (generators)', zorder=6)

# Highlight Voronoi vertices
ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'g^', 
       markersize=10, label='Voronoi vertices', zorder=7)

ax.legend(loc='upper right', fontsize=12)
ax.text(0.02, 0.98, f'Voronoi cells (one per edge): {len(edge_midpoints_np)}\nVoronoi vertices: {len(vor.vertices)}', 
       transform=ax.transAxes, fontsize=11, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()

print("✓ Step 3: Voronoi diagram of edges visualized")

##%% Visualize Step 4: Combined View - Polygons + Edge Voronoi
fig, ax = plt.subplots(figsize=(14, 14))

ax.set_title('Step 4: Original Polygons + Edge Voronoi Diagram', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(bounds[0] - 1, bounds[1] + 1)
ax.set_ylim(bounds[0] - 1, bounds[1] + 1)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)

colors = plt.cm.Set3(np.linspace(0, 1, len(polygons)))

# Plot original polygons
for i, polygon in enumerate(polygons):
    poly_np = polygon.cpu().numpy()
    patch = MPLPolygon(poly_np, facecolor=colors[i], edgecolor='black', 
                      linewidth=2, alpha=0.3)
    ax.add_patch(patch)

# Draw edges with thicker lines
edge_idx = 0
for poly_idx, polygon in enumerate(polygons):
    num_vertices = polygon.shape[0]
    for i in range(num_vertices):
        start, end = edges[edge_idx]
        start_np = start.cpu().numpy()
        end_np = end.cpu().numpy()
        ax.plot([start_np[0], end_np[0]], [start_np[1], end_np[1]], 
               '-', color=colors[poly_idx], linewidth=4, alpha=0.7, zorder=3)
        edge_idx += 1

# Plot Voronoi diagram on top
voronoi_plot_2d(vor, ax=ax, show_vertices=True, line_colors='blue',
                line_width=1.5, point_size=0, line_alpha=0.6)

# Plot edge midpoints
ax.plot(edge_midpoints_np[:, 0], edge_midpoints_np[:, 1], 'ro', 
       markersize=6, label='Edge midpoints', zorder=5, alpha=0.8)

# Highlight Voronoi vertices
ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'g^', 
       markersize=8, label='Voronoi vertices', zorder=6, alpha=0.7)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', linewidth=3, label='Polygon edges'),
    Line2D([0], [0], color='blue', linewidth=1.5, label='Voronoi edges'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
           markersize=8, label='Edge midpoints'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
           markersize=8, label='Voronoi vertices')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

ax.text(0.02, 0.98, 'Each Voronoi cell corresponds to one polygon edge', 
       transform=ax.transAxes, fontsize=12, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.show()

print("✓ Step 4: Combined view visualized")

##%% Analyze Voronoi Cells for Each Edge
print("\n" + "="*60)
print("VORONOI CELL ANALYSIS")
print("="*60)

for i, region_idx in enumerate(vor.point_region):
    region = vor.regions[region_idx]
    
    if -1 not in region and len(region) > 0:
        print(f"\nEdge {i}:")
        print(f"  Voronoi cell vertices: {region}")
        print(f"  Number of vertices: {len(region)}")
    else:
        print(f"\nEdge {i}:")
        print(f"  Unbounded Voronoi cell (extends to infinity)")

print("\n" + "="*60)
print("All steps completed successfully!")
print("="*60)
# %%
