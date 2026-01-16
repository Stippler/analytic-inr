import torch
import torch.nn as nn
from typing import List
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import matplotlib

import copy

class ReluMLP(nn.Module):
    
    def __init__(self, input_dim=2, hidden_dim=3, num_layers=3, output_dim=1):
        """ReLU MLP with arbitrary input and output dimensions."""
        super().__init__()
        layers = []

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=False))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=False))
        
        # Output layer (no ReLU)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    def eval_activations(self, x):
        preacts = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                preacts.append(x.clone())
                x = torch.relu(x)
        return x, torch.stack(preacts, dim=1)

def plot_cell_sdf(
        net,
        resolution: int = 300,
        device = None,
        line_segments = None
    ):
    """
    Plot SDF heatmap with activation boundaries and line segments.
    
    Args:
        net: Neural network model
        resolution: Grid resolution for the heatmap
        device: Device to run computations on
        line_segments: Optional list of LineSegments objects to overlay on the plot
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if net.input_dim != 2 or net.output_dim != 1:
        print(f"Warning: plot_cell_sdf expects 2D input and 1D output, got input_dim={net.input_dim}, output_dim={net.output_dim}")
        return
    
    # Grid & network pass
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)

    with torch.no_grad():
        net = net.to(device)
        sdf_vals, preacts = net.eval_activations(grid_pts.to(device))

    L = net.num_layers
    H = net.hidden_dim

    sdf_grid = sdf_vals.view(resolution, resolution).cpu().numpy()
    preacts = preacts.view(resolution, resolution, L, H).cpu()

    fig, ax = plt.subplots(figsize=(10, 8))

    # SDF heat-map
    vmin, vmax = sdf_grid.min(), sdf_grid.max()
    if vmin >= 0:
        cmap = LinearSegmentedColormap.from_list('pos', ['#ffcccc','red','darkred'])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    elif vmax <= 0:
        cmap = LinearSegmentedColormap.from_list('neg', ['darkblue','blue','#ccccff'])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = LinearSegmentedColormap.from_list('cent', ['blue','white','red'])
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax.contourf(xx.cpu().numpy(), yy.cpu().numpy(), sdf_grid, levels=500, cmap=cmap, norm=norm)

    # Activation boundaries
    colors = ['cyan','orange','magenta','yellow','lime','red','purple','brown'][:L]
    rgba = [plt.matplotlib.colors.to_rgba(c) for c in colors]

    for l in range(L):
        for h in range(H):
            ax.contour(xx.cpu(), yy.cpu(),
                    preacts[:, :, l, h], levels=[0],
                    linewidths=1.2,
                    colors=rgba[l],
                    alpha=0.9,
                    zorder=1)

    # Predicted boundary
    ax.contour(xx.cpu().numpy(), yy.cpu().numpy(), sdf_grid, levels=[0], colors='black',
               linewidths=1.2, linestyles='--')
    
    # Draw line segments if provided
    if line_segments is not None:
        # Group segments by activation pattern
        patterns = {}
        for seg in line_segments:
            pattern = tuple(seg.activations)
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(seg)
        
        for pattern_idx, (pattern, segs) in enumerate(patterns.items()):
            for seg_idx, seg in enumerate(segs):
                verts = seg.vertices
                
                if seg.closed:
                    plot_verts = np.vstack([verts, verts[0]])
                else:
                    plot_verts = verts
                
                # Plot with black color
                ax.plot(plot_verts[:, 0], plot_verts[:, 1], 
                    color='black', linewidth=2.5, alpha=0.9, 
                    linestyle='-', zorder=10)
                
                # Mark vertices
                ax.scatter(verts[:, 0], verts[:, 1], 
                        c='black', s=40, zorder=11, 
                        edgecolors='white', linewidths=1.5)
                
                # Add label with number of vertices
                midpoint = np.mean(verts, axis=0)
                ax.text(midpoint[0], midpoint[1], f'{len(verts)}v', 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.7, edgecolor='black'),
                       zorder=12)
        
    # Add pattern info to title
    
    # Legend
    handles = [mpatches.Patch(color=colors[i], label=f'layer {i+1} boundary') for i in range(L)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'SDF Heat-Map ({len(line_segments)} segments, {len(patterns)} patterns)')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax, format=FormatStrFormatter('%.3f'))
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('SDF value', fontsize=10)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=7)
    cbar.update_ticks()

    plt.tight_layout()
    plt.show()

def generate_polygons(spec, convex=True, star_ratio=0.5, stretch=(1.0, 1.0)):
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
    
    Returns:
    --------
    list of np.ndarray
        List of polygon vertex arrays, each of shape (n_vertices+1, 2)
        The last vertex is repeated to close the polygon
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
        
        # Polygon radius (inscribed in cell with padding, accounting for stretch)
        # Ensure stretched polygon fits within cell boundaries
        radius_x = (cell_width / 2) * 0.8 / sx if sx > 0 else cell_width * 0.4
        radius_y = (cell_height / 2) * 0.8 / sy if sy > 0 else cell_height * 0.4
        radius = min(radius_x, radius_y)
        
        if convex:
            # Generate regular convex polygon vertices
            angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
            angles += idx * 0.3
            
            vertices = np.zeros((n_vertices, 2))
            vertices[:, 0] = cell_center_x + radius * sx * np.cos(angles)
            vertices[:, 1] = cell_center_y + radius * sy * np.sin(angles)
        else:
            # Generate non-convex star-like polygon
            n_points = n_vertices * 2
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            angles += idx * 0.3
            
            # Alternate between outer and inner radii
            radii = np.zeros(n_points)
            radii[::2] = radius
            radii[1::2] = radius * star_ratio
            
            vertices = np.zeros((n_points, 2))
            vertices[:, 0] = cell_center_x + radii * sx * np.cos(angles)
            vertices[:, 1] = cell_center_y + radii * sy * np.sin(angles)
        
        polygons.append(vertices)
    
    return polygons

    

def plot_polygons(polygons, figsize=(8, 8), title="Polygons"):
    """Plot polygons (assumes closed polygons without duplicated last vertex)"""
    plt.figure(figsize=figsize)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(polygons)))
    
    for i, polygon in enumerate(polygons):
        # Close polygon by adding first vertex at end
        closed_poly = np.vstack([polygon, polygon[0]])
        plt.plot(closed_poly[:, 0], closed_poly[:, 1], 'k-', linewidth=2)
        plt.fill(closed_poly[:, 0], closed_poly[:, 1], color=colors[i], alpha=0.3)
        
        # Mark all vertices
        plt.scatter(polygon[:, 0], polygon[:, 1], c='red', s=20, zorder=5)
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


class LineSegments:
    def __init__(self, vertices: np.ndarray, normals: np.ndarray, offsets: np.ndarray, 
                 closed=True, activations=None):
        self.vertices = vertices
        self.closed = closed
        self.normals = normals
        self.offsets = offsets
        self.activations = activations if activations is not None else []
    
    def split(self, n, d, eps=1e-8):
        # Split by hyperplane n·x + d = 0
        SD = self.vertices @ n + d
        
        SD_min = SD.min()
        SD_max = SD.max()

        if SD_min>-eps:
            # Case 1.1: No split - all vertices are positive
            activation = 1
            return [LineSegments(self.vertices.copy(), self.normals.copy(), 
                                 self.offsets.copy(), self.closed, 
                                 self.activations + [activation])]
        elif SD_max<eps:
            # Case 1.2: No split - all vertices are negative
            activation = 0
            return [LineSegments(self.vertices.copy(), self.normals.copy(), 
                                 self.offsets.copy(), self.closed, 
                                 self.activations + [activation])]

        
        V  = self.vertices
        N = self.normals
        O = self.offsets
        ON = np.abs(SD) <= eps
        S  = np.sign(SD).astype(np.int8)
        S[ON] = 0

        n_verts = len(V)
        
        current_V = [V[0]]
        current_N = [N[0]]
        current_O = [O[0]]
        segments = []

        for i in range(1, n_verts):
            if ON[i]:
                current_V.append(V[i])
                current_N.append(N[i])
                current_O.append(O[i])
                segment = LineSegments(
                    np.array(current_V),
                    np.array(current_N),
                    np.array(current_O),
                    False,
                    self.activations + [0 if S[i-1]==-1 else 1]
                )
                segments.append(segment)
                current_V = [V[i]]
                current_N = [N[i]]
                current_O = [O[i]]
                continue
            elif S[i]*S[i-1]==-1:
                denom = SD[i]-SD[i-1]
                t = -SD[i-1]/denom
                P = V[i-1]+t*(V[i]-V[i-1])
                current_V.append(P)
                current_N.append(N[i-1])
                current_O.append(O[i-1])

                segment = LineSegments(
                    np.array(current_V),
                    np.array(current_N),
                    np.array(current_O),
                    False,
                    self.activations + [0 if S[i-1]==-1 else 1]
                )
                segments.append(segment)

                current_V = [P, V[i]]
                current_N = [N[i], N[i]]
                current_O = [O[i], O[i]]
                continue

            current_V.append(V[i])
            current_N.append(N[i])
            current_O.append(O[i])
        
        if self.closed:
            segments[0].vertices = np.concatenate([np.array(current_V), segments[0].vertices])
            segments[0].normals = np.concatenate([np.array(current_N), segments[0].normals])
            segments[0].offsets = np.concatenate([np.array(current_O), segments[0].offsets])
        elif len(current_V)>1:
            segment = LineSegments(
                np.array(current_V),
                np.array(current_N),
                np.array(current_O),
                False,
                self.activations + [0 if S[-1]==-1 else 1]
            )
            segments.append(segment)

        return segments



polygons = generate_polygons('1x32', convex=True)
line_segments = []
for vertices in polygons:
    n_verts = len(vertices)
    closed = True
    
    # Vectorized computation
    # Get previous and next vertices for all vertices at once
    vertices_prev = np.roll(vertices, 1, axis=0)  # Shift down by 1
    vertices_next = np.roll(vertices, -1, axis=0)  # Shift up by 1
    
    # Compute edge vectors
    edges_before = vertices - vertices_prev
    edges_after = vertices_next - vertices
    
    # Compute normals (rotate 90° CCW: (x,y) -> (-y,x))
    normals_before = np.stack([-edges_before[:, 1], edges_before[:, 0]], axis=1)
    normals_after = np.stack([-edges_after[:, 1], edges_after[:, 0]], axis=1)
    
    # Average and normalize
    avg_edge_normals = (normals_before + normals_after) / 2
    norms = np.linalg.norm(avg_edge_normals, axis=1, keepdims=True) + 1e-8
    avg_edge_normals = avg_edge_normals / norms
    
    # Boundary normals (rotate 90° CCW again)
    boundary_normals = np.stack([-avg_edge_normals[:, 1], avg_edge_normals[:, 0]], axis=1)
    
    # Compute offsets
    offsets = -np.sum(boundary_normals * vertices, axis=1)
    
    line_segment = LineSegments(vertices, boundary_normals, offsets, closed)
    line_segments.append(line_segment)

