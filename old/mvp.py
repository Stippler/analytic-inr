#%%
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

#%%
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

#%%

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
        ON = np.abs(SD) <= eps
        S  = np.sign(SD).astype(np.int8)
        S[ON] = 0

        N = len(V)

        current_V = [V[0]]
        current_N = []
        current_O = []
        segments = []

        for i in range(0, N-1):
            if ON[i]:
                # vertex is on the edge
                current_V.append[V[0]]
                segment = LineSegments(
                    vertices,
                    np.array(current_N),
                    np.array(current_O),
                    False,
                    0 if S[i-1]==-1 else 1
                )
                segments.append(segment)

                current_V = [V[i]]
                current_N = []
                current_O = []
                continue
            elif S[i-1]*S[i] == -1:
                # edge is cut
                denom = ...
                t = ...
                P = ...
                current_V.append(P)

                segment = LineSegments(
                    vertices,
                    np.array(current_N),
                    np.array(current_O),
                    False,
                    0 if S[i-1]==-1 else 1
                )

                segments.append(segment)
                current_V = [P, S[i]]
                continue
            
            
                

        if self.closed:
            V0, V1  = V,  np.roll(V,  -1, axis=0)
            S0, S1  = S,  np.roll(S,  -1, axis=0)
            SD0,SD1 = SD, np.roll(SD, -1, axis=0)
            n_edges = len(V)
        else:
            V0, V1  = V[:-1], V[1:]
            S0, S1  = S[:-1], S[1:]
            SD0,SD1 = SD[:-1], SD[1:]
            n_edges = len(V) - 1

        # 4 possibilites:
        # 1. Edge crosses
        intersect = (S0*S1) < 0 # True crossing
        # 2. Start vertex on edge
        touch0 = (S0==0) & (S1!=0)
        # 3. End vertex on edge
        touch1 = (S1==0) & (S0!=0)
        # 4. Both vertices on edge
        touch2 = (S0==0) & (S1==0)

        cross = intersect | touch0 | touch1 | touch2

        # Build new segments
        # Get edge chains
        indices0 = np.argwhere(cross)
        indices1 = np.roll(indices0, -1)

        if not self.closed:
            indices0[:-1]
            indices1[:-1]
        
        # Easiest way I can think of is to start with first crossing edge
        # Depending on start/end edge there are all combination of cases depending on the starting and ending edge
        # For example:
        # touch2 is true, then the edge is a new line segment and we are done
        # intersect for first and last edge is true then we have to add two new vertices in start and end and we are done
        # normals and offsets must only be stored for intermediate vertices, i.e. not for start or end vertices
        # only if it is closed one normal/offset has to be stored for the start and the end

        return result

line_segments: List[LineSegments] = []

polygons = generate_polygons('1x32', convex=True)

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

print(f"Created {len(line_segments)} line segments")
for i, seg in enumerate(line_segments):
    print(f"  Segment {i}: {len(seg.vertices)} vertices, closed={seg.closed}")


#%%

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

#%%

# Network parameters
width = 8   # neurons per layer
depth = 1   # number of hidden layers

# Initialize model
model = ReluMLP(input_dim=2, hidden_dim=width, num_layers=depth, output_dim=1)

# Initialize first layer by sampling vertices from line segments and splitting
current_segments = line_segments.copy()
first_layer_normals = []
first_layer_offsets = []

with torch.no_grad():
    first_layer = model.layers[0]
    
    for neuron_idx in range(width):
        # Collect all vertices with normals from current segments
        all_verts = []
        all_norms = []
        all_offs = []
        all_seg_indices = []
        
        for seg_idx, seg in enumerate(current_segments):
            n_verts = len(seg.vertices)
            
            # Skip segments with too few vertices
            if n_verts < 2:
                continue
            
            if seg.closed:
                # Closed: all vertices have normals
                for v_idx in range(n_verts):
                    if v_idx < len(seg.normals):
                        all_verts.append(seg.vertices[v_idx])
                        all_norms.append(seg.normals[v_idx])
                        all_offs.append(seg.offsets[v_idx])
                        all_seg_indices.append(seg_idx)
            else:
                # Open: only interior vertices have normals
                # normals[i] corresponds to vertices[i+1]
                for v_idx in range(1, n_verts - 1):
                    norm_idx = v_idx - 1
                    if norm_idx < len(seg.normals):
                        all_verts.append(seg.vertices[v_idx])
                        all_norms.append(seg.normals[norm_idx])
                        all_offs.append(seg.offsets[norm_idx])
                        all_seg_indices.append(seg_idx)
        
        # Verify lists are in sync
        assert len(all_verts) == len(all_norms) == len(all_offs), \
            f"List sizes don't match: {len(all_verts)} verts, {len(all_norms)} norms, {len(all_offs)} offs"
        
        if len(all_verts) == 0:
            print(f"Warning: No valid vertices found for neuron {neuron_idx}/{width}")
            print(f"  Current segments: {len(current_segments)}")
            for i, seg in enumerate(current_segments[:5]):  # Show first 5
                print(f"    Seg {i}: {len(seg.vertices)} verts, closed={seg.closed}, "
                      f"{len(seg.normals)} normals")
            break
        
        # Randomly sample one vertex
        sample_idx = np.random.randint(0, len(all_verts))
        sampled_normal = all_norms[sample_idx]
        sampled_offset = all_offs[sample_idx]
        
        if len(sampled_normal) != 2:
            print(f"Warning: Invalid normal dimension: {sampled_normal.shape}")
            break
        
        # Set neuron weights
        first_layer.weight[neuron_idx] = torch.tensor(sampled_normal, dtype=torch.float32)
        first_layer.bias[neuron_idx] = torch.tensor(sampled_offset, dtype=torch.float32)
        
        first_layer_normals.append(sampled_normal)
        first_layer_offsets.append(sampled_offset)
        
        # Split all current segments with this neuron's hyperplane
        new_segments = []
        for seg in current_segments:
            split_segs = seg.split(sampled_normal, sampled_offset)
            new_segments.extend(split_segs)
        
        current_segments = new_segments
        print(f"Neuron {neuron_idx+1}/{width}: {len(current_segments)} segments after split")
    
    # Initialize output layer analytically with gradient constraints
    # For proper SDF: output=0 on surface, positive outside, negative inside
    
    print("\nInitializing output layer from linear regions...")
    
    # Get the output layer
    output_layer = model.layers[-1]
    
    # Collect constraints with gradient information
    A_list = []
    targets = []
    epsilon = 0.02  # Distance to sample points inside/outside
    
    for seg_idx, segment in enumerate(current_segments):
        # Skip closed segments as they form closed loops
        if segment.closed:
            print(f"  Skipping closed segment {seg_idx}")
            continue
        
        # Skip segments with insufficient vertices
        if len(segment.vertices) < 2:
            print(f"  Skipping segment {seg_idx} with only {len(segment.vertices)} vertices")
            continue
        
        vertices = segment.vertices
        
        # Try to get normals from the segment's stored normals
        # If not available, compute from edge direction
        if len(segment.normals) > 0:
            # Use the first available normal from the segment
            normal = segment.normals[0]
            normal = normal / (np.linalg.norm(normal) + 1e-8)
        else:
            # Fallback: compute normal from segment direction
            direction = vertices[-1] - vertices[0]
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm < 1e-8:
                # Degenerate segment
                print(f"  Skipping degenerate segment {seg_idx}")
                continue
            
            direction = direction / direction_norm
            # Normal (perpendicular, rotate 90 degrees CCW)
            normal = np.array([-direction[1], direction[0]])
        
        # Use first and last vertices
        # For each vertex, add three constraints:
        # 1. output = 0 at the vertex
        # 2. output > 0 slightly outside (along normal)
        # 3. output < 0 slightly inside (against normal)
        for vertex in [vertices[0], vertices[-1]]:
            # 1. Constraint at the surface: output = 0
            x_surface = torch.tensor(vertex, dtype=torch.float32).unsqueeze(0)
            h_surface = x_surface
            for layer in model.layers[:-1]:
                if isinstance(layer, nn.Linear):
                    h_surface = layer(h_surface)
                elif isinstance(layer, nn.ReLU):
                    h_surface = torch.relu(h_surface)
            
            h_aug = torch.cat([h_surface.squeeze(0), torch.ones(1)])
            A_list.append(h_aug)
            targets.append(0.0)
            
            # 2. Constraint outside: output = +epsilon (positive)
            x_outside = vertex + epsilon * normal
            x_outside = torch.tensor(x_outside, dtype=torch.float32).unsqueeze(0)
            h_outside = x_outside
            for layer in model.layers[:-1]:
                if isinstance(layer, nn.Linear):
                    h_outside = layer(h_outside)
                elif isinstance(layer, nn.ReLU):
                    h_outside = torch.relu(h_outside)
            
            h_aug = torch.cat([h_outside.squeeze(0), torch.ones(1)])
            A_list.append(h_aug)
            targets.append(epsilon)
            
            # 3. Constraint inside: output = -epsilon (negative)
            x_inside = vertex - epsilon * normal
            x_inside = torch.tensor(x_inside, dtype=torch.float32).unsqueeze(0)
            h_inside = x_inside
            for layer in model.layers[:-1]:
                if isinstance(layer, nn.Linear):
                    h_inside = layer(h_inside)
                elif isinstance(layer, nn.ReLU):
                    h_inside = torch.relu(h_inside)
            
            h_aug = torch.cat([h_inside.squeeze(0), torch.ones(1)])
            A_list.append(h_aug)
            targets.append(-epsilon)
    
    print(f"  Collected {len(targets)} constraints from {len(current_segments)} segments")
    
    if len(A_list) == 0:
        print("  Warning: No constraints collected, using random initialization")
    else:
        A_matrix = torch.stack(A_list, dim=0)  # Shape: (num_constraints, hidden_dim + 1)
        target_vector = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
        # Solve least-squares
        solution = torch.linalg.lstsq(A_matrix, target_vector).solution
        
        # Extract weights and bias
        W_out = solution[:-1, :].T  # Shape: (1, hidden_dim)
        b_out = solution[-1, :]     # Shape: (1,)
        
        # Set the output layer parameters
        output_layer.weight.data = W_out
        output_layer.bias.data = b_out
        
        # Verify the solution
        residuals = A_matrix @ solution - target_vector
        rms_error = torch.sqrt(torch.mean(residuals ** 2))
        max_error = torch.abs(residuals).max()
        print(f"  Solution: RMS error = {rms_error.item():.6f}, Max error = {max_error.item():.6f}")

print(f"\nFinal: {len(current_segments)} segments after all splits")

# Diagnostic: count segment types
closed_count = sum(1 for seg in current_segments if seg.closed)
open_count = len(current_segments) - closed_count
open_with_normals = sum(1 for seg in current_segments if not seg.closed and len(seg.normals) > 0)
print(f"  Closed segments: {closed_count}")
print(f"  Open segments: {open_count}")
print(f"  Open segments with normals: {open_with_normals}")

# Show a few segment examples
print("\nSample segments:")
for i, seg in enumerate(current_segments[:5]):
    print(f"  Segment {i}: {len(seg.vertices)} vertices, closed={seg.closed}, "
          f"{len(seg.normals)} normals, activations={seg.activations}")

# Visualize the result
plot_cell_sdf(model, line_segments=current_segments)
#%%

"""


        # den = SD1 - SD0
        # valid = cross & (np.abs(den)>0)
        # t = np.where(valid, -SD0/den, 0.0)
        # P = V0+t[:, None] * (V1-V0)

        start_idx = np.argmax()
        idx = start_idx

        segments = []
        current_segment = []
        while idx!=start_idx:

            
            
            idx = (idx+1) % n_edges


        # TODO:
        # Find flips (1 to -1, -1 to 1) and insert new vertices there
        # Find on edges (1 to 0, -1 to 0)
        # handle edge cases (0 to 0)
        # Put them into different line segments with different activations


        
        result = []
        current_verts = []
        current_normals = []
        current_offsets = []
        
        n_verts = len(self.vertices)
        num_edges = n_verts if self.closed else n_verts - 1
        
        for i in range(num_edges):
            v0, v1 = self.vertices[i], self.vertices[(i + 1) % n_verts]
            dist0, dist1 = SD[i], SD[(i + 1) % n_verts]
            
            # Classify vertices relative to split line
            v0_on_line = abs(dist0) <= eps
            v1_on_line = abs(dist1) <= eps
            
            # Check if v0 is on the line AND we have accumulated vertices - finalize segment
            if v0_on_line and len(current_verts) >= 1:
                # v0 is on split line - add v0 with its normal (if interior) to current segment
                current_verts.append(v0)
                # Add normal for v0 if it's an interior vertex of the segment being finalized
                if len(current_verts) > 1:  # v0 is not the first vertex of this segment
                    if self.closed:
                        if i < len(self.normals):
                            current_normals.append(self.normals[i])
                            current_offsets.append(self.offsets[i])
                    elif 0 < i < n_verts - 1:
                        # For open segments: normals[i-1] corresponds to vertices[i]
                        if i - 1 < len(self.normals):
                            current_normals.append(self.normals[i - 1])
                            current_offsets.append(self.offsets[i - 1])
                
                # Finalize current segment
                if len(current_verts) >= 2:
                    verts = np.array(current_verts)
                    norms = np.array(current_normals) if current_normals else np.zeros((0, 2))
                    offs = np.array(current_offsets) if current_offsets else np.zeros(0)
                    activation = 1 if (verts[0] @ n + d) > 0 else 0
                    new_activations = self.activations + [activation]
                    result.append(LineSegments(verts, norms, offs, closed=False, 
                                              activations=new_activations))
                
                # Start new segment with v0 (no normal for first vertex of open segment)
                current_verts = [v0]
                current_normals = []
                current_offsets = []
                continue
            
            # Check if edge crosses the line between vertices (neither on the line)
            if not v0_on_line and not v1_on_line and dist0 * dist1 < 0:
                t = np.clip(-dist0 / (dist1 - dist0 + eps), 0, 1)
                intersection = v0 + t * (v1 - v0)
                
                # Add v0 with its normal/offset if applicable
                current_verts.append(v0)
                if len(current_verts) > 1:  # v0 is interior vertex
                    if self.closed:
                        current_normals.append(self.normals[i])
                        current_offsets.append(self.offsets[i])
                    elif 0 < i < n_verts - 1:
                        # For open segments: normals[i-1] corresponds to vertices[i]
                        if i - 1 < len(self.normals):
                            current_normals.append(self.normals[i - 1])
                            current_offsets.append(self.offsets[i - 1])
                
                # Add intersection point (ends this segment)
                current_verts.append(intersection)
                
                if len(current_verts) >= 2:
                    verts = np.array(current_verts)
                    norms = np.array(current_normals) if current_normals else np.zeros((0, 2))
                    offs = np.array(current_offsets) if current_offsets else np.zeros(0)
                    activation = 1 if (verts[0] @ n + d) > 0 else 0
                    new_activations = self.activations + [activation]
                    result.append(LineSegments(verts, norms, offs, closed=False, 
                                              activations=new_activations))
                
                # Start new segment with intersection (no normal for endpoints)
                current_verts = [intersection]
                current_normals = []
                current_offsets = []
            elif v0_on_line and len(current_verts) == 0:
                # First vertex is on the line - start new segment
                current_verts = [v0]
                current_normals = []
                current_offsets = []
            elif v1_on_line and not v0_on_line:
                # v1 is on line but v0 is not - add v0 and prepare to finalize at v1
                current_verts.append(v0)
                if len(current_verts) > 1:  # v0 is interior vertex
                    if self.closed:
                        current_normals.append(self.normals[i])
                        current_offsets.append(self.offsets[i])
                    elif 0 < i < n_verts - 1:
                        if i - 1 < len(self.normals):
                            current_normals.append(self.normals[i - 1])
                            current_offsets.append(self.offsets[i - 1])
            elif v0_on_line and v1_on_line:
                # Both on line - add v0 and continue (edge lies on the split line)
                if len(current_verts) == 0:
                    current_verts.append(v0)
                # Don't add normal for edges that lie on the split line
            else:
                # No split - just add v0 with its normal if applicable
                current_verts.append(v0)
                if len(current_verts) > 1:  # v0 is interior vertex
                    if self.closed:
                        current_normals.append(self.normals[i])
                        current_offsets.append(self.offsets[i])
                    elif 0 < i < n_verts - 1:
                        if i - 1 < len(self.normals):
                            current_normals.append(self.normals[i - 1])
                            current_offsets.append(self.offsets[i - 1])
        
        # Handle final segment
        if not self.closed:
            # For open segments, add final vertex
            if len(current_verts) > 0:
                current_verts.append(self.vertices[-1])
        
        # Finalize last segment
        if len(current_verts) >= 2:
            verts = np.array(current_verts)
            norms = np.array(current_normals) if current_normals else np.zeros((0, 2))
            offs = np.array(current_offsets) if current_offsets else np.zeros(0)
            activation = 1 if (verts[0] @ n + d) > 0 else 0
            new_activations = self.activations + [activation]
            result.append(LineSegments(verts, norms, offs, closed=False, 
                                      activations=new_activations))
        
        # Safety check - should not happen due to early return for no-split case
        if len(result) == 0:
            activation = 1 if SD[0] > 0 else 0
            return [LineSegments(self.vertices.copy(), self.normals.copy(), 
                                self.offsets.copy(), self.closed, 
                                self.activations + [activation])]
        
        # For closed polygons: merge first and last segments if they're on the same side
        if self.closed and len(result) >= 2:
            first_seg = result[0]
            last_seg = result[-1]
            
            # Check if first and last segments have the same activation
            if first_seg.activations[-1] == last_seg.activations[-1]:
                # Merge: last segment + first segment
                # Check if they share an endpoint to avoid duplicating it
                eps_dist = 1e-6
                skip_first = 1 if np.linalg.norm(last_seg.vertices[-1] - first_seg.vertices[0]) < eps_dist else 0
                
                merged_verts = np.vstack([last_seg.vertices, first_seg.vertices[skip_first:]])
                merged_norms = np.vstack([last_seg.normals, first_seg.normals]) if len(last_seg.normals) > 0 and len(first_seg.normals) > 0 else np.zeros((0, 2))
                merged_offs = np.concatenate([last_seg.offsets, first_seg.offsets]) if len(last_seg.offsets) > 0 and len(first_seg.offsets) > 0 else np.zeros(0)
                
                # Replace first and last with merged segment
                merged_seg = LineSegments(merged_verts, merged_norms, merged_offs, 
                                         closed=False, activations=last_seg.activations)
                result = result[1:-1] + [merged_seg]
        
"""