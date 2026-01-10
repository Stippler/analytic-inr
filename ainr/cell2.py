from dataclasses import dataclass
from typing import List
import numpy as np
from ainr.model import ReluMLP
from ainr.vis import plot_cell_sdf2 as plot_cell_sdf
from typing import Tuple
import torch
import copy
from sklearn.decomposition import PCA
from ainr.ground_truth import generate_polygons
from ainr.vis import plot_polygons

@dataclass
class Surface2D:
    vertices: np.ndarray
    closed: bool

@dataclass
class Task:
    normal: np.ndarray
    offset: float
    
    threshold: float

    double: bool

@dataclass
class Cell:
    surfaces: List[Surface2D]
    weight: np.ndarray
    bias: np.ndarray
    
    tasks: List[Task]

@dataclass
class Hyperplane:
    neuron_weight: np.ndarray
    neuron_bias: float

def split_surfaces(surfaces: List[Surface2D], normal: np.ndarray, offset: float, eps=1e-8):
    inside_surfaces = []
    outside_surfaces = []
    
    for orig_surface in surfaces:
        SD = orig_surface.vertices @ normal + offset
    
        SD_min = SD.min()
        SD_max = SD.max()

        if SD_min > -eps:
            # Case 1.1: No split - all vertices are positive
            outside_surfaces.append(orig_surface)
            continue
        elif SD_max < eps:
            # Case 1.2: No split - all vertices are negative
            inside_surfaces.append(orig_surface)
            continue
        
        V = orig_surface.vertices
        ON = np.abs(SD) <= eps
        S = np.sign(SD).astype(np.int8)
        S[ON] = 0

        n_verts = len(V)
        
        current_V = []
        first_inside_idx = len(inside_surfaces)
        first_outside_idx = len(outside_surfaces)

        for i in range(0, n_verts):
            if ON[i]:
                current_V.append(V[i])

                # Determine previous sign
                if orig_surface.closed or i > 0:
                    prev_sign = S[i-1]
                else:
                    # i=0 and open surface - first vertex is on plane, no previous sign
                    current_V = [V[i]]
                    continue
                
                new_surface = Surface2D(
                    np.array(current_V),
                    False,
                )
                if prev_sign == -1:
                    inside_surfaces.append(new_surface)
                else:
                    outside_surfaces.append(new_surface)

                current_V = [V[i]]
                continue
            elif (orig_surface.closed or i > 0) and S[i] * S[i-1] == -1:
                denom = SD[i] - SD[i-1]
                t = -SD[i-1] / denom
                P = V[i-1] + t * (V[i] - V[i-1])
                current_V.append(P)

                new_surface = Surface2D(
                    np.array(current_V),
                    False,
                )
                if S[i-1]==-1:
                    inside_surfaces.append(new_surface)
                else:
                    outside_surfaces.append(new_surface)
                current_V = [P, V[i]]
                continue

            current_V.append(V[i])
        
        # Handle remaining vertices  
        if orig_surface.closed and len(current_V) > 0 and (len(inside_surfaces) > first_inside_idx or 
                                    len(outside_surfaces) > first_outside_idx):
            wrap_to_inside = (S[-1] == -1) or (S[-1] == 0 and S[0] == -1)
            
            if wrap_to_inside and len(inside_surfaces) > first_inside_idx:
                inside_surfaces[first_inside_idx].vertices = np.concatenate([
                    np.array(current_V), 
                    inside_surfaces[first_inside_idx].vertices
                ])
            elif not wrap_to_inside and len(outside_surfaces) > first_outside_idx:
                outside_surfaces[first_outside_idx].vertices = np.concatenate([
                    np.array(current_V), 
                    outside_surfaces[first_outside_idx].vertices
                ])
            elif len(current_V) > 1:  # Fallback: create new surface if wrapping failed
                new_surface = Surface2D(np.array(current_V), False)
                if wrap_to_inside:
                    inside_surfaces.append(new_surface)
                else:
                    outside_surfaces.append(new_surface)
        elif len(current_V) > 1:
            new_surface = Surface2D(np.array(current_V), False)
            if S[-1] == -1:
                inside_surfaces.append(new_surface)
            else:
                outside_surfaces.append(new_surface)
    
    return inside_surfaces, outside_surfaces

def split_cell(cell: Cell, hyperplane: Hyperplane, eps=1e-8):
    # project to 2d
    effective_normal_unnorm = hyperplane.neuron_weight @ cell.weight
    effective_offset_unnorm = hyperplane.neuron_weight @ cell.bias + hyperplane.neuron_bias

    norm = np.linalg.norm(effective_normal_unnorm)
    if norm < 1e-8:
        return [cell]

    effective_normal = effective_normal_unnorm / norm
    effective_offset = effective_offset_unnorm / norm

    inside_surfaces, outside_surfaces = split_surfaces(cell.surfaces, effective_normal, effective_offset, eps)
    new_cells = []
    if len(inside_surfaces) > 0:
        new_cells.append(Cell(inside_surfaces, cell.weight, cell.bias, cell.tasks.copy()))
    if len(outside_surfaces) > 0:
        new_cells.append(Cell(outside_surfaces, cell.weight, cell.bias, cell.tasks.copy()))
    return new_cells

def get_cell_vertices(cell: Cell):
    vertices = []
    for surface in cell.surfaces:
        vertices.extend(surface.vertices)
    return np.array(vertices)

def compute_pca(cell: Cell):
    """
    Compute PCA on cell vertices to extract principal directions.
    Returns a list of Tasks, each representing a hyperplane split along a principal component.
    Each component is doubled (positive and negative directions).
    """
    # Collect all vertices
    vertices = get_cell_vertices(cell)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(vertices)
    
    # Get principal components (these become our normals)
    components = pca.components_  # Shape: (2, 2)
    mean = pca.mean_  # Center of the data
    
    tasks = []
    
    # For each principal component
    for i in range(2):
        normal = components[i]  # Principal axis direction
        
        # Compute offset: project mean onto the normal
        # Hyperplane equation: normal @ x + offset = 0
        # For plane passing through mean: offset = -normal @ mean
        offset = -np.dot(normal, mean)
        
        # Create threshold for constraint solving (can be tuned)
        threshold = 1e-6
        
        # Add the task with this normal direction
        tasks.append(Task(
            normal=normal,
            offset=offset,
            threshold=threshold,
            double=True
        ))
    
    return tasks

def compute_tasks(cell: Cell):
    tasks = compute_pca(cell)
    cell.tasks = tasks
    return cell

def compute_vertex_plane(surface: Surface2D):
    V = surface.vertices
    middle_index = len(V)//2
    left_index = middle_index - 1
    right_index = middle_index + 1

    vertex = V[middle_index]
    vertex_prev = V[left_index]
    vertex_next = V[right_index]

    # Compute edge vectors
    edge_before = vertex - vertex_prev  # Points TO current vertex
    edge_after = vertex_next - vertex   # Points FROM current vertex

    # Normalize edge directions
    edge_before_norm = edge_before / (np.linalg.norm(edge_before) + 1e-8)
    edge_after_norm = edge_after / (np.linalg.norm(edge_after) + 1e-8)

    # FLIP edge_before since it points backwards (into the vertex)
    # We want both to point "forward" along the polygon boundary
    edge_before_norm = -edge_before_norm

    # Average the edge directions to get the tangent bisector
    tangent_bisector = (edge_before_norm + edge_after_norm) / 2
    tangent_bisector = tangent_bisector / (np.linalg.norm(tangent_bisector) + 1e-8)

    # Rotate tangent bisector 90° CCW to get the boundary normal (perpendicular to bisector)
    boundary_normal = np.stack([-tangent_bisector[1], tangent_bisector[0]])

    # Compute offsets
    offset = -np.sum(boundary_normal * vertex)
    
    return boundary_normal, offset

def get_split_for_largest_surface(surfaces: List[Surface2D]):
    surface_count = np.array([
        len(surface.vertices) if surface.closed else len(surface.vertices)-2
        for surface in surfaces
    ])

    if np.all(surface_count==0):
        return None, None

    surface_idx = np.argmax(surface_count)
    surface = surfaces[surface_idx]
    return compute_vertex_plane(surface)

def get_largest_surfaces(surfaces: List[Surface2D]):
    surface_count = np.array([
        len(surface.vertices) if surface.closed else len(surface.vertices)-2
        for surface in surfaces
    ])
    
    if np.all(surface_count==0):
        return None, None
    
    surface_indices = np.argsort(surface_count)[::-1]
    return surface_indices

def get_largest_cells(cells: List[Cell]):
    vertex_counts = np.array([
        np.sum([len(surface.vertices) if surface.closed else len(surface.vertices)-2 for surface in cell.surfaces])
        for cell in cells
    ])

    if np.all(vertex_counts==0):
        return None
    
    cell_indices = np.argsort(vertex_counts)[::-1]
    return cell_indices

def add_hyperplane_constraint(weight: np.ndarray,
                              bias: np.ndarray,
                              A: np.ndarray,
                              y: np.ndarray,
                              normal: np.ndarray,
                              offset: float,
                              hidden_dim: int,
                              surface_idx: int,
                              threshold: float):
    """
    Add a constraint that the layer should produce a specific hyperplane split.
    
    We want: w_layer @ (weight @ x + bias) + b_layer = normal @ x + offset
    This gives us 3 equations (2 for normal components, 1 for offset).
    """
    row_x = np.zeros(hidden_dim + 1)
    row_x[:hidden_dim] = weight[:, 0]

    row_y = np.zeros(hidden_dim + 1)
    row_y[:hidden_dim] = weight[:, 1]

    row_c = np.zeros(hidden_dim + 1)
    row_c[:hidden_dim] = bias
    row_c[hidden_dim] = 1.0

    A_new = np.vstack([row_x, row_y, row_c])
    y_new = np.array([normal[0], normal[1], offset])

    A_candidate = np.vstack([A, A_new]) if A.shape[0] > 0 else A_new
    y_candidate = np.hstack([y, y_new]) if y.shape[0] > 0 else y_new

    # Solve the system
    solution, residuals, rank, s = np.linalg.lstsq(A_candidate, y_candidate, rcond=None)
    
    # Verify solution accuracy
    verification = A_candidate @ solution
    error = np.abs(verification - y_candidate)
    max_error = np.max(error)

    if max_error > threshold:
        print(f"Surface {surface_idx}: Solution error too large - skipping")
        return False, A, y, None
    
    return True, A_candidate, y_candidate, solution


def solve_layer_weights(cells: List[Cell], 
                        hidden_dim: int,
                        layer_idx: int,
                        neuron_pair_idx: int,
                        mlp: ReluMLP=None,
                        vis=False):
    """
    Solve for layer weights by accumulating hyperplane constraints from all surfaces.
    
    Returns:
        Tuple of (w_hidden_data, w_hidden_offset) or (None, None) if no solution found
    """
    solution = None
    A = np.empty((0, hidden_dim + 1))
    y = np.empty(0)
    
    sort = True
    if sort:
        indices = get_largest_cells(cells)
        cells = [cells[i] for i in indices]
    for cell_idx, cell in enumerate(cells):
        if len(cell.tasks)==0:
            compute_tasks(cell)
        
        success = False
        for i, task in enumerate(cell.tasks):
            success, A, y, new_solution = add_hyperplane_constraint(
                cell.weight, cell.bias, A, y, task.normal, task.offset, hidden_dim, cell_idx, task.threshold
            )
            if success:
                # Delete the task that was successfully used
                del cell.tasks[i]
                break
        
        if success:
            if vis:
                mlp.layers[layer_idx].weight.data[2*neuron_pair_idx, :] = torch.tensor(new_solution[:hidden_dim], dtype=torch.float32)
                mlp.layers[layer_idx].bias.data[2*neuron_pair_idx] = torch.tensor(new_solution[hidden_dim], dtype=torch.float32)
                mlp.layers[layer_idx].weight.data[2*neuron_pair_idx+1, :] = torch.tensor(-new_solution[:hidden_dim], dtype=torch.float32)
                mlp.layers[layer_idx].bias.data[2*neuron_pair_idx+1] = torch.tensor(-new_solution[hidden_dim], dtype=torch.float32)
                plot_cell_sdf(mlp, cells=cells, title=f"Neuron for {cell_idx}: Successfully added", highlight_idx=cell_idx)
            solution = new_solution
        elif vis:
            plot_cell_sdf(mlp, cells=cells, title=f"Neuron for {cell_idx}: Failed to add", highlight_idx=cell_idx)
    
    if solution is None:
        return None, None
    
    w_hidden_data = solution[:hidden_dim]
    w_hidden_offset = solution[hidden_dim]
    return w_hidden_data, w_hidden_offset


def apply_layer_splits(cells: List[Cell],
                       w_hidden_data: np.ndarray,
                       w_hidden_offset: float):
    """
    Apply a layer's learned weights to split all surfaces.
    
    For each cell, compute the effective hyperplane in input space and split the cell into two new cells.
    """
    new_cells = []
    hyperplane = Hyperplane(neuron_weight=w_hidden_data, neuron_bias=w_hidden_offset)
    
    for cell in cells:
        new_cells += split_cell(cell, hyperplane)
    return new_cells


def collapse_layers(cells: List[Cell],
                    next_layer_weight: torch.Tensor,
                    next_layer_bias: torch.Tensor):
    new_cells = []
    
    for cell in cells:
        # Compute previous layer output at a representative point
        vertices = get_cell_vertices(cell)
        vertex = np.mean(vertices, axis=0)
        prev_out = cell.weight @ vertex + cell.bias
        
        # Get next layer weights
        next_weight = np.array(next_layer_weight)
        next_bias = np.array(next_layer_bias)
        
        # Check which next layer neurons are active
        next_preact = next_weight @ prev_out + next_bias
        mask = next_preact < 0  # Inactive neurons
        
        # Zero out inactive neurons
        next_weight_collapsed = next_weight.copy()
        next_bias_collapsed = next_bias.copy()
        next_weight_collapsed[mask, :] = 0
        next_bias_collapsed[mask] = 0
        
        # Compose the layers
        composed_weight = next_weight_collapsed @ cell.weight
        composed_bias = next_weight_collapsed @ cell.bias + next_bias_collapsed
        
        new_cells.append(Cell(cell.surfaces, composed_weight, composed_bias, cell.tasks.copy()))
    
    return new_cells


def solve_output_layer_analytically(mlp: ReluMLP,
                                   cells: List[Cell]):
    A_rows = []
    y_targets = []
    
    # For each cell, iterate through surfaces
    for cell in cells:
        weight = cell.weight
        bias = cell.bias
        
        for surface in cell.surfaces:
            if len(surface.vertices) < 2:
                continue
            
            # Get first and last vertex
            V0, V1 = surface.vertices[0], surface.vertices[-1]
            edge = V1 - V0
            
            # Normal vector (-dy, dx)
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            
            if norm_len < 1e-9:
                continue
            
            # Normalize and negate to flip SDF: negative inside, positive outside
            unit_normal = normal / norm_len
            target_n = -unit_normal
            target_c = np.dot(V0, unit_normal)
            
            # Build constraint matrix
            # We want: W_out @ (weight @ x + bias) + b_out = target_n @ x + target_c
            # Expanding: (W_out @ weight) @ x + (W_out @ bias + b_out) = target_n @ x + target_c
            
            # Row 1: Match x-component of normal
            row_x = np.zeros(mlp.hidden_dim + 1)
            row_x[:mlp.hidden_dim] = weight[:, 0]
            A_rows.append(row_x)
            y_targets.append(target_n[0])
            
            # Row 2: Match y-component of normal
            row_y = np.zeros(mlp.hidden_dim + 1)
            row_y[:mlp.hidden_dim] = weight[:, 1]
            A_rows.append(row_y)
            y_targets.append(target_n[1])
            
            # Row 3: Match offset
            row_c = np.zeros(mlp.hidden_dim + 1)
            row_c[:mlp.hidden_dim] = bias
            row_c[mlp.hidden_dim] = 1.0
            A_rows.append(row_c)
            y_targets.append(target_c)
    
    # Solve the least squares system
    A = np.array(A_rows)
    y = np.array(y_targets)
    
    solution, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    
    # Verify solution quality
    verification = A @ solution
    error = np.abs(verification - y)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error ** 2))
    
    print(f"  Rank={rank}/{A.shape[1]}, RMS error={rms_error:.6f}, max error={max_error:.6f}")
    
    # Extract weights and bias
    w_out = solution[:mlp.hidden_dim]
    b_out = solution[mlp.hidden_dim]
    
    # Set the output layer parameters
    output_layer_idx = mlp.num_layers
    mlp.layers[output_layer_idx].weight.data = torch.tensor(w_out.reshape(1, -1), dtype=torch.float32)
    mlp.layers[output_layer_idx].bias.data = torch.tensor([b_out], dtype=torch.float32)
    
    return True


def build_layer(mlp: ReluMLP, 
               layer_idx: int,
               cells: List[Cell],
               vis=False):
    """
    Build a single layer by solving for neuron weights and applying splits.
    
    Args:
        mlp: The MLP being constructed
        layer_idx: Which layer to build (0, 1, 2, ...)
        cells: Current list of cells
        vis: Whether to visualize the process
        
    Returns:
        Updated cells after this layer's splits
    """
    
    for neuron_pair_idx in range(mlp.hidden_dim // 2):
        hidden_dim = mlp.hidden_dim if layer_idx > 0 else mlp.input_dim
        w_hidden_data, w_hidden_offset = solve_layer_weights(cells, hidden_dim, layer_idx, neuron_pair_idx, mlp, vis)
        
        if w_hidden_data is None:
            continue
        
        # Apply the splits to all surfaces
        cells = apply_layer_splits(cells, w_hidden_data, w_hidden_offset)
        
        # Set neuron weights (positive and negative pairs)
        mlp.layers[layer_idx].weight.data[2*neuron_pair_idx, :] = torch.tensor(w_hidden_data, dtype=torch.float32)
        mlp.layers[layer_idx].bias.data[2*neuron_pair_idx] = torch.tensor(w_hidden_offset, dtype=torch.float32)
        mlp.layers[layer_idx].weight.data[2*neuron_pair_idx+1, :] = torch.tensor(-w_hidden_data, dtype=torch.float32)
        mlp.layers[layer_idx].bias.data[2*neuron_pair_idx+1] = torch.tensor(-w_hidden_offset, dtype=torch.float32)
        if vis:
            plot_cell_sdf(mlp, cells=cells, title=f"Layer {layer_idx + 1}, Neuron pair {neuron_pair_idx}: Applied splits")
    
    return cells