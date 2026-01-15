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
class Params:
    weights: torch.Tensor
    biases: torch.Tensor

@dataclass
class Spline:
    t_values: List[torch.Tensor]
    vertices: List[torch.Tensor]
    network_values: List[torch.Tensor]

    knots: torch.Tensor
    values: torch.Tensor

@dataclass
class SubCell:
    splines: List[Spline]
    activations: torch.Tensor

@dataclass
class Cell:
    subcells: List[SubCell]
    weight: torch.Tensor
    bias: torch.Tensor

def split_surfaces(splines: List[Spline], normal: torch.Tensor, offset: torch.Tensor, eps=1e-8):
    inside_splines = []
    outside_splines = []
    
    for original_spline in splines:
        SD = original_spline.vertices @ normal + offset
    
        SD_min = SD.min()
        SD_max = SD.max()

        if SD_min > -eps:
            # Case 1.1: No split - all vertices are positive
            outside_surfaces.append(original_spline)
            continue
        elif SD_max < eps:
            # Case 1.2: No split - all vertices are negative
            inside_surfaces.append(original_spline)
            continue
        
        V = original_spline.vertices
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
                if original_spline.closed or i > 0:
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
            elif (original_spline.closed or i > 0) and S[i] * S[i-1] == -1:
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
        if original_spline.closed and len(current_V) > 0 and (len(inside_surfaces) > first_inside_idx or 
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

def split_cell(cell: Cell, neuron_weight, neuron_bias, neuron_idx, is_double, eps=1e-8):
    """
    Split a cell based on hidden layer weights.
    
    Args:
        cell: Cell to split
        neuron_weight: Weight vector for the hyperplane
        neuron_bias: Bias for the hyperplane
        neuron_idx: Index of the neuron being split
        is_double: If True, create two subcells (inside/outside). If False, create only one.
        eps: Tolerance for numerical comparisons
        
    Returns:
        List of new cells
    """
    # project to 2d
    effective_normal_unnorm = neuron_weight @ cell.weight
    effective_offset_unnorm = neuron_weight @ cell.bias + neuron_bias

    norm = np.linalg.norm(effective_normal_unnorm)
    if norm < 1e-8:
        return [cell]

    effective_normal = effective_normal_unnorm / norm
    effective_offset = effective_offset_unnorm / norm

    # Split all surfaces across all subcells
    new_subcells_inside = []
    new_subcells_outside = []
    
    for subcell in cell.subcells:
        inside_surfaces, outside_surfaces = split_surfaces(subcell.surfaces, effective_normal, effective_offset, eps)
        
        if len(inside_surfaces) > 0:
            # Inside = negative side of hyperplane, so activation is 0
            new_activations = subcell.activations.copy()
            new_activations[neuron_idx] = 0.0
            if is_double:
                new_activations[neuron_idx+1] = 1.0
            new_subcells_inside.append(SubCell(inside_surfaces, new_activations))
        
        if len(outside_surfaces) > 0:
            # Outside = positive side of hyperplane, so activation is 1
            new_activations = subcell.activations.copy()
            new_activations[neuron_idx] = 1.0
            if is_double:
                new_activations[neuron_idx+1] = 0.0
            new_subcells_outside.append(SubCell(outside_surfaces, new_activations))
    
    new_cells = []
    # Always combine inside and outside subcells into a single cell
    all_subcells = new_subcells_inside + new_subcells_outside
    if len(all_subcells) > 0:
        new_cells.append(Cell(all_subcells, cell.weight, cell.bias, cell.tasks.copy()))
    
    return new_cells

def get_cell_vertices(cell: Cell):
    vertices = []
    for subcell in cell.subcells:
        for surface in subcell.surfaces:
            vertices.extend(surface.vertices)
    return np.array(vertices)


def compute_pca(cell: Cell):
    # Collect all vertices
    vertices = get_cell_vertices(cell)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(vertices)
    
    # Get principal components (these become our normals)
    components = pca.components_  # Shape: (2, 2)
    mean = pca.mean_  # Center of the data
    
    return components, mean

def compute_tasks_from_pca(components: np.ndarray, mean: np.ndarray):
    tasks = []
    for i in range(2):
        normal = components[i]
        offset = -np.dot(normal, mean)
        threshold = 1e-6
        tasks.append(Task(
            normal=normal,
            offset=offset,
            threshold=threshold,
            double=True
        ))
    
    return tasks

def compute_tasks(cell: Cell):
    components, mean = compute_pca(cell)
    tasks = compute_tasks_from_pca(components, mean)
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
        np.sum([
            np.sum([len(surface.vertices) if surface.closed else len(surface.vertices)-2 for surface in subcell.surfaces])
            for subcell in cell.subcells
        ])
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
                        neuron_idx: int,
                        mlp: ReluMLP=None,
                        vis=False):
    solution = None
    A = np.empty((0, hidden_dim + 1))
    y = np.empty(0)
    is_double = None
    
    sort = True
    if sort:
        indices = get_largest_cells(cells)
        cells = [cells[i] for i in indices]
    for cell_idx, cell in enumerate(cells):
        if len(cell.tasks)==0:
            compute_tasks(cell)
        
        success = False
        for i, task in enumerate(cell.tasks):
            # Skip if we've already committed to double/single and this task doesn't match
            if is_double is not None and task.double != is_double:
                continue
                
            success, A, y, new_solution = add_hyperplane_constraint(
                cell.weight, cell.bias, A, y, task.normal, task.offset, hidden_dim, cell_idx, task.threshold
            )
            if success:
                # Set the mode based on the first successful task
                if is_double is None:
                    is_double = task.double
                # Delete the task that was successfully used
                del cell.tasks[i]
                break
        
        if success:
            if vis and mlp is not None:
                # Check if we have enough neurons to visualize
                layer_size = mlp.layers[layer_idx].weight.data.shape[0]
                can_visualize = True
                
                if is_double and neuron_idx + 1 >= layer_size:
                    can_visualize = False
                elif not is_double and neuron_idx >= layer_size:
                    can_visualize = False
                
                if can_visualize:
                    if is_double:
                        mlp.layers[layer_idx].weight.data[neuron_idx, :] = torch.tensor(new_solution[:hidden_dim], dtype=torch.float32)
                        mlp.layers[layer_idx].bias.data[neuron_idx] = torch.tensor(new_solution[hidden_dim], dtype=torch.float32)
                        mlp.layers[layer_idx].weight.data[neuron_idx+1, :] = torch.tensor(-new_solution[:hidden_dim], dtype=torch.float32)
                        mlp.layers[layer_idx].bias.data[neuron_idx+1] = torch.tensor(-new_solution[hidden_dim], dtype=torch.float32)
                    else:
                        mlp.layers[layer_idx].weight.data[neuron_idx, :] = torch.tensor(new_solution[:hidden_dim], dtype=torch.float32)
                        mlp.layers[layer_idx].bias.data[neuron_idx] = torch.tensor(new_solution[hidden_dim], dtype=torch.float32)
                    plot_cell_sdf(mlp, cells=cells, title=f"Neuron for {cell_idx}: Successfully added", highlight_idx=cell_idx)
            solution = new_solution
        elif vis:
            plot_cell_sdf(mlp, cells=cells, title=f"Neuron for {cell_idx}: Failed to add", highlight_idx=cell_idx)
    
    if solution is None:
        return None, None, None
    
    neuron_weight = solution[:hidden_dim]
    neuron_bias = solution[hidden_dim]
    return neuron_weight, neuron_bias, is_double


def apply_layer_splits(cells: List[Cell],
                       neuron_weight: np.ndarray,
                       neuron_bias: float,
                       neuron_idx: int,
                       is_double: bool):
    new_cells = []
    
    for cell in cells:
        new_cells += split_cell(cell, neuron_weight, neuron_bias, neuron_idx, is_double)
    return new_cells


def collapse_layers(cells: List[Cell],
                    layer_weight: torch.Tensor,
                    layer_bias: torch.Tensor,
                    next_layer_dim: int):
    
    
    
    new_cells = []
    next_weight = np.array(layer_weight)
    next_bias = np.array(layer_bias)
    
    for cell in cells:
        for subcell in cell.subcells:
            composed_weight = next_weight @ cell.weight
            composed_bias = next_weight @ cell.bias + next_bias

            composed_weight[subcell.activations == 0] = 0
            composed_bias[composed_bias == 0] = 0
            
            new_subcell = SubCell(subcell.surfaces, np.zeros(next_layer_dim))
            new_cells.append(Cell([new_subcell], composed_weight, composed_bias, []))
    
    return new_cells


def solve_output_layer_analytically(mlp: ReluMLP,
                                   cells: List[Cell]):
    A_rows = []
    y_targets = []
    
    # For each cell, iterate through subcells and surfaces
    for cell in cells:
        weight = cell.weight
        bias = cell.bias
        
        for subcell in cell.subcells:
            for surface in subcell.surfaces:
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
    
    neuron_idx = 0
    
    while neuron_idx < mlp.hidden_dim:
        hidden_dim = mlp.hidden_dim if layer_idx > 0 else mlp.input_dim
        neuron_weight, neuron_bias, is_double = solve_layer_weights(
            cells, hidden_dim, layer_idx, neuron_idx, mlp, vis
        )
        
        if neuron_weight is None:
            break
        
        # Check if we have enough neurons for double mode
        if is_double and neuron_idx + 1 >= mlp.hidden_dim:
            print(f"Warning: Not enough neurons for double mode at neuron_idx={neuron_idx}, skipping")
            break
        
        # Apply the splits to all cells
        cells = apply_layer_splits(cells, neuron_weight, neuron_bias, neuron_idx, is_double)
        
        # Set neuron weights
        if is_double:
            mlp.layers[layer_idx].weight.data[neuron_idx, :] = torch.tensor(neuron_weight, dtype=torch.float32)
            mlp.layers[layer_idx].bias.data[neuron_idx] = torch.tensor(neuron_bias, dtype=torch.float32)
            mlp.layers[layer_idx].weight.data[neuron_idx+1, :] = torch.tensor(-neuron_weight, dtype=torch.float32)
            mlp.layers[layer_idx].bias.data[neuron_idx+1] = torch.tensor(-neuron_bias, dtype=torch.float32)
            neuron_idx += 2
        else:
            mlp.layers[layer_idx].weight.data[neuron_idx, :] = torch.tensor(neuron_weight, dtype=torch.float32)
            mlp.layers[layer_idx].bias.data[neuron_idx] = torch.tensor(neuron_bias, dtype=torch.float32)
            neuron_idx += 1
        
        if vis:
            plot_cell_sdf(mlp, cells=cells, title=f"Layer {layer_idx + 1}, Neuron {'pair ' if is_double else ''}{neuron_idx}: Applied splits")
    
    return cells
