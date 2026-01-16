from typing import Dict, List, Tuple
import jax.numpy as jnp
import jax
import numpy as np
from marching.activation import ReluActivation
import marching.arch as arch
from functools import partial
from tqdm import tqdm
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from marching.cell import Cell
from marching.check import make_check
from marching.range import AffineContext, make_range
from marching.split import make_split
from marching.collapse import make_collapse
from marching.surface import make_extract_polygon

class LayerStatistics:
    def __init__(self):
        self.total_checks = 0
        self.total_splits = 0
        self.splits_two_cells = 0  # splits that created both inside and outside cells
        self.splits_inside_only = 0  # splits that only created inside cell
        self.splits_outside_only = 0  # splits that only created outside cell
        self.final_cell_count = 0
        self.average_breakpoints = 0.0
        self.filtered_upper = 0  # cells filtered where upper < 0
        self.filtered_lower = 0  # cells filtered where lower > 0
        # Histograms for vertex and edge counts
        self.vertex_count_histogram = defaultdict(int)  # key: vertex count, value: frequency
        self.edge_count_histogram = defaultdict(int)    # key: edge count, value: frequency
    
    def add_cell_to_histogram(self, cell):
        """Add cell's vertex and edge counts to histograms."""
        self.vertex_count_histogram[int(cell.vertex_count)] += 1
        self.edge_count_histogram[int(cell.edge_count)] += 1
    
    def get_histogram_stats(self):
        """Convert histograms to a format suitable for JSON serialization."""
        vertex_hist = dict(sorted(self.vertex_count_histogram.items()))
        edge_hist = dict(sorted(self.edge_count_histogram.items()))
        
        # Calculate some basic statistics
        if vertex_hist:
            vertex_counts = list(vertex_hist.keys())
            vertex_stats = {
                "min": min(vertex_counts),
                "max": max(vertex_counts),
                "most_common": max(vertex_hist.items(), key=lambda x: x[1])[0],
                "histogram": {str(k): int(v) for k, v in vertex_hist.items()}
            }
        else:
            vertex_stats = {"histogram": {}}
            
        if edge_hist:
            edge_counts = list(edge_hist.keys())
            edge_stats = {
                "min": min(edge_counts),
                "max": max(edge_counts),
                "most_common": max(edge_hist.items(), key=lambda x: x[1])[0],
                "histogram": {str(k): int(v) for k, v in edge_hist.items()}
            }
        else:
            edge_stats = {"histogram": {}}
            
        return {
            "vertex_statistics": vertex_stats,
            "edge_statistics": edge_stats
        }

class ExperimentStatistics:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.timestamp = datetime.now().isoformat()
        self.total_checks = 0
        self.total_splits = 0
        self.total_splits_two_cells = 0
        self.total_splits_inside_only = 0
        self.total_splits_outside_only = 0
        self.final_total_cells = 0
        self.layer_stats = []
        self.total_polygons = 0
        self.execution_time = 0.0
        self.total_filtered_upper = 0
        self.total_filtered_lower = 0
        # Combined histograms across all layers
        self.total_vertex_count_histogram = defaultdict(int)
        self.total_edge_count_histogram = defaultdict(int)
        # Keep track of final layer cells for accurate surface percentage
        self.final_layer_cell_count = 0

    def add_layer_stats(self, layer_idx: int, stats: LayerStatistics, is_final_layer=False):
        self.total_checks += stats.total_checks
        self.total_splits += stats.total_splits
        self.total_splits_two_cells += stats.splits_two_cells
        self.total_splits_inside_only += stats.splits_inside_only
        self.total_splits_outside_only += stats.splits_outside_only
        self.final_total_cells += stats.final_cell_count
        self.total_filtered_upper += stats.filtered_upper
        self.total_filtered_lower += stats.filtered_lower
        
        if is_final_layer:
            self.final_layer_cell_count = stats.final_cell_count
        
        # Combine histograms
        for count, freq in stats.vertex_count_histogram.items():
            self.total_vertex_count_histogram[count] += freq
        for count, freq in stats.edge_count_histogram.items():
            self.total_edge_count_histogram[count] += freq
        
        layer_dict = {
            "layer_index": layer_idx,
            "checks": int(stats.total_checks),
            "splits": int(stats.total_splits),
            "splits_two_cells": int(stats.splits_two_cells),
            "splits_inside_only": int(stats.splits_inside_only),
            "splits_outside_only": int(stats.splits_outside_only),
            "final_cell_count": int(stats.final_cell_count),
            "average_breakpoints": float(stats.average_breakpoints),
            "filtered_upper": int(stats.filtered_upper),
            "filtered_lower": int(stats.filtered_lower)
        }
        
        # Add percentages if we have splits
        if stats.total_splits > 0:
            layer_dict.update({
                "percent_splits_two_cells": (stats.splits_two_cells / stats.total_splits) * 100,
                "percent_splits_single_cell": ((stats.splits_inside_only + stats.splits_outside_only) / stats.total_splits) * 100
            })
        
        # Add histogram statistics
        layer_dict.update(stats.get_histogram_stats())
        
        self.layer_stats.append(layer_dict)

    def get_total_histogram_stats(self):
        """Get histogram statistics for the entire experiment."""
        vertex_hist = dict(sorted(self.total_vertex_count_histogram.items()))
        edge_hist = dict(sorted(self.total_edge_count_histogram.items()))
        
        if vertex_hist:
            vertex_counts = list(vertex_hist.keys())
            vertex_stats = {
                "min": min(vertex_counts),
                "max": max(vertex_counts),
                "most_common": max(vertex_hist.items(), key=lambda x: x[1])[0],
                "histogram": {str(k): int(v) for k, v in vertex_hist.items()}
            }
        else:
            vertex_stats = {"histogram": {}}
            
        if edge_hist:
            edge_counts = list(edge_hist.keys())
            edge_stats = {
                "min": min(edge_counts),
                "max": max(edge_counts),
                "most_common": max(edge_hist.items(), key=lambda x: x[1])[0],
                "histogram": {str(k): int(v) for k, v in edge_hist.items()}
            }
        else:
            edge_stats = {"histogram": {}}
            
        return {
            "vertex_statistics": vertex_stats,
            "edge_statistics": edge_stats
        }

    def save(self):
        # Calculate percentages for total statistics
        total_stats = {
            "total_checks": int(self.total_checks),
            "total_splits": int(self.total_splits),
            "total_splits_two_cells": int(self.total_splits_two_cells),
            "total_splits_inside_only": int(self.total_splits_inside_only),
            "total_splits_outside_only": int(self.total_splits_outside_only),
            "final_total_cells": int(self.final_total_cells),
            "total_polygons": int(self.total_polygons),
            "total_filtered_upper": int(self.total_filtered_upper),
            "total_filtered_lower": int(self.total_filtered_lower)
        }

        # Add derived statistics
        if self.total_splits > 0:
            total_stats.update({
                "percent_splits_two_cells": (self.total_splits_two_cells / self.total_splits) * 100,
                "percent_splits_single_cell": ((self.total_splits_inside_only + self.total_splits_outside_only) / self.total_splits) * 100
            })
        
        if self.final_layer_cell_count > 0:
            total_stats["percent_cells_with_surface"] = (self.total_polygons / self.final_layer_cell_count) * 100
        
        # Add histogram statistics
        total_stats.update(self.get_total_histogram_stats())

        stats_dict = {
            "input_path": self.input_path,
            "timestamp": self.timestamp,
            "execution_time_seconds": float(self.execution_time),
            "total_statistics": total_stats,
            "layer_statistics": self.layer_stats
        }

        # Create experiments directory if it doesn't exist
        experiments_dir = Path("experiments")
        experiments_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = experiments_dir / f"stats_{timestamp_str}.json"
        
        with open(filename, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Saved statistics to {filename}")

class SplitValidationError(Exception):
    """Exception raised for errors in the cell splitting process."""
    pass

class CheckValidationError(Exception):
    """Exception raised for errors in the cell checking process."""
    pass

def load(filename):
    out_params = {}
    param_count = 0
    with np.load(filename) as data:
        for key,val in data.items():
            # print(f"mlp layer key: {key}")
            # convert numpy to jax arrays
            if isinstance(val, np.ndarray):
                param_count += val.size
                val = jnp.array(val)
            out_params[key] = val
    print(f"Loaded MLP with {param_count} params")
    return out_params

def parse_ops(params: Dict[str, jnp.ndarray], op_names: List[str]) -> List[Tuple[jnp.ndarray, jnp.ndarray, str]]:
    """Get the pipeline operations from the parameters."""
    ops = []
    num_layers = len(params)//3
    for i in range(num_layers):
        num = i*2
        A = params[f'{num:04d}.dense.A']
        b = params[f'{num:04d}.dense.b']
        op = None
        for op_name in op_names:
            if f'{num+1:04d}.{op_name}._' in params:
                op = op_name
        if op is None:
            raise ValueError(f'Unknown operation: {f"{num+1:04d}.{op}._"}')
        ops.append((A, b, op))
    return ops

def write_obj(filename: str, polygons: List[Tuple[jnp.ndarray, jnp.ndarray]]):
    """Write polygons to an OBJ file.
    
    Args:
        filename: Output OBJ file path
        polygons: List of vertex arrays and counts
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("# OBJ file created by marching neurons\n")
        
        # Write vertices (v)
        vertex_offset = 1  # OBJ indices start at 1
        face_definitions = []
        
        for polygon, count in tqdm(polygons, total=len(polygons)):
            # Write vertices for this polygon
            for i in range(count):
                v = polygon[i]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Store face definition (will write after all vertices)
            vertices_idx = list(range(vertex_offset, vertex_offset + count))
            face_str = "f " + " ".join(str(idx) for idx in vertices_idx) + "\n"
            face_definitions.append(face_str)
            
            vertex_offset += count

        # Write faces (f)
        for face_def in face_definitions:
            f.write(face_def)

def main():
    start_time = time.time()
    
    # input_path = "data/all_networks/relu_mlp_d3_w128/Stanford_bunny/net.pt"
    input_path = "data/additional_networks/relu_mlp_d4_w32/ABC_00003459/net.pt"

    # Initialize experiment statistics
    experiment_stats = ExperimentStatistics(input_path)

    if input_path.endswith(".npz"):
        params = load(input_path)
        dim = params['0000.dense.A'].shape[0]
        if dim not in [2, 3]:
            params = {k: v.T for k, v in params.items()}
    elif input_path.endswith(".pt"):
        arch_name = input_path.split("/")[-3].split(".")[0]
        params = arch.load_pt(input_path, arch_name)
    else:
        raise ValueError("Invalid input path")

    activations = {
        'relu': ReluActivation(),
    }

    op_names = ['relu', 'squeeze_last']
    ops = parse_ops(params, op_names)
    input_dim = ops[0][0].shape[0]
    layer_width = ops[0][1].shape[0]

    A, b, op_name = ops[0]
    neuron_idx = 0

    is_jit = True

    cell = Cell.create_initial(ops)
    affine_ctx = AffineContext('affine_all')
    range_fn = make_range(ops=ops, activations=activations, ctx=affine_ctx, jit=is_jit)
    split_fn = make_split(jit=is_jit)
    check_fn = make_check(ops=ops, activations=activations, jit=is_jit)
    collapse_fn = make_collapse(ops=ops, activations=activations, jit=is_jit)
    extract_polygon_fn = make_extract_polygon(ops=ops, activations=activations, jit=is_jit)

    if is_jit:
        range_fn = jax.jit(range_fn)
        split_fn = jax.jit(split_fn)
        check_fn = jax.jit(check_fn)
        collapse_fn = jax.jit(collapse_fn)
        extract_polygon_fn = jax.jit(extract_polygon_fn)

    queue: List[Tuple[Cell, jnp.ndarray, jnp.ndarray]] = [(cell, A, b)]
    polygons: List[jnp.ndarray] = []
    # done_cells: List[Tuple[Cell, jnp.ndarray, jnp.ndarray]] = []
    # Initialize statistics for first layer
    layer_stats = LayerStatistics()

    reference_count = jnp.zeros((1024,), jnp.int32)
    weights = jnp.zeros((1024, input_dim, layer_width), jnp.float32)
    biases = jnp.zeros((1024, layer_width), jnp.float32)
    weights = weights.at[0, :, :].set(A)
    biases = biases.at[0, :].set(b)
    reference_count = reference_count.at[0].add(1)

    max_queue_size = 0
    max_polygons = 0
    while queue:
        cell, A, b = queue.pop()
        cell, contains_surface = range_fn(cell, A, b)
        if contains_surface:
            cell, A_i, b_i, send_idx = check_fn(cell, A, b)
            if send_idx == 0:
                # split, send to range 0
                cell1, cell2 = split_fn(cell, A_i, b_i)
                queue.append((cell1, A, b))
                queue.append((cell2, A, b))
            elif send_idx in [1, 3]:
                # split, done, send back 1
                if send_idx == 1:
                    cell, A, b = collapse_fn(cell, A, b)
                    queue.append((cell, A, b))
                else:
                    polygon, polygon_count = extract_polygon_fn(cell, A, b)
                    if polygon_count>0:
                        polygons.append((polygon, polygon_count))
            elif send_idx in [2, 4]:
                # done, send to collapse 2
                cell1, cell2 = split_fn(cell, A_i, b_i)
                if send_idx == 2:
                    cell1, A1, b1 = collapse_fn(cell1, A, b)
                    cell2, A2, b2 = collapse_fn(cell2, A, b)
                    queue.append((cell1, A1, b1))
                    queue.append((cell2, A2, b2))
                else:
                    polygon, polygon_count = extract_polygon_fn(cell1, A1, b1)
                    if polygon_count>0:
                        polygons.append((polygon, polygon_count))
                    polygon, polygon_count = extract_polygon_fn(cell2, A2, b2)
                    if polygon_count>0:
                        polygons.append((polygon, polygon_count))
        if (len(queue) > max_queue_size) or (len(polygons) > max_polygons):
            max_queue_size = len(queue)
            max_polygons = len(polygons)
            print(f"Queue size: {len(queue)}, Polygons: {len(polygons)}")

    write_obj('bunny.obj', polygons)
    print(f"Exported mesh to bunny.obj")

if __name__ == '__main__':
    main()
