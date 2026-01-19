from dataclasses import replace
from functools import partial
from typing import Dict, List, Tuple
import jax.numpy as jnp
import jax
import numpy as np
from marching.activation import ReluActivation, SinActivation
import marching.arch as arch
from tqdm import tqdm
import time

from marching.cell import Cell
from marching.check import make_check, v_check_fn, check_step
from marching.range import AffineContext, make_range, v_range_fn, range_step
from marching.split import make_split, v_split_fn, split_step
from marching.collapse import make_collapse, v_collapse_fn, collapse_step
from marching.surface import make_extract_triangles, v_extract_triangles_fn, make_extract_polygon, v_extract_polygons_fn, extract_step
from marching.utils import gather_indices, get_cells, update_cells
from marching.buffers import Buffers

from typing import Optional

jax.config.update("jax_enable_x64", True)

def make_runner(params: Dict[str, jnp.ndarray], batch_size: int = 1024, use_polygons: bool = False):
    """Create a runner function for the marching neurons algorithm.
    
    Args:
        params: Dictionary of network parameters
        batch_size: Size of batches to process
        use_polygons: Whether to use polygon buffers instead of triangle buffers
        
    Returns:
        A JIT-compiled function that runs the algorithm
    """
    # Setup activations and operations
    activations = {
        'relu': ReluActivation(),
        'sin': SinActivation(),
    }
    op_names = ['relu', 'squeeze_last', 'sin']
    ops = parse_ops(params, op_names)

    is_jit = True
    affine_ctx = AffineContext('affine_all')
    
    # Configure buffer sizes
    cell_buffer_size = batch_size * 100
    weight_buffer_size = cell_buffer_size // 2
    triangle_buffer_size = 30_000_000
    
    # All index buffers are sized to cell_buffer_size
    check_index_buffer_size = cell_buffer_size
    range_index_buffer_size = cell_buffer_size
    split_index_buffer_size = cell_buffer_size
    collapse_index_buffer_size = cell_buffer_size
    extract_index_buffer_size = cell_buffer_size

    # Create base functions
    range_fn = make_range(ops=ops, activations=activations, ctx=affine_ctx, jit=is_jit)
    split_fn = make_split(jit=is_jit)
    check_fn = make_check(ops=ops, activations=activations, jit=is_jit)
    collapse_fn = make_collapse(ops=ops, activations=activations, jit=is_jit)
    
    # Create extraction function based on mode
    if use_polygons:
        extract_fn = make_extract_polygon(ops=ops, activations=activations, jit=is_jit)
    else:
        extract_fn = make_extract_triangles(ops=ops, activations=activations, jit=is_jit)
    
    # Create vectorized functions with partial application
    v_range = jax.jit(partial(v_range_fn, range_fn))
    v_split = jax.jit(partial(v_split_fn, split_fn))
    v_check = jax.jit(partial(v_check_fn, check_fn))
    v_collapse = jax.jit(partial(v_collapse_fn, collapse_fn))
    
    # Create vectorized extraction function based on mode
    if use_polygons:
        v_extract = jax.jit(partial(v_extract_polygons_fn, extract_fn))
    else:
        v_extract = jax.jit(partial(v_extract_triangles_fn, extract_fn))
    
    # Create JIT compiled step functions
    extract_step_jit = jax.jit(partial(extract_step, v_extract=v_extract, batch_size=batch_size, use_polygons=use_polygons))
    collapse_step_jit = jax.jit(partial(collapse_step, v_collapse=v_collapse, batch_size=batch_size))
    split_step_jit = jax.jit(partial(split_step, v_split=v_split, batch_size=batch_size))
    range_step_jit = jax.jit(partial(range_step, v_range=v_range, batch_size=batch_size))
    check_step_jit = jax.jit(partial(check_step, v_check=v_check, batch_size=batch_size))


    def run(count: jnp.ndarray = jnp.array(1, dtype=jnp.int32), jit: bool = True):
        """Run the marching neurons algorithm.
        
        Args:
            count: Initial cell buffer count (default 1)
            jit: Whether to use JIT compilation
            
        Returns:
            For triangles: Tuple of (triangle_buffer, triangle_buffer_count)
            For polygons: Tuple of (vertex_buffer, vertex_buffer_count, face_vertex_count_buffer, face_buffer_count)
        """
        
        # Create buffers inside the function
        cell = Cell.create_initial(ops)
        buffers: Buffers = Buffers.create(
            proto_cell=cell,
            ops=ops,
            cell_buffer_size=cell_buffer_size,
            weight_buffer_size=weight_buffer_size,
            range_index_buffer_size=range_index_buffer_size,
            check_index_buffer_size=check_index_buffer_size,
            split_index_buffer_size=split_index_buffer_size,
            collapse_index_buffer_size=collapse_index_buffer_size,
            extract_index_buffer_size=extract_index_buffer_size,
            triangle_buffer_size=triangle_buffer_size,
            use_polygon_buffer=use_polygons,
        )
        buffers = replace(buffers, cell_buffer_count=count)
        
        # Create pipeline steps
        pipeline_step = (
            extract_step_jit,
            collapse_step_jit,
            split_step_jit,
            range_step_jit,
            check_step_jit
        )

        debug = False
        
        if jit:
            def cond(buffers: Buffers):
                continue_condition = (buffers.cell_buffer_count > 0)
                if use_polygons:
                    # Use vertex buffer count for polygons
                    output_condition = (buffers.vertex_buffer_count < triangle_buffer_size-(batch_size+1))
                else:
                    # Use triangle buffer count for triangles
                    output_condition = (buffers.triangle_buffer_count < triangle_buffer_size-(batch_size+1))
                cell_condition = (buffers.cell_buffer_count < cell_buffer_size-(batch_size+1))
                weight_condition = (jnp.sum(buffers.weight_references!=0) < weight_buffer_size-(batch_size+1))
                return continue_condition & output_condition & cell_condition & weight_condition

            def body(buffers: Buffers):
                counts = jnp.array([
                    buffers.extract_count,
                    buffers.collapse_count,
                    buffers.split_count,
                    buffers.range_count,
                    buffers.check_count
                ])
                idx = jnp.argmax(counts)
                buffers = jax.lax.switch(idx, pipeline_step, buffers)
                
                if debug:
                    jax.debug.print('--------------------------------')
                    if use_polygons:
                        jax.debug.print('{idx}, {cell_buffer_count}, {vertex_buffer_count}, {max_cell_split_count}', 
                                      idx=idx, 
                                      cell_buffer_count=buffers.cell_buffer_count, 
                                      vertex_buffer_count=buffers.vertex_buffer_count, 
                                      max_cell_split_count=buffers.max_cell_split_count)
                    else:
                        jax.debug.print('{idx}, {cell_buffer_count}, {triangle_buffer_count}, {max_cell_split_count}', 
                                      idx=idx, 
                                      cell_buffer_count=buffers.cell_buffer_count, 
                                      triangle_buffer_count=buffers.triangle_buffer_count, 
                                      max_cell_split_count=buffers.max_cell_split_count)
                    jax.debug.print('{counts}', counts=counts)

                    buffers = replace(buffers,
                        max_vertex_count=jnp.maximum(jnp.max(buffers.cell_buffer.vertex_count), buffers.max_vertex_count).astype(jnp.int32),
                        max_edge_count=jnp.maximum(jnp.max(buffers.cell_buffer.edge_count), buffers.max_edge_count).astype(jnp.int32),
                        max_cell_buffer_count=jnp.maximum(buffers.cell_buffer_count, buffers.max_cell_buffer_count).astype(jnp.int32),
                        max_weight_buffer_count=jnp.maximum(jnp.sum(buffers.weight_references!=0), buffers.max_weight_buffer_count).astype(jnp.int32),
                        max_range_index_buffer_count=jnp.maximum(buffers.range_count, buffers.max_range_index_buffer_count).astype(jnp.int32),
                        max_check_index_buffer_count=jnp.maximum(buffers.check_count, buffers.max_check_index_buffer_count).astype(jnp.int32),
                        max_split_index_buffer_count=jnp.maximum(buffers.split_count, buffers.max_split_index_buffer_count).astype(jnp.int32),
                        max_collapse_index_buffer_count=jnp.maximum(buffers.collapse_count, buffers.max_collapse_index_buffer_count).astype(jnp.int32),
                        max_extract_index_buffer_count=jnp.maximum(buffers.extract_count, buffers.max_extract_index_buffer_count).astype(jnp.int32),
                        total_iterations=buffers.total_iterations+1,
                        max_cell_split_count=jnp.maximum(jnp.max(buffers.cell_buffer.cell_split_count).astype(jnp.int32), buffers.max_cell_split_count).astype(jnp.int32)
                    )
                return buffers

            buffers = jax.lax.while_loop(cond, body, buffers)

            if debug:
                jax.debug.print('Max vertex count: {max_vertex_count}, Max edge count: {max_edge_count}, Max polygon vertices: {max_polygon_vertices}',
                                max_vertex_count=buffers.max_vertex_count, 
                                max_edge_count=buffers.max_edge_count, 
                                max_polygon_vertices=buffers.max_polygon_vertices)
                jax.debug.print('Max cell buffer count: {max_cell_buffer_count}, Max weight buffer count: {max_weight_buffer_count}',
                                max_cell_buffer_count=buffers.max_cell_buffer_count, 
                                max_weight_buffer_count=buffers.max_weight_buffer_count)
                jax.debug.print('Max range index buffer count: {max_range_index_buffer_count}, Max check index buffer count: {max_check_index_buffer_count}',
                                max_range_index_buffer_count=buffers.max_range_index_buffer_count, 
                                max_check_index_buffer_count=buffers.max_check_index_buffer_count)
                jax.debug.print('Max split index buffer count: {max_split_index_buffer_count}, Max collapse index buffer count: {max_collapse_index_buffer_count}',
                                max_split_index_buffer_count=buffers.max_split_index_buffer_count, 
                                max_collapse_index_buffer_count=buffers.max_collapse_index_buffer_count)
                jax.debug.print('Max extract index buffer count: {max_extract_index_buffer_count}, Total iterations: {total_iterations}',
                                max_extract_index_buffer_count=buffers.max_extract_index_buffer_count, 
                                total_iterations=buffers.total_iterations)
                jax.debug.print('Max cell split count: {max_cell_split_count}', 
                              max_cell_split_count=buffers.max_cell_split_count)
        else:
            times = {}
            while buffers.cell_buffer_count > 0:
                counts = jnp.array([
                    buffers.extract_count,
                    buffers.collapse_count,
                    buffers.split_count,
                    buffers.range_count,
                    buffers.check_count
                ])
                idx = jnp.argmax(counts)
                print('--------------------------------')
                print(f"Stage: {pipeline_step[idx].__name__}")
                start_time = time.time()
                buffers = pipeline_step[idx](buffers)
                end_time = time.time()
                times[pipeline_step[idx].__name__] = end_time - start_time
                
                # ---- progress print -------------------------------------------------
                print(f"Range: {int(buffers.range_count)}, Check: {int(buffers.check_count)}, Split: {int(buffers.split_count)}, Collapse: {int(buffers.collapse_count)}, Extract: {int(buffers.extract_count)}")
                print(f"Weight References: {int(jnp.sum(buffers.weight_references[:-1]))}, Cell References: {int(jnp.sum(buffers.cell_references[:-1]))} Allocated Weights: {int(jnp.sum(buffers.weight_references[:-1]!=0))}")
                if use_polygons:
                    print(f"Cells in buffer: {int(buffers.cell_buffer_count)}, vertices: {int(buffers.vertex_buffer_count)}, faces: {int(buffers.face_buffer_count)}")
                else:
                    print(f"Cells in buffer: {int(buffers.cell_buffer_count)}, triangles: {int(buffers.triangle_buffer_count)}")
                assert int(buffers.cell_buffer_count) == int(jnp.sum(buffers.cell_references[:-1]))
                assert int(jnp.sum(buffers.weight_references[:-1])) == int(jnp.sum(buffers.cell_references[:-1]))
                assert (buffers.range_count + buffers.check_count
                        + buffers.split_count + buffers.collapse_count
                        + buffers.extract_count) == buffers.cell_buffer_count

        if use_polygons:
            return (buffers.vertex_buffer, buffers.vertex_buffer_count, 
                   buffers.face_vertex_count_buffer, buffers.face_buffer_count)
        else:
            return buffers.triangle_buffer, buffers.triangle_buffer_count
    
    # JIT compile the entire run function including buffer creation
    run_jit = jax.jit(partial(run, jit=True))
    
    # Compile with a warmup call
    print("Compiling runner...")
    tik = time.time()
    _ = run_jit(jnp.array(0, dtype=jnp.int32))
    tok = time.time()
    print(f"Runner compilation time: {tok - tik} seconds")
    
    return run_jit

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

def write_obj(filename: str, triangle_buffer: jnp.ndarray, triangle_count:jnp.ndarray):
    """Write triangles to an OBJ file.
    
    Args:
        filename: Output OBJ file path
        triangles: List of vertex arrays and counts
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("# OBJ file created by marching neurons\n")
        
        # Write vertices (v)
        vertex_offset = 1  # OBJ indices start at 1
        face_definitions = []
        
        for i in tqdm(range(triangle_count), total=triangle_count):
            # Write vertices for this triangle
            for j in range(3):
                v = triangle_buffer[i, j]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Store face definition (will write after all vertices)
            vertices_idx = list(range(vertex_offset, vertex_offset + 3))
            face_str = "f " + " ".join(str(idx) for idx in vertices_idx) + "\n"
            face_definitions.append(face_str)
            
            vertex_offset += 3

        # Write faces (f)
        for face_def in face_definitions:
            f.write(face_def)

import trimesh

def export_triangles_as_ply(triangles, triangle_count, output_file):
    """
    Export triangles to PLY format using trimesh library.
    
    Args:
        triangles: array of shape (N, 3, 3) containing N triangles, each with 3 vertices of 3 coordinates
        triangle_count: number of valid triangles
        output_file: path to output PLY file
    """

    if triangle_count == 0:
        print("Warning: No triangles to export!")
        return
    
    # Convert JAX arrays to NumPy
    triangles_np = np.array(triangles)
    vertices = triangles_np[:triangle_count].reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    
    # Create mesh and export
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(output_file)

def export_polygons_as_ply(vertex_buffer, vertex_count, face_buffer, face_sizes, face_count, output_file):
    """
    Export polygons to PLY format using custom binary PLY writer for n-gons.
    
    Args:
        vertex_buffer: array of vertices (flattened, with valid vertices first)
        vertex_count: number of valid vertices
        face_buffer: array of face vertex indices (can be None for new structure)
        face_sizes: array of face sizes (vertices per face)
        face_count: number of valid faces
        output_file: path to output PLY file
    """
    if face_count == 0:
        print("Warning: No polygons to export!")
        return
    
    # Convert JAX arrays to NumPy
    vertices_np = np.array(vertex_buffer[:vertex_count])
    face_sizes_np = np.array(face_sizes[:face_count])
    
    # Build face list
    faces_list = []
    vertex_offset = 0
    
    if face_buffer is not None:
        # Old structure: use face_buffer
        face_buffer_np = np.array(face_buffer[:face_count])
        for i in range(face_count):
            face_size = face_sizes_np[i]
            if face_size >= 3:  # Valid face
                face_indices = face_buffer_np[i, :face_size]
                faces_list.append(face_indices.tolist())
    else:
        # New structure: vertices are consecutive, reconstruct faces
        for i in range(face_count):
            face_size = face_sizes_np[i]
            if face_size >= 3:  # Valid face
                face_indices = np.arange(vertex_offset, vertex_offset + face_size)
                faces_list.append(face_indices.tolist())
            vertex_offset += face_size
    
    # Export using custom PLY writer for n-gons
    if faces_list:
        write_ply_ngon_binary_le(output_file, vertices_np, faces_list)
    else:
        print("Warning: No valid faces to export!")

def write_ply_ngon_binary_le(path, vertices, faces_list):
    """Write PLY file with n-gon faces in binary little-endian format."""
    import struct
    vertices = np.asarray(vertices, dtype=np.float32)
    
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces_list)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    
    with open(path, "wb") as f:
        f.write(header)
        vertices.tofile(f)
        for face in faces_list:
            f.write(struct.pack("<B", len(face)))  # uchar for face size
            np.asarray(face, dtype=np.int32).tofile(f)

def export_polygons_as_obj(vertex_buffer, vertex_count, face_buffer, face_sizes, face_count, output_file):
    """
    Export polygons to OBJ format.
    
    Args:
        vertex_buffer: array of vertices
        vertex_count: number of valid vertices
        face_buffer: array of face vertex indices
        face_sizes: array of face sizes (vertices per face)
        face_count: number of valid faces
        output_file: path to output OBJ file
    """
    if face_count == 0:
        print("Warning: No polygons to export!")
        return
    
    # Convert JAX arrays to NumPy
    vertices_np = np.array(vertex_buffer[:vertex_count])
    face_buffer_np = np.array(face_buffer[:face_count])
    face_sizes_np = np.array(face_sizes[:face_count])
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("# OBJ file created by marching neurons (polygons)\n")
        
        # Write vertices
        for v in vertices_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ indices start at 1)
        for i in range(face_count):
            face_size = face_sizes_np[i]
            if face_size >= 3:  # Valid face
                face_indices = face_buffer_np[i, :face_size]
                # Convert to 1-based indexing for OBJ
                face_str = "f " + " ".join(str(idx + 1) for idx in face_indices) + "\n"
                f.write(face_str)

def load_network(input_path: str):
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
    return params