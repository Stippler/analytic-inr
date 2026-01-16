from dataclasses import replace
import jax
import jax.numpy as jnp
from typing import Tuple, Dict
from marching.activation import Activation
from marching.cell import Cell
from marching.utils import gather_indices, get_cells, update_cells
from marching.buffers import Buffers

def robust_basis_for_plane(plane_normal):
    """Create robust orthonormal basis for a plane with given normal (n,u,v)."""
    
    # 1) Normalize the normal
    n_len = jnp.linalg.norm(plane_normal)
    n = plane_normal / n_len

    # 2) Select axis with the smallest component in n
    #    so it's guaranteed to be non-parallel to n
    min_idx = jnp.argmin(jnp.abs(n))
    v_tmp = jnp.zeros(3)
    v_tmp = v_tmp.at[min_idx].set(1.0)
    
    # 3) Cross to get a vector in the plane
    u2 = jnp.cross(v_tmp, n)
    u2 = u2 / jnp.linalg.norm(u2)
    
    # 4) Cross again to complete orthonormal triple
    u3 = jnp.cross(n, u2)
    u3 = u3 / jnp.linalg.norm(u3)
    
    return n, u2, u3

def onb_duff(n):
    sign = jnp.sign(jnp.copysign(1.0, n[2]))
    a = -1.0 / (sign + n[2])
    b = n[0] * n[1] * a
    u = jnp.array([1.0 + sign * n[0] * n[0] * a,
                   sign * b,
                   -sign * n[0]])
    v = jnp.array([b,
                   sign + n[1] * n[1] * a,
                   -n[1]])
    return n, u, v   # already unit-length and orthogonal

def arg_sort_ccw(points, valid_points, plane_normal):
    # 1) Compute centroid of valid points
    valid_mask = valid_points
    point_count = jnp.sum(valid_mask)
    points = jnp.where(
        valid_mask[:, None],
        points, jnp.array([0, 0, 0], dtype=points.dtype)
    )
    centroid = jnp.sum(points * valid_mask[:, None], axis=0) / point_count
    
    # 2) Create basis vectors
    # u1, u2, u3 = onb_duff(plane_normal)# robust_basis_for_plane(plane_normal)
    u1, u2, u3 = robust_basis_for_plane(plane_normal)
    
    # Use u2 and u3 as our basis vectors in the plane
    u = u2  # First basis vector in plane
    v = u3  # Second basis vector in plane
    
    # 3) Project points to 2D
    diffs = points - centroid
    x_coords = jnp.einsum('ij,j->i', diffs, u)
    y_coords = jnp.einsum('ij,j->i', diffs, v)
    
    # 4) Calculate angles
    angles = jnp.where(valid_mask, jnp.arctan2(y_coords, x_coords), jnp.inf)
    
    # 5) Sort points by angle
    sort_idx = jnp.argsort(angles)
    
    return sort_idx


def clip_3d(cell: Cell, signs, inside_mask, outside_mask, A_i, b_i):
    vertices = cell.vertices.astype(jnp.float64)
    vertex_count = cell.vertex_count
    edges = cell.edges
    edge_count = cell.edge_count

    max_vertex_count = vertices.shape[0]
    max_edge_count = edges.shape[0]

    # 3) Classify edges
    valid_edge_mask = jnp.arange(max_edge_count, dtype=jnp.int32) < edge_count
    in0 = inside_mask[edges[:, 0]] & valid_edge_mask
    in1 = inside_mask[edges[:, 1]] & valid_edge_mask
    
    out0 = outside_mask[edges[:, 0]] & valid_edge_mask
    out1 = outside_mask[edges[:, 1]] & valid_edge_mask
    
    edge_inside_mask = in0 & in1  # both vertices inside
    edge_cross_in2out_mask = in0 & out1  # first inside, second outside
    edge_cross_out2in_mask = out0 & in1  # first outside, second inside
    edge_cross_mask = edge_cross_in2out_mask | edge_cross_out2in_mask

    # 4) Find intersection points
    cross_edge_vert = vertices[edges] # edges[edge_idx] # shape: (2, 2) 
    cross_edge_vec = cross_edge_vert[:,1] - cross_edge_vert[:,0] # shape: (2, 3)
    denom = jnp.einsum('d,md->m', A_i, cross_edge_vec)  # shape (2,)

    eps = 1e-12
    denom_safe = jnp.where(jnp.abs(denom) < eps, jnp.inf, denom)
    t_raw      = -signs[edges[:,0]] / denom_safe
    # Optionally clamp to [0,1] so the point never leaves the segment
    t_raw      = jnp.clip(t_raw, 0.0, 1.0)

    # t_raw = -signs[edges[:, 0]] / denom # shape (max_edges,)

    inter_points = vertices[edges[:, 0]] + t_raw[:,None] * cross_edge_vec
    intersection_count = jnp.sum(edge_cross_mask).astype(jnp.int32)
    
    # 5) Append ordered intersection points to vertices
    vertex_mask = (jnp.arange(max_vertex_count, dtype=jnp.int32) < vertex_count) & (signs < 0)
    remain_vert_idx = gather_indices(vertex_mask)
    remain_vertex_count = jnp.sum(vertex_mask).astype(jnp.int32)
    
    inter_idx = arg_sort_ccw(inter_points, edge_cross_mask, A_i)
    inter_idx_concat = inter_idx[:max_vertex_count]
    inter_idx_concat = jnp.roll(inter_idx_concat, remain_vertex_count)
    
    new_vertices = jnp.where(
        (jnp.arange(max_vertex_count, dtype=jnp.int32) < remain_vertex_count)[:, None],
        vertices[remain_vert_idx],
        inter_points[inter_idx_concat]
    )
    new_vertex_count = (remain_vertex_count + intersection_count).astype(jnp.int32)

    # 6) Update edges to new vertex indices
    indexof_vertices = jnp.zeros_like(remain_vert_idx)
    tmp = jnp.arange(remain_vert_idx.shape[0], dtype=jnp.int32)
    indexof_vertices = indexof_vertices.at[remain_vert_idx].set(tmp)
    edges = indexof_vertices[edges].astype(jnp.int32)

    # 7) Update edges with intersection points
    indexof_inter_points = jnp.zeros_like(inter_idx)
    tmp = jnp.arange(inter_idx.shape[0], dtype=jnp.int32)
    indexof_inter_points = indexof_inter_points.at[inter_idx].set(tmp)
    indexof_inter_points = (indexof_inter_points+remain_vertex_count).astype(jnp.int32)

    edges = jnp.where(
        edge_cross_in2out_mask[:, None],
        jnp.stack([edges[:, 0], indexof_inter_points], axis=1).astype(jnp.int32),
        edges
    )
    edges = jnp.where(
        edge_cross_out2in_mask[:, None],
        jnp.stack([indexof_inter_points, edges[:, 1]], axis=1).astype(jnp.int32),
        edges
    )

    # 8) Move edges to the front of the array
    edge_mask = edge_cross_mask | edge_inside_mask
    idx = gather_indices(edge_mask)
    edges = edges[idx].astype(jnp.int32)
    edge_count = jnp.sum(edge_mask).astype(jnp.int32)

    # 9) Create new edges
    idx = (jnp.arange(max_edge_count, dtype=jnp.int32) + remain_vertex_count).astype(jnp.int32)
    next_idx = (jnp.arange(max_edge_count, dtype=jnp.int32) + jnp.array(1, dtype=jnp.int32) + remain_vertex_count).astype(jnp.int32)
    next_idx = next_idx.at[intersection_count-1].set(remain_vertex_count)
    new_edges = jnp.stack([idx, next_idx], axis=1).astype(jnp.int32)
    new_edges = new_edges[:max_edge_count]
    new_edges = jnp.roll(new_edges, edge_count, axis=0)

    edges = jnp.where(
        (jnp.arange(max_edge_count) < edge_count)[:, None],
        edges,
        new_edges,
    ).astype(jnp.int32)

    edge_count = (edge_count + intersection_count).astype(jnp.int32)

    new_vertices = jnp.where(
        (jnp.arange(max_vertex_count, dtype=jnp.int32) < new_vertex_count)[:, None],
        new_vertices,
        jnp.zeros_like(new_vertices)
    )

    new_edges = jnp.where(
        (jnp.arange(max_edge_count, dtype=jnp.int32) < edge_count)[:, None],
        edges,
        jnp.zeros_like(edges)
    ).astype(jnp.int32)

    # 10) Return clipped cell
    return replace(
        cell,
        vertices=new_vertices.astype(jnp.float32),
        vertex_count=new_vertex_count,
        edges=edges,
        edge_count=edge_count,
        cell_split_count=(cell.cell_split_count+1).astype(jnp.int32)
    )

def make_split(jit=True):
    def split(cell: Cell, A_i: jnp.ndarray, b_i: jnp.ndarray) -> Tuple[Cell, Cell]:
        # Find the critical neuron index within the current layer's range
        vertices = cell.vertices.astype(jnp.float64)
        vertex_count = cell.vertex_count
        max_vertex_count = vertices.shape[0]
        A_i = A_i.astype(jnp.float64)
        b_i = b_i.astype(jnp.float64)
    
        # 1) Calculate signs for each vertex
        signs = jnp.dot(vertices, A_i) + b_i
    
        # 2) Classify vertices
        valid_vertices_mask = jnp.arange(max_vertex_count, dtype=jnp.int32) < vertex_count
        inside_mask = (signs < 0) & valid_vertices_mask
        outside_mask = (signs > 0) & valid_vertices_mask

        inside_cell = clip_3d(cell, signs, inside_mask, outside_mask, A_i, b_i)
        outside_cell = clip_3d(cell, -signs, outside_mask, inside_mask, -A_i, -b_i)

        return inside_cell, outside_cell
    return split

def v_split_fn(split_fn, cells, A_i_batch, b_i_batch, count):
    """Vectorized version of split function.
    
    Args:
        split_fn: Base split function to vectorize
        cells: Cell batch to process
        A_i_batch: Batch of A_i matrices
        b_i_batch: Batch of b_i vectors
        count: Number of valid cells in batch
        
    Returns:
        Tuple of (inside_cells, outside_cells)
    """
    def short(valid, cell, A_i, b_i):
        return jax.lax.cond(
            valid,
            split_fn,
            lambda cell, A_i, b_i: (cell, cell),
            cell, A_i, b_i
        )
    valid = jnp.arange(A_i_batch.shape[0]) < count
    v_short = jax.vmap(short, in_axes=(0, 0, 0, 0))
    inside_cells, outside_cells = v_short(valid, cells, A_i_batch, b_i_batch)
    return inside_cells, outside_cells

def split_step(buffers: Buffers, v_split, batch_size):
    """Split cells that are ready for splitting.
    
    Args:
        buffers: Current pipeline buffers
        v_split: Vectorized split function
        batch_size: Size of batches to process
        
    Returns:
        Updated buffers
    """
    split_count = buffers.split_count
    split_index_buffer = buffers.split_index_buffer
    A_i_buffer = buffers.A_i_buffer
    b_i_buffer = buffers.b_i_buffer
    cell_references = buffers.cell_references
    weight_references = buffers.weight_references
    cell_buffer_count = buffers.cell_buffer_count
    range_count = buffers.range_count
    range_index_buffer = buffers.range_index_buffer
    cell_buffer = buffers.cell_buffer
    cell_weight_map = buffers.cell_weight_map

    cells, A_batch, b_batch, cell_count, split_count, buffer_indices, weight_indices = \
        get_cells(buffers, split_index_buffer, split_count, batch_size)
    
    idx = jnp.arange(batch_size, dtype=jnp.int32)+split_count
    A_i = A_i_buffer[idx]
    b_i = b_i_buffer[idx]
    inside_cells, outside_cells = v_split(
        cells,
        A_i,
        b_i,
        cell_count
    )
    
    # Update inside cells in place
    cell_buffer = update_cells(cell_buffer, inside_cells, buffer_indices)
    
    # Find slots for outside cells
    free_cell_slots = jnp.argwhere(
        cell_references[:-1] == 0,
        size=batch_size,
        fill_value=-1
    )[:, 0].astype(jnp.int32)

    idx = jnp.arange(batch_size, dtype=jnp.int32)
    valid_mask = idx<cell_count

    free_cell_slots = jnp.where(valid_mask, free_cell_slots, cell_references.shape[0]-1)
    cell_references = cell_references.at[free_cell_slots].set(1)
    
    increment = jnp.bincount(
        weight_indices,
        weights=valid_mask.astype(jnp.int32),
        length=weight_references.shape[0]
    )
    weight_references = weight_references + increment
    
    # Store outside cells
    cell_buffer = jax.tree.map(
        lambda x, y: x.at[free_cell_slots].set(y),
        cell_buffer, outside_cells
    )
    
    cell_weight_map = cell_weight_map.at[free_cell_slots].set(weight_indices)
    
    # Add both to range queue
    combined_indices = jnp.concatenate([buffer_indices, free_cell_slots])
    combined_valid_mask = jnp.concatenate([valid_mask, valid_mask])
    gather_idx = gather_indices(combined_valid_mask)
    combined_indices = combined_indices[gather_idx]
    combined_valid_mask = combined_valid_mask[gather_idx]
    
    range_idx = jnp.arange(batch_size * 2) + range_count
    range_index_buffer = range_index_buffer.at[range_idx].set(combined_indices)
    range_count = range_count + cell_count * 2
    cell_buffer_count = cell_buffer_count + cell_count

    buffers = replace(
        buffers,
        cell_buffer=cell_buffer,
        range_count=range_count,
        range_index_buffer=range_index_buffer,
        cell_weight_map=cell_weight_map,
        split_count=split_count,
        split_index_buffer=split_index_buffer,
        A_i_buffer=A_i_buffer,
        b_i_buffer=b_i_buffer,
        weight_references=weight_references,
        cell_references=cell_references,
        cell_buffer_count=cell_buffer_count
    )
    return buffers
