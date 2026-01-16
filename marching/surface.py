from dataclasses import replace
from typing import Callable, Dict, List, Tuple
import jax
import jax.numpy as jnp

from marching.activation import Activation
from marching.cell import Cell
from marching.split import arg_sort_ccw
from marching.utils import gather_indices, get_cells, update_cells
from marching.buffers import Buffers

def make_extract_polygon(ops: List[Tuple[jnp.ndarray, jnp.ndarray, str]], activations: Dict[str, Activation], jit=True):
    
    A_next, b_next, op_name = ops[-1]
    op_name = ops[-2][2]

    def collapse_fn(cell: Cell, A: jnp.ndarray, b: jnp.ndarray):
        A_scaled, b_scaled = activations[op_name].collapse(A, b, cell.indices[cell.layer_idx])
        A_composed = A_scaled @ A_next
        b_composed = b_scaled @ A_next + b_next
        cell = replace(
            cell,
            layer_idx=cell.layer_idx+1,
        )
        return cell, A_composed, b_composed
    
    def extract_polygon(cell: Cell, A: jnp.ndarray, b: jnp.ndarray):
        cell, A, b = collapse_fn(cell, A, b)
        vertices = cell.vertices.astype(jnp.float64)
        edges = cell.edges
        edge_count = cell.edge_count

        w = A[:, 0]
        b = b[0]

        max_edge_count = edges.shape[0]

        # 1) Calculate signs for each vertex
        signs = vertices @ w + b  # shape (max_edges, 2)

        # 2) Create masks for classification
        valid_edge_mask = jnp.arange(max_edge_count) < edge_count
        inside_mask = (signs[edges] < 0) & valid_edge_mask[:, None]  # shape (max_edges, 2)
        outside_mask = (signs[edges] > 0) & valid_edge_mask[:, None]

        # 3) Classify edges
        edge_cross_in2out_mask = inside_mask[:,0] & outside_mask[:,1] & valid_edge_mask  # first inside, second outside
        edge_cross_out2in_mask = outside_mask[:,0] & inside_mask[:,1] & valid_edge_mask  # first outside, second inside

        edge_cross_mask = edge_cross_in2out_mask | edge_cross_out2in_mask

        # 4) Find intersection points
        cross_edge_vert = vertices[edges] # edges[edge_idx] # shape: (2, 2) 
        cross_edge_vec = cross_edge_vert[:,1] - cross_edge_vert[:,0] # shape: (2, 3)
        denom = jnp.einsum('d,md->m', w, cross_edge_vec)  # shape (2,)
        t_raw = -signs[edges[:,0]] / denom # shape (max_edges,)

        inter_points = vertices[edges[:, 0]] + t_raw[:,None] * cross_edge_vec
        intersection_count = jnp.sum(edge_cross_mask).astype(jnp.int32)

        # 5) Append ordered intersection points to vertices
        inter_idx = arg_sort_ccw(inter_points, edge_cross_mask, w)
        polygon_vertices = inter_points[inter_idx]
        polygon_vertex_counts = intersection_count

        return polygon_vertices.astype(jnp.float32), polygon_vertex_counts

    return extract_polygon

def v_extract_polygons_fn(extract_polygon_fn, cells, A_batch, b_batch, count, eps=1e-6):
    def extract_single(valid, cell, A, b):
        return jax.lax.cond(
            valid,
            extract_polygon_fn,
            lambda cell, A, b: (jnp.zeros((90, 3), jnp.float32), jnp.array(0, jnp.int32)),
            cell, A, b
        )
    
    valid = jnp.arange(A_batch.shape[0]) < count
    v_extract = jax.vmap(extract_single, in_axes=(0, 0, 0, 0))
    
    polygon_vertices, polygon_vertex_counts = v_extract(valid, cells, A_batch, b_batch)
    
    return polygon_vertices, polygon_vertex_counts
    



def make_extract_triangles(ops: List[Tuple[jnp.ndarray, jnp.ndarray, str]], activations: Dict[str, Activation], jit=True):
    
    A_next, b_next, op_name = ops[-1]
    op_name = ops[-2][2]

    def collapse_fn(cell: Cell, A: jnp.ndarray, b: jnp.ndarray):
        A_scaled, b_scaled = activations[op_name].collapse(A.astype(jnp.float64), b.astype(jnp.float64), cell.indices[cell.layer_idx])
        A_composed = jnp.dot(A_scaled.astype(jnp.float64), A_next.astype(jnp.float64))
        b_composed = jnp.dot(b_scaled.astype(jnp.float64), A_next.astype(jnp.float64)) + b_next.astype(jnp.float64)
        cell = replace(
            cell,
            layer_idx=cell.layer_idx+1,
        )
        return cell, A_composed, b_composed
    
    def extract_triangles(cell: Cell, A: jnp.ndarray, b: jnp.ndarray):
        cell, A, b = collapse_fn(cell, A, b)
        vertices = cell.vertices.astype(jnp.float64)
        edges = cell.edges
        edge_count = cell.edge_count

        w = A[:, 0]
        b = b[0]

        max_edge_count = edges.shape[0]

        # 1) Calculate signs for each vertex
        signs = vertices @ w + b  # shape (max_edges, 2)


        # 2) Create masks for classification
        def get_triangles(edges, signs, vertices, w, b):
            valid_edge_mask = jnp.arange(max_edge_count) < edge_count
            inside_mask = (signs[edges] < 0) & valid_edge_mask[:, None]  # shape (max_edges, 2)
            outside_mask = (signs[edges] > 0) & valid_edge_mask[:, None]

            # 3) Classify edges
            edge_cross_in2out_mask = inside_mask[:,0] & outside_mask[:,1] & valid_edge_mask  # first inside, second outside
            edge_cross_out2in_mask = outside_mask[:,0] & inside_mask[:,1] & valid_edge_mask  # first outside, second inside

            edge_cross_mask = edge_cross_in2out_mask | edge_cross_out2in_mask

            # 4) Find intersection points
            cross_edge_vert = vertices[edges] # edges[edge_idx] # shape: (2, 2) 
            cross_edge_vec = cross_edge_vert[:,1] - cross_edge_vert[:,0] # shape: (2, 3)
            denom = jnp.einsum('d,md->m', w, cross_edge_vec)  # shape (2,)
            t_raw = -signs[edges[:,0]] / denom # shape (max_edges,)

            inter_points = vertices[edges[:, 0]] + t_raw[:,None] * cross_edge_vec
            intersection_count = jnp.sum(edge_cross_mask).astype(jnp.int32)

            # 5) Append ordered intersection points to vertices
            inter_idx = arg_sort_ccw(inter_points, edge_cross_mask, w)
            polygon = inter_points[inter_idx]
            polygon_count = intersection_count

            # Create indices for the triangles using fixed shapes
            idx0 = jnp.zeros(polygon.shape[0]-2, dtype=jnp.int32)  # always use first vertex
            idx1 = jnp.arange(1, polygon.shape[0]-2 + 1, dtype=jnp.int32)
            idx2 = jnp.arange(2, polygon.shape[0]-2 + 2, dtype=jnp.int32)
            
            # Create triangles array with fixed shape
            triangles = jnp.stack([
                polygon[idx0],
                polygon[idx1],
                polygon[idx2]
            ], axis=1)
            
            # Valid triangles are those where both idx1 and idx2 are less than point_count
            valid_mask = (idx1 < polygon_count) & (idx2 < polygon_count)
            triangle_count = jnp.sum(valid_mask).astype(jnp.int32)
            
            return triangles.astype(jnp.float32), triangle_count, polygon_count
        
        def no_triangles(edges, signs, vertices, w, b):
            return jnp.zeros((edges.shape[0]-2, 3, 3), dtype=jnp.float32), jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)

        triangles, triangle_count, polygon_count = jax.lax.cond(jnp.any(signs<0)&jnp.any(signs>0), get_triangles, no_triangles, edges, signs, vertices, w, b)

        return triangles.astype(jnp.float32), triangle_count, polygon_count

    return extract_triangles


def v_extract_triangles_fn(extract_triangles_fn, cells, A_batch, b_batch, count):
    """Vectorized version of extract triangles function.
    
    Args:
        extract_triangles_fn: Base extract triangles function to vectorize
        cells: Cell batch to process
        A_batch: Batch of A matrices
        b_batch: Batch of b vectors
        count: Number of valid cells in batch
        
    Returns:
        Tuple of (triangles, total_triangles, polygon_vertex_counts)
    """
    def short(valid, cell, A, b):
        return jax.lax.cond(
            valid,
            extract_triangles_fn,
            lambda cell, A, b: (jnp.zeros((88, 3, 3), jnp.float32), jnp.array(0, jnp.int32), jnp.array(0, jnp.int32)),
            cell, A, b
        )
    valid = jnp.arange(A_batch.shape[0]) < count
    v_short = jax.vmap(short, in_axes=(0, 0, 0, 0))
    
    triangles, triangle_counts, polygon_vertex_counts = v_short(valid, cells, A_batch, b_batch)
    
    # Filter valid cells
    cell_mask = jnp.arange(A_batch.shape[0]) < count
    valid_mask = (triangle_counts > 0) & cell_mask
    
    # Get valid triangle counts
    valid_counts = jnp.where(valid_mask, triangle_counts, 0)
    total_triangles = jnp.sum(valid_counts).astype(jnp.int32)
    
    # Reshape triangles to (batch_size * max_triangles, 3, 3)
    max_triangles = triangles.shape[1]  # This is the max_triangles per cell
    triangles = triangles.reshape(A_batch.shape[0] * max_triangles, 3, 3)
    
    # Create mask for all valid triangles
    triangle_mask = jnp.repeat(valid_mask, max_triangles) & \
                (jnp.arange(A_batch.shape[0] * max_triangles) % max_triangles < jnp.repeat(triangle_counts, max_triangles))
    
    # Gather valid triangles
    indices = gather_indices(triangle_mask)
    triangles = triangles[indices]
    
    return triangles, total_triangles, polygon_vertex_counts

def extract_step(buffers: Buffers, v_extract, batch_size, use_polygons=False):
    """Extract triangles or polygons from cells that are ready for extraction.
    
    Args:
        buffers: Current pipeline buffers
        v_extract: Vectorized extract function (triangles or polygons)
        batch_size: Size of batches to process
        use_polygons: Whether to extract polygons (True) or triangles (False)
        
    Returns:
        Updated buffers
    """
    extract_count = buffers.extract_count
    extract_index_buffer = buffers.extract_index_buffer
    cell_references = buffers.cell_references
    weight_references = buffers.weight_references
    cell_buffer_count = buffers.cell_buffer_count

    cells, A_batch, b_batch, cell_count, extract_count, buffer_indices, weight_indices = \
        get_cells(buffers, extract_index_buffer, extract_count, batch_size)

    valid_mask = jnp.arange(batch_size) < cell_count
    cell_references = cell_references.at[buffer_indices].set(0)
    decrement = jnp.bincount(
        weight_indices,
        weights=valid_mask.astype(jnp.int32),
        length=weight_references.shape[0]
    )
    weight_references = weight_references - decrement
    cell_buffer_count = cell_buffer_count - cell_count

    if use_polygons:
        # 1) extract
        polygon_vertices, polygon_vertex_counts = v_extract(cells, A_batch, b_batch, cell_count)  # [B,E,3],[B]
        B = A_batch.shape[0]
        E = polygon_vertices.shape[1]
        N_cap = B * E

        # 2) stable-pack valid faces to front via sort-key (no duplicate padding)
        cell_mask  = (jnp.arange(B, dtype=jnp.int32) < cell_count)
        valid_face = (polygon_vertex_counts >= 3) & cell_mask
        face_key   = (jnp.where(valid_face, 0, 1).astype(jnp.int32) * B
                    + jnp.arange(B, dtype=jnp.int32))
        perm_faces = jnp.argsort(face_key)              # [B]
        counts     = polygon_vertex_counts[perm_faces]  # [B]
        polys      = polygon_vertices[perm_faces]       # [B,E,3]

        face_count = jnp.sum(valid_face).astype(jnp.int32)
        # Total vertices across valid faces; avoids dynamic slice
        prefix     = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(counts)])
        total_verts = prefix[face_count]                # scalar int32

        # 3) compact vertices of first face_count faces to the front
        flat    = polys.reshape(N_cap, 3)                               # [B*E,3]
        face_id = jnp.repeat(jnp.arange(B, dtype=jnp.int32), E)         # [B*E]
        col     = jnp.tile(jnp.arange(E, dtype=jnp.int32), B)           # [B*E]
        valid_v = (face_id < face_count) & (col < counts[face_id])      # [B*E]

        v_key   = (jnp.where(valid_v, 0, 1).astype(jnp.int32) * N_cap
                + jnp.arange(N_cap, dtype=jnp.int32))
        perm_v  = jnp.argsort(v_key)                                    # [B*E]
        packed_vertices = flat[perm_v]                                   # valid first

        # 4) masked scatters into global buffers
        dst_v = buffers.vertex_buffer_count + jnp.arange(N_cap, dtype=jnp.int32)
        write_v = jnp.arange(N_cap, dtype=jnp.int32) < total_verts
        vertex_buffer = buffers.vertex_buffer.at[dst_v].set(
            jnp.where(write_v[:, None], packed_vertices, buffers.vertex_buffer[dst_v])
        )
        vertex_buffer_count = buffers.vertex_buffer_count + total_verts

        dst_f = buffers.face_buffer_count + jnp.arange(B, dtype=jnp.int32)
        write_f = jnp.arange(B, dtype=jnp.int32) < face_count
        face_vcount_buffer = buffers.face_vertex_count_buffer.at[dst_f].set(
            jnp.where(write_f, counts, buffers.face_vertex_count_buffer[dst_f])
        )
        face_buffer_count = buffers.face_buffer_count + face_count

        buffers = replace(
            buffers,
            cell_references=cell_references,
            weight_references=weight_references,
            cell_buffer_count=cell_buffer_count,
            extract_count=extract_count,
            extract_index_buffer=extract_index_buffer,
            vertex_buffer=vertex_buffer,
            vertex_buffer_count=vertex_buffer_count,
            face_vertex_count_buffer=face_vcount_buffer,
            face_buffer_count=face_buffer_count,
        )

    else:
        # Extract triangles and update triangle buffers
        triangles, triangle_count, polygon_vertex_counts = v_extract(
            cells, A_batch, b_batch, cell_count
        )
        
        triangle_buffer = buffers.triangle_buffer
        triangle_buffer_count = buffers.triangle_buffer_count
        
        idx = jnp.arange(triangles.shape[0], dtype=jnp.int32) + triangle_buffer_count
        triangle_buffer = triangle_buffer.at[idx].set(triangles)
        triangle_buffer_count += triangle_count
        
        buffers = replace(
            buffers,
            cell_references=cell_references,
            weight_references=weight_references,
            cell_buffer_count=cell_buffer_count,
            extract_count=extract_count,
            extract_index_buffer=extract_index_buffer,
            triangle_buffer=triangle_buffer,
            triangle_buffer_count=triangle_buffer_count,
            max_polygon_vertices=jnp.maximum(jnp.max(polygon_vertex_counts), buffers.max_polygon_vertices).astype(jnp.int32)
        )
    
    return buffers
