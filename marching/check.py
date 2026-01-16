from dataclasses import replace
from typing import Tuple, TypeAlias
from marching.cell import Cell
import jax.numpy as jnp
import jax
from marching.utils import gather_indices, get_cells, update_cells
from marching.buffers import Buffers

State: TypeAlias = Tuple[Cell, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

def make_check(ops, activations, jit=True):

    
    def make_runner(k):
        op_name = ops[k][2]
        def runner(lower, upper, op_name=op_name):
            activation = activations[op_name]
            segment_idx, bp_count, offset = activation.query_single(lower.astype(jnp.float64), upper.astype(jnp.float64))
            return segment_idx.astype(jnp.int32), bp_count.astype(jnp.int32), offset.astype(jnp.float32)
        return runner

    runners = [make_runner(i) for i in range(0, len(ops)-1)]
    runners = tuple(runners)

    def check(cell: Cell, A: jnp.ndarray, b: jnp.ndarray) -> Cell:
        def cond_fn(state: State) -> bool:
            return state[-1] < 0

        def body_fn(state: State) -> State:
            cell, A, b, A_i, b_i, send_idx = state
            neuron_idx = jnp.argmax(cell.ranges[cell.layer_idx, :])
            A_i = A[:, neuron_idx].astype(jnp.float64)
            b_i = b[neuron_idx].astype(jnp.float64)

            vertices = cell.vertices.astype(jnp.float64)
            vertex_count = cell.vertex_count
            max_vertex_count = vertices.shape[0]
            valid_vertices_mask = jnp.arange(max_vertex_count, dtype=jnp.int32) < vertex_count

            signs = jnp.dot(vertices, A_i) + b_i # batch size, max_vertex_count

            lower = jnp.min(jnp.where(valid_vertices_mask, signs, jnp.inf))
            upper = jnp.max(jnp.where(valid_vertices_mask, signs, -jnp.inf))
            
            segment_idx, bp_count, offset = jax.lax.switch(cell.layer_idx, runners, lower, upper)
            b_i = b_i + offset
            
            new_range = jnp.where(bp_count==0, -jnp.inf, upper-lower).astype(jnp.float32)
            new_index = segment_idx
            cell = replace(
                cell,
                ranges=cell.ranges.at[cell.layer_idx, neuron_idx].set(new_range),
                indices=cell.indices.at[cell.layer_idx, neuron_idx].set(new_index),
            )
            
            done = jnp.all(cell.ranges[cell.layer_idx, :]<0)

            send_idx = -1
            # continue: no breakpoints crossed and not done
            continue_loop = (~done) & (bp_count==0)

            # split: breakpoints crossed and not done
            split = (~done) & (bp_count!=0)

            # done: send to collapse
            collapse = done & (cell.layer_idx!=(len(ops)-2))

            # done: send to extract
            extract = done & (cell.layer_idx==(len(ops)-2))

            send_idx = jnp.where(continue_loop, -1, send_idx)
            send_idx = jnp.where(split, 0, send_idx)
            send_idx = jnp.where(collapse, 1, send_idx)
            send_idx = jnp.where(extract, 2, send_idx)

            return cell, A, b, A_i.astype(jnp.float32), b_i.astype(jnp.float32), send_idx

        done = jnp.all(cell.ranges[cell.layer_idx, :]<0)
        collapse = done & (cell.layer_idx!=(len(ops)-2))
        extract = done & (cell.layer_idx==(len(ops)-2))
        send_idx = -1
        send_idx = jnp.where(collapse, 1, send_idx)
        send_idx = jnp.where(extract, 2, send_idx)

        state = (cell, A, b, jnp.zeros_like(A[:, 0]), jnp.zeros_like(b[0]), send_idx)
        if jit:
            state = jax.lax.while_loop(cond_fn, body_fn, state)
        else:
            while cond_fn(state):
                state = body_fn(state)
        cell, A, b, A_i, b_i, send_idx = state
        return cell, A_i, b_i, send_idx
    return check

def v_check_fn(check_fn, cells, A_batch, b_batch, count):
    """Vectorized version of check function.
    
    Args:
        check_fn: Base check function to vectorize
        cells: Cell batch to process
        A_batch: Batch of A matrices
        b_batch: Batch of b vectors
        count: Number of valid cells in batch
        
    Returns:
        Tuple of (cells, A_i, b_i, send_indices)
    """
    def short(valid, cell, A, b):
        return jax.lax.cond(
            valid,
            check_fn,
            lambda cell, A, b: (cell, jnp.zeros_like(A[:, 0]), jnp.zeros_like(b[0]), -1),
            cell, A, b
        )
    valid = jnp.arange(A_batch.shape[0]) < count
    v_short = jax.vmap(short, in_axes=(0, 0, 0, 0))
    cells, A_i, b_i, send_indices = v_short(valid, cells, A_batch, b_batch)
    return cells, A_i, b_i, send_indices

def check_step(buffers: Buffers, v_check, batch_size):
    """Check cells and route them to appropriate next stages.
    
    Args:
        buffers: Current pipeline buffers
        v_check: Vectorized check function
        batch_size: Size of batches to process
        
    Returns:
        Updated buffers
    """
    check_count = buffers.check_count
    check_index_buffer = buffers.check_index_buffer
    split_count = buffers.split_count
    split_index_buffer = buffers.split_index_buffer
    A_i_buffer = buffers.A_i_buffer
    b_i_buffer = buffers.b_i_buffer
    collapse_count = buffers.collapse_count
    collapse_index_buffer = buffers.collapse_index_buffer
    extract_count = buffers.extract_count
    extract_index_buffer = buffers.extract_index_buffer
    cell_buffer = buffers.cell_buffer

    bp_counts = cell_buffer.bp_count[check_index_buffer]
    bp_counts = jnp.where(jnp.arange(check_index_buffer.shape[0])<check_count, bp_counts, -100)
    idx = jnp.argsort(bp_counts, descending=True)
    check_index_buffer = check_index_buffer[idx]
    buffers = replace(buffers, check_index_buffer=check_index_buffer)

    cells, A_batch, b_batch, cell_count, check_count, buffer_indices, weight_indices = \
        get_cells(buffers, check_index_buffer, check_count, batch_size)
    
    cells, A_i, b_i, send_indices = v_check(
        cells,
        A_batch,
        b_batch,
        cell_count
    )

    cell_buffer = update_cells(cell_buffer, cells, buffer_indices)

    valid_mask = jnp.arange(batch_size) < cell_count

    # Add to appropriate queues based on send_indices
    split_mask = (send_indices == 0) & valid_mask
    split_idx = gather_indices(split_mask)
    valid_split_count = jnp.sum(split_mask).astype(jnp.int32)

    collapse_mask = (send_indices == 1) & valid_mask
    collapse_idx = gather_indices(collapse_mask)
    collapse_valid_count = jnp.sum(collapse_mask).astype(jnp.int32)

    extract_mask = (send_indices == 2) & valid_mask
    extract_idx = gather_indices(extract_mask)
    extract_valid_count = jnp.sum(extract_mask).astype(jnp.int32)
    
    # Add to split queue
    idx = jnp.arange(batch_size)
    split_index_buffer = split_index_buffer.at[idx+split_count].set(buffer_indices[split_idx])
    A_i_buffer = A_i_buffer.at[idx+split_count].set(A_i[split_idx])
    b_i_buffer = b_i_buffer.at[idx+split_count].set(b_i[split_idx])
    split_count = split_count + valid_split_count
    
    # Add to collapse queue
    collapse_index_buffer = collapse_index_buffer.at[idx+collapse_count].set(buffer_indices[collapse_idx])
    collapse_count = collapse_count + collapse_valid_count
    
    # Add to extract queue
    extract_index_buffer = extract_index_buffer.at[idx+extract_count].set(buffer_indices[extract_idx])
    extract_count = extract_count + extract_valid_count

    buffers = replace(
        buffers,
        cell_buffer=cell_buffer,
        check_count=check_count,
        check_index_buffer=check_index_buffer,
        split_count=split_count,
        split_index_buffer=split_index_buffer,
        A_i_buffer=A_i_buffer,
        b_i_buffer=b_i_buffer,
        collapse_count=collapse_count,
        collapse_index_buffer=collapse_index_buffer,
        extract_count=extract_count,
        extract_index_buffer=extract_index_buffer
    )
    return buffers