from dataclasses import replace
from typing import Dict, List, Tuple
from marching.activation import Activation
import jax.numpy as jnp
import jax
from marching.cell import Cell
from marching.utils import gather_indices, get_cells, update_cells
from marching.buffers import Buffers

def make_collapse(ops: List[Tuple[jnp.ndarray, jnp.ndarray, str]], activations: Dict[str, Activation], jit=True):

    def make_runner(k):
        A_next, b_next, _ = ops[k]
        op_name = ops[k-1][2]
        def runner(cell: Cell, A, b, A_next=A_next, b_next=b_next, op_name=op_name):
            A_scaled, b_scaled = activations[op_name].collapse(A, b, cell.indices[cell.layer_idx, :])
            A_composed = jnp.dot(A_scaled.astype(jnp.float64), A_next.astype(jnp.float64)).astype(jnp.float32) 
            b_composed = jnp.dot(b_scaled.astype(jnp.float64), A_next.astype(jnp.float64)).astype(jnp.float32) + b_next
            return A_composed, b_composed
        return runner

    runners = [make_runner(i) for i in range(1, len(ops)-1)]
    runners = tuple(runners)

    def collapse(cell: Cell, A: jnp.ndarray, b: jnp.ndarray) -> Tuple[Cell, jnp.ndarray, jnp.ndarray]:
        if jit:
            A_composed, b_composed = jax.lax.switch(
                cell.layer_idx,
                runners,
                cell, A, b
            )
        else:
            A_composed, b_composed = runners[cell.layer_idx](cell, A, b)
        cell = replace(
            cell,
            layer_idx=cell.layer_idx+1
        )
        return cell, A_composed, b_composed
    return collapse

def v_collapse_fn(collapse_fn, cells, A_batch, b_batch, count):
    """Vectorized version of collapse function.
    
    Args:
        collapse_fn: Base collapse function to vectorize
        cells: Cell batch to process
        A_batch: Batch of A matrices
        b_batch: Batch of b vectors
        count: Number of valid cells in batch
        
    Returns:
        Tuple of (cells, A_composed, b_composed)
    """
    def short(valid, cell, A, b):
        return jax.lax.cond(
            valid,
            collapse_fn,
            lambda cell, A, b: (cell, A, b),
            cell, A, b
        )
    valid = jnp.arange(A_batch.shape[0]) < count
    v_short = jax.vmap(short, in_axes=(0, 0, 0, 0))
    cells, A_composed, b_composed = v_short(valid, cells, A_batch, b_batch)
    return cells, A_composed, b_composed

def collapse_step(buffers: Buffers, v_collapse, batch_size):
    """Collapse cells that are ready for collapsing.
    
    Args:
        buffers: Current pipeline buffers
        v_collapse: Vectorized collapse function
        batch_size: Size of batches to process
        
    Returns:
        Updated buffers
    """
    collapse_count = buffers.collapse_count
    collapse_index_buffer = buffers.collapse_index_buffer
    weight_references = buffers.weight_references
    range_count = buffers.range_count
    range_index_buffer = buffers.range_index_buffer
    cell_buffer = buffers.cell_buffer
    A_buffer = buffers.A_buffer
    b_buffer = buffers.b_buffer
    cell_weight_map = buffers.cell_weight_map

    cells, A_batch, b_batch, cell_count, collapse_count, buffer_indices, weight_indices = \
        get_cells(buffers, collapse_index_buffer, collapse_count, batch_size)
    
    idx = jnp.arange(batch_size, dtype=jnp.int32)
    valid_mask = idx<cell_count
    decrement = jnp.bincount(
        weight_indices,
        weights=valid_mask.astype(jnp.int32),
        length=weight_references.shape[0]
    )
    weight_references = weight_references - decrement

    cells, A_composed, b_composed = v_collapse(
        cells,
        A_batch,
        b_batch,
        cell_count
    )
    cell_buffer = update_cells(cell_buffer, cells, buffer_indices)
    
    free_weight_slots = jnp.argwhere(
        weight_references[:-1] == 0,
        size=batch_size,
        fill_value=-1
    )[:, 0].astype(jnp.int32)

    free_weight_slots = jnp.where(valid_mask, free_weight_slots, weight_references.shape[0]-1)

    weight_references = weight_references.at[free_weight_slots].set(1)

    A_buffer = A_buffer.at[free_weight_slots].set(A_composed)
    b_buffer = b_buffer.at[free_weight_slots].set(b_composed)

    cell_weight_map = cell_weight_map.at[buffer_indices].set(free_weight_slots)
    
    # Add collapsed cells to range queue
    range_idx = jnp.arange(batch_size) + range_count
    range_index_buffer = range_index_buffer.at[range_idx].set(buffer_indices)
    range_count = range_count + cell_count

    buffers = replace(
        buffers,
        cell_buffer=cell_buffer,
        range_count=range_count,
        range_index_buffer=range_index_buffer,
        cell_weight_map=cell_weight_map,
        collapse_count=collapse_count,
        collapse_index_buffer=collapse_index_buffer,
        A_buffer=A_buffer,
        b_buffer=b_buffer,
        weight_references=weight_references
    )
    return buffers
