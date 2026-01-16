import jax.numpy as jnp
import jax
from marching.buffers import Buffers

def gather_indices(mask):
    N = mask.shape[0]
    idx = jnp.nonzero(mask, size=N, fill_value=N-1)[0]
    return idx

def get_cells(buffers: Buffers, indices, count, batch_size):
    """Get cells from buffers for processing.
    
    Args:
        buffers: Current pipeline buffers
        indices: Indices of cells to get
        count: Number of valid indices
        batch_size: Size of batch to process
        
    Returns:
        Tuple of (cells, A_batch, b_batch, cell_count, new_count, buffer_indices, weight_indices)
    """
    cell_buffer = buffers.cell_buffer
    A_buffer = buffers.A_buffer
    b_buffer = buffers.b_buffer
    cell_weight_map = buffers.cell_weight_map

    new_count = jnp.maximum(0, count-batch_size).astype(jnp.int32) # find new count
    idx = jnp.arange(batch_size, dtype=jnp.int32)+new_count # find the indices of the cells to get
    valid_mask = idx<count
    buffer_indices = jnp.where(valid_mask, indices[idx], cell_weight_map.shape[0]-1)
    cells = jax.tree.map(
        lambda x: x[buffer_indices],
        cell_buffer
    )
    weight_indices = jnp.where(
        valid_mask,
        cell_weight_map[buffer_indices],
        A_buffer.shape[0]-1
    )
    A_batch = A_buffer[weight_indices]
    b_batch = b_buffer[weight_indices]
    cell_count = jnp.minimum(batch_size, count).astype(jnp.int32)
    return cells, A_batch, b_batch, cell_count, new_count, buffer_indices, weight_indices

def update_cells(cell_buffer, cells, buffer_indices):
    """Update cells in the cell buffer.
    
    Args:
        cell_buffer: Buffer containing all cells
        cells: New cells to update
        buffer_indices: Indices where to update cells
        
    Returns:
        Updated cell buffer
    """
    cell_buffer = jax.tree.map(
        lambda x, y: x.at[buffer_indices].set(y),
        cell_buffer, cells
    )
    return cell_buffer
