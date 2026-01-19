from typing import Callable, Dict, List, Tuple
import jax.numpy as jnp
import jax
from dataclasses import dataclass, replace
import numpy as np
from marching.activation import Activation, ReluActivation
from marching.cell import Cell
from functools import partial
from marching.utils import gather_indices, get_cells, update_cells
from marching.buffers import Buffers

@dataclass(frozen=True)
class AffineContext():
    mode: str = 'affine_fixed'
    truncate_count: int = -777
    truncate_policy: str = 'absolute'
    affine_domain_terms: int = 0
    n_append: int = 0

    def __post_init__(self):
        if self.mode not in ['interval', 'affine_fixed', 'affine_truncate', 'affine_append', 'affine_all']:
            raise ValueError("invalid mode")

        if self.mode == 'affine_truncate':
            if self.truncate_count is None:
                raise ValueError("must specify truncate count")

def coordinates_in_general_box(ctx, center, vecs):
    base = center
    if ctx.mode == 'interval':
        aff = jnp.zeros((0,center.shape[-1]))
        err = jnp.sum(jnp.abs(vecs), axis=0)
    else:
        aff = vecs
        err = jnp.zeros_like(center)
    return base, aff, err

def radius(input):
    if is_const(input): return 0.
    base, aff, err = input
    rad = jnp.sum(jnp.abs(aff), axis=0)
    if err is not None:
        rad += err
    return rad

def is_const(input):
    base, aff, err = input
    if err is not None: return False
    return aff is None or aff.shape[0] == 0

def may_contain_bounds(input):
    '''
    An interval range of values that `input` _may_ take along the domain
    '''
    base, aff, err = input
    rad = radius(input)
    return base-rad, base+rad

def sin_bound(lower, upper):
    '''
    Bound sin([lower,upper])
    '''
    f_lower = jnp.sin(lower)
    f_upper = jnp.sin(upper)

    # test if there is an interior peak in the range
    lower /= 2. * jnp.pi
    upper /= 2. * jnp.pi
    contains_min = jnp.ceil(lower - .75) < (upper - .75)
    contains_max = jnp.ceil(lower - .25) < (upper - .25)

    # result is either at enpoints or maybe an interior peak
    out_lower = jnp.minimum(f_lower, f_upper)
    out_lower = jnp.where(contains_min, -1., out_lower)
    out_upper = jnp.maximum(f_lower, f_upper)
    out_upper = jnp.where(contains_max, 1., out_upper)

    return out_lower, out_upper

def cos_bound(lower, upper):
    return sin_bound(lower + jnp.pi/2, upper + jnp.pi/2)

def minimum_all(vals):
    '''
    Take elementwise minimum of a list of arrays
    '''
    combined = jnp.stack(vals, axis=0)
    return jnp.min(combined, axis=0)

def maximum_all(vals):
    '''
    Take elementwise maximum of a list of arrays
    '''
    combined = jnp.stack(vals, axis=0)
    return jnp.max(combined, axis=0)

def truncate_affine(ctx, input):
    # do nothing if the input is a constant or we are not in truncate mode
    if is_const(input): return input
    if ctx.mode != 'affine_truncate':
        return input

    # gather values
    base, aff, err = input
    n_keep = ctx.truncate_count

    # if the affine list is shorter than the truncation length, nothing to do
    if aff.shape[0] <= n_keep:
        return input

    # compute the magnitudes of each affine value
    # TODO fanicier policies?
    if ctx.truncate_policy == 'absolute':
        affine_mags = jnp.sum(jnp.abs(aff), axis=-1)
    elif ctx.truncate_policy == 'relative':
        affine_mags = jnp.sum(jnp.abs(aff), axis=-1) / jnp.abs(base)
    else:
        raise RuntimeError("bad policy")

    # sort the affine terms by by magnitude
    sort_inds = jnp.argsort(-affine_mags, axis=-1) # sort to decreasing order
    aff = aff[sort_inds,:]

    # keep the n_keep highest-magnitude entries
    aff_keep = aff[:n_keep,:]
    aff_drop = aff[n_keep:,:]

    # for all the entries we aren't keeping, add their contribution to the interval error
    err = err + jnp.sum(jnp.abs(aff_drop), axis=0)

    return base, aff_keep, err

def apply_linear_approx(ctx, input, alpha, beta, delta):
    base, aff, err = input
    base = alpha * base + beta
    if aff is not None:
        aff = alpha * aff

    # This _should_ always be positive by definition. Always be sure your 
    # approximation routines are generating positive delta.
    # At most, we defending against floating point error here.
    delta = jnp.abs(delta)

    if ctx.mode in ['interval', 'affine_fixed']:
        err = alpha * err + delta
    elif ctx.mode in ['affine_truncate', 'affine_all']:
        err = alpha * err
        new_aff = jnp.diag(delta)
        aff = jnp.concatenate((aff, new_aff), axis=0)
        base, aff, err = truncate_affine(ctx, (base, aff, err))

    elif ctx.mode in ['affine_append']:
        err = alpha * err
        
        keep_vals, keep_inds = jax.lax.top_k(delta, ctx.n_append)
        row_inds = jnp.arange(ctx.n_append)
        new_aff = jnp.zeros((ctx.n_append, aff.shape[-1]))
        new_aff = new_aff.at[row_inds, keep_inds].set(keep_vals)
        aff = jnp.concatenate((aff, new_aff), axis=0)
        err = err + (jnp.sum(delta) - jnp.sum(keep_vals)) # add in the error for the affs we didn't keep

    return base, aff, err


def dense(input, A, b):
    # seg_idx, ranges, bp_count, base, aff, err = input
    base, aff, err = input
    
    if(is_const((base, aff, err))):
        out = jnp.dot(base, A)
        if b is not None:
            out += b
        return out, None, None

    def dot(x, with_abs=False):
        myA = jnp.abs(A) if with_abs else A 
        return jnp.dot(x, myA)
 
    base = dot(base)
    aff = jax.vmap(dot)(aff)
    err = dot(err, with_abs=True)

    if b is not None:
        base += b

    return base, aff, err


def relu(input, ctx):
    # Chebyshev bound
    base, aff, err = input

    if is_const((base, aff, err)):
        return jax.nn.relu(base), aff, err

    lower, upper = may_contain_bounds((base, aff, err))

    # Compute the linearized approximation
    alpha = (jax.nn.relu(upper) - jax.nn.relu(lower)) / (upper - lower)
    alpha = jnp.where(lower >= 0, 1., alpha)
    alpha = jnp.where(upper < 0, 0., alpha)
    # handle numerical badness in the denominator above
    alpha = jnp.nan_to_num(alpha, nan=0.0, copy=False) # necessary?
    alpha = jnp.clip(alpha, a_min=0., a_max=1.) 

    # here, alpha/beta are necessarily positive, which makes this simpler
    beta = (jax.nn.relu(lower) - alpha * lower) / 2
    delta = beta

    base, aff, err = apply_linear_approx(ctx, (base, aff, err), alpha, beta, delta)
    return (base, aff, err), lower, upper


def sin(input, ctx):
    # not-quite Chebyshev bound
    base, aff, err = input
    pi = jnp.pi

    if is_const(input):
        return jnp.sin(base), aff, err

    lower, upper = may_contain_bounds(input)

    slope_lower, slope_upper = cos_bound(lower, upper)
    alpha = 0.5 * (slope_lower + slope_upper) # this is NOT the Chebyshev value, but seems reasonable
    alpha = jnp.clip(alpha, a_min=-1., a_max=1.) # (should already be there, this is for numerics only)

    # We want to find the minima/maxima of (sin(x) - alpha*x) on [lower, upper] to compute our 
    # beta and delta. In addition to the endpoints, some calc show there can be interior 
    # extrema at +-arccos(alpha) + 2kpi for some integer k.
    # The extrema will 
    intA = jnp.arccos(alpha)
    intB = -intA

    # The the first and last occurence of a value which repeats mod 2pi on the domain [lower, upper]
    # (these give the only possible locations for our extrema)
    def first(x): return 2.*pi*jnp.ceil((lower + x) / (2.*pi)) - x
    def last(x): return 2.*pi*jnp.floor((upper - x) / (2.*pi)) + x

    extrema_locs = [lower, upper, first(intA), last(intA), first(intB), last(intB)]
    extrema_locs = [jnp.clip(x, a_min=lower, a_max=upper) for x in extrema_locs]
    extrema_vals = [jnp.sin(x) - alpha * x for x in extrema_locs]

    r_lower = minimum_all(extrema_vals)
    r_upper = maximum_all(extrema_vals)

    beta = 0.5 * (r_upper + r_lower)
    delta = r_upper - beta

    output = apply_linear_approx(ctx, input, alpha, beta, delta)
    return output, lower, upper

def squeeze_last(input):
    base, aff, err = input
    s = lambda x : jnp.squeeze(x, axis=0)
    base = s(base)
    if is_const((base, aff, err)):
        return base, None, None
    aff = jax.vmap(s)(aff)
    err = s(err)
    return base, aff, err

def init_aabb(cell: Cell, ctx: AffineContext):
    vertices = cell.vertices
    vertex_count = cell.vertex_count
    mask = jnp.arange(vertices.shape[0], dtype=jnp.int32) < vertex_count

    min_x = jnp.min(jnp.where(mask, vertices[:, 0], jnp.inf))
    max_x = jnp.max(jnp.where(mask, vertices[:, 0], -jnp.inf))
    min_y = jnp.min(jnp.where(mask, vertices[:, 1], jnp.inf))
    max_y = jnp.max(jnp.where(mask, vertices[:, 1], -jnp.inf))
    min_z = jnp.min(jnp.where(mask, vertices[:, 2], jnp.inf))
    max_z = jnp.max(jnp.where(mask, vertices[:, 2], -jnp.inf))

    box_center = jnp.stack([
        (min_x + max_x)/2.0,
        (min_y + max_y)/2.0,
        (min_z + max_z)/2.0
    ], axis=-1)

    v0 = jnp.stack([(max_x - min_x)*0.5, 
                   jnp.zeros_like(min_x),
                   jnp.zeros_like(min_x)], axis=-1)
    v1 = jnp.stack([jnp.zeros_like(min_x),
                   (max_y - min_y)*0.5,
                   jnp.zeros_like(min_y)], axis=-1)
    v2 = jnp.stack([jnp.zeros_like(min_x),
                   jnp.zeros_like(min_y),
                   (max_z - min_z)*0.5], axis=-1)
    
    box_vecs = jnp.stack([v0, v1, v2], axis=-1)
    base, aff, err = coordinates_in_general_box(ctx, box_center, box_vecs)
    return base, aff, err

def init_obb_pca(cell: Cell, ctx: AffineContext):
    vertices = cell.vertices
    vertex_count = cell.vertex_count
    mask      = (jnp.arange(vertices.shape[0]) < vertex_count)
    mask_f    = mask[:, None]
    center    = jnp.sum(vertices * mask_f, axis=0) / vertex_count

    scale     = jnp.sqrt(vertex_count).astype(vertices.dtype)
    centred   = (vertices - center) * mask_f / scale

    cov       = centred.T @ centred
    eig_vals, eig_vecs = jnp.linalg.eigh(cov)
    order     = eig_vals.argsort()[::-1]
    eig_vecs  = eig_vecs[:, order]

    proj      = centred @ eig_vecs
    min_p     = jnp.min(jnp.where(mask_f, proj,  jnp.inf), axis=0)
    max_p     = jnp.max(jnp.where(mask_f, proj, -jnp.inf), axis=0)
    half_len_s = jnp.maximum((max_p - min_p) * 0.5, 1e-6)
    offset_s   = (max_p + min_p) * 0.5

    half_len  = half_len_s * scale
    offset    = offset_s   * scale

    box_center = center + eig_vecs @ offset
    box_vecs   = eig_vecs @ jnp.diag(half_len)

    base, aff, err = coordinates_in_general_box(ctx, box_center, box_vecs)
    return base, aff, err

def check_layer(cell: Cell, x: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], A: jnp.ndarray, b: jnp.ndarray, ctx: AffineContext, activation: Activation, layer_idx: int):
    x = dense(x, A=A, b=b)
    if activation.name == 'relu':
        x, lower, upper = relu(x, ctx=ctx)
    elif activation.name == 'sin':
        x, lower, upper = sin(x, ctx=ctx)
    else:
        raise ValueError(f"Invalid activation: {activation.name}")
    lower = lower-1e-5
    upper = upper+1e-5
    seg_idx, bps, offset = activation.query(lower, upper)

    new_indices = jnp.where(
        bps<0,
        seg_idx,
        cell.indices[layer_idx, :]
    )

    new_ranges = upper-lower
    old_ranges = cell.ranges[layer_idx, :]
    new_ranges = jnp.where(
        old_ranges>new_ranges,
        new_ranges,
        old_ranges
    )

    cell = replace(
        cell,
        indices=cell.indices.at[layer_idx, :].set(new_indices),
        ranges=cell.ranges.at[layer_idx, :].set(new_ranges),
        bp_count=cell.bp_count+jnp.sum(bps).astype(jnp.int32)
    )
    return cell, x

def make_range(ops, ctx: AffineContext, activations: Dict[str, Activation], jit=True):

    init = init_aabb

    def make_runner(k):
        def runner(cell, x, A, b):
            cell, x = check_layer(cell, x, A, b, ctx=ctx, activation=activations[ops[k][2]], layer_idx=k)
            cell: Cell
            cell.bp_count = jnp.sum(cell.ranges[cell.layer_idx, :] > 0).astype(jnp.int32)
            for i, (A, b, op_name) in enumerate(ops[k+1:len(ops)-1], start=k+1):
                cell, x = check_layer(cell, x, A, b, ctx=ctx, activation=activations[op_name], layer_idx=i)
            x = dense(x, A=ops[-1][0], b=ops[-1][1])
            x = squeeze_last(x)
            base, aff, err = x
            may_lower, may_upper = may_contain_bounds((base, aff, err))
            contains_surface = (may_lower < -1e-6) & (1e-6 < may_upper) & (cell.vertex_count > 3) & (cell.cell_split_count<90)
            # contains_surface = contains_surface & ((cell.vertex_count*3//2)==cell.edges)
            # vertices = jnp.where(60<cell.vertex_count, cell.vertices, jnp.zeros_like(cell.vertices[0]))
            # contains_surface = ~jnp.any(jnp.abs(vertices)>1)
            return cell, contains_surface
        return runner
    
    runners = tuple([make_runner(i) for i in range(len(ops)-1)])
    
    def range_fn(cell: Cell, A: jnp.ndarray, b: jnp.ndarray)->Cell:
        x = init(cell, ctx=ctx)

        if jit:
            cell, contains_surface = jax.lax.switch(
                cell.layer_idx,
                runners,
                cell, x, A, b
            )
        else:
            cell, contains_surface = runners[cell.layer_idx](cell, x, A, b)
        
        return cell, contains_surface
    return range_fn

def v_range_fn(range_fn, cells, A_batch, b_batch, count):
    """Vectorized version of range function.
    
    Args:
        range_fn: Base range function to vectorize
        cells: Cell batch to process
        A_batch: Batch of A matrices
        b_batch: Batch of b vectors
        count: Number of valid cells in batch
        
    Returns:
        Tuple of (processed_cells, contains_surface)
    """
    def short(valid, cell, A, b):
        return jax.lax.cond(
            valid,
            range_fn,
            lambda cell, A, b: (cell, jnp.array(False)),
            cell, A, b
        )
    valid = jnp.arange(A_batch.shape[0]) < count
    v_short = jax.vmap(short, in_axes=(0, 0, 0, 0))
    cells, contains_surface = v_short(valid, cells, A_batch, b_batch)
    return cells, contains_surface

def range_step(buffers: Buffers, v_range, batch_size):
    """Process cells in the range stage.
    
    Args:
        buffers: Current pipeline buffers
        v_range: Vectorized range function
        batch_size: Size of batches to process
        
    Returns:
        Updated buffers
    """
    range_count = buffers.range_count
    range_index_buffer = buffers.range_index_buffer
    cell_references = buffers.cell_references
    weight_references = buffers.weight_references
    cell_buffer_count = buffers.cell_buffer_count
    check_count = buffers.check_count
    check_index_buffer = buffers.check_index_buffer
    cell_buffer = buffers.cell_buffer

    cells, A_batch, b_batch, cell_count, range_count, buffer_indices, weight_indices = \
        get_cells(buffers, range_index_buffer, range_count, batch_size)
    
    cells, contains_surface = v_range(
        cells,
        A_batch,
        b_batch,
        cell_count
    )
    cell_buffer = update_cells(cell_buffer, cells, buffer_indices)

    idx = jnp.arange(batch_size, dtype=jnp.int32)
    valid_mask = (idx<cell_count)
    remove_mask = valid_mask & (~contains_surface)
    keep_mask = valid_mask & contains_surface

    # Release cells not containing surface
    cell_references = cell_references.at[buffer_indices].set(
        jnp.where(remove_mask, 
            0,
            cell_references[buffer_indices]
        )
    )

    # Decrement weight references of cells not containing surface
    decrement = jnp.bincount(
        weight_indices,
        weights=remove_mask.astype(jnp.int32),
        length=weight_references.shape[0]
    )
    weight_references = weight_references - decrement           

    new_cell_count = jnp.sum(keep_mask).astype(jnp.int32)
    cell_buffer_count = cell_buffer_count - cell_count + new_cell_count

    # Move buffer_indices to the front of the array
    gather_idx = gather_indices(keep_mask)
    buffer_indices = buffer_indices[gather_idx]
    
    # Add cells containing surface to check queue
    check_index_buffer = check_index_buffer.at[idx+check_count].set(buffer_indices)
    
    # Increment check_count by the number of cells containing surface
    check_count = check_count + new_cell_count

    buffers = replace(
        buffers,
        cell_buffer=cell_buffer,
        range_count=range_count,
        range_index_buffer=range_index_buffer,
        cell_references=cell_references,
        weight_references=weight_references,
        cell_buffer_count=cell_buffer_count,
        check_count=check_count,
        check_index_buffer=check_index_buffer
    )
    return buffers
