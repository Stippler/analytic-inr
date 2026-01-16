from dataclasses import dataclass
import jax.numpy as jnp
import jax
from marching.cell import Cell

@jax.tree_util.register_pytree_node_class
@dataclass
class Buffers:
    # ─── cell queue ---------------------------------------------------------
    cell_buffer:               Cell          # shape prefix: (cell_buffer_size, …)
    cell_references:           jnp.ndarray   # [cell_buffer_size]
    cell_weight_map:           jnp.ndarray   # [cell_buffer_size]
    cell_buffer_count:         jnp.ndarray   # ()

    # ─── affine (A,b) store -------------------------------------------------
    A_buffer:                  jnp.ndarray   # [weight_buffer_size, in_dim, layer_width]
    b_buffer:                  jnp.ndarray   # [weight_buffer_size, layer_width]
    weight_references:         jnp.ndarray   # [weight_buffer_size]

    # ─── stage-specific index queues ---------------------------------------
    range_index_buffer:        jnp.ndarray   # [range_index_buffer_size]
    check_index_buffer:        jnp.ndarray   # [check_index_buffer_size]
    split_index_buffer:        jnp.ndarray   # [split_index_buffer_size]
    A_i_buffer:                jnp.ndarray   # [split_index_buffer_size, in_dim]
    b_i_buffer:                jnp.ndarray   # [split_index_buffer_size]
    collapse_index_buffer:     jnp.ndarray   # [collapse_index_buffer_size]
    extract_index_buffer:      jnp.ndarray   # [extract_index_buffer_size]

    # ─── per-stage counters -------------------------------------------------
    range_count:               jnp.ndarray   # ()
    check_count:               jnp.ndarray   # ()
    split_count:               jnp.ndarray   # ()
    collapse_count:            jnp.ndarray   # ()
    extract_count:             jnp.ndarray   # ()

    # ─── out-buffer -------------------------------------------------
    
    vertex_buffer:               jnp.ndarray   # [vertex_buffer_size, 3]
    vertex_buffer_count:         jnp.ndarray   # ()
    face_vertex_count_buffer:    jnp.ndarray   # [face_buffer_size]
    face_buffer_count:           jnp.ndarray   # ()
    
    triangle_buffer:             jnp.ndarray   # [triangle_buffer_size, 3, 3]
    triangle_buffer_count:       jnp.ndarray   # ()

    # statistics
    max_vertex_count:          jnp.ndarray   # ()
    max_edge_count:            jnp.ndarray   # ()
    max_polygon_vertices:      jnp.ndarray   # ()
    max_cell_buffer_count:     jnp.ndarray   # ()
    max_weight_buffer_count:   jnp.ndarray   # ()
    max_range_index_buffer_count: jnp.ndarray   # ()
    max_check_index_buffer_count: jnp.ndarray   # ()
    max_split_index_buffer_count: jnp.ndarray   # ()
    max_collapse_index_buffer_count: jnp.ndarray   # ()
    max_extract_index_buffer_count: jnp.ndarray   # ()
    total_iterations:          jnp.ndarray   # ()
    max_cell_split_count:      jnp.ndarray   # ()

    # ─── flags (must be last due to default values) -------------------------------------------------
    use_polygon_buffer:       bool = False

    # ======================================================================
    #  factory
    # ======================================================================
    @classmethod
    def create(
        cls,
        *,
        ops: list,
        proto_cell:               Cell,
        cell_buffer_size:         int,
        range_index_buffer_size:  int,
        check_index_buffer_size:  int,
        split_index_buffer_size:  int,
        collapse_index_buffer_size: int,
        extract_index_buffer_size:  int,
        triangle_buffer_size:     int,
        weight_buffer_size:       int = None,
        float_dtype               = jnp.float32,
        use_polygon_buffer:       bool = False,
    ) -> "Buffers":
        """
        Allocate every array at its fixed, maximum size and
        seed slot 0 with the prototype cell and its (A, b).
        """
        # ——— basic problem sizes ————————————————————————————————
        in_dim        = proto_cell.vertices.shape[-1]
        layer_width   = proto_cell.indices.shape[-1]
        A, b, op_name = ops[0]

        if weight_buffer_size is None:
            weight_buffer_size = cell_buffer_size

        # ——— cell queue ————————————————————————————————————————
        cell_buffer         = _make_buffer_with_proto(cell_buffer_size, proto_cell)
        cell_references     = jnp.zeros((cell_buffer_size,),  jnp.int32).at[0].set(1)
        cell_weight_map     = jnp.zeros((cell_buffer_size,),  jnp.int32)         # slot 0 → 0 by default
        cell_buffer_count   = jnp.array(1, dtype=jnp.int32)

        # ——— affine store ——————————————————————————————————————
        A_buffer            = jnp.zeros((weight_buffer_size, in_dim, layer_width),
                                        float_dtype).at[0].set(A)  # placeholder
        b_buffer            = jnp.zeros((weight_buffer_size, layer_width),
                                        float_dtype).at[0].set(b)
        weight_references   = jnp.zeros((weight_buffer_size,), jnp.int32).at[0].set(1)

        # ——— stage queues & counters ————————————————————————————
        range_index_buffer   = jnp.zeros(range_index_buffer_size,  jnp.int32).at[0].set(0)
        check_index_buffer   = jnp.zeros(check_index_buffer_size,  jnp.int32)
        split_index_buffer   = jnp.zeros(split_index_buffer_size,  jnp.int32)
        A_i_buffer           = jnp.zeros((split_index_buffer_size, in_dim), float_dtype)
        b_i_buffer           = jnp.zeros(split_index_buffer_size,           float_dtype)
        collapse_index_buffer= jnp.zeros(collapse_index_buffer_size, jnp.int32)
        extract_index_buffer = jnp.zeros(extract_index_buffer_size,  jnp.int32)

        range_count    = jnp.array(1, dtype=jnp.int32)   # slot 0 enqueued already
        check_count    = jnp.array(0, dtype=jnp.int32)
        split_count    = jnp.array(0, dtype=jnp.int32)
        collapse_count = jnp.array(0, dtype=jnp.int32)
        extract_count  = jnp.array(0, dtype=jnp.int32)

        # ——— output buffers (conditional allocation) ————————————————————————————————
        if use_polygon_buffer:
            vertex_buffer             = jnp.zeros((triangle_buffer_size * 3, 3), float_dtype)  # Assume max 3 vertices per triangle for sizing
            vertex_buffer_count       = jnp.array(0, dtype=jnp.int32)
            face_vertex_count_buffer  = jnp.zeros((triangle_buffer_size,), jnp.int32)
            face_buffer_count         = jnp.array(0, dtype=jnp.int32)
            triangle_buffer           = None
            triangle_buffer_count     = None
        else:
            triangle_buffer             = jnp.zeros((triangle_buffer_size, 3, 3), float_dtype)
            triangle_buffer_count       = jnp.array(0, dtype=jnp.int32)
            vertex_buffer               = None
            vertex_buffer_count         = None
            face_vertex_count_buffer    = None
            face_buffer_count           = None

        # statistics
        max_vertex_count = jnp.array(0, dtype=jnp.int32)
        max_edge_count = jnp.array(0, dtype=jnp.int32)
        max_polygon_vertices = jnp.array(0, dtype=jnp.int32)
        max_cell_buffer_count = jnp.array(0, dtype=jnp.int32)
        max_weight_buffer_count = jnp.array(0, dtype=jnp.int32)
        max_range_index_buffer_count = jnp.array(0, dtype=jnp.int32)
        max_check_index_buffer_count = jnp.array(0, dtype=jnp.int32)
        max_split_index_buffer_count = jnp.array(0, dtype=jnp.int32)
        max_collapse_index_buffer_count = jnp.array(0, dtype=jnp.int32)
        max_extract_index_buffer_count = jnp.array(0, dtype=jnp.int32)
        total_iterations = jnp.array(0, dtype=jnp.int64)
        max_cell_split_count = jnp.array(0, dtype=jnp.int32)

        # ==================================================================
        return cls(
            # cell queue
            cell_buffer, cell_references, cell_weight_map, cell_buffer_count,
            # affine
            A_buffer, b_buffer, weight_references,
            # stage queues
            range_index_buffer, check_index_buffer, split_index_buffer,
            A_i_buffer, b_i_buffer,
            collapse_index_buffer, extract_index_buffer,
            # counters
            range_count, check_count, split_count, collapse_count, extract_count,
            # output buffers
            vertex_buffer, vertex_buffer_count, face_vertex_count_buffer, face_buffer_count,
            triangle_buffer, triangle_buffer_count,
            # statistics
            max_vertex_count, max_edge_count, max_polygon_vertices,
            max_cell_buffer_count, max_weight_buffer_count,
            max_range_index_buffer_count, max_check_index_buffer_count, max_split_index_buffer_count,
            max_collapse_index_buffer_count, max_extract_index_buffer_count, total_iterations,
            max_cell_split_count,
            # flags
            use_polygon_buffer
        )

    # ======================================================================
    #  pytree plumbing
    # ======================================================================
    def tree_flatten(self):
        # Base children (always included)
        base_children = [
            # cell queue
            self.cell_buffer, self.cell_references, self.cell_weight_map, self.cell_buffer_count,
            # affine
            self.A_buffer, self.b_buffer, self.weight_references,
            # stage queues
            self.range_index_buffer, self.check_index_buffer, self.split_index_buffer,
            self.A_i_buffer, self.b_i_buffer,
            self.collapse_index_buffer, self.extract_index_buffer,
            # counters
            self.range_count, self.check_count, self.split_count,
            self.collapse_count, self.extract_count,
        ]
        
        # Conditional output buffers
        if self.use_polygon_buffer:
            output_children = [
                self.vertex_buffer, self.vertex_buffer_count, 
                self.face_vertex_count_buffer, self.face_buffer_count
            ]
        else:
            output_children = [
                self.triangle_buffer, self.triangle_buffer_count
            ]
        
        # Statistics (always included)
        stats_children = [
            self.max_vertex_count, self.max_edge_count, self.max_polygon_vertices,
            self.max_cell_buffer_count, self.max_weight_buffer_count,
            self.max_range_index_buffer_count, self.max_check_index_buffer_count, self.max_split_index_buffer_count,
            self.max_collapse_index_buffer_count, self.max_extract_index_buffer_count, self.total_iterations,
            self.max_cell_split_count
        ]
        
        children = tuple(base_children + output_children + stats_children)
        aux_data = {'use_polygon_buffer': self.use_polygon_buffer}  # Store flag in aux_data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        use_polygon_buffer = aux_data['use_polygon_buffer']
        
        # Convert to iterator for easier consumption
        children_iter = iter(children)
        
        # Base children (explicit construction)
        # Cell queue
        cell_buffer = next(children_iter)
        cell_references = next(children_iter)
        cell_weight_map = next(children_iter)
        cell_buffer_count = next(children_iter)
        
        # Affine store
        A_buffer = next(children_iter)
        b_buffer = next(children_iter)
        weight_references = next(children_iter)
        
        # Stage queues
        range_index_buffer = next(children_iter)
        check_index_buffer = next(children_iter)
        split_index_buffer = next(children_iter)
        A_i_buffer = next(children_iter)
        b_i_buffer = next(children_iter)
        collapse_index_buffer = next(children_iter)
        extract_index_buffer = next(children_iter)
        
        # Counters
        range_count = next(children_iter)
        check_count = next(children_iter)
        split_count = next(children_iter)
        collapse_count = next(children_iter)
        extract_count = next(children_iter)
        
        # Output buffers (conditional)
        if use_polygon_buffer:
            vertex_buffer = next(children_iter)
            vertex_buffer_count = next(children_iter)
            face_vertex_count_buffer = next(children_iter)
            face_buffer_count = next(children_iter)
            triangle_buffer = None
            triangle_buffer_count = None
        else:
            triangle_buffer = next(children_iter)
            triangle_buffer_count = next(children_iter)
            vertex_buffer = None
            vertex_buffer_count = None
            face_vertex_count_buffer = None
            face_buffer_count = None
        
        # Statistics
        max_vertex_count = next(children_iter)
        max_edge_count = next(children_iter)
        max_polygon_vertices = next(children_iter)
        max_cell_buffer_count = next(children_iter)
        max_weight_buffer_count = next(children_iter)
        max_range_index_buffer_count = next(children_iter)
        max_check_index_buffer_count = next(children_iter)
        max_split_index_buffer_count = next(children_iter)
        max_collapse_index_buffer_count = next(children_iter)
        max_extract_index_buffer_count = next(children_iter)
        total_iterations = next(children_iter)
        max_cell_split_count = next(children_iter)
        
        # Construct object with explicit arguments
        return cls(
            # Cell queue
            cell_buffer, cell_references, cell_weight_map, cell_buffer_count,
            # Affine
            A_buffer, b_buffer, weight_references,
            # Stage queues
            range_index_buffer, check_index_buffer, split_index_buffer,
            A_i_buffer, b_i_buffer,
            collapse_index_buffer, extract_index_buffer,
            # Counters
            range_count, check_count, split_count, collapse_count, extract_count,
            # Output buffers
            vertex_buffer, vertex_buffer_count, face_vertex_count_buffer, face_buffer_count,
            triangle_buffer, triangle_buffer_count,
            # Statistics
            max_vertex_count, max_edge_count, max_polygon_vertices,
            max_cell_buffer_count, max_weight_buffer_count,
            max_range_index_buffer_count, max_check_index_buffer_count, max_split_index_buffer_count,
            max_collapse_index_buffer_count, max_extract_index_buffer_count, total_iterations,
            max_cell_split_count,
            # Flags (with default values, must be last)
            use_polygon_buffer
        )

def _make_buffer_with_proto(n: int, proto: Cell) -> Cell:
    """
    Light re-implementation of your helper:
    replicate a prototype Cell `n` times by giving every leaf
    a leading batch dimension of size `n`.
    """
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), proto) 
