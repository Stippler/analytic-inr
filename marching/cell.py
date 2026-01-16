import jax
from jax import numpy as jnp
from dataclasses import dataclass
import itertools
import matplotlib.pyplot as plt
from typing import Tuple


@jax.tree_util.register_pytree_node_class
@dataclass
class Cell:
    vertices:      jnp.ndarray     # [max_vertices, dim]
    edges:         jnp.ndarray     # [max_edges, 2]
    vertex_count:  jnp.ndarray # scalar
    edge_count:    jnp.ndarray # scalar

    indices:       jnp.ndarray # [activation_layer_count, layer_width]
    ranges:        jnp.ndarray # [activation_layer_count, layer_width]

    bp_count:      jnp.ndarray # scalar
    layer_idx:     jnp.ndarray # scalar
    reference_idx: jnp.ndarray # scalar

    cell_split_count:   jnp.ndarray # scalar

    @classmethod
    def create_initial(cls,
                       ops,
                       max_vertices: int = 60,
                       max_edges:    int = 90,
                       float_dtype=jnp.float32):
        input_dim = ops[0][0].shape[0]
        layer_width = ops[0][1].shape[0]
        activation_layer_count = len(ops)-1

        # 1) build the [-1,1]^d cube vertices
        initial_verts = jnp.array(
            list(itertools.product([-1.,1.], repeat=input_dim)),
            dtype=float_dtype
        )
        V0 = initial_verts.shape[0]

        # 2) build the edges by connecting points that differ in one coordinate
        diffs = jnp.abs(initial_verts[:, None, :] - initial_verts[None, :, :]).sum(-1)
        V0 = 2**input_dim  # 8
        i_idx = jnp.arange(V0)[:, None]
        j_idx = jnp.arange(V0)[None, :]
        mask  = (diffs == 2.0) & (i_idx < j_idx)  # shape (8,8), exactly 12 True’s

        # 4) statically grab those 12 pairs
        # 1) your mask of shape (8,8), exactly 12 True’s for the unique edges i<j
        V0      = initial_verts.shape[0]           # == 8
        i_idx   = jnp.arange(V0)[:, None]
        j_idx   = jnp.arange(V0)[None, :]
        mask    = (diffs == 2.0) & (i_idx < j_idx)  # shape (8,8)

        # 2) fixed‐size nonzero → two (12,) arrays of row and col indices
        rows, cols = jnp.nonzero(
            mask,
            size=12,         # reserve exactly 12 hits
            fill_value=0     # unused slots if any (there won’t be) are zero
        )

        # 3) stack into a (12,2) array just like argwhere would give
        initial_edges = jnp.stack([rows, cols], axis=1)  # shape (12,2)
        E0 = initial_edges.shape[0]                      # == 12
        # diffs   = jnp.abs(initial_verts[:, None, :] - initial_verts[None, :, :]).sum(-1)
        # pairs   = jnp.argwhere(diffs == 2.)                 # (E⋆ , 2)
        # initial_edges = pairs[pairs[:, 0] < pairs[:, 1]]    # keep i < j
        # E0 = initial_edges.shape[0]

        # 3) pad out to fixed size
        verts = jnp.zeros((max_vertices, input_dim), dtype=float_dtype)
        verts = verts.at[:V0].set(initial_verts)

        eds = jnp.zeros((max_edges, 2), dtype=jnp.int32)
        eds = eds.at[:E0].set(initial_edges.astype(jnp.int32))

        # 4) counters and neuron-slots
        vc = jnp.array(V0, dtype=jnp.int32)
        ec = jnp.array(E0, dtype=jnp.int32)
        indices = jnp.full((activation_layer_count, layer_width), -1, dtype=jnp.int32)
        ranges = jnp.full((activation_layer_count, layer_width), jnp.inf, dtype=float_dtype)
        bp_count = jnp.array(0, dtype=jnp.int32)
        reference_idx = jnp.array(0, dtype=jnp.int32)

        return cls(
            vertices=verts,
            edges=eds,
            vertex_count=vc,
            edge_count=ec,
            indices=indices,
            bp_count=bp_count,
            ranges=ranges,
            reference_idx=reference_idx,
            layer_idx=jnp.array(0, dtype=jnp.int32),
            cell_split_count=jnp.array(0, dtype=jnp.int32)
        )

    @property
    def collapsable(self):
        return jnp.all(self.indices[self.layer_idx] != -1)
    
    def tree_flatten(self):
        # A tuple of "children" (dynamic) and "auxiliary data" (static)
        children = (self.vertices,
                    self.edges,
                    self.vertex_count,
                    self.edge_count,
                    self.indices,
                    self.ranges,
                    self.bp_count,
                    self.layer_idx,
                    self.reference_idx,
                    self.cell_split_count)
        return children, None                     # nothing static

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    def export(self, path: str):
        # export_polytopes_to_obj(polytope_cells, len(polytope_cells['id']), "experiments/sampled_cubes.obj")
        with open(path, "w") as f:
            # Write header information
            f.write("# Exported Polytope\n")
            
            # Define this polytope as a separate object
            f.write(f"o {path.split('/')[-1].split('.')[0]}\n")
            
            # Comment for this polytope
            f.write(f"# Polytope {0}\n")
            
            # Write out vertices
            for j in range(self.vertex_count):
                # Assuming v is a 3D coordinate
                v = self.vertices[j]
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            
            for j in range(self.edge_count):
                edge = self.edges[j]
                start_idx = edge[0] + 1
                end_idx = edge[1] + 1
                f.write("l {} {}\n".format(start_idx, end_idx))
            
            # Add a blank line between objects for readability
            f.write("\n")
        

def plot_activation(activation, x_range: Tuple[float, float] = (-2, 2), num_points: int = 1000):
    """Plots the FixedActivation function."""
    x_plot = jnp.linspace(x_range[0], x_range[1], num_points)
    y_plot = activation(x_plot) # Calls the activation function

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, 'b-', label='Activation function')
    plt.plot(activation.breakpoints, activation.values, 'ro', label='Breakpoints')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Fixed Activation Function')
    plt.legend()
    plt.show()