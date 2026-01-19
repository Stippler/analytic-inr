"""
Surface extraction methods for neural implicit representations.

This module provides the marching neurons extraction algorithm.
"""

from dataclasses import replace
import jax.numpy as jnp


def by_name(extractor_name, net):
    """Get an extractor by name and apply it to a network."""
    if extractor_name == "marching_neurons":
        return marching_neurons(net, triangulate=False)
    elif extractor_name == "marching_neurons_triangulated":
        return marching_neurons(net, triangulate=True)
    else:
        raise ValueError(f"Unknown extractor: {extractor_name}")


def marching_neurons(net, triangulate):
    """Extract surface using Marching Neurons algorithm (our method)."""
    from marching.runner import make_runner

    cached_net, cached_tri, runner = globals().get("_cached_mn", (None,) * 3)
    if cached_net is not net or cached_tri != triangulate:
        params = _collect_params(net)
        runner = make_runner(params, batch_size=2 ** 13, use_polygons=not triangulate)
        globals()["_cached_mn"] = net, triangulate, runner
    
    if triangulate:
        triangle_buffer, triangle_count = runner(jnp.array(1, dtype=jnp.int32))
        vertices = triangle_buffer[:triangle_count].reshape(-1, 3)
        faces = jnp.arange(triangle_count * 3).reshape(-1, 3)
        return vertices, faces
    else:
        vertex_buffer, vertex_count, face_sizes, face_count = runner(jnp.array(1, dtype=jnp.int32))
        return vertex_buffer[:vertex_count], jnp.arange(vertex_count), face_sizes[:face_count]


def _collect_params(net):
    """Convert PyTorch network to JAX parameters."""
    import jax.numpy as jnp
    import torch.nn as nn
    from marching.arch import Sine, SirenLinear
    
    params = {}
    idx = 0
    for module in net:
        if isinstance(module, nn.ReLU):
            params[f"{idx:04}.relu._"] = jnp.array([])
        elif isinstance(module, Sine):
            params[f"{idx:04}.sin._"] = jnp.array([])
        elif isinstance(module, nn.Linear):
            params[f"{idx:04}.dense.A"] = jnp.from_dlpack(module.weight.detach()).T
            params[f"{idx:04}.dense.b"] = jnp.from_dlpack(module.bias.detach())
        elif isinstance(module, SirenLinear):
            params[f"{idx:04}.dense.A"] = jnp.from_dlpack(module.omega_0 * module.linear.weight.detach()).T
            params[f"{idx:04}.dense.b"] = jnp.from_dlpack(module.omega_0 * module.linear.bias.detach())
        else:
            raise ValueError(f"Unknown module: {module}")
        idx += 1
    params[f"{idx:04}.squeeze_last._"] = jnp.array([])
    return params


