import inspect
from itertools import chain

import torch
from torch import nn
# import jax.numpy as jnp
import numpy as np
def by_name(arch_name):
    for i in range(arch_name.count("_") + 1):
        kind, *args = arch_name.rsplit("_", i)
        if (constr := globals().get(kind, None)) is not None:
            sig = inspect.signature(constr)
            req_keys = []
            for param_name, param in sig.parameters.items():
                # Skip parameters with default values
                if param.default == inspect.Parameter.empty:
                    req_keys.append(param_name[:1])
            got_keys = [arg[:1] for arg in args]
            
            # Check if we have all required arguments
            if all(key in got_keys for key in req_keys):
                # Convert arguments and use defaults for missing optional params
                arg_values = []
                for arg in args:
                    # Special case for input dimension: i2 -> input_dim=2, i3 -> input_dim=3
                    if arg.startswith('i'):
                        arg_values.append(int(arg[1:]))  # input_dim value
                    else:
                        arg_values.append(int(arg[1:]))
                return constr(*arg_values)
            
            # If we're missing required args, continue trying other splits
            continue
    raise ValueError(f"Unknown architecture: {arch_name}")


def relu_mlp(depth, width, input_dim=3):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        nn.ReLU(),
        *chain.from_iterable([nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 2)),
        nn.Linear(width, 1)
    )


def geo_relu_mlp(depth, width, input_dim=3):
    net = relu_mlp(depth, width, input_dim)
    for i, module in enumerate(net):
        if isinstance(module, nn.Linear):
            if i == len(net) - 1:
                nn.init.normal_(module.weight, mean=(torch.pi / module.in_features) ** 0.5, std=0.00001)
                nn.init.constant_(module.bias, -1)
            else:
                nn.init.normal_(module.weight, 0.0, (2 / module.out_features) ** 0.5)
                nn.init.zeros_(module.bias)
    return net


def pe_relu_mlp(depth, width, freq, input_dim=3):
    return nn.Sequential(
        *fourier_encoding(input_dim, width, axis_aligned=True, max_scale=freq),
        *chain.from_iterable([nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 1)),
        nn.Linear(width, 1)
    )


def pe30_relu_mlp(depth, width, input_dim=3):
    return nn.Sequential(
        *fourier_encoding(input_dim, width, axis_aligned=True, max_scale=30),
        *chain.from_iterable([nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 1)),
        nn.Linear(width, 1)
    )


def pe50_relu_mlp(depth, width, input_dim=3):
    return nn.Sequential(
        *fourier_encoding(input_dim, width, axis_aligned=True, max_scale=50),
        *chain.from_iterable([nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 1)),
        nn.Linear(width, 1)
    )


def pe70_relu_mlp(depth, width, input_dim=3):
    return nn.Sequential(
        *fourier_encoding(input_dim, width, axis_aligned=True, max_scale=70),
        *chain.from_iterable([nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 1)),
        nn.Linear(width, 1)
    )


def pe100_relu_mlp(depth, width, input_dim=3):
    return nn.Sequential(
        *fourier_encoding(input_dim, width, axis_aligned=True, max_scale=100),
        *chain.from_iterable([nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 1)),
        nn.Linear(width, 1)
    )


def pe1000_relu_mlp(depth, width, input_dim=3):
    return nn.Sequential(
        *fourier_encoding(input_dim, width, axis_aligned=True, max_scale=1000),
        *chain.from_iterable([nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 1)),
        nn.Linear(width, 1)
    )


def pe10000_relu_mlp(depth, width, input_dim=3):
    return nn.Sequential(
        *fourier_encoding(input_dim, width, axis_aligned=True, max_scale=10_000),
        *chain.from_iterable([nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 1)),
        nn.Linear(width, 1)
    )


def grff_relu_mlp(depth, width, freq, input_dim=3):
    return nn.Sequential(
        *fourier_encoding(input_dim, width, axis_aligned=False, max_scale=freq),
        *chain.from_iterable([nn.Linear(width, width), nn.ReLU()] for _ in range(depth - 1)),
        nn.Linear(width, 1)
    )


def siren(depth, width, input_dim=3):
    return nn.Sequential(
        SirenLinear(input_dim, width, True),
        Sine(),
        *chain.from_iterable([SirenLinear(width, width, False), Sine()] for _ in range(depth - 2)),
        SirenLinear(width, 1, False).linear
    )


def fourier_encoding(in_features, out_features, axis_aligned, max_scale):
    """
    Positional Encoding & Gaussian Random Fourier Features encoding from:
    M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron,
    R. Ng. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains."
    Advances in Neural Information Processing Systems (NeurIPS). 2020.
    """

    linear = nn.Linear(in_features, out_features)
    linear.weight.requires_grad = False
    linear.bias.requires_grad = False
    if axis_aligned:
        linear.weight.zero_()
        freqs = out_features // 2
        i = 0
        for c, n in enumerate(freqs // in_features + (torch.arange(in_features) < freqs % in_features)):
            linear.weight[i * 2:(i + n) * 2:2, c] = torch.pi * torch.logspace(0, 1, n, max_scale)
            i += n
    else:
        linear.weight[0::2].normal_(std=torch.pi * max_scale / 3)
    linear.weight[1::2] = linear.weight[0::2]
    linear.bias[0::2] = torch.pi / 2
    linear.bias[1::2] = 0
    return linear, Sine()

class Sin(nn.Module):
    def forward(self, X):
        return X.sin()

class Sine(nn.Module):

    def forward(self, X):
        return X.sin()


class SirenLinear(nn.Module):

    def __init__(self, in_features, out_features, is_first):
        super().__init__()
        self.omega_0 = 30
        self.linear = nn.Linear(in_features, out_features)
        bound = (1 / in_features) if is_first else ((6 / in_features) ** 0.5 / self.omega_0)
        nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, X):
        return self.omega_0 * self.linear(X)

# def load_pt(input_pt, arch_name):
#     net = by_name(arch_name)
#     net.load_state_dict(torch.load(input_pt, weights_only=True))

#     params = {}
#     idx = 0
#     for module in net.modules():
#         if isinstance(module, nn.ReLU):
#             params[f"{idx:04}.relu._"] = jnp.array([])
#         elif isinstance(module, Sine):
#             params[f"{idx:04}.sin._"] = jnp.array([])
#         elif isinstance(module, nn.Linear):
#             params[f"{idx:04}.dense.A"] = jnp.array(module.weight.T.numpy(force=True))
#             params[f"{idx:04}.dense.b"] = jnp.array(module.bias.numpy(force=True))
#         elif isinstance(module, SirenLinear):
#             params[f"{idx:04}.dense.A"] = jnp.array(module.omega_0 * module.linear.weight.T.numpy(force=True))
#             params[f"{idx:04}.dense.b"] = jnp.array(module.omega_0 * module.linear.bias.numpy(force=True))
#         elif isinstance(module, nn.Sequential):
#             continue
#         else:
#             raise ValueError(f"Unknown module: {module}")
#         idx += 1
#     params[f"{idx:04}.squeeze_last._"] = jnp.array([])
#     # params = {k: v.T for k, v in params.items()}
#     return params
    
# def load_npz(filename):
#     out_params = {}
#     param_count = 0
#     with np.load(filename) as data:
#         for key,val in data.items():
#             # print(f"mlp layer key: {key}")
#             # convert numpy to jax arrays
#             if isinstance(val, np.ndarray):
#                 param_count += val.size
#                 val = jnp.array(val)
#             out_params[key] = val
#     print(f"Loaded MLP with {param_count} params")
#     return out_params