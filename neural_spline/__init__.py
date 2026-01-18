"""
Neural Spline Learning - High-Performance Version

Optimized with NVIDIA Warp, Numba, and PyTorch 2.x compile.
"""

# Legacy imports (for backward compatibility)
# from .model import ReluMLP as ReluMLP_legacy
# from .spline import Spline as Spline_legacy
# from .train import train_model, compute_metrics
# from .train_fast import train_model_fast, update_spline_predictions as update_spline_predictions_legacy

# New optimized modules
from .network import ReluMLP, create_mlp
from .types import Spline, PCAComponent, PCANode, ConvexPolytope
from .geometry import constrained_recursive_pca, flatten_pca_tree
from .spline import compute_splines
from .simplification import simplify_sdf_to_knots_batch, reduce_knots_geometric
from .train_optimized import train_model_optimized, update_spline_predictions

__all__ = [
    # New optimized modules (preferred)
    'ReluMLP',
    'create_mlp',
    'Spline',
    'PCAComponent',
    'PCANode',
    'ConvexPolytope',
    'constrained_recursive_pca',
    'flatten_pca_tree',
    'compute_sdf',
    'simplify_sdf_to_knots_batch',
    'reduce_knots_geometric',
    'train_model_optimized',
    'update_spline_predictions',
    # Legacy compatibility
    # 'ReluMLP_legacy',
    # 'Spline_legacy',
    # 'train_model',
    # 'train_model_fast',
    # 'update_spline_predictions_legacy',
    # 'compute_metrics',
]

