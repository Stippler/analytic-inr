"""
Spline dataclass for representing 1D line segments with ground truth and predicted values.
"""

import torch
from dataclasses import dataclass


@dataclass
class Spline:
    """Represents a 1D spline along a line segment."""
    start_point: torch.Tensor  # (2,) or (3,) start point in space
    end_point: torch.Tensor    # (2,) or (3,) end point in space
    
    pred_knots: torch.Tensor   # (N,) predicted knot positions along [0, 1]
    pred_values: torch.Tensor  # (N,) predicted SDF values at knots
    
    gt_knots: torch.Tensor     # (M,) ground truth knot positions
    gt_values: torch.Tensor    # (M,) ground truth SDF values
    
    # Metadata for visualization and analysis
    label: str = ""
    depth: int = 0
    pc_type: str = "pc1"

