"""
Geometric knot reduction using Numba JIT for fast CPU computation.

This module implements linearity-aware knot simplification:
- Linear fit test for face/edge regions
- Curvature-based sampling for vertex regions
- Hybrid Douglas-Peucker with adaptive tolerance
"""

import numpy as np
import torch
from numba import jit
from typing import List, Tuple
from tqdm import tqdm


@jit(nopython=True)
def point_line_distance(px: float, py: float, 
                        x1: float, y1: float, 
                        x2: float, y2: float) -> float:
    """
    Compute perpendicular distance from point to line segment in 2D.
    
    Args:
        px, py: Point coordinates
        x1, y1: Line start
        x2, y2: Line end
        
    Returns:
        Perpendicular distance
    """
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        # Degenerate segment
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Perpendicular distance formula
    num = abs(dy * px - dx * py + x2 * y1 - y2 * x1)
    denom = np.sqrt(dx**2 + dy**2)
    return num / denom


@jit(nopython=True)
def is_linear_region(t_vals: np.ndarray, sdf_vals: np.ndarray,
                     start_idx: int, end_idx: int,
                     tolerance: float = 1e-6) -> bool:
    """
    Test if a region is linear (constant gradient).
    
    In linear regions (closest feature is a face/edge), the SDF has
    constant gradient. We test if all intermediate points lie on the
    line between endpoints.
    
    Args:
        t_vals: Array of t parameters
        sdf_vals: Array of SDF values
        start_idx: Start index
        end_idx: End index (inclusive)
        tolerance: Maximum deviation for linearity
        
    Returns:
        True if region is linear
    """
    if end_idx - start_idx <= 1:
        return True
    
    # Line through start and end points
    t1, t2 = t_vals[start_idx], t_vals[end_idx]
    sdf1, sdf2 = sdf_vals[start_idx], sdf_vals[end_idx]
    
    # Check all intermediate points
    max_error = 0.0
    for i in range(start_idx + 1, end_idx):
        t = t_vals[i]
        sdf = sdf_vals[i]
        
        # Expected value from linear interpolation
        if abs(t2 - t1) > 1e-12:
            alpha = (t - t1) / (t2 - t1)
            sdf_expected = sdf1 + alpha * (sdf2 - sdf1)
            error = abs(sdf - sdf_expected)
            max_error = max(max_error, error)
    
    return max_error < tolerance


@jit(nopython=True)
def douglas_peucker_recursive(
    t_vals: np.ndarray,
    sdf_vals: np.ndarray,
    start_idx: int,
    end_idx: int,
    tolerance: float,
    linear_tolerance_factor: float = 10.0
) -> List[int]:
    """
    Recursive Douglas-Peucker with linearity-aware tolerance.
    
    Args:
        t_vals: Array of t parameters
        sdf_vals: Array of SDF values
        start_idx: Start index
        end_idx: End index
        tolerance: Base tolerance
        linear_tolerance_factor: Tolerance multiplier for linear regions
        
    Returns:
        List of indices to keep
    """
    if end_idx - start_idx <= 1:
        return [start_idx, end_idx]
    
    # Check if region is linear
    is_linear = is_linear_region(t_vals, sdf_vals, start_idx, end_idx, tolerance * 10)
    
    # Adjust tolerance based on linearity
    if is_linear:
        effective_tolerance = tolerance * linear_tolerance_factor
    else:
        effective_tolerance = tolerance
    
    # Find point with maximum distance
    max_dist = 0.0
    max_idx = start_idx
    
    t1, t2 = t_vals[start_idx], t_vals[end_idx]
    sdf1, sdf2 = sdf_vals[start_idx], sdf_vals[end_idx]
    
    for i in range(start_idx + 1, end_idx):
        # Compute perpendicular distance to line
        dist = point_line_distance(
            t_vals[i], sdf_vals[i],
            t1, sdf1,
            t2, sdf2
        )
        
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    # If max distance exceeds tolerance, split
    if max_dist > effective_tolerance:
        # Recursively simplify left and right
        left = douglas_peucker_recursive(
            t_vals, sdf_vals, start_idx, max_idx, 
            tolerance, linear_tolerance_factor
        )
        right = douglas_peucker_recursive(
            t_vals, sdf_vals, max_idx, end_idx,
            tolerance, linear_tolerance_factor
        )
        
        # Merge (remove duplicate at max_idx)
        result = left[:-1] + right
        return result
    else:
        # Keep only endpoints
        return [start_idx, end_idx]


@jit(nopython=True)
def reduce_knots_geometric_core(
    t_vals: np.ndarray,
    sdf_vals: np.ndarray,
    tolerance: float,
    linear_tolerance_factor: float = 10.0
) -> np.ndarray:
    """
    Core knot reduction algorithm (Numba-compiled).
    
    Applies hybrid Douglas-Peucker with linearity awareness.
    
    Args:
        t_vals: (N,) array of t parameters in [0, 1]
        sdf_vals: (N,) array of SDF values
        tolerance: Base simplification tolerance
        linear_tolerance_factor: How much to relax tolerance in linear regions
        
    Returns:
        (M,) array of indices to keep (M <= N)
    """
    if len(t_vals) <= 2:
        return np.arange(len(t_vals), dtype=np.int64)
    
    # Run Douglas-Peucker
    keep_indices = douglas_peucker_recursive(
        t_vals, sdf_vals, 0, len(t_vals) - 1,
        tolerance, linear_tolerance_factor
    )
    
    # Convert list to array and sort
    result = np.array(keep_indices, dtype=np.int64)
    result.sort()
    
    return result


def reduce_knots_geometric(
    t_vals: np.ndarray,
    sdf_vals: np.ndarray,
    tolerance: float = 0.005,
    linear_tolerance_factor: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce knots using geometric analysis (Python wrapper).
    
    Args:
        t_vals: (N,) array of t parameters
        sdf_vals: (N,) array of SDF values
        tolerance: Simplification tolerance
        linear_tolerance_factor: Tolerance multiplier for linear regions
        
    Returns:
        (t_reduced, sdf_reduced): Simplified arrays
    """
    # Convert to numpy if needed
    if isinstance(t_vals, torch.Tensor):
        t_vals = t_vals.cpu().numpy()
    if isinstance(sdf_vals, torch.Tensor):
        sdf_vals = sdf_vals.cpu().numpy()
    
    # Ensure contiguous arrays for Numba
    t_vals = np.ascontiguousarray(t_vals, dtype=np.float64)
    sdf_vals = np.ascontiguousarray(sdf_vals, dtype=np.float64)
    
    # Run core algorithm
    keep_indices = reduce_knots_geometric_core(
        t_vals, sdf_vals, tolerance, linear_tolerance_factor
    )
    
    # Extract simplified arrays
    t_reduced = t_vals[keep_indices]
    sdf_reduced = sdf_vals[keep_indices]
    
    return t_reduced, sdf_reduced


def simplify_sdf_to_knots_batch(
    t_values_list: List[torch.Tensor],
    sdf_values_list: List[torch.Tensor],
    tolerance: float = 0.005,
    linear_tolerance_factor: float = 10.0
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Batch simplification of multiple SDF curves.
    
    Args:
        t_values_list: List of (N_i,) tensors with t parameters
        sdf_values_list: List of (N_i,) tensors with SDF values
        tolerance: Simplification tolerance
        linear_tolerance_factor: Tolerance multiplier for linear regions
        
    Returns:
        knot_t_list: List of simplified t values
        knot_sdf_list: List of simplified SDF values
        max_errors: (B,) tensor of maximum errors per curve
        mean_errors: (B,) tensor of mean errors per curve
    """
    knot_t_list = []
    knot_sdf_list = []
    max_errors = []
    mean_errors = []
    
    for t_vals, sdf_vals in tqdm(
        zip(t_values_list, sdf_values_list),
        total=len(t_values_list),
        desc="Simplifying knots (Numba)",
        unit="segment"
    ):
        # Reduce knots
        t_reduced, sdf_reduced = reduce_knots_geometric(
            t_vals.cpu().numpy(),
            sdf_vals.cpu().numpy(),
            tolerance,
            linear_tolerance_factor
        )
        
        # Compute error metrics
        # Interpolate reduced curve back to original sampling
        t_orig = t_vals.cpu().numpy()
        sdf_orig = sdf_vals.cpu().numpy()
        sdf_interp = np.interp(t_orig, t_reduced, sdf_reduced)
        
        errors = np.abs(sdf_orig - sdf_interp)
        max_error = errors.max()
        mean_error = errors.mean()
        
        # Convert back to tensors
        knot_t_list.append(torch.from_numpy(t_reduced).float())
        knot_sdf_list.append(torch.from_numpy(sdf_reduced).float())
        max_errors.append(max_error)
        mean_errors.append(mean_error)
    
    return (
        knot_t_list,
        knot_sdf_list,
        torch.tensor(max_errors, dtype=torch.float32),
        torch.tensor(mean_errors, dtype=torch.float32)
    )


@jit(nopython=True)
def compute_curvature(t_vals: np.ndarray, sdf_vals: np.ndarray) -> np.ndarray:
    """
    Compute approximate curvature at each point.
    
    Uses finite differences to estimate second derivative.
    
    Args:
        t_vals: (N,) t parameters
        sdf_vals: (N,) SDF values
        
    Returns:
        (N,) curvature estimates
    """
    n = len(t_vals)
    curvature = np.zeros(n)
    
    for i in range(1, n - 1):
        # Central difference approximation of second derivative
        dt_left = t_vals[i] - t_vals[i-1]
        dt_right = t_vals[i+1] - t_vals[i]
        
        if dt_left > 1e-12 and dt_right > 1e-12:
            # First derivatives
            deriv_left = (sdf_vals[i] - sdf_vals[i-1]) / dt_left
            deriv_right = (sdf_vals[i+1] - sdf_vals[i]) / dt_right
            
            # Second derivative (curvature approximation)
            dt_avg = (dt_left + dt_right) / 2
            second_deriv = (deriv_right - deriv_left) / dt_avg
            
            curvature[i] = abs(second_deriv)
    
    # Boundary points get neighbor's curvature
    if n > 2:
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
    
    return curvature
