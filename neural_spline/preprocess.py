import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class AxisSegment:
    """Single PCA axis line segment with metrics."""
    start: np.ndarray
    end: np.ndarray
    variance: float
    n_left: int
    n_right: int
    axis_idx: int


@dataclass
class PCANode:
    """Node in recursive PCA tree."""
    center: np.ndarray
    axes: List[AxisSegment] = field(default_factory=list)
    children: List['PCANode'] = field(default_factory=list)
    depth: int = 0
    n_points: int = 0


def compute_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA. Returns (center, axes, variances)."""
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    return center, eigenvectors[:, order].T, eigenvalues[order]


def clip_line_to_halfspaces(origin: np.ndarray, direction: np.ndarray, 
                             planes: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    """
    Clip line to region defined by halfspace planes.
    Each plane is (normal, point) where normal points inward.
    Returns (t_min, t_max) or (inf, -inf) if no valid segment.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    t_min, t_max = -np.inf, np.inf
    
    for normal, point in planes:
        d_dot_n = np.dot(direction, normal)
        if abs(d_dot_n) > 1e-12:
            t = np.dot(point - origin, normal) / d_dot_n
            if d_dot_n > 0:
                t_min = max(t_min, t)
            else:
                t_max = min(t_max, t)
        elif np.dot(origin - point, normal) < -1e-12:
            return np.inf, -np.inf
    
    return (t_min, t_max) if t_min <= t_max else (np.inf, -np.inf)


def make_box_planes(bounds: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create inward-pointing halfspace planes for bounding box."""
    n_dims = bounds.shape[0]
    planes = []
    for i in range(n_dims):
        normal_neg = np.zeros(n_dims)
        normal_neg[i] = 1.0
        point_neg = bounds[i, 0] * normal_neg
        planes.append((normal_neg, point_neg))
        
        normal_pos = np.zeros(n_dims)
        normal_pos[i] = -1.0
        point_pos = bounds[i, 1] * normal_pos
        planes.append((normal_pos, point_pos))
    
    return planes


def recursive_pca(points: np.ndarray,
                  bounds: Optional[np.ndarray] = None,
                  planes: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                  min_points: int = 3,
                  min_variance: float = 1e-4,
                  max_depth: int = 4,
                  depth: int = 0) -> Optional[PCANode]:
    """
    Recursive PCA with boundary plane clipping.
    
    bounds: (n_dims, 2) array of [min, max] for each dimension
    planes: list of (normal, point) tuples defining halfspaces
    """
    n_dims = points.shape[1]
    
    if len(points) < min_points or depth >= max_depth:
        return None
    
    # Initialize bounds and planes if root
    if bounds is None:
        bounds = np.stack([points.min(axis=0), points.max(axis=0)], axis=1)
    if planes is None:
        planes = make_box_planes(bounds)
    
    center, pca_axes, variances = compute_pca(points)
    node = PCANode(center=center, depth=depth, n_points=len(points))
    
    # Process each PCA axis
    for axis_idx in range(n_dims):
        axis = pca_axes[axis_idx]
        variance = variances[axis_idx]
        
        if variance < min_variance:
            continue
        
        # Clip axis to boundary planes
        t_min, t_max = clip_line_to_halfspaces(center, axis, planes)
        if not np.isfinite(t_min) or not np.isfinite(t_max):
            continue
        
        start = center + t_min * axis
        end = center + t_max * axis
        
        projections = (points - center) @ axis
        n_left = (projections <= 0).sum()
        n_right = (projections >= 0).sum()
        
        node.axes.append(AxisSegment(
            start=start,
            end=end,
            variance=variance,
            n_left=n_left,
            n_right=n_right,
            axis_idx=axis_idx
        ))
    
    # Recurse into 2^n_dims subspaces
    projections = (points - center) @ pca_axes.T
    n_splits = 2 ** n_dims
    
    for split_idx in range(n_splits):
        mask = np.ones(len(points), dtype=bool)
        child_planes = list(planes)
        
        for axis_idx in range(n_dims):
            sign = 1 if (split_idx >> axis_idx) & 1 else -1
            axis_mask = (projections[:, axis_idx] * sign >= 0)
            mask &= axis_mask
            
            # Add splitting plane for this child
            child_planes.append((sign * pca_axes[axis_idx], center))
        
        subspace_points = points[mask]
        
        if len(subspace_points) >= min_points:
            child = recursive_pca(
                subspace_points,
                bounds=bounds,
                planes=child_planes,
                min_points=min_points,
                min_variance=min_variance,
                max_depth=max_depth,
                depth=depth + 1
            )
            if child is not None:
                node.children.append(child)
    
    return node


def flatten_tree(node: Optional[PCANode]) -> List[AxisSegment]:
    """Flatten PCA tree into list of all axis segments."""
    if node is None:
        return []
    
    segments = list(node.axes)
    for child in node.children:
        segments.extend(flatten_tree(child))
    
    return segments

