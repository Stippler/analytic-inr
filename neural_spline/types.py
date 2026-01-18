"""
Shared type definitions for high-performance neural spline learning.

This module provides type-safe dataclasses used across the pipeline:
- ConvexPolytope: Represents convex regions for PCA clipping
- PCAComponent: Enhanced PCA segment with constraint tracking
- Spline: Unified spline representation for training
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ConvexPolytope:
    """
    Represents a convex polytope (convex region) defined by half-spaces.
    
    Each half-space is defined as: normal · x <= offset
    Used for tracking Voronoi-like cells during recursive PCA.
    
    Attributes:
        normals: (N, D) tensor of outward-pointing normal vectors
        offsets: (N,) tensor of offset values for each half-space
        dimension: Dimensionality of the space (2 or 3)
    """
    normals: torch.Tensor  # (N, D)
    offsets: torch.Tensor  # (N,)
    dimension: int = 3
    
    def __post_init__(self):
        """Validate dimensions after initialization."""
        assert self.normals.shape[0] == self.offsets.shape[0], \
            "Number of normals must match number of offsets"
        assert self.normals.shape[1] == self.dimension, \
            f"Normal dimension {self.normals.shape[1]} must match polytope dimension {self.dimension}"
    
    def add_constraint(self, normal: torch.Tensor, offset: float) -> 'ConvexPolytope':
        """
        Add a new half-space constraint to the polytope.
        
        Args:
            normal: (D,) normal vector for the new half-space
            offset: Scalar offset value
            
        Returns:
            New ConvexPolytope with added constraint
        """
        new_normals = torch.cat([self.normals, normal.unsqueeze(0)], dim=0)
        new_offsets = torch.cat([self.offsets, torch.tensor([offset], device=self.offsets.device)])
        return ConvexPolytope(new_normals, new_offsets, self.dimension)
    
    def clip_ray(self, origin: torch.Tensor, direction: torch.Tensor) -> tuple[Optional[float], Optional[float]]:
        """
        Clip a ray against the polytope to find entry/exit points.
        
        Uses the Cyrus-Beck clipping algorithm.
        
        Args:
            origin: (D,) ray origin point
            direction: (D,) ray direction (need not be normalized)
            
        Returns:
            (t_min, t_max): Parameter values where ray enters/exits polytope
                           Returns (None, None) if no intersection
        """
        # Normalize direction
        dir_norm = torch.norm(direction)
        if dir_norm < 1e-12:
            return None, None
        direction = direction / dir_norm
        
        t_min = float('-inf')
        t_max = float('inf')
        
        # For each half-space: normal · x <= offset
        for i in range(len(self.normals)):
            n = self.normals[i]
            d = self.offsets[i]
            
            # Compute: normal · direction
            denom = torch.dot(n, direction).item()
            
            # Compute: offset - normal · origin
            numer = d - torch.dot(n, origin).item()
            
            if abs(denom) < 1e-12:
                # Ray is parallel to plane
                if numer < 0:
                    # Ray origin is outside this half-space
                    return None, None
                # Otherwise, continue (ray is inside or on boundary)
            else:
                # Ray intersects plane at t = numer / denom
                t = numer / denom
                
                if denom < 0:
                    # Ray is entering this half-space
                    t_min = max(t_min, t)
                else:
                    # Ray is exiting this half-space
                    t_max = min(t_max, t)
        
        # Check if interval is valid
        if t_min > t_max or t_max < 0:
            return None, None
        
        return t_min, t_max


@dataclass
class PCAComponent:
    """
    Represents a single PCA component (line segment) with metadata.
    
    Enhanced version with constraint tracking for robust recursive PCA.
    
    Attributes:
        start: (D,) Start point of segment in space
        end: (D,) End point of segment in space
        variance: Variance along this principal component
        component_idx: Which PC this is (0=PC1, 1=PC2, 2=PC3)
        depth: Recursion depth in PCA tree
        label: Human-readable label for tracking hierarchy
        polytope: Optional convex polytope representing valid region
        parent_axis: Optional parent's principal axis for degeneracy handling
    """
    start: torch.Tensor  # (D,)
    end: torch.Tensor    # (D,)
    variance: float
    component_idx: int
    depth: int
    label: str
    polytope: Optional[ConvexPolytope] = None
    parent_axis: Optional[torch.Tensor] = None
    
    @property
    def direction(self) -> torch.Tensor:
        """Get normalized direction vector."""
        d = self.end - self.start
        return d / (torch.norm(d) + 1e-12)
    
    @property
    def length(self) -> float:
        """Get length of segment."""
        return torch.norm(self.end - self.start).item()
    
    @property
    def midpoint(self) -> torch.Tensor:
        """Get midpoint of segment."""
        return (self.start + self.end) / 2.0


@dataclass
class Spline:
    """
    Represents a 1D spline along a line segment with ground truth and predictions.
    
    Used for training the neural network to approximate SDF along PCA components.
    
    Attributes:
        start_point: (D,) Start point in space (D=2 or 3)
        end_point: (D,) End point in space
        gt_knots: (M,) Ground truth knot positions in [0, 1]
        gt_values: (M,) Ground truth SDF values at knots
        normals: (M,D) Surface normals at gt_knots (optional)
        pred_knots: (N,) Predicted knot positions (populated during training)
        pred_values: (N,) Predicted SDF values at knots
        label: Human-readable label for visualization
        depth: PCA recursion depth
        pc_type: Principal component type (pc1, pc2, pc3)
        component_idx: Which PC this is (0, 1, 2)
    """
    start_point: torch.Tensor      # (D,)
    end_point: torch.Tensor        # (D,)
    gt_knots: torch.Tensor         # (M,)
    gt_values: torch.Tensor        # (M,)
    normals: Optional[torch.Tensor] = None      # (M,D) - normals at gt_knots
    pred_knots: Optional[torch.Tensor] = None   # (N,) - filled during training
    pred_values: Optional[torch.Tensor] = None  # (N,)
    label: str = ""
    depth: int = 0
    pc_type: str = "pc1"
    component_idx: int = 0
    
    @property
    def direction(self) -> torch.Tensor:
        """Get normalized direction vector."""
        d = self.end_point - self.start_point
        return d / (torch.norm(d) + 1e-12)
    
    @property
    def length(self) -> float:
        """Get length of segment."""
        return torch.norm(self.end_point - self.start_point).item()
    
    def has_predictions(self) -> bool:
        """Check if predictions have been computed."""
        return self.pred_knots is not None and self.pred_values is not None


@dataclass
class PCANode:
    """
    Node in recursive PCA tree.
    
    Internal structure used during PCA decomposition.
    
    Attributes:
        center: (D,) Centroid of points in this node
        components: List of PCAComponent objects at this node
        children: List of child PCANode objects
        depth: Recursion depth
        n_points: Number of points in this node
        polytope: Convex polytope representing this node's region
    """
    center: torch.Tensor
    components: List[PCAComponent] = field(default_factory=list)
    children: List['PCANode'] = field(default_factory=list)
    depth: int = 0
    n_points: int = 0
    polytope: Optional[ConvexPolytope] = None


def create_bounding_box_polytope(bbox_min: float = -1.0, bbox_max: float = 1.0, 
                                  dimension: int = 3, device: torch.device = torch.device("cuda")) -> ConvexPolytope:
    """
    Create a ConvexPolytope representing an axis-aligned bounding box.
    
    Args:
        bbox_min: Minimum coordinate value
        bbox_max: Maximum coordinate value
        dimension: Number of dimensions (2 or 3)
        
    Returns:
        ConvexPolytope representing the bounding box
    """
    # For each axis, we need two constraints: x >= min and x <= max
    # Rewrite as: -x <= -min and x <= max
    normals_list = []
    offsets_list = []
    
    for i in range(dimension):
        # Lower bound: -x_i <= -bbox_min  =>  x_i >= bbox_min
        normal_lower = torch.zeros(dimension, device=device)
        normal_lower[i] = -1.0
        normals_list.append(normal_lower)
        offsets_list.append(-bbox_min)
        
        # Upper bound: x_i <= bbox_max
        normal_upper = torch.zeros(dimension, device=device)
        normal_upper[i] = 1.0
        normals_list.append(normal_upper)
        offsets_list.append(bbox_max)
    
    normals = torch.stack(normals_list, dim=0)  # (2*D, D)
    offsets = torch.tensor(offsets_list, device=device)  # (2*D,)
    
    return ConvexPolytope(normals, offsets, dimension)


@dataclass
class Splines:
    start_points: torch.Tensor
    end_points: torch.Tensor

    knots: torch.Tensor
    values: torch.Tensor

    normals: torch.Tensor
