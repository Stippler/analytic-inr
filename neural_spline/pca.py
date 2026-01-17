import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PCAComponent:
    """Single PCA component (line segment) with metadata."""
    start: torch.Tensor  # (D,)
    end: torch.Tensor    # (D,)
    variance: float
    component_idx: int  # Which PC this is (0=PC1, 1=PC2, etc.)
    depth: int
    label: str


@dataclass
class PCANode:
    """Node in recursive PCA tree."""
    center: torch.Tensor
    components: List[PCAComponent] = field(default_factory=list)
    children: List['PCANode'] = field(default_factory=list)
    depth: int = 0
    n_points: int = 0


def compute_pca(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PCA for any dimension using PyTorch.
    
    Args:
        points: (N, D) tensor where D is 2 or 3
    
    Returns:
        center: (D,) centroid
        axes: (D, D) principal axes (rows), sorted by variance
        variances: (D,) variance along each axis
    """
    center = points.mean(dim=0)
    centered = points - center
    cov = (centered.T @ centered) / len(points)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort by variance (descending)
    order = torch.argsort(eigenvalues, descending=True)
    return center, eigenvectors[:, order].T, eigenvalues[order]


def clip_line_to_box(origin: torch.Tensor, direction: torch.Tensor, 
                     bbox_min: float = -1.0, bbox_max: float = 1.0) -> Tuple[Optional[float], Optional[float]]:
    """
    Clip line to bounding box.
    
    Returns:
        (t_min, t_max) parameters or (None, None) if no intersection
    """
    direction = direction / (torch.norm(direction) + 1e-12)
    t_min, t_max = float('-inf'), float('inf')
    
    for i in range(len(origin)):
        if abs(direction[i].item()) > 1e-12:
            t1 = (bbox_min - origin[i].item()) / direction[i].item()
            t2 = (bbox_max - origin[i].item()) / direction[i].item()
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        elif origin[i].item() < bbox_min or origin[i].item() > bbox_max:
            return None, None
    
    return (t_min, t_max) if t_min <= t_max else (None, None)


def recursive_pca_decomposition(
    points: torch.Tensor,
    min_points: int = 3,
    max_depth: int = 3,
    depth: int = 0,
    label_prefix: str = ""
) -> Optional[PCANode]:
    """
    Recursively decompose point cloud using PCA.
    Works for both 2D and 3D data.
    
    Args:
        points: (N, D) tensor where D is 2 or 3
        min_points: Minimum points required to continue recursion (default: 3)
        max_depth: Maximum recursion depth
        depth: Current depth
        label_prefix: Label prefix for tracking hierarchy
    
    Returns:
        PCANode with components and children, or None if insufficient points
    """
    # Early return if not enough points or max depth reached
    if len(points) < min_points or depth >= max_depth:
        return None
    
    # Compute PCA
    center, axes, variances = compute_pca(points)
    n_dims = points.shape[1]
    
    # Create node
    node = PCANode(center=center, depth=depth, n_points=len(points))
    
    # Always add all components (PC1, PC2 for 2D or PC1, PC2, PC3 for 3D)
    for i in range(n_dims):
        # Clip to bounding box
        t_min, t_max = clip_line_to_box(center, axes[i])
        if t_min is None:
            continue
        
        # Create component
        pc_name = f"pc{i+1}"
        label = f"D{depth}:{label_prefix}{pc_name}" if label_prefix else f"D{depth}:{pc_name}"
        
        component = PCAComponent(
            start=center + t_min * axes[i],
            end=center + t_max * axes[i],
            variance=variances[i].item(),
            component_idx=i,
            depth=depth,
            label=label
        )
        node.components.append(component)
    
    # Recursively subdivide into quadrants (2D) or octants (3D)
    # Project points onto principal axes
    projections = (points - center) @ axes.T  # (N, D)
    
    # Generate all sign combinations for subdivision
    n_subdivisions = 2 ** n_dims  # 4 for 2D, 8 for 3D
    
    for subdiv_idx in range(n_subdivisions):
        # Create mask for this subdivision
        mask = torch.ones(len(points), dtype=torch.bool, device=points.device)
        subdiv_label = ""
        
        for dim in range(n_dims):
            # Extract bit for this dimension
            sign = 1 if (subdiv_idx >> dim) & 1 else -1
            
            # If a point is exactly on the plane (projection == 0), include it in both
            # This is done by using >= 0 for positive side and <= 0 for negative side
            if sign > 0:
                dim_mask = (projections[:, dim] >= 0)  # Include points on plane
            else:
                dim_mask = (projections[:, dim] <= 0)  # Include points on plane
            
            mask &= dim_mask
            subdiv_label += '+' if sign > 0 else '-'
        
        # Recurse if enough points
        subpoints = points[mask]
        if len(subpoints) >= min_points:
            child = recursive_pca_decomposition(
                subpoints,
                min_points=min_points,
                max_depth=max_depth,
                depth=depth + 1,
                label_prefix=f"{label_prefix}{subdiv_label}/"
            )
            if child is not None:
                node.children.append(child)
    
    return node


def flatten_pca_tree(node: Optional[PCANode]) -> List[PCAComponent]:
    """
    Flatten PCA tree into a list of all components.
    
    Args:
        node: Root PCANode
    
    Returns:
        List of all PCAComponent objects in the tree
    """
    if node is None:
        return []
    
    components = list(node.components)
    for child in node.children:
        components.extend(flatten_pca_tree(child))
    
    return components
