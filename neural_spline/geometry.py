import torch
from typing import List, Tuple, Optional
from .types import PCAComponent, PCANode, ConvexPolytope, create_bounding_box_polytope


def compute_pca(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PCA with stable covariance computation.
    
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


def orient_eigenvector_by_mass(points: torch.Tensor, center: torch.Tensor, 
                               eigenvector: torch.Tensor) -> torch.Tensor:
    """
    Orient eigenvector towards the bulk of the data distribution.
    
    Projects points onto the eigenvector and ensures it points towards
    the majority of the data (more than half of points have positive projection).
    
    Args:
        points: (N, D) point cloud
        center: (D,) centroid
        eigenvector: (D,) principal axis to orient
        
    Returns:
        (D,) Oriented eigenvector (possibly flipped)
    """
    centered = points - center
    projections = centered @ eigenvector
    
    n_positive = (projections > 0).sum()
    n_total = points.shape[0]
    
    if n_positive < n_total / 2:
        return -eigenvector
    return eigenvector


def check_eigenvalue_degeneracy(eigenvalues: torch.Tensor, 
                               threshold: float = 0.1) -> List[int]:
    """
    Identify degenerate eigenvalues (nearly equal).
    
    Args:
        eigenvalues: (D,) sorted eigenvalues (descending)
        threshold: Relative threshold for considering eigenvalues close
        
    Returns:
        List of indices where degeneracy occurs
    """
    degenerate_pairs = []
    for i in range(len(eigenvalues) - 1):
        # Check relative difference
        if eigenvalues[i] > 1e-9:  # Avoid division by very small numbers
            rel_diff = abs(eigenvalues[i] - eigenvalues[i+1]) / eigenvalues[i]
            if rel_diff < threshold:
                degenerate_pairs.append(i)
    return degenerate_pairs


def resolve_degeneracy(axes: torch.Tensor, eigenvalues: torch.Tensor,
                       parent_axes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Resolve degenerate eigenvectors by aligning with parent or global axes.
    
    When eigenvalues are very close, the eigenvector directions can be arbitrary.
    We prefer axes that align with the parent's axes or global coordinate axes.
    
    Args:
        axes: (D, D) current principal axes
        eigenvalues: (D,) eigenvalues
        parent_axes: Optional (D, D) parent's principal axes
        
    Returns:
        (D, D) Resolved axes
    """
    degenerate = check_eigenvalue_degeneracy(eigenvalues)
    
    if not degenerate:
        return axes
    
    # For degenerate pairs, try to align with parent or global axes
    resolved_axes = axes.clone()
    
    for deg_idx in degenerate:
        # We have a degenerate pair at indices deg_idx and deg_idx+1
        # Try to align these with parent axes or global axes
        
        if parent_axes is not None:
            # Align with parent's axes
            # Find which parent axis is most aligned with this degenerate subspace
            for parent_idx in range(parent_axes.shape[0]):
                parent_axis = parent_axes[parent_idx]
                
                # Check alignment with both degenerate axes
                align_0 = abs(torch.dot(axes[deg_idx], parent_axis))
                align_1 = abs(torch.dot(axes[deg_idx + 1], parent_axis))
                
                # If strongly aligned, use parent axis
                if align_0 > 0.7 or align_1 > 0.7:
                    # Use parent axis as one of the degenerate axes
                    resolved_axes[deg_idx] = parent_axis
                    break
        else:
            # Align with global coordinate axes
            dim = axes.shape[0]
            for axis_idx in range(dim):
                global_axis = torch.zeros(dim, device=axes.device)
                global_axis[axis_idx] = 1.0
                
                # Check alignment
                align_0 = abs(torch.dot(axes[deg_idx], global_axis))
                align_1 = abs(torch.dot(axes[deg_idx + 1], global_axis))
                
                if align_0 > 0.7 or align_1 > 0.7:
                    resolved_axes[deg_idx] = global_axis
                    break
    
    return resolved_axes


def constrained_recursive_pca(
    points: torch.Tensor,
    min_points: int = 3,
    max_depth: int = 3,
    depth: int = 0,
    label_prefix: str = "",
    polytope: Optional[ConvexPolytope] = None,
    parent_axes: Optional[torch.Tensor] = None
) -> Optional[PCANode]:
    """
    Recursively decompose point cloud using constrained PCA with slab method.
    
    This implements robust recursive PCA with:
    - Mass-based eigenvector orientation for stability
    - Degeneracy handling for near-symmetric regions
    - Convex polytope tracking for accurate clipping
    
    Args:
        points: (N, D) tensor where D is 2 or 3
        min_points: Minimum points required to continue recursion
        max_depth: Maximum recursion depth
        depth: Current depth
        label_prefix: Label prefix for tracking hierarchy
        polytope: Current convex polytope (None = use bounding box)
        parent_axes: Parent's principal axes for degeneracy resolution
    
    Returns:
        PCANode with components and children, or None if insufficient points
    """
    # Early return if not enough points or max depth reached
    if len(points) < min_points or depth >= max_depth:
        return None
    
    # Initialize polytope if not provided (root node)
    if polytope is None:
        dim = points.shape[1]
        polytope = create_bounding_box_polytope(
            bbox_min=-1.0, bbox_max=1.0, dimension=dim
        )
    
    # Compute PCA
    center, axes, variances = compute_pca(points)
    n_dims = points.shape[1]
    
    # Apply mass-based orientation to all axes (vectorized)
    centered = points - center  # (N, D)
    projections = centered @ axes.T  # (N, n_dims)
    n_positive = (projections > 0).sum(dim=0)  # (n_dims,)
    n_total = points.shape[0]
    flip = n_positive < n_total / 2  # (n_dims,)
    axes = torch.where(flip[:, None], -axes, axes)  # Flip entire rows where needed
    
    # Resolve degeneracy if present (DISABLED - not needed for neural network training)
    # axes = resolve_degeneracy(axes, variances, parent_axes)
    
    # Create node
    node = PCANode(
        center=center, 
        depth=depth, 
        n_points=len(points),
        polytope=polytope
    )
    
    # Create components by clipping to polytope
    for i in range(n_dims):
        # Clip principal axis ray against the current polytope
        t_min, t_max = polytope.clip_ray(center, axes[i])
        
        if t_min is None:
            # No intersection with polytope (shouldn't happen, but handle gracefully)
            continue
        
        # Create component with clipped endpoints
        pc_name = f"pc{i+1}"
        label = f"D{depth}:{label_prefix}{pc_name}" if label_prefix else f"D{depth}:{pc_name}"
        
        component = PCAComponent(
            start=center + t_min * axes[i],
            end=center + t_max * axes[i],
            variance=variances[i],
            component_idx=i,
            depth=depth,
            label=label,
            polytope=polytope,
            parent_axis=axes[i] if i == 0 else None
        )
        node.components.append(component)
    
    # Recursively subdivide using splitting planes
    # Project points onto principal axes
    projections = (points - center) @ axes.T  # (N, D)
    
    # Exclude points that lie on or very close to any splitting plane
    # These points are already captured by the parent components
    eps = 1e-6
    near_plane = torch.any(torch.abs(projections) < eps, dim=1)  # (N,)
    
    # Generate all sign combinations for subdivision (2^D octants/quadrants)
    n_subdivisions = 2 ** n_dims  # 4 for 2D, 8 for 3D
    
    for subdiv_idx in range(n_subdivisions):
        # Create mask for this subdivision, starting with points NOT near any plane
        mask = ~near_plane
        subdiv_label = ""
        
        # Create child polytope by adding splitting plane constraints
        child_polytope = polytope
        
        for dim in range(n_dims):
            # Extract bit for this dimension
            sign = 1 if (subdiv_idx >> dim) & 1 else -1
            
            if sign > 0:
                dim_mask = (projections[:, dim] > 0)
                subdiv_label += '+'
                normal = -axes[dim]
                offset = -torch.dot(axes[dim], center)
            else:
                dim_mask = (projections[:, dim] < 0)
                subdiv_label += '-'
                normal = axes[dim]
                offset = torch.dot(axes[dim], center)
            
            mask &= dim_mask
            child_polytope = child_polytope.add_constraint(normal, offset)
        
        # Recurse if enough points
        subpoints = points[mask]
        if len(subpoints) >= min_points:
            child = constrained_recursive_pca(
                subpoints,
                min_points=min_points,
                max_depth=max_depth,
                depth=depth + 1,
                label_prefix=f"{label_prefix}{subdiv_label}/",
                polytope=child_polytope,
                parent_axes=axes
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