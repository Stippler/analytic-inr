"""
Edge-based spline initialization for neural spline learning.

This module extracts edges from 2D polygons or 3D meshes and initializes
Splines with exact surface values (SDF=0 on edges).
"""

import torch
from typing import Dict, Any
from .types import Splines


def initialize_edge_splines(data: Dict[str, Any]) -> Splines:
    """
    Initialize Splines from mesh edges with exact surface values.
    
    This function:
    1. Extracts edges from the data dict (2D polygons or 3D meshes)
    2. Computes normals for each edge
    3. Initializes Splines with:
       - start_points: edge start vertices
       - end_points: edge end vertices
       - normals: edge normals at both knot positions (perpendicular for 2D, bisector for 3D)
       - knots: [0, 1] for each edge (start and end)
       - values: [0, 0] for each edge (SDF=0 on surface)
    
    For 2D:
    - Normals are perpendicular to edges (90° CCW rotation)
    - Note: Normal direction is arbitrary unless edges are consistently ordered
    
    For 3D:
    - Uses equal-weighted edge-bisector normals (unit normals from adjacent faces)
    - Pure PyTorch implementation (no trimesh adjacency)
    - GPU-friendly and handles boundary/non-manifold edges naturally
    - Robustly ignores degenerate faces and edges with cancelling normals
    - Note: Requires consistent face winding for meaningful edge normals
    
    Parameters:
    -----------
    data : Dict[str, Any]
        Dictionary containing:
        - 'type': '2d' or '3d'
        - 'vertices': torch.Tensor of shape (N, 2) for 2d or (N, 3) for 3d
        - 'edges': torch.Tensor of shape (E, 2) [for 2d]
        - 'faces': torch.Tensor of shape (F, 3) [for 3d]
        - 'mesh': trimesh.Trimesh object (not used for edge extraction)
        
    Returns:
    --------
    Splines
        Initialized Splines object with edge-based data
    """
    data_type = data['type']
    vertices = data['vertices']
    device = vertices.device
    dtype = vertices.dtype
    
    print(f"\nInitializing edge-based splines for {data_type} data")
    
    if data_type == '2d':
        # For 2D, use provided edges
        edges = data['edges'].to(device=device, dtype=torch.long)
        print(f"  Vertices: {vertices.shape[0]}, Edges: {edges.shape[0]}")
        
        # Get edge start and end points (2D)
        start_points = vertices[edges[:, 0]]  # (E, 2)
        end_points = vertices[edges[:, 1]]    # (E, 2)
        
        # Compute edge vectors
        edge_vectors = end_points - start_points  # (E, 2)
        
        # Compute edge lengths
        edge_lengths = torch.norm(edge_vectors, dim=1)  # (E,)
        
        # Filter out zero-length edges
        valid_mask = edge_lengths > 1e-12
        if not valid_mask.all():
            num_removed = (~valid_mask).sum().item()
            print(f"  Warning: Removing {num_removed} zero-length edges")
            edges = edges[valid_mask]
            start_points = start_points[valid_mask]
            end_points = end_points[valid_mask]
            edge_vectors = edge_vectors[valid_mask]
            edge_lengths = edge_lengths[valid_mask]
        
        # For 2D edges, normal is perpendicular rotation 90° CCW
        # nx = -dy, ny = dx
        normals = torch.stack([-edge_vectors[:, 1], edge_vectors[:, 0]], dim=1)
        
        # Normalize to length 1
        normals = normals / edge_lengths.unsqueeze(1)
        
        # Repeat for both knots [0, 1] - shape: (E, 2, 2)
        # Edge normal is constant along the edge, so we use the same normal at both endpoints
        normals = normals.unsqueeze(1).expand(-1, 2, -1).contiguous()
        
    else:  # 3d
        # For 3D, extract edges from faces and compute equal-weighted edge-bisector normals
        # Pure PyTorch implementation - no trimesh adjacency needed
        
        faces_t = torch.as_tensor(data['faces'], device=device, dtype=torch.long)
        print(f"  Vertices: {vertices.shape[0]}, Faces: {faces_t.shape[0]}")
        
        # Compute face unit normals (equal weight per face, not area-weighted)
        v0 = vertices[faces_t[:, 0]]  # (F, 3)
        v1 = vertices[faces_t[:, 1]]  # (F, 3)
        v2 = vertices[faces_t[:, 2]]  # (F, 3)
        
        face_n_raw = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
        face_len = torch.linalg.norm(face_n_raw, dim=1, keepdim=True)  # (F, 1)
        
        # Robustly handle degenerate faces (ignore them rather than letting them inject NaNs/zeros)
        valid_face = face_len.squeeze(1) > 1e-12
        face_n = torch.where(valid_face.unsqueeze(1), face_n_raw / face_len, torch.zeros_like(face_n_raw))  # (F, 3)
        
        # Build all edges from faces (3 edges per triangle)
        e01 = faces_t[:, [0, 1]]
        e12 = faces_t[:, [1, 2]]
        e20 = faces_t[:, [2, 0]]
        e_all = torch.cat([e01, e12, e20], dim=0)  # (3F, 2)
        
        # Face index for each edge
        f_all = torch.arange(faces_t.shape[0], device=device, dtype=torch.long).repeat(3)  # (3F,)
        
        # Make edges undirected by sorting vertex indices
        e_sorted, _ = torch.sort(e_all, dim=1)
        
        # Get unique edges and inverse mapping
        edges, inv = torch.unique(e_sorted, dim=0, return_inverse=True)  # edges: (E, 2), inv: (3F,)
        
        print(f"  Edges: {edges.shape[0]}")
        
        # Get edge start and end points
        start_points = vertices[edges[:, 0]]  # (E, 3)
        end_points = vertices[edges[:, 1]]    # (E, 3)
        
        # Sum incident face unit normals per edge -> bisector
        n_sum = torch.zeros((edges.shape[0], 3), device=device, dtype=dtype)
        n_sum.index_add_(0, inv, face_n[f_all])
        
        # Normalize to get edge bisector normals
        # Handle near-zero sums (sharp creases with cancelling normals, inconsistent winding, etc.)
        n_len = torch.linalg.norm(n_sum, dim=1, keepdim=True)
        normals = torch.where(n_len > 1e-8, n_sum / n_len, torch.zeros_like(n_sum))
        
        # Repeat for both knots [0, 1] - shape: (E, 2, 3)
        # Edge bisector normal is constant along the edge, so we use the same normal at both endpoints
        normals = normals.unsqueeze(1).expand(-1, 2, -1).contiguous()
    
    num_edges = start_points.shape[0]
    print(f"  Final edge count: {num_edges}")
    
    # Initialize knots: [0, 1] for each edge (keep dtype consistent with vertices)
    knots = vertices.new_tensor([0.0, 1.0]).expand(num_edges, 2).clone()
    
    # Initialize values: [0, 0] for each edge (SDF=0 on surface)
    values = vertices.new_zeros((num_edges, 2))
    
    # Create Splines object
    splines = Splines(
        start_points=start_points,
        end_points=end_points,
        knots=knots,
        values=values,
        normals=normals,
        nconf=None,
        sconf=None,
        sign_uncertain=None
    )
    
    print(f"  ✓ Initialized Splines with {num_edges} edges")
    print(f"    - start_points: {splines.start_points.shape} ({data_type})")
    print(f"    - end_points: {splines.end_points.shape}")
    print(f"    - knots: {splines.knots.shape}")
    print(f"    - values: {splines.values.shape}")
    print(f"    - normals: {splines.normals.shape}")
    
    return splines
