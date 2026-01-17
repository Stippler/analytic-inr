import torch
import numpy as np
from typing import Tuple, List, Optional, Union
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
from pytorch3d.loss import point_mesh_distance
from tqdm import tqdm


def _ray_triangle_intersection_batch(
    ray_origins: torch.Tensor,      # (B, 3)
    ray_directions: torch.Tensor,   # (B, 3)
    triangle_verts: torch.Tensor,   # (F, 3, 3)
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized Möller-Trumbore ray-triangle intersection.
    
    Returns:
        hit_ray_idx: Indices of rays that hit triangles
        hit_t: t values where hits occur
    """
    B = ray_origins.shape[0]
    F = triangle_verts.shape[0]
    
    # Expand for broadcasting: (B, 1, 3) and (1, F, 3)
    origins = ray_origins.unsqueeze(1).to(device)  # (B, 1, 3)
    directions = ray_directions.unsqueeze(1).to(device)  # (B, 1, 3)
    
    # Triangle vertices
    v0 = triangle_verts[:, 0].unsqueeze(0).to(device)  # (1, F, 3)
    v1 = triangle_verts[:, 1].unsqueeze(0).to(device)
    v2 = triangle_verts[:, 2].unsqueeze(0).to(device)
    
    # Möller-Trumbore algorithm (vectorized over B rays and F triangles)
    edge1 = v1 - v0  # (1, F, 3)
    edge2 = v2 - v0  # (1, F, 3)
    
    h = torch.cross(directions, edge2, dim=2)  # (B, F, 3)
    a = (edge1 * h).sum(dim=2)  # (B, F)
    
    # Parallel rays (no intersection)
    valid = torch.abs(a) > 1e-8  # (B, F)
    
    f = 1.0 / (a + 1e-10)  # (B, F)
    s = origins - v0  # (B, F, 3)
    u = f * (s * h).sum(dim=2)  # (B, F)
    
    valid &= (u >= 0.0) & (u <= 1.0)
    
    q = torch.cross(s, edge1, dim=2)  # (B, F, 3)
    v = f * (directions * q).sum(dim=2)  # (B, F)
    
    valid &= (v >= 0.0) & (u + v <= 1.0)
    
    t = f * (edge2 * q).sum(dim=2)  # (B, F)
    valid &= (t > 1e-8) & (t < 1.0)  # Only hits along segment
    
    # Extract hits (keep on GPU)
    hit_indices = torch.nonzero(valid, as_tuple=False)  # (num_hits, 2)
    if len(hit_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.float32, device=device)
    
    hit_ray_idx = hit_indices[:, 0]
    hit_t = t[valid]
    
    return hit_ray_idx, hit_t


def compute_sdf_along_segments_3d(
    components: List,
    mesh_verts: torch.Tensor,
    mesh_faces: torch.Tensor,
    n_samples_per_unit: int = 1000,
    device: str = 'cuda',
    exact: bool = True
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute SDF along line segments for 3D mesh using GPU acceleration (fully vectorized).
    
    Args:
        components: List of PCAComponent objects
        mesh_verts: (V, 3) mesh vertices
        mesh_faces: (F, 3) mesh faces
        n_samples_per_unit: Number of samples per unit length
        device: Device for computation
        exact: If True, use exact point-to-mesh distance (slower but accurate).
               If False, use KNN approximation (faster but less accurate).
        
    Returns:
        t_values_list: List of (N,) tensors with t parameters
        sdf_values_list: List of (N,) tensors with SDF values
    """
    print(f"Computing SDF for {len(components)} segments...")
    
    # ========================================
    # 1. Collect all segments and compute sampling
    # ========================================
    starts = torch.stack([c.start for c in components])  # (B, 3)
    ends = torch.stack([c.end for c in components])  # (B, 3)
    
    segment_lengths = torch.norm(ends - starts, dim=1)  # (B,)
    n_samples_per_segment = (segment_lengths * n_samples_per_unit).int()
    n_samples_per_segment = torch.clamp(n_samples_per_segment, min=10)
    
    max_samples = n_samples_per_segment.max().item()
    B = len(components)
    
    # ========================================
    # 2. Generate all sample points (batched)
    # ========================================
    print(f"  Sampling {B} segments with max {max_samples} points each...")
    
    # Create t values for each segment (padded to max_samples)
    t_grid = torch.linspace(0, 1, max_samples).unsqueeze(0).expand(B, -1)  # (B, max_samples)
    
    # Compute points: (B, max_samples, 3)
    directions = ends - starts  # (B, 3)
    points = starts.unsqueeze(1) + t_grid.unsqueeze(2) * directions.unsqueeze(1)
    
    # Flatten for batch processing
    points_flat = points.reshape(-1, 3)  # (B * max_samples, 3)
    
    # ========================================
    # 3. Compute distances (GPU batch)
    # ========================================
    print(f"  Computing distances for {len(points_flat):,} points (exact={exact})...")
    
    mesh_pytorch3d = Meshes(
        verts=mesh_verts.unsqueeze(0).to(device),
        faces=mesh_faces.unsqueeze(0).to(device)
    )
    
    # Get triangle vertices for distance computation
    triangle_verts = mesh_verts[mesh_faces.long()].to(device)  # (F, 3, 3)
    
    if exact:
        # Use exact point-to-mesh distance from pytorch3d (more accurate)
        batch_size = 50000  # Can use larger batches with pytorch3d
        all_distances = []
        
        for i in tqdm(range(0, len(points_flat), batch_size), desc="  Distance batches (exact)", leave=False):
            batch_points = points_flat[i:i+batch_size].float().to(device)
            
            # point_mesh_distance from pytorch3d.loss
            # Computes distance from point cloud (batch_points) to mesh
            # Create a point cloud as a Meshes object with no faces
            pcl = Meshes(
                verts=[batch_points], 
                faces=[torch.zeros((0, 3), dtype=torch.long, device=device)]
            )
            
            # Returns (point_to_mesh, mesh_to_point)
            # point_to_mesh: distance from each point in pcl to nearest point on mesh
            loss_dict = point_mesh_distance.point_mesh_distance(mesh_pytorch3d, pcl)
            point_to_mesh_dist = torch.sqrt(loss_dict)  # Already gives us the distance
            
            all_distances.append(point_to_mesh_dist)
        
        distances_flat = torch.cat(all_distances)
    else:
        # Use KNN approximation (faster but less accurate)
        mesh_samples = sample_points_from_meshes(mesh_pytorch3d, num_samples=50000)
        
        batch_size = 100000  # Larger batches for KNN
        all_distances = []
        
        for i in tqdm(range(0, len(points_flat), batch_size), desc="  Distance batches (KNN)", leave=False):
            batch_points = points_flat[i:i+batch_size].float().to(device).unsqueeze(0)
            knn_result = knn_points(batch_points, mesh_samples, K=1)
            batch_dist = torch.sqrt(knn_result.dists[0, :, 0])
            all_distances.append(batch_dist)
        
        distances_flat = torch.cat(all_distances)
    
    distances = distances_flat.reshape(B, max_samples).to(device)  # (B, max_samples) on GPU
    
    # ========================================
    # 4. Compute signs using vectorized ray-triangle intersection
    # ========================================
    print(f"  Computing signs via ray tracing...")
    
    # Ray tracing for all segments at once (reuse triangle_verts from above)
    hit_ray_idx, hit_t = _ray_triangle_intersection_batch(
        starts, directions, triangle_verts, device=device
    )
    
    # Process signs using vectorized sweep-line algorithm (all on GPU)
    signs = torch.ones(B, max_samples, dtype=torch.float32, device=device)  # (B, max_samples) on GPU
    
    if len(hit_ray_idx) > 0:
        # Vectorized sign computation via sweep-line
        # Create combined events: samples (flag=0) and hits (flag=1)
        sample_ray_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_samples).reshape(-1)  # (B*max_samples,)
        sample_t = t_grid.to(device).reshape(-1)  # (B*max_samples,)
        sample_flags = torch.zeros(B * max_samples, dtype=torch.int8, device=device)
        
        # Concatenate samples and hits
        all_ray_idx = torch.cat([sample_ray_idx, hit_ray_idx])
        all_t = torch.cat([sample_t, hit_t])
        all_flags = torch.cat([sample_flags, torch.ones(len(hit_t), dtype=torch.int8, device=device)])
        
        # Sort by (ray_idx, t, flag) - flags ensure hits come before samples at same t
        sort_keys = all_ray_idx.float() * 2.0 + all_t + all_flags.float() * 0.5  # Prioritize hits before samples
        sort_idx = torch.argsort(sort_keys)
        
        sorted_ray_idx = all_ray_idx[sort_idx]
        sorted_flags = all_flags[sort_idx]
        
        # Cumulative count of crossings
        crossing_counts = torch.cumsum(sorted_flags.long(), dim=0)
        
        # Extract only sample positions
        is_sample = sorted_flags == 0
        sample_counts = crossing_counts[is_sample]
        
        # Compute offset per ray (reset count at start of each ray)
        # Count hits per ray (vectorized)
        hit_counts_per_ray = torch.bincount(hit_ray_idx.long(), minlength=B)
        
        # Cumulative hits before each ray
        ray_offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(hit_counts_per_ray[:-1], dim=0)])
        
        # Expand offsets to match samples
        sample_ray_positions = sorted_ray_idx[is_sample]
        corrections = ray_offsets[sample_ray_positions]
        
        # Corrected crossing counts
        true_counts = sample_counts - corrections
        
        # Even crossings = outside (+1), odd = inside (-1)
        signs_flat = torch.where(true_counts % 2 == 0, 
                                torch.tensor(1.0, device=device), 
                                torch.tensor(-1.0, device=device))
        
        # Reshape back
        signs = signs_flat.reshape(B, max_samples)
    
    # ========================================
    # 5. Apply signs and extract results
    # ========================================
    sdf = distances * signs  # (B, max_samples) on GPU
    
    # Extract valid samples for each segment (move to CPU only here at the end)
    t_values_list = []
    sdf_values_list = []
    
    for i in range(B):
        n = n_samples_per_segment[i].item()
        t_values_list.append(t_grid[i, :n].cpu())
        sdf_values_list.append(sdf[i, :n].cpu())
    
    print(f"✓ SDF computation complete")
    return t_values_list, sdf_values_list


def compute_sdf_along_segments_2d(
    components: List,
    polygons: List[np.ndarray],
    n_samples_per_unit: int = 1000
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute SDF along line segments for 2D polygons.
    
    Args:
        components: List of PCAComponent objects
        polygons: List of polygon vertex arrays
        n_samples_per_unit: Number of samples per unit length
        
    Returns:
        t_values_list: List of (N,) tensors with t parameters
        sdf_values_list: List of (N,) tensors with SDF values
    """
    t_values_list = []
    sdf_values_list = []
    
    for comp in tqdm(components, desc="Computing SDF (2D)", unit="segment"):
        # Get line segment (keep as tensors)
        p_start = comp.start
        p_end = comp.end
        
        # Compute number of samples based on segment length
        segment_length = torch.norm(p_end - p_start).item()
        n_samples = int(segment_length * n_samples_per_unit)
        n_samples = max(10, n_samples)  # Minimum 10 samples per segment
        
        # Sample points along segment (as tensors)
        t = torch.linspace(0, 1, n_samples)
        points = p_start + t.unsqueeze(1) * (p_end - p_start)
        
        # Convert to numpy for polygon operations (easier with numpy)
        points_np = points.cpu().numpy()
        
        # Compute SDF for each polygon and take minimum (union)
        sdf = np.full(n_samples, np.inf)
        
        for poly in polygons:
            # Compute distance to polygon edges
            poly_distances = np.full(n_samples, np.inf)
            for i in range(len(poly)):
                v1, v2 = poly[i], poly[(i + 1) % len(poly)]
                edge_vec = v2 - v1
                edge_len_sq = np.dot(edge_vec, edge_vec)
                
                if edge_len_sq < 1e-10:
                    continue
                
                t_closest = np.clip(np.dot(points_np - v1, edge_vec) / edge_len_sq, 0, 1)
                closest_points = v1 + t_closest[:, None] * edge_vec
                edge_distances = np.linalg.norm(points_np - closest_points, axis=1)
                poly_distances = np.minimum(poly_distances, edge_distances)
            
            # Ray casting for inside/outside
            v_next = np.roll(poly, -1, axis=0)
            py = points_np[:, 1:2]
            px = points_np[:, 0:1]
            
            v_y, v_next_y = poly[:, 1], v_next[:, 1]
            v_x, v_next_x = poly[:, 0], v_next[:, 0]
            
            # Edges that cross the horizontal ray
            crosses = ((v_y <= py) & (py < v_next_y)) | ((v_next_y <= py) & (py < v_y))
            
            # Safe division
            dy = v_next_y - v_y
            valid = np.abs(dy) > 1e-12
            safe_dy = np.where(valid, dy, 1.0)
            
            t_cross = (py - v_y) / safe_dy
            x_cross = v_x + t_cross * (v_next_x - v_x)
            
            crossings_right = crosses & valid & (x_cross > px)
            inside_poly = np.sum(crossings_right, axis=1) % 2 == 1
            
            # Signed distance for this polygon
            poly_sdf = np.where(inside_poly, -poly_distances, poly_distances)
            
            # Union: take minimum
            sdf = np.minimum(sdf, poly_sdf)
        
        # Convert SDF to tensor
        sdf_tensor = torch.from_numpy(sdf).float()
        
        # Store tensors
        t_values_list.append(t)
        sdf_values_list.append(sdf_tensor)
    
    return t_values_list, sdf_values_list


def simplify_sdf_to_knots_batch(
    t_values_list: List[torch.Tensor],
    sdf_values_list: List[torch.Tensor],
    tolerance: float = 0.005
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Simplify SDF curves to knots using Douglas-Peucker algorithm (parallelized).
    
    Args:
        t_values_list: List of (N,) tensors with t parameters
        sdf_values_list: List of (N,) tensors with SDF values
        tolerance: Maximum perpendicular distance for simplification
        
    Returns:
        knot_t_list: List of simplified t values
        knot_sdf_list: List of simplified SDF values
        max_errors: (B,) tensor of maximum errors
        mean_errors: (B,) tensor of mean errors
    """
    knot_t_list = []
    knot_sdf_list = []
    max_errors = []
    mean_errors = []
    
    for t_values, sdf_values in tqdm(zip(t_values_list, sdf_values_list), 
                                      total=len(t_values_list),
                                      desc="Simplifying to knots", 
                                      unit="segment"):
        # Convert to numpy for Douglas-Peucker (easier to implement recursively)
        t_np = t_values.cpu().numpy()
        sdf_np = sdf_values.cpu().numpy()
        
        def perpendicular_distance(point_idx, start_idx, end_idx):
            x0, y0 = t_np[point_idx], sdf_np[point_idx]
            x1, y1 = t_np[start_idx], sdf_np[start_idx]
            x2, y2 = t_np[end_idx], sdf_np[end_idx]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / np.sqrt(dx**2 + dy**2)
        
        def douglas_peucker(start_idx, end_idx):
            if end_idx - start_idx <= 1:
                return [start_idx, end_idx]
            
            max_dist, max_idx = 0, start_idx
            for i in range(start_idx + 1, end_idx):
                dist = perpendicular_distance(i, start_idx, end_idx)
                if dist > max_dist:
                    max_dist, max_idx = dist, i
            
            if max_dist > tolerance:
                left = douglas_peucker(start_idx, max_idx)
                right = douglas_peucker(max_idx, end_idx)
                return left + right[1:]
            return [start_idx, end_idx]
        
        # Apply Douglas-Peucker
        keep_indices = sorted(set(douglas_peucker(0, len(t_np) - 1)))
        
        knot_t = t_np[keep_indices]
        knot_sdf = sdf_np[keep_indices]
        
        # Compute error
        sdf_interp = np.interp(t_np, knot_t, knot_sdf)
        max_error = np.max(np.abs(sdf_np - sdf_interp))
        mean_error = np.mean(np.abs(sdf_np - sdf_interp))
        
        # Convert back to tensors
        knot_t_list.append(torch.from_numpy(knot_t).float())
        knot_sdf_list.append(torch.from_numpy(knot_sdf).float())
        max_errors.append(max_error)
        mean_errors.append(mean_error)
    
    return (knot_t_list, knot_sdf_list, 
            torch.tensor(max_errors), torch.tensor(mean_errors))


def compute_sdf_and_knots(
    components: List,
    data: dict,
    n_samples_per_unit: int = 1000,
    tolerance: float = 0.005,
    device: str = 'cuda',
    exact: bool = True
) -> Tuple[List[torch.Tensor], List[torch.Tensor], 
           List[torch.Tensor], List[torch.Tensor],
           torch.Tensor, torch.Tensor]:
    """
    Complete SDF computation and knot simplification pipeline.
    
    Args:
        components: List of PCAComponent objects
        data: Dictionary with 'type', 'vertices', and either 'verts'/'faces' (3D) or 'polygons' (2D)
        n_samples_per_unit: Number of samples per unit length
        tolerance: Douglas-Peucker tolerance
        device: Device for GPU computation
        exact: If True, use exact point-to-mesh distance for 3D (slower but accurate).
               If False, use KNN approximation (faster but less accurate).
        
    Returns:
        t_values_list: List of full t sampling
        sdf_values_list: List of full SDF values
        knot_t_list: List of simplified t knots
        knot_sdf_list: List of simplified SDF knots
        max_errors: Maximum errors per component
        mean_errors: Mean errors per component
    """
    # Compute SDF along segments
    if data['type'] == '3d':
        t_values_list, sdf_values_list = compute_sdf_along_segments_3d(
            components=components,
            mesh_verts=data['verts'],
            mesh_faces=data['faces'],
            n_samples_per_unit=n_samples_per_unit,
            device=device,
            exact=exact
        )
    else:  # 2d
        t_values_list, sdf_values_list = compute_sdf_along_segments_2d(
            components=components,
            polygons=data['polygons'],
            n_samples_per_unit=n_samples_per_unit
        )
    
    # Simplify to knots
    knot_t_list, knot_sdf_list, max_errors, mean_errors = simplify_sdf_to_knots_batch(
        t_values_list, sdf_values_list, tolerance=tolerance
    )
    
    return (t_values_list, sdf_values_list,
            knot_t_list, knot_sdf_list,
            max_errors, mean_errors)

