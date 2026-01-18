from dataclasses import dataclass
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points
from .types import PCAComponent, Splines
import trimesh
import time


def sample_surface_pointcloud(
    verts: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample points AND normals on a triangle mesh surface using PyTorch3D.

    Args:
        verts: (V, 3) float tensor
        faces: (F, 3) long tensor
        num_samples: number of points to sample
        device: torch device (optional)

    Returns:
        points:  (num_samples, 3) float tensor
        normals: (num_samples, 3) float tensor (unit length)
    """

    if device is None:
        device = verts.device

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
    )

    # (1, N, 3), (1, N, 3)
    points, normals = sample_points_from_meshes(
        mesh,
        num_samples,
        return_normals=True,
    )

    return points[0], normals[0]

def compute_ray_geometry(start_points: torch.Tensor, end_points: torch.Tensor, mesh: trimesh.Trimesh):
    """
    Returns:
        t_hit: (R, Cmax)
        hit_count: (R,)
        surface_points: (N, 3)
        surface_normals: (N, 3)
    """

    assert trimesh.ray.has_embree, "Trimesh must have embree installed"

    bbox_min = torch.tensor([-1.0, -1.0, -1.0], device=start_points.device)
    bbox_max = torch.tensor([ 1.0,  1.0,  1.0], device=start_points.device)

    ray_directions = end_points - start_points
    ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)

    # Inverse direction (avoid division by zero)
    inv_dir = 1.0 / torch.where(
        ray_directions.abs() < 1e-8,
        torch.full_like(ray_directions, 1e-8),
        ray_directions,
    )

    # Per-axis intersection times
    t0 = (bbox_min - start_points) * inv_dir
    t1 = (bbox_max - start_points) * inv_dir

    # Entry / exit times
    tmin = torch.minimum(t0, t1).amax(dim=1)
    tmax = torch.maximum(t0, t1).amin(dim=1)

    # Sanity check
    ray_start = start_points + tmin[:, None] * ray_directions

    valid = tmax >= tmin
    assert torch.all(valid), "Not all rays hit the box"
    
    # Convert to numpy and run ray tracing using trimesh, mesh is in data['mesh]
    ray_origins_np = ray_start.detach().cpu().numpy()
    ray_dirs_np    = ray_directions.detach().cpu().numpy()

    hit_points_np, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins_np,
        ray_directions=ray_dirs_np,
        multiple_hits=True,
    )
    hit_normals_np = mesh.face_normals[index_tri]

    start_np = start_points.detach().cpu().numpy()
    s0 = start_np[index_ray]
    d = ray_dirs_np[index_ray]
    
    t_hit_flat = np.einsum("ij,ij->i", hit_points_np-s0, d)

    # --- group + sort t_hit per ray ---
    R = start_points.shape[0]
    t_per_ray = []
    p_per_ray = []
    n_per_ray = []
    hit_count = []
    maxC = 0

    for r in tqdm(range(R)):
        mask = index_ray == r
        if not np.any(mask):
            t_per_ray.append(np.empty((0,), dtype=np.float32))
            p_per_ray.append(np.empty((0, 3), dtype=np.float32))
            n_per_ray.append(np.empty((0, 3), dtype=np.float32))
            hit_count.append(0)
            maxC = max(maxC, 0)
            continue  

        t_r = t_hit_flat[mask]
        p_r = hit_points_np[mask]
        n_r = hit_normals_np[mask]
        
        order = np.argsort(t_r)
        t_r = t_r[order]
        p_r = p_r[order]
        n_r = n_r[order]

        t_q = np.round(t_r, decimals=6)
        if t_q.size == 0 or np.all(t_q[1:] != t_q[:-1]):
            # no duplicates → zero cost
            t_r_out = t_r
            p_r_out = p_r
            n_r_out = n_r
        else:
            # slow path: vectorized merge
            t_unique, inv = np.unique(t_q, return_inverse=True)
            K = len(t_unique)

            p_out = np.zeros((K, 3), dtype=np.float32)
            n_out = np.zeros((K, 3), dtype=np.float32)
            counts = np.bincount(inv, minlength=K)

            np.add.at(p_out, inv, p_r)
            np.add.at(n_out, inv, n_r)

            p_out /= counts[:, None]
            n_out /= counts[:, None]
            n_out /= np.linalg.norm(n_out, axis=1, keepdims=True) + 1e-12

            t_r_out = t_unique
            p_r_out = p_out
            n_r_out = n_out

        t_per_ray.append(t_r_out)
        p_per_ray.append(p_r_out)
        n_per_ray.append(n_r_out)
        hit_count.append(t_r_out.size)
        maxC = max(maxC, t_r_out.size)

    t_pad = np.zeros((R, maxC), dtype=np.float32)
    p_pad = np.zeros((R, maxC, 3), dtype=np.float32)
    n_pad = np.zeros((R, maxC, 3), dtype=np.float32)
    hit_count = np.asarray(hit_count, dtype=np.int32)
    
    for r in range(R):
        C = hit_count[r]
        if C == 0:
            continue
        t_pad[r, :C] = t_per_ray[r]
        p_pad[r, :C] = p_per_ray[r]
        n_pad[r, :C] = n_per_ray[r]

    
    # --- to torch ---
    device = start_points.device
    t_hit = torch.from_numpy(t_pad).to(device)
    hit_count = torch.from_numpy(hit_count).to(device)
    hit_points = torch.from_numpy(p_pad).to(device)
    hit_normals = torch.from_numpy(n_pad).to(device)
    
    surface_points = np.concatenate(p_per_ray, axis=0)
    surface_normals = np.concatenate(n_per_ray, axis=0)
    surface_points = torch.from_numpy(surface_points.astype(np.float32)).to(device)
    surface_normals = torch.from_numpy(surface_normals.astype(np.float32)).to(device)

    return t_hit, hit_count, hit_points, hit_normals, surface_points, surface_normals

def scale_segments(start_points: torch.Tensor, end_points: torch.Tensor):
    # Scale start and end points to new lengths
    ray_len = (end_points - start_points).norm(dim=1)  # (R,)
    assert torch.all(ray_len > 0), "Ray length is zero"
    L = torch.quantile(ray_len, 0.1).item()
    q = torch.ceil(ray_len / L).clamp(min=1)   # (R,)
    ray_len_q = q * L                          # (R,)
    delta = ray_len_q - ray_len    # (R,)
    half  = 0.5 * delta            # (R,)
    
    ray_directions = (end_points - start_points) / ray_len[:, None] # (R,3)

    start_points = start_points - half[:, None] * ray_directions
    end_points   = end_points   + half[:, None] * ray_directions

    return start_points, end_points, L

def subdivide_segments(
    start_points: torch.Tensor,
    end_points: torch.Tensor,
    L: float,
):
    """
    Subdivide rays into equal-length segments.

    Args:
        start_points : (R,3) torch
        end_points   : (R,3) torch
        L            : float (segment length)

    Returns:
        new_start_points : (S,3) torch
        new_end_points   : (S,3) torch
        t_start          : (S,) torch
        t_end            : (S,) torch
        mapping_idx      : (S,)  torch  (segment -> ray index)
    """
    device = start_points.device

    # Ray directions and lengths
    dirs = end_points - start_points
    ray_len = torch.norm(dirs, dim=1)                    # (R,)

    # Number of segments per ray (must be integer by construction)
    n_seg = torch.ceil(ray_len / L).to(torch.long)      # (R,)
    assert torch.all(n_seg >= 1)

    # Map each segment to its parent ray
    mapping_idx = torch.repeat_interleave(
        torch.arange(start_points.shape[0], device=device),
        n_seg
    )                                                     # (S,)

    # Segment start offset along ray
    seg_start = torch.cumsum(n_seg, dim=0) - n_seg        # (R,)
    seg_idx = torch.arange(mapping_idx.shape[0], device=device) \
          - seg_start[mapping_idx]
    t_start = seg_idx.float() / n_seg[mapping_idx].float()                        # (S,)
    t_end = (seg_idx + 1).float() / n_seg[mapping_idx].float()                        # (S,)

    # Compute segment start/end points
    new_start_points = start_points[mapping_idx] + t_start[:, None] * dirs[mapping_idx]
    new_end_points   = start_points[mapping_idx] + t_end[:, None] * dirs[mapping_idx]

    return new_start_points, new_end_points, t_start, t_end, mapping_idx


def compute_splines(data: dict, components: List[PCAComponent], num_samples: int = 50_000) -> List[torch.Tensor]:
    # Get all samples
    sample_points, sample_normals = sample_surface_pointcloud(
        data['vertices'],
        data['faces'],
        num_samples,
    )
    
    # Get all rays
    start_points = torch.stack([c.start for c in components], dim=0)
    end_points = torch.stack([c.end for c in components], dim=0)
    print(f"Scale Segments")
    tik = time.time()
    start_points, end_points, L = scale_segments(start_points, end_points)
    tok = time.time()
    print(f"Scale Segments took {tok - tik} seconds")
    print(f"Compute Ray Geometry")
    tik = time.time()
    t_hit, hit_count, hit_points, hit_normals, surface_points, surface_normals = compute_ray_geometry(start_points, end_points, data['mesh'])
    tok = time.time()
    print(f"Compute Ray Geometry took {tok - tik} seconds")
    print(f"Subdivide Segments")
    tik = time.time()
    new_start_points, new_end_points, t_start, t_end, mapping_idx = subdivide_segments(start_points, end_points, L)
    tok = time.time()
    print(f"Subdivide Segments took {tok - tik} seconds")
    print(f"Sample Surface Pointcloud")
    tik = time.time()
    sample_points, sample_normals = sample_surface_pointcloud(data['vertices'], data['faces'], num_samples)
    sample_points = torch.cat([sample_points, surface_points], dim=0)
    sample_normals = torch.cat([sample_normals, surface_normals], dim=0)
    tok = time.time()
    print(f"Sample Surface Pointcloud took {tok - tik} seconds")
    print(f"Concatenate Points and Normals")
    tik = time.time()

    # For 2d remove points that look to the top or bottom
    if data['type'] == '2d':
        eps = 1e-4
        mask = sample_normals[:, 2].abs() < eps
        sample_points = sample_points[mask][:, :2]
        sample_normals = sample_normals[mask][:, :2]
    
    # Calculate SDF for each segment
    print(f"Computing SDF for segments")
    tik = time.time()
    batch_size = 1024

    n_samples_per_segment = 32
    n_selected_samples = 16
    


    device = start_points.device
    u_local = torch.linspace(0.0, 1.0, n_samples_per_segment, device=device)
    all_sdf = []
    all_normals = []
    all_knots = []
    for i in tqdm(range(0, len(new_start_points), batch_size), desc="Computing SDF"):
        start_points_batch = new_start_points[i:i+batch_size]
        end_points_batch = new_end_points[i:i+batch_size]
        t_start_batch = t_start[i:i+batch_size]
        t_end_batch = t_end[i:i+batch_size]
        mapping_idx_batch = mapping_idx[i:i+batch_size]
        
        t_global = t_start_batch[:, None] + u_local[None, :] * (t_end_batch - t_start_batch)[:, None]  # (B, T)

        parent_t_hit = t_hit[mapping_idx_batch]           # (B, Cmax)
        parent_hit_count = hit_count[mapping_idx_batch]   # (B,)
        parent_hit_points = hit_points[mapping_idx_batch]     # (B, Cmax, D)
        parent_hit_normals = hit_normals[mapping_idx_batch]   # (B, Cmax, D)
        
        # Convert to segment-local u
        segment_len = t_end_batch - t_start_batch
        u_hit = (parent_t_hit - t_start_batch[:, None]) / segment_len[:, None]  # (B, Cmax)
        
        cmax = parent_t_hit.shape[1]
        valid_in_segment = (torch.arange(cmax, device=device)[None, :] < parent_hit_count[:, None]) & \
                        (u_hit >= 0) & (u_hit <= 1)  # (B, Cmax)
        
        # Sample to compute sdf
        dirs = end_points_batch - start_points_batch          # (B, D)
        query_points = start_points_batch[:, None, :] + u_local[None, :, None] * dirs[:, None, :] # (B, 512, D)
        
        B, T, D = query_points.shape
        query_flat = query_points.reshape(-1, D)
        
        # KNN search for unsigned distance
        knn = knn_points(query_flat[None], sample_points[None], K=1)
        idx = knn.idx[0, :, 0]
        dist2 = knn.dists[0, :, 0]

        unsigned_dist = torch.sqrt(dist2).view(B, T)          # (B, 512)
        nearest_normals = sample_normals[idx].view(B, T, D)   # (B, 512, D)
        nearest_points = sample_points[idx].view(B, T, D)     # (B, 512, D)

        # Count crossings
        crossings = (parent_t_hit[:, None, :] < t_global[:, :, None]) # (B, T, Cmax)
        crossing_count = crossings.sum(dim=2)

        sign = torch.where(crossing_count % 2 ==1, -1.0, 1.0)
        sdf = sign*unsigned_dist
        normals = nearest_normals

        importance = ((unsigned_dist[:, :-2]+unsigned_dist[:, 2:])/2 - unsigned_dist[:, 1:-1]).abs()
        first_element = torch.tensor([[0]], device=device, dtype=torch.long).expand(B, -1)
        last_element = torch.tensor([[n_samples_per_segment-1]], device=device, dtype=torch.long).expand(B, -1)
        indices = torch.cat(
            [first_element, 
             torch.argsort(importance, dim=1, descending=True)[:,:n_selected_samples-2]+1,
             last_element], dim=1
        )

        knots = u_local.unsqueeze(0).expand(B, -1).gather(1, indices)
        sdf = sdf.gather(1, indices)
        normals = normals.gather(1, indices.unsqueeze(2).expand(-1, -1, D))

        all_sdf.append(sdf)
        all_normals.append(normals)
        all_knots.append(knots)
        
        
    tok = time.time()
    print(f"Computing SDF for segments took {tok - tik} seconds")

    splines = Splines(
        start_points=new_start_points,
        end_points=new_end_points,
        knots=torch.cat(all_knots, dim=0),
        values=torch.cat(all_sdf, dim=0),
        normals=torch.cat(all_normals, dim=0),
    )

    return splines

