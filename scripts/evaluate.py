from argparse import ArgumentParser
from functools import reduce
from itertools import count, product
from pathlib import Path
from shutil import rmtree
import sys
from time import time

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree, cKDTree
from geomloss import SamplesLoss
from tqdm import tqdm

from marching import arch as architectures
import trimesh

root_dir = Path(__file__).parent.parent


def compute_meshless_metrics(model, gt_mesh_path, num_samples=30000):
    """
    Computes Chamfer and Hausdorff without Marching Cubes.
    
    Args:
        model: Your trained neural SDF (torch.nn.Module)
        gt_mesh_path: Path to the ground truth .obj/.ply
        num_samples: Number of points to sample for evaluation
    """
    device = next(model.parameters()).device
    
    # --- PREPARE DATA ---
    # 1. Load GT Mesh
    mesh_gt = trimesh.load(gt_mesh_path)
    
    # 2. Sample points on GT surface (for Direction 1)
    points_gt, _ = trimesh.sample.sample_surface(mesh_gt, num_samples)
    points_gt_tensor = torch.tensor(points_gt, dtype=torch.float32, device=device)

    # 3. Prepare "Near Surface" points (for Direction 2)
    # We add noise to GT points to start "near" the surface, then project them
    noise = torch.randn_like(points_gt_tensor) * 0.05
    points_to_project = points_gt_tensor + noise
    points_to_project.requires_grad_(True)

    # --- DIRECTION 1: GT -> NEURAL (Easy) ---
    # The distance is simply the absolute SDF value
    with torch.no_grad():
        sdf_pred_gt = model(points_gt_tensor)
    
    dist_gt_to_neural = torch.abs(sdf_pred_gt).squeeze()

    # --- DIRECTION 2: NEURAL -> GT (Hard) ---
    # 1. Project points to the neural zero-level set (Newton's method)
    # We do 5-10 iterations to ensure they are exactly on the zero-isosurface
    p_current = points_to_project
    for _ in range(5):
        sdf_val = model(p_current)
        grad = torch.autograd.grad(
            outputs=sdf_val, 
            inputs=p_current, 
            grad_outputs=torch.ones_like(sdf_val), 
            create_graph=False, 
            retain_graph=True
        )[0]
        
        # Project: x_new = x - f(x) * n
        # Normalize gradient to ensure stable step
        grad_norm = torch.nn.functional.normalize(grad, dim=1)
        p_current = p_current - sdf_val * grad_norm

    # Now p_current lies on the neural surface (approx).
    # We need the distance from these points to the REAL mesh.
    # We use scipy cKDTree for fast nearest-neighbor search.
    
    neural_points_np = p_current.detach().cpu().numpy()
    
    # Query distances against GT mesh vertices 
    # (For exactness, you should use trimesh.nearest.on_surface, but KDTree on vertices is a fast approx)
    # Method A: Exact surface distance (slower but accurate)
    closest_points, distances, triangle_id = mesh_gt.nearest.on_surface(neural_points_np)
    dist_neural_to_gt = torch.tensor(distances, device=device, dtype=torch.float32)

    # --- COMPUTE METRICS ---
    
    # Chamfer Distance (L1 or L2)
    # Mean of both directions
    chamfer_l1 = 0.5 * (torch.mean(dist_gt_to_neural) + torch.mean(dist_neural_to_gt))
    chamfer_l1_median = 0.5 * (torch.median(dist_gt_to_neural) + torch.median(dist_neural_to_gt))
    
    # Hausdorff Distance
    # Max of the distances (worst-case error)
    # We take the max of both directions to find the biggest outlier
    hausdorff = torch.max(
        torch.max(dist_gt_to_neural),
        torch.max(dist_neural_to_gt)
    )

    # Define the loss (p=1 implies Earth Mover's Distance)
    # blur=0.005 controls the "fuzziness" of the matching (regularization)
    emd_loss = SamplesLoss("sinkhorn", p=1, blur=0.005, backend="tensorized")

    # Example Data (Batch, N_points, 3)
    # neural_points: (1, 10000, 3) projected from your SDF
    # gt_points: (1, 10000, 3) sampled from mesh
    with torch.no_grad():
        emd_loss_tensor = emd_loss(p_current[500:], points_gt_tensor[500:])

    return {
        "chamfer_l1": chamfer_l1.item(),
        "chamfer_l1_median": chamfer_l1_median.item(),
        "hausdorff": hausdorff.item(),
        "dist_gt_to_neural_mean": torch.mean(dist_gt_to_neural).item(),
        "dist_neural_to_gt_mean": torch.mean(dist_neural_to_gt).item(),
        "emd_loss": emd_loss_tensor.item()
    }

def eval(task_name, arch_name, mesh_name):
    net_dir = root_dir / "nets" / task_name / arch_name / mesh_name
    
    if not net_dir.exists():
        raise FileNotFoundError(f"Net directory {net_dir} not found")
    
    net = architectures.by_name(arch_name).cuda()
    net.load_state_dict(torch.load(net_dir / "net.pt", weights_only=True))
    net.cuda()

    net.eval()

    gt_mesh_path = root_dir / "data" / "meshes" / f"{mesh_name}.ply"
    if not gt_mesh_path.exists():
        raise FileNotFoundError(f"Ground truth mesh file {gt_mesh_path} not found")
    metrics = compute_meshless_metrics(net, gt_mesh_path)

    metrics["details"] = {
        "arch": arch_name,
        "mesh": mesh_name,
    }

    import json
    with open(net_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    from sdf_meshing import create_mesh
    create_mesh(net, net_dir / "mesh.ply", N=512)

if __name__ == "__main__":
    parser = ArgumentParser(description="Trains SDF neural networks from point clouds.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--arch", action="extend", nargs="+", required=True)
    parser.add_argument("--mesh", action="extend", nargs="+", required=True)
    args = parser.parse_args()
    for call in tqdm(list(product(args.arch, args.mesh)), desc="nets"):
        eval(args.task, *call)