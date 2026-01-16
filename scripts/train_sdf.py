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

def train(task_name, arch_name, mesh_name, n_iters):
    net_dir = root_dir / "nets" / task_name / arch_name / mesh_name
    rmtree(net_dir, ignore_errors=True)

    def save():
        net_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, net_dir / "net.pt")
        torch.save(torch.tensor(losses), net_dir / "losses.pt")

        fig, ax1 = plt.subplots(figsize=(6, 3))
        fig.suptitle(f"{arch_name} / {mesh_name}")
        ax2 = ax1.twinx()
        ax2.semilogy()
        ax1.plot(losses, color="C0")
        ax2.plot(losses, color="C1")
        ax1.tick_params("y", labelcolor="C0")
        ax2.tick_params("y", labelcolor="C1")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Loss")
        fig.tight_layout()
        fig.savefig(net_dir / "losses.png", bbox_inches="tight")
        plt.close()

    if task_name == "from_sdf":
        pcd = o3d.t.io.read_point_cloud(root_dir / "data" / "sdf_point_clouds" / f"{mesh_name}.ply")
        X = torch.tensor(pcd.point.positions.numpy(), device="cuda")
        y = torch.tensor(pcd.point.signed_distances.numpy(), device="cuda")
    elif task_name == "from_surface":
        pcd = o3d.t.io.read_point_cloud(root_dir / "data" / "surface_point_clouds" / f"{mesh_name}.ply")
        X = pcd.point.positions.numpy()
        pert_sigmas = torch.tensor(KDTree(X).query(X, 51)[0][:, -1], dtype=torch.float32, device="cuda")
        X = torch.tensor(X, device="cuda")
        N = torch.tensor(pcd.point.normals.numpy(), device="cuda")
    elif task_name == "primitives":
        if mesh_name == "sphere":
            r = 0.9
            X = torch.vstack([
                torch.empty((10000, 3), device="cuda").uniform_(-1.1, 1.1),
                torch.normal(r * F.normalize(torch.empty((100000, 3), device="cuda").normal_(), dim=1), 0.05)
            ])
            y = X.norm(dim=1, keepdim=True) - r
            print(X.shape, y.shape)
        else:
            raise ValueError(f"Unknown primitive: {mesh_name}")
    else:
        raise ValueError(f"Unknown task: {task_name}")

    net = architectures.by_name(arch_name).cuda()
    optim = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=1e-4)
    batch_size = 250_000 if arch_name.startswith("siren") else 10_000

    losses = []
    save_interval = 1000
    progress = tqdm(range(n_iters), desc="iterations", leave=False)
    for itr in progress:
        # Note: Currently, the only way to sample without replacement is via torch.randperm(), which becomes slow for
        # very large source size. So we just sample with replacement and accept the occasional redundant sample.
        ind = torch.randint(len(X), (batch_size,), device="cuda")
        if task_name == "from_surface":
            X_surf = X[ind].requires_grad_()
            X_rand = torch.cat([
                torch.normal(X, pert_sigmas[:, None]),
                torch.empty((batch_size // 8, 3), device="cuda").uniform_(-1.1, 1.1)
            ], dim=0).requires_grad_()
            X_rand_grad = torch.autograd.grad([net(X_rand).sum()], [X_rand], create_graph=True)[0]
            loss = net(X_surf).abs().mean() + 0.1 * (X_rand_grad.norm(dim=1) - 1).square().mean()
            if itr < 20_000:
                X_surf_grad = torch.autograd.grad([net(X_surf).sum()], [X_surf], create_graph=True)[0]
                loss = loss + 1 / (1 + 0.01 * itr) * (X_surf_grad - N[ind]).abs().norm(dim=1).mean()
        else:
            loss = (net(X[ind]) - y[ind]).abs().mean()
        optim.zero_grad()
        loss.backward(inputs=optim.param_groups[0]["params"])
        optim.step()
        losses.append(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if itr % save_interval == 0:
            save()

    save()

    net.eval()

    gt_mesh_path = root_dir / "data" / "meshes" / f"{mesh_name}.ply"
    if not gt_mesh_path.exists():
        raise FileNotFoundError(f"Ground truth mesh file {gt_mesh_path} not found")
    metrics = compute_meshless_metrics(net, gt_mesh_path)

    metrics["details"] = {
        "arch": arch_name,
        "mesh": mesh_name,
        "n_iters": n_iters
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
    parser.add_argument("--n_iters", type=int, required=True)
    args = parser.parse_args()
    for call in tqdm(list(product(args.arch, args.mesh)), desc="nets"):
        train(args.task, *call, args.n_iters)