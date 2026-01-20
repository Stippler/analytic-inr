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
from neural_spline.model import ReluMLP
from geomloss import SamplesLoss
from tqdm import tqdm
import numpy as np
from marching import arch as architectures
import trimesh

root_dir = Path(__file__).parent.parent

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

from pytorch3d.loss import chamfer_distance

def compute_meshless_metrics(model, gt_mesh_path, num_samples=30000):
    """
    Computes Chamfer and Hausdorff without Marching Cubes.
    
    Args:
        model: Your trained neural SDF (torch.nn.Module)
        gt_mesh_path: Path to the ground truth .obj/.ply
        num_samples: Number of points to sample for evaluation
    """
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    # --- PREPARE DATA ---
    # 1. Load GT Mesh
    mesh_gt = trimesh.load(gt_mesh_path)
    
    # 2. Sample points on GT surface (for Direction 1)
    points_gt, _ = trimesh.sample.sample_surface(mesh_gt, num_samples, seed=SEED)
    points_gt_tensor = torch.tensor(points_gt, dtype=torch.float32, device=device)

    # 3. Prepare "Near Surface" points (for Direction 2)
    # We add noise to GT points to start "near" the surface, then project them
    noise = torch.randn(points_gt_tensor.size(), generator=generator, device=device) * 0.05
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
    # take the first 5000 points to compute the EMD loss
    # to avoid OOM
    torch.cuda.empty_cache()
    with torch.no_grad():
        emd_loss_tensor = emd_loss(p_current[:5000].cuda(), points_gt_tensor[:5000].cuda())

    return {
        "chamfer_l1": chamfer_l1.item(),
        "chamfer_l1_median": chamfer_l1_median.item(),
        "hausdorff": hausdorff.item(),
        "dist_gt_to_neural_mean": torch.mean(dist_gt_to_neural).item(),
        "dist_neural_to_gt_mean": torch.mean(dist_neural_to_gt).item(),
        "emd_loss": emd_loss_tensor.item()
    }

def compute_IoU(data, model):
    model.eval()
    
    assert "pcd_vol" in data , "pcd_vol are required"
    
    points_to_project = data["pcd_vol"]
    gt_sdf = data["pcd_vol_sdf"]
    
    with torch.no_grad():
        pred_sdf = model(points_to_project)
    
    pred_inside = (pred_sdf < 0).squeeze()
    gt_inside = (gt_sdf < 0).squeeze()
    
    intersection = torch.logical_and(pred_inside, gt_inside).float().sum()
    union = torch.logical_or(pred_inside, gt_inside).float().sum()

    model.train()
    
    return intersection / (union + 1e-6)

def compute_chamfer_distance(data, model):
    model.eval()
    
    assert "pcd_surf" in data, "pcd_surf is required"
    
    device = next(model.parameters()).device
    
    # 1. Get points on the predicted surface using Newton's Method
    p_current = data["pcd_surf"].to(device).detach().requires_grad_(True)
    
    # We must enable grad for the projection even if in eval mode
    with torch.enable_grad():
        for _ in range(5):
            sdf_val = model(p_current)
            # Calculate gradient of SDF w.r.t. input points
            grad = torch.autograd.grad(
                outputs=sdf_val, 
                inputs=p_current, 
                grad_outputs=torch.ones_like(sdf_val),
            )[0]
            
            grad_norm = torch.nn.functional.normalize(grad, dim=1)
            # Project points: x = x - sdf(x) * normal
            p_current = p_current - sdf_val * grad_norm
            p_current = p_current.detach().requires_grad_(True)

    # 2. Compute Distance
    # Direction: Pred -> GT
    pcd_surf_gt = data["pcd_surf"].to(device)
    cd_p_gt, _ = chamfer_distance(p_current[None], pcd_surf_gt[None])
    
    # Direction: GT -> Pred (Requires projecting GT points or sampling SDF)
    # Usually, for Neural SDFs, we sample the SDF at GT surface points 
    # and treat that as the distance error.
    with torch.no_grad():
        model_preds_abs = model(pcd_surf_gt).abs()
        dist_gt_p = model_preds_abs.mean() 

    model.train()
    
    # old, not completely correct (was just pcd_surf to mesh)
    # cd, _ = chamfer_distance(p_current[None], data["pcd_surf"][None])
    # hd, _ = chamfer_distance(p_current[None], data["pcd_surf"][None], point_reduction="max")
    
    cd_2 = (cd_p_gt + dist_gt_p) / 2
    hd_2 = torch.max(chamfer_distance(p_current[None], pcd_surf_gt[None], point_reduction="max")[0], model_preds_abs.max())

    model.train()
    
    return cd_2, hd_2

def compute_trimesh_chamfer(gt_points, gen_mesh, num_mesh_samples=100000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples, seed=SEED)[0]

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.detach().cpu().numpy()

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, _ = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, _ = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    chamfer_l2 = gt_to_gen_chamfer + gen_to_gt_chamfer 

    # Hausdorff Distance 
    # (The maximum of the minimum distances)
    # Note: KDTree returns Euclidean distances, not squared.
    hausdorff_one = np.max(one_distances)
    hausdorff_two = np.max(two_distances)
    hausdorff_dist = max(hausdorff_one, hausdorff_two)

    THRESHOLD = 0.005

    # Precision & Recall
    # Precision: % of generated points within threshold of GT
    precision = np.mean(two_distances < THRESHOLD)
    
    # Recall: % of GT points within threshold of generated mesh
    recall = np.mean(one_distances < THRESHOLD)
    
    # F-Score (Harmonic mean of P and R)
    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0.0

    return {
        'chamfer_l2': chamfer_l2.item(),
        'hausdorff': hausdorff_dist.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f_score': f_score.item()
    }

def eval_fast(net, data, mesh_path, gt_mesh_path):
    # step 1: marching cubes
    print("Step 1: Marching Cubes")
    mesh_path = Path(mesh_path)
    assert mesh_path.exists(), f"Mesh path {mesh_path} not found"

    ground_truth_points = data['pcd_surf']
    reconstruction = trimesh.load(mesh_path)

    print("Step 2: Compute Chamfer Distance")
    trimesh_metrics = compute_trimesh_chamfer(
                    ground_truth_points,
                    reconstruction,
                )

    print("Step 3: Compute IoU")
    iou = compute_IoU(data, net)

    metrics = {}
    metrics["old"] = compute_meshless_metrics(net, gt_mesh_path)
    metrics |= trimesh_metrics
    metrics["iou"] = iou.item()

    total_params = sum(p.numel() for p in net.parameters())
    metrics["total_params"] = total_params

    import json
    with open(mesh_path.parent / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def eval(task_name, arch_name, mesh_name, sdf_path=None, save_mesh=True):
    # path is to get rid of the .ply extension
    try:
        net_dir = root_dir / "nets" / task_name / arch_name / Path(mesh_name).stem
        
        if not net_dir.exists():
            return
            raise FileNotFoundError(f"Net directory {net_dir} not found")
        
        net = architectures.by_name(arch_name).cuda()
        net.load_state_dict(torch.load(net_dir / "net.pt", weights_only=True))
    except Exception as e:
        # here, try the sdf_path
        if sdf_path is None:
            raise e
        config = torch.load(sdf_path, weights_only=True)
        net = ReluMLP.restore_from_config(config["config"])
        net.load_state_dict(config["model_state_dict"])
        
        # net_dir is where the outputs flow to
        net_dir = sdf_path.parent
        

    net.cuda()
    net.eval()

    gt_mesh_path = root_dir / "data" / "meshes" / f"{Path(mesh_name).stem}.ply"
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

if __name__ == "__main__":
    parser = ArgumentParser(description="Trains SDF neural networks from point clouds.")
    parser.add_argument("--task", required=False)
    # if the following arguments are not provided, all nets and meshes will be evaluated
    parser.add_argument("--arch", action="extend", nargs="+", required=False)
    parser.add_argument("--mesh", action="extend", nargs="+", required=False)
    parser.add_argument("--save_mesh", action="store_true", required=False)
    parser.add_argument("--sdf_path", type=str, required=False)
    args = parser.parse_args()
    
    # for the new script (more versatile)
    if args.sdf_path:
        assert args.mesh is not None, "Mesh is required when sdf_path is provided"
        sdf_path = Path(args.sdf_path)
        assert sdf_path.exists(), f"SDF path {sdf_path} not found"
        eval(args.task, args.arch, args.mesh[0], sdf_path, args.save_mesh)
    else:
        calls = []
        task_dir = root_dir / "nets" / args.task
        mesh_dir = root_dir / "data" / "meshes"
        
        arch = args.arch or (p.name for p in task_dir.iterdir())
        mesh = args.mesh or (p.name for p in mesh_dir.iterdir())
        
        for call in tqdm(list(product(arch, mesh)), desc="nets"):
            eval(args.task, *call, args.save_mesh)