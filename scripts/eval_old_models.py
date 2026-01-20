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
from scripts.evaluate import eval_fast
from neural_spline.utils import extract_mesh_marching_cubes, load_mesh_data

root_dir = Path(__file__).parent.parent

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def eval(task_name, arch_name, mesh_name, sdf_path=None, save_mesh=True):

    net_dir = root_dir / "nets" / task_name / arch_name / Path(mesh_name).stem
    
    if not net_dir.exists():
        return
        raise FileNotFoundError(f"Net directory {net_dir} not found")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = architectures.by_name(arch_name).to(device)
    net.load_state_dict(torch.load(net_dir / "net.pt", weights_only=True))
    net.eval()

    # step 0: load data
    data = load_mesh_data(mesh_name, '3d')
    if data is None:
        return
        raise FileNotFoundError(f"Data not found for mesh {mesh_name}")

    # step 1: marching cubes
    final_mesh_path = net_dir / 'mesh_final.ply'
    extract_mesh_marching_cubes(net, save_path=final_mesh_path, resolution=256, device=device)

    eval_fast(net, data, final_mesh_path, Path("data") / "meshes" / f"{mesh_name}.ply")


    # gt_mesh_path = root_dir / "data" / "meshes" / f"{Path(mesh_name).stem}.ply"
    # if not gt_mesh_path.exists():
    #     raise FileNotFoundError(f"Ground truth mesh file {gt_mesh_path} not found")
    # metrics = compute_meshless_metrics(net, gt_mesh_path)

    # metrics["details"] = {
    #     "arch": arch_name,
    #     "mesh": mesh_name,
    # }

    # import json
    # with open(net_dir / "metrics.json", "w") as f:
    #     json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser(description="Trains SDF neural networks from point clouds.")
    parser.add_argument("--task", required=False)
    # if the following arguments are not provided, all nets and meshes will be evaluated
    parser.add_argument("--arch", action="extend", nargs="+", required=False)
    parser.add_argument("--mesh", action="extend", nargs="+", required=False)
    parser.add_argument("--save_mesh", action="store_true", required=False)
    args = parser.parse_args()
    

    calls = []
    task_dir = root_dir / "nets" / args.task
    mesh_dir = root_dir / "data" / "meshes"
    
    arch = args.arch or (p.name for p in task_dir.iterdir())
    mesh = args.mesh or (p.name for p in mesh_dir.iterdir())
    
    for call in tqdm(list(product(arch, mesh)), desc="nets"):
        eval(args.task, *call, args.save_mesh)