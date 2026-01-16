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
from scipy.spatial import KDTree
from tqdm import tqdm

from marching import arch as architectures

root_dir = Path(__file__).parent.parent


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


if __name__ == "__main__":
    parser = ArgumentParser(description="Trains SDF neural networks from point clouds.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--arch", action="extend", nargs="+", required=True)
    parser.add_argument("--mesh", action="extend", nargs="+", required=True)
    parser.add_argument("--n_iters", type=int, required=True)
    args = parser.parse_args()
    for call in tqdm(list(product(args.arch, args.mesh)), desc="nets"):
        train(args.task, *call, args.n_iters)