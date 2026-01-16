import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import randint
from tempfile import gettempdir

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import open3d as o3d
from scipy.stats.qmc import Sobol
from tqdm import tqdm

root_dir = Path(__file__).parent.parent


def preprocess(mesh_name):
    n = 2 ** 24
    mesh_path = root_dir / "data" / "meshes" / f"{mesh_name}.ply"
    
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file {mesh_path} not found")

    # Sample n points from a [-1.1, 1.1] volume (widened to protect against undefined behavior of the neural network
    # right at the valid domain border). Some points or sampled uniformly, but most of them are on or near the surface.
    # For ShapeNet meshes, we need to use DeepSDF's special sampler because those shapes are very much non-watertight.
    if "ShapeNet" not in mesh_name:
        n_uniform = 2 ** 20
        n_surface = n - n_uniform
        n_surface_on = n_surface // 2
        n_surface_near = n_surface - n_surface_on
        uniform_points = (Sobol(3).random(n_uniform).astype(np.float32) - 0.5) * 2.2
        mesh = o3d.t.io.read_triangle_mesh(mesh_path)
        surface_points_on = mesh.sample_points_uniformly(n_surface_on).point.positions.numpy()
        surface_points_near = mesh.sample_points_uniformly(n_surface_near).point.positions.numpy() + \
                              np.random.normal(scale=0.05, size=(n_surface_near, 3)).astype(np.float32)
        sdf_points = np.concatenate([uniform_points, surface_points_near])
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        sdf_samples = scene.compute_signed_distance(o3d.core.Tensor.from_numpy(sdf_points), nsamples=9).numpy()
        points = np.concatenate([sdf_points, surface_points_on])
        dists = np.concatenate([sdf_samples, np.zeros(n_surface_on, dtype=np.float32)])
    else:
        temp_path = Path(gettempdir(), f"marching_neurons_deep_sdf_preprocessor_{randint(0, 1000000000000)}.npz")
        cmd = [root_dir / "bin" / "PreprocessMesh", "-m", mesh_path, "-o", temp_path, "-s", str(n), "-b", "1.1"]
        subprocess.run(cmd, env=dict(os.environ, PANGOLIN_WINDOW_URI="headless://")).check_returncode()
        npz = np.load(temp_path)
        temp_path.unlink()
        points_and_dists = np.concatenate([npz["pos"], npz["neg"]])
        points = points_and_dists[:, :3]
        dists = points_and_dists[:, 3]

    # blue for interior points, red for exterior points
    cd = ((1 - np.abs(dists * 10).clip(0, 1)) * 255).astype(np.uint8)
    c1 = np.full_like(cd, 255)
    colors = np.where((dists < 0)[:, None], np.column_stack([cd, cd, c1]), np.column_stack([c1, cd, cd]))

    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor.from_numpy(points)
    pcd.point.signed_distances = o3d.core.Tensor.from_numpy(dists[:, None])
    pcd.point.colors = o3d.core.Tensor.from_numpy(colors)

    out_dir = root_dir / "data" / "sdf_point_clouds"
    out_dir.mkdir(exist_ok=True, parents=True)
    o3d.t.io.write_point_cloud(out_dir / f"{mesh_name}.ply", pcd)


if __name__ == "__main__":
    parser = ArgumentParser(description="Turns ground truth meshes into SDF point clouds.")
    parser.add_argument("--mesh", action="extend", nargs="+")
    args = parser.parse_args()
    for mesh_name in tqdm(args.mesh or [p.stem for p in (root_dir / "data" / "meshes").iterdir()], desc="meshes"):
        preprocess(mesh_name)