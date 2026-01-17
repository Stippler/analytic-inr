"""
Batch-rewrite all PLY meshes in data/meshes to a PyTorch3D-compatible format.

Assumptions:
- All meshes are already valid triangle meshes
- No cleaning, no triangulation
- Only header / property normalization is needed
"""

import os
import open3d as o3d
from pytorch3d.io import load_ply

MESH_DIR = "data/meshes"


def rewrite_ply_inplace(ply_path):
    # ------------------------------------------------------------
    # 1) Try PyTorch3D directly
    # ------------------------------------------------------------
    try:
        load_ply(ply_path)
        print(f"[OK] {ply_path}")
        return
    except Exception as e:
        print(f"[REWRITE] {ply_path}")
        print(f"          PyTorch3D error: {e}")

    # ------------------------------------------------------------
    # 2) Load with Open3D (assumed triangle mesh already)
    # ------------------------------------------------------------
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        print(f"[SKIP] {ply_path} (empty or unreadable)")
        return

    # ------------------------------------------------------------
    # 3) Overwrite with minimal ASCII PLY
    #    (forces canonical face list: uchar + int vertex_indices)
    # ------------------------------------------------------------
    o3d.io.write_triangle_mesh(
        ply_path,
        mesh,
        write_ascii=False,          # binary
        write_vertex_normals=False,
        write_vertex_colors=False,
        write_triangle_uvs=False,
        compressed=False,
    )

    # ------------------------------------------------------------
    # 4) Verify PyTorch3D can now load it
    # ------------------------------------------------------------
    load_ply(ply_path)
    print(f"[FIXED] {ply_path}")


def main():
    for root, _, files in os.walk(MESH_DIR):
        for name in files:
            if name.lower().endswith(".ply"):
                rewrite_ply_inplace(os.path.join(root, name))


if __name__ == "__main__":
    main()
