from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import trimesh
from pathlib import Path
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from skimage import measure
import open3d as o3d


def load_mesh_data(model: str, dim: str) -> Optional[Dict[str, Any]]:
    """
    Load mesh data for 2D or 3D models.
    
    Parameters:
    -----------
    model : str
        Model name. For 2D: 'simple' or 'hard'. For 3D: mesh filename (e.g., 'Armadillo')
    dim : str
        Dimension: '2d' or '3d'
    
    Returns:
    --------
    dict or None
        Dictionary containing:
        - 'type': '2d' or '3d'
        - 'mesh': trimesh.Trimesh object
        - 'vertices': torch.Tensor of shape (N, 3)
        - 'faces': torch.Tensor of shape (F, 3)
        
        Returns None on error (with error messages printed)
    """
    from .polygons import generate_polygons
    
    if dim == '2d':
        # Generate 2D polygons
        if model.lower() == "simple":
            print("  Generating simple convex polygons...")
            polygons_2d = generate_polygons("1x16", convex=True)
        elif model.lower() == "hard":
            print("  Generating hard non-convex polygons...")
            polygons_2d = generate_polygons(
                "3x16",
                convex=False,
                stretch=(1, 0.5),
                star_ratio=0.9,
                rotation=[np.pi / 4, -np.pi / 3, np.pi / 5],
            )
        else:
            print(f"ERROR: Unknown model '{model}'. Use 'simple' or 'hard' for 2D, or a mesh name for 3D")
            return None

        # ------------------------------------------------------------
        # Extrude polygons into a thin 3D triangle mesh
        # ------------------------------------------------------------
        thickness = 1e-3  # small but non-zero (ray-tracing safe)
        meshes = []

        for poly in polygons_2d:
            # Ensure CCW order
            poly = np.asarray(poly, dtype=np.float64)

            prism = trimesh.creation.extrude_polygon(
                polygon=trimesh.path.polygons.Polygon(poly),
                height=thickness,
            )

            # Center extrusion around z = 0
            prism.vertices[:, 2] -= thickness * 0.5
            meshes.append(prism)

        # Merge all prisms into one mesh
        mesh = trimesh.util.concatenate(meshes)

        # Optional cleanup (safe)
        # mesh.remove_duplicate_faces()
        # mesh.remove_unreferenced_vertices()
        # mesh.process(validate=True)

        print(f"  Mesh vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")

        # Torch tensors (from mesh)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        verts = torch.from_numpy(mesh.vertices).float().to(device)
        faces = torch.from_numpy(mesh.faces).long().to(device)

        data = {
            "type": "2d",
            "mesh": mesh,        # for ray tracing
            "vertices": verts,   # for PCA
            "faces": faces,
        }

        return data
        
    else:  # 3d
        # Locate mesh
        mesh_path = Path("data") / "meshes" / f"{model}.ply"
        if not mesh_path.exists():
            mesh_path = Path("data") / "stanford" / f"{model}.ply"
            if not mesh_path.exists():
                print(f"ERROR: Mesh not found: {model}.ply in data/meshes/ or data/stanford/")
                return None

        print(f"  Loading mesh from: {mesh_path}")

        # Load mesh exactly as-is
        mesh = trimesh.load(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            print("ERROR: Loaded object is not a triangle mesh")
            return None

        print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

        # ------------------------------------------------------------------
        # Normalize mesh to [-1, 1] with padding (APPLY TO TRIMESH ITSELF)
        # ------------------------------------------------------------------
        padding = 0.1

        verts_np = mesh.vertices.astype("float32")

        # Center by bounding box center (not centroid) for symmetric padding
        min_coords = verts_np.min(axis=0)
        max_coords = verts_np.max(axis=0)
        bbox_center = (min_coords + max_coords) / 2
        verts_np = verts_np - bbox_center

        # Now compute extents and scale
        extents = max_coords - min_coords
        max_extent = extents.max()

        if max_extent > 0:
            scale = 2.0 * (1.0 - padding) / max_extent
            verts_np = verts_np * scale

        # Write back to trimesh
        mesh.vertices = verts_np

        # ------------------------------------------------------------------
        # Create torch tensors FROM the normalized mesh
        # ------------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        verts = torch.from_numpy(mesh.vertices).float().to(device)
        faces = torch.from_numpy(mesh.faces).long().to(device)
        
        # also load volume/surface points
        pcd_surf_path = Path("data") / "point_clouds" / f"{mesh_path.stem}_surf.ply"
        assert pcd_surf_path.exists(), f"Surface point cloud not found: {pcd_surf_path}"
        pcd_surf = o3d.t.io.read_point_cloud(pcd_surf_path)
        surf_points = pcd_surf.point.positions.numpy()
        
        pcd_vol_path = Path("data") / "point_clouds" / f"{mesh_path.stem}_vol.ply"
        assert pcd_vol_path.exists(), f"Surface point cloud not found: {pcd_vol_path}"
        pcd_vol = o3d.t.io.read_point_cloud(pcd_surf_path)
        vol_points = pcd_vol.point.positions.numpy()
        vol_sdf = pcd_vol.point.signed_distances.numpy().squeeze()
        
        data = {
            'type': '3d',
            'mesh': mesh,
            'vertices': verts,
            'faces': faces,
            # these 3 are needed for evaluation
            'pcd_surf': torch.tensor(surf_points).to(device),
            'pcd_vol': torch.tensor(vol_points).to(device),
            'pcd_vol_sdf': torch.tensor(vol_sdf).to(device),
        }
        
        return data


def create_polygons2d(spec, convex=True, star_ratio=0.5, stretch=(1.0, 1.0), rotation=0.0) -> List[np.ndarray]:
    """
    Generate non-overlapping polygons in [-1, 1]^2 based on a specification string.
    
    Parameters:
    -----------
    spec : str
        Specification string in format "n_polygons x vertices_per_polygon"
        Examples: "3x4" (3 polygons with 4 vertices each)
                  "4x5" (4 polygons with 5 vertices each)
        
        Alternatively, can specify different vertices per polygon:
        "4,3,5" (3 polygons with 4, 3, and 5 vertices respectively)
    
    convex : bool
        If True, generate convex polygons (regular polygons)
        If False, generate non-convex star-like polygons
    
    star_ratio : float (0 to 1)
        For non-convex polygons, ratio of inner radius to outer radius
        Smaller values create more pronounced star shapes
        Only used when convex=False
    
    stretch : tuple, list, or float
        Stretching factor for polygons in (x, y) directions.
        - Single float: uniform scaling (e.g., 1.5)
        - Tuple (sx, sy): stretch all polygons by sx in x and sy in y
        - List of tuples: per-polygon stretch factors
        Examples: (2.0, 1.0) stretches 2x in x-direction (rectangles from squares)
                  (1.0, 0.5) compresses in y-direction (flattened shapes)
    
    rotation : float, list, or np.ndarray
        Rotation angle(s) in radians for the polygons.
        - Single float: rotate all polygons by the same angle
        - List/array: per-polygon rotation angles
        Examples: 0.5 (rotate all by 0.5 radians)
                  [0, np.pi/4, np.pi/2] (different rotation per polygon)
    
    Returns:
    --------
    list of np.ndarray
        List of polygon vertex arrays, each of shape (n_vertices, 2)
    """
    # Parse the specification
    if 'x' in spec:
        # Format: "n_polygons x vertices_per_polygon"
        parts = spec.split('x')
        n_polygons = int(parts[0])
        vertices_per_polygon = int(parts[1])
        vertices_list = [vertices_per_polygon] * n_polygons
    else:
        # Format: "v1,v2,v3,..."
        vertices_list = [int(v) for v in spec.split(',')]
        n_polygons = len(vertices_list)
    
    # Parse stretch parameter
    if isinstance(stretch, (int, float)):
        # Single float: uniform scaling
        stretch_list = [(stretch, stretch)] * n_polygons
    elif isinstance(stretch, tuple):
        # Tuple: same stretch for all polygons
        stretch_list = [stretch] * n_polygons
    elif isinstance(stretch, list):
        # List: per-polygon stretches
        stretch_list = stretch
        if len(stretch_list) < n_polygons:
            # Extend with (1.0, 1.0) if not enough specified
            stretch_list.extend([(1.0, 1.0)] * (n_polygons - len(stretch_list)))
    else:
        stretch_list = [(1.0, 1.0)] * n_polygons
    
    # Parse rotation parameter
    if isinstance(rotation, (int, float)):
        # Single value: same rotation for all polygons
        rotation_list = [float(rotation)] * n_polygons
    elif isinstance(rotation, (list, np.ndarray)):
        # List/array: per-polygon rotations
        rotation_list = list(rotation)
        if len(rotation_list) < n_polygons:
            # Extend with 0.0 if not enough specified
            rotation_list.extend([0.0] * (n_polygons - len(rotation_list)))
    else:
        rotation_list = [0.0] * n_polygons
    
    # Determine grid layout to fit polygons without overlap
    grid_cols = int(np.ceil(np.sqrt(n_polygons)))
    grid_rows = int(np.ceil(n_polygons / grid_cols))
    
    # Cell dimensions (with padding)
    padding = 0.05
    cell_width = 2.0 / grid_cols
    cell_height = 2.0 / grid_rows
    
    polygons = []
    
    for idx, n_vertices in enumerate(vertices_list):
        # Determine cell position
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Cell center in [-1, 1]^2
        cell_center_x = -1.0 + cell_width * (col + 0.5)
        cell_center_y = -1.0 + cell_height * (row + 0.5)
        
        # Get stretch factors for this polygon
        sx, sy = stretch_list[idx]
        
        # Get rotation angle for this polygon
        angle_offset = rotation_list[idx]
        
        # Polygon radius (inscribed in cell with padding, accounting for stretch)
        # Ensure stretched polygon fits within cell boundaries
        radius_x = (cell_width / 2) * 0.8 / sx if sx > 0 else cell_width * 0.4
        radius_y = (cell_height / 2) * 0.8 / sy if sy > 0 else cell_height * 0.4
        radius = min(radius_x, radius_y)
        
        if convex:
            # Generate regular convex polygon vertices (centered at origin)
            angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
            angles += idx * 0.3
            
            vertices = np.zeros((n_vertices, 2))
            vertices[:, 0] = radius * sx * np.cos(angles)
            vertices[:, 1] = radius * sy * np.sin(angles)
        else:
            # Generate non-convex star-like polygon (centered at origin)
            n_points = n_vertices * 2
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            angles += idx * 0.3
            
            # Alternate between outer and inner radii
            radii = np.zeros(n_points)
            radii[::2] = radius
            radii[1::2] = radius * star_ratio
            
            vertices = np.zeros((n_points, 2))
            vertices[:, 0] = radii * sx * np.cos(angles)
            vertices[:, 1] = radii * sy * np.sin(angles)
        
        # Apply rotation using rotation matrix (after stretch)
        if angle_offset != 0:
            cos_a = np.cos(angle_offset)
            sin_a = np.sin(angle_offset)
            rotation_matrix = np.array([[cos_a, -sin_a],
                                       [sin_a, cos_a]])
            vertices = vertices @ rotation_matrix.T
        
        # Translate to cell center
        vertices[:, 0] += cell_center_x
        vertices[:, 1] += cell_center_y
        
        polygons.append(vertices)
    
    return polygons


def extract_mesh_marching_cubes(model, save_path: Optional[Path] = None, resolution: int = 128, device=None, batch_size: int = 100000):
    """
    Extract 3D mesh from implicit model using marching cubes with batched evaluation.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Neural network that outputs SDF values
    save_path : Path, optional
        Path to save the extracted mesh (as .ply file)
    resolution : int
        Grid resolution for marching cubes (default: 128)
    device : torch.device, optional
        Device to run evaluation on
    batch_size : int
        Number of points to evaluate at once (default: 100000)
    
    Returns:
    --------
    tuple
        (vertices, faces, normals) numpy arrays from marching cubes
    """
    if device is None:
        device = next(model.parameters()).device
    
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    total_points = grid_points.shape[0]
    
    # Evaluate in batches to avoid OOM
    sdf_values = []
    with torch.no_grad():
        for i in range(0, total_points, batch_size):
            batch = grid_points[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            batch_sdf = model(batch_tensor).squeeze().cpu().numpy()
            sdf_values.append(batch_sdf)
    
    sdf_values = np.concatenate(sdf_values)
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    vertices_mc, faces_mc, normals_mc, _ = measure.marching_cubes(
        sdf_grid, 
        level=0.0, 
        spacing=(2/resolution, 2/resolution, 2/resolution)
    )
    
    vertices_mc -= 1.0
    
    if save_path is not None:
        mesh = trimesh.Trimesh(vertices=vertices_mc, faces=faces_mc, vertex_normals=normals_mc)
        mesh.export(save_path)
    
    return vertices_mc, faces_mc, normals_mc

