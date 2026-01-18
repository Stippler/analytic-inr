"""
Integration test for sphere tracing implementation.

Run with:
    conda activate pytorch3d
    python test_sphere_tracing.py
"""

import torch
import numpy as np
from neural_spline.spline import lift_polygons_to_mesh, surface_query, sphere_trace_rays, compute_sdf
from neural_spline.types import PCAComponent
from pytorch3d.structures import Meshes


def test_polygon_lifting():
    """Test 2D polygon to 3D mesh lifting."""
    print('='*60)
    print('Testing 2D Polygon Lifting')
    print('='*60)
    
    # Simple square polygon
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    verts, faces = lift_polygons_to_mesh([square], epsilon=1e-3, device='cuda')
    print(f'Square lifted: {len(verts)} verts, {len(faces)} faces')
    assert len(verts) == 16, f'Expected 16 vertices (4 edges × 4 verts), got {len(verts)}'
    assert len(faces) == 8, f'Expected 8 faces (4 edges × 2 triangles), got {len(faces)}'
    print('✓ Polygon lifting works\n')
    return verts, faces


def test_surface_query(verts, faces):
    """Test surface query functionality."""
    print('='*60)
    print('Testing Surface Query')
    print('='*60)
    
    # Move to CUDA
    verts_cuda = verts.cuda()
    faces_cuda = faces.cuda()
    mesh = Meshes(verts=[verts_cuda], faces=[faces_cuda])
    query_points = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32).cuda()
    distances, tri_ids, normals = surface_query(query_points, mesh, chunk_size=100)
    print(f'Distances: {distances}')
    print(f'Normals shape: {normals.shape}')
    print(f'Normals (first 2): {normals[:2]}')
    print(f'Normal lengths: {torch.norm(normals, dim=1)}')
    
    # Check normals are in XY plane (z component should be very small)
    z_components = torch.abs(normals[:, 2])
    print(f'Z-components of normals: {z_components} (should be ~0 for 2D)')
    assert z_components.max() < 0.1, f'Z-components should be near 0 for 2D, got max {z_components.max()}'
    print('✓ Surface query works\n')
    return mesh


def test_sphere_tracing(mesh):
    """Test sphere tracing along a ray."""
    print('='*60)
    print('Testing Sphere Tracing')
    print('='*60)
    
    # Single ray from outside to outside
    p0 = torch.tensor([[-0.5, 0.5, 0.0]], dtype=torch.float32).cuda()
    p1 = torch.tensor([[1.5, 0.5, 0.0]], dtype=torch.float32).cuda()
    t_list, sdf_list, normals_list = sphere_trace_rays(
        p0, p1, mesh, 
        alpha=0.8, 
        min_step=0.001, 
        max_step=0.1,
        max_steps_per_ray=100,
        device='cuda',
        verbose=True
    )
    print(f'Ray traced: {len(t_list[0])} samples')
    print(f'Sample t values (first 5): {t_list[0][:5]}')
    print(f'Sample SDF values (first 5): {sdf_list[0][:5]}')
    print('✓ Sphere tracing works\n')


def test_full_pipeline():
    """Test the complete compute_sdf pipeline."""
    print('='*60)
    print('Testing Full compute_sdf Pipeline (2D)')
    print('='*60)
    
    # Create test component
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    component = PCAComponent(
        start=torch.tensor([0.5, 0.5]),
        end=torch.tensor([1.5, 0.5]),
        variance=1.0,
        component_idx=0,
        depth=0,
        label='test'
    )
    data = {'type': '2d', 'polygons': [square]}
    
    t_vals, sdf_vals, normals_vals = compute_sdf(
        components=[component],
        data=data,
        device_id=0,  # Use CUDA device 0
        return_normals=True,
        verbose=True
    )
    print(f'Pipeline produced {len(t_vals[0])} samples')
    print(f'Sample distances (first 5): {sdf_vals[0][:5]}')
    print(f'Normals shape: {normals_vals[0].shape}')
    print('✓ Full pipeline works\n')


def test_3d_mesh():
    """Test with a simple 3D mesh."""
    print('='*60)
    print('Testing 3D Mesh (cube)')
    print('='*60)
    
    # Simple cube
    verts = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=torch.float32)
    
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ], dtype=torch.int64)
    
    component = PCAComponent(
        start=torch.tensor([-2.0, 0.0, 0.0]),
        end=torch.tensor([2.0, 0.0, 0.0]),
        variance=1.0,
        component_idx=0,
        depth=0,
        label='test_3d'
    )
    
    data = {'type': '3d', 'verts': verts, 'faces': faces}
    
    t_vals, sdf_vals, normals_vals = compute_sdf(
        components=[component],
        data=data,
        device_id=0,  # Use CUDA device 0
        return_normals=True,
        verbose=True
    )
    
    print(f'3D pipeline produced {len(t_vals[0])} samples')
    print(f'Sample distances (first 5): {sdf_vals[0][:5]}')
    print(f'Normals shape: {normals_vals[0].shape}')
    
    # Check normals are unit length
    all_normals = normals_vals[0]
    lengths = torch.norm(all_normals, dim=1)
    print(f'Normal lengths: mean={lengths.mean():.6f}, std={lengths.std():.6f}')
    assert (lengths > 0.99).all() and (lengths < 1.01).all(), "Normals should be unit length"
    print('✓ 3D mesh works\n')


if __name__ == '__main__':
    print('\n' + '='*60)
    print('SPHERE TRACING INTEGRATION TESTS')
    print('='*60 + '\n')
    
    try:
        verts, faces = test_polygon_lifting()
        mesh = test_surface_query(verts, faces)
        test_sphere_tracing(mesh)
        test_full_pipeline()
        test_3d_mesh()
        
        print('='*60)
        print('ALL TESTS PASSED!')
        print('='*60)
    except Exception as e:
        print(f'\n❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
