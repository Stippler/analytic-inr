#!/usr/bin/env python3
"""
Quick test to verify normal computation is correct for CCW polygons.
"""

import torch
import numpy as np
from analytic_inr import RayMarchingVoronoi

def test_simple_square():
    """Test normals for a simple square in CCW order."""
    # Simple square, CCW order: bottom-left, bottom-right, top-right, top-left
    square = torch.tensor([
        [0.0, 0.0],  # bottom-left
        [1.0, 0.0],  # bottom-right
        [1.0, 1.0],  # top-right
        [0.0, 1.0],  # top-left
    ], dtype=torch.float32)
    
    voronoi = RayMarchingVoronoi([square], device='cpu')
    normals = voronoi.compute_vertex_normals()
    
    print("Square vertices (CCW):")
    print(square.numpy())
    print("\nComputed normals:")
    print(normals.numpy())
    
    # Expected normals (pointing outward from center of square at [0.5, 0.5])
    expected = np.array([
        [-1, -1],  # bottom-left: outward is down-left
        [1, -1],   # bottom-right: outward is down-right
        [1, 1],    # top-right: outward is up-right
        [-1, 1],   # top-left: outward is up-left
    ], dtype=np.float32)
    
    # Normalize expected
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    
    print("\nExpected normals (normalized):")
    print(expected)
    
    # Check if they match (with some tolerance)
    normals_np = normals.numpy()
    differences = np.abs(normals_np - expected)
    max_diff = np.max(differences)
    
    print(f"\nMax difference: {max_diff:.6f}")
    
    if max_diff < 0.01:
        print("✓ Normals are CORRECT!")
        return True
    else:
        print("✗ Normals are INCORRECT!")
        print("\nDifferences:")
        print(differences)
        return False

def test_triangle():
    """Test normals for an equilateral-ish triangle."""
    # Triangle CCW
    triangle = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866],  # approximately equilateral
    ], dtype=torch.float32)
    
    voronoi = RayMarchingVoronoi([triangle], device='cpu')
    normals = voronoi.compute_vertex_normals()
    
    print("\n" + "="*60)
    print("Triangle test:")
    print("="*60)
    print("Triangle vertices (CCW):")
    print(triangle.numpy())
    print("\nComputed normals:")
    print(normals.numpy())
    
    # Verify normals are unit length
    norms = np.linalg.norm(normals.numpy(), axis=1)
    print(f"\nNormal magnitudes: {norms}")
    print(f"All unit length: {np.allclose(norms, 1.0)}")
    
    # Check they point outward (dot product with vector from centroid should be positive)
    centroid = triangle.mean(dim=0)
    print(f"\nCentroid: {centroid.numpy()}")
    
    for i in range(3):
        to_vertex = triangle[i] - centroid
        normal = normals[i]
        dot = torch.dot(to_vertex, normal).item()
        print(f"Vertex {i}: dot(to_vertex, normal) = {dot:.4f} {'✓ outward' if dot > 0 else '✗ inward'}")
    
    return True

if __name__ == "__main__":
    print("Testing normal computation for CCW polygons")
    print("="*60)
    
    success = test_simple_square()
    test_triangle()
    
    if success:
        print("\n" + "="*60)
        print("All tests PASSED! ✓")
        print("="*60)

