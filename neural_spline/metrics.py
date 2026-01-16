"""
Metrics for evaluating Implicit Neural Representation (INR) quality for SDF prediction.

These metrics compare the learned SDF from the neural network against the ground truth SDF
computed from polygons/meshes. Inspired by metrics used in SIREN, DeepSDF, and related work.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes


def _compute_gt_sdf_batch_gpu(mesh, points_np, device, batch_size=50000):
    """
    Compute ground truth SDF using GPU-accelerated distance computation.
    
    Args:
        mesh: trimesh.Trimesh object
        points_np: (N, 3) numpy array of query points
        device: torch device
        batch_size: Process points in batches to avoid OOM
    
    Returns:
        sdf: (N,) numpy array of signed distance values
    """
    n_points = len(points_np)
    sdf_values = np.zeros(n_points, dtype=np.float32)
    
    # Convert mesh to PyTorch3D format once
    verts_gpu = torch.tensor(mesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)  # (1, V, 3)
    faces_gpu = torch.tensor(mesh.faces, dtype=torch.int64, device=device).unsqueeze(0)  # (1, F, 3)
    mesh_pytorch3d = Meshes(verts=verts_gpu, faces=faces_gpu)
    mesh_samples = sample_points_from_meshes(mesh_pytorch3d, num_samples=50000)  # (1, N_samples, 3)
    
    # Process in batches to avoid OOM
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        batch_points = points_np[i:end_idx]
        
        # GPU distance computation using KNN
        points_gpu = torch.from_numpy(batch_points).float().to(device).unsqueeze(0)  # (1, N, 3)
        knn_result = knn_points(points_gpu, mesh_samples, K=1)
        distances = torch.sqrt(knn_result.dists[0, :, 0]).cpu().numpy()
        
        # CPU inside/outside check (still faster overall)
        inside = mesh.contains(batch_points)
        
        # Signed distance
        sdf_batch = distances
        sdf_batch[inside] *= -1
        sdf_values[i:end_idx] = sdf_batch
    
    return sdf_values


def compute_ground_truth_sdf_2d(points: np.ndarray, polygons: List[np.ndarray]) -> np.ndarray:
    """
    Compute ground truth SDF for 2D points given polygons.
    
    Args:
        points: (N, 2) array of query points
        polygons: List of polygon vertex arrays
    
    Returns:
        sdf: (N,) array of signed distance values
    """
    sdf = np.full(len(points), np.inf)
    
    for poly in polygons:
        v = np.asarray(poly)
        
        # Compute distance to this polygon's edges
        poly_distances = np.full(len(points), np.inf)
        for i in range(len(v)):
            v1, v2 = v[i], v[(i + 1) % len(v)]
            edge_vec = v2 - v1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            
            if edge_len_sq < 1e-10:
                continue
            
            t_closest = np.clip(np.dot(points - v1, edge_vec) / edge_len_sq, 0, 1)
            closest_points = v1 + t_closest[:, None] * edge_vec
            edge_distances = np.linalg.norm(points - closest_points, axis=1)
            poly_distances = np.minimum(poly_distances, edge_distances)
        
        # Ray casting for inside/outside
        v_next = np.roll(v, -1, axis=0)
        py = points[:, 1:2]
        px = points[:, 0:1]
        
        v_y, v_next_y = v[:, 1], v_next[:, 1]
        v_x, v_next_x = v[:, 0], v_next[:, 0]
        
        # Edges that cross the horizontal ray
        crosses = ((v_y <= py) & (py < v_next_y)) | ((v_next_y <= py) & (py < v_y))
        
        # Safe division
        dy = v_next_y - v_y
        valid = np.abs(dy) > 1e-12
        safe_dy = np.where(valid, dy, 1.0)
        
        t_cross = (py - v_y) / safe_dy
        x_cross = v_x + t_cross * (v_next_x - v_x)
        
        crossings_right = crosses & valid & (x_cross > px)
        inside_poly = np.sum(crossings_right, axis=1) % 2 == 1
        
        poly_sdf = np.where(inside_poly, -poly_distances, poly_distances)
        
        # Union: take minimum SDF
        sdf = np.minimum(sdf, poly_sdf)
    
    return sdf


def compute_sdf_metrics(
    model,
    polygons: List[np.ndarray],
    n_samples: int = 10000,
    bbox_min: float = -1.0,
    bbox_max: float = 1.0,
    device=None
) -> Dict[str, float]:
    """
    Compute comprehensive SDF quality metrics for a 2D INR model.
    
    Metrics computed:
    - MAE (Mean Absolute Error): Average absolute difference between predicted and GT SDF
    - RMSE (Root Mean Square Error): sqrt(mean squared error)
    - Max Error: Maximum absolute error
    - Median Error: Median absolute error
    - IoU: Intersection over Union of interior regions (SDF < 0)
    - Boundary IoU: IoU near the boundary (|SDF| < threshold)
    - Sign Accuracy: Percentage of points with correct sign (inside/outside)
    
    Args:
        model: Neural network model (ReluMLP)
        polygons: List of ground truth polygon vertex arrays
        n_samples: Number of random points to sample for evaluation
        bbox_min: Minimum coordinate of bounding box
        bbox_max: Maximum coordinate of bounding box
        device: Device to run computations on
    
    Returns:
        Dictionary of metric names to values
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample random points in the domain
    points_np = np.random.uniform(bbox_min, bbox_max, (n_samples, 2))
    
    # Compute ground truth SDF
    gt_sdf = compute_ground_truth_sdf_2d(points_np, polygons)
    
    # Compute predicted SDF
    points_torch = torch.from_numpy(points_np).float().to(device)
    with torch.no_grad():
        model = model.to(device)
        pred_sdf = model(points_torch).squeeze().cpu().numpy()
    
    # Compute metrics
    abs_errors = np.abs(pred_sdf - gt_sdf)
    squared_errors = (pred_sdf - gt_sdf) ** 2
    
    metrics = {
        # Basic error metrics
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(squared_errors))),
        'max_error': float(np.max(abs_errors)),
        'median_error': float(np.median(abs_errors)),
        'std_error': float(np.std(abs_errors)),
        
        # Sign accuracy (inside/outside classification)
        'sign_accuracy': float(np.mean((np.sign(pred_sdf) == np.sign(gt_sdf)))),
        
        # IoU metrics
        'iou_interior': compute_iou(pred_sdf < 0, gt_sdf < 0),
        'iou_boundary_01': compute_boundary_iou(pred_sdf, gt_sdf, threshold=0.1),
        'iou_boundary_005': compute_boundary_iou(pred_sdf, gt_sdf, threshold=0.05),
        
        # Error distribution (percentiles)
        'error_95th_percentile': float(np.percentile(abs_errors, 95)),
        'error_99th_percentile': float(np.percentile(abs_errors, 99)),
        
        # Separate interior/exterior metrics
        'mae_interior': float(np.mean(abs_errors[gt_sdf < 0])) if np.any(gt_sdf < 0) else 0.0,
        'mae_exterior': float(np.mean(abs_errors[gt_sdf >= 0])) if np.any(gt_sdf >= 0) else 0.0,
        
        # Near-surface accuracy (important for visualization)
        'mae_near_surface': compute_near_surface_error(pred_sdf, gt_sdf, threshold=0.1),
    }
    
    return metrics


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union for binary masks.
    
    Args:
        pred_mask: Predicted binary mask (boolean or 0/1)
        gt_mask: Ground truth binary mask (boolean or 0/1)
    
    Returns:
        IoU value between 0 and 1
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0  # Both empty
    
    return float(intersection / union)


def compute_boundary_iou(pred_sdf: np.ndarray, gt_sdf: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute IoU for the boundary region (points near the zero level set).
    
    This is particularly important for evaluating how well the INR captures
    the shape boundary.
    
    Args:
        pred_sdf: Predicted SDF values
        gt_sdf: Ground truth SDF values
        threshold: Distance threshold for "near boundary"
    
    Returns:
        IoU for boundary region
    """
    pred_boundary = np.abs(pred_sdf) < threshold
    gt_boundary = np.abs(gt_sdf) < threshold
    
    return compute_iou(pred_boundary, gt_boundary)


def compute_near_surface_error(pred_sdf: np.ndarray, gt_sdf: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute mean absolute error only for points near the surface.
    
    This metric focuses on how well the network captures the exact boundary location,
    which is critical for visualization and downstream applications.
    
    Args:
        pred_sdf: Predicted SDF values
        gt_sdf: Ground truth SDF values  
        threshold: Distance threshold for "near surface"
    
    Returns:
        MAE for points near the surface
    """
    near_surface = np.abs(gt_sdf) < threshold
    
    if not np.any(near_surface):
        return 0.0
    
    return float(np.mean(np.abs(pred_sdf[near_surface] - gt_sdf[near_surface])))


def compute_chamfer_distance_2d(
    pred_boundary_points: np.ndarray,
    gt_boundary_points: np.ndarray
) -> Tuple[float, float]:
    """
    Compute bidirectional Chamfer distance between predicted and GT boundary points.
    
    This measures how close the predicted zero level set is to the ground truth boundary.
    
    Args:
        pred_boundary_points: (N, 2) predicted boundary points
        gt_boundary_points: (M, 2) ground truth boundary points
    
    Returns:
        Tuple of (pred_to_gt_distance, gt_to_pred_distance)
    """
    if len(pred_boundary_points) == 0 or len(gt_boundary_points) == 0:
        return float('inf'), float('inf')
    
    # Pred to GT
    dists_pred_to_gt = np.min(
        np.linalg.norm(
            pred_boundary_points[:, None, :] - gt_boundary_points[None, :, :],
            axis=2
        ),
        axis=1
    )
    pred_to_gt = np.mean(dists_pred_to_gt)
    
    # GT to Pred
    dists_gt_to_pred = np.min(
        np.linalg.norm(
            gt_boundary_points[:, None, :] - pred_boundary_points[None, :, :],
            axis=2
        ),
        axis=1
    )
    gt_to_pred = np.mean(dists_gt_to_pred)
    
    return float(pred_to_gt), float(gt_to_pred)


def extract_zero_level_set_2d(
    model,
    resolution: int = 200,
    bbox_min: float = -1.0,
    bbox_max: float = 1.0,
    device=None
) -> np.ndarray:
    """
    Extract the zero level set (boundary) from the learned SDF using marching squares.
    
    Args:
        model: Neural network model
        resolution: Grid resolution for extraction
        bbox_min: Minimum coordinate
        bbox_max: Maximum coordinate
        device: Device to run on
    
    Returns:
        Boundary points as (N, 2) array
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create grid
    x = torch.linspace(bbox_min, bbox_max, resolution)
    y = torch.linspace(bbox_min, bbox_max, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    
    # Evaluate SDF on grid
    with torch.no_grad():
        model = model.to(device)
        sdf_vals = model(grid_pts).squeeze().cpu().numpy()
    
    sdf_grid = sdf_vals.reshape(resolution, resolution)
    
    # Simple boundary extraction: find points near zero
    # For a more sophisticated approach, use skimage.measure.find_contours
    threshold = 0.01
    near_zero = np.abs(sdf_grid) < threshold
    
    # Get coordinates of near-zero points
    y_coords, x_coords = np.where(near_zero)
    
    if len(x_coords) == 0:
        return np.array([]).reshape(0, 2)
    
    # Convert to actual coordinates
    x_range = np.linspace(bbox_min, bbox_max, resolution)
    y_range = np.linspace(bbox_min, bbox_max, resolution)
    
    boundary_points = np.stack([
        x_range[x_coords],
        y_range[y_coords]
    ], axis=1)
    
    return boundary_points


def compute_sdf_metrics_3d(
    model,
    mesh,
    n_samples: int = 10000,
    bbox_min: float = -1.0,
    bbox_max: float = 1.0,
    device=None
) -> Dict[str, float]:
    """
    Compute comprehensive SDF quality metrics for a 3D INR model.
    Uses GPU-accelerated batch computation with PyTorch3D.
    
    Args:
        model: Neural network model (ReluMLP)
        mesh: Trimesh mesh object
        n_samples: Number of random points to sample
        bbox_min: Minimum coordinate of bounding box
        bbox_max: Maximum coordinate of bounding box
        device: Device to run computations on
    
    Returns:
        Dictionary of metric names to values
    """
    print(f"  Sampling {n_samples} random points...")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample random points in the domain
    points_np = np.random.uniform(bbox_min, bbox_max, (n_samples, 3))
    
    # Compute ground truth SDF using GPU-accelerated method
    print("  Computing ground truth SDF using GPU-accelerated method...")
    gt_sdf = _compute_gt_sdf_batch_gpu(mesh, points_np, device)
    
    # Compute predicted SDF
    print("  Computing predicted SDF from model...")
    points_torch = torch.from_numpy(points_np).float().to(device)
    with torch.no_grad():
        model = model.to(device)
        pred_sdf = model(points_torch).squeeze().cpu().numpy()
    
    # Compute metrics
    print("  Computing error metrics...")
    abs_errors = np.abs(pred_sdf - gt_sdf)
    squared_errors = (pred_sdf - gt_sdf) ** 2
    
    metrics = {
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(squared_errors))),
        'max_error': float(np.max(abs_errors)),
        'median_error': float(np.median(abs_errors)),
        'std_error': float(np.std(abs_errors)),
        'sign_accuracy': float(np.mean((np.sign(pred_sdf) == np.sign(gt_sdf)))),
        'iou_interior': compute_iou(pred_sdf < 0, gt_sdf < 0),
        'iou_boundary_01': compute_boundary_iou(pred_sdf, gt_sdf, threshold=0.1),
        'iou_boundary_005': compute_boundary_iou(pred_sdf, gt_sdf, threshold=0.05),
        'error_95th_percentile': float(np.percentile(abs_errors, 95)),
        'error_99th_percentile': float(np.percentile(abs_errors, 99)),
        'mae_interior': float(np.mean(abs_errors[gt_sdf < 0])) if np.any(gt_sdf < 0) else 0.0,
        'mae_exterior': float(np.mean(abs_errors[gt_sdf >= 0])) if np.any(gt_sdf >= 0) else 0.0,
        'mae_near_surface': compute_near_surface_error(pred_sdf, gt_sdf, threshold=0.1),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "SDF Metrics"):
    """
    Pretty print metrics dictionary.
    
    Args:
        metrics: Dictionary of metric names to values
        title: Title to print above metrics
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Group metrics by category
    basic_metrics = ['mae', 'rmse', 'max_error', 'median_error', 'std_error']
    accuracy_metrics = ['sign_accuracy']
    iou_metrics = ['iou_interior', 'iou_boundary_01', 'iou_boundary_005']
    percentile_metrics = ['error_95th_percentile', 'error_99th_percentile']
    region_metrics = ['mae_interior', 'mae_exterior', 'mae_near_surface']
    
    def print_group(group_name: str, metric_keys: List[str]):
        print(f"\n{group_name}:")
        for key in metric_keys:
            if key in metrics:
                value = metrics[key]
                # Format based on typical value ranges
                if 'accuracy' in key or 'iou' in key:
                    print(f"  {key:30s}: {value:7.4f} ({value*100:.2f}%)")
                else:
                    print(f"  {key:30s}: {value:7.6f}")
    
    print_group("Basic Error Metrics", basic_metrics)
    print_group("Classification Metrics", accuracy_metrics)
    print_group("IoU Metrics", iou_metrics)
    print_group("Error Distribution", percentile_metrics)
    print_group("Region-Specific Metrics", region_metrics)
    
    print(f"{'='*60}\n")

