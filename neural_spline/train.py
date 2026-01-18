"""
Training loop and related functions for neural spline learning.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List
from tqdm import tqdm

from .model import ReluMLP

"""
Spline dataclass for representing 1D line segments with ground truth and predicted values.
"""

import torch
from dataclasses import dataclass


@dataclass
class Spline:
    """Represents a 1D spline along a line segment."""
    start_point: torch.Tensor  # (2,) or (3,) start point in space
    end_point: torch.Tensor    # (2,) or (3,) end point in space
    
    pred_knots: torch.Tensor   # (N,) predicted knot positions along [0, 1]
    pred_values: torch.Tensor  # (N,) predicted SDF values at knots
    
    gt_knots: torch.Tensor     # (M,) ground truth knot positions
    gt_values: torch.Tensor    # (M,) ground truth SDF values
    
    # Metadata for visualization and analysis
    label: str = ""
    depth: int = 0
    pc_type: str = "pc1"




def sort_and_drop_duplicates(t_candidates):
    """Sort tensor and drop duplicate values."""
    t_candidates, _ = torch.sort(t_candidates)
    if len(t_candidates) > 1:
        # Keep first element and any element that differs from previous by > threshold
        mask = torch.cat([
            torch.tensor([True], device=t_candidates.device),
            torch.abs(t_candidates[1:] - t_candidates[:-1]) > 1e-6
        ])
        t_candidates = t_candidates[mask]
    return t_candidates


def get_spline_value(t_query, knots, values):
    """Linear interpolation to find height at specific t points."""
    # Find segment indices
    indices = torch.searchsorted(knots, t_query)
    indices = indices.clamp(1, len(knots) - 1)
    
    t_left = knots[indices - 1]
    t_right = knots[indices]
    h_left = values[indices - 1]
    h_right = values[indices]
    
    # Interpolate
    denom = (t_right - t_left) + 1e-8
    alpha = (t_query - t_left) / denom
    return h_left + alpha * (h_right - h_left)


def calculate_exact_integral_loss(pred_t, pred_h, gt_t, gt_h):
    """
    Calculates the exact integral of (f_pred(t) - f_gt(t))^2 dt.
    """
    # 1. MERGE KNOTS
    t_union = torch.cat([pred_t, gt_t])
    t_union, _ = torch.sort(t_union)
    
    # Remove duplicates to avoid div/0
    mask = torch.cat([
        torch.tensor([True], device=t_union.device),
        (t_union[1:] - t_union[:-1]) > 1e-6
    ])
    t_union = t_union[mask]

    # 2. EVALUATE BOTH SPLINES ON UNION GRID
    h_on_grid_pred = get_spline_value(t_union, pred_t, pred_h)
    h_on_grid_gt = get_spline_value(t_union, gt_t, gt_h)
    
    # 3. CALCULATE DIFFERENCE
    diff = h_on_grid_pred - h_on_grid_gt
    
    # 4. INTEGRATE EXACTLY using trapezoidal rule for linear segments squared
    dt = t_union[1:] - t_union[:-1]
    y_left = diff[:-1]
    y_right = diff[1:]
    
    # Simpson's rule for linear difference squared
    segment_areas = (dt / 3.0) * (y_left**2 + y_left * y_right + y_right**2)
    
    total_integral = torch.sum(segment_areas)
    
    return total_integral


def reset_predictions(mlp: ReluMLP, splines: List[Spline]):
    """
    Reset predictions for all splines by analytically computing knots through the network.
    This is the key to making the network learn implicit representations efficiently.
    """
    for spline in splines:
        W0 = mlp.layers[0].weight
        b0 = mlp.layers[0].bias

        p0 = spline.start_point
        p1 = spline.end_point
        d = p1 - p0

        alpha = W0 @ d
        beta = W0 @ p0 + b0

        eps_safe = 1e-5
        eps = 1e-8

        t_start = torch.tensor(0.0, dtype=torch.float32)
        t_end = torch.tensor(1.0, dtype=torch.float32)
        valid_mask = torch.abs(alpha) > eps_safe
        t_new = -beta[valid_mask] / alpha[valid_mask]

        mask = (t_new > t_start + eps) & (t_new < t_end - eps)
        t_new = t_new[mask]
        t_new = sort_and_drop_duplicates(t_new)

        if len(t_new) > 0:
            t_knots = torch.cat([t_start.unsqueeze(0), t_new, t_end.unsqueeze(0)])
        else:
            t_knots = torch.stack([t_start, t_end])

        h_values = (alpha.unsqueeze(1) * t_knots.unsqueeze(0) + beta.unsqueeze(1)).T

        # Propagate through hidden layers
        for layer in mlp.layers[1:-1]:
            W = layer.weight
            b = layer.bias

            h_values = torch.relu(h_values)
            h_values = (W @ h_values.T).T + b

            t_new = []
            h_new = []
            for t_start, t_end, h_start, h_end in zip(t_knots[:-1], t_knots[1:], h_values[:-1], h_values[1:]):
                t_new.append(t_start.unsqueeze(0))
                h_new.append(h_start.unsqueeze(0))

                # Find all flipping h values (zero crossings)
                sign_flips = (h_start * h_end) < 0
                flip_idx = torch.where(sign_flips)[0]

                # Calculate new t values
                if len(flip_idx) > 0:
                    t_cross = t_start - h_start[flip_idx] * (t_end - t_start) / (h_end[flip_idx] - h_start[flip_idx])

                    valid_mask = (t_cross > t_start + eps) & (t_cross < t_end - eps)
                    t_cross = t_cross[valid_mask]
                    t_cross = sort_and_drop_duplicates(t_cross)

                    if len(t_cross) > 0:
                        alpha_t = (t_cross - t_start) / (t_end - t_start)
                        h_cross = h_start + alpha_t.unsqueeze(1) * (h_end - h_start)

                        t_new.append(t_cross)
                        h_new.append(h_cross)

            t_new.append(t_knots[-1].unsqueeze(0))
            h_new.append(h_values[-1].unsqueeze(0))

            t_knots = torch.cat(t_new)
            if t_knots.requires_grad:
                t_knots.retain_grad()
            h_values = torch.cat(h_new)
            if h_values.requires_grad:
                h_values.retain_grad()

        # Final layer (apply ReLU before output)
        h_values = torch.relu(h_values)
        W = mlp.layers[-1].weight
        b = mlp.layers[-1].bias
        h_values = (W @ h_values.T).T + b
        
        spline.pred_knots = t_knots
        spline.pred_values = h_values.squeeze(1)


def train_model(mlp: ReluMLP,
                splines: List[Spline],
                epochs: int = 1000,
                batch_size: int = 8,
                lr: float = 0.01,
                clip_grad_norm: float = 1.0,
                save_path=None):
    """
    Train the MLP to predict SDF values along line segments.
    
    Args:
        mlp: The neural network to train
        splines: List of Spline objects with ground truth data
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        clip_grad_norm: Maximum gradient norm for clipping
        save_path: Optional path to save best model checkpoint
    
    Returns:
        loss_history: List of average loss per epoch
    """
    # GPU Diagnostic
    device = next(mlp.parameters()).device
    print("\n" + "="*60)
    print("GPU DIAGNOSTIC - Regular Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Model on GPU: {device.type == 'cuda'}")
    print("="*60 + "\n")
    
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    loss_history = []
    best_loss = float('inf')
    best_epoch = 0
    
    n_splines = len(splines)
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        # Shuffle spline indices for this epoch
        indices = np.arange(n_splines)
        np.random.shuffle(indices)
        
        epoch_loss = 0.0
        epoch_batches = 0
        
        # Process splines in batches
        for batch_start in range(0, n_splines, batch_size):
            batch_end = min(batch_start + batch_size, n_splines)
            batch_indices = indices[batch_start:batch_end]
            
            optimizer.zero_grad()
            batch_splines = [splines[i] for i in batch_indices]
            
            # Reset predictions for all splines in batch
            reset_predictions(mlp, batch_splines)
            
            # Calculate loss for splines in this batch
            batch_loss = 0
            valid_splines = 0
            for spline in batch_splines:
                if len(spline.pred_knots) > 1:
                    # Calculate the Area Between Curves
                    loss = calculate_exact_integral_loss(
                        spline.pred_knots,
                        spline.pred_values,
                        spline.gt_knots,
                        spline.gt_values
                    )
                    
                    batch_loss += loss
                    valid_splines += 1
            
            if valid_splines > 0:
                batch_loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=clip_grad_norm)
                optimizer.step()
                epoch_loss += batch_loss.item()
                epoch_batches += 1
        
        # Update progress bar with epoch loss
        if epoch_batches > 0:
            avg_epoch_loss = epoch_loss / epoch_batches
            loss_history.append(avg_epoch_loss)
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_epoch = epoch
                if save_path is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': mlp.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, save_path / 'best_model.pt')
            
            pbar.set_postfix({'loss': f'{avg_epoch_loss:.6f}', 'best': f'{best_loss:.6f}'})
    
    return loss_history


def compute_metrics(splines: List[Spline]) -> dict:
    """
    Compute metrics comparing predicted and ground truth splines.
    
    Returns dictionary with metrics:
        - mean_l2_error: Mean L2 error across all splines
        - max_l2_error: Maximum L2 error
        - mean_num_knots_pred: Average number of predicted knots
        - mean_num_knots_gt: Average number of ground truth knots
    """
    l2_errors = []
    num_knots_pred = []
    num_knots_gt = []
    
    for spline in splines:
        if spline.pred_knots is not None and len(spline.pred_knots) > 1:
            # Compute L2 error by sampling both splines uniformly
            t_samples = torch.linspace(0, 1, 1000)
            pred_samples = get_spline_value(t_samples, spline.pred_knots, spline.pred_values)
            gt_samples = get_spline_value(t_samples, spline.gt_knots, spline.gt_values)
            
            l2_error = torch.sqrt(torch.mean((pred_samples - gt_samples) ** 2)).item()
            l2_errors.append(l2_error)
            
            num_knots_pred.append(len(spline.pred_knots))
            num_knots_gt.append(len(spline.gt_knots))
    
    metrics = {
        'mean_l2_error': np.mean(l2_errors) if l2_errors else 0,
        'max_l2_error': np.max(l2_errors) if l2_errors else 0,
        'mean_num_knots_pred': np.mean(num_knots_pred) if num_knots_pred else 0,
        'mean_num_knots_gt': np.mean(num_knots_gt) if num_knots_gt else 0,
        'num_splines': len(splines)
    }
    
    return metrics

