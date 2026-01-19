"""
Fast, batched training loop for neural spline learning.

This module implements a vectorized version of the training loop that processes
multiple splines simultaneously using a sweep-line algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import profile, record_function, ProfilerActivity

from neural_spline.types import Splines
from neural_spline.model import ReluMLP


# ==========================================
# 1. Vectorized Core Logic (Sweep-Line)
# ==========================================

def sort_and_drop_duplicates(t_vals, batch_ids, values=None, eps=1e-6):
    """
    Sorts flattened knots by (Batch_ID, T) and removes duplicates 
    within the same batch.
    
    Args:
        t_vals: (N,) Flattened tensor of t values
        batch_ids: (N,) Batch indices for each t value
        values: (N, D) Optional values associated with each t
        eps: Threshold for considering two t values as duplicates
    
    Returns:
        Sorted and deduplicated tensors
    """
    device = t_vals.device
    
    # Stable sort key: Batch_ID is primary, T is secondary.
    # We use a float encoding for the sort key: batch_id + t_vals
    # (Assumes t is in [0, 1] and batch_ids are integers)
    sort_key = batch_ids.float() + (t_vals * 0.99)  # 0.99 ensures t=1 doesn't spill to next batch
    perm = torch.argsort(sort_key)
    
    t_sorted = t_vals[perm]
    b_sorted = batch_ids[perm]
    
    # Detect duplicates: same batch AND close t
    if len(t_sorted) > 1:
        same_batch = b_sorted[1:] == b_sorted[:-1]
        close_t = torch.abs(t_sorted[1:] - t_sorted[:-1]) < eps
        is_duplicate = same_batch & close_t
        
        # Keep first occurrence, drop subsequent duplicates
        mask_keep = torch.cat([torch.tensor([True], device=device), ~is_duplicate])
    else:
        mask_keep = torch.ones_like(t_sorted, dtype=torch.bool)

    final_perm = perm[mask_keep]
    
    if values is not None:
        return t_vals[final_perm], batch_ids[final_perm], values[final_perm]
    return t_vals[final_perm], batch_ids[final_perm], None


def forward(mlp: nn.Module, p0: torch.Tensor, p1: torch.Tensor, max_knots=128):
    device = p0.device
    B = p0.shape[0]
    
    # Get linear layers from the MLP
    layers = mlp.layers

    mask = torch.tensor((B, 128), device = device, dtype = torch.bool)
    # TODO: first two are true
    knots = torch.tensor((B, 128), device = device, dtype = p0.dtype)
    # First two are 0 and 1, after that repeat 1
    # Calculate first layer weights and bias togheter with direction and offset so that multiplications/addition are then 1 input, n outputs
    # Then multiply this with knots to get the values
    normals = torch.tensor((B, 127, layers[0].weight.shape), device = device, dtype = p0.dtype)
    offsets = torch.tensor((B, 127, layers[0].bias.shape), device = device, dtype = p0.dtype)
    mask = torch.zeros((B, 128), device = device, dtype = torch.float32)
    mask[:, 0:1] = 1.0

    d = p1 - p0  # (B, D)
    W = layers[0].weight
    b = layers[0].bias

    eps = 1e-8
    
    for i, layer in enumerate(layers):
        W = layer.weight  # (D_out, D_in)
        b = layer.bias    # (D_out,)
        
        # 1. Pre-activation Z at current knots
        z_flat = h_flat @ W.T + b  # (N_knots, D_out)
        
        # If output layer, stop here (no ReLU processing)
        if i == len(layers) - 1:
            h_flat = z_flat
            break
            
        # 2. Find Zero Crossings (New Knots)
        # We look at segments defined by adjacent points in the flat array.
        # Mask: Valid segment if batch_ids[i] == batch_ids[i+1]
        if len(batch_ids) > 1:
            mask_seg = batch_ids[:-1] == batch_ids[1:]
            # True True True True False True True True True True False
            
            if mask_seg.any():
                z_L = z_flat[:-1][mask_seg]  # (N_segs, D_out)
                z_R = z_flat[1:][mask_seg]    # (N_segs, D_out)
                t_L = t_flat[:-1][mask_seg]   # (N_segs,)
                t_R = t_flat[1:][mask_seg]    # (N_segs,)
                b_seg = batch_ids[:-1][mask_seg]  # (N_segs,)
                
                # Solve Linearity: Z_L + alpha * (Z_R - Z_L) = 0
                denom = z_R - z_L  # (N_segs, D_out)
                # Avoid div/0
                safe_denom = torch.where(
                    torch.abs(denom) < 1e-9,
                    torch.tensor(1e-9, device=device),
                    denom
                )
                alpha = -z_L / safe_denom  # (N_segs, D_out)
                
                # Valid crossing if 0 < alpha < 1
                is_crossing = (alpha > eps) & (alpha < 1.0 - eps)
                
                # Get indices of (Segment, Neuron) that crossed
                seg_idx, neuron_idx = torch.where(is_crossing)
                
                if len(seg_idx) > 0:
                    valid_alphas = alpha[seg_idx, neuron_idx]
                    
                    # Compute t_new
                    dt = t_R[seg_idx] - t_L[seg_idx]
                    t_new = t_L[seg_idx] + valid_alphas * dt
                    b_new = b_seg[seg_idx]
                    
                    # Interpolate Z to get values at new knots
                    # For the crossing neuron, value is 0. For others, it's linear interp.
                    z_new = z_L[seg_idx] + valid_alphas.unsqueeze(1) * (z_R[seg_idx] - z_L[seg_idx])
                    
                    # 3. Merge & Sort
                    t_combined = torch.cat([t_flat, t_new])
                    b_combined = torch.cat([batch_ids, b_new])
                    z_combined = torch.cat([z_flat, z_new])
                    
                    t_flat, batch_ids, z_flat = sort_and_drop_duplicates(
                        t_combined, b_combined, z_combined
                    )
        
        # Apply ReLU
        h_flat = torch.relu(z_flat)

    # --- Reconstruct Batched Tensors (Padding) ---
    # We now have a flat array. We need to split it back into (B, Max_Knots).
    
    # Count knots per batch
    counts = torch.bincount(batch_ids, minlength=B)
    
    # Split
    t_list = torch.split(t_flat, counts.tolist())
    v_list = torch.split(h_flat, counts.tolist())
    
    # Squeeze values if needed (N, 1) -> (N,)
    v_list = [v.squeeze(-1) if v.dim() > 1 else v for v in v_list]
    
    # Pad
    # Padding t with 1.0 means dt=0 at the end -> no loss contribution
    pred_t_pad = pad_sequence(t_list, batch_first=True, padding_value=1.0)
    # Padding v with 0.0 (value doesn't matter if dt=0)
    pred_v_pad = pad_sequence(v_list, batch_first=True, padding_value=0.0)
    
    return pred_t_pad, pred_v_pad


# ==========================================
# 2. Vectorized Loss Function
# ==========================================

def integral_loss(pred_t, pred_v, gt_t, gt_v):
    """
    Computes exact integral of (Pred - GT)^2 for a batch using Simpson's rule.
    
    This implements the same exact integration as the single-spline version,
    but vectorized across the entire batch.
    
    Args:
        pred_t: (B, N_p) Predicted knot positions (padded)
        pred_v: (B, N_p) Predicted values at knots (padded)
        gt_t: (B, N_g) Ground truth knot positions (padded)
        gt_v: (B, N_g) Ground truth values at knots (padded)
    
    Returns:
        Scalar loss (sum over batch)
    """
    # 1. Merge Knots: Union of Pred and GT
    # Concatenate along time dimension: (B, N_p + N_gt)
    t_union = torch.cat([pred_t, gt_t], dim=1)
    
    # Sort the union grid
    t_union, _ = torch.sort(t_union, dim=1)
    
    # We don't strictly need to remove duplicates here. 
    # If duplicates exist, dt=0, so area=0. It handles itself.
    
    # 2. Vectorized Interpolation
    def batch_interp(t_grid, t_source, v_source):
        """
        Batch linear interpolation.
        
        Args:
            t_grid: (B, N_union) Points to interpolate at
            t_source: (B, N_source) Source knot positions
            v_source: (B, N_source) Source values
        
        Returns:
            (B, N_union) Interpolated values
        """
        # Find position of t_grid in t_source
        idx = torch.searchsorted(t_source.contiguous(), t_grid.contiguous())
        
        # Clamp to ensure we can look at idx-1 and idx
        max_idx = t_source.shape[1] - 1
        idx = idx.clamp(1, max_idx)
        
        # Gather values
        t_left = torch.gather(t_source, 1, idx - 1)
        t_right = torch.gather(t_source, 1, idx)
        v_left = torch.gather(v_source, 1, idx - 1)
        v_right = torch.gather(v_source, 1, idx)
        
        # Interpolate
        denom = (t_right - t_left) + 1e-8
        alpha = (t_grid - t_left) / denom
        return v_left + alpha * (v_right - v_left)

    v_pred_interp = batch_interp(t_union, pred_t, pred_v)
    v_gt_interp = batch_interp(t_union, gt_t, gt_v)
    
    # 3. Integrate Square Difference
    diff = v_pred_interp - v_gt_interp
    
    dt = t_union[:, 1:] - t_union[:, :-1]
    y_L = diff[:, :-1]
    y_R = diff[:, 1:]
    
    # Simpson's rule / Exact integration of linear function squared
    # Area = dt/3 * (yL^2 + yL*yR + yR^2)
    segment_areas = (dt / 3.0) * (y_L**2 + y_L * y_R + y_R**2)
    
    # Sum over all segments (dim 1) then sum over batch (dim 0)
    return segment_areas.sum()

# ==========================================
# 4. Optimized Training Loop
# ==========================================

def train_model_fast(mlp: ReluMLP,
                     splines: Splines,
                     epochs: int = 1000,
                     batch_size: int = 64,
                     lr: float = 0.01,
                     clip_grad_norm: float = 1.0,
                     save_path=None,
                     profile_enabled: bool = False,
                     profile_wait: int = 1,
                     profile_warmup: int = 1,
                     profile_active: int = 1,
                     profile_repeat: int = 1):
    device = splines.start_points.device
    mlp = mlp.to(device)
    
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    
    N = splines.start_points.shape[0]
    
    loss_history = []
    best_loss = float('inf')
    
    # Setup profiler if enabled
    prof_context = None
    if profile_enabled:
        activities = [ProfilerActivity.CPU]
        activities.append(ProfilerActivity.CUDA)
        
        prof_context = profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=profile_wait,
                warmup=profile_warmup,
                active=profile_active,
                repeat=profile_repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    
    pbar = tqdm(range(epochs))
    
    # Start profiler context if enabled
    if prof_context is not None:
        prof_context.__enter__()
    
    for epoch in pbar:
        # GPU diagnostic on first epoch
        if epoch == 0 and torch.cuda.is_available():
            print(f"\n[Epoch 0] GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
        # Shuffle indices
        perm = torch.randperm(N, device=device)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx in perm.chunk(batch_size):
            with record_function("batch_iteration"):
                # Slice Batch
                with record_function("data_slicing"):
                    p0 = splines.start_points[batch_idx]
                    p1 = splines.end_points[batch_idx]
                    knots_gt = splines.knots[batch_idx]
                    values_gt = splines.values[batch_idx]
                
                with record_function("optimizer_zero_grad"):
                    optimizer.zero_grad()
                
                # A. Vectorized Prediction
                with record_function("forward_pass"):
                    knots_pred, values_pred = forward(mlp, p0, p1)
                
                # B. Vectorized Loss
                with record_function("loss_computation"):
                    loss_sum = integral_loss(knots_pred, values_pred, knots_gt, values_gt)
                    loss_mean = loss_sum / len(batch_idx)
                
                with record_function("backward_pass"):
                    loss_mean.backward()
                
                with record_function("gradient_clipping"):
                    torch.nn.utils.clip_grad_norm_(mlp.parameters(), clip_grad_norm)
                
                with record_function("optimizer_step"):
                    optimizer.step()
                
                epoch_loss += loss_mean.item()
                n_batches += 1
            
            # Step profiler if enabled
            if prof_context is not None:
                prof_context.step()
            
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        loss_history.append(avg_loss)
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': mlp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, save_path / 'best_model_fast.pt')
                
        pbar.set_postfix({'loss': f"{avg_loss:.6f}", 'best': f"{best_loss:.6f}"})
    
    return loss_history


def update_spline_predictions(mlp: ReluMLP, splines: Splines):
    """
    Update all splines with final predictions from the trained model.
    
    This is useful after training to populate the pred_knots and pred_values
    for each spline for visualization and metrics computation.
    
    Args:
        mlp: Trained neural network
        splines: List of Spline objects to update
    """
    device = splines[0].start_point.device
    mlp = mlp.to(device)
    
    # Process in batches to avoid memory issues
    batch_size = 128
    
    for i in range(0, len(splines), batch_size):
        batch_splines = splines[i:i+batch_size]
        
        # Gather start/end points
        p0 = torch.stack([s.start_point for s in batch_splines]).to(device)
        p1 = torch.stack([s.end_point for s in batch_splines]).to(device)
        
        # Compute predictions
        with torch.no_grad():
            pred_t, pred_v = forward(mlp, p0, p1)
        
        # Update each spline
        for j, spline in enumerate(batch_splines):
            # Find valid knots (not padding)
            valid_mask = pred_t[j] <= 1.0
            if valid_mask.sum() > 0:
                # Detect padding by checking for duplicate 1.0 values at the end
                t_vals = pred_t[j][valid_mask]
                v_vals = pred_v[j][valid_mask]
                
                # Remove padding (consecutive 1.0s at the end)
                if len(t_vals) > 1:
                    diffs = t_vals[1:] - t_vals[:-1]
                    if torch.all(diffs[-1:] < 1e-6):  # Last segment is padding
                        non_padding = torch.cat([
                            torch.ones(1, dtype=torch.bool, device=device),
                            diffs > 1e-6
                        ])
                        t_vals = t_vals[non_padding]
                        v_vals = v_vals[non_padding]
                
                spline.pred_knots = t_vals.cpu()
                spline.pred_values = v_vals.cpu()

