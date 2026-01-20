import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from neural_spline.types import Splines
from neural_spline.model import ReluMLP
from neural_spline.utils import extract_mesh_marching_cubes
from scripts.evaluate import compute_chamfer_distance, compute_IoU
torch.set_float32_matmul_precision('high')

def insert_zero_crossings(t: torch.Tensor,
                          valid: torch.Tensor,
                          z: torch.Tensor,
                          max_knots: int,
                          max_candidates_per_segment: int,
                          eps: float = 1e-6):
    """
    Optimized version using segment-wise sorting + stable compaction.
    Exploits the fact that t is already sorted and new knots belong to specific segments.
    
    Returns:
        t_out, v_out, z_out
    """
    device = t.device
    B, K = t.shape
    H = z.shape[-1]

    # Adjacent segments
    tL, tR = t[:, :-1], t[:, 1:]                 # (B, K-1)
    zL, zR = z[:, :-1, :], z[:, 1:, :]           # (B, K-1, H)
    seg_valid = valid[:, :-1] & valid[:, 1:]     # (B, K-1)

    # alpha for each (segment, neuron)
    denom = (zR - zL)                             # (B, K-1, H)
    denom_valid = denom.abs() > eps
    denom_safe = torch.where(denom_valid, denom, torch.ones_like(denom))
    alpha = (-zL) / denom_safe                    # (B, K-1, H)

    valid_new = seg_valid[:, :, None] & denom_valid & (alpha > eps) & (alpha < 1.0 - eps)
    # (B, K-1, )
    max_insert_knots_seg = torch.max(torch.sum(valid_new, dim=2))
    
    # Limit candidates per segment - keep only the top max_candidates_per_segment valid candidates
    # Strategy: Sort by validity and alpha value, keep first max_candidates_per_segment
    
    # Create sort keys: valid candidates get their alpha value, invalid get a large number
    top_v, top_idx = torch.topk(valid_new.byte(), dim=2, k=max_candidates_per_segment)
    top_v = top_v.bool()
    top_idx = torch.where(top_v, top_idx, H-1)
    sort_keys = torch.gather(alpha, dim=2, index=top_idx)
    sort_keys = torch.where(top_v, sort_keys, 2)
    # TODO: think again if this is correct
    
    # Sort and keep only top max_candidates_per_segment
    _, sort_idx = torch.sort(sort_keys, dim=2)  # (B, K-1, max_candidates_per_segment)
    
    # Gather the selected candidates
    alpha = torch.gather(alpha, 2, torch.gather(top_idx, 2, sort_idx))  # (B, K-1, max_cand)
    valid_new = top_v # torch.gather(valid_new, 2, sort_idx)  # (B, K-1, max_cand)
    alpha = torch.where(top_v, alpha, 2)
    
    # Note: zL and zR remain unchanged as full z-vectors at endpoints (B, K-1, H)
    # We're just reducing the number of candidate positions, not the z-vector dimension

    # candidate times in each segment (B, K-1, max_candidates_per_segment)
    dt = (tR - tL)                                # (B, K-1)
    t_new = tL[:, :, None] + alpha * dt[:, :, None]

    t_new_eff = torch.where(valid_new, t_new, 2)

    # candidate z-vectors: (B, K-1, max_candidates_per_segment, H)
    # Interpolate the full z-vector at each candidate position
    z_new = zL[:, :, None, :] + alpha[:, :, :, None] * (zR - zL)[:, :, None, :]
    z_new = torch.where(valid_new[:, :, :, None], z_new, torch.zeros_like(z_new))

    # B, K-1, max_candidates
    # B, K-1 => B, K-1, 1
    # B, K, max_candidates
    t_merged = torch.cat([tL[:, :, None], t_new_eff], dim=2) 
    t_flat = t_merged.reshape(B, (K-1)*(max_candidates_per_segment+1))
    t_flat = torch.cat([t_flat, t[:, [-1]]], dim=1)

    z_merged = torch.cat([zL[:, :, None, :], z_new], dim=2) 
    z_merged = z_merged.reshape(B, (K-1)*(max_candidates_per_segment+1), H)
    z_merged = torch.cat([z_merged, z[:, -1:]], dim=1)

    top_t, top_idx = torch.topk(t_flat, k=max_knots, largest=False, sorted=True)
    # top_v = top_t<=1.0
    
    top_z = torch.gather(z_merged, 1, top_idx.unsqueeze(-1).expand(-1, -1, H))
    top_v = top_t<=1.0

    return top_t, top_v, top_z, max_insert_knots_seg


def forward_knots(mlp: nn.Module, end_points: torch.Tensor, start_points: torch.Tensor, return_grad=False,
                  max_knots=64, max_candidates_per_segment=8, eps=1e-6):
    device = end_points.device
    B, D = end_points.shape
    layers = mlp.layers
    n_layers = len(layers)

    # knot buffers
    t = torch.full((B, max_knots), 2.0, device=device, dtype=torch.float32)
    valid = torch.zeros((B, max_knots), device=device, dtype=torch.bool)
    t[:, 0] = 0.0
    t[:, 1] = 1.0
    valid[:, 0:2] = True

    # segment direction
    d = end_points - start_points  # (B, Din)
    
    # Initialize statistics tensors
    # max_knots_layer[i] = max knots present before layer i processes (i=0 is initial state)
    # max_insert_knots_layer[i] = max knots inserted in any segment by layer i
    max_knots_layer = torch.zeros(n_layers, dtype=torch.int64, device=device)
    max_insert_knots_layer = torch.zeros(n_layers - 1, dtype=torch.int64, device=device)

    # ----- layer 0 preactivation via 1D projection -----
    W0 = layers[0].weight          # (H, Din)
    b0 = layers[0].bias            # (H,)

    if return_grad:
        jac = W0
    
    alpha0 = d @ W0.T              # (B, H)
    beta0  = start_points @ W0.T + b0  # (B, H)
    z = alpha0[:, None, :] * t[:, :, None] + beta0[:, None, :]  # (B, K, H)
    z = torch.where(valid.unsqueeze(-1), z, torch.zeros_like(z))

    max_knots_layer[0] = 2
    t, valid, z, max_insert_knots_seg = insert_zero_crossings(t, valid, z, max_knots, max_candidates_per_segment, eps)
    max_insert_knots_layer[0] = max_insert_knots_seg

    h = torch.where(valid.unsqueeze(-1), torch.relu(z), torch.zeros_like(z))  # (B, K, H)

    # ----- hidden layers 1..(n_layers-2) -----
    for li in range(1, n_layers - 1):
        layer = layers[li]  # Linear(H+D,H) for hidden with skip, Linear(H,H) without skip
        
        if mlp.skip_connections:
            # Compute coordinates at each knot: p0 + t*d
            coords = start_points[:, None, :] + t[:, :, None] * d[:, None, :]  # (B, K, D)
            coords = torch.where(valid.unsqueeze(-1), coords, torch.zeros_like(coords))
            # Concatenate hidden activations with input coordinates
            h_input = torch.cat([h, coords], dim=-1)  # (B, K, H+D)
        else:
            h_input = h
        
        z = torch.matmul(h_input, layer.weight.t()) + layer.bias
        z = torch.where(valid.unsqueeze(-1), z, torch.zeros_like(z))
        
        # if this is not the output layer, insert crossings and relu
        if li != n_layers - 1:
            # li runs only to n_layers-2 here, so always true
            max_knots_layer[li] = torch.max(torch.sum(valid, dim=1))
            t, valid, z, max_insert_knots_seg = insert_zero_crossings(t, valid, z, max_knots, max_candidates_per_segment, eps)
            max_insert_knots_layer[li] = max_insert_knots_seg

            h = torch.where(valid.unsqueeze(-1), torch.relu(z), torch.zeros_like(z))

    max_knots_layer[n_layers - 1] = torch.max(torch.sum(valid, dim=1))
    # ----- output layer (no ReLU, no crossings) -----
    out_layer = layers[-1]  # Linear(H,1)
    y = torch.matmul(h, out_layer.weight.t()) + out_layer.bias
    y = torch.where(valid.unsqueeze(-1), y, torch.zeros_like(y))  # (B, K, 1)

    return t, valid, y, max_knots_layer, max_insert_knots_layer

class KnotForward(nn.Module):
    def __init__(self, mlp, max_knots=128, max_candidates_per_segment=8, eps=1e-6):
        super().__init__()
        self.mlp = mlp
        self.max_knots = max_knots
        self.max_candidates_per_segment = max_candidates_per_segment
        self.eps = eps

    def forward(self, end_points, start_points):
        return forward_knots(self.mlp, end_points, start_points,
                             max_knots=self.max_knots,
                             max_candidates_per_segment=self.max_candidates_per_segment,
                             eps=self.eps)


def integral_loss(pred_t, pred_v, gt_t, gt_v, pred_valid, eps=1e-6):
    if pred_v.dim() == 3 and pred_v.size(-1) == 1:
        pred_v = pred_v.squeeze(-1)
    if gt_v.dim() == 3 and gt_v.size(-1) == 1:
        gt_v = gt_v.squeeze(-1)

    pred_t_src = torch.where(pred_valid, pred_t, pred_t.new_tensor(1.0))
    pred_v_src = torch.where(pred_valid, pred_v, torch.zeros_like(pred_v))

    # 1) Union grid in [0,1]
    t_union = torch.cat([pred_t_src, gt_t], dim=1)
    t_union, _ = torch.sort(t_union, dim=1)

    # 2) Batch linear interpolation
    def batch_interp(t_grid, t_source, v_source):
        idx = torch.searchsorted(t_source.contiguous(), t_grid.contiguous())
        max_idx = t_source.shape[1] - 1
        idx = idx.clamp(1, max_idx)

        t_left  = torch.gather(t_source, 1, idx - 1)
        t_right = torch.gather(t_source, 1, idx)
        v_left  = torch.gather(v_source, 1, idx - 1)
        v_right = torch.gather(v_source, 1, idx)

        denom = (t_right - t_left)
        valid_denom = denom.abs() > eps
        denom_safe = torch.where(valid_denom, denom, torch.ones_like(denom))
        alpha = (t_grid - t_left) / denom_safe
        return v_left + alpha * (v_right - v_left)

    v_pred = batch_interp(t_union, pred_t_src, pred_v_src)
    v_gt   = batch_interp(t_union, gt_t, gt_v)

    # 3) Exact integral of squared linear diff on each segment
    diff = v_pred - v_gt
    dt = t_union[:, 1:] - t_union[:, :-1]
    yL = diff[:, :-1]
    yR = diff[:, 1:]

    segment_areas = (dt / 3.0) * (yL * yL + yL * yR + yR * yR)
    return segment_areas.sum()


def integral_norm_loss(pred_t, pred_norm, gt_t, gt_norm, pred_valid, eps=1e-6):
    if pred_v.dim() == 3 and pred_v.size(-1) == 1:
        pred_v = pred_v.squeeze(-1)
    if gt_v.dim() == 3 and gt_v.size(-1) == 1:
        gt_v = gt_v.squeeze(-1)

    pred_t_src = torch.where(pred_valid, pred_t, pred_t.new_tensor(1.0))
    pred_v_src = torch.where(pred_valid, pred_v, torch.zeros_like(pred_v))

    # 1) Union grid in [0,1]
    t_union = torch.cat([pred_t_src, gt_t], dim=1)
    t_union, _ = torch.sort(t_union, dim=1)

    # 2) Batch linear interpolation
    def batch_interp(t_grid, t_source, v_source):
        idx = torch.searchsorted(t_source.contiguous(), t_grid.contiguous())
        max_idx = t_source.shape[1] - 1
        idx = idx.clamp(1, max_idx)

        t_left  = torch.gather(t_source, 1, idx - 1)
        t_right = torch.gather(t_source, 1, idx)
        v_left  = torch.gather(v_source, 1, idx - 1)
        v_right = torch.gather(v_source, 1, idx)

        denom = (t_right - t_left)
        valid_denom = denom.abs() > eps
        denom_safe = torch.where(valid_denom, denom, torch.ones_like(denom))
        alpha = (t_grid - t_left) / denom_safe
        return v_left + alpha * (v_right - v_left)

    v_pred = batch_interp(t_union, pred_t_src, pred_v_src)
    v_gt   = batch_interp(t_union, gt_t, gt_v)

    # 3) Exact integral of squared linear diff on each segment
    diff = v_pred - v_gt
    dt = t_union[:, 1:] - t_union[:, :-1]
    yL = diff[:, :-1]
    yR = diff[:, 1:]

    segment_areas = (dt / 3.0) * (yL * yL + yL * yR + yR * yR)
    return segment_areas.sum()


def train_model_fast(mlp: ReluMLP,
                     splines: Splines,
                     epochs: int = 1000,
                     batch_size: int = 64,
                     lr: float = 0.01,
                     clip_grad_norm: float = 1.0,
                     save_path=None,
                     extract_mesh: bool = False,
                     mesh_resolution: int = 256,
                     mesh_save_interval: int = 50,
                     max_knots: int = 32,
                     data: dict = None,
                     max_seg_insertions: int = 8):
    device = splines.start_points.device
    mlp = mlp.to(device)

    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    N = splines.start_points.shape[0]
    n_layers = len(mlp.layers)

    loss_history = []
    best_loss = float('inf')
    
    # Global statistics across all epochs
    global_max_knots_layer = torch.zeros(n_layers, dtype=torch.int64, device=device)
    global_max_insert_knots_layer = torch.zeros(n_layers - 1, dtype=torch.int64, device=device)

    pbar = tqdm(range(epochs))
    
    knot_fwd = KnotForward(mlp, max_knots=max_knots, max_candidates_per_segment=max_seg_insertions, eps=1e-6).to(device)

    knot_fwd = torch.compile(
        knot_fwd,
        backend="inductor",
        mode="reduce-overhead",   # try "max-autotune" later
        fullgraph=False
    )

    compiled_loss = torch.compile(integral_loss, backend="inductor", mode="reduce-overhead", fullgraph=False)
    

    for epoch in pbar:
        perm = torch.randperm(N, device=device)

        epoch_loss = 0.0
        n_batches = 0
        
        # Epoch statistics (max over batches)
        epoch_max_knots_layer = torch.zeros(n_layers, dtype=torch.int64, device=device)
        epoch_max_insert_knots_layer = torch.zeros(n_layers - 1, dtype=torch.int64, device=device)

        for i in range(0, N, batch_size):
            batch_idx = perm[i:i + batch_size]

            p0 = splines.start_points[batch_idx]
            p1 = splines.end_points[batch_idx]
            knots_gt = splines.knots[batch_idx]
            values_gt = splines.values[batch_idx]

            optimizer.zero_grad(set_to_none=True)

            pred_t, pred_valid, pred_y, batch_max_knots_layer, batch_max_insert_knots_layer = knot_fwd(p1, p0)
            assert batch_max_knots_layer[-1] < max_knots
            assert torch.max(batch_max_insert_knots_layer[-1]) < max_seg_insertions
            
            # mid_t = (pred_t[:,1:].detach() + pred_t[:,:-1].detach()) / 2.0 
            # mid_points = p0[:, None, :] + mid_t[:, :, None] * (p1 - p0)[:, None, :]
            # mid_points.requires_grad_(True)
            # z = mlp(mid_points)
            # pred_norm = torch.autograd.grad(
            #     z[pred_valid[:, 1:] & pred_valid[:, :-1]].sum(), mid_points, create_graph=True
            # )[0]
            
            # Update epoch maximums
            epoch_max_knots_layer = torch.max(epoch_max_knots_layer, batch_max_knots_layer)
            epoch_max_insert_knots_layer = torch.max(epoch_max_insert_knots_layer, batch_max_insert_knots_layer)

            loss_sum = compiled_loss(pred_t, pred_y, knots_gt, values_gt, pred_valid)
            loss_mean = loss_sum / batch_idx.numel()

            loss_mean.backward()

            torch.nn.utils.clip_grad_norm_(mlp.parameters(), clip_grad_norm)

            optimizer.step()

            epoch_loss += loss_mean.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        # compute metrics based on the pcd_surf and pcd_vol
        cd, hd = compute_chamfer_distance(data, mlp)
        iou = compute_IoU(data, mlp)
        
        # Update global maximums
        global_max_knots_layer = torch.max(global_max_knots_layer, epoch_max_knots_layer)
        global_max_insert_knots_layer = torch.max(global_max_insert_knots_layer, epoch_max_insert_knots_layer)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'config': mlp.config(),
                    'model_state_dict': mlp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, save_path / 'best_model_fast.pt')

        if extract_mesh and save_path and epoch % mesh_save_interval == 0:
            # Print global statistics summary
            print(f"\n{'='*60}")
            print(f"Global Knot Statistics Across All Epochs")
            print(f"{'='*60}")
            for i in range(n_layers - 1):
                knots_before = global_max_knots_layer[i].item()
                knots_inserted = global_max_insert_knots_layer[i].item()
                print(f"  Layer {i}: {knots_before}, {knots_inserted}")
            print(f"  Layer {n_layers - 1}: {global_max_knots_layer[-1].item()}")
            print(f"{'='*60}\n")
            mesh_path = save_path / f'mesh_epoch_{epoch:04d}.ply'
            try:
                extract_mesh_marching_cubes(mlp, save_path=mesh_path, resolution=mesh_resolution, device=device)
            except Exception as e:
                print(f"Error extracting mesh: {e}")
                pass
        
        # Format knot statistics for display
        knots_str = ' '.join([f"{k.item()}" for k in epoch_max_knots_layer[:-1]])
        insert_str = ' '.join([f"{k.item()}" for k in epoch_max_insert_knots_layer])
        final_knots = epoch_max_knots_layer[-1].item()

        pbar.set_postfix({
            'loss': f"{avg_loss:.6f}", 
            'best': f"{best_loss:.6f}",
            'knots': f"{knots_str} | {final_knots}",
            'ins': insert_str,
            'cd': f"{cd * 1e3:.4f}",
            'hd': f"{hd * 1e3:.4f}",
            'iou': f"{iou:.4f}"
        })

    # Print global statistics summary
    print(f"\n{'='*60}")
    print(f"Global Knot Statistics Across All Epochs")
    print(f"{'='*60}")
    for i in range(n_layers - 1):
        knots_before = global_max_knots_layer[i].item()
        knots_inserted = global_max_insert_knots_layer[i].item()
        print(f"  Layer {i}: {knots_before}, {knots_inserted}")
    print(f"  Layer {n_layers - 1}: {global_max_knots_layer[-1].item()}")
    print(f"{'='*60}\n")
    
    # Save final mesh at the end of training
    if extract_mesh and save_path:
        final_mesh_path = save_path / 'mesh_final.ply'
        extract_mesh_marching_cubes(mlp, save_path=final_mesh_path, resolution=mesh_resolution, device=device)

    return loss_history
