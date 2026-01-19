import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from neural_spline.types import Splines
from neural_spline.model import ReluMLP
from neural_spline.utils import extract_mesh_marching_cubes
torch.set_float32_matmul_precision('high')

def insert_zero_crossings(t: torch.Tensor,
                          valid: torch.Tensor,
                          z: torch.Tensor,
                          max_knots: int,
                          eps: float = 1e-6):
    device = t.device
    B, K = t.shape
    H = z.shape[-1]

    # Adjacent segments
    tL, tR = t[:, :-1], t[:, 1:]             # (B, K-1)
    zL, zR = z[:, :-1, :], z[:, 1:, :]       # (B, K-1, H)
    seg_valid = valid[:, :-1] & valid[:, 1:] # (B, K-1)

    denom = (zR - zL)
    denom_valid = denom.abs() > eps
    denom_safe = torch.where(denom_valid, denom, torch.ones_like(denom))
    alpha = (-zL) / denom_safe

    is_cross = seg_valid[:, :, None] & denom_valid & (alpha > eps) & (alpha < 1.0 - eps)

    # Flatten candidates
    M = (K - 1) * H
    alpha_f = alpha.reshape(B, M)            # (B, M)
    valid_new = is_cross.reshape(B, M)       # (B, M) bool

    seg_idx_base = torch.arange(K - 1, device=device).repeat_interleave(H)  # (M,)
    seg_idx = seg_idx_base[None, :].expand(B, -1)                           # (B, M)

    tL_f = torch.gather(tL, 1, seg_idx)
    tR_f = torch.gather(tR, 1, seg_idx)

    alpha_f = torch.where(valid_new, alpha_f, torch.zeros_like(alpha_f))
    t_new = tL_f + alpha_f * (tR_f - tL_f)      # (B, M)

    # Interpolate full z-vector at t_new
    seg_idx_3 = seg_idx.unsqueeze(-1).expand(-1, -1, H)
    zL_c = torch.gather(zL, 1, seg_idx_3)
    zR_c = torch.gather(zR, 1, seg_idx_3)

    z_new = zL_c + alpha_f.unsqueeze(-1) * (zR_c - zL_c)
    z_new = z_new * valid_new.unsqueeze(-1)

    # Merge + sort + truncate
    BIG = t.new_tensor(2.0)  # invalid => goes to end

    t_old_eff = torch.where(valid, t, BIG)
    t_new_eff = torch.where(valid_new, t_new, BIG)

    t_all = torch.cat([t_old_eff, t_new_eff], dim=1)        # (B, K+M)
    z_all = torch.cat([z, z_new], dim=1)                    # (B, K+M, H)
    v_all = torch.cat([valid, valid_new], dim=1)            # (B, K+M)

    t_sorted, idx = torch.sort(t_all, dim=1)
    idx_z = idx.unsqueeze(-1).expand(-1, -1, H)
    z_sorted = torch.gather(z_all, 1, idx_z)
    v_sorted = torch.gather(v_all, 1, idx)

    t2 = t_sorted[:, :max_knots].contiguous()
    z2 = z_sorted[:, :max_knots].contiguous()
    valid2 = v_sorted[:, :max_knots].contiguous()

    t2 = torch.where(valid2, t2, t2.new_tensor(1.0))
    z2 = torch.where(valid2.unsqueeze(-1), z2, torch.zeros_like(z2))

    return t2, valid2, z2

def forward_knots(mlp: nn.Module, end_points: torch.Tensor, start_points: torch.Tensor,
                  max_knots=64, eps=1e-6):
    device = end_points.device
    B, D = end_points.shape
    layers = mlp.layers
    n_layers = len(layers)

    # knot buffers
    t = torch.full((B, max_knots), 1.0, device=device, dtype=torch.float32)
    valid = torch.zeros((B, max_knots), device=device, dtype=torch.bool)
    t[:, 0] = 0.0
    t[:, 1] = 1.0
    valid[:, 0:2] = True

    # segment direction
    d = end_points - start_points  # (B, Din)

    # ----- layer 0 preactivation via 1D projection -----
    W0 = layers[0].weight          # (H, Din)
    b0 = layers[0].bias            # (H,)
    alpha0 = d @ W0.T              # (B, H)
    beta0  = start_points @ W0.T + b0  # (B, H)
    z = alpha0[:, None, :] * t[:, :, None] + beta0[:, None, :]  # (B, K, H)
    z = torch.where(valid.unsqueeze(-1), z, torch.zeros_like(z))

    # insert crossings for layer 0 (since it has ReLU after it)
    t, valid, z = insert_zero_crossings(t, valid, z, max_knots, eps)
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
            t, valid, z = insert_zero_crossings(t, valid, z, max_knots, eps)
            h = torch.where(valid.unsqueeze(-1), torch.relu(z), torch.zeros_like(z))

    # ----- output layer (no ReLU, no crossings) -----
    out_layer = layers[-1]  # Linear(H,1)
    y = torch.matmul(h, out_layer.weight.t()) + out_layer.bias
    y = torch.where(valid.unsqueeze(-1), y, torch.zeros_like(y))  # (B, K, 1)

    return t, valid, y

class KnotForward(nn.Module):
    def __init__(self, mlp, max_knots=128, eps=1e-6):
        super().__init__()
        self.mlp = mlp
        self.max_knots = max_knots
        self.eps = eps

    def forward(self, end_points, start_points):
        return forward_knots(self.mlp, end_points, start_points,
                             max_knots=self.max_knots, eps=self.eps)


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


def train_model_fast(mlp: ReluMLP,
                     splines: Splines,
                     epochs: int = 1000,
                     batch_size: int = 64,
                     lr: float = 0.01,
                     clip_grad_norm: float = 1.0,
                     save_path=None,
                     extract_mesh: bool = False,
                     mesh_resolution: int = 128,
                     mesh_save_interval: int = 50,
                     store_sdf: bool = True,
                     sdf_save_interval: int = 50):
    device = splines.start_points.device
    mlp = mlp.to(device)

    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    N = splines.start_points.shape[0]

    loss_history = []
    best_loss = float('inf')

    pbar = tqdm(range(epochs))
    
    knot_fwd = KnotForward(mlp, max_knots=32, eps=1e-6).to(device)

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

        for i in range(0, N, batch_size):
            batch_idx = perm[i:i + batch_size]

            p0 = splines.start_points[batch_idx]
            p1 = splines.end_points[batch_idx]
            knots_gt = splines.knots[batch_idx]
            values_gt = splines.values[batch_idx]

            optimizer.zero_grad()

            pred_t, pred_valid, pred_y = knot_fwd(p1, p0)

            loss_sum = compiled_loss(pred_t, pred_y, knots_gt, values_gt, pred_valid)
            loss_mean = loss_sum / batch_idx.numel()

            loss_mean.backward()

            torch.nn.utils.clip_grad_norm_(mlp.parameters(), clip_grad_norm)

            optimizer.step()

            epoch_loss += loss_mean.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

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
            mesh_path = save_path / f'mesh_epoch_{epoch:04d}.ply'
            try:
                extract_mesh_marching_cubes(mlp, save_path=mesh_path, resolution=mesh_resolution, device=device)
            except Exception as e:
                print(f"Error extracting mesh: {e}")
                
        if store_sdf and save_path and epoch % sdf_save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "config": mlp.config(),
                "model_state_dict": mlp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }
            torch.save(checkpoint, save_path / f"sdf_epoch_{epoch:04d}.pt")

        pbar.set_postfix({'loss': f"{avg_loss:.6f}", 'best': f"{best_loss:.6f}"})

    # Save final mesh at the end of training
    if extract_mesh and save_path:
        final_mesh_path = save_path / 'mesh_final.ply'
        extract_mesh_marching_cubes(mlp, save_path=final_mesh_path, resolution=mesh_resolution, device=device)

    return loss_history
