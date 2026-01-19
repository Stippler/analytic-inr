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
                          eps: float = 1e-6,
                          collect_stats: bool = False,
                          max_candidates_per_segment: int = None):
    """
    Optimized version using segment-wise sorting + stable compaction.
    Exploits the fact that t is already sorted and new knots belong to specific segments.
    
    Args:
        collect_stats: If True, returns additional statistics about candidates
    
    Returns:
        t_out, v_out, z_out[, stats_dict if collect_stats=True]
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
    
    # Limit candidates per segment if max_candidates_per_segment is specified
    # This reduces the dimension from H to max_candidates_per_segment for all subsequent operations
    if max_candidates_per_segment is not None and max_candidates_per_segment < H:
        # For each segment, we want to keep only the top max_candidates_per_segment valid candidates
        # Strategy: Sort by validity and alpha value, keep first max_candidates_per_segment
        
        # Create sort keys: valid candidates get their alpha value, invalid get a large number
        BIG_SORT = alpha.new_tensor(10.0)
        sort_keys = torch.where(valid_new, alpha, BIG_SORT)  # (B, K-1, H)
        
        # Sort and keep only top max_candidates_per_segment
        _, sort_idx = torch.sort(sort_keys, dim=2)  # (B, K-1, H)
        sort_idx_topk = sort_idx[:, :, :max_candidates_per_segment]  # (B, K-1, max_cand)
        
        # Gather the selected candidates
        alpha = torch.gather(alpha, 2, sort_idx_topk)  # (B, K-1, max_cand)
        valid_new = torch.gather(valid_new, 2, sort_idx_topk)  # (B, K-1, max_cand)
        
        # Note: zL and zR remain unchanged as full z-vectors at endpoints (B, K-1, H)
        # We're just reducing the number of candidate positions, not the z-vector dimension
        
        # Update the candidate dimension
        H_reduced = max_candidates_per_segment
    else:
        H_reduced = H
    
    # Collect candidate statistics if requested
    stats = None
    if collect_stats:
        n_candidates = valid_new.sum().item()
        stats = {'n_candidates': n_candidates}
        
        # Per-segment candidate counts: how many candidates per segment
        # valid_new is (B, K-1, H) - sum over H dimension to get candidates per segment
        candidates_per_seg = valid_new.sum(dim=2).cpu()  # (B, K-1)
        
        # Aggregate across batch: mean, min, max candidates per segment position
        if candidates_per_seg.numel() > 0:
            # Flatten to get all segment counts
            all_seg_counts = candidates_per_seg.flatten()
            # Filter out segments with no candidates for stats
            active_seg_counts = all_seg_counts[all_seg_counts > 0]
            
            if len(active_seg_counts) > 0:
                stats['seg_cand_min'] = active_seg_counts.min().item()
                stats['seg_cand_max'] = active_seg_counts.max().item()
                stats['seg_cand_mean'] = active_seg_counts.float().mean().item()
                stats['seg_cand_median'] = active_seg_counts.float().median().item()
                stats['n_active_segments'] = len(active_seg_counts)
                stats['total_segments'] = candidates_per_seg.numel()
            else:
                stats['seg_cand_min'] = 0
                stats['seg_cand_max'] = 0
                stats['seg_cand_mean'] = 0
                stats['seg_cand_median'] = 0
                stats['n_active_segments'] = 0
                stats['total_segments'] = candidates_per_seg.numel()

    # candidate times in each segment (B, K-1, H_reduced)
    dt = (tR - tL)                                # (B, K-1)
    t_new = tL[:, :, None] + alpha * dt[:, :, None]
    
    # Collect spatial distribution if requested
    if collect_stats:
        # Get valid candidate t values for spatial distribution
        valid_t_vals = t_new[valid_new].cpu()
        if len(valid_t_vals) > 0:
            # Spatial bins: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
            spatial_bins = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            spatial_hist = torch.histc(valid_t_vals, bins=5, min=0.0, max=1.0).tolist()
            stats['spatial_distribution'] = spatial_hist
        else:
            stats['spatial_distribution'] = [0, 0, 0, 0, 0]

    BIG = t.new_tensor(2.0)
    t_new_eff = torch.where(valid_new, t_new, BIG)

    # candidate z-vectors: (B, K-1, H_reduced, H)
    # Interpolate the full z-vector at each candidate position
    z_new = zL[:, :, None, :] + alpha[:, :, :, None] * (zR - zL)[:, :, None, :]
    z_new = torch.where(valid_new[:, :, :, None], z_new, torch.zeros_like(z_new))

    # Sort candidates within each segment (O(H_reduced log H_reduced) per segment)
    t_sorted, idx = torch.sort(t_new_eff, dim=2)  # (B, K-1, H_reduced)
    idx_z = idx.unsqueeze(-1).expand(-1, -1, -1, H)  # (B, K-1, H_reduced, H)
    z_sorted = torch.gather(z_new, 2, idx_z)         # (B, K-1, H_reduced, H)
    v_sorted = torch.gather(valid_new, 2, idx)       # (B, K-1, H_reduced)

    # Interleave: [t[i], candidates(seg i)] for i=0..K-2, then append t[K-1]
    t_old = t[:, :-1].unsqueeze(-1)                   # (B, K-1, 1)
    z_old = z[:, :-1].unsqueeze(2)                    # (B, K-1, 1, H)
    v_old = valid[:, :-1].unsqueeze(-1)               # (B, K-1, 1)

    t_seg = torch.cat([t_old, t_sorted], dim=2)       # (B, K-1, 1+H_reduced)
    z_seg = torch.cat([z_old, z_sorted], dim=2)       # (B, K-1, 1+H_reduced, H)
    v_seg = torch.cat([v_old, v_sorted], dim=2)       # (B, K-1, 1+H_reduced)

    t_all = t_seg.reshape(B, (K-1) * (1 + H_reduced))
    z_all = z_seg.reshape(B, (K-1) * (1 + H_reduced), H)
    v_all = v_seg.reshape(B, (K-1) * (1 + H_reduced))

    t_all = torch.cat([t_all, t[:, -1:]], dim=1)      # (B, Nall)
    z_all = torch.cat([z_all, z[:, -1:, :]], dim=1)   # (B, Nall, H)
    v_all = torch.cat([v_all, valid[:, -1:]], dim=1)  # (B, Nall)

    # STABLE COMPACTION: keep first max_knots valid entries per row, preserving order
    v_int = v_all.to(torch.int64)
    pos = torch.cumsum(v_int, dim=1) - 1               # target positions for valid entries
    keep = v_all & (pos < max_knots)

    idx_out = torch.where(keep, pos, torch.zeros_like(pos)).to(torch.int64)

    # scatter_add is safe because for keep==True, pos are unique within each row
    t_out = torch.zeros((B, max_knots), device=device, dtype=t.dtype)
    t_src = torch.where(keep, t_all, torch.zeros_like(t_all))
    t_out = t_out.scatter_add(1, idx_out, t_src)

    z_out = torch.zeros((B, max_knots, H), device=device, dtype=z.dtype)
    z_src = torch.where(keep.unsqueeze(-1), z_all, torch.zeros_like(z_all))
    z_out = z_out.scatter_add(1, idx_out.unsqueeze(-1).expand(-1, -1, H), z_src)

    v_out_i = torch.zeros((B, max_knots), device=device, dtype=torch.int64)
    v_out_i = v_out_i.scatter_add(1, idx_out, keep.to(torch.int64))
    v_out = v_out_i > 0

    # normalize padding
    t_out = torch.where(v_out, t_out, t_out.new_tensor(1.0))
    z_out = torch.where(v_out.unsqueeze(-1), z_out, torch.zeros_like(z_out))

    if collect_stats:
        return t_out, v_out, z_out, stats
    return t_out, v_out, z_out


def forward_knots(mlp: nn.Module, end_points: torch.Tensor, start_points: torch.Tensor,
                  max_knots=64, eps=1e-6, collect_stats=False, max_candidates_per_segment=None):
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
    
    # Per-layer statistics collection
    layer_stats = [] if collect_stats else None

    # ----- layer 0 preactivation via 1D projection -----
    W0 = layers[0].weight          # (H, Din)
    b0 = layers[0].bias            # (H,)
    alpha0 = d @ W0.T              # (B, H)
    beta0  = start_points @ W0.T + b0  # (B, H)
    z = alpha0[:, None, :] * t[:, :, None] + beta0[:, None, :]  # (B, K, H)
    z = torch.where(valid.unsqueeze(-1), z, torch.zeros_like(z))

    # insert crossings for layer 0 (since it has ReLU after it)
    if collect_stats:
        t, valid, z, stats = insert_zero_crossings(t, valid, z, max_knots, eps, collect_stats=True, max_candidates_per_segment=max_candidates_per_segment)
        stats['layer'] = 0
        layer_stats.append(stats)
    else:
        t, valid, z = insert_zero_crossings(t, valid, z, max_knots, eps, max_candidates_per_segment=max_candidates_per_segment)
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
            if collect_stats:
                t, valid, z, stats = insert_zero_crossings(t, valid, z, max_knots, eps, collect_stats=True, max_candidates_per_segment=max_candidates_per_segment)
                stats['layer'] = li
                layer_stats.append(stats)
            else:
                t, valid, z = insert_zero_crossings(t, valid, z, max_knots, eps, max_candidates_per_segment=max_candidates_per_segment)
            h = torch.where(valid.unsqueeze(-1), torch.relu(z), torch.zeros_like(z))

    # ----- output layer (no ReLU, no crossings) -----
    out_layer = layers[-1]  # Linear(H,1)
    y = torch.matmul(h, out_layer.weight.t()) + out_layer.bias
    y = torch.where(valid.unsqueeze(-1), y, torch.zeros_like(y))  # (B, K, 1)

    if collect_stats:
        return t, valid, y, layer_stats
    return t, valid, y

class KnotForward(nn.Module):
    def __init__(self, mlp, max_knots=128, eps=1e-6, collect_stats=False, max_candidates_per_segment=None):
        super().__init__()
        self.mlp = mlp
        self.max_knots = max_knots
        self.eps = eps
        self.collect_stats = collect_stats
        self.max_candidates_per_segment = max_candidates_per_segment

    def forward(self, end_points, start_points):
        return forward_knots(self.mlp, end_points, start_points,
                             max_knots=self.max_knots, eps=self.eps,
                             collect_stats=self.collect_stats,
                             max_candidates_per_segment=self.max_candidates_per_segment)


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
                     mesh_resolution: int = 256,
                     mesh_save_interval: int = 50,
                     collect_detailed_stats: bool = False,
                     max_knots: int = 32,
                     max_candidates_per_segment: int = None):
    device = splines.start_points.device
    mlp = mlp.to(device)

    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    N = splines.start_points.shape[0]

    loss_history = []
    best_loss = float('inf')

    pbar = tqdm(range(epochs))
    
    knot_fwd = KnotForward(mlp, max_knots=max_knots, eps=1e-6, collect_stats=False, max_candidates_per_segment=max_candidates_per_segment).to(device)

    knot_fwd = torch.compile(
        knot_fwd,
        backend="inductor",
        mode="reduce-overhead",   # try "max-autotune" later
        fullgraph=False
    )
    
    # Separate instance for detailed statistics collection (not compiled for flexibility)
    knot_fwd_stats = KnotForward(mlp, max_knots=max_knots, eps=1e-6, collect_stats=True, max_candidates_per_segment=max_candidates_per_segment).to(device) if collect_detailed_stats else None

    compiled_loss = torch.compile(integral_loss, backend="inductor", mode="reduce-overhead", fullgraph=False)
    

    for epoch in pbar:
        perm = torch.randperm(N, device=device)

        epoch_loss = 0.0
        n_batches = 0
        epoch_knot_counts = []

        for i in range(0, N, batch_size):
            batch_idx = perm[i:i + batch_size]

            p0 = splines.start_points[batch_idx]
            p1 = splines.end_points[batch_idx]
            knots_gt = splines.knots[batch_idx]
            values_gt = splines.values[batch_idx]

            optimizer.zero_grad(set_to_none=True)

            pred_t, pred_valid, pred_y = knot_fwd(p1, p0)

            # Track knot statistics
            knot_counts = pred_valid.sum(dim=1).cpu()  # (B,) - number of valid knots per sample
            epoch_knot_counts.append(knot_counts)

            loss_sum = compiled_loss(pred_t, pred_y, knots_gt, values_gt, pred_valid)
            loss_mean = loss_sum / batch_idx.numel()

            loss_mean.backward()

            torch.nn.utils.clip_grad_norm_(mlp.parameters(), clip_grad_norm)

            optimizer.step()

            epoch_loss += loss_mean.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        
        # Compute knot statistics for this epoch
        all_knot_counts = torch.cat(epoch_knot_counts)
        knot_min = all_knot_counts.min().item()
        knot_max = all_knot_counts.max().item()
        knot_mean = all_knot_counts.float().mean().item()
        knot_median = all_knot_counts.float().median().item()
        knot_std = all_knot_counts.float().std().item()
        knot_q25 = all_knot_counts.float().quantile(0.25).item()
        knot_q75 = all_knot_counts.float().quantile(0.75).item()
        knot_saturation = (all_knot_counts >= max_knots).float().mean().item() * 100  # % hitting max_knots limit

        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                torch.save({
                    'epoch': epoch,
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

        pbar.set_postfix({
            'loss': f"{avg_loss:.6f}", 
            'best': f"{best_loss:.6f}",
            'knots': f"μ={knot_mean:.1f}±{knot_std:.1f} [{knot_min:.0f},{knot_median:.0f},{knot_max:.0f}]",
            'sat': f"{knot_saturation:.1f}%"
        })
        
        # Print detailed statistics every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} - Detailed Knot Statistics")
            print(f"{'='*60}")
            print(f"  Min / Q25 / Median / Q75 / Max: {knot_min:.0f} / {knot_q25:.0f} / {knot_median:.0f} / {knot_q75:.0f} / {knot_max:.0f}")
            print(f"  Mean ± Std: {knot_mean:.2f} ± {knot_std:.2f}")
            print(f"  Saturation (at max_knots={max_knots}): {knot_saturation:.1f}%")
            print(f"\n  Distribution:")
            bins = [0, 8, 16, 24, 32, float('inf')]
            bin_labels = ['0-7', '8-15', '16-23', '24-31', '32+']
            for i in range(len(bins)-1):
                if bins[i+1] == float('inf'):
                    count = (all_knot_counts >= bins[i]).sum().item()
                else:
                    count = ((all_knot_counts >= bins[i]) & (all_knot_counts < bins[i+1])).sum().item()
                pct = count / len(all_knot_counts) * 100
                bar = '█' * int(pct / 2)  # Scale for display
                print(f"    {bin_labels[i]:>6s}: {pct:5.1f}% {bar}")
            
            # Collect detailed per-layer and spatial statistics
            if collect_detailed_stats and knot_fwd_stats is not None:
                print(f"\n  {'='*58}")
                print(f"  Per-Layer Candidate Analysis")
                print(f"  {'='*58}")
                
                # Sample a batch for detailed analysis
                sample_size = min(256, N)
                sample_idx = torch.randperm(N, device=device)[:sample_size]
                p0_sample = splines.start_points[sample_idx]
                p1_sample = splines.end_points[sample_idx]
                
                with torch.no_grad():
                    _, _, _, layer_stats_list = knot_fwd_stats(p1_sample, p0_sample)
                
                # Aggregate stats across all batches
                total_candidates_per_layer = {}
                spatial_dist_per_layer = {}
                segment_stats_per_layer = {}
                
                for stats in layer_stats_list:
                    layer_id = stats['layer']
                    if layer_id not in total_candidates_per_layer:
                        total_candidates_per_layer[layer_id] = 0
                        spatial_dist_per_layer[layer_id] = [0, 0, 0, 0, 0]
                        segment_stats_per_layer[layer_id] = {
                            'min': stats.get('seg_cand_min', 0),
                            'max': stats.get('seg_cand_max', 0),
                            'mean': stats.get('seg_cand_mean', 0),
                            'median': stats.get('seg_cand_median', 0),
                            'n_active': stats.get('n_active_segments', 0),
                            'total': stats.get('total_segments', 0)
                        }
                    
                    total_candidates_per_layer[layer_id] += stats['n_candidates']
                    for i, count in enumerate(stats['spatial_distribution']):
                        spatial_dist_per_layer[layer_id][i] += count
                
                # Print per-layer statistics
                print(f"\n  Candidates generated per layer (sample of {sample_size} splines):")
                total_all_layers = sum(total_candidates_per_layer.values())
                for layer_id in sorted(total_candidates_per_layer.keys()):
                    n_cand = total_candidates_per_layer[layer_id]
                    pct = (n_cand / total_all_layers * 100) if total_all_layers > 0 else 0
                    bar = '█' * int(pct / 2)
                    print(f"    Layer {layer_id}: {n_cand:6d} candidates ({pct:5.1f}%) {bar}")
                
                # Print spatial distribution per layer
                print(f"\n  Spatial distribution of candidates (t-values in [0,1]):")
                spatial_bins_labels = ['[0.0-0.2)', '[0.2-0.4)', '[0.4-0.6)', '[0.6-0.8)', '[0.8-1.0]']
                for layer_id in sorted(spatial_dist_per_layer.keys()):
                    print(f"\n    Layer {layer_id}:")
                    spatial_counts = spatial_dist_per_layer[layer_id]
                    total_spatial = sum(spatial_counts)
                    if total_spatial > 0:
                        for i, (label, count) in enumerate(zip(spatial_bins_labels, spatial_counts)):
                            pct = count / total_spatial * 100
                            bar = '█' * int(pct / 2.5)  # Smaller bars for readability
                            print(f"      {label:>12s}: {pct:5.1f}% {bar}")
                    else:
                        print(f"      No candidates in this layer")
                
                # Print segment-level statistics
                print(f"\n  Segment-level candidate distribution:")
                print(f"  (Shows how candidates are distributed across segments between existing knots)")
                for layer_id in sorted(segment_stats_per_layer.keys()):
                    seg_stats = segment_stats_per_layer[layer_id]
                    print(f"\n    Layer {layer_id}:")
                    if seg_stats['n_active'] > 0:
                        print(f"      Active segments: {seg_stats['n_active']} / {seg_stats['total']}")
                        print(f"      Candidates per active segment:")
                        print(f"        Min:    {seg_stats['min']:.1f}")
                        print(f"        Median: {seg_stats['median']:.1f}")
                        print(f"        Mean:   {seg_stats['mean']:.1f}")
                        print(f"        Max:    {seg_stats['max']:.1f}")
                        
                        # Interpret the results
                        if seg_stats['max'] > 3 * seg_stats['mean']:
                            print(f"      ⚠️  High variance! Some segments get {seg_stats['max']:.0f}x more candidates than average")
                        elif seg_stats['max'] < 2 * seg_stats['mean']:
                            print(f"      ✓  Candidates fairly uniformly distributed across segments")
                    else:
                        print(f"      No active segments")
            
            print(f"{'='*60}\n")

    # Save final mesh at the end of training
    if extract_mesh and save_path:
        final_mesh_path = save_path / 'mesh_final.ply'
        extract_mesh_marching_cubes(mlp, save_path=final_mesh_path, resolution=mesh_resolution, device=device)

    return loss_history
