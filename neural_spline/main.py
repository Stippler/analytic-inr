import click
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from pytorch3d.io import load_ply

from .polygons import generate_polygons
from .geometry import constrained_recursive_pca, flatten_pca_tree
from .fields import compute_sdf_warp
from .simplification import simplify_sdf_to_knots_batch
from .types import Spline
from .network import create_mlp
from .train_optimized import train_model_optimized, update_spline_predictions


@click.command()
@click.option('--model', type=str, default='Armadillo',
              help='Model name. For 2D: simple/hard. For 3D: mesh name (e.g., Armadillo)')
@click.option('--epochs', type=int, default=1000, help='Number of training epochs')
@click.option('--hidden-dim', type=int, default=32, help='Hidden dimension of the MLP')
@click.option('--num-layers', type=int, default=4, help='Number of hidden layers')
@click.option('--batch-size', type=int, default=8, help='Batch size for training')
@click.option('--lr', type=float, default=0.01, help='Learning rate')
@click.option('--max-depth', type=int, default=3, help='Maximum PCA recursion depth')
@click.option('--exact/--no-exact', default=True, 
              help='Use exact distance computation (slower, more accurate) vs KNN approximation (faster)')
@click.option('--no-compile', is_flag=True, default=False,
              help='Disable torch.compile (enabled by default)')
@click.option('--precision', type=click.Choice(['fp32', 'bfloat16', 'fp16']), default='fp32',
              help='Training precision (fp32, bfloat16, fp16)')
@click.option('--warp-device', type=int, default=0,
              help='CUDA device ID for Warp SDF computation')
def main(model, epochs, hidden_dim, num_layers, batch_size, lr, max_depth, exact, no_compile, precision, warp_device):
    """Unified training command for 2D and 3D."""
    
    # ========================================
    # 1. Infer dimension from model name
    # ========================================
    if model.lower() in ['simple', 'hard']:
        dim = '2d'
        input_dim = 2
    else:
        dim = '3d'
        input_dim = 3
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Neural Spline Training - {dim.upper()} (OPTIMIZED)")
    click.echo(f"Model: {model}")
    click.echo(f"Precision: {precision}")
    click.echo(f"torch.compile: {'Disabled' if no_compile else 'Enabled'}")
    click.echo(f"Warp Device: cuda:{warp_device}")
    click.echo(f"{'='*60}\n")
    
    # ========================================
    # 2. Setup Experiment Directory
    # ========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path('data') / 'experiments' / dim / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Experiment directory: {exp_dir}")
    
    # ========================================
    # 3. Load Data
    # ========================================
    click.echo(f"\nLoading {dim} data...")
    
    if dim == '2d':
        # Generate 2D polygons
        if model.lower() == 'simple':
            click.echo("  Generating simple convex polygons...")
            polygons = generate_polygons('1x16', convex=True)
        elif model.lower() == 'hard':
            click.echo("  Generating hard non-convex polygons...")
            polygons = generate_polygons('3x16', convex=False, stretch=(1, 0.5), 
                                         star_ratio=0.9, rotation=[np.pi/4, -np.pi/3, np.pi/5])
        else:
            click.echo(f"ERROR: Unknown model '{model}'. Use 'simple' or 'hard' for 2D, or a mesh name for 3D", err=True)
            return
        
        # Extract vertices from polygons and convert to tensor
        vertices_np = np.concatenate(polygons, axis=0)
        vertices = torch.from_numpy(vertices_np).float()
        click.echo(f"  Total vertices: {len(vertices)}")
        data = {'type': '2d', 'polygons': polygons, 'vertices': vertices}
        
    else:  # 3d
        # Load 3D mesh using PyTorch3D
        mesh_path = Path('data') / 'meshes' / f'{model}.ply'
        
        if not mesh_path.exists():
            # Try alternate location
            mesh_path = Path('data') / 'stanford' / f'{model}.ply'
            if not mesh_path.exists():
                click.echo(f"ERROR: Mesh not found: {model}.ply in data/meshes/ or data/stanford/", err=True)
                return
        
        click.echo(f"  Loading mesh from: {mesh_path}")
        verts, faces = load_ply(str(mesh_path))
    
        click.echo(f"  Vertices: {len(verts)}, Faces: {len(faces)}")
        
        # Normalize mesh to [-1, 1] with padding (keep as tensor)
        padding = 0.1
        centroid = verts.mean(dim=0)
        verts = verts - centroid
        
        # Compute extents
        min_coords = verts.min(dim=0)[0]
        max_coords = verts.max(dim=0)[0]
        extents = max_coords - min_coords
        max_extent = extents.max().item()
        
        if max_extent > 0:
            scale = 2.0 * (1.0 - padding) / max_extent
            verts = verts * scale
        
        bounds = torch.stack([verts.min(dim=0)[0], verts.max(dim=0)[0]], dim=1)
        click.echo(f"  Normalized bounds:")
        click.echo(f"    {bounds.T}")
        
        vertices = verts
        data = {
            'type': '3d',
            'verts': verts,
            'faces': faces,
            'vertices': vertices
        }
    
    # ========================================
    # 4. Constrained Recursive PCA Decomposition (Optimized)
    # ========================================
    click.echo(f"\nPerforming constrained recursive PCA decomposition...")
    click.echo(f"  Input: {len(vertices)} vertices in {dim}")
    click.echo(f"  Features: Mass-based orientation, degeneracy handling, slab clipping")
    
    # Use minimum of 3 points for subdivision
    min_points = 3
    
    click.echo(f"  Parameters: min_points={min_points}, max_depth={max_depth}")
    
    pca_tree = constrained_recursive_pca(
        vertices,
        min_points=min_points,
        max_depth=max_depth
    )
    
    if pca_tree is None:
        click.echo("ERROR: PCA decomposition failed (insufficient points)", err=True)
        return
    
    # Extract all components
    components = flatten_pca_tree(pca_tree)
    click.echo(f"  ✓ Generated {len(components)} PCA components")
    
    # Print component breakdown
    by_depth = {}
    by_type = {}
    for comp in components:
        by_depth[comp.depth] = by_depth.get(comp.depth, 0) + 1
        pc_type = f"PC{comp.component_idx + 1}"
        by_type[pc_type] = by_type.get(pc_type, 0) + 1
    
    click.echo(f"\n  Components by depth: {dict(sorted(by_depth.items()))}")
    click.echo(f"  Components by type: {dict(sorted(by_type.items()))}")
    
    # ========================================
    # 5. Save Configuration
    # ========================================
    config = {
        'dimension': dim,
        'input_dim': input_dim,
        'model': model,
        'epochs': epochs,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'lr': lr,
        'n_vertices': len(vertices),
        'n_components': len(components),
        'pca_config': {
            'min_points': min_points,
            'max_depth': max_depth
        },
        'sdf_config': {
            'exact': exact,
            'warp_device': warp_device
        },
        'training_config': {
            'precision': precision,
            'use_compile': not no_compile
        },
        'timestamp': timestamp
    }
    
    if dim == '3d':
        config['n_faces'] = len(data['faces'])
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    click.echo(f"\n✓ Configuration saved to {exp_dir / 'config.json'}")
    
    # ========================================
    # 6. Compute SDF using Warp (Optimized)
    # ========================================
    click.echo(f"\n{'='*60}")
    click.echo(f"COMPUTING SDF USING NVIDIA WARP")
    click.echo(f"{'='*60}")
    
    # Compute SDF using Warp kernels
    click.echo(f"\nProcessing {len(components)} components with Warp...")
    
    t_values_list, sdf_values_list = compute_sdf_warp(
        components=components,
        data=data,
        n_samples_per_unit=1000,
        device_id=warp_device
    )
    
    # ========================================
    # 7. Simplify to Knots using Numba (Optimized)
    # ========================================
    click.echo(f"\n{'='*60}")
    click.echo(f"SIMPLIFYING TO KNOTS USING NUMBA")
    click.echo(f"{'='*60}")
    
    knot_t_list, knot_sdf_list, max_errors, mean_errors = simplify_sdf_to_knots_batch(
        t_values_list=t_values_list,
        sdf_values_list=sdf_values_list,
        tolerance=0.005,
        linear_tolerance_factor=10.0
    )
    
    # Print statistics
    total_samples = sum(len(t) for t in t_values_list)
    total_knots = sum(len(t) for t in knot_t_list)
    avg_error = mean_errors.mean().item()
    max_error = max_errors.max().item()
    
    click.echo(f"\n✓ SDF Computation Complete:")
    click.echo(f"  Total samples:     {total_samples:,}")
    click.echo(f"  Total knots:       {total_knots:,}")
    click.echo(f"  Compression ratio: {total_samples / total_knots:.1f}x")
    click.echo(f"  Mean error:        {avg_error:.6f}")
    click.echo(f"  Max error:         {max_error:.6f}")
    
    # ========================================
    # 8. Create Splines for Training
    # ========================================
    click.echo(f"\n{'='*60}")
    click.echo(f"CREATING SPLINE OBJECTS")
    click.echo(f"{'='*60}")
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    click.echo(f"Using device: {device.upper()}")
    
    splines = []
    for comp, gt_t, gt_v in zip(components, knot_t_list, knot_sdf_list):
        spline = Spline(
            start_point=comp.start,
            end_point=comp.end,
            gt_knots=gt_t,
            gt_values=gt_v,
            label=comp.label,
            depth=comp.depth,
            pc_type=f"pc{comp.component_idx + 1}",
            component_idx=comp.component_idx
        )
        splines.append(spline)
    
    click.echo(f"✓ Created {len(splines)} splines for training")
    
    # ========================================
    # 9. Train Model (Optimized)
    # ========================================
    click.echo(f"\n{'='*60}")
    click.echo(f"TRAINING NEURAL NETWORK (OPTIMIZED)")
    click.echo(f"{'='*60}")
    
    # Create model
    mlp = create_mlp(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_bfloat16=(precision == 'bfloat16'),
        device=device
    )
    
    click.echo(f"\nModel Summary:")
    click.echo(mlp.summary())
    
    # Train
    use_compile = not no_compile
    loss_history = train_model_optimized(
        mlp=mlp,
        splines=splines,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        clip_grad_norm=1.0,
        save_path=exp_dir,
        use_compile=use_compile,
        precision=precision
    )
    
    # Save loss history
    np.save(exp_dir / 'loss_history.npy', np.array(loss_history))
    
    # Update predictions
    click.echo(f"\nUpdating spline predictions...")
    update_spline_predictions(mlp, splines, use_compile=use_compile)
    
    # ========================================
    # 10. Summary
    # ========================================
    actual_max_depth = max(comp.depth for comp in components)
    
    click.echo(f"\n{'='*60}")
    click.echo(f"PIPELINE COMPLETE (OPTIMIZED)")
    click.echo(f"{'='*60}")
    click.echo(f"Dimension:       {dim}")
    click.echo(f"Model:           {model}")
    click.echo(f"Vertices:        {len(vertices):,}")
    click.echo(f"Components:      {len(components)}")
    click.echo(f"Max depth:       {actual_max_depth} (requested: {max_depth})")
    click.echo(f"Knots:           {total_knots:,}")
    click.echo(f"Compression:     {total_samples / total_knots:.1f}x")
    click.echo(f"Precision:       {precision}")
    click.echo(f"torch.compile:   {'Enabled' if use_compile else 'Disabled'}")
    click.echo(f"Final loss:      {loss_history[-1]:.6f}")
    click.echo(f"Experiment:      {exp_dir}")
    click.echo(f"{'='*60}\n")


if __name__ == "__main__":
    main()
