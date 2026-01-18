import click
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

from neural_spline.train_fast import train_model_fast

from .polygons import generate_polygons
from .geometry import constrained_recursive_pca, flatten_pca_tree
from .spline import compute_splines
from .simplification import simplify_sdf_to_knots_batch
from .types import Spline
from .model import ReluMLP
from .train_optimized import train_model_optimized, update_spline_predictions
from .utils import load_mesh_data


@click.command()
@click.option('--model', type=str, default='Armadillo',
              help='Model name. For 2D: simple/hard. For 3D: mesh name (e.g., Armadillo)')
@click.option('--epochs', type=int, default=1000, help='Number of training epochs')
@click.option('--hidden-dim', type=int, default=32, help='Hidden dimension of the MLP')
@click.option('--num-layers', type=int, default=4, help='Number of hidden layers')
@click.option('--batch-size', type=int, default=8, help='Batch size for training')
@click.option('--lr', type=float, default=0.01, help='Learning rate')
@click.option('--max-depth', type=int, default=5, help='Maximum PCA recursion depth')
@click.option('--exact/--no-exact', default=True, 
              help='Use exact distance computation (slower, more accurate) vs KNN approximation (faster)')
@click.option('--no-compile', is_flag=True, default=False,
              help='Disable torch.compile (enabled by default)')
@click.option('--precision', type=click.Choice(['fp32', 'bfloat16', 'fp16']), default='fp32',
              help='Training precision (fp32, bfloat16, fp16)')
def main(model, epochs, hidden_dim, num_layers, batch_size, lr, max_depth, exact, no_compile, precision):
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
    
    data = load_mesh_data(model, dim)
    if data is None:
        return
    
    # ========================================
    # 4. Constrained Recursive PCA Decomposition (Optimized)
    # ========================================
    click.echo(f"\nPerforming constrained recursive PCA decomposition...")
    click.echo(f"  Input: {len(data['vertices'])} vertices in {dim}")
    click.echo(f"  Features: Mass-based orientation, degeneracy handling, slab clipping")
    # click.echo(f"  Parameters: min_points={min_points}, max_depth={max_depth}")
    
    pca_tree = constrained_recursive_pca(
        data['vertices'],
        min_points=5,
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
    # 5. Create Splines
    # ========================================
    splines = compute_splines(data, components, 50_000)

    
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
        'n_vertices': len(data['vertices']),
        'n_components': len(components),
        'pca_config': {
            'min_points': 15,
            'max_depth': max_depth
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
    # 9. Train Model (Optimized)
    # ========================================
    click.echo(f"\n{'='*60}")
    click.echo(f"TRAINING NEURAL NETWORK (OPTIMIZED)")
    click.echo(f"{'='*60}")
    
    # Create model
    mlp = ReluMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = mlp.to(device)
    loss_history = train_model_fast(
        mlp=mlp,
        splines=splines,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        clip_grad_norm=1.0,
        save_path=exp_dir,
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
    click.echo(f"Vertices:        {len(data['vertices']):,}")
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
