
import click
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from .polygons import generate_polygons
from .preprocess import preprocess_polygons_to_splines
from .spline import Spline
from .model import ReluMLP
from .train import train_model, reset_predictions, compute_metrics
from .metrics import compute_sdf_metrics, print_metrics
from .vis import plot_sdf_heatmap, plot_splines


@click.group()
def cli():
    """Neural Spline Training CLI - Train on 2D or 3D data."""
    pass


@cli.command()
@click.option('--dataset', type=click.Choice(['simple', 'hard'], case_sensitive=False), 
              default='simple', help='Dataset: simple (convex) or hard (non-convex star polygons)')
@click.option('--epochs', type=int, default=1000, help='Number of training epochs')
@click.option('--hidden-dim', type=int, default=16, help='Hidden dimension of the MLP')
@click.option('--num-layers', type=int, default=3, help='Number of hidden layers')
@click.option('--batch-size', type=int, default=8, help='Batch size for training')
@click.option('--lr', type=float, default=0.01, help='Learning rate')
@click.option('--save/--no-save', default=True, help='Save experiment results')
def train_2d(dataset, epochs, hidden_dim, num_layers, batch_size, lr, save):
    """Train on 2D polygon datasets."""
    input_dim = 2
    
    # Setup experiment directory
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path('data') / 'experiments' / f'{input_dim}d' / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"Saving to: {exp_dir}")
    else:
        exp_dir = None
    
    # Generate polygons
    if dataset.lower() == 'simple':
        polygons = generate_polygons('1x16', convex=True)
    else:
        polygons = generate_polygons('3x16', convex=False, stretch=(1, 0.5), 
                                    star_ratio=0.9, rotation=[np.pi/4, -np.pi/3, np.pi/5])
    
    # Preprocess to splines
    all_knots, hierarchy = preprocess_polygons_to_splines(polygons, max_depth=2, 
                                                           n_samples=1000, tolerance=0.005)
    
    splines = []
    for knot_data in all_knots:
        p_start, p_end = knot_data['line']
        splines.append(Spline(
            start_point=torch.tensor(p_start, dtype=torch.float32),
            end_point=torch.tensor(p_end, dtype=torch.float32),
            pred_values=None, pred_knots=None,
            gt_knots=torch.tensor(knot_data['knot_t'], dtype=torch.float32),
            gt_values=torch.tensor(knot_data['knot_sdf'], dtype=torch.float32),
            label=knot_data['label'], depth=knot_data['depth'], pc_type=knot_data['pc_type']
        ))
    
    # Create and train model
    torch.manual_seed(0)
    mlp = ReluMLP(input_dim, hidden_dim, num_layers, skip_connections=False)
    n_params = sum(p.numel() for p in mlp.parameters())
    
    click.echo(f"Training {dataset} | {n_params} params | {len(splines)} splines | {epochs} epochs")
    
    loss_history = train_model(mlp, splines, epochs=epochs, batch_size=batch_size, 
                               lr=lr, clip_grad_norm=1.0, save_path=exp_dir if save else None)
    
    # Compute metrics
    reset_predictions(mlp, splines)
    spline_metrics = compute_metrics(splines)
    sdf_metrics = compute_sdf_metrics(mlp, polygons, n_samples=10000, bbox_min=-1.0, bbox_max=1.0)
    
    click.echo(f"\nResults: MAE={sdf_metrics['mae']:.6f} | IoU={sdf_metrics['iou_interior']:.4f} | "
               f"Sign Acc={sdf_metrics['sign_accuracy']:.4f}")
    
    if save:
        # Save plots
        plot_sdf_heatmap(mlp, splines=splines, polygons=polygons, resolution=300,
                        title="Learned SDF", save_path=exp_dir / 'sdf_heatmap.png')
        plot_splines(splines, save_path=exp_dir / 'spline_predictions.png')
        
        # Loss curve
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(loss_history, linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Training Loss'); ax.grid(alpha=0.3); ax.set_yscale('log')
        plt.tight_layout()
        fig.savefig(exp_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save final model and data
        torch.save({'model_state_dict': mlp.state_dict(), 'input_dim': input_dim,
                   'hidden_dim': hidden_dim, 'num_layers': num_layers}, exp_dir / 'final_model.pt')
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump({'dataset': dataset, 'input_dim': input_dim, 'epochs': epochs,
                      'hidden_dim': hidden_dim, 'num_layers': num_layers, 'batch_size': batch_size,
                      'lr': lr, 'n_params': n_params, 'n_splines': len(splines),
                      'n_polygons': len(polygons)}, f, indent=2)
        
        with open(exp_dir / 'metrics.json', 'w') as f:
            json.dump({'spline_metrics': spline_metrics, 'sdf_metrics': sdf_metrics,
                      'final_loss': float(loss_history[-1])}, f, indent=2)
        
        np.save(exp_dir / 'loss_history.npy', np.array(loss_history))
        
        click.echo(f"Saved to: {exp_dir}")
    else:
        # Show plots if not saving
        plot_sdf_heatmap(mlp, splines=splines, polygons=polygons, resolution=300, title="Learned SDF")
        plot_splines(splines)


@cli.command()
@click.option('--dataset', type=str, default='stanford', help='Dataset name')
@click.option('--model-name', type=str, default='Armadillo.ply', help='Model filename')
@click.option('--epochs', type=int, default=1000, help='Number of training epochs')
@click.option('--hidden-dim', type=int, default=32, help='Hidden dimension of the MLP')
@click.option('--num-layers', type=int, default=4, help='Number of hidden layers')
@click.option('--batch-size', type=int, default=8, help='Batch size for training')
@click.option('--lr', type=float, default=0.01, help='Learning rate')
@click.option('--save/--no-save', default=True, help='Save experiment results')
def train_3d(dataset, model_name, epochs, hidden_dim, num_layers, batch_size, lr, save):
    """Train on 3D mesh datasets."""
    import trimesh
    
    input_dim = 3
    mesh_path = Path('data') / dataset / model_name
    
    if not mesh_path.exists():
        click.echo(f"Error: Mesh not found at {mesh_path}", err=True)
        return
    
    # Setup experiment directory
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path('data') / 'experiments' / f'{input_dim}d' / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"Saving to: {exp_dir}")
    else:
        exp_dir = None
    
    # Load and normalize mesh
    click.echo("Loading mesh...")
    mesh = trimesh.load(str(mesh_path), force='mesh')
    click.echo(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    
    padding = 0.1
    mesh.vertices -= mesh.centroid
    max_extent = np.max(mesh.extents)
    if max_extent > 0:
        scale = 2.0 * (1.0 - padding) / max_extent
        mesh.vertices *= scale
    click.echo(f"  Normalized bounds: {mesh.bounds.T}")
    
    # Use mesh vertices directly for PCA (no sampling bias)
    click.echo("Using mesh vertices for PCA...")
    surface_points = mesh.vertices.copy()
    click.echo(f"  Using {len(surface_points)} vertices")
    
    # Preprocess using hierarchical PCA (works for 3D too)
    from .preprocess import build_hierarchical_pca, extract_all_lines, compute_sdf_sampling_3d
    
    click.echo("Building hierarchical PCA decomposition...")
    hierarchy = build_hierarchical_pca(surface_points, max_depth=2, box_min=-1, box_max=1)
    all_lines_data = extract_all_lines(hierarchy)
    line_segments = [line for line, _ in all_lines_data]
    labels = [label for _, label in all_lines_data]
    click.echo(f"  Generated {len(line_segments)} line segments")
    
    # Compute SDF along each line using trimesh
    click.echo("Computing SDF along lines and simplifying to knots...")
    all_knots = []
    from .preprocess import simplify_sdf_to_knots
    
    for idx, ((p_start, p_end), label) in enumerate(zip(line_segments, labels)):
        if idx % 10 == 0:
            click.echo(f"  Processing line {idx+1}/{len(line_segments)}...")
        
        # Reduce samples for 3D - still enough resolution but much faster
        t, sdf = compute_sdf_sampling_3d(p_start, p_end, mesh, n_samples=200)
        
        # Simplify to knots
        knot_t, knot_sdf, max_err, mean_err = simplify_sdf_to_knots(t, sdf, tolerance=0.005)
        
        depth = int(label.split(':')[0].replace('D', '')) if 'D' in label else 0
        pc_type = 'pc1' if 'pc1' in label else ('pc2' if 'pc2' in label else 'pc3')
        
        all_knots.append({
            't': t, 'sdf': sdf,
            'knot_t': knot_t, 'knot_sdf': knot_sdf,
            'max_error': max_err, 'mean_error': mean_err,
            'label': label, 'line': (p_start, p_end),
            'depth': depth, 'pc_type': pc_type
        })
    
    click.echo(f"  Generated {len(all_knots)} splines with knots")
    
    # Convert to Spline objects
    click.echo("Converting to Spline objects...")
    splines = []
    for knot_data in all_knots:
        p_start, p_end = knot_data['line']
        splines.append(Spline(
            start_point=torch.tensor(p_start, dtype=torch.float32),
            end_point=torch.tensor(p_end, dtype=torch.float32),
            pred_values=None, pred_knots=None,
            gt_knots=torch.tensor(knot_data['knot_t'], dtype=torch.float32),
            gt_values=torch.tensor(knot_data['knot_sdf'], dtype=torch.float32),
            label=knot_data['label'], depth=knot_data['depth'], pc_type=knot_data['pc_type']
        ))
    
    # Create and train model
    click.echo("Creating model...")
    torch.manual_seed(0)
    mlp = ReluMLP(input_dim, hidden_dim, num_layers, skip_connections=False)
    n_params = sum(p.numel() for p in mlp.parameters())
    
    click.echo(f"Training {dataset}/{model_name} | {n_params} params | {len(splines)} splines | {epochs} epochs")
    click.echo("Starting training...")
    
    loss_history = train_model(mlp, splines, epochs=epochs, batch_size=batch_size,
                               lr=lr, clip_grad_norm=1.0, save_path=exp_dir if save else None)
    
    # Compute metrics
    click.echo("\nComputing spline metrics...")
    reset_predictions(mlp, splines)
    spline_metrics = compute_metrics(splines)
    
    # Compute 3D SDF metrics
    click.echo("Computing 3D SDF metrics...")
    from .metrics import compute_sdf_metrics_3d
    sdf_metrics = compute_sdf_metrics_3d(mlp, mesh, n_samples=10000, bbox_min=-1.0, bbox_max=1.0)
    
    click.echo(f"\nResults: MAE={sdf_metrics['mae']:.6f} | IoU={sdf_metrics['iou_interior']:.4f} | "
               f"Sign Acc={sdf_metrics['sign_accuracy']:.4f}")
    
    if save:
        click.echo("Saving results...")
        # Loss curve
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(loss_history, linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Training Loss'); ax.grid(alpha=0.3); ax.set_yscale('log')
        plt.tight_layout()
        fig.savefig(exp_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save final model and data
        torch.save({'model_state_dict': mlp.state_dict(), 'input_dim': input_dim,
                   'hidden_dim': hidden_dim, 'num_layers': num_layers}, exp_dir / 'final_model.pt')
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump({'dataset': dataset, 'model_name': model_name, 'input_dim': input_dim,
                      'epochs': epochs, 'hidden_dim': hidden_dim, 'num_layers': num_layers,
                      'batch_size': batch_size, 'lr': lr, 'n_params': n_params,
                      'n_splines': len(splines), 'n_vertices': len(mesh.vertices),
                      'n_faces': len(mesh.faces)}, f, indent=2)
        
        with open(exp_dir / 'metrics.json', 'w') as f:
            json.dump({'spline_metrics': spline_metrics, 'sdf_metrics': sdf_metrics,
                      'final_loss': float(loss_history[-1])}, f, indent=2)
        
        np.save(exp_dir / 'loss_history.npy', np.array(loss_history))
        
        # Save mesh for reference
        mesh.export(exp_dir / 'mesh.ply')
        
        click.echo(f"Saved to: {exp_dir}")


if __name__ == "__main__":
    cli()