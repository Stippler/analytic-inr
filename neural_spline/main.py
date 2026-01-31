import click
import torch
torch._dynamo.reset()
import numpy as np
from pathlib import Path
from datetime import datetime
from neural_spline.train_fast import train_model_fast
from neural_spline.geometry import constrained_recursive_pca, flatten_pca_tree
from .spline import compute_splines
from .model import ReluMLP
from .utils import load_mesh_data
from .types import Splines
from .edge import initialize_edge_splines
import hashlib


def get_cache_key(model: str, max_depth: int, use_knn: bool, use_edges: bool) -> str:
    """Generate a unique cache key based on spline computation parameters."""
    key_string = f"{model}_{max_depth}_{use_knn}_{use_edges}"
    return hashlib.md5(key_string.encode()).hexdigest()


def save_splines(splines: Splines, cache_path: Path) -> None:
    """Save splines to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as a dictionary of tensors
    splines_dict = {
        'start_points': splines.start_points,
        'end_points': splines.end_points,
        'knots': splines.knots,
        'values': splines.values,
        'normals': splines.normals,
        'nconf': splines.nconf,
        'sconf': splines.sconf,
        'sign_uncertain': splines.sign_uncertain,
    }
    
    torch.save(splines_dict, cache_path)
    print(f"Saved splines to {cache_path}")


def load_splines(cache_path: Path) -> Splines:
    """Load splines from disk."""
    splines_dict = torch.load(cache_path)
    
    splines = Splines(
        start_points=splines_dict['start_points'],
        end_points=splines_dict['end_points'],
        knots=splines_dict['knots'],
        values=splines_dict['values'],
        normals=splines_dict['normals'],
        nconf=splines_dict.get('nconf'),
        sconf=splines_dict.get('sconf'),
        sign_uncertain=splines_dict.get('sign_uncertain'),
    )
    
    print(f"Loaded splines from {cache_path}")
    return splines


@click.command()
@click.option('--model', type=str, default='Stanford_armadillo')
@click.option('--epochs', type=int, default=1000)
@click.option('--hidden-dim', type=int, default=64)
@click.option('--num-layers', type=int, default=3)
@click.option('--batch-size', type=int, default=4096)
@click.option('--lr', type=float, default=0.001)
@click.option('--max-depth', type=int, default=5)
@click.option('--use-knn/--no-use-knn', default=True)
@click.option('--use-edges/--no-use-edges', default=True, help='Use edge-based splines (default) instead of PCA-based splines')
@click.option('--extract-mesh/--no-extract-mesh', default=True)
@click.option('--mesh-resolution', type=int, default=256)
@click.option('--save-dir', type=str, default=None)
@click.option('--mesh-save-interval', type=int, default=50)
@click.option('--skip-connections/--no-skip-connections', default=False)
@click.option('--max-knots', type=int, default=64, help='Maximum number of knots per spline')
@click.option('--max-seg-insertions', type=int, default=16, help='Maximum candidates per segment (None = no limit, e.g., 4 or 8 for speedup)')
@click.option('--weight-segment', type=float, default=1.0, help='Weight for segment loss')
@click.option('--weight-normal', type=float, default=1e-3, help='Weight for normal loss')
@click.option('--weight-eikonal', type=float, default=1e-6, help='Weight for eikonal loss')
def main(model, epochs, hidden_dim, num_layers, batch_size, lr, max_depth, use_knn, use_edges, extract_mesh, mesh_resolution, save_dir, mesh_save_interval, skip_connections, max_knots, max_seg_insertions, weight_segment, weight_normal, weight_eikonal):
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for cached splines
    cache_key = get_cache_key(model, max_depth, use_knn, use_edges)
    cache_dir = Path('data/cache')
    cache_path = cache_dir / f"splines_{cache_key}.pt"
    
    if cache_path.exists():
        print(f"Found cached splines for model={model}, use_edges={use_edges}, max_depth={max_depth}, use_knn={use_knn}")
        splines = load_splines(cache_path)
    else:
        print(f"No cached splines found. Computing from scratch...")
        
        data = load_mesh_data(model)
        if data is None:
            return
        
        if use_edges:
            # Use edge-based spline initialization
            print("Using edge-based spline initialization...")
            splines = initialize_edge_splines(data)
        else:
            # Use PCA-based spline initialization
            print("Using PCA-based spline initialization...")
            pca_tree = constrained_recursive_pca(
                data['vertices'],
                min_points=15,
                max_depth=max_depth
            )
            
            if pca_tree is None:
                return
            
            components = flatten_pca_tree(pca_tree)

            splines = compute_splines(data, components, 500_000, use_knn_method=use_knn)
        
        print(f"Computed {len(splines.start_points)} splines")
        
        # Save splines to cache
        save_splines(splines, cache_path)

        del data
        torch.cuda.empty_cache()

    mlp = ReluMLP(
        input_dim=2 if data['type'] == '2d' else 3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        skip_connections=skip_connections,
    )

    total_params = sum(p.numel() for p in mlp.parameters())
    print(f"Total parameters: {total_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = mlp.to(device)
    
    output_path = None
    if save_dir or extract_mesh:
        if save_dir:
            output_path = Path(save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path('outputs') / model / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
    
    loss_histories = train_model_fast(
        mlp=mlp,
        splines=splines,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        clip_grad_norm=1.0,
        save_path=output_path,
        extract_mesh=extract_mesh and dim == '3d',
        mesh_resolution=mesh_resolution,
        mesh_save_interval=mesh_save_interval,
        max_knots=max_knots,
        max_seg_insertions=max_seg_insertions,
        weight_segment=weight_segment,
        weight_normal=weight_normal,
        weight_eikonal=weight_eikonal,
    )
    
    # Plot loss histories
    if output_path:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total loss
        axes[0, 0].plot(loss_histories['total'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Segment loss
        axes[0, 1].plot(loss_histories['segment'])
        axes[0, 1].set_title('Segment Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Normal loss
        axes[1, 0].plot(loss_histories['normal'])
        axes[1, 0].set_title('Normal Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Eikonal loss
        axes[1, 1].plot(loss_histories['eikonal'])
        axes[1, 1].set_title('Eikonal Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'loss_history.png', dpi=150)
        print(f"Saved loss plot to {output_path / 'loss_history.png'}")
        plt.close()


if __name__ == "__main__":
    main()
