import click
import torch
from pathlib import Path
from datetime import datetime
from neural_spline.train_fast import train_model_fast
from neural_spline.geometry import constrained_recursive_pca, flatten_pca_tree
from .spline import compute_splines
from .model import ReluMLP
from .utils import load_mesh_data


@click.command()
@click.option('--model', type=str, default='Armadillo')
@click.option('--epochs', type=int, default=1000)
@click.option('--hidden-dim', type=int, default=64)
@click.option('--num-layers', type=int, default=4)
@click.option('--batch-size', type=int, default=2048)
@click.option('--lr', type=float, default=0.01)
@click.option('--max-depth', type=int, default=5)
@click.option('--use-knn/--no-use-knn', default=True)
@click.option('--extract-mesh/--no-extract-mesh', default=True)
@click.option('--mesh-resolution', type=int, default=256)
@click.option('--save-dir', type=str, default=None)
@click.option('--mesh-save-interval', type=int, default=50)
@click.option('--skip-connections/--no-skip-connections', default=False)
def main(model, epochs, hidden_dim, num_layers, batch_size, lr, max_depth, use_knn, extract_mesh, mesh_resolution, save_dir, mesh_save_interval, skip_connections):
    if model.lower() in ['simple', 'hard']:
        dim = '2d'
        input_dim = 2
    else:
        dim = '3d'
        input_dim = 3
    
    data = load_mesh_data(model, dim)
    if data is None:
        return
    
    pca_tree = constrained_recursive_pca(
        data['vertices'],
        min_points=5,
        max_depth=max_depth
    )
    
    if pca_tree is None:
        return
    
    components = flatten_pca_tree(pca_tree)

    splines = compute_splines(data, components, 500_000, use_knn_method=use_knn)
    print(f"Computed {len(splines.start_points)} splines")

    mlp = ReluMLP(
        input_dim=input_dim,
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
    
    train_model_fast(
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
    )


if __name__ == "__main__":
    main()
