"""
Demo script for Analytic INR cell construction
"""
import numpy as np
import torch
from ainr.cell import Surface2D, Cell, SubCell, build_layer, collapse_layers, solve_output_layer_analytically
from ainr.ground_truth import generate_polygons
from ainr.vis import plot_polygons, plot_cell_sdf2 as plot_cell_sdf
from ainr.model import ReluMLP


if __name__ == "__main__":
    polygons = generate_polygons('1x32', convex=False, stretch=(1, 0.5), star_ratio=0.9)
    # plot_polygons(polygons)
    
    surfaces = [Surface2D(polygon, True) for polygon in polygons]
    mlp = ReluMLP(2, 8, 2, skip_connections=False)
    
    # Create initial subcell with activations initialized to zeros (size = hidden_dim)
    initial_subcell = SubCell(surfaces, np.zeros(mlp.hidden_dim, dtype=np.float32))
    cells = [Cell([initial_subcell], np.eye(2), np.zeros(2), [])]

    for layer in mlp.layers:
        layer.weight.data = torch.zeros_like(layer.weight.data)
        layer.bias.data = torch.zeros_like(layer.bias.data)

    # Build all layers iteratively
    for layer_idx in range(mlp.num_layers):
        # Build layer
        cells = build_layer(mlp, layer_idx, cells, vis=False)
        
        # Collapse previous layers with this layer
        cells = collapse_layers(
            cells, 
            mlp.layers[layer_idx].weight.data, 
            mlp.layers[layer_idx].bias.data,
            mlp.hidden_dim if layer_idx < mlp.num_layers else 0
        )

    # Solve output layer analytically using boundary constraints
    solve_output_layer_analytically(mlp, cells)

    plot_cell_sdf(mlp, cells=cells, title="Final Trained Model")

