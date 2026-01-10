"""
Visualization utilities for analytic implicit neural representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import matplotlib
from typing import List, Optional


def plot_cell_sdf(
    net,
    surfaces=None,
    resolution: int = 300,
    device=None,
    highlight_idx=None,
    title=None
):
    """
    Plot SDF heatmap with activation boundaries and surfaces.
    
    Args:
        net: Neural network model (ReluMLP)
        surfaces: Optional list of Surface2D objects to overlay on the plot
        resolution: Grid resolution for the heatmap
        device: Device to run computations on
        highlight_idx: Optional index of surface to highlight in red
        title: Optional title for the plot
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if net.input_dim != 2 or net.output_dim != 1:
        print(f"Warning: plot_cell_sdf expects 2D input and 1D output, got input_dim={net.input_dim}, output_dim={net.output_dim}")
        return
    
    # Grid & network pass
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)

    with torch.no_grad():
        net = net.to(device)
        sdf_vals, preacts = net.eval_activations(grid_pts.to(device))

    L = net.num_layers
    H = net.hidden_dim

    sdf_grid = sdf_vals.view(resolution, resolution).cpu().numpy()
    preacts = preacts.view(resolution, resolution, L, H).cpu()

    fig, ax = plt.subplots(figsize=(10, 8))

    # SDF heat-map
    vmin, vmax = sdf_grid.min(), sdf_grid.max()
    if vmin >= 0:
        cmap = LinearSegmentedColormap.from_list('pos', ['white', 'red', 'darkred'])
        norm = plt.Normalize(vmin=0, vmax=vmax)
    elif vmax <= 0:
        cmap = LinearSegmentedColormap.from_list('neg', ['darkblue', 'blue', 'white'])
        norm = plt.Normalize(vmin=vmin, vmax=0)
    else:
        cmap = LinearSegmentedColormap.from_list('cent', ['blue', 'white', 'red'])
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax.contourf(xx.cpu().numpy(), yy.cpu().numpy(), sdf_grid, levels=500, cmap=cmap, norm=norm)

    # Activation boundaries
    colors = ['cyan', 'orange', 'magenta', 'yellow', 'lime', 'red', 'purple', 'brown'][:L]
    rgba = [plt.matplotlib.colors.to_rgba(c) for c in colors]

    for l in range(L):
        for h in range(H):
            ax.contour(xx.cpu(), yy.cpu(),
                       preacts[:, :, l, h], levels=[0],
                       linewidths=1.2,
                       colors=rgba[l],
                       alpha=0.9,
                       zorder=1)

    # Predicted boundary
    ax.contour(xx.cpu().numpy(), yy.cpu().numpy(), sdf_grid, levels=[0], colors='black',
               linewidths=1.2, linestyles='--')
    
    
    # Draw surfaces if provided
    if surfaces is not None:
        # Flatten nested lists if needed
        flat_surfaces = []
        for item in surfaces:
            if isinstance(item, list):
                flat_surfaces.extend(item)
            else:
                flat_surfaces.append(item)
        
        for surf_idx, surf in enumerate(flat_surfaces):
            verts = surf.vertices
            
            if surf.closed:
                plot_verts = np.vstack([verts, verts[0:1]])  # Use [0:1] to keep 2D shape
            else:
                plot_verts = verts
            
            # Determine color based on whether this surface is highlighted
            if highlight_idx is not None and surf_idx == highlight_idx:
                edge_color = 'red'
                vertex_color = 'red'
                linewidth = 3.5
                edge_alpha = 1.0
            else:
                edge_color = 'black'
                vertex_color = 'black'
                linewidth = 2.5
                edge_alpha = 0.9
            
            # Plot with appropriate color
            ax.plot(plot_verts[:, 0], plot_verts[:, 1], 
                    color=edge_color, linewidth=linewidth, alpha=edge_alpha, 
                    linestyle='-', zorder=10)
            
            # Mark vertices
            ax.scatter(verts[:, 0], verts[:, 1], 
                      c=vertex_color, s=40, zorder=11, 
                      edgecolors='white', linewidths=1.5)
            
            # Add label with number of vertices
            if len(verts) > 0:
                midpoint = np.mean(verts, axis=0)
                ax.text(midpoint[0], midpoint[1], f'{len(verts)}v', 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 alpha=0.7, edgecolor='black'),
                       zorder=12)
        
        if title is None:
            title = f'SDF Heat-Map ({len(flat_surfaces)} surfaces)'
    else:
        if title is None:
            title = 'SDF Heat-Map'
    
    # Legend
    handles = [mpatches.Patch(color=colors[i], label=f'layer {i+1} boundary') for i in range(L)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax, format=FormatStrFormatter('%.3f'))
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('SDF value', fontsize=10)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=7)
    cbar.update_ticks()

    plt.tight_layout()
    plt.show()


def plot_cell_sdf2(
    net,
    cells=None,
    resolution: int = 300,
    device=None,
    highlight_idx=None,
    title=None
):
    """
    Plot SDF heatmap with activation boundaries and cells.
    
    Args:
        net: Neural network model (ReluMLP)
        cells: Optional list of Cell objects to overlay on the plot
        resolution: Grid resolution for the heatmap
        device: Device to run computations on
        highlight_idx: Optional index of cell to highlight in red
        title: Optional title for the plot
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if net.input_dim != 2 or net.output_dim != 1:
        print(f"Warning: plot_cell_sdf2 expects 2D input and 1D output, got input_dim={net.input_dim}, output_dim={net.output_dim}")
        return
    
    # Grid & network pass
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)

    with torch.no_grad():
        net = net.to(device)
        sdf_vals, preacts = net.eval_activations(grid_pts.to(device))

    L = net.num_layers
    H = net.hidden_dim

    sdf_grid = sdf_vals.view(resolution, resolution).cpu().numpy()
    preacts = preacts.view(resolution, resolution, L, H).cpu()

    fig, ax = plt.subplots(figsize=(10, 8))

    # SDF heat-map
    vmin, vmax = sdf_grid.min(), sdf_grid.max()
    if vmin >= 0:
        cmap = LinearSegmentedColormap.from_list('pos', ['white', 'red', 'darkred'])
        norm = plt.Normalize(vmin=0, vmax=vmax)
    elif vmax <= 0:
        cmap = LinearSegmentedColormap.from_list('neg', ['darkblue', 'blue', 'white'])
        norm = plt.Normalize(vmin=vmin, vmax=0)
    else:
        cmap = LinearSegmentedColormap.from_list('cent', ['blue', 'white', 'red'])
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax.contourf(xx.cpu().numpy(), yy.cpu().numpy(), sdf_grid, levels=500, cmap=cmap, norm=norm)

    # Activation boundaries
    colors = ['cyan', 'orange', 'magenta', 'yellow', 'lime', 'red', 'purple', 'brown'][:L]
    rgba = [plt.matplotlib.colors.to_rgba(c) for c in colors]

    for l in range(L):
        for h in range(H):
            ax.contour(xx.cpu(), yy.cpu(),
                       preacts[:, :, l, h], levels=[0],
                       linewidths=1.2,
                       colors=rgba[l],
                       alpha=0.9,
                       zorder=1)

    # Predicted boundary
    ax.contour(xx.cpu().numpy(), yy.cpu().numpy(), sdf_grid, levels=[0], colors='black',
               linewidths=1.2, linestyles='--')
    
    
    # Draw cells if provided
    if cells is not None:
        total_surfaces = 0
        for cell_idx, cell in enumerate(cells):
            # Determine color based on whether this cell is highlighted
            if highlight_idx is not None and cell_idx == highlight_idx:
                edge_color = 'red'
                vertex_color = 'red'
                linewidth = 3.5
                edge_alpha = 1.0
            else:
                edge_color = 'black'
                vertex_color = 'black'
                linewidth = 2.5
                edge_alpha = 0.9
            
            # Plot all surfaces in this cell
            for surf in cell.surfaces:
                verts = surf.vertices
                total_surfaces += 1
                
                if surf.closed:
                    plot_verts = np.vstack([verts, verts[0:1]])  # Use [0:1] to keep 2D shape
                else:
                    plot_verts = verts
                
                # Plot with appropriate color
                ax.plot(plot_verts[:, 0], plot_verts[:, 1], 
                        color=edge_color, linewidth=linewidth, alpha=edge_alpha, 
                        linestyle='-', zorder=10)
                
                # Mark vertices
                ax.scatter(verts[:, 0], verts[:, 1], 
                          c=vertex_color, s=40, zorder=11, 
                          edgecolors='white', linewidths=1.5)
                
                # Add label with number of vertices
                if len(verts) > 0:
                    midpoint = np.mean(verts, axis=0)
                    ax.text(midpoint[0], midpoint[1], f'{len(verts)}v', 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                     alpha=0.7, edgecolor='black'),
                           zorder=12)
        
        if title is None:
            title = f'SDF Heat-Map ({len(cells)} cells, {total_surfaces} surfaces)'
    else:
        if title is None:
            title = 'SDF Heat-Map'
    
    # Legend
    handles = [mpatches.Patch(color=colors[i], label=f'layer {i+1} boundary') for i in range(L)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax, format=FormatStrFormatter('%.3f'))
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('SDF value', fontsize=10)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=7)
    cbar.update_ticks()

    plt.tight_layout()
    plt.show()


def plot_polygons(polygons, figsize=(8, 8), title="Polygons"):
    """
    Plot polygons (assumes closed polygons without duplicated last vertex)
    
    Args:
        polygons: List of polygon vertex arrays or list of LineSegments objects
        figsize: Figure size tuple
        title: Plot title
    """
    plt.figure(figsize=figsize)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Handle both numpy arrays and LineSegments objects
    vertices_list = []
    for item in polygons:
        if hasattr(item, 'vertices'):
            # It's a LineSegments object
            vertices_list.append(item.vertices)
        elif isinstance(item, list):
            # It's a list (maybe nested list of LineSegments)
            for subitem in item:
                if hasattr(subitem, 'vertices'):
                    vertices_list.append(subitem.vertices)
                else:
                    vertices_list.append(subitem)
        else:
            # It's a numpy array
            vertices_list.append(item)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(vertices_list)))
    
    for i, polygon in enumerate(vertices_list):
        # Close polygon by adding first vertex at end
        closed_poly = np.vstack([polygon, polygon[0:1]])  # Use [0:1] to keep 2D shape
        plt.plot(closed_poly[:, 0], closed_poly[:, 1], 'k-', linewidth=2)
        plt.fill(closed_poly[:, 0], closed_poly[:, 1], color=colors[i], alpha=0.3)
        
        # Mark all vertices
        plt.scatter(polygon[:, 0], polygon[:, 1], c='red', s=20, zorder=5)
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
