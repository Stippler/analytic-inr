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
            
            # Plot all surfaces in all subcells of this cell
            for subcell in cell.subcells:
                for surf in subcell.surfaces:
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


def plot_cell_sdf3(
    net,
    polygons=None,
    resolution: int = 300,
    device=None,
    highlight_idx=None,
    title=None
):
    """
    Plot SDF heatmap with activation boundaries and polygons.
    
    Args:
        net: Neural network model (ReluMLP)
        polygons: Optional list of polygon vertex arrays (numpy arrays) to overlay on the plot
        resolution: Grid resolution for the heatmap
        device: Device to run computations on
        highlight_idx: Optional index of polygon to highlight in red
        title: Optional title for the plot
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if net.input_dim != 2 or net.output_dim != 1:
        print(f"Warning: plot_cell_sdf3 expects 2D input and 1D output, got input_dim={net.input_dim}, output_dim={net.output_dim}")
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
    
    
    # Draw polygons if provided
    if polygons is not None:
        # Flatten nested lists if needed and ensure all are numpy arrays
        flat_polygons = []
        for item in polygons:
            if isinstance(item, list):
                flat_polygons.extend(item)
            else:
                flat_polygons.append(item)
        
        for poly_idx, polygon in enumerate(flat_polygons):
            # Ensure polygon is a numpy array
            verts = np.asarray(polygon)
            
            # Close polygon by adding first vertex at end
            plot_verts = np.vstack([verts, verts[0:1]])  # Use [0:1] to keep 2D shape
            
            # Determine color based on whether this polygon is highlighted
            if highlight_idx is not None and poly_idx == highlight_idx:
                edge_color = 'red'
                vertex_color = 'red'
                linewidth = 3.5
                edge_alpha = 1.0
            else:
                edge_color = 'green'
                vertex_color = 'green'
                linewidth = 2.5
                edge_alpha = 0.9
            
            # Plot polygon edges
            ax.plot(plot_verts[:, 0], plot_verts[:, 1], 
                    color=edge_color, linewidth=linewidth, alpha=edge_alpha, 
                    linestyle='-', zorder=10)
            
            # Mark vertices
            ax.scatter(verts[:, 0], verts[:, 1], 
                      c=vertex_color, s=40, zorder=11, 
                      edgecolors='white', linewidths=1.5)
        
        if title is None:
            title = f'SDF Heat-Map ({len(flat_polygons)} polygons)'
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


def plot_sdf_heatmap(net, splines=None, polygons=None, resolution=300, device=None, 
                     highlight_idx=None, title=None, interactive=False):
    """
    Plot SDF heatmap with activation boundaries, polygons, and spline lines.
    
    Args:
        net: Neural network model (ReluMLP)
        splines: Optional list of Spline objects to show as lines
        polygons: Optional list of polygon vertex arrays
        resolution: Grid resolution for the heatmap
        device: Device to run computations on
        highlight_idx: Optional index of polygon to highlight in red
        title: Optional title for the plot
        interactive: If True, use plotly for interactive plot (requires plotly)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if net.input_dim != 2 or net.output_dim != 1:
        print(f"Warning: expects 2D input and 1D output, got input_dim={net.input_dim}, output_dim={net.output_dim}")
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
    
    # Color mapping for splines
    def get_color_for_spline(spline):
        depth = spline.depth if hasattr(spline, 'depth') else 0
        label = spline.label if hasattr(spline, 'label') else ''
        
        if depth == 0:
            return '#8B00FF'
        elif depth == 1:
            quadrant_colors = {
                '++': '#FF0000',
                '+-': '#FF6B00',
                '--': '#FFD700',
                '-+': '#00C853'
            }
            for quad in ['++', '+-', '--', '-+']:
                if quad in label:
                    return quadrant_colors[quad]
            return '#FF0000'
        else:
            return '#00B8D4'
    
    if interactive:
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add SDF heatmap
            fig.add_trace(go.Heatmap(
                x=x.numpy(),
                y=y.numpy(),
                z=sdf_grid,
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(title='SDF value')
            ))
            
            # Add spline lines
            if splines is not None:
                for spline in splines:
                    p_start = spline.start_point.cpu().numpy()
                    p_end = spline.end_point.cpu().numpy()
                    color = get_color_for_spline(spline)
                    pc_type = spline.pc_type if hasattr(spline, 'pc_type') else 'pc1'
                    dash = 'solid' if pc_type == 'pc1' else 'dash'
                    
                    fig.add_trace(go.Scatter(
                        x=[p_start[0], p_end[0]],
                        y=[p_start[1], p_end[1]],
                        mode='lines',
                        line=dict(color=color, width=2, dash=dash),
                        name=spline.label if hasattr(spline, 'label') else 'spline',
                        showlegend=False
                    ))
            
            # Add polygons
            if polygons is not None:
                flat_polygons = []
                for item in polygons:
                    if isinstance(item, list):
                        flat_polygons.extend(item)
                    else:
                        flat_polygons.append(item)
                
                for poly_idx, polygon in enumerate(flat_polygons):
                    verts = np.asarray(polygon)
                    # Close polygon
                    plot_verts = np.vstack([verts, verts[0:1]])
                    
                    edge_color = 'red' if (highlight_idx is not None and poly_idx == highlight_idx) else 'green'
                    
                    fig.add_trace(go.Scatter(
                        x=plot_verts[:, 0],
                        y=plot_verts[:, 1],
                        mode='lines+markers',
                        line=dict(color=edge_color, width=2.5),
                        marker=dict(size=6, color='white', line=dict(color=edge_color, width=2)),
                        name=f'polygon {poly_idx}',
                        showlegend=False
                    ))
            
            fig.update_layout(
                title=title or 'SDF Heat-Map',
                xaxis_title='x',
                yaxis_title='y',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                width=800,
                height=800,
                hovermode='closest'
            )
            
            fig.show()
            return
            
        except ImportError:
            print("Plotly not installed. Falling back to matplotlib.")
    
    # Matplotlib version
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
    
    # Draw polygons if provided
    if polygons is not None:
        flat_polygons = []
        for item in polygons:
            if isinstance(item, list):
                flat_polygons.extend(item)
            else:
                flat_polygons.append(item)
        
        for poly_idx, polygon in enumerate(flat_polygons):
            verts = np.asarray(polygon)
            plot_verts = np.vstack([verts, verts[0:1]])
            
            if highlight_idx is not None and poly_idx == highlight_idx:
                edge_color = 'red'
                vertex_color = 'red'
                linewidth = 3.5
                edge_alpha = 1.0
            else:
                edge_color = 'green'
                vertex_color = 'green'
                linewidth = 2.5
                edge_alpha = 0.9
            
            ax.plot(plot_verts[:, 0], plot_verts[:, 1], 
                    color=edge_color, linewidth=linewidth, alpha=edge_alpha, 
                    linestyle='-', zorder=10)
            
            ax.scatter(verts[:, 0], verts[:, 1], 
                      c=vertex_color, s=40, zorder=11, 
                      edgecolors='white', linewidths=1.5)
    
    # Draw spline lines if provided
    if splines is not None:
        for spline in splines:
            p_start = spline.start_point.cpu().numpy()
            p_end = spline.end_point.cpu().numpy()
            color = get_color_for_spline(spline)
            pc_type = spline.pc_type if hasattr(spline, 'pc_type') else 'pc1'
            ls = '-' if pc_type == 'pc1' else '--'
            lw = 2.5 if (hasattr(spline, 'depth') and spline.depth == 0) else 1.5
            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                   color=color, ls=ls, lw=lw, alpha=0.8, zorder=5)
    
    # Title
    if title is None:
        if polygons is not None:
            flat_polygons = []
            for item in polygons:
                if isinstance(item, list):
                    flat_polygons.extend(item)
                else:
                    flat_polygons.append(item)
            title = f'SDF Heat-Map ({len(flat_polygons)} polygons)'
        else:
            title = 'SDF Heat-Map'
    
    # Legend
    handles = [mpatches.Patch(color=colors[i], label=f'layer {i+1} boundary') for i in range(L)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax, format=FormatStrFormatter('%.3f'))
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('SDF value', fontsize=10)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=7)
    cbar.update_ticks()

    plt.tight_layout()
    plt.show()


def plot_splines(splines, interactive=False):
    """
    Plot splines with PC1 and PC2 arranged per row.
    
    Args:
        splines: List of Spline objects
        interactive: If True, use plotly for interactive plots
    """
    if splines is None or len(splines) == 0:
        return
    
    # Color mapping function
    def get_color_for_spline(spline):
        depth = spline.depth if hasattr(spline, 'depth') else 0
        label = spline.label if hasattr(spline, 'label') else ''
        
        if depth == 0:
            return '#8B00FF'
        elif depth == 1:
            quadrant_colors = {
                '++': '#FF0000',
                '+-': '#FF6B00',
                '--': '#FFD700',
                '-+': '#00C853'
            }
            for quad in ['++', '+-', '--', '-+']:
                if quad in label:
                    return quadrant_colors[quad]
            return '#FF0000'
        else:
            return '#00B8D4'
    
    # Organize splines by pairs (PC1, PC2)
    spline_pairs = []
    pc1_splines = [s for s in splines if hasattr(s, 'pc_type') and s.pc_type == 'pc1']
    pc2_splines = [s for s in splines if hasattr(s, 'pc_type') and s.pc_type == 'pc2']
    
    # Match PC1 and PC2 by their base label (remove /pc1 or /pc2)
    pc1_dict = {}
    for s in pc1_splines:
        base_label = s.label.replace('/pc1', '').replace('pc1', '').strip()
        pc1_dict[base_label] = s
    
    pc2_dict = {}
    for s in pc2_splines:
        base_label = s.label.replace('/pc2', '').replace('pc2', '').strip()
        pc2_dict[base_label] = s
    
    # Create pairs
    all_labels = sorted(set(list(pc1_dict.keys()) + list(pc2_dict.keys())))
    for label in all_labels:
        pair = []
        if label in pc1_dict:
            pair.append(pc1_dict[label])
        if label in pc2_dict:
            pair.append(pc2_dict[label])
        if pair:
            spline_pairs.append((label, pair))
    
    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            n_rows = len(spline_pairs)
            n_cols = 2  # PC1 and PC2
            
            subplot_titles = []
            for label, pair in spline_pairs:
                for s in pair:
                    subplot_titles.append(s.label if hasattr(s, 'label') else '')
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=subplot_titles,
                vertical_spacing=0.08,
                horizontal_spacing=0.10
            )
            
            row_idx = 1
            for label, pair in spline_pairs:
                for col_idx, spline in enumerate(pair, start=1):
                    color = get_color_for_spline(spline)
                    
                    # Ground truth
                    gt_t = spline.gt_knots.cpu().numpy()
                    gt_sdf = spline.gt_values.cpu().numpy()
                    
                    fig.add_trace(
                        go.Scatter(x=gt_t, y=gt_sdf, mode='lines+markers',
                                  name=f'GT {spline.label}',
                                  line=dict(color=color, width=2),
                                  marker=dict(size=6, color='white', 
                                            line=dict(color=color, width=2)),
                                  showlegend=(row_idx == 1 and col_idx == 1)),
                        row=row_idx, col=col_idx
                    )
                    
                    # Predicted
                    if hasattr(spline, 'pred_knots') and spline.pred_knots is not None and spline.pred_values is not None:
                        pred_t = spline.pred_knots.detach().cpu().numpy()
                        pred_sdf = spline.pred_values.detach().cpu().numpy()
                        
                        fig.add_trace(
                            go.Scatter(x=pred_t, y=pred_sdf, mode='lines+markers',
                                      name=f'Pred',
                                      line=dict(color='#00CED1', width=2, dash='dash'),
                                      marker=dict(size=6, symbol='square', color='white',
                                                line=dict(color='#00CED1', width=2)),
                                      showlegend=(row_idx == 1 and col_idx == 1)),
                            row=row_idx, col=col_idx
                        )
                    
                    # Zero line
                    fig.add_hline(y=0, line_dash="dot", line_color="black", 
                                 opacity=0.3, row=row_idx, col=col_idx)
                    
                    # Update axes
                    fig.update_xaxes(title_text="t", row=row_idx, col=col_idx)
                    fig.update_yaxes(title_text="SDF", row=row_idx, col=col_idx)
                
                row_idx += 1
            
            fig.update_layout(
                height=300 * n_rows,
                title_text="Spline Predictions (PC1 | PC2 per row)",
                showlegend=True,
                hovermode='closest'
            )
            
            fig.show()
            return
            
        except ImportError:
            print("Plotly not installed. Falling back to matplotlib.")
    
    # Matplotlib version
    n_rows = len(spline_pairs)
    n_cols = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (label, pair) in enumerate(spline_pairs):
        for col_idx, spline in enumerate(pair):
            ax = axes[row_idx, col_idx]
            color = get_color_for_spline(spline)
            
            # Ground truth
            gt_t = spline.gt_knots.cpu().numpy()
            gt_sdf = spline.gt_values.cpu().numpy()
            
            ax.plot(gt_t, gt_sdf, 'o-', color=color, 
                   lw=2, markersize=4, alpha=1.0, 
                   markerfacecolor='white', markeredgewidth=1.5, 
                   markeredgecolor=color, label='GT')
            
            # Predicted
            if hasattr(spline, 'pred_knots') and spline.pred_knots is not None and spline.pred_values is not None:
                pred_t = spline.pred_knots.detach().cpu().numpy()
                pred_sdf = spline.pred_values.detach().cpu().numpy()
                
                ax.plot(pred_t, pred_sdf, 's-', color='#00CED1', 
                       lw=2, markersize=4, alpha=0.8, 
                       markerfacecolor='white', markeredgewidth=1.5, 
                       markeredgecolor='#00CED1', label='Pred')
            
            ax.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=1)
            ax.grid(alpha=0.2, linestyle='--')
            ax.set_title(spline.label if hasattr(spline, 'label') else '', 
                        fontsize=9, fontweight='bold')
            ax.set_xlabel('t', fontsize=8)
            ax.set_ylabel('SDF', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc='best')
        
        # Hide unused subplots in the row
        for col_idx in range(len(pair), n_cols):
            axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_cell_sdf3(
    net,
    splines=None,
    polygons=None,
    resolution: int = 300,
    device=None,
    highlight_idx=None,
    title=None,
    interactive=False,
    separate_plots=True
):
    """
    Legacy combined plot: SDF heatmap with activation boundaries, polygons, and spline plots.
    
    Note: Consider using plot_sdf_heatmap() and plot_splines() separately for cleaner output.
    
    Args:
        net: Neural network model (ReluMLP)
        splines: Optional list of Spline objects to plot on the right
        polygons: Optional list of polygon vertex arrays (numpy arrays) to overlay on the left plot
        resolution: Grid resolution for the heatmap
        device: Device to run computations on
        highlight_idx: Optional index of polygon to highlight in red
        title: Optional title for the plot
        interactive: If True, use plotly for interactive plots in Jupyter (requires plotly)
        separate_plots: If True, call plot_splines separately after
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if net.input_dim != 2 or net.output_dim != 1:
        print(f"Warning: plot_cell_sdf3 expects 2D input and 1D output, got input_dim={net.input_dim}, output_dim={net.output_dim}")
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

    # If splines are provided, create multi-panel plot
    if splines is not None and len(splines) > 0 and not separate_plots:
        # Separate PC1 and PC2 splines
        pc1_splines = [s for s in splines if hasattr(s, 'pc_type') and s.pc_type == 'pc1']
        pc2_splines = [s for s in splines if hasattr(s, 'pc_type') and s.pc_type == 'pc2']
        n_plots = max(len(pc1_splines), len(pc2_splines))
        
        # Create figure with grid layout
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(18, 6))
        width_ratios = [2] + [0.6] * n_plots
        gs = gridspec.GridSpec(3, n_plots + 1, figure=fig, 
                              height_ratios=[10, 10, 1], 
                              width_ratios=width_ratios)
        
        # Left: SDF heatmap
        ax = fig.add_subplot(gs[0:2, 0])
    else:
        # Single plot mode
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
    
    # Color mapping function for splines
    def get_color_for_spline(spline):
        depth = spline.depth if hasattr(spline, 'depth') else 0
        label = spline.label if hasattr(spline, 'label') else ''
        
        if depth == 0:
            return '#8B00FF'
        elif depth == 1:
            quadrant_colors = {
                '++': '#FF0000',
                '+-': '#FF6B00',
                '--': '#FFD700',
                '-+': '#00C853'
            }
            for quad in ['++', '+-', '--', '-+']:
                if quad in label:
                    return quadrant_colors[quad]
            return '#FF0000'
        else:
            return '#00B8D4'
    
    # Draw polygons if provided
    if polygons is not None:
        flat_polygons = []
        for item in polygons:
            if isinstance(item, list):
                flat_polygons.extend(item)
            else:
                flat_polygons.append(item)
        
        for poly_idx, polygon in enumerate(flat_polygons):
            verts = np.asarray(polygon)
            plot_verts = np.vstack([verts, verts[0:1]])
            
            if highlight_idx is not None and poly_idx == highlight_idx:
                edge_color = 'red'
                vertex_color = 'red'
                linewidth = 3.5
                edge_alpha = 1.0
            else:
                edge_color = 'green'
                vertex_color = 'green'
                linewidth = 2.5
                edge_alpha = 0.9
            
            ax.plot(plot_verts[:, 0], plot_verts[:, 1], 
                    color=edge_color, linewidth=linewidth, alpha=edge_alpha, 
                    linestyle='-', zorder=10)
            
            ax.scatter(verts[:, 0], verts[:, 1], 
                      c=vertex_color, s=40, zorder=11, 
                      edgecolors='white', linewidths=1.5)
    
    # Draw spline lines if provided
    if splines is not None:
        for spline in splines:
            p_start = spline.start_point.cpu().numpy()
            p_end = spline.end_point.cpu().numpy()
            color = get_color_for_spline(spline)
            pc_type = spline.pc_type if hasattr(spline, 'pc_type') else 'pc1'
            ls = '-' if pc_type == 'pc1' else '--'
            lw = 2.5 if (hasattr(spline, 'depth') and spline.depth == 0) else 1.5
            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                   color=color, ls=ls, lw=lw, alpha=0.8, zorder=5)
    
    # Title for left panel
    if title is None:
        if polygons is not None:
            flat_polygons = []
            for item in polygons:
                if isinstance(item, list):
                    flat_polygons.extend(item)
                else:
                    flat_polygons.append(item)
            title = f'SDF Heat-Map ({len(flat_polygons)} polygons)'
        else:
            title = 'SDF Heat-Map'
    
    # Legend
    handles = [mpatches.Patch(color=colors[i], label=f'layer {i+1} boundary') for i in range(L)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.3)
    
    # Add colorbar for SDF values
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax, format=FormatStrFormatter('%.3f'))
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('SDF value', fontsize=10)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=7)
    cbar.update_ticks()
    
    # Plot splines on the right side if provided (and not using separate_plots)
    if splines is not None and len(splines) > 0 and not separate_plots:
        # Top row: PC1
        for idx, spline in enumerate(pc1_splines):
            ax_pc1 = fig.add_subplot(gs[0, idx + 1])
            color = get_color_for_spline(spline)
            
            # Plot ground truth SDF (thin, transparent)
            gt_t = spline.gt_knots.cpu().numpy()
            gt_sdf = spline.gt_values.cpu().numpy()
            ax_pc1.plot(gt_t, gt_sdf, color='gray', ls='-', lw=1, alpha=0.3, label='GT')
            
            # Plot GT knots (circles)
            ax_pc1.plot(gt_t, gt_sdf, 'o-', color=color, 
                    lw=2, markersize=4, alpha=1.0, markerfacecolor='white', 
                    markeredgewidth=1.5, markeredgecolor=color)
            
            # Plot predicted spline if available
            if hasattr(spline, 'pred_knots') and spline.pred_knots is not None and spline.pred_values is not None:
                pred_t = spline.pred_knots.detach().cpu().numpy()
                pred_sdf = spline.pred_values.detach().cpu().numpy()
                ax_pc1.plot(pred_t, pred_sdf, 's-', color='#00CED1', 
                        lw=2, markersize=4, alpha=0.8, markerfacecolor='white', 
                        markeredgewidth=1.5, markeredgecolor='#00CED1', label='Pred')
            
            ax_pc1.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=1)
            ax_pc1.grid(alpha=0.2, linestyle='--')
            
            short_label = spline.label.split(': ')[1] if ': ' in (spline.label if hasattr(spline, 'label') else '') else (spline.label if hasattr(spline, 'label') else '')
            ax_pc1.set_title(f"{short_label}\n{len(gt_t)} knots", 
                        fontsize=8, fontweight='bold')
            
            if idx == 0:
                ax_pc1.set_ylabel('SDF', fontsize=8)
            ax_pc1.tick_params(labelsize=7)
        
        # Bottom row: PC2
        for idx, spline in enumerate(pc2_splines):
            ax_pc2 = fig.add_subplot(gs[1, idx + 1])
            color = get_color_for_spline(spline)
            
            # Plot ground truth SDF (thin, transparent)
            gt_t = spline.gt_knots.cpu().numpy()
            gt_sdf = spline.gt_values.cpu().numpy()
            ax_pc2.plot(gt_t, gt_sdf, color='gray', ls='--', lw=1, alpha=0.3, label='GT')
            
            # Plot GT knots (circles)
            ax_pc2.plot(gt_t, gt_sdf, 'o--', color=color, 
                    lw=2, markersize=4, alpha=1.0, markerfacecolor='white', 
                    markeredgewidth=1.5, markeredgecolor=color)
            
            # Plot predicted spline if available
            if hasattr(spline, 'pred_knots') and spline.pred_knots is not None and spline.pred_values is not None:
                pred_t = spline.pred_knots.detach().cpu().numpy()
                pred_sdf = spline.pred_values.detach().cpu().numpy()
                ax_pc2.plot(pred_t, pred_sdf, 's--', color='#00CED1', 
                        lw=2, markersize=4, alpha=0.8, markerfacecolor='white', 
                        markeredgewidth=1.5, markeredgecolor='#00CED1', label='Pred')
            
            ax_pc2.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=1)
            ax_pc2.grid(alpha=0.2, linestyle='--')
            
            short_label = spline.label.split(': ')[1] if ': ' in (spline.label if hasattr(spline, 'label') else '') else (spline.label if hasattr(spline, 'label') else '')
            ax_pc2.set_title(f"{short_label}\n{len(gt_t)} knots", 
                        fontsize=8, fontweight='bold')
            ax_pc2.set_xlabel('t', fontsize=8)
            
            if idx == 0:
                ax_pc2.set_ylabel('SDF', fontsize=8)
            ax_pc2.tick_params(labelsize=7)
        
        # Bottom: Legend
        ax_legend = fig.add_subplot(gs[2, :])
        ax_legend.axis('off')
        
        from matplotlib.lines import Line2D
        legend_handles = []
        
        # D0: Purple
        legend_handles.append(Line2D([0], [0], color='#8B00FF', lw=2, ls='-', 
                                     label='D0 PC1'))
        legend_handles.append(Line2D([0], [0], color='#8B00FF', lw=2, ls='--', 
                                     label='D0 PC2'))
        
        # D1 quadrants with unique colors
        d1_colors = {
            '++': ('#FF0000', 'Red'),
            '+-': ('#FF6B00', 'Orange'),
            '--': ('#FFD700', 'Gold'),
            '-+': ('#00C853', 'Green')
        }
        
        for quad, (color, name) in d1_colors.items():
            legend_handles.append(Line2D([0], [0], color=color, lw=2, ls='-', 
                                         label=f'D1 {quad} PC1'))
            legend_handles.append(Line2D([0], [0], color=color, lw=2, ls='--', 
                                         label=f'D1 {quad} PC2'))
        
        # Add GT/Pred explanation
        legend_handles.append(Line2D([0], [0], color='gray', lw=2, marker='o', 
                                     markerfacecolor='white', markeredgewidth=1.5,
                                     markeredgecolor='gray', label='GT Knots'))
        legend_handles.append(Line2D([0], [0], color='#00CED1', lw=2, marker='s', 
                                     markerfacecolor='white', markeredgewidth=1.5,
                                     markeredgecolor='#00CED1', label='Pred Knots'))
        
        ax_legend.legend(handles=legend_handles, loc='center', ncol=6, fontsize=8, 
                        frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()
    
    # If separate_plots is True and we have splines, plot them separately
    if separate_plots and splines is not None and len(splines) > 0:
        plot_splines(splines, interactive=interactive)


def plot_sdf_with_knots(all_knots_data, polygons):
    """
    Plot SDF curves with knot overlays and error information.
    
    Args:
        all_knots_data: List of dicts with keys: 't', 'sdf', 'knot_t', 'knot_sdf', 
                       'max_error', 'label', 'line', 'depth', 'pc_type'
        polygons: List of polygon vertex arrays
    """
    # Separate PC1 and PC2
    pc1_data = [d for d in all_knots_data if d['pc_type'] == 'pc1']
    pc2_data = [d for d in all_knots_data if d['pc_type'] == 'pc2']
    
    # Color mapping function
    def get_color_for_data(data):
        depth = data['depth']
        label = data['label']
        
        # D0: Purple
        if depth == 0:
            return '#8B00FF'
        
        # D1: Unique color per quadrant
        elif depth == 1:
            quadrant_colors = {
                '++': '#FF0000',  # Red
                '+-': '#FF6B00',  # Orange  
                '--': '#FFD700',  # Gold
                '-+': '#00C853'   # Green
            }
            # Extract quadrant from label (e.g., "D1: ++/pc1" -> "++")
            for quad in ['++', '+-', '--', '-+']:
                if quad in label:
                    return quadrant_colors[quad]
            return '#FF0000'  # Fallback
        
        # D2+: Cyan/Blue
        else:
            return '#00B8D4'
    
    n_plots = max(len(pc1_data), len(pc2_data))
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    import matplotlib.gridspec as gridspec
    width_ratios = [2] + [0.6] * n_plots
    gs = gridspec.GridSpec(3, n_plots + 1, figure=fig, 
                          height_ratios=[10, 10, 1], 
                          width_ratios=width_ratios)
    
    # Left: 2D visualization
    ax_2d = fig.add_subplot(gs[0:2, 0])
    poly_colors = plt.cm.tab10(np.linspace(0, 1, len(polygons)))
    for j, poly in enumerate(polygons):
        poly_closed = np.vstack([poly, poly[0]])
        ax_2d.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', linewidth=2, alpha=0.7)
        ax_2d.fill(poly_closed[:, 0], poly_closed[:, 1], color=poly_colors[j], alpha=0.15)
    
    for data in all_knots_data:
        p_start, p_end = data['line']
        color = get_color_for_data(data)
        ls = '-' if data['pc_type'] == 'pc1' else '--'
        lw = 2.5 if data['depth'] == 0 else 1.5
        ax_2d.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                   color=color, ls=ls, lw=lw, alpha=0.8)
    
    ax_2d.set_xlim(-1, 1)
    ax_2d.set_ylim(-1, 1)
    ax_2d.set_aspect('equal')
    ax_2d.grid(alpha=0.3)
    ax_2d.set_title('All PCA Lines', fontsize=12, fontweight='bold')
    
    # Right: Individual plots with knots
    # Top row: PC1
    for idx, data in enumerate(pc1_data):
        ax = fig.add_subplot(gs[0, idx + 1])
        color = get_color_for_data(data)
        
        # Plot dense SDF (thin, transparent)
        ax.plot(data['t'], data['sdf'], color=color, ls='-', lw=1, alpha=0.3)
        
        # Plot knots (thick, opaque)
        ax.plot(data['knot_t'], data['knot_sdf'], 'o-', color=color, 
                lw=2, markersize=4, alpha=1.0, markerfacecolor='white', 
                markeredgewidth=1.5, markeredgecolor=color)
        
        ax.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=1)
        ax.grid(alpha=0.2, linestyle='--')
        
        short_label = data['label'].split(': ')[1] if ': ' in data['label'] else data['label']
        ax.set_title(f"{short_label}\n{len(data['knot_t'])} knots, err={data['max_error']:.4f}", 
                    fontsize=8, fontweight='bold')
        
        if idx == 0:
            ax.set_ylabel('SDF', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Bottom row: PC2
    for idx, data in enumerate(pc2_data):
        ax = fig.add_subplot(gs[1, idx + 1])
        color = get_color_for_data(data)
        
        # Plot dense SDF (thin, transparent)
        ax.plot(data['t'], data['sdf'], color=color, ls='--', lw=1, alpha=0.3)
        
        # Plot knots (thick, opaque)
        ax.plot(data['knot_t'], data['knot_sdf'], 'o--', color=color, 
                lw=2, markersize=4, alpha=1.0, markerfacecolor='white', 
                markeredgewidth=1.5, markeredgecolor=color)
        
        ax.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=1)
        ax.grid(alpha=0.2, linestyle='--')
        
        short_label = data['label'].split(': ')[1] if ': ' in data['label'] else data['label']
        ax.set_title(f"{short_label}\n{len(data['knot_t'])} knots, err={data['max_error']:.4f}", 
                    fontsize=8, fontweight='bold')
        ax.set_xlabel('t', fontsize=8)
        
        if idx == 0:
            ax.set_ylabel('SDF', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Bottom: Legend
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    
    from matplotlib.lines import Line2D
    legend_handles = []
    
    # D0: Purple
    legend_handles.append(Line2D([0], [0], color='#8B00FF', lw=2, ls='-', 
                                 label='D0 PC1'))
    legend_handles.append(Line2D([0], [0], color='#8B00FF', lw=2, ls='--', 
                                 label='D0 PC2'))
    
    # D1 quadrants with unique colors
    d1_colors = {
        '++': ('#FF0000', 'Red'),
        '+-': ('#FF6B00', 'Orange'),
        '--': ('#FFD700', 'Gold'),
        '-+': ('#00C853', 'Green')
    }
    
    for quad, (color, name) in d1_colors.items():
        legend_handles.append(Line2D([0], [0], color=color, lw=2, ls='-', 
                                     label=f'D1 {quad} PC1'))
        legend_handles.append(Line2D([0], [0], color=color, lw=2, ls='--', 
                                     label=f'D1 {quad} PC2'))
    
    # Add knot explanation
    legend_handles.append(Line2D([0], [0], color='gray', lw=2, marker='o', 
                                 markerfacecolor='white', markeredgewidth=1.5,
                                 label='Knots (simplified)'))
    
    ax_legend.legend(handles=legend_handles, loc='center', ncol=5, fontsize=8, 
                    frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.show()


def plot_splines_separately(splines, interactive=False):
    """
    Plot splines in a separate figure for extensibility (legacy 5-column grid version).
    
    Note: Consider using plot_splines() for cleaner PC1|PC2 per-row layout.
    
    Args:
        splines: List of Spline objects
        interactive: If True, use plotly for interactive plots in Jupyter
    """
    if splines is None or len(splines) == 0:
        return
    
    # Color mapping function
    def get_color_for_spline(spline):
        depth = spline.depth if hasattr(spline, 'depth') else 0
        label = spline.label if hasattr(spline, 'label') else ''
        
        if depth == 0:
            return '#8B00FF'
        elif depth == 1:
            quadrant_colors = {
                '++': '#FF0000',
                '+-': '#FF6B00',
                '--': '#FFD700',
                '-+': '#00C853'
            }
            for quad in ['++', '+-', '--', '-+']:
                if quad in label:
                    return quadrant_colors[quad]
            return '#FF0000'
        else:
            return '#00B8D4'
    
    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Calculate grid layout
            n_splines = len(splines)
            n_cols = min(5, n_splines)  # Max 5 columns
            n_rows = int(np.ceil(n_splines / n_cols))
            
            # Create subplots
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[s.label if hasattr(s, 'label') else '' for s in splines],
                vertical_spacing=0.12,
                horizontal_spacing=0.05
            )
            
            for idx, spline in enumerate(splines):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                color = get_color_for_spline(spline)
                
                # Ground truth
                gt_t = spline.gt_knots.cpu().numpy()
                gt_sdf = spline.gt_values.cpu().numpy()
                
                fig.add_trace(
                    go.Scatter(x=gt_t, y=gt_sdf, mode='lines+markers',
                              name=f'GT {spline.label if hasattr(spline, "label") else ""}',
                              line=dict(color=color, width=2),
                              marker=dict(size=6, color='white', 
                                        line=dict(color=color, width=2)),
                              showlegend=(idx == 0)),
                    row=row, col=col
                )
                
                # Predicted
                if hasattr(spline, 'pred_knots') and spline.pred_knots is not None and spline.pred_values is not None:
                    pred_t = spline.pred_knots.detach().cpu().numpy()
                    pred_sdf = spline.pred_values.detach().cpu().numpy()
                    
                    fig.add_trace(
                        go.Scatter(x=pred_t, y=pred_sdf, mode='lines+markers',
                                  name=f'Pred',
                                  line=dict(color='#00CED1', width=2, dash='dash'),
                                  marker=dict(size=6, symbol='square', color='white',
                                            line=dict(color='#00CED1', width=2)),
                                  showlegend=(idx == 0)),
                        row=row, col=col
                    )
                
                # Zero line
                fig.add_hline(y=0, line_dash="dot", line_color="black", 
                             opacity=0.3, row=row, col=col)
                
                # Update axes
                fig.update_xaxes(title_text="t", row=row, col=col)
                fig.update_yaxes(title_text="SDF", row=row, col=col)
            
            fig.update_layout(
                height=300 * n_rows,
                title_text="Spline Predictions",
                showlegend=True,
                hovermode='closest'
            )
            
            fig.show()
            return
            
        except ImportError:
            print("Plotly not installed. Falling back to matplotlib.")
    
    # Matplotlib version
    n_splines = len(splines)
    n_cols = min(5, n_splines)
    n_rows = int(np.ceil(n_splines / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_splines == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, spline in enumerate(splines):
        ax = axes[idx]
        color = get_color_for_spline(spline)
        
        # Ground truth
        gt_t = spline.gt_knots.cpu().numpy()
        gt_sdf = spline.gt_values.cpu().numpy()
        
        ax.plot(gt_t, gt_sdf, 'o-', color=color, 
               lw=2, markersize=4, alpha=1.0, 
               markerfacecolor='white', markeredgewidth=1.5, 
               markeredgecolor=color, label='GT')
        
        # Predicted
        if hasattr(spline, 'pred_knots') and spline.pred_knots is not None and spline.pred_values is not None:
            pred_t = spline.pred_knots.detach().cpu().numpy()
            pred_sdf = spline.pred_values.detach().cpu().numpy()
            
            ax.plot(pred_t, pred_sdf, 's-', color='#00CED1', 
                   lw=2, markersize=4, alpha=0.8, 
                   markerfacecolor='white', markeredgewidth=1.5, 
                   markeredgecolor='#00CED1', label='Pred')
        
        ax.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=1)
        ax.grid(alpha=0.2, linestyle='--')
        ax.set_title(spline.label if hasattr(spline, 'label') else '', fontsize=9, fontweight='bold')
        ax.set_xlabel('t', fontsize=8)
        ax.set_ylabel('SDF', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    
    # Hide unused subplots
    for idx in range(n_splines, len(axes)):
        axes[idx].axis('off')
    
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
