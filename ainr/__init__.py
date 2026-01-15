"""
AINR (Analytic Implicit Neural Representations)
A module for working with ReLU MLPs and their analytic properties.
"""

from .model import ReluMLP
from .cell import Surface2D, Cell
from .vis import (plot_cell_sdf, plot_cell_sdf2, plot_cell_sdf3, plot_polygons,
                  plot_sdf_heatmap, plot_splines, plot_sdf_with_knots, plot_splines_separately)
from .ground_truth import generate_polygons, generate_circle, generate_rectangle, generate_star

__all__ = [
    'ReluMLP',
    'Surface2D',
    'Cell',
    'plot_cell_sdf',
    'plot_cell_sdf2',
    'plot_cell_sdf3',
    'plot_sdf_heatmap',
    'plot_splines',
    'plot_sdf_with_knots',
    'plot_splines_separately',
    'plot_polygons',
    'generate_polygons',
    'generate_circle',
    'generate_rectangle',
    'generate_star',
]

