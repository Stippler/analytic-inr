"""
AINR (Analytic Implicit Neural Representations)
A module for working with ReLU MLPs and their analytic properties.
"""

from .model import ReluMLP
from .cell import Surface2D, Cell
from .vis import plot_cell_sdf, plot_polygons
from .ground_truth import generate_polygons, generate_circle, generate_rectangle, generate_star

__all__ = [
    'ReluMLP',
    'LineSegments',
    'plot_cell_sdf',
    'plot_polygons',
    'generate_polygons',
    'generate_circle',
    'generate_rectangle',
    'generate_star',
]

