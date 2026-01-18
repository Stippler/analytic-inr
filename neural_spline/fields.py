"""
Legacy module - functions moved to spline.py.

This module is kept for backward compatibility but the main SDF computation
has been refactored to use sphere tracing in spline.py.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from .types import PCAComponent

# All functions have been moved to spline.py
# Import from there if needed for compatibility
from .spline import compute_sdf

__all__ = ['compute_sdf']

