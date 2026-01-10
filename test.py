from typing import List
import numpy as np

from ainr import plot_polygons
from ainr.ground_truth import generate_polygons

polygons = generate_polygons('1x32')

def find_hyperplane(polygons):
    

