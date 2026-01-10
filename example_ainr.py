import sys
import importlib
import numpy as np

def reload_ainr():
    """Reload all ainr modules and re-import classes/functions."""
    import ainr
    
    # Reload submodules first (in dependency order)
    if 'ainr.model' in sys.modules:
        importlib.reload(sys.modules['ainr.model'])
    if 'ainr.cell' in sys.modules:
        importlib.reload(sys.modules['ainr.cell'])
    if 'ainr.ground_truth' in sys.modules:
        importlib.reload(sys.modules['ainr.ground_truth'])
    if 'ainr.vis' in sys.modules:
        importlib.reload(sys.modules['ainr.vis'])
    
    # Reload main module
    importlib.reload(ainr)
    
    # Re-import into global namespace
    global ReluMLP, LineSegments, generate_polygons, plot_polygons, plot_cell_sdf
    from ainr import ReluMLP, LineSegments, generate_polygons, plot_polygons, plot_cell_sdf
    
    print("✓ ainr modules reloaded")


# Reload modules (run this after making changes to ainr)
reload_ainr()

# Create models
def create_model(input_dim=2, hidden_dim=8, num_layers=1, output_dim=1, skip_connections=True):
    return ReluMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        skip_connections=skip_connections
    )

model_skip = create_model(skip_connections=True)
model_no_skip = create_model(skip_connections=False)

# Generate and visualize polygons
polygons = generate_polygons('1x32', convex=True)
plot_polygons(polygons)

# Create line segments from polygons
line_segments = LineSegments.from_polygons(polygons, model_skip, closed=True)

def get_split(line_segments):
    segment_count = [
        len(seg.vertices) if seg.closed else len(seg.vertices)-2
        for seg in line_segments
    ]
    seg_idx = np.argmax(segment_count)
    seg = line_segments[seg_idx]
    vert_idx = len(seg.vertices)//2
    return seg.normals[vert_idx], seg.offsets[vert_idx]

normal, offset = get_split(line_segments)
