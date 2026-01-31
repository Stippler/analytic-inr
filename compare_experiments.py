#!/usr/bin/env python3
"""
Compare loss plots from multiple experiments.
Usage: python compare_experiments.py <experiment_dir>
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def compare_experiments(experiment_dir: str):
    """Compare loss plots from all experiments in a directory."""
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        print(f"Error: Directory {experiment_dir} does not exist")
        return
    
    # Find all subdirectories with loss_history.png
    exp_dirs = []
    for subdir in sorted(exp_path.iterdir()):
        if subdir.is_dir():
            loss_plot = subdir / 'loss_history.png'
            if loss_plot.exists():
                exp_dirs.append(subdir)
    
    if not exp_dirs:
        print(f"No experiments with loss plots found in {experiment_dir}")
        return
    
    print(f"Found {len(exp_dirs)} experiments with loss plots")
    
    # Create comparison grid
    n_experiments = len(exp_dirs)
    cols = 3
    rows = (n_experiments + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, exp_dir in enumerate(exp_dirs):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Load and display the loss plot
        loss_plot = exp_dir / 'loss_history.png'
        img = Image.open(loss_plot)
        ax.imshow(img)
        ax.set_title(exp_dir.name, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_experiments, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = exp_path / 'comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")
    plt.close()
    
    # Also try to create a summary plot if we can extract loss values
    try:
        create_summary_plots(exp_path, exp_dirs)
    except Exception as e:
        print(f"Could not create summary plots: {e}")

def create_summary_plots(exp_path: Path, exp_dirs: list):
    """Create summary plots comparing final losses across experiments."""
    # This would require parsing the loss values from saved data
    # For now, just print a message
    print("To create summary plots, save loss histories as JSON/CSV in each experiment")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_experiments.py <experiment_dir>")
        print("Example: python compare_experiments.py experiments/20260120_140000")
        sys.exit(1)
    
    compare_experiments(sys.argv[1])
