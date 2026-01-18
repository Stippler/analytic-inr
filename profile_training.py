#!/usr/bin/env python3
"""
Example script demonstrating how to use the PyTorch profiler with train_fast.py

This script shows different profiling configurations and how to interpret results.
"""

import torch
from pathlib import Path
from neural_spline.model import ReluMLP
from neural_spline.train_fast import train_model_fast
from neural_spline.types import Splines

def example_profiling():
    """
    Example: Profile training with various configurations
    """
    
    # Setup: Create dummy data (replace with your actual data loading)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy splines for demonstration
    n_splines = 1000
    input_dim = 2
    
    splines = Splines(
        start_points=torch.randn(n_splines, input_dim, device=device),
        end_points=torch.randn(n_splines, input_dim, device=device),
        knots=torch.rand(n_splines, 10, device=device).sort(dim=1)[0],
        values=torch.randn(n_splines, 10, device=device)
    )
    
    # Create model
    mlp = ReluMLP(
        input_dim=input_dim,
        hidden_dims=[32, 32],
        output_dim=1
    )
    
    print("="*80)
    print("PROFILING EXAMPLE 1: Quick Profile (5 batches)")
    print("="*80)
    
    # Example 1: Quick profiling - profile just a few batches
    # This is good for getting a quick overview
    loss_history = train_model_fast(
        mlp=mlp,
        splines=splines,
        epochs=1,
        batch_size=64,
        lr=0.01,
        profile_enabled=True,
        profile_wait=1,      # Skip first batch (may have initialization overhead)
        profile_warmup=1,    # Warmup for 1 batch
        profile_active=3,    # Profile 3 batches
        profile_repeat=1     # Do this once
    )
    
    print("\n" + "="*80)
    print("PROFILING EXAMPLE 2: Longer Profile (1 full epoch)")
    print("="*80)
    
    # Example 2: Profile a full epoch
    # This gives more comprehensive statistics
    loss_history = train_model_fast(
        mlp=mlp,
        splines=splines,
        epochs=1,
        batch_size=64,
        lr=0.01,
        profile_enabled=True,
        profile_wait=0,      # Start immediately
        profile_warmup=2,    # Warmup for 2 batches
        profile_active=10,   # Profile 10 batches
        profile_repeat=1     # Do this once
    )
    
    print("\n" + "="*80)
    print("PROFILING EXAMPLE 3: Multiple Cycles")
    print("="*80)
    
    # Example 3: Profile multiple cycles across epochs
    # This helps identify variance between batches/epochs
    loss_history = train_model_fast(
        mlp=mlp,
        splines=splines,
        epochs=3,
        batch_size=64,
        lr=0.01,
        profile_enabled=True,
        profile_wait=2,      # Wait 2 batches
        profile_warmup=1,    # Warmup 1 batch
        profile_active=3,    # Profile 3 batches
        profile_repeat=2     # Repeat this cycle 2 times
    )


def view_profiler_results():
    """
    Instructions for viewing profiler results
    """
    print("\n" + "="*80)
    print("HOW TO VIEW PROFILER RESULTS")
    print("="*80)
    
    print("\n1. COMMAND LINE TABLE (already printed above)")
    print("   - Shows key operations sorted by time")
    print("   - Look for 'cuda_time_total' (GPU) or 'cpu_time_total' (CPU)")
    print("   - Identify bottlenecks in your code")
    
    print("\n2. TENSORBOARD (Interactive Web UI - RECOMMENDED)")
    print("   Run this command in your terminal:")
    print("   $ tensorboard --logdir=./profiler_logs")
    print("\n   Then open: http://localhost:6006")
    print("\n   In TensorBoard, go to the 'PyTorch Profiler' tab")
    print("   You'll see:")
    print("   - Timeline view: See GPU/CPU activity over time")
    print("   - Operator view: Breakdown of operation times")
    print("   - Kernel view: Detailed CUDA kernel information")
    print("   - Memory view: Memory allocation patterns")
    
    print("\n3. CHROME TRACE (Advanced)")
    print("   The profiler also saves .json trace files in ./profiler_logs")
    print("   Open chrome://tracing in Chrome browser")
    print("   Load the .json file for detailed timeline visualization")
    
    print("\n" + "="*80)
    print("KEY METRICS TO LOOK FOR")
    print("="*80)
    print("- forward_pass: Time spent in model forward pass")
    print("- backward_pass: Time spent computing gradients")
    print("- loss_computation: Time spent calculating loss")
    print("- optimizer_step: Time spent updating weights")
    print("- data_slicing: Time spent preparing batch data")
    
    print("\nLook for:")
    print("✓ Which operation takes the most time?")
    print("✓ Are you GPU-bound or CPU-bound?")
    print("✓ Any unexpected memory allocations?")
    print("✓ Are kernels efficiently batched?")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("PyTorch Profiler Tutorial for train_fast.py\n")
    
    # Run examples
    example_profiling()
    
    # Show how to view results
    view_profiler_results()
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE!")
    print("="*80)
    print("\nQuick start:")
    print("1. Run: tensorboard --logdir=./profiler_logs")
    print("2. Open: http://localhost:6006")
    print("3. Click 'PyTorch Profiler' tab")
    print("="*80 + "\n")
