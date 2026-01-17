"""
Neural network model definition with BFloat16 support.

This module provides the ReluMLP model with:
- Optional BFloat16 mixed precision
- Layer introspection helpers for analytical training
- Compatibility with torch.compile
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ReluMLP(nn.Module):
    """
    Multi-layer perceptron with ReLU activations.
    
    Optimized for analytical training with support for:
    - Mixed precision (BFloat16)
    - Layer introspection
    - torch.compile compatibility
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        skip_connections: bool = False,
        use_bfloat16: bool = False
    ):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Input dimension (2 or 3 for spatial coordinates)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            skip_connections: Whether to use skip connections (not used in analytical training)
            use_bfloat16: Whether to use BFloat16 precision
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.use_bfloat16 = use_bfloat16
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, 1))
        
        # Initialize with small weights for stability
        self._initialize_weights()
        
        # Convert to BFloat16 if requested
        if use_bfloat16:
            self.to(dtype=torch.bfloat16)
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (N, input_dim)
            
        Returns:
            Output tensor (N, 1)
        """
        h = x
        
        # All layers except last
        for layer in self.layers[:-1]:
            h = torch.relu(layer(h))
        
        # Output layer (no activation)
        h = self.layers[-1](h)
        
        return h
    
    def eval_activations(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate network and return both output and pre-activation values.
        
        Used for visualization of activation boundaries.
        
        Args:
            x: Input tensor (N, input_dim)
        
        Returns:
            output: (N, 1) network output
            preactivations: (N, num_layers, hidden_dim) pre-activation values
        """
        h = x
        preacts = []
        
        # All layers except last
        for layer in self.layers[:-1]:
            h_pre = layer(h)
            preacts.append(h_pre)
            h = torch.relu(h_pre)
        
        # Output layer (no activation)
        output = self.layers[-1](h)
        
        # Stack pre-activations
        preacts = torch.stack(preacts, dim=1)  # (N, num_layers, hidden_dim)
        
        return output, preacts
    
    def get_layer_params(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get weight and bias for a specific layer.
        
        Helper for analytical training.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            (weight, bias) tuple
        """
        layer = self.layers[layer_idx]
        return layer.weight, layer.bias
    
    def num_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def to_precision(self, dtype: torch.dtype):
        """
        Convert model to specified precision.
        
        Args:
            dtype: Target dtype (torch.float32, torch.bfloat16, etc.)
        """
        self.to(dtype=dtype)
        self.use_bfloat16 = (dtype == torch.bfloat16)
    
    def get_precision(self) -> torch.dtype:
        """Get current model precision."""
        return next(self.parameters()).dtype
    
    def summary(self) -> str:
        """
        Get a summary string of the model architecture.
        
        Returns:
            Human-readable summary
        """
        lines = []
        lines.append("="*60)
        lines.append("ReluMLP Summary")
        lines.append("="*60)
        lines.append(f"Input dimension:    {self.input_dim}")
        lines.append(f"Hidden dimension:   {self.hidden_dim}")
        lines.append(f"Number of layers:   {self.num_layers}")
        lines.append(f"Output dimension:   1")
        lines.append(f"Skip connections:   {self.skip_connections}")
        lines.append(f"Precision:          {self.get_precision()}")
        lines.append(f"Total parameters:   {self.num_parameters():,}")
        lines.append("="*60)
        lines.append("\nLayer structure:")
        for i, layer in enumerate(self.layers):
            in_feat = layer.in_features
            out_feat = layer.out_features
            params = in_feat * out_feat + out_feat
            if i == 0:
                lines.append(f"  Layer 0 (Input):   {in_feat:4d} → {out_feat:4d}  ({params:6,} params)")
            elif i == len(self.layers) - 1:
                lines.append(f"  Layer {i} (Output):  {in_feat:4d} → {out_feat:4d}  ({params:6,} params)")
            else:
                lines.append(f"  Layer {i} (Hidden):  {in_feat:4d} → {out_feat:4d}  ({params:6,} params)")
        lines.append("="*60)
        
        return "\n".join(lines)


def create_mlp(
    input_dim: int,
    hidden_dim: int = 32,
    num_layers: int = 4,
    use_bfloat16: bool = False,
    device: str = 'cuda'
) -> ReluMLP:
    """
    Factory function to create and initialize an MLP.
    
    Args:
        input_dim: Input dimension (2 or 3)
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        use_bfloat16: Whether to use BFloat16 precision
        device: Device to place model on
        
    Returns:
        Initialized ReluMLP model
    """
    model = ReluMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_bfloat16=use_bfloat16
    )
    
    model = model.to(device)
    
    return model
