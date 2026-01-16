"""
Simple ReLU MLP for learning implicit neural representations.
"""

import torch
import torch.nn as nn


class ReluMLP(nn.Module):
    """Multi-layer perceptron with ReLU activations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, skip_connections: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, 1))
    
    def forward(self, x):
        """Forward pass through the network."""
        h = x
        
        # All layers except last
        for layer in self.layers[:-1]:
            h = torch.relu(layer(h))
        
        # Output layer (no activation)
        h = self.layers[-1](h)
        
        return h
    
    def eval_activations(self, x):
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
        preacts = torch.stack(preacts, dim=1)  # (N, num_layers-1, hidden_dim)
        
        return output, preacts

