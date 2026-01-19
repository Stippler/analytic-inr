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
        
        # Hidden layers (with skip connections if enabled)
        for _ in range(num_layers - 1):
            # If skip connections, concatenate input to hidden state
            layer_input_dim = hidden_dim + input_dim if skip_connections else hidden_dim
            self.layers.append(nn.Linear(layer_input_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, 1))
    
    def forward(self, x):
        """Forward pass through the network."""
        # First layer
        h = torch.relu(self.layers[0](x))
        
        # Hidden layers (with skip connections if enabled)
        for layer in self.layers[1:-1]:
            if self.skip_connections:
                # Concatenate input to hidden state
                h_input = torch.cat([h, x], dim=-1)
            else:
                h_input = h
            h = torch.relu(layer(h_input))
        
        # Output layer (no activation)
        output = self.layers[-1](h)
        
        return output
    
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
        # First layer
        h_pre = self.layers[0](x)
        preacts = [h_pre]
        h = torch.relu(h_pre)
        
        # Hidden layers (with skip connections if enabled)
        for layer in self.layers[1:-1]:
            if self.skip_connections:
                # Concatenate input to hidden state
                h_input = torch.cat([h, x], dim=-1)
            else:
                h_input = h
            h_pre = layer(h_input)
            preacts.append(h_pre)
            h = torch.relu(h_pre)
        
        # Output layer (no activation)
        output = self.layers[-1](h)
        
        # Stack pre-activations
        preacts = torch.stack(preacts, dim=1)  # (N, num_layers-1, hidden_dim)
        
        return output, preacts

