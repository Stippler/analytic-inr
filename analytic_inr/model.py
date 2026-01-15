"""
Neural network models for analytic implicit neural representations.
"""

import torch
import torch.nn as nn


class ReluMLP(nn.Module):
    """
    ReLU MLP with optional skip connections from input to each layer.
    
    With skip connections:
        - layers[0]: input_dim -> hidden_dim (first hidden layer)
        - layers[1..num_layers-1]: (hidden_dim + input_dim) -> hidden_dim (remaining hidden layers)
        - layers[num_layers]: (hidden_dim + input_dim) -> output_dim (output layer)
    
    Without skip connections:
        - layers[0]: input_dim -> hidden_dim (first hidden layer)
        - layers[1..num_layers-1]: hidden_dim -> hidden_dim (remaining hidden layers)
        - layers[num_layers]: hidden_dim -> output_dim (output layer)
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of hidden layers (not counting output layer)
        output_dim: Dimension of output
        skip_connections: If True, concatenate input with each layer's output
    """
    
    def __init__(self, input_dim=2, hidden_dim=3, num_layers=3, output_dim=1, skip_connections=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip_connections = skip_connections
        
        # All layers in a single ModuleList for easy access
        self.layers = nn.ModuleList()
        
        # First hidden layer: input_dim -> hidden_dim
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        if skip_connections:
            # Remaining hidden layers: (hidden_dim + input_dim) -> hidden_dim
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            
            # Output layer: (hidden_dim + input_dim) -> output_dim
            self.layers.append(nn.Linear(hidden_dim + input_dim, output_dim))
        else:
            # Remaining hidden layers: hidden_dim -> hidden_dim
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            # Output layer: hidden_dim -> output_dim
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # for layer in self.layers:
        #     layer.weight.data = torch.zeros_like(layer.weight.data)
        #     layer.bias.data = torch.zeros_like(layer.bias.data)
    
    def forward(self, x):
        """Forward pass with optional skip connections."""
        if self.skip_connections:
            x_input = x
            
            # First hidden layer + ReLU
            x = torch.relu(self.layers[0](x))
            
            # Remaining hidden layers with skip connections + ReLU
            for i in range(1, self.num_layers):
                x = torch.cat([x, x_input], dim=-1)
                x = torch.relu(self.layers[i](x))
            
            # Output layer with skip connection (no ReLU)
            x = torch.cat([x, x_input], dim=-1)
            x = self.layers[self.num_layers](x)
            
            return x
        else:
            # First hidden layer + ReLU
            x = torch.relu(self.layers[0](x))
            
            # Remaining hidden layers + ReLU
            for i in range(1, self.num_layers):
                x = torch.relu(self.layers[i](x))
            
            # Output layer (no ReLU)
            x = self.layers[self.num_layers](x)
            
            return x
    
    def eval_activations(self, x):
        """
        Evaluate network and return both output and pre-activations.
        
        Returns:
            output: Network output
            preacts: Stacked pre-activation values from each hidden layer (before ReLU)
                     Shape: (batch, num_layers, hidden_dim)
        """
        if self.skip_connections:
            x_input = x
            preacts = []
            
            # First hidden layer
            x = self.layers[0](x)
            preacts.append(x.clone())
            x = torch.relu(x)
            
            # Remaining hidden layers with skip connections
            for i in range(1, self.num_layers):
                x = torch.cat([x, x_input], dim=-1)
                x = self.layers[i](x)
                preacts.append(x.clone())
                x = torch.relu(x)
            
            # Output layer with skip connection (no ReLU, no preact recording)
            x = torch.cat([x, x_input], dim=-1)
            x = self.layers[self.num_layers](x)
            
            return x, torch.stack(preacts, dim=1)
        else:
            preacts = []
            
            # First hidden layer
            x = self.layers[0](x)
            preacts.append(x.clone())
            x = torch.relu(x)
            
            # Remaining hidden layers
            for i in range(1, self.num_layers):
                x = self.layers[i](x)
                preacts.append(x.clone())
                x = torch.relu(x)
            
            # Output layer (no ReLU, no preact recording)
            x = self.layers[self.num_layers](x)
            
            return x, torch.stack(preacts, dim=1)

