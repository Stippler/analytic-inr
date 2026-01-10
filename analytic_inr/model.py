import torch
import torch.nn as nn

class ReluMLP(nn.Module):
    
    def __init__(self, input_dim=2, hidden_dim=3, num_layers=3, output_dim=1):
        """
        ReLU MLP with arbitrary input and output dimensions.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers
            output_dim: Dimension of output
        """
        super().__init__()
        layers = []

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=False))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=False))
        
        # Output layer (no ReLU)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    def eval_activations(self, x):
        preacts = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                preacts.append(x.clone())
                x = torch.relu(x)
        return x, torch.stack(preacts, dim=1)
    
    def linearize_at_point(self, x):
        """
        Linearize the network at a given point by collapsing layers based on ReLU activations.
        
        For a ReLU network at a specific point, the activation pattern (which neurons fire) is fixed.
        This allows us to collapse the entire network into a single linear transformation.
        
        Args:
            x: Input point (shape: (input_dim,))
        
        Returns:
            radii: List of radius tensors for each hidden layer (not including output layer)
            final_W: Final combined weight matrix after collapsing all layers (shape: (output_dim, input_dim))
            final_b: Final combined bias vector after collapsing all layers (shape: (output_dim,))
            outputs: List of pre-activation values at each layer (including output)
        """
        radii = []
        outputs = []
        
        with torch.no_grad():
            # Start with first layer
            W_current = self.layers[0].weight.clone()  # Shape: (H, input_dim)
            b_current = self.layers[0].bias.clone()     # Shape: (H,)
            
            # Iterate through all layers
            layer_idx = 0
            for i in range(0, len(self.layers), 2):  # Step by 2 (Linear + ReLU pairs)
                if i == len(self.layers) - 1:  # Last layer (output, no ReLU)
                    # Final output layer - compute output but skip radius
                    z = W_current @ x + b_current
                    outputs.append(z.clone())
                    # Skip radius computation for output layer (not meaningful for classification/multi-output)
                    break
                
                # Compute pre-activation
                z = W_current @ x + b_current
                outputs.append(z.clone())
                
                # Compute radius (normalized distance to hyperplane)
                radius = (z / torch.norm(W_current, dim=1)).clone()
                radii.append(radius)
                
                # Determine activation mask
                mask = (z > 0).to(torch.float32)
                
                # Apply mask to current weights and biases (zero out inactive neurons)
                W_masked = W_current * mask[:, None]
                b_masked = b_current * mask
                
                # If there's a next layer, combine it
                if i + 2 < len(self.layers):
                    next_linear_idx = i + 2
                    W_next = self.layers[next_linear_idx].weight.clone()
                    b_next = self.layers[next_linear_idx].bias.clone()
                    
                    # Combine: W_next @ (mask * (W_current @ x + b_current)) + b_next
                    #        = W_next @ mask @ W_current @ x + W_next @ mask @ b_current + b_next
                    W_current = W_next @ W_masked
                    b_current = W_next @ b_masked + b_next
                
                layer_idx += 1
            
            final_W = W_current  # Shape: (output_dim, input_dim)
            final_b = b_current  # Shape: (output_dim,)
        
        return radii, final_W, final_b, outputs