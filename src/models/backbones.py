import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable hidden layers and activation functions.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        hidden_features: List of hidden layer sizes
        activation: Activation function ('relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu', 'swish')
        dropout: Dropout probability (0.0 means no dropout)
        use_batch_norm: Whether to use batch normalization
        bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        hidden_features: List[int], 
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        bias: bool = True
    ):
        super(MLP, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation_name = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build the network architecture
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList() if dropout > 0.0 else None
        
        # Input layer
        prev_features = in_features
        
        # Hidden layers
        for hidden_size in hidden_features:
            self.layers.append(nn.Linear(prev_features, hidden_size, bias=bias))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            
            if dropout > 0.0:
                self.dropouts.append(nn.Dropout(dropout))
            
            prev_features = hidden_size
        
        # Output layer
        self.layers.append(nn.Linear(prev_features, out_features, bias=bias))
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, x):
        """Apply the specified activation function."""
        if self.activation_name == "relu":
            return F.relu(x)
        elif self.activation_name == "gelu":
            return F.gelu(x)
        elif self.activation_name == "tanh":
            return torch.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
    
    def forward(self, x):
        """Forward pass through the MLP."""
        # Flatten input if it has more than 2 dimensions
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers[:-1]):  # All layers except the last one
            x = layer(x)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Apply activation function
            x = self._get_activation(x)
            
            # Apply dropout if enabled
            if self.dropout > 0.0 and self.dropouts is not None:
                x = self.dropouts[i](x)
        
        # Output layer (no activation, no dropout)
        x = self.layers[-1](x)
        
        return x
    