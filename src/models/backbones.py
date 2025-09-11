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
    

class VGGish(nn.Module):
    """
    VGGish-style architecture with parametrizable layer channels, output fully connected layer,
    and residual skip connections within each block.
    Args:
        in_channels (int): Number of input channels (e.g., 1 for mel spectrograms).
        channels (list of int): List of output channels for each conv block.
        proj_dim (int): Output dimension of the final fully connected layer.
    """
    def __init__(self, in_channels=1, channels=[64, 128, 256, 512], proj_dim=128):
        super(VGGish, self).__init__()
        self.in_channels = in_channels
        self.channels = channels

        self.blocks = nn.ModuleList()
        input_c = in_channels
        for i, out_c in enumerate(channels):
            block = []
            # First conv
            block.append(nn.Conv2d(input_c, out_c, kernel_size=3, padding=1))
            block.append(nn.BatchNorm2d(out_c))
            block.append(nn.ReLU(inplace=True))
            # Second conv
            block.append(nn.Conv2d(out_c, out_c, kernel_size=3, padding=1))
            block.append(nn.BatchNorm2d(out_c))
            # No activation here, will be after residual
            self.blocks.append(nn.Sequential(*block))
            input_c = out_c

        # For skip connections, if in/out channels differ, use 1x1 conv
        self.res_convs = nn.ModuleList()
        input_c = in_channels
        for out_c in channels:
            if input_c != out_c:
                self.res_convs.append(nn.Conv2d(input_c, out_c, kernel_size=1))
            else:
                self.res_convs.append(nn.Identity())
            input_c = out_c

        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2) for _ in channels
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], proj_dim)

    def forward(self, x):
        # Accepts input of shape (B, C, F, T) or (B, F, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, F, T)
        for i, block in enumerate(self.blocks):
            identity = self.res_convs[i](x)
            out = block(x)
            out = out + identity
            out = nn.functional.relu(out, inplace=True)
            out = self.maxpools[i](out)
            x = out
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return {
            "z": x
        }

# Example instantiation:
# vggish = VGGish(in_channels=1, channels=[64, 128, 256, 512], proj_dim=128)
vggish = VGGish(in_channels=1, channels=[128, 256, 512, 512], proj_dim=512)
