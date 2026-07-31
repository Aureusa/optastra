from typing import Union
import torch
import torch.nn as nn

from .._pytorch_primitives import get_norm, get_activation, get_dropout


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) block.
    
    For num_layers = 2, the architecture is:
    Linear -> Norm -> Activation -> Dropout -> Linear (standard MLP for ViT)
    For num_layers > 2, the architecture is:
    Linear -> Norm -> Activation -> Dropout -> ... -> Linear
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        activation: str = "gelu",
        norm: Union[str, None] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            input_dim = in_features if i == 0 else hidden_features
            output_dim = out_features if i == num_layers - 1 else hidden_features
            layers.append(nn.Linear(input_dim, output_dim))
            if i < num_layers - 1:
                # Linear -> Norm -> Activation -> Dropout
                layers.append(get_norm(norm, num_features=hidden_features))
                layers.append(get_activation(activation))
                layers.append(get_dropout("dropout", p=dropout))
            # Ends with a linear layer

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Union[torch.Tensor, torch.nn.Module]) -> torch.Tensor:
        """Forward pass through the MLP.

        :param x: Input tensor of shape (batch_size, in_features).
        :return: Output tensor of shape (batch_size, out_features).
        """
        return self.mlp(x)
    