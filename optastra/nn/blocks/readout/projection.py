import torch
import torch.nn as nn


class FlattenAndProjectLinear(nn.Module):
    """A simple readout block that flattens the input feature map
    and projects it to a desired output dimension using a linear layer.

    This block is typically used as a readout layer in neural networks,
    where the spatial dimensions of the feature map are flattened and
    then projected to a lower-dimensional space using a linear layer.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FlattenAndProject block.

        :param x: Input tensor of shape (batch_size, in_channels, height, width).
        :return: Output tensor of shape (batch_size, out_channels).
        """
        x = self.flatten(x)  # Flatten the spatial dimensions
        x = self.projection(x)  # Project to the desired output dimension
        return x


class FlattenAndProjectConv(nn.Module):
    """
    A readout block that flattens the input feature map and projects it to a
    desired output dimension using a convolutional layer.

    This block is useful for scenarios where you want to reduce the spatial
    dimensions of the feature map while preserving
    some spatial information through convolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FlattenAndProjectConv block.

        :param x: Input tensor of shape (batch_size, in_channels, height, width).
        :return: Output tensor of shape (batch_size, out_channels).
        """
        x = self.conv(x)  # Apply 1x1 convolution to project channels
        x = self.flatten(x)  # Flatten the spatial dimensions
        return x
    