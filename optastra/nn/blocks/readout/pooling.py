import torch
import torch.nn as nn


class GlobalAvgPool2d(nn.Module):
    """Global Average Pooling (GAP) layer.

    This layer computes the average of each feature map across its spatial dimensions,
    resulting in a single value per feature map. It is commonly used in CNN architectures
    to reduce the spatial dimensions of feature maps before feeding them into fully connected layers.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass through the Global Average Pooling layer.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Output tensor of shape (batch_size, channels).
        """
        # x is expected to be of shape (batch_size, channels, height, width)
        return torch.mean(x, dim=(2, 3))  # Average over height and width


class GlobalMaxPool2d(nn.Module):
    """Global Max Pooling (GMP) layer.

    This layer computes the maximum value of each feature map across its spatial dimensions,
    resulting in a single value per feature map. It is often used in CNN architectures
    to reduce the spatial dimensions of feature maps while retaining the most salient features.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass through the Global Max Pooling layer.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Output tensor of shape (batch_size, channels).
        """
        # x is expected to be of shape (batch_size, channels, height, width)
        return torch.amax(x, dim=(2, 3))  # Max over height and width


class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling (GeM) layer.

    This layer computes the generalized mean of each feature map across its spatial dimensions,
    allowing for a flexible pooling operation that can interpolate between average and max pooling.
    The pooling parameter 'p' controls the behavior of the pooling operation.
    """

    def __init__(self, p=1.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Forward pass through the Generalized Mean Pooling layer.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Output tensor of shape (batch_size, channels).
        """
        # x is expected to be of shape (batch_size, channels, height, width)
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(2, 3)).pow(1.0 / self.p)


class TokenPooling(nn.Module):
    """
    Token Pooling layer for transformer-based architectures.

    This layer is designed to pool token embeddings, typically used in transformer models.
    It can be used to aggregate information from multiple tokens into a single representation.
    """

    def __init__(self, method='mean'):
        super().__init__()
        self.method = method

    def forward(self, x):
        """
        Forward pass through the Token Pooling layer.

        :param x: Input tensor of shape (batch_size, num_tokens, embedding_dim).
        :return: Output tensor of shape (batch_size, embedding_dim).
        """
        # x is expected to be of shape (batch_size, num_tokens, embedding_dim)
        if self.method == 'mean':
            return torch.mean(x, dim=1)  # Average over tokens
        elif self.method == 'max':
            return torch.max(x, dim=1).values  # Max over tokens
        else:
            raise ValueError(f"Unsupported pooling method: {self.method}")
        