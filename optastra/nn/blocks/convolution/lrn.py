"""
This module contains the implementation of Local Response Normalization (LRN).
It was originally introduced in the AlexNet architecture and is used to normalize
the activations of a layer across the channels.
References:
    - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
    ImageNet Classification with Deep Convolutional Neural Networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalResponseNorm(nn.Module):
    def __init__(
            self,
            size: int = 5,
            alpha: float = 1e-4,
            beta: float = 0.75,
            k: float = 2.0
        ):
        """
        Initializes the Local Response Normalization layer.

        :param size: The number of adjacent channels to normalize over.
        :param alpha: The scaling parameter.
        :param beta: The exponent parameter.
        :param k: An offset (usually set to 2).
        """
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LRN layer:
            b_c = a_c / (k + alpha / size * sum_{c' in window(c)} a_{c'}^2) ** beta
        (same convention as torch.nn.LocalResponseNorm / Caffe).

        :param x: Input tensor of shape (N, C, H, W).
        :return: Normalized tensor of the same shape as input.
        """
        # LRN in AlexNet normalizes across adjacent channels for each spatial location.
        # We implement this as a zero-padded local average of a^2 over the channel dimension
        # for each (H, W). The average already divides by `size`, so alpha is NOT divided again.
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 1, c)          # (N*H*W, 1, C)
        squared_input = x_flat.pow(2)
        # window centred on c: size // 2 channels before, (size - 1) // 2 after (also correct for even sizes)
        squared_input = F.pad(squared_input, (self.size // 2, (self.size - 1) // 2))
        mean_sq = F.avg_pool1d(squared_input, kernel_size=self.size, stride=1)   # (N*H*W, 1, C)

        scale = self.k + self.alpha * mean_sq
        normalized_flat = x_flat / scale.pow(self.beta)
        return normalized_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
