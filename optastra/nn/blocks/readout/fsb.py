import torch
import torch.nn as nn


class FeatureSelectionBlock(nn.Module):
    """
    Feature Selection Block (FSB) for selecting and combining features from multiple stages.

    This block takes a list of feature maps from different stages of a backbone network,
    applies a selection mechanism, and combines them into a single output feature map.
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # Define a convolutional layer for each input feature map
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        # Define a final convolutional layer to combine the selected features
        self.final_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        """
        Forward pass through the Feature Selection Block.

        :param features: List of input feature maps from different stages.
        :return: Output tensor after selecting and combining features.
        """
        # Apply the convolutional layers to each input feature map
        selected_features = [conv(feature) for conv, feature in zip(self.convs, features)]

        # Concatenate the selected features along the channel dimension
        combined_features = torch.cat(selected_features, dim=1)

        # Apply the final convolution to combine the features
        output = self.final_conv(combined_features)

        return output
    