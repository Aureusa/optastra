import torch
import torch.nn as nn

from ..backbones import get_backbone_entrypoint
from ..necks import get_neck_entrypoint
from ..heads import get_head_entrypoint


def create_model(
        backbone: str,
        neck: str,
        head: str,
        **kwargs,
    ) -> nn.Module:
    """Create a model based on the provided configuration.

    :return: An instance of the model defined by the configuration.
    """
    # Example implementation: Create a model based on the config
    backbone = _create_backbone(backbone)
    neck = _create_neck(neck, backbone.out_channels)
    last_stage = list(neck.out_channels.keys())[-1]  # Assuming the last stage is the one to be used for the head
    in_features = neck.out_channels[last_stage]
    head = _create_head(head, in_features=in_features, stage=last_stage)

    # Combine backbone, neck, and head into a single model
    model = nn.Sequential(backbone, neck, head)
    return model

def _create_backbone(name):
    """Create a backbone model based on the provided configuration.

    :param name: Name of the backbone.
    :return: An instance of the backbone model.
    """
    # Placeholder for actual backbone creation logic
    return get_backbone_entrypoint(name)()

def _create_neck(name, in_channels):
    """Create a neck module based on the provided configuration.

    :param name: Name of the neck.
    :return: An instance of the neck module.
    """
    # Placeholder for actual neck creation logic
    return get_neck_entrypoint(name)(in_channels=in_channels)

def _create_head(name, in_features, stage):
    """Create a head module based on the provided configuration.

    :param name: Name of the head.
    :return: An instance of the head module.
    """
    # Placeholder for actual head creation logic
    return get_head_entrypoint(name)(in_features=in_features, hidden_features=4 * in_features, stage=stage)
