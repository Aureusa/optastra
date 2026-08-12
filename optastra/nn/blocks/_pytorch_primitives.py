from torch import nn


_NORMS = {
    "batchnorm": nn.BatchNorm2d,
    "batchnorm1d": nn.BatchNorm1d,
    "layernorm": nn.LayerNorm,
    "groupnorm": nn.GroupNorm,
    None: None
}
_ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "silu": nn.SiLU,
    None: None
}
_DROPS = {
    "dropout": nn.Dropout,
    "dropout2d": nn.Dropout2d,
    None: None
}

def list_norms():
    """List all available normalization types."""
    return list(_NORMS.keys())

def list_activations():
    """List all available activation types."""
    return list(_ACTS.keys())

def list_dropouts():
    """List all available dropout types."""
    return list(_DROPS.keys())

def get_norm(norm_name: str, num_features: int, num_groups: int = 32) -> nn.Module:
    """Get the normalization layer based on the provided name.

    :param norm_name: Name of the normalization type. Use list_norms() to see available options.
    :param num_features: Number of features (channels) for the normalization layer.
    :param num_groups: Number of groups for group normalization. Default is 32.
    :return: An instance of the requested normalization layer.
    """
    if norm_name not in _NORMS:
        raise ValueError(f"Unsupported normalization type: {norm_name}")
    norm_class = _NORMS[norm_name]
    if norm_class is None:
        return nn.Identity()
    if norm_name == "groupnorm":
        # For group normalization, we need to specify the number of groups.
        # Here, we use 32 as a common choice, but this can be adjusted as needed.
        return norm_class(num_groups=num_groups, num_channels=num_features)
    return norm_class(num_features)

def get_activation(act_name: str) -> nn.Module:
    """Get the activation layer based on the provided name.

    :param act_name: Name of the activation type. Use list_activations() to see available options.
    :return: An instance of the requested activation layer.
    """
    if act_name not in _ACTS:
        raise ValueError(f"Unsupported activation type: {act_name}")
    act_class = _ACTS[act_name]
    if act_class is None:
        return nn.Identity()
    return act_class()

def get_dropout(drop_name: str, p: float = 0.5) -> nn.Module:
    """Get the dropout layer based on the provided name.

    :param drop_name: Name of the dropout type. Use list_dropouts() to see available options.
    :param p: Dropout probability. Default is 0.5.
    :return: An instance of the requested dropout layer.
    """
    if drop_name not in _DROPS:
        raise ValueError(f"Unsupported dropout type: {drop_name}")
    drop_class = _DROPS[drop_name]
    if drop_class is None:
        return nn.Identity()
    return drop_class(p)
