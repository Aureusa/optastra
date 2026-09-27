"""
Pretrained-weight flow for backbones: load a backbone's weights out of any
state_dict or optastra checkpoint, and export just the backbone from a
training checkpoint.

The same backbone is stored under different key prefixes depending on the
model that trained it:

    BYOLModel                 online_backbone.layer1.0.conv1.weight
    SimCLRModel               backbone.layer1.0.conv1.weight
    build_sequential_model    0.layer1.0.conv1.weight
    torch.compile(model)      _orig_mod.<any of the above>

`load_backbone_weights` strips wrapper prefixes (`_orig_mod.`, `module.`)
and then picks whichever known prefix matches the backbone's own keys, so
    Backbone.create("resnet50", weights="runs/byol/ckpt_1000.pt")
just works; pass `weights_prefix=` to choose explicitly.
"""
from __future__ import annotations
import os
from typing import Any, Mapping

import torch
import torch.nn as nn


__all__ = ["load_backbone_weights", "export_backbone", "read_state_dict"]


# Prepended to every key by a wrapper around the whole model, never part of a real module name.
WRAPPER_PREFIXES = ("_orig_mod.", "module.")   # torch.compile, DataParallel/DDP
# Where the backbone lives inside the models optastra builds, in order of preference.
KNOWN_PREFIXES = ("", "online_backbone.", "backbone.", "0.")


def read_state_dict(source: str | os.PathLike | Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """
    A model state_dict from a raw state_dict (file or dict) or an optastra
    checkpoint (`{"model": state_dict, "optimizer": ..., ...}`), with
    wrapper prefixes (`_orig_mod.`, `module.`) removed.
    """
    obj = torch.load(source, map_location="cpu", weights_only=True) if isinstance(source, (str, os.PathLike)) else source
    if isinstance(obj, Mapping) and isinstance(obj.get("model"), Mapping):
        obj = obj["model"]   # optastra Checkpointer format
    if not isinstance(obj, Mapping) or not all(torch.is_tensor(v) for v in obj.values()):
        raise ValueError(
            "Expected a state_dict (str -> Tensor) or an optastra checkpoint with a 'model' entry, "
            f"got {type(obj).__name__} with keys {list(obj)[:5] if isinstance(obj, Mapping) else '-'}."
        )
    return {_strip_wrappers(k): v for k, v in obj.items()}


def _strip_wrappers(key: str) -> str:
    stripped = True
    while stripped:
        stripped = False
        for w in WRAPPER_PREFIXES:
            if key.startswith(w):
                key, stripped = key[len(w):], True
    return key


def _select_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _top_level_prefixes(keys) -> list[str]:
    return sorted({k.split(".", 1)[0] + "." for k in keys if "." in k})


def load_backbone_weights(
        backbone: nn.Module,
        source: str | os.PathLike | Mapping[str, Any],
        prefix: str | None = None,
        strict: bool = True,
    ) -> nn.Module:
    """
    Load backbone weights from `source` (path or dict; raw state_dict or
    optastra checkpoint) into `backbone`, in place.

    :param prefix: key prefix the backbone sits under in `source`, e.g.
        "online_backbone.". None = auto-detect: the first of
        "", "online_backbone.", "backbone.", "0." that matches the most of
        the backbone's keys.
    :param strict: raise if any backbone key is missing from `source`, or
        `source` has keys under `prefix` the backbone doesn't know.
    """
    state_dict = read_state_dict(source)
    expected = set(backbone.state_dict().keys())

    if prefix is None:
        overlap = {p: len(expected & _select_prefix(state_dict, p).keys()) for p in KNOWN_PREFIXES}
        prefix = max(KNOWN_PREFIXES, key=lambda p: overlap[p])   # max() keeps the first on ties
        if overlap[prefix] == 0:
            raise ValueError(
                f"Could not find {type(backbone).__name__} weights in the given state_dict under any of "
                f"the prefixes {list(KNOWN_PREFIXES)}. Its top-level prefixes are "
                f"{_top_level_prefixes(state_dict)} -- pass weights_prefix=... explicitly."
            )

    weights = _select_prefix(state_dict, prefix)
    missing = sorted(expected - weights.keys())
    unexpected = sorted(weights.keys() - expected)
    if strict and (missing or unexpected):
        raise ValueError(
            f"Weights under prefix {prefix!r} don't match {type(backbone).__name__}:\n"
            f"  missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
            f"  unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}\n"
            "Check the backbone name/config, or pass weights_prefix=... / strict=False."
        )
    backbone.load_state_dict(weights, strict=strict)
    return backbone


def export_backbone(
        checkpoint_path: str | os.PathLike,
        out_path: str | os.PathLike,
        prefix: str | None = None,
    ) -> dict[str, torch.Tensor]:
    """
    Extract the backbone's weights from a training checkpoint (e.g. a BYOL
    run) and save them as a plain, prefix-free state_dict that
    `Backbone.create(name, weights=out_path)` loads directly.

    :param prefix: key prefix of the backbone inside the checkpoint. None =
        the first of "online_backbone.", "backbone.", "0." present in it.
    :return: the exported state_dict.
    """
    state_dict = read_state_dict(checkpoint_path)
    if prefix is None:
        candidates = [p for p in KNOWN_PREFIXES if p and any(k.startswith(p) for k in state_dict)]
        if not candidates:
            raise ValueError(
                f"No known backbone prefix {list(KNOWN_PREFIXES[1:])} in {checkpoint_path}. Its top-level "
                f"prefixes are {_top_level_prefixes(state_dict)} -- pass prefix=... explicitly."
            )
        prefix = candidates[0]
    weights = _select_prefix(state_dict, prefix)
    if not weights:
        raise ValueError(f"No keys start with {prefix!r} in {checkpoint_path}.")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(weights, out_path)
    return weights
