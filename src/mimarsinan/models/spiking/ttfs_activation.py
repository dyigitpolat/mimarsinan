"""Map IR activation_type strings to torch activation callables."""

from __future__ import annotations

import torch.nn.functional as F


def ttfs_activation_from_type(activation_type: str | None):
    """Compound strings use the base name before ' + '."""
    if activation_type is None or (
        isinstance(activation_type, str) and activation_type.strip() in ("", "ReLU")
    ):
        return F.relu
    base = activation_type.split(" + ")[0].strip()
    name_map = {
        "LeakyReLU": "leaky_relu",
        "LeakyGradReLU": "relu",
        "ReLU": "relu",
        "GELU": "gelu",
        "Identity": "identity",
    }
    f_name = name_map.get(base, "relu")
    if f_name == "identity":
        return lambda x: x
    try:
        return getattr(F, f_name)
    except AttributeError:
        return F.relu
