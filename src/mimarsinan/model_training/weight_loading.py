"""
Weight loading strategies for pretrained model initialization.

Provides a strategy pattern for loading weights from various sources
(torchvision pretrained, local checkpoints, URLs) into native PyTorch
models before or instead of training from scratch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _filter_compatible(
    model: nn.Module, src_sd: Dict[str, Any]
) -> tuple[Dict[str, Any], list[str]]:
    """Return only state-dict entries whose shapes match the model's parameters.

    Returns ``(compatible_sd, skipped_keys)`` where *skipped_keys* lists
    the names of parameters dropped due to shape mismatch.
    """
    model_sd = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in src_sd.items():
        if key in model_sd:
            if value.shape == model_sd[key].shape:
                compatible[key] = value
            else:
                skipped.append(key)
        else:
            compatible[key] = value
    return compatible, skipped


class WeightLoadingStrategy(ABC):
    """Base class for weight loading strategies."""

    @abstractmethod
    def load(self, model: nn.Module, **kwargs) -> tuple[nn.Module, Dict[str, Any]]:
        """Load weights into *model* and return ``(model, info)``.

        *info* is a dict with metadata about what was loaded, e.g.
        ``{"matched": 150, "missing": 2, "unexpected": 0, "source": "..."}``.
        """


class TorchvisionWeightStrategy(WeightLoadingStrategy):
    """Load pretrained weights from a torchvision model factory.

    The strategy creates a fresh pretrained instance using the provided
    factory, extracts its ``state_dict``, and loads compatible parameters
    into the target model.  Shape-mismatched parameters (e.g. a classifier
    head with different ``num_classes``) are silently skipped so the target
    model keeps its randomly-initialised values for those layers.
    """

    def __init__(self, pretrained_factory):
        """
        Args:
            pretrained_factory: Callable that returns a pretrained nn.Module.
                E.g. ``lambda: torchvision.models.vgg16_bn(weights="DEFAULT")``.
        """
        self._factory = pretrained_factory

    def load(self, model: nn.Module, **kwargs) -> tuple[nn.Module, Dict[str, Any]]:
        pretrained = self._factory()
        src_sd = pretrained.state_dict()
        compatible_sd, skipped = _filter_compatible(model, src_sd)
        result = model.load_state_dict(compatible_sd, strict=False)

        info = {
            "source": "torchvision",
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
            "shape_skipped_keys": skipped,
            "matched": len(compatible_sd) - len(result.unexpected_keys),
        }
        del pretrained
        return model, info


class CheckpointWeightStrategy(WeightLoadingStrategy):
    """Load weights from a local checkpoint file (``.pt`` / ``.pth``)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, model: nn.Module, **kwargs) -> tuple[nn.Module, Dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.path}")

        state = torch.load(self.path, map_location="cpu", weights_only=False)

        if isinstance(state, dict) and "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        elif isinstance(state, dict):
            sd = state
        else:
            raise ValueError(
                f"Unexpected checkpoint format from {self.path}. "
                "Expected a state_dict or a dict with 'model_state_dict'/'state_dict' key."
            )

        compatible_sd, skipped = _filter_compatible(model, sd)
        result = model.load_state_dict(compatible_sd, strict=False)
        info = {
            "source": str(self.path),
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
            "shape_skipped_keys": skipped,
            "matched": len(compatible_sd) - len(result.unexpected_keys),
        }
        return model, info


class URLWeightStrategy(WeightLoadingStrategy):
    """Download weights from a URL and load them."""

    def __init__(self, url: str):
        self.url = url

    def load(self, model: nn.Module, **kwargs) -> tuple[nn.Module, Dict[str, Any]]:
        sd = torch.hub.load_state_dict_from_url(
            self.url, map_location="cpu", check_hash=False
        )
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        compatible_sd, skipped = _filter_compatible(model, sd)
        result = model.load_state_dict(compatible_sd, strict=False)
        info = {
            "source": self.url,
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
            "shape_skipped_keys": skipped,
            "matched": len(compatible_sd) - len(result.unexpected_keys),
        }
        return model, info


def resolve_weight_strategy(
    weight_source: str,
    model_builder=None,
) -> Optional[WeightLoadingStrategy]:
    """Resolve a ``weight_source`` config string into a loading strategy.

    Args:
        weight_source: One of:
            - ``"torchvision"`` -- use the builder's pretrained factory
            - A file path ending in ``.pt``, ``.pth``, ``.ckpt`` -- checkpoint
            - A string starting with ``http://`` or ``https://`` -- URL download
        model_builder: The model builder (must have ``get_pretrained_factory()``
            when ``weight_source="torchvision"``).

    Returns:
        A ``WeightLoadingStrategy``, or ``None`` if ``weight_source`` is falsy.
    """
    if not weight_source:
        return None

    if weight_source == "torchvision":
        if model_builder is None or not hasattr(model_builder, "get_pretrained_factory"):
            raise ValueError(
                "weight_source='torchvision' requires a model builder with "
                "get_pretrained_factory(). The current builder does not support it."
            )
        factory = model_builder.get_pretrained_factory()
        return TorchvisionWeightStrategy(factory)

    if weight_source.startswith("http://") or weight_source.startswith("https://"):
        return URLWeightStrategy(weight_source)

    path = Path(weight_source)
    if path.suffix in (".pt", ".pth", ".ckpt", ".bin"):
        return CheckpointWeightStrategy(path)

    raise ValueError(
        f"Cannot resolve weight_source='{weight_source}'. Expected 'torchvision', "
        f"a file path (.pt/.pth/.ckpt), or a URL."
    )
