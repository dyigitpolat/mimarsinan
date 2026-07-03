"""Weight loading strategies for pretrained model initialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _filter_compatible(
    model: nn.Module, src_sd: Dict[str, Any]
) -> tuple[Dict[str, Any], list[str]]:
    """Return state-dict entries compatible with the model, and the list of shape-mismatched keys skipped."""
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
        """Load weights into *model* and return ``(model, info)``; *info* holds load metadata."""


class TorchvisionWeightStrategy(WeightLoadingStrategy):
    """Load pretrained weights from a torchvision model factory; shape-mismatched
    params (e.g. a differently-sized head) are skipped so the target keeps its init.
    """

    def __init__(self, pretrained_factory):
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


class UnsupportedPreloadError(ValueError):
    """Raised when a pretrained deploy is requested for a builder with no pretrained source.

    Typed so the pipeline records a clean ``UNSUPPORTED`` skip instead of an opaque
    ``rc=1``; subclasses ``ValueError`` for back-compat.
    """


def torchvision_source_supported(model_builder=None) -> bool:
    """Whether a ``weight_source='torchvision'`` preload can resolve for this builder.

    Non-raising predicate; the raising path lives in ``resolve_weight_strategy``.
    """
    return model_builder is not None and hasattr(
        model_builder, "get_pretrained_factory"
    )


def resolve_weight_strategy(
    weight_source: str,
    model_builder=None,
) -> Optional[WeightLoadingStrategy]:
    """Resolve a ``weight_source`` string (``'torchvision'``, a checkpoint path, or an http(s) URL) into a strategy, or ``None`` if falsy.

    Raises ``UnsupportedPreloadError`` early when torchvision is requested for a
    builder without ``get_pretrained_factory()``.
    """
    if not weight_source:
        return None

    if weight_source == "torchvision":
        if not torchvision_source_supported(model_builder):
            raise UnsupportedPreloadError(
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
