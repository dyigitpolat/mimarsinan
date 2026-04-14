"""Dataset-agnostic input preprocessing (resize + normalization).

A ``PreprocessingSpec`` is constructed from a config dict (``preprocessing``
block under ``deployment_parameters``) and consumed by every ``DataProvider``
to wrap its native transforms with a consistent resize + normalize policy.

This keeps preprocessing a dataset/pipeline concern rather than a per-provider
hardcoded choice, so any model/dataset combination can dictate input shape
(e.g. ViT pretrained weights expect 224x224 + ImageNet normalization) without
editing provider code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import torchvision.transforms as transforms


NORMALIZATION_PRESETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "cifar": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


_INTERPOLATION_MODES = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "nearest": transforms.InterpolationMode.NEAREST,
}


@dataclass(frozen=True)
class PreprocessingSpec:
    """Resolved preprocessing policy. Apply via :meth:`compose`."""

    resize_to: Optional[int] = None
    interpolation: str = "bicubic"
    mean: Optional[Sequence[float]] = None
    std: Optional[Sequence[float]] = None

    def _resize_transform(self):
        if self.resize_to is None:
            return None
        mode = _INTERPOLATION_MODES.get(self.interpolation, transforms.InterpolationMode.BICUBIC)
        return transforms.Resize(
            (int(self.resize_to), int(self.resize_to)),
            interpolation=mode,
            antialias=True,
        )

    def _normalize_transform(self):
        if self.mean is None or self.std is None:
            return None
        return transforms.Normalize(list(self.mean), list(self.std))

    def compose(self, base_transforms: Iterable) -> transforms.Compose:
        """Wrap ``base_transforms`` with resize (before ToTensor) and normalize (after).

        Resize is inserted **before** the first ``ToTensor`` so PIL-based
        augmentations still see the target resolution; Normalize is appended
        after all tensor ops.
        """
        base = list(base_transforms)
        resize = self._resize_transform()
        normalize = self._normalize_transform()

        if resize is not None:
            insert_at = 0
            for i, t in enumerate(base):
                if isinstance(t, transforms.ToTensor):
                    insert_at = i
                    break
                insert_at = i + 1
            base.insert(insert_at, resize)

        if normalize is not None:
            base.append(normalize)

        return transforms.Compose(base)


def _resolve_normalize(value) -> tuple[Optional[Sequence[float]], Optional[Sequence[float]]]:
    if value is None:
        return None, None
    if isinstance(value, str):
        key = value.lower()
        if key in ("none", "off", ""):
            return None, None
        if key not in NORMALIZATION_PRESETS:
            raise ValueError(
                f"Unknown normalization preset {value!r}. "
                f"Known presets: {sorted(NORMALIZATION_PRESETS)}"
            )
        mean, std = NORMALIZATION_PRESETS[key]
        return mean, std
    if isinstance(value, dict):
        return tuple(value["mean"]), tuple(value["std"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        mean, std = value
        return tuple(mean), tuple(std)
    raise ValueError(f"Unsupported normalization spec: {value!r}")


def resolve_preprocessing(
    preprocessing: Union[PreprocessingSpec, dict, None],
) -> Optional[PreprocessingSpec]:
    """Build a :class:`PreprocessingSpec` from a config dict (or pass through).

    Returns ``None`` when no preprocessing is requested so callers can keep
    their native transforms untouched.
    """
    if preprocessing is None:
        return None
    if isinstance(preprocessing, PreprocessingSpec):
        return preprocessing
    if not isinstance(preprocessing, dict):
        raise TypeError(f"preprocessing must be dict or PreprocessingSpec, got {type(preprocessing)!r}")

    resize_to = preprocessing.get("resize_to")
    if resize_to is not None:
        resize_to = int(resize_to)

    interpolation = preprocessing.get("interpolation", "bicubic")
    mean, std = _resolve_normalize(preprocessing.get("normalize"))

    if resize_to is None and mean is None and std is None:
        return None

    return PreprocessingSpec(
        resize_to=resize_to,
        interpolation=str(interpolation),
        mean=mean,
        std=std,
    )
