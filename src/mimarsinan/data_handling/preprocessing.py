"""Dataset-agnostic input preprocessing (resize + normalization) shared by every ``DataProvider``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import torchvision.transforms as transforms


# Named normalization presets are PROVIDER-REGISTERED dataset facts: each
# data-provider module registers its canonical mean/std at import (the
# data_providers package import guarantees registration before any lookup
# through the provider factory). The framework ships no dataset constants here.
NORMALIZATION_PRESETS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}


def register_normalization_preset(
    name: str,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    aliases: Sequence[str] = (),
) -> None:
    """Register a named canonical normalization (idempotent; providers call this)."""
    for key in (name, *aliases):
        NORMALIZATION_PRESETS[str(key).lower()] = (tuple(mean), tuple(std))


_INTERPOLATION_MODES = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "nearest": transforms.InterpolationMode.NEAREST,
}


def interpolation_mode_names() -> tuple:
    """The declared interpolation option names (wizard schema surface)."""
    return tuple(_INTERPOLATION_MODES)


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
                f"Known presets: {sorted(NORMALIZATION_PRESETS)} "
                f"(presets are provider-registered; importing "
                f"mimarsinan.data_handling.data_providers registers the "
                f"shipped ones)"
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
