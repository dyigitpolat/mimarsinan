"""Declarative spec for an FFCV pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class FieldSpec:
    """One serialized field's name + the write-time type-class name.

    Decoders live in each split's chain (``SplitSpec.transforms``), not on the
    field, so one field can carry split-asymmetric decoders.
    """

    name: str
    write_type: str
    write_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SplitSpec:
    """Per-split (train/val/test) operation chain after decoding."""

    transforms: tuple = ()
    drop_last: bool = False
    shuffle: bool = False


@dataclass(frozen=True)
class PipelineSpec:
    """Complete FFCV pipeline declaration for one (provider, dataset)."""

    id: str
    fields: tuple[FieldSpec, ...]
    splits: dict[str, SplitSpec]
    notes: str = ""

    def stable_hash(self) -> str:
        """Deterministic short hash; suffix of the on-disk cache key."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:12]


def normalize_split_name(split: str) -> str:
    """Canonical split key — accepts ``val`` / ``validation`` etc."""
    s = split.lower().strip()
    if s in ("val", "validation"):
        return "val"
    if s in ("train", "training"):
        return "train"
    if s in ("test", "testing"):
        return "test"
    raise ValueError(f"unknown split: {split!r}")
