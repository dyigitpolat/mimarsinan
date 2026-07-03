"""Shared types and helpers for the joint architecture + hardware search problem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params


def json_key(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(float(v))))))


def effective_max_dims(cores: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    """Return effective (max_axons, max_neurons) for IR tiling (legacy bias axon reserved)."""
    params = resolve_platform_mapping_params(cores)
    return params.effective_max_axons, params.effective_max_neurons


BuilderFactory = Callable[..., Any]
ModelConfigAssembler = Callable[[Dict[str, Any]], Dict[str, Any]]
ValidateFn = Callable[[Dict[str, Any], Dict[str, Any], Tuple[int, ...]], bool]
ConstraintFn = Callable[[Dict[str, Any], Dict[str, Any], Tuple[int, ...]], float]


@dataclass
class HwOnlyCache:
    softcores: List[Any]
    total_params: float
    host_side_segment_count: int


@dataclass
class ValidationEntry:
    model: Any
    total_params: float
    hw_objectives: Dict[str, float]


VALIDATION_CACHE_MAX_SIZE = 16
