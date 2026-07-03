"""Shared types and helpers for the joint architecture + hardware search problem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.search.problem import ValidationResult
from mimarsinan.search.results import ObjectiveSpec


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


class JointHostContract:
    """Declares host members used across the joint mixins; empty at runtime."""

    data_provider_factory: Any
    device: Any
    input_shape: Tuple[int, ...]
    num_classes: int
    target_tq: int
    lr: float
    search_mode: str
    builder_factory: BuilderFactory
    validate_fn: Optional[ValidateFn]
    constraint_fn: Optional[ConstraintFn]
    fixed_model_config: Optional[Dict[str, Any]]
    fixed_platform_constraints: Optional[Dict[str, Any]]
    accuracy_seed: int
    warmup_fraction: float
    training_batch_size: Optional[int]
    accuracy_evaluator: str
    extrapolation_num_train_epochs: int
    extrapolation_num_checkpoints: int
    extrapolation_target_epochs: int
    _cache: Dict[str, Dict[str, float]]
    _hw_only_cache: Optional[HwOnlyCache]
    _validation_cache: Dict[str, ValidationEntry]
    _validation_errors: Dict[str, ValidationResult]

    if TYPE_CHECKING:

        @property
        def objectives(self) -> Sequence[ObjectiveSpec]: ...

        def validate_detailed(self, configuration: Dict) -> ValidationResult: ...

        def _penalty_objectives(self) -> Dict[str, float]: ...

        def _ensure_hw_only_cache(self) -> HwOnlyCache: ...

        def _build_raw_model(self, model_config: Dict, pcfg: Dict) -> Tuple[Any, float]: ...

        def _ensure_mapper_repr(self, model: Any) -> Any: ...

        def _collect_softcores(
            self, model: Any, pcfg: Dict,
        ) -> Tuple[List[LayoutSoftCoreSpec], int]: ...

        def _compute_hw_objectives(
            self,
            softcores: List[LayoutSoftCoreSpec],
            pcfg: Dict,
            total_params: float,
            host_side_segment_count: int,
        ) -> Tuple[Optional[Dict[str, float]], Optional[str]]: ...
