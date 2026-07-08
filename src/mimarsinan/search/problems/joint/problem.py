"""Generic joint architecture + hardware search problem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.platform.coalescing import CANONICAL_KEY, normalize_coalescing_config
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.problem import ValidationResult
from mimarsinan.search.results import ObjectiveSpec, resolve_active_objectives
from mimarsinan.search.search_space_description import (
    DEFAULT_CORE_AXONS_BOUNDS,
    DEFAULT_CORE_COUNT_BOUNDS,
    DEFAULT_CORE_NEURONS_BOUNDS,
)

from .evaluate import JointEvaluateMixin
from .layout_hook import JointLayoutMixin
from .types import (
    BuilderFactory,
    ConstraintFn,
    HwOnlyCache,
    ModelConfigAssembler,
    ValidateFn,
    ValidationEntry,
    clip_int,
    effective_max_dims,
    json_key,
)
from .validate import JointValidateMixin


@dataclass
class JointArchHwProblem(
    JointValidateMixin,
    JointLayoutMixin,
    JointEvaluateMixin,
    EncodedProblem[Dict[str, Any]],
):
    """Model-agnostic search problem with variable geometry."""

    data_provider_factory: Any
    device: Any
    input_shape: Tuple[int, ...]
    num_classes: int
    target_tq: int
    lr: float

    search_mode: str = "joint"

    builder_factory: BuilderFactory = None  # type: ignore[assignment]
    arch_options: Sequence[Tuple[str, Sequence[Any]]] = ()
    model_config_assembler: ModelConfigAssembler = None  # type: ignore[assignment]
    validate_fn: Optional[ValidateFn] = None
    constraint_fn: Optional[ConstraintFn] = None

    fixed_model_config: Optional[Dict[str, Any]] = None
    fixed_platform_constraints: Optional[Dict[str, Any]] = None

    active_objective_names: Sequence[str] = ()

    num_core_types: int = 1
    core_axons_bounds: Tuple[int, int] = DEFAULT_CORE_AXONS_BOUNDS
    core_neurons_bounds: Tuple[int, int] = DEFAULT_CORE_NEURONS_BOUNDS
    core_count_bounds: Tuple[int, int] = DEFAULT_CORE_COUNT_BOUNDS

    accuracy_seed: int = 0
    warmup_fraction: float = 0.10
    training_batch_size: Optional[int] = None
    accuracy_evaluator: str = "extrapolating"
    extrapolation_num_train_epochs: int = 1
    extrapolation_num_checkpoints: int = 5
    extrapolation_target_epochs: int = 10
    pruning_fraction: float = 0.0

    _cache: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    _hw_only_cache: Optional[HwOnlyCache] = field(default=None, init=False)
    _validation_cache: Dict[str, ValidationEntry] = field(default_factory=dict, init=False)
    _validation_errors: Dict[str, ValidationResult] = field(default_factory=dict, init=False)

    @property
    def _searches_model(self) -> bool:
        return self.search_mode in ("model", "joint")

    @property
    def _searches_hw(self) -> bool:
        return self.search_mode in ("hardware", "joint")

    @property
    def _n_arch_vars(self) -> int:
        return len(self.arch_options) if self._searches_model else 0

    @property
    def _n_hw_vars(self) -> int:
        return (3 * int(self.num_core_types)) if self._searches_hw else 0

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        if self.active_objective_names:
            return resolve_active_objectives(self.search_mode, self.active_objective_names)
        return resolve_active_objectives(self.search_mode)

    @property
    def n_var(self) -> int:
        return self._n_arch_vars + self._n_hw_vars

    @property
    def xl(self) -> np.ndarray:
        xl: List[float] = []
        if self._searches_model:
            xl.extend([0.0] * len(self.arch_options))
        if self._searches_hw:
            for _ in range(int(self.num_core_types)):
                xl.extend([
                    float(self.core_axons_bounds[0]),
                    float(self.core_neurons_bounds[0]),
                    float(self.core_count_bounds[0]),
                ])
        return np.array(xl, dtype=float)

    @property
    def xu(self) -> np.ndarray:
        xu: List[float] = []
        if self._searches_model:
            xu.extend([float(len(opts) - 1) for _, opts in self.arch_options])
        if self._searches_hw:
            for _ in range(int(self.num_core_types)):
                xu.extend([
                    float(self.core_axons_bounds[1]),
                    float(self.core_neurons_bounds[1]),
                    float(self.core_count_bounds[1]),
                ])
        return np.array(xu, dtype=float)

    def _decode_arch(self, x: np.ndarray, offset: int) -> Dict[str, Any]:
        raw_arch: Dict[str, Any] = {}
        for i, (key, options) in enumerate(self.arch_options):
            idx = clip_int(x[offset + i], 0, len(options) - 1)
            raw_arch[key] = options[idx]
        return self.model_config_assembler(raw_arch)

    def _decode_hw(self, x: np.ndarray, offset: int) -> Dict[str, Any]:
        core_types: List[Dict[str, int]] = []
        idx = offset
        for _ in range(int(self.num_core_types)):
            ax = clip_int(x[idx], int(self.core_axons_bounds[0]), int(self.core_axons_bounds[1]))
            neu = clip_int(
                x[idx + 1], int(self.core_neurons_bounds[0]), int(self.core_neurons_bounds[1]),
            )
            count = clip_int(x[idx + 2], int(self.core_count_bounds[0]), int(self.core_count_bounds[1]))
            idx += 3
            ax = max(8, int(round(ax / 8)) * 8)
            neu = max(8, int(round(neu / 8)) * 8)
            core_types.append({
                "max_axons": ax,
                "max_neurons": neu,
                "count": count,
            })

        pcfg: Dict[str, Any] = {
            "cores": core_types,
            "target_tq": int(self.target_tq),
            "weight_bits": 8,
        }
        if self.fixed_platform_constraints:
            for key in ("allow_scheduling", "allow_neuron_splitting", "has_bias"):
                if key in self.fixed_platform_constraints:
                    pcfg.setdefault(key, self.fixed_platform_constraints[key])
            if CANONICAL_KEY in self.fixed_platform_constraints:
                pcfg.setdefault(
                    CANONICAL_KEY,
                    bool(self.fixed_platform_constraints[CANONICAL_KEY]),
                )
        normalize_coalescing_config(pcfg)
        return pcfg

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.array(x, dtype=float).flatten()
        if x.shape[0] != self.n_var:
            raise ValueError(f"Expected x of length {self.n_var}, got {x.shape}")

        offset = 0

        if self._searches_model:
            model_config = self._decode_arch(x, offset)
            offset += self._n_arch_vars
        else:
            model_config = dict(self.fixed_model_config or {})

        if self._searches_hw:
            platform_constraints = self._decode_hw(x, offset)
        else:
            platform_constraints = dict(self.fixed_platform_constraints or {})
            normalize_coalescing_config(platform_constraints)

        return {
            "model_config": model_config,
            "platform_constraints": platform_constraints,
        }


__all__ = ["JointArchHwProblem", "effective_max_dims", "json_key"]
