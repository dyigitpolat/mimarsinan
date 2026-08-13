"""Generic joint architecture + hardware search problem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.deployment_record.objectives import ObjectiveSpecV2
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.problem import ValidationResult
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.search.results import ObjectiveSpec, resolve_active_specs
from mimarsinan.search.search_space_description import (
    CORE_DIM_GRANULARITY,
    DEFAULT_CORE_AXONS_BOUNDS,
    DEFAULT_CORE_COUNT_BOUNDS,
    DEFAULT_CORE_NEURONS_BOUNDS,
)

from .evaluate import JointEvaluateMixin
from .layout_hook import JointLayoutMixin
from .types import (
    BuilderFactory,
    CandidatePlatformError,
    ConstraintFn,
    HwOnlyCache,
    ModelConfigAssembler,
    PlatformResolver,
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
    #: The deployment's platform resolution, curried over this run's declared
    #: platform. Every candidate chip — decoded, LLM-declared, or the base — is
    #: this one function's output, so no candidate can differ from its
    #: deployed twin by a key nobody remembered to carry.
    platform_resolver: Optional[PlatformResolver] = None

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
    #: The deployment's ``encoding_layer_placement``. A candidate's layout is
    #: only the deployed model's layout if its encoder sits where deployment
    #: will put it, so the search resolves the SAME placement the run will.
    encoding_placement: str = "subsume"

    _cache: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    _hw_only_cache: Optional[HwOnlyCache] = field(default=None, init=False)
    _validation_cache: Dict[str, ValidationEntry] = field(default_factory=dict, init=False)
    _validation_errors: Dict[str, ValidationResult] = field(default_factory=dict, init=False)
    _resolved_base: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)

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
    def active_specs(self) -> Sequence[ObjectiveSpecV2]:
        """The ACTIVE registry axes — what an evaluation of this problem produces.

        Gated on the run's OWN physics (read off the resolved platform, the same
        declaration the deployment carries), so a candidate can never be scored on
        an axis its target cannot back.
        """
        return resolve_active_specs(
            self.search_mode, self.active_objective_names or None,
            physics=self.candidate_physics,
        )

    @property
    def candidate_physics(self) -> Optional[PlatformPhysics]:
        """The physics every candidate of this problem is priced with."""
        base = self.fixed_platform_constraints or {}
        payload = base.get("platform_physics_resolved")
        return None if payload is None else PlatformPhysics.from_dict(payload)

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        return tuple(ObjectiveSpec(s.name, s.goal) for s in self.active_specs)

    def _require_platform_resolver(self) -> PlatformResolver:
        if self.platform_resolver is None:
            raise ValueError(
                "the search problem has no platform_resolver: a candidate chip "
                "is the deployment's platform resolution of the declared "
                "platform plus the searched dimensions, and without the "
                "resolver it cannot be built"
            )
        return self.platform_resolver

    @property
    def fixed_platform_constraints(self) -> Optional[Dict[str, Any]]:
        """The declared platform, resolved — a candidate with an empty overlay."""
        if self.platform_resolver is None:
            return None
        if self._resolved_base is None:
            self._resolved_base = self.platform_resolver({})
        return self._resolved_base

    def resolve_candidate_platform(self, overlay: Mapping[str, Any]) -> Dict[str, Any]:
        """The ONE candidate-platform seam: the deployment resolution of *overlay*.

        Idempotent — re-resolving an already resolved platform returns it
        unchanged — so every entry point (decode, an LLM-declared candidate, the
        base itself) may pass through it unconditionally.
        """
        return self._require_platform_resolver()(overlay)

    def _resolved_configuration(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """A candidate whose platform is the chip a deployment would build.

        The run's DECLARED platform resolves first: a declared platform that
        does not resolve is problem-level breakage and aborts here — which is
        also what makes every remaining failure the candidate's own, and so
        scorable rather than fatal.
        """
        if self.fixed_platform_constraints is None:
            self._require_platform_resolver()
        try:
            platform = self.resolve_candidate_platform(
                configuration.get("platform_constraints") or {}
            )
        except ValueError as exc:
            raise CandidatePlatformError(
                f"candidate platform does not resolve into a chip: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return {**configuration, "platform_constraints": platform}

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

    @staticmethod
    def _snap_core_dim(value: int) -> int:
        """Core dimensions live on the declared grid, never between its lines."""
        snapped = int(round(value / CORE_DIM_GRANULARITY)) * CORE_DIM_GRANULARITY
        return max(CORE_DIM_GRANULARITY, snapped)

    def _decode_hw(self, x: np.ndarray, offset: int) -> Dict[str, Any]:
        """The searched dimensions, resolved into a chip by the deployment resolver."""
        base = self.fixed_platform_constraints
        if not base:
            raise ValueError(
                "hardware search requires a platform_resolver: the candidate "
                "chip is the declared platform re-resolved with the searched "
                "core dimensions"
            )
        # A searched core type declares dimensions only; every other core
        # property is the declared platform's — including whether the chip can
        # deliver a bias on-core, which the resolver stamps onto each core.
        base_cores = base.get("cores") or []
        hardware_bias = resolve_platform_mapping_params(base_cores).hardware_bias

        core_types: List[Dict[str, Any]] = []
        idx = offset
        for _ in range(int(self.num_core_types)):
            ax = clip_int(x[idx], int(self.core_axons_bounds[0]), int(self.core_axons_bounds[1]))
            neu = clip_int(
                x[idx + 1], int(self.core_neurons_bounds[0]), int(self.core_neurons_bounds[1]),
            )
            count = clip_int(x[idx + 2], int(self.core_count_bounds[0]), int(self.core_count_bounds[1]))
            idx += 3
            core_types.append({
                "max_axons": self._snap_core_dim(ax),
                "max_neurons": self._snap_core_dim(neu),
                "count": count,
                "has_bias": hardware_bias,
            })

        return self.resolve_candidate_platform({
            "cores": core_types,
            "target_tq": int(self.target_tq),
        })

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
            platform_constraints = self.resolve_candidate_platform({})

        return {
            "model_config": model_config,
            "platform_constraints": platform_constraints,
        }


__all__ = ["JointArchHwProblem", "effective_max_dims", "json_key"]
