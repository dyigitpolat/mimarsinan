"""Generic joint architecture + hardware search problem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


from mimarsinan.deployment_record.objectives import ObjectiveSpecV2
from mimarsinan.search.optimizers.budget import EvaluationBudget
from mimarsinan.search.problems.encoded_problem import EncodedProblem
from mimarsinan.search.option_axes import OptionAxis, candidate_option
from mimarsinan.search.problem import ValidationResult
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.search.results import ObjectiveSpec, resolve_active_specs
from mimarsinan.search.search_space_description import (
    DEFAULT_CORE_AXONS_BOUNDS,
    DEFAULT_CORE_COUNT_BOUNDS,
    DEFAULT_CORE_NEURONS_BOUNDS,
)

from .constrain import JointConstrainMixin
from .encoding import JointEncodingMixin
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
    effective_max_dims,
    json_key,
)
from .validate import JointValidateMixin


@dataclass
class JointArchHwProblem(
    JointEncodingMixin,
    JointValidateMixin,
    JointConstrainMixin,
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
    #: The run's DECLARED pruning [P] — never searched (accuracy impact is
    #: unmodeled at candidate time; option_axes refuses the keys by name).
    #: Candidate models are shrunk to the shapes deployment will map:
    #: ``prune_sparsity`` exactly (the deployed chain shrink itself),
    #: ``pruning``/``pruning_fraction`` by the mask floor-count bound.
    pruning_fraction: float = 0.0
    pruning: bool = False
    prune_sparsity: float = 0.0
    prune_criterion: str = "row_col_l1"
    firing_mode: str = "Default"
    #: [E1] The firing semantics the executed-window rule branches on, from
    #: the run's config — so the candidate sizes the SAME wall the runner runs.
    spiking_mode: str = "lif"
    ttfs_cycle_schedule: str = "cascaded"
    per_hop_retiming: bool = False
    #: [H2] The run's pass-boundary transfer discipline (VERBATIM/COLLAPSE),
    #: resolved by the deployment's own rule; None leaves carry unpriced.
    pass_transfer: Optional[str] = None
    #: The deployment's ``encoding_layer_placement``. A candidate's layout is
    #: only the deployed model's layout if its encoder sits where deployment
    #: will put it, so the search resolves the SAME placement the run will.
    encoding_placement: str = "subsume"

    #: Deployment options promoted to decision variables (C3). Empty = the run's
    #: declaration stands and the encoding is byte-identical to before.
    option_axes: Tuple[OptionAxis, ...] = ()

    #: The on-chip parameter floor this search must respect. 0 = no constraint.
    onchip_min_fraction: float = 0.0

    #: [TS1] The run's evaluation accountant, set by the step or a campaign
    #: driver. The problem only METERS at its cache seam; stopping is the
    #: driver's decision at its own boundary. None = unmetered, and every
    #: evaluation is byte-identical to a run without an accountant.
    evaluation_budget: Optional[EvaluationBudget] = None

    _cache: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    _hw_only_cache: Dict[str, HwOnlyCache] = field(
        default_factory=dict, init=False
    )
    _validation_cache: Dict[str, ValidationEntry] = field(default_factory=dict, init=False)
    _validation_errors: Dict[str, ValidationResult] = field(default_factory=dict, init=False)
    _resolved_base: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _constraint_census: Dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    #: [H4] Per-PROBLEM constants the eval loop was re-deriving per candidate
    #: (measured: 5 resolutions + 4 probe builds per eval, ~half the wall).
    _active_specs_cache: Optional[Tuple[ObjectiveSpecV2, ...]] = field(
        default=None, init=False, repr=False
    )
    _fragment_needs_cache: Dict[str, bool] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def _searches_model(self) -> bool:
        return self.search_mode in ("model", "joint")

    @property
    def _searches_hw(self) -> bool:
        return self.search_mode in ("hardware", "joint")

    @property
    def active_specs(self) -> Sequence[ObjectiveSpecV2]:
        """The ACTIVE registry axes — what an evaluation of this problem produces.

        Gated on the run's OWN physics (read off the resolved platform, the same
        declaration the deployment carries), so a candidate can never be scored on
        an axis its target cannot back.
        """
        if self._active_specs_cache is not None:
            return self._active_specs_cache
        object.__setattr__(self, "_active_specs_cache", tuple(resolve_active_specs(
            self.search_mode, self.active_objective_names or None,
            physics=self.candidate_physics,
            activity_factor=(self.fixed_platform_constraints or {}).get(
                "activity_factor", 0.0,
            ),
        )))
        assert self._active_specs_cache is not None
        return self._active_specs_cache

    @property
    def candidate_physics(self) -> Optional[PlatformPhysics]:
        """The physics every candidate of this problem is priced with."""
        base = self.fixed_platform_constraints or {}
        payload = base.get("platform_physics_resolved")
        return None if payload is None else PlatformPhysics.from_dict(payload)

    def candidate_encoding_placement(self, configuration: Mapping[str, Any]) -> str:
        """Where THIS candidate puts its encoder — searched value, else declared."""
        return str(candidate_option(
            configuration, "encoding_layer_placement", self.encoding_placement,
        ))

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


__all__ = ["JointArchHwProblem", "effective_max_dims", "json_key"]
