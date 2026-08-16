"""Shared types and helpers for the joint architecture + hardware search problem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from mimarsinan.deployment_record.objectives import (
    CandidateStaticView,
    ObjectiveSpecV2,
)
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.verification.layout_verification_types import (
    LayoutVerificationStats,
)
from mimarsinan.search.option_axes import OptionAxis
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

#: Resolve a candidate platform: the deployment's platform resolution applied to
#: the declared platform overlaid with the candidate's searched declarations.
PlatformResolver = Callable[[Mapping[str, Any]], Dict[str, Any]]


@dataclass
class HwOnlyCache:
    """The candidate-INDEPENDENT model fixture of a hardware-only search.

    Only the model is candidate-independent: its LAYOUT is re-derived per
    candidate, because tiling depends on the candidate's core geometry. The
    model-side mapper representation is not — converting a torch model into
    mapper form is a function of the model alone — so it is memoized here the
    first time a candidate needs it.
    """

    model: Any
    total_params: float
    mapper_repr: Any = None
    # (params, macs) OnchipFractionEstimate pair — a function of the fixed
    # model and placement alone, so computed once like the mapper repr.
    onchip_census: Any = None


@dataclass
class ValidationEntry:
    """A validated candidate: its static view, plus the model accuracy may need."""

    model: Any
    view: CandidateStaticView


@dataclass(frozen=True)
class CandidateLayout:
    """One candidate, laid out: its chip, its softcores, its packing, its view.

    :class:`ValidationEntry` keeps only what an EVALUATION needs (the view);
    an introspection caller — the compilagent layout backend — needs the
    softcores themselves. Both come off the one resolution path, so the numbers
    an agent is shown cannot drift from the numbers the search scores.
    """

    platform: Dict[str, Any]
    softcores: List[LayoutSoftCoreSpec]
    host_side_segment_count: int
    stats: LayoutVerificationStats
    view: CandidateStaticView
    # LayoutNocFragments when a NoC axis asked for them (else None).
    noc: Optional[Any] = None


class CandidatePlatformError(ValueError):
    """A candidate DECLARED a platform that does not resolve into a chip.

    Candidate-scoped by construction: the run's own declared platform is
    resolved first, so a failure past that point is the candidate's declaration
    (zero cores, a tile grid too small for its capacity, a retired key), and it
    is scored — never allowed to abort a whole search.
    """


#: Candidate-scoped failure phases, in the order the pipeline meets them.
STRUCTURAL_PHASE = "structural"
MODEL_BUILD_PHASE = "model_build"
HW_CONVERSION_PHASE = "hw_conversion"
HW_PACKING_PHASE = "hw_packing"


@dataclass(frozen=True)
class CandidateFailure:
    """One candidate-scoped failure, rendered per boundary (invalid vs typed raise)."""

    phase: str
    message: str
    cause: Optional[BaseException] = None

    def as_validation_result(self) -> ValidationResult:
        return ValidationResult(
            is_valid=False, error_message=self.message, failure_phase=self.phase,
        )


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
    encoding_placement: str
    pruning_fraction: float
    pruning: bool
    prune_sparsity: float
    prune_criterion: str
    firing_mode: str
    spiking_mode: str
    ttfs_cycle_schedule: str
    per_hop_retiming: bool
    option_axes: Tuple[OptionAxis, ...]
    arch_options: Sequence[Tuple[str, Sequence[Any]]]
    model_config_assembler: ModelConfigAssembler
    num_core_types: int
    core_axons_bounds: Tuple[int, int]
    core_neurons_bounds: Tuple[int, int]
    core_count_bounds: Tuple[int, int]
    validate_fn: Optional[ValidateFn]
    constraint_fn: Optional[ConstraintFn]
    fixed_model_config: Optional[Dict[str, Any]]
    platform_resolver: Optional[PlatformResolver]
    accuracy_seed: int
    warmup_fraction: float
    training_batch_size: Optional[int]
    accuracy_evaluator: str
    extrapolation_num_train_epochs: int
    extrapolation_num_checkpoints: int
    extrapolation_target_epochs: int
    _cache: Dict[str, Dict[str, float]]
    _hw_only_cache: Dict[str, HwOnlyCache]
    _validation_cache: Dict[str, ValidationEntry]
    _validation_errors: Dict[str, ValidationResult]
    onchip_min_fraction: float
    _constraint_census: Dict[str, int]

    if TYPE_CHECKING:

        @property
        def objectives(self) -> Sequence[ObjectiveSpec]: ...

        @property
        def active_specs(self) -> Sequence[ObjectiveSpecV2]: ...

        @property
        def _searches_model(self) -> bool: ...

        @property
        def _searches_hw(self) -> bool: ...

        @property
        def fixed_platform_constraints(self) -> Optional[Dict[str, Any]]: ...

        def resolve_candidate_platform(
            self, overlay: Mapping[str, Any],
        ) -> Dict[str, Any]: ...

        def candidate_encoding_placement(
            self, configuration: Mapping[str, Any],
        ) -> str: ...

        def _resolved_configuration(
            self, configuration: Dict[str, Any],
        ) -> Dict[str, Any]: ...

        def validate_detailed(self, configuration: Dict) -> ValidationResult: ...

        def _penalty_objectives(self) -> Dict[str, float]: ...

        def _requires_fragment(self, fragment: str) -> bool: ...

        def _ensure_hw_only_cache(self, placement: str) -> HwOnlyCache: ...

        def _build_raw_model(
            self, model_config: Dict, pcfg: Dict, placement: str,
        ) -> Tuple[Any, float]: ...

        def _candidate_model(
            self, mc: Dict, pcfg: Dict, placement: str,
        ) -> Tuple[Any, float]: ...

        def _ensure_mapper_repr(self, model: Any, placement: str) -> Any: ...

        def onchip_constraint(self, configuration: Dict) -> Optional[Any]: ...

        def _collect_softcores(
            self, model: Any, pcfg: Dict, *, collect_census: bool = False,
        ) -> Tuple[List[LayoutSoftCoreSpec], int, Optional[Any]]: ...

        def _pack_candidate(
            self, softcores: List[LayoutSoftCoreSpec], pcfg: Dict,
        ) -> Tuple[LayoutVerificationStats, Optional[str]]: ...

        def _packing_failure(
            self,
            stats: LayoutVerificationStats,
            error: Optional[str],
            softcores: List[LayoutSoftCoreSpec],
            pcfg: Dict,
        ) -> CandidateFailure: ...

        def _static_view(
            self,
            stats: LayoutVerificationStats,
            pcfg: Dict,
            total_params: float,
            host_side_segment_count: int,
            census: Any = None,
            noc: Any = None,
            latency_steps: Any = None,
        ) -> CandidateStaticView: ...

        @property
        def stage_semantics(self) -> Any: ...

        @staticmethod
        def _make_core_types(pcfg: Dict) -> List[Any]: ...

        def _layoutless_view(
            self, pcfg: Dict, total_params: float,
        ) -> CandidateStaticView: ...

        def _onchip_census(self, model: Any, placement: str) -> Any: ...

        def _resolve_entry(
            self, mc: Dict, pcfg: Dict, placement: str,
        ) -> Tuple[Optional[ValidationEntry], Optional[CandidateFailure]]: ...
