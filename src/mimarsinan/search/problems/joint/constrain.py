"""The DECLARED-constraint channel of the joint architecture + hardware search.

A constraint shapes a region the optimizer can SEE; it is never an exception a
run discovers after it has picked a winner. The declared search space is the
first such region: the vector encodings cannot leave it (``decode`` clips and
snaps), but a JSON proposal — an LLM's, a plan codec's, a hand-written
record's — can name any chip, so every entrance asks ``_domain_failure`` before
anything is built or charged. This channel is also one of the seams the TS1
accountant charges, so it names itself when it asks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.mapping.verification.onchip_fraction import (
    estimate_onchip_fraction,
)
from mimarsinan.search.constraints import ConstraintReport, onchip_floor_violation
from mimarsinan.search.option_axes import OptionAxis
from mimarsinan.search.search_space_description import CORE_DIM_GRANULARITY

from .types import (
    CONSTRAINT_CHANNEL,
    DOMAIN_PHASE,
    REPLICATE_KEY,
    CandidateFailure,
    CandidatePlatformError,
    JointHostContract,
)

logger = logging.getLogger(__name__)

_TOP_LEVEL_KEYS = frozenset({
    "model_config", "platform_constraints", "deployment_options", REPLICATE_KEY,
})
_CORE_DIMS = ("max_axons", "max_neurons", "count")


class JointConstrainMixin(JointHostContract):
    """Declared-constraint scoring for :class:`JointArchHwProblem`."""

    def _domain_failure(self, configuration: Dict) -> Optional[CandidateFailure]:
        """Does this RESOLVED candidate lie outside the declared search space?"""
        violations = domain_violations(
            configuration,
            searches_hw=self._searches_hw,
            num_core_types=int(self.num_core_types),
            core_axons_bounds=self.core_axons_bounds,
            core_neurons_bounds=self.core_neurons_bounds,
            core_count_bounds=self.core_count_bounds,
            target_tq=int(self.target_tq),
            searches_model=self._searches_model,
            arch_options=self.arch_options,
            fixed_model_config=self.fixed_model_config,
            option_axes=self.option_axes,
            declared_platform=self.fixed_platform_constraints or {},
        )
        if not violations:
            return None
        return CandidateFailure(
            phase=DOMAIN_PHASE,
            message="outside the declared search space: " + "; ".join(violations),
        )

    def onchip_fraction(self, configuration: Dict) -> float:
        """The share of this candidate's parameters that would sit on chip cores.

        Measured through the deployment's OWN estimator, under the CANDIDATE's
        placement — the NeuralOps/ComputeOps boundary the encoder lands on.
        """
        placement = self.candidate_encoding_placement(configuration)
        model, _params = self._candidate_model(
            configuration.get("model_config") or {},
            configuration.get("platform_constraints") or {},
            placement,
        )
        return estimate_onchip_fraction(
            model,
            tuple(self.input_shape),
            int(self.num_classes),
            encoding_placement=placement,
            metric="params",
        ).fraction

    def onchip_constraint(self, configuration: Dict) -> Optional[ConstraintReport]:
        """The on-chip floor report for this candidate, or None when satisfied."""
        if self.onchip_min_fraction <= 0.0:
            return None
        return onchip_floor_violation(
            fraction=self.onchip_fraction(configuration),
            floor=float(self.onchip_min_fraction),
        )

    def constraint_violation(self, configuration: Dict) -> float:
        try:
            resolved = self._resolved_configuration(configuration)
        except CandidatePlatformError:
            return 1.0
        declared = self._declared_violation(resolved, configuration)
        if declared is not None:
            return declared
        if not self.validate_detailed(resolved, channel=CONSTRAINT_CHANNEL).is_valid:
            return 1.0
        report = self.onchip_constraint(resolved)
        if report is None:
            return 0.0
        self._constraint_census[report.constraint] = (
            self._constraint_census.get(report.constraint, 0) + 1
        )
        logger.warning(
            "[JointArchHwProblem] candidate violates %s: %s",
            report.constraint, report.detail,
        )
        return report.violation

    def _declared_violation(
        self, resolved: Dict, configuration: Dict,
    ) -> Optional[float]:
        """The caller's own constraint reading, or None when it does not object.

        A cheap predicate over the DECLARATION: it builds nothing, so it costs
        the run no evaluation, and its own breakage is the candidate's problem
        rather than the search's.
        """
        if self.constraint_fn is None:
            return None
        try:
            cv = float(self.constraint_fn(
                resolved["model_config"],
                resolved["platform_constraints"],
                self.input_shape,
            ))
        except Exception as exc:
            logger.warning(
                "[JointArchHwProblem] constraint_fn failed (%s: %s) for candidate "
                "%.500s; recording constraint violation 1e6",
                type(exc).__name__, exc, configuration, exc_info=True,
            )
            return 1e6
        return cv if cv > 0 else None

    def constraint_census(self) -> Dict[str, int]:
        """How many candidates each declared constraint has rejected so far."""
        return dict(self._constraint_census)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _core_triples(cores: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int, int]]:
    return [
        (int(core["max_axons"]), int(core["max_neurons"]), int(core.get("count", 1)))
        for core in cores
    ]


def domain_violations(
    configuration: Mapping[str, Any],
    *,
    searches_hw: bool,
    num_core_types: int,
    core_axons_bounds: Tuple[int, int],
    core_neurons_bounds: Tuple[int, int],
    core_count_bounds: Tuple[int, int],
    target_tq: int,
    searches_model: bool,
    arch_options: Sequence[Tuple[str, Sequence[Any]]],
    fixed_model_config: Optional[Mapping[str, Any]],
    option_axes: Sequence[OptionAxis],
    declared_platform: Mapping[str, Any],
) -> List[str]:
    """Every way a RESOLVED candidate lies outside the declaration (empty = inside)."""
    problems: List[str] = []
    unknown = sorted(set(configuration) - _TOP_LEVEL_KEYS)
    if unknown:
        problems.append(f"unsupported candidate keys {unknown}")
    replicate = configuration.get(REPLICATE_KEY, 0)
    if not _is_int(replicate) or replicate < 0:
        problems.append(f"{REPLICATE_KEY} must be a non-negative integer, got {replicate!r}")
    options = configuration.get("deployment_options") or {}
    problems.extend(_platform_violations(
        configuration.get("platform_constraints") or {},
        searches_hw=searches_hw, num_core_types=num_core_types,
        bounds={
            "max_axons": (core_axons_bounds, CORE_DIM_GRANULARITY),
            "max_neurons": (core_neurons_bounds, CORE_DIM_GRANULARITY),
            "count": (core_count_bounds, 1),
        },
        target_tq=target_tq, option_axes=option_axes,
        declared_platform=declared_platform, options=options,
    ))
    problems.extend(_model_violations(
        configuration.get("model_config") or {}, searches_model=searches_model,
        arch_options=arch_options, fixed_model_config=fixed_model_config,
    ))
    problems.extend(_option_violations(options, option_axes))
    return problems


def _platform_violations(
    pcfg: Mapping[str, Any], *, searches_hw: bool, num_core_types: int,
    bounds: Mapping[str, Tuple[Tuple[int, int], int]], target_tq: int,
    option_axes: Sequence[OptionAxis], declared_platform: Mapping[str, Any],
    options: Mapping[str, Any],
) -> List[str]:
    problems: List[str] = []
    cores = list(pcfg.get("cores") or [])
    if searches_hw:
        if len(cores) != int(num_core_types):
            problems.append(
                f"{len(cores)} core types proposed; the declaration searches "
                f"exactly {int(num_core_types)}"
            )
        for index, core in enumerate(cores):
            for dim in _CORE_DIMS:
                value = core.get(dim, 1 if dim == "count" else None)
                (low, high), step = bounds[dim]
                if not _is_int(value):
                    problems.append(f"core {index} {dim} must be an integer, got {value!r}")
                elif not int(low) <= value <= int(high):
                    problems.append(
                        f"core {index} {dim}={value} outside the declared [{int(low)}, {int(high)}]"
                    )
                elif value % step:
                    problems.append(f"core {index} {dim}={value} is not a multiple of {step}")
    elif cores and _core_triples(cores) != _core_triples(declared_platform.get("cores") or []):
        problems.append("the declaration does not search the chip; cores may not move")

    if "target_tq" in pcfg and int(pcfg["target_tq"]) != int(target_tq):
        problems.append(f"target_tq={pcfg['target_tq']} moves the declared {int(target_tq)}")
    searched = {axis.key for axis in option_axes}
    if "weight_bits" in pcfg:
        declared = declared_platform.get("weight_bits")
        if "weight_bits" in searched:
            if "weight_bits" in options and int(pcfg["weight_bits"]) != int(options["weight_bits"]):
                problems.append("platform weight_bits disagrees with the searched deployment option")
        elif declared is not None and int(pcfg["weight_bits"]) != int(declared):
            problems.append(f"weight_bits={pcfg['weight_bits']} moves the declared {int(declared)}")
    return problems


def _model_violations(
    model_config: Mapping[str, Any], *, searches_model: bool,
    arch_options: Sequence[Tuple[str, Sequence[Any]]],
    fixed_model_config: Optional[Mapping[str, Any]],
) -> List[str]:
    if searches_model:
        return [
            f"model_config.{key}={model_config[key]!r} is not one of the declared {list(choices)}"
            for key, choices in arch_options
            if key in model_config and model_config[key] not in tuple(choices)
        ]
    # An EMPTY overlay means the fixed model, as an empty platform overlay
    # means the declared chip (the fidelity twin asks exactly that way).
    if fixed_model_config is not None and model_config and dict(model_config) != dict(fixed_model_config):
        return ["the declaration does not search the model; model_config may not move"]
    return []


def _option_violations(
    options: Mapping[str, Any], option_axes: Sequence[OptionAxis],
) -> List[str]:
    problems: List[str] = []
    axes = {axis.key: axis for axis in option_axes}
    for key, value in options.items():
        axis = axes.get(key)
        if axis is None:
            problems.append(f"deployment option {key!r} is not a declared search axis")
        elif axis.is_choice:
            if value not in axis.choices:
                problems.append(f"deployment option {key}={value!r} is not one of {list(axis.choices)}")
        elif not (_is_int(value) or isinstance(value, float)) or (axis.integral and not _is_int(value)):
            problems.append(f"deployment option {key}={value!r} is not a number of the axis' kind")
        elif not axis.lower <= float(value) <= axis.upper:
            problems.append(
                f"deployment option {key}={value} outside the declared [{axis.lower}, {axis.upper}]"
            )
    return problems
