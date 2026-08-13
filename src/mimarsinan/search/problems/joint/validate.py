"""Validation and candidate-view resolution for joint architecture + hardware search."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from mimarsinan.search.option_axes import candidate_option
from mimarsinan.search.problem import CandidateInfeasibleError, ValidationResult

from .types import (
    HW_CONVERSION_PHASE,
    MODEL_BUILD_PHASE,
    STRUCTURAL_PHASE,
    VALIDATION_CACHE_MAX_SIZE,
    CandidateFailure,
    CandidateLayout,
    CandidatePlatformError,
    JointHostContract,
    ValidationEntry,
    json_key,
)

logger = logging.getLogger(__name__)


class JointValidateMixin(JointHostContract):
    """Feasibility validation for :class:`JointArchHwProblem`."""

    def _record_invalid(
        self, key: str, message: str, phase: str,
    ) -> ValidationResult:
        vr = ValidationResult(
            is_valid=False, error_message=message, failure_phase=phase,
        )
        self._validation_errors[key] = vr
        return vr

    @staticmethod
    def _failure(phase: str, what: str, exc: Exception) -> CandidateFailure:
        message = f"{what}: {type(exc).__name__}: {exc}"
        logger.warning(
            "[JointArchHwProblem] %s; candidate rejected", message, exc_info=True,
        )
        return CandidateFailure(phase=phase, message=message, cause=exc)

    def validate(self, configuration: Dict) -> bool:
        return self.validate_detailed(configuration).is_valid

    def validate_detailed(self, configuration: Dict) -> ValidationResult:
        """Full feasibility check: structural → model build → HW packing."""
        try:
            configuration = self._resolved_configuration(configuration)
        except CandidatePlatformError as exc:
            return self._record_invalid(
                json_key(configuration), str(exc), STRUCTURAL_PHASE,
            )

        key = json_key(configuration)
        if key in self._validation_cache:
            return ValidationResult(is_valid=True)
        if key in self._validation_errors:
            return self._validation_errors[key]
        if key in self._cache:
            return ValidationResult(is_valid=True)

        mc = configuration.get("model_config", {})
        pcfg = dict(configuration.get("platform_constraints", {}))

        structural = self._structural_failure(mc, pcfg)
        if structural is not None:
            return self._record_invalid(key, structural.message, structural.phase)

        entry, failure = self._resolve_entry(
            mc, pcfg, str(candidate_option(
                configuration, "encoding_layer_placement",
                self.encoding_placement,
            )),
        )
        if failure is not None:
            return self._record_invalid(key, failure.message, failure.phase)

        assert entry is not None
        self._validation_cache[key] = entry
        self._evict_validation_cache()
        return ValidationResult(is_valid=True)

    def _structural_failure(self, mc: Dict, pcfg: Dict) -> Optional[CandidateFailure]:
        """The caller-supplied structural check, if any."""
        if self.validate_fn is None:
            return None
        try:
            if not self.validate_fn(mc, pcfg, self.input_shape):
                return CandidateFailure(
                    phase=STRUCTURAL_PHASE,
                    message="Structural validation failed (validate_fn returned False)",
                )
        except Exception as exc:
            return self._failure(STRUCTURAL_PHASE, "Structural validation error", exc)
        return None

    def _resolve_model(
        self, mc: Dict, pcfg: Dict, placement: str,
    ) -> Tuple[Optional[Tuple[Any, float]], Optional[CandidateFailure]]:
        """The candidate's model, or the candidate-scoped reason there is none."""
        try:
            return self._candidate_model(mc, pcfg, placement), None
        except Exception as exc:
            if not self._searches_model:
                # The model does not depend on the candidate here: its failure
                # is problem-level breakage, not candidate infeasibility.
                raise
            return None, self._failure(MODEL_BUILD_PHASE, "Model build failed", exc)

    def _resolve_layout(
        self, model: Any, pcfg: Dict, total_params: float, placement: str,
    ) -> Tuple[Optional[CandidateLayout], Optional[CandidateFailure]]:
        """Lay a built model onto the candidate chip: conversion → softcores → packing."""
        try:
            mapped_model = self._ensure_mapper_repr(model, placement)
        except Exception as exc:
            return None, self._failure(HW_CONVERSION_PHASE, "HW conversion failed", exc)

        try:
            softcores, host_segments = self._collect_softcores(mapped_model, pcfg)
        except Exception as exc:
            return None, self._failure(
                HW_CONVERSION_PHASE, "Softcore collection failed", exc,
            )

        stats, error = self._pack_candidate(softcores, pcfg)
        if not stats.feasible:
            return None, self._packing_failure(stats, error, softcores, pcfg)

        return CandidateLayout(
            platform=pcfg,
            softcores=softcores,
            host_side_segment_count=host_segments,
            stats=stats,
            view=self._static_view(stats, pcfg, total_params, host_segments),
        ), None

    def _resolve_entry(
        self, mc: Dict, pcfg: Dict, placement: str,
    ) -> Tuple[Optional[ValidationEntry], Optional[CandidateFailure]]:
        """The ONE candidate-facts path: model → layout → static view.

        Every search mode walks it. Candidate-scoped breakage comes back as a
        :class:`CandidateFailure` the boundary renders (an invalid result, or a
        typed raise); problem-level breakage — a fixture the candidate does not
        influence — propagates untyped.
        """
        facts, failure = self._resolve_model(mc, pcfg, placement)
        if failure is not None:
            return None, failure
        assert facts is not None
        model, total_params = facts

        if not self._requires_fragment("layout"):
            return ValidationEntry(
                model=model, view=self._layoutless_view(pcfg, total_params),
            ), None

        layout, failure = self._resolve_layout(model, pcfg, total_params, placement)
        if failure is not None:
            return None, failure
        assert layout is not None
        return ValidationEntry(model=model, view=layout.view), None

    def candidate_layout(self, configuration: Dict) -> CandidateLayout:
        """This candidate, laid out on the chip a deployment would build for it.

        The introspection seam (a layout backend needs the softcores an
        evaluation throws away) walking the SAME path an evaluation walks, so
        the two cannot disagree. A candidate-scoped failure crosses typed; a
        problem-level one propagates untyped, exactly as in ``evaluate``.
        """
        resolved = self._resolved_configuration(configuration)
        pcfg = resolved["platform_constraints"]
        placement = str(candidate_option(
            resolved, "encoding_layer_placement", self.encoding_placement,
        ))
        facts, failure = self._resolve_model(
            resolved["model_config"], pcfg, placement,
        )
        if facts is not None:
            layout, failure = self._resolve_layout(
                facts[0], pcfg, facts[1], placement,
            )
            if layout is not None:
                return layout
        assert failure is not None
        raise CandidateInfeasibleError(failure.message) from failure.cause

    def _evict_validation_cache(self) -> None:
        while len(self._validation_cache) > VALIDATION_CACHE_MAX_SIZE:
            oldest_key = next(iter(self._validation_cache))
            del self._validation_cache[oldest_key]

    def constraint_violation(self, configuration: Dict) -> float:
        try:
            resolved = self._resolved_configuration(configuration)
        except CandidatePlatformError:
            return 1.0
        try:
            if self.constraint_fn is not None:
                cv = float(self.constraint_fn(
                    resolved["model_config"],
                    resolved["platform_constraints"],
                    self.input_shape,
                ))
                if cv > 0:
                    return cv
        except Exception as exc:
            logger.warning(
                "[JointArchHwProblem] constraint_fn failed (%s: %s) for candidate "
                "%.500s; recording constraint violation 1e6",
                type(exc).__name__, exc, configuration, exc_info=True,
            )
            return 1e6
        if not self.validate_detailed(resolved).is_valid:
            return 1.0
        # A DECLARED deployment constraint: infeasible is a region of the search
        # space the optimizer can see, not an exception a run discovers after it
        # has already picked a winner.
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

    def constraint_census(self) -> Dict[str, int]:
        """How many candidates each declared constraint has rejected so far."""
        return dict(self._constraint_census)
