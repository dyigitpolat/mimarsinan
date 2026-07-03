"""Validation and constraint checking for joint architecture + hardware search."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch

from mimarsinan.mapping.platform.coalescing import CoalescingConfigError, normalize_coalescing_config
from mimarsinan.search.problem import ValidationResult
from mimarsinan.search.results import ACCURACY_OBJECTIVE_NAME

from .types import VALIDATION_CACHE_MAX_SIZE, JointHostContract, ValidationEntry, json_key

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

    def _record_invalid_exception(
        self, key: str, what: str, exc: Exception, phase: str,
    ) -> ValidationResult:
        message = f"{what}: {type(exc).__name__}: {exc}"
        logger.warning(
            "[JointArchHwProblem] %s for candidate %.500s; marking invalid",
            message, key, exc_info=True,
        )
        return self._record_invalid(key, message, phase)

    def validate(self, configuration: Dict) -> bool:
        return self.validate_detailed(configuration).is_valid

    def validate_detailed(self, configuration: Dict) -> ValidationResult:
        """Full feasibility check: structural → model build → HW packing."""
        key = json_key(configuration)

        if key in self._validation_cache:
            return ValidationResult(is_valid=True)

        if key in self._validation_errors:
            return self._validation_errors[key]

        if key in self._cache:
            return ValidationResult(is_valid=True)

        mc = configuration.get("model_config", {})
        pcfg = dict(configuration.get("platform_constraints", {}))
        try:
            normalize_coalescing_config(pcfg)
        except CoalescingConfigError as exc:
            return self._record_invalid(key, str(exc), "structural")

        try:
            if self.validate_fn is not None:
                if not self.validate_fn(mc, pcfg, self.input_shape):
                    return self._record_invalid(
                        key,
                        "Structural validation failed (validate_fn returned False)",
                        "structural",
                    )
        except Exception as exc:
            return self._record_invalid_exception(
                key, "Structural validation error", exc, "structural",
            )

        torch.manual_seed(int(self.accuracy_seed))
        np.random.seed(int(self.accuracy_seed))

        if self.search_mode == "hardware":
            return self._validate_hw_only(key, pcfg)
        return self._validate_model_or_joint(key, mc, pcfg)

    def _validate_hw_only(self, key: str, pcfg: Dict) -> ValidationResult:
        # The fixed model does not depend on the candidate: a build failure is
        # problem-level breakage, not candidate infeasibility — fail loud.
        cache = self._ensure_hw_only_cache()

        hw_obj, error = self._compute_hw_objectives(
            cache.softcores, pcfg, cache.total_params, cache.host_side_segment_count,
        )
        if hw_obj is None:
            return self._record_invalid(
                key, error or "HW bin-packing infeasible", "hw_packing",
            )

        self._validation_cache[key] = ValidationEntry(
            model=None, total_params=cache.total_params, hw_objectives=hw_obj,
        )
        self._evict_validation_cache()
        return ValidationResult(is_valid=True)

    def _validate_model_or_joint(
        self, key: str, mc: Dict, pcfg: Dict,
    ) -> ValidationResult:
        active_names = {spec.name for spec in self.objectives}
        hw_names = active_names - {ACCURACY_OBJECTIVE_NAME}

        try:
            raw_model, total_params = self._build_raw_model(mc, pcfg)
        except Exception as exc:
            return self._record_invalid_exception(
                key, "Model build failed", exc, "model_build",
            )

        if not hw_names:
            self._validation_cache[key] = ValidationEntry(
                model=raw_model, total_params=total_params, hw_objectives={},
            )
            self._evict_validation_cache()
            return ValidationResult(is_valid=True)

        try:
            mapped_model = self._ensure_mapper_repr(raw_model)
        except Exception as exc:
            return self._record_invalid_exception(
                key, "HW conversion failed", exc, "hw_conversion",
            )

        try:
            softcores, host_segments = self._collect_softcores(mapped_model, pcfg)
        except Exception as exc:
            return self._record_invalid_exception(
                key, "Softcore collection failed", exc, "hw_conversion",
            )

        hw_obj, error = self._compute_hw_objectives(
            softcores, pcfg, total_params, host_segments,
        )
        if hw_obj is None:
            return self._record_invalid(
                key, error or "HW bin-packing infeasible", "hw_packing",
            )

        self._validation_cache[key] = ValidationEntry(
            model=raw_model, total_params=total_params, hw_objectives=hw_obj,
        )
        self._evict_validation_cache()
        return ValidationResult(is_valid=True)

    def _evict_validation_cache(self) -> None:
        while len(self._validation_cache) > VALIDATION_CACHE_MAX_SIZE:
            oldest_key = next(iter(self._validation_cache))
            del self._validation_cache[oldest_key]

    def constraint_violation(self, configuration: Dict) -> float:
        try:
            if self.constraint_fn is not None:
                cv = float(self.constraint_fn(
                    configuration["model_config"],
                    configuration["platform_constraints"],
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
        return 0.0 if self.validate_detailed(configuration).is_valid else 1.0
