"""Validation and constraint checking for joint architecture + hardware search."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.mapping.platform.coalescing import CoalescingConfigError, normalize_coalescing_config
from mimarsinan.search.problem import ValidationResult
from mimarsinan.search.results import ACCURACY_OBJECTIVE_NAME

from .types import VALIDATION_CACHE_MAX_SIZE, ValidationEntry, json_key


class JointValidateMixin:
    """Feasibility validation for :class:`JointArchHwProblem`."""

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
            vr = ValidationResult(
                is_valid=False,
                error_message=str(exc),
                failure_phase="structural",
            )
            self._validation_errors[key] = vr
            return vr

        try:
            if self.validate_fn is not None:
                if not self.validate_fn(mc, pcfg, self.input_shape):
                    vr = ValidationResult(
                        is_valid=False,
                        error_message="Structural validation failed (validate_fn returned False)",
                        failure_phase="structural",
                    )
                    self._validation_errors[key] = vr
                    return vr
        except Exception as exc:
            vr = ValidationResult(
                is_valid=False,
                error_message=f"Structural validation error: {type(exc).__name__}: {exc}",
                failure_phase="structural",
            )
            self._validation_errors[key] = vr
            return vr

        torch.manual_seed(int(self.accuracy_seed))
        np.random.seed(int(self.accuracy_seed))

        if self.search_mode == "hardware":
            return self._validate_hw_only(key, pcfg)
        return self._validate_model_or_joint(key, mc, pcfg)

    def _validate_hw_only(self, key: str, pcfg: Dict) -> ValidationResult:
        try:
            cache = self._ensure_hw_only_cache()
        except Exception as exc:
            vr = ValidationResult(
                is_valid=False,
                error_message=f"HW-only model build failed: {type(exc).__name__}: {exc}",
                failure_phase="model_build",
            )
            self._validation_errors[key] = vr
            return vr

        hw_obj, error = self._compute_hw_objectives(
            cache.softcores, pcfg, cache.total_params, cache.host_side_segment_count,
        )
        if hw_obj is None:
            vr = ValidationResult(
                is_valid=False, error_message=error, failure_phase="hw_packing",
            )
            self._validation_errors[key] = vr
            return vr

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
            vr = ValidationResult(
                is_valid=False,
                error_message=f"Model build failed: {type(exc).__name__}: {exc}",
                failure_phase="model_build",
            )
            self._validation_errors[key] = vr
            return vr

        if not hw_names:
            self._validation_cache[key] = ValidationEntry(
                model=raw_model, total_params=total_params, hw_objectives={},
            )
            self._evict_validation_cache()
            return ValidationResult(is_valid=True)

        try:
            mapped_model = self._ensure_mapper_repr(raw_model)
        except Exception as exc:
            vr = ValidationResult(
                is_valid=False,
                error_message=f"HW conversion failed: {type(exc).__name__}: {exc}",
                failure_phase="hw_conversion",
            )
            self._validation_errors[key] = vr
            return vr

        try:
            softcores, host_segments = self._collect_softcores(mapped_model, pcfg)
        except Exception as exc:
            vr = ValidationResult(
                is_valid=False,
                error_message=f"Softcore collection failed: {type(exc).__name__}: {exc}",
                failure_phase="hw_conversion",
            )
            self._validation_errors[key] = vr
            return vr

        hw_obj, error = self._compute_hw_objectives(
            softcores, pcfg, total_params, host_segments,
        )
        if hw_obj is None:
            vr = ValidationResult(
                is_valid=False, error_message=error, failure_phase="hw_packing",
            )
            self._validation_errors[key] = vr
            return vr

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
            return 0.0 if self.validate_detailed(configuration).is_valid else 1.0
        except Exception:
            return 1e6
