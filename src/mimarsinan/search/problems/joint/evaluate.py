"""Candidate evaluation for joint architecture + hardware search."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch

from mimarsinan.mapping.platform.coalescing import normalize_coalescing_config
from mimarsinan.search.evaluators.extrapolating_accuracy_evaluator import ExtrapolatingAccuracyEvaluator
from mimarsinan.search.evaluators.fast_accuracy_evaluator import FastAccuracyEvaluator
from mimarsinan.search.results import ACCURACY_OBJECTIVE_NAME

from .types import ValidationEntry, json_key

logger = logging.getLogger(__name__)


def _warn_penalized(
    stage: str, exc: Exception, configuration: Any, penalty_desc: str,
) -> None:
    logger.warning(
        "[JointArchHwProblem] %s failed (%s: %s) for candidate %.500s; %s",
        stage, type(exc).__name__, exc, configuration, penalty_desc,
        exc_info=True,
    )


class JointEvaluateMixin:
    """Accuracy and objective evaluation for :class:`JointArchHwProblem`."""

    def evaluate(self, configuration: Dict[str, Any]) -> Dict[str, float]:
        key = json_key(configuration)
        if key in self._cache:
            return self._cache[key]

        vr = self.validate_detailed(configuration)
        if not vr.is_valid:
            obj = self._penalty_objectives()
            self._cache[key] = obj
            return obj

        vc = self._validation_cache.get(key)
        if vc is not None:
            try:
                obj = self._evaluate_from_cache(vc, configuration)
            except Exception as exc:
                _warn_penalized(
                    "_evaluate_from_cache", exc, configuration,
                    "recording full penalty objectives",
                )
                obj = self._penalty_objectives()
        else:
            mc = configuration["model_config"]
            pcfg = configuration["platform_constraints"]
            torch.manual_seed(int(self.accuracy_seed))
            np.random.seed(int(self.accuracy_seed))
            try:
                obj = self._evaluate_inner(mc, pcfg)
            except Exception as exc:
                _warn_penalized(
                    "_evaluate_inner", exc, configuration,
                    "recording full penalty objectives",
                )
                obj = self._penalty_objectives()

        self._cache[key] = obj
        return obj

    def _evaluate_from_cache(
        self,
        vc: ValidationEntry,
        configuration: Dict[str, Any],
    ) -> Dict[str, float]:
        """Build objectives from a cached validation entry."""
        active_names = {spec.name for spec in self.objectives}
        needs_accuracy = ACCURACY_OBJECTIVE_NAME in active_names

        obj: Dict[str, float] = {
            k: v for k, v in vc.hw_objectives.items() if k in active_names
        }

        if needs_accuracy:
            if vc.model is not None:
                try:
                    obj[ACCURACY_OBJECTIVE_NAME] = self._evaluate_accuracy(vc.model)
                except Exception as exc:
                    _warn_penalized(
                        "Accuracy evaluation", exc, configuration,
                        "recording penalty accuracy 0.0",
                    )
                    obj[ACCURACY_OBJECTIVE_NAME] = 0.0
                finally:
                    vc.model = None
            else:
                mc = configuration["model_config"]
                pcfg = configuration["platform_constraints"]
                torch.manual_seed(int(self.accuracy_seed))
                np.random.seed(int(self.accuracy_seed))
                try:
                    raw_model, _ = self._build_raw_model(mc, pcfg)
                    obj[ACCURACY_OBJECTIVE_NAME] = self._evaluate_accuracy(raw_model)
                except Exception as exc:
                    _warn_penalized(
                        "Accuracy evaluation", exc, configuration,
                        "recording penalty accuracy 0.0",
                    )
                    obj[ACCURACY_OBJECTIVE_NAME] = 0.0

        return obj

    def _evaluate_inner(
        self,
        mc: Dict[str, Any],
        pcfg: Dict[str, Any],
    ) -> Dict[str, float]:
        pcfg = dict(pcfg)
        normalize_coalescing_config(pcfg)
        active_names = {spec.name for spec in self.objectives}
        needs_accuracy = ACCURACY_OBJECTIVE_NAME in active_names
        hw_names = active_names - {ACCURACY_OBJECTIVE_NAME}

        if self.search_mode == "hardware":
            cache = self._ensure_hw_only_cache()
            hw_obj, _err = self._compute_hw_objectives(
                cache.softcores, pcfg, cache.total_params, cache.host_side_segment_count,
            )
            if hw_obj is None:
                return self._penalty_objectives()
            return {k: v for k, v in hw_obj.items() if k in active_names}

        raw_model, total_params = self._build_raw_model(mc, pcfg)

        obj: Dict[str, float] = {}

        if hw_names:
            try:
                mapped_model = self._ensure_mapper_repr(raw_model)
                softcores, host_segments = self._collect_softcores(mapped_model, pcfg)
                hw_obj, _err = self._compute_hw_objectives(
                    softcores, pcfg, total_params, host_segments,
                )
                if hw_obj is None:
                    print("[JointArchHwProblem] Packing infeasible – returning full penalty")
                    return self._penalty_objectives()
                for k, v in hw_obj.items():
                    if k in active_names:
                        obj[k] = v
            except Exception as exc:
                _warn_penalized(
                    "HW objective computation", exc,
                    {"model_config": mc, "platform_constraints": pcfg},
                    "recording full penalty objectives",
                )
                return self._penalty_objectives()

        if needs_accuracy:
            try:
                accuracy = self._evaluate_accuracy(raw_model)
                obj[ACCURACY_OBJECTIVE_NAME] = accuracy
            except Exception as exc:
                _warn_penalized(
                    "Accuracy evaluation", exc,
                    {"model_config": mc, "platform_constraints": pcfg},
                    "recording penalty accuracy 0.0",
                )
                obj[ACCURACY_OBJECTIVE_NAME] = 0.0

        return obj

    def _evaluate_accuracy(self, model) -> float:
        if self.accuracy_evaluator == "extrapolating":
            acc_eval = ExtrapolatingAccuracyEvaluator(
                data_provider_factory=self.data_provider_factory,
                device=self.device,
                lr=float(self.lr),
                num_train_epochs=int(self.extrapolation_num_train_epochs),
                num_checkpoints=int(self.extrapolation_num_checkpoints),
                target_epochs=int(self.extrapolation_target_epochs),
                warmup_fraction=float(self.warmup_fraction),
                num_workers=0,
                training_batch_size=self.training_batch_size,
                seed=int(self.accuracy_seed),
            )
        else:
            acc_eval = FastAccuracyEvaluator(
                data_provider_factory=self.data_provider_factory,
                device=self.device,
                lr=float(self.lr),
                warmup_fraction=float(self.warmup_fraction),
                num_workers=0,
                training_batch_size=self.training_batch_size,
                seed=int(self.accuracy_seed),
            )
        return float(acc_eval.evaluate(model))
