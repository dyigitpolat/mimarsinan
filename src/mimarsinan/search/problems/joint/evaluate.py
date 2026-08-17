"""The one evaluation contract for joint architecture + hardware search.

A candidate becomes a :class:`CandidateStaticView`, and the ACTIVE registry
objectives are read off that view. There is no per-mode objective assembly:
accuracy attaches exactly where the registry says the mode can carry the axis,
and every other axis is a question asked of the same view.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict

from mimarsinan.deployment_record.objectives import CandidateStaticView
from mimarsinan.search.evaluators.extrapolating_accuracy_evaluator import ExtrapolatingAccuracyEvaluator
from mimarsinan.search.evaluators.fast_accuracy_evaluator import FastAccuracyEvaluator
from mimarsinan.search.optimizers.budget import charge_evaluation
from mimarsinan.search.option_axes import candidate_option
from mimarsinan.search.problem import CandidateInfeasibleError
from mimarsinan.search.results import ACCURACY_OBJECTIVE_NAME

from .types import (
    HW_PACKING_PHASE,
    CandidateFailure,
    CandidatePlatformError,
    JointHostContract,
    ValidationEntry,
    json_key,
)

logger = logging.getLogger(__name__)

# A candidate that simply does not FIT is scored, not raised about; a candidate
# that broke while being built crosses the boundary typed. Structural rejection
# is not listed because it cannot arrive here: ``validate_fn`` runs in
# ``validate_detailed``, before any candidate fact is resolved, so the phases
# ``_resolve_entry`` can report are model build, conversion and packing.
_PENALIZED_PHASES = frozenset({HW_PACKING_PHASE})


class JointEvaluateMixin(JointHostContract):
    """Objective evaluation for :class:`JointArchHwProblem`.

    Candidate-scoped failures cross the problem boundary as
    :class:`CandidateInfeasibleError`; anything else propagates untyped.
    """

    def evaluate(self, configuration: Dict[str, Any]) -> Dict[str, float]:
        try:
            configuration = self._resolved_configuration(configuration)
        except CandidatePlatformError:
            return self._penalty_objectives()

        key = json_key(configuration)
        cached = self._cache.get(key)
        # [TS1] THE distinct-evaluation seam: this cache already knows whether a
        # candidate identity costs evaluator work (miss) or was merely proposed
        # again (hit), so the run's accountant is told here and nowhere else.
        charge_evaluation(self.evaluation_budget, key, hit=cached is not None)
        if cached is not None:
            return cached

        vr = self.validate_detailed(configuration)
        if not vr.is_valid:
            obj = self._penalty_objectives()
            self._cache[key] = obj
            return obj

        # The validation cache is an OPTIMIZATION, not a dependency: it is
        # bounded, so a validated candidate whose entry has been evicted is
        # simply resolved again rather than scored off something stale.
        entry = self._validation_cache.get(key)
        if entry is None:
            obj = self._evaluate_inner(
                configuration["model_config"],
                configuration["platform_constraints"],
                str(candidate_option(
                    configuration, "encoding_layer_placement",
                    self.encoding_placement,
                )),
            )
        else:
            obj = self._objectives_from_entry(entry)

        self._cache[key] = obj
        return obj

    def _evaluate_inner(
        self, mc: Dict[str, Any], pcfg: Dict[str, Any], placement: str,
    ) -> Dict[str, float]:
        """Evaluate one candidate pair directly, without the configuration cache."""
        entry, failure = self._resolve_entry(mc, pcfg, placement)
        if entry is None:
            assert failure is not None
            return self._raise_or_penalize(failure)
        return self._objectives_from_entry(entry)

    def _raise_or_penalize(self, failure: CandidateFailure) -> Dict[str, float]:
        """Render a candidate-scoped failure at the evaluate boundary."""
        if failure.phase in _PENALIZED_PHASES:
            logger.warning(
                "[JointArchHwProblem] %s – returning full penalty", failure.message,
            )
            return self._penalty_objectives()
        raise CandidateInfeasibleError(failure.message) from failure.cause

    def _objectives_from_entry(self, entry: ValidationEntry) -> Dict[str, float]:
        """Attach the accuracy estimate where the mode carries one, then read the axes."""
        view = entry.view
        # ``estimated_accuracy`` names both the objective and the view fragment
        # backing it, so "does an active axis need this fragment" IS "must this
        # search train" — asked of the registry rather than of a mode string.
        if self._requires_fragment(ACCURACY_OBJECTIVE_NAME):
            view = replace(
                view, estimated_accuracy=self._accuracy_estimate(entry),
            )
        return self._objectives_from_view(view)

    def _objectives_from_view(self, view: CandidateStaticView) -> Dict[str, float]:
        """THE evaluation contract: every ACTIVE registry axis, read off one view."""
        return {spec.key: spec.value(view) for spec in self.active_specs}

    def _accuracy_estimate(self, entry: ValidationEntry) -> float:
        """The training proxy; a failed estimate is a penalty, never a lost candidate."""
        model = entry.model
        if model is None:
            raise CandidateInfeasibleError(
                "accuracy is an active objective but the candidate carries no model"
            )
        try:
            return self._evaluate_accuracy(model)
        except Exception as exc:
            logger.warning(
                "[JointArchHwProblem] Accuracy evaluation failed (%s: %s); "
                "recording penalty accuracy 0.0", type(exc).__name__, exc,
                exc_info=True,
            )
            return 0.0
        finally:
            # The estimate is produced once; the model is the search's largest
            # retained object, so it is released as soon as it has answered.
            entry.model = None

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
