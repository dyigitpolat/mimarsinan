"""Tuner for gradual activation quantization.

Rollout strategy
----------------
The cycle variable ``rate`` is **binary-discrete**: at rate ``t`` the
tuner installs the hard ``QuantizeDecorator`` (``StaircaseFunction``)
on the first ``round(t * N)`` perceptrons in the cached
*sensitivity order* (least-sensitive first), and leaves the rest
un-quantised.  ``rate == 0`` is the all-float baseline; ``rate == 1``
has every perceptron hard-quantised.  There is no smooth annealing
inside a single perceptron -- the soft DSQ path was deleted because
it collapsed every activation to ~0.5 at low β, which produced
catastrophic cycle outcomes and never gave the optimiser a useful
gradient signal.

Per-perceptron sensitivity is measured empirically once at the
start of :meth:`run`: for each perceptron, the hard quantiser is
installed alone, ``trainer.validate_fast`` is run, and the accuracy
drop vs. the unquantised baseline is recorded.  The ascending order
of drops is the rollout order.

Per-perceptron state is routed through
:meth:`AdaptationManager.set_per_perceptron_rate` so the existing
Phase-B1 dispatch in ``get_rate`` / ``update_activation`` does the
right thing with zero knowledge of this tuner's internals.
``_clone_state`` / ``_restore_state`` snapshot the override dict so
a failed cycle rolls back to the previous per-layer k.
"""

from __future__ import annotations

import copy

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class ActivationQuantizationTuner(SmoothAdaptationTuner):
    def __init__(self, pipeline, model, target_tq, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.target_tq = target_tq
        self.adaptation_manager = adaptation_manager
        self._final_metric = None
        # Ascending-sensitivity perceptron names, filled lazily by
        # ``_measure_layer_sensitivities``.  ``None`` until the probe
        # has run; tests seed this directly to bypass the probe.
        self._sensitivity_order: list[str] | None = None

    # ------------------------------------------------------------------
    # Per-perceptron state <-> AdaptationManager override dict
    # ------------------------------------------------------------------

    def _perceptron_names(self) -> list[str]:
        return [p.name for p in self.model.get_perceptrons()]

    def _apply_overrides(self, enabled_names) -> None:
        """Install the hard quantiser on ``enabled_names`` and remove it
        from every other perceptron.  The scalar ``quantization_rate``
        is left at 0.0 throughout -- per-perceptron overrides carry
        the full signal."""
        enabled = set(enabled_names)
        am = self.adaptation_manager
        # Clear any stale overrides first, then re-install the ones we
        # want.  ``set_per_perceptron_rate(..., 0.0)`` also works, but
        # wiping the dict first keeps the override set minimal and
        # makes snapshot diffs easy to read in logs.
        am.clear_per_perceptron_overrides("quantization_rate")
        for name in enabled:
            am.set_per_perceptron_rate("quantization_rate", name, 1.0)
        for p in self.model.get_perceptrons():
            am.update_activation(self.pipeline.config, p)

    def _current_overrides_snapshot(self) -> dict:
        """Deep-copy the per-perceptron quantisation overrides so a
        later restore can't be mutated by subsequent cycles."""
        am = self.adaptation_manager
        return copy.deepcopy(am._per_perceptron_rates.get("quantization_rate", {}))

    # ------------------------------------------------------------------
    # Sensitivity probe
    # ------------------------------------------------------------------

    def _measure_layer_sensitivities(self) -> list[str]:
        """Quantise each perceptron alone, record the
        ``validate_fast`` drop vs. the un-quantised baseline, and
        return perceptron names sorted by drop *ascending*
        (least-sensitive first).

        The probe is side-effect free: it snapshots the per-perceptron
        override dict before the first probe and restores it at the
        end.  Callers can therefore run the probe at any point in the
        tuner's life without perturbing state."""
        am = self.adaptation_manager
        pre_overrides = copy.deepcopy(am._per_perceptron_rates)

        # Baseline: every perceptron un-quantised (scalar already 0 at
        # probe time; we defensively wipe any lingering overrides).
        am.clear_per_perceptron_overrides("quantization_rate")
        for p in self.model.get_perceptrons():
            am.update_activation(self.pipeline.config, p)
        baseline_acc = float(self.trainer.validate_fast())

        drops: dict[str, float] = {}
        names = self._perceptron_names()
        for name in names:
            am.clear_per_perceptron_overrides("quantization_rate")
            am.set_per_perceptron_rate("quantization_rate", name, 1.0)
            for p in self.model.get_perceptrons():
                am.update_activation(self.pipeline.config, p)
            acc = float(self.trainer.validate_fast())
            drops[name] = baseline_acc - acc

        # Restore the pre-probe override dict and rebuild activations
        # to match.  Using ``update`` + a clear-then-copy keeps the
        # semantics of ``clear_per_perceptron_overrides`` (empty dict,
        # not a re-assigned one) intact.
        for rate_name in list(am._per_perceptron_rates):
            am._per_perceptron_rates[rate_name].clear()
        for rate_name, mapping in pre_overrides.items():
            am._per_perceptron_rates[rate_name].update(mapping)
        for p in self.model.get_perceptrons():
            am.update_activation(self.pipeline.config, p)

        return sorted(names, key=lambda n: drops[n])

    # ------------------------------------------------------------------
    # SmoothAdaptationTuner hooks
    # ------------------------------------------------------------------

    def _before_cycle(self):
        """Populate the sensitivity order on the very first cycle.
        The probe is expensive (one ``validate_fast`` per perceptron)
        so we cache it for the whole run."""
        if self._sensitivity_order is None:
            self._sensitivity_order = self._measure_layer_sensitivities()

    def _rate_to_k(self, rate: float) -> int:
        """Map the cycle's continuous rate to an integer perceptron
        count.  ``round`` (banker's rounding) matches the contract
        tested in
        ``test_activation_quantization_per_layer_rollout.py``."""
        n = len(self._sensitivity_order or self._perceptron_names())
        k = int(round(float(rate) * n))
        return max(0, min(n, k))

    def _update_and_evaluate(self, rate):
        # Ensure the sensitivity order is primed -- tests may call
        # ``_update_and_evaluate`` directly without going through
        # ``run`` / ``_before_cycle``.
        if self._sensitivity_order is None:
            self._sensitivity_order = self._perceptron_names()
        k = self._rate_to_k(rate)
        enabled = self._sensitivity_order[:k]
        self._apply_overrides(enabled)
        return self.trainer.validate_fast()

    # ------------------------------------------------------------------
    # State snapshot: per-perceptron overrides, not the scalar rate
    # ------------------------------------------------------------------

    def _get_extra_state(self):
        return {
            "per_perceptron_quant_overrides": self._current_overrides_snapshot(),
        }

    def _set_extra_state(self, extra):
        am = self.adaptation_manager
        overrides = (extra or {}).get("per_perceptron_quant_overrides", {})
        am.clear_per_perceptron_overrides("quantization_rate")
        for name, value in overrides.items():
            am.set_per_perceptron_rate("quantization_rate", name, float(value))
        for p in self.model.get_perceptrons():
            am.update_activation(self.pipeline.config, p)

    # ------------------------------------------------------------------
    # Validation + final commit
    # ------------------------------------------------------------------

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate_full()

    def _after_run(self):
        # Contract: every perceptron must end the step hard-quantised
        # regardless of the tuner's natural progress.  ``_continue_to_full_rate``
        # tries to climb gradually; if that fails, we force k = N here
        # so the downstream pipeline always sees a fully-quantised model.
        self._continue_to_full_rate()

        if self._sensitivity_order is None:
            self._sensitivity_order = self._perceptron_names()
        self._apply_overrides(self._sensitivity_order)

        recovered_val = self._attempt_recovery_if_below_floor()
        if recovered_val >= self._validation_floor_for_commit():
            self._committed_rate = 1.0
        self._final_metric = recovered_val
        self._flush_enforcement_hooks()
        return self._final_metric

    def run(self):
        return super().run()
