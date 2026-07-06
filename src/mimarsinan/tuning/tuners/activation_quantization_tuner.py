"""Tuner for gradual activation quantization."""

import dataclasses

from mimarsinan.mapping.support.bias_compensation import (
    apply_sync_exact_entry_half_step,
)
from mimarsinan.tuning.adaptation_rate_tuner import AdaptationRateTuner
from mimarsinan.tuning.orchestration.adaptation_manager import (
    install_sync_entry_grid_snap,
    sync_exact_qat_active,
)
from mimarsinan.tuning.orchestration.endpoint_recovery import run_endpoint_recovery
from mimarsinan.tuning.orchestration.hop_staging import (
    capture_hop_reference,
    frontier_rates,
    resolve_sync_hop_staging,
    run_hop_stage_reaffine,
)


class ActivationQuantizationTuner(AdaptationRateTuner):
    rate_attr = "quantization_rate"
    _budget_multiplier = 2.0

    def __init__(self, pipeline, model, target_tq, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr, adaptation_manager)
        self.target_tq = target_tq
        self._final_metric = None
        # [MBH T6] exact-endpoint QAT also trains through the deployed per-stage
        # input grid snap (no-op unless the sync_exact_qat recipe knob + synchronized).
        install_sync_entry_grid_snap(self.model, self.pipeline.config)
        # [5v B1(ii)] enter the exact-ceil endpoint through the half-step: the
        # fold is trainable bias, applied once, and the QAT may train it away.
        if sync_exact_qat_active(self.pipeline.config) and bool(
            self.pipeline.config.get("sync_entry_half_step", False)
        ):
            folded = apply_sync_exact_entry_half_step(
                self.model, int(self.pipeline.config["simulation_steps"])
            )
            print(f"[MBH-B1] sync entry half-step folded on {folded} hops", flush=True)
        # [5v B1(iii)] the hop frontier arms only when the A6 gauge fails on a
        # chain past the proven-recovery depth; the ladder then walks one hop
        # level per rung with a keep-best re-affine at each frontier step.
        self._hop_stage_levels = resolve_sync_hop_staging(self)
        if self._hop_stage_levels:
            self.adaptation_manager.quantization_hop_levels = self._hop_stage_levels
            capture_hop_reference(self)
            self._adopt_optimization_driver(dataclasses.replace(
                self._optimization_driver,
                fast_ladder_rates=frontier_rates(self._hop_stage_levels),
            ))

    def _fast_ramp(self, rate) -> None:
        if getattr(self, "_hop_stage_levels", None):
            run_hop_stage_reaffine(self, rate)
        super()._fast_ramp(rate)

    def _stabilization_budget(self):
        if sync_exact_qat_active(self.pipeline.config):
            # The sync AQ endpoint IS the conversion endpoint: the bounded
            # P1'' stage below replaces the open-ended stabilize.
            return 0
        return 4 * int(self._budget.max_training_steps)

    def _post_stabilization_hook(self):
        if not sync_exact_qat_active(self.pipeline.config):
            return
        if not getattr(self, "_fixed_ladder_policy", False):
            return
        # P1'' for sync: rate 1.0 through the ceil kernel + grid snap IS the
        # exact deployed composition (T6) — train it to the D-hat high-water.
        run_endpoint_recovery(
            self,
            base_steps=int(self.pipeline.config.get("endpoint_recovery_steps", 0)),
        )

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()
