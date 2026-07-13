"""Tuner for gradual activation quantization."""

import dataclasses

import torch

from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.mapping.support.bias_compensation import (
    apply_lif_half_step_bias_compensation,
    apply_sync_exact_entry_half_step,
)
from mimarsinan.models.nn.activations.autograd import LIF_EXACT_QAT_THETA_FLOOR
from mimarsinan.spiking.dfq_bias_correction import preactivation_channel_means
from mimarsinan.spiking.sync_first_moment import apply_sync_first_moment_fold
from mimarsinan.spiking.theta_cotrain import promote_theta_for_exact_qat
from mimarsinan.tuning.adaptation_rate_tuner import AdaptationRateTuner
from mimarsinan.tuning.orchestration.blend_ramp import kd_loss_from_config
from mimarsinan.tuning.teacher import freeze_module
from mimarsinan.tuning.orchestration.adaptation_manager import (
    install_sync_entry_grid_snap,
    sync_exact_qat_active,
)
from mimarsinan.tuning.orchestration.lif_exact_qat import (
    install_lif_entry_input_quantizers,
    lif_exact_qat_active,
)
from mimarsinan.tuning.orchestration.frontier import frontier_ladder
from mimarsinan.tuning.orchestration.frontier.endpoint_recovery import (
    run_endpoint_recovery,
)
from mimarsinan.tuning.orchestration.frontier.hop_staging import (
    capture_hop_reference,
    resolve_sync_hop_staging,
    run_hop_stage_reaffine,
)
from mimarsinan.tuning.orchestration.mbh_ledger import (
    _fixed_validation_batch,
    _measurement_guard,
)


class ActivationQuantizationTuner(AdaptationRateTuner):
    rate_attr = "quantization_rate"
    _budget_multiplier = 2.0

    def __init__(self, pipeline, model, target_tq, target_accuracy, lr,
                 adaptation_manager, kd_teacher=None):
        super().__init__(pipeline, model, target_accuracy, lr, adaptation_manager)
        self.target_tq = target_tq
        self._kd_teacher = kd_teacher
        self._final_metric = None
        # [S3/R6] the float twin must be captured BEFORE the grid snap and the
        # half-step folds change the forward (it is the reference the endpoint
        # fold matches against).
        self._first_moment_armed = sync_exact_qat_active(self.pipeline.config) and bool(
            self.pipeline.config.get("sync_first_moment_fold", False)
        )
        self._fm_cal_x = None
        self._fm_float_preact = None
        if self._first_moment_armed:
            self._capture_first_moment_reference()
        # [MBH T6] exact-endpoint QAT also trains through the deployed per-stage
        # input grid snap (no-op unless the sync_exact_qat recipe knob + synchronized).
        install_sync_entry_grid_snap(self.model, self.pipeline.config)
        # [5v B1(iii)] the hop frontier arms only when the A6 gauge fails on a
        # chain past the proven-recovery depth; the ladder then walks one hop
        # level per rung with a keep-best re-affine at each frontier step.
        self._hop_stage_levels = resolve_sync_hop_staging(self)
        if self._hop_stage_levels:
            self.adaptation_manager.quantization_hop_levels = self._hop_stage_levels
            capture_hop_reference(self)
            self._adopt_optimization_driver(dataclasses.replace(
                self._optimization_driver,
                fast_ladder_rates=frontier_ladder(self._hop_stage_levels),
            ))
        # [5v B1(ii)] enter the exact-ceil endpoint through the half-step: the
        # fold assumes the CEIL KERNEL, so a hop-staged run defers it to the
        # conversion endpoint (rate 1.0) — applied at init it poisons the
        # k-hybrid's float suffix (fbb1: live k=1 read 0.25, every staged rung
        # refused). Monolithic runs fold at init as before.
        self._half_step_armed = sync_exact_qat_active(self.pipeline.config) and bool(
            self.pipeline.config.get("sync_entry_half_step", False)
        )
        if self._half_step_armed and not self._hop_stage_levels:
            self._fold_entry_half_step()
        # [lif_exact_qat_program §6.1(2)] the LIF exact-QAT install: theta
        # trainable in-loop, the half-step folded ONCE as trainable entry bias
        # (P-L6), and the deployed entry round installed before the ladder.
        self._lif_exact_armed = lif_exact_qat_active(self.pipeline.config)
        if self._lif_exact_armed:
            self._install_lif_exact_qat()
            self._install_exact_qat_kd_teacher()

    def _install_lif_exact_qat(self) -> None:
        report = promote_theta_for_exact_qat(self.model)
        folded = apply_lif_half_step_bias_compensation(
            self.model, int(self.pipeline.config["simulation_steps"]),
        )
        snaps = install_lif_entry_input_quantizers(self.model, self.pipeline.config)
        witness = {
            "installed": True,
            "folded": int(folded),
            "entry_snaps": int(snaps),
            "theta_per_channel": len(report["per_channel"]),
            "theta_scalar": len(report["scalar"]),
            "retimed": bool(self.pipeline.config.get("lif_per_hop_retiming", False)),
        }
        print(
            "[LIF-EXACT-QAT] installed: "
            f"theta per_channel={witness['theta_per_channel']} "
            f"{report['per_channel']} scalar={witness['theta_scalar']} "
            f"{report['scalar']} folded={witness['folded']} "
            f"entry_snaps={witness['entry_snaps']} retimed={witness['retimed']}",
            flush=True,
        )
        emit_reporter_event(self.pipeline.reporter, "lif_exact_qat", witness)

    def _install_exact_qat_kd_teacher(self) -> None:
        """[lif_exact_qat_program §8] Distil the exact-QAT ladder AND endpoint
        to the post-structural float teacher (both share ``self.trainer``);
        on-pipeline the stage otherwise trains plain CE — the measured worst
        KD arm. No teacher (knob off) leaves the loss byte-identical."""
        if self._kd_teacher is None:
            return
        teacher = freeze_module(self._kd_teacher)
        self.trainer.loss_function = kd_loss_from_config(self.pipeline.config, teacher)
        emit_reporter_event(
            self.pipeline.reporter, "lif_exact_qat_kd",
            {"kd": True, "alpha": float(self.pipeline.config.get("kd_ce_alpha", 0.3)),
             "temperature": float(self.pipeline.config.get("kd_temperature", 3.0))},
        )
        print(
            "[LIF-EXACT-QAT] KD teacher installed (post-structural float): "
            f"alpha={self.pipeline.config.get('kd_ce_alpha', 0.3)} "
            f"T={self.pipeline.config.get('kd_temperature', 3.0)}",
            flush=True,
        )

    def _pin_lif_theta_positivity_floor(self) -> None:
        """Pin trained theta at the kernel's forward floor so the finalize-built
        LIF nodes run exactly the theta the QAT trained."""
        pinned = 0
        for perceptron in self.model.get_perceptrons():
            theta = perceptron.activation_scale
            if torch.is_tensor(theta) and bool(
                (theta.detach() < LIF_EXACT_QAT_THETA_FLOOR).any()
            ):
                with torch.no_grad():
                    theta.clamp_(min=LIF_EXACT_QAT_THETA_FLOOR)
                pinned += 1
        if pinned:
            print(
                f"[LIF-EXACT-QAT] theta positivity floor pinned on {pinned} "
                f"perceptron(s) (min {LIF_EXACT_QAT_THETA_FLOOR})",
                flush=True,
            )

    def _fold_entry_half_step(self) -> None:
        folded = apply_sync_exact_entry_half_step(
            self.model,
            int(self.pipeline.config["simulation_steps"]),
            encoding_layer_placement=str(
                self.pipeline.config.get("encoding_layer_placement", "subsume")
            ),
        )
        print(f"[MBH-B1] sync entry half-step folded on {folded} hops", flush=True)

    def _capture_first_moment_reference(self) -> None:
        batch = _fixed_validation_batch(self.trainer)
        if batch is None:
            print(
                "[MBH-S3] sync first-moment fold: no calibration batch; "
                "the endpoint fold will skip",
                flush=True,
            )
            return
        x, _ = batch
        self._fm_cal_x = x.to(self.pipeline.config["device"])
        with _measurement_guard(self.trainer):
            self._fm_float_preact = preactivation_channel_means(
                self.model, self._fm_cal_x,
            )

    def _apply_first_moment_fold(self) -> None:
        # [S3] closed-form sequential fold at the conversion endpoint (rate 1.0,
        # half-step folded), BEFORE endpoint recovery so the QAT trains from
        # the corrected state.
        if self._fm_cal_x is None or self._fm_float_preact is None:
            print(
                "[MBH-S3] sync first-moment fold skipped: no float reference",
                flush=True,
            )
            return
        with _measurement_guard(self.trainer):
            stats = apply_sync_first_moment_fold(
                self.model,
                self._fm_cal_x,
                self._fm_float_preact,
                int(self.pipeline.config["simulation_steps"]),
            )
        print(
            f"[MBH-S3] sync first-moment fold: folded on {stats['folded']} hops "
            f"(mean |delta| {stats['mean_abs_delta']:.6f}, "
            f"skipped {stats['skipped']})",
            flush=True,
        )

    def _fast_ramp(self, rate) -> None:
        if getattr(self, "_hop_stage_levels", None):
            run_hop_stage_reaffine(self, rate)
        super()._fast_ramp(rate)

    def _stabilization_budget(self):
        if sync_exact_qat_active(self.pipeline.config) or getattr(
            self, "_lif_exact_armed", False
        ):
            # The exact-QAT AQ endpoint IS the conversion endpoint: the bounded
            # P1'' stage below replaces the open-ended stabilize.
            return 0
        return 4 * int(self._budget.max_training_steps)

    def _post_stabilization_hook(self):
        if getattr(self, "_lif_exact_armed", False):
            # [lif_exact_qat_program §6.1(2)/§6.3] P1'' through the staircase
            # forward. Wall equivalence: the recipe's endpoint constant is
            # convergence-grounded on the O(S)-per-step cycle-accurate forward;
            # the staircase pays O(1), so the same recipe wall funds S x the
            # steps (the [C1] convergence stop and the run-total step ledger
            # keep the budget a ceiling, never a mandatory burn).
            if getattr(self, "_fixed_ladder_policy", False):
                base = int(self.pipeline.config.get("endpoint_recovery_steps", 0))
                base *= max(1, int(self.pipeline.config["simulation_steps"]))
                run_endpoint_recovery(self, base_steps=base)
            self._pin_lif_theta_positivity_floor()
            return
        if not sync_exact_qat_active(self.pipeline.config):
            return
        if not getattr(self, "_fixed_ladder_policy", False):
            return
        if getattr(self, "_hop_stage_levels", None) and getattr(
            self, "_half_step_armed", False
        ):
            # [5v B1(ii)] the deferred fold: the kernel is fully installed at
            # rate 1.0, so this IS the exact-kernel QAT's entry bias.
            self._fold_entry_half_step()
        if getattr(self, "_first_moment_armed", False):
            self._apply_first_moment_fold()
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
