"""TTFS-cycle fine-tuning: gradual value-domain ramp, genuine cascade at finalize."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.training.blended_genuine_forward import (
    BlendedGenuineForward,
)
from mimarsinan.models.spiking.training.prefix_genuine_forward import (
    PrefixGenuineForward,
)
from mimarsinan.tuning.axes.blend_axis import GenuineBlendAxis, PrefixConversionAxis
from mimarsinan.tuning.forward_install import LazyExecutorForward
from mimarsinan.tuning.orchestration.blend_ramp import (
    KDClassificationLoss,
    PlainClassificationLoss,
    run_teacher_distmatch,
)
from mimarsinan.tuning.orchestration.endpoint_recovery import run_endpoint_recovery
from mimarsinan.tuning.orchestration.genuine_probe import iter_val_batches
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
)
from mimarsinan.tuning.orchestration.mbh_ledger import live_model_acc_fp32
from mimarsinan.tuning.orchestration.ramp_strategy import RampStrategy
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY


class _SegmentSpikeForward(LazyExecutorForward):
    """Picklable ``model.forward`` override driving the segment-aware spike sim.

    ``boundary_surrogate_temp`` (None = severed) enables the offload-boundary STE
    so the genuine backward trains every segment; the forward is unchanged."""

    def __init__(self, model, T: int, *, boundary_surrogate_temp: float | None = None):
        super().__init__(model, T)
        self.boundary_surrogate_temp = boundary_surrogate_temp

    def _build_executor(self):
        from mimarsinan.models.spiking.training.ttfs_segment_forward import (
            TTFSSegmentForward,
        )

        return TTFSSegmentForward(
            self.model.get_mapper_repr(), self.T,
            boundary_surrogate_temp=self.boundary_surrogate_temp,
        )

    def _run(self, x):
        return self._ensure_executor(self._build_executor)(x)


class _Rung2TeacherFlow(nn.Module):
    """Frozen KD teacher evaluating the identity-mapped contract semantics.

    Wraps an identity-mapped ``SpikingHybridCoreFlow``; count-scaled logits (÷T) are
    restored to the value domain via per-output node scales.
    """

    def __init__(self, flow, simulation_length: int, output_scales):
        super().__init__()
        self.flow = flow
        self.simulation_length = int(simulation_length)
        self.register_buffer(
            "_output_scales", torch.as_tensor(output_scales, dtype=torch.float32),
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.flow(x) / float(self.simulation_length)
        return logits * self._output_scales.to(logits.device, logits.dtype)


class _ContractCalibration:
    """Calibration-pipeline resolver bound to the resolved deployment contract."""

    def __init__(self, contract):
        self._contract = contract

    def __call__(self, config, *, synchronized, distmatch_driven):
        return self._contract.calibration_pipeline(
            config, distmatch_driven=distmatch_driven,
        )


class _BlendGenuineKDLoss(KDClassificationLoss):
    """KD (vs frozen teacher) on the blend output + a CE on the PURE genuine logits.

    The extra ``genuine_ce_alpha · CE(genuine_logits, y)`` term sharpens the pure cascade
    so intermediate-rate training lifts the rate-1 endpoint; skipped when no blend forward.
    """

    def __init__(self, teacher, *, genuine_ce_alpha: float, blend_forward_provider,
                 temperature: float = 3.0, alpha: float = 0.3):
        super().__init__(teacher, temperature=temperature, alpha=alpha)
        self.genuine_ce_alpha = float(genuine_ce_alpha)
        self._blend_forward = blend_forward_provider

    def __call__(self, model, x, y):
        loss = super().__call__(model, x, y)
        if self.genuine_ce_alpha > 0.0:
            blend = self._blend_forward()
            if blend is not None:
                loss = loss + self.genuine_ce_alpha * F.cross_entropy(
                    blend.genuine_logits(x), y,
                )
        return loss


class _GenuineRamp(RampStrategy):
    """Shared base of the genuine cascade ramps: ``base_activation`` IS the bare
    ``TTFSActivation`` target (no value blend) and the deployed on-chip rebuild
    runs before the ramp forward is installed."""

    def is_bare_target(self, tuner) -> bool:
        return True

    def make_blend(self, tuner, old, target, rate):
        target.rate = float(rate)
        return target

    def after_install_blend_pre(self, tuner) -> None:
        tuner._finalize_rebuild()


class PrefixConversionRamp(_GenuineRamp):
    """P4 for multi-segment cascaded vehicles: the axis walks converted-prefix k
    (``PrefixGenuineForward``), every rung a genuine partial deployment, trained
    with plain CE (the float suffix hands the frontier segment its gradient — no
    KD teacher); the D-hat gate reads the k-hybrid, the P1'' endpoint closes at
    k=n."""

    def make_axis(self, tuner):
        return PrefixConversionAxis()

    def ramp_forward(self, tuner, model):
        tuner._prefix_forward = PrefixGenuineForward(
            model, tuner._T, rate=0.0,
            boundary_surrogate_temp=tuner._boundary_surrogate_temp,
        )
        return tuner._prefix_forward

    def make_kd_loss(self, tuner):
        return PlainClassificationLoss()

    def after_install_blend_pre(self, tuner) -> None:
        super().after_install_blend_pre(tuner)
        tuner._calibrate_to_teacher_distribution()

    def on_remove_forward(self, tuner) -> None:
        tuner._prefix_forward = None


def _prefix_ann_channel_means(tuner) -> tuple:
    """(teacher per-perceptron channel means, calibration batch), cached per run."""
    from mimarsinan.spiking.dfq_bias_correction import teacher_channel_means

    cached = getattr(tuner, "_prefix_ann_mean_cache", None)
    if cached is None:
        cal_x = tuner._calibration_inputs()
        cached = (teacher_channel_means(tuner._teacher, cal_x), cal_x)
        tuner._prefix_ann_mean_cache = cached
    return cached


def run_prefix_stage_reaffine(tuner, rate: float) -> dict:
    """One P4 stage's keep-best DFQ re-affine measured through the k-hybrid.

    Sets the frontier to ``rate`` first so both the cascade means and the
    keep-best probe read the genuine partial deployment the rung will train.
    """
    from mimarsinan.spiking.dfq_bias_correction import dfq_correct_biases
    from mimarsinan.spiking.distribution_matching import (
        node_values_by_perceptron_index,
    )

    tuner._fast_set_rate(float(rate))
    forward = tuner._prefix_forward
    assert forward is not None, (
        "run_prefix_stage_reaffine requires the installed PrefixGenuineForward"
    )
    ann_mean, cal_x = _prefix_ann_channel_means(tuner)

    def hybrid_channel_values() -> dict:
        with torch.no_grad():
            _, node_values = forward.forward_with_node_values(cal_x)
        return node_values_by_perceptron_index(tuner.model, node_values)

    stats = dfq_correct_biases(
        tuner.model,
        ann_mean,
        hybrid_channel_values,
        bias_iters=int(tuner.pipeline.config.get(
            "ttfs_prefix_stage_dfq_iters", TUNING_POLICY.prefix_stage_dfq_iters,
        )),
        eta=tuner._calibration.distmatch_bias_eta,
        probe=lambda: live_model_acc_fp32(tuner),
        probe_patience=TUNING_POLICY.dfq_keepbest_patience,
    )
    print(
        f"[MBH-PREFIX] tuner={type(tuner).__name__} "
        f"k={forward.prefix_k}/{forward.n_segments} rate={float(rate):.6f} "
        f"dfq_probe_entry={float(stats.get('probe_entry') or 0.0):.6f} "
        f"dfq_probe_best={float(stats.get('probe_best') or 0.0):.6f} "
        f"dfq_iters={int(stats.get('probe_iters_run', 0))}",
        flush=True,
    )
    return stats


class GenuineBlendRamp(_GenuineRamp):
    """Ramp a teacher<->genuine OUTPUT blend (``BlendedGenuineForward``, driven by
    ``GenuineBlendAxis``) calibrated to the teacher distribution; finalize deploys
    the pure cascade."""

    def make_axis(self, tuner):
        return GenuineBlendAxis()

    def ramp_forward(self, tuner, model):
        tuner._blend_forward = BlendedGenuineForward(
            model, tuner._teacher, tuner._T, rate=0.0,
            boundary_surrogate_temp=tuner._boundary_surrogate_temp,
        )
        return tuner._blend_forward

    def make_kd_loss(self, tuner):
        return _BlendGenuineKDLoss(
            tuner._teacher,
            genuine_ce_alpha=tuner._genuine_ce_alpha,
            blend_forward_provider=lambda: tuner._blend_forward,
        )

    def after_install_blend_pre(self, tuner) -> None:
        super().after_install_blend_pre(tuner)
        tuner._calibrate_to_teacher_distribution()

    def on_remove_forward(self, tuner) -> None:
        tuner._blend_forward = None


class TTFSCycleAdaptationTuner(KDBlendAdaptationTuner):
    """Ramp to the TTFS spike node, training through the schedule's NF dynamics."""

    _target_activation_type = "TTFS"

    def _configure(self) -> None:
        self.name = "TTFS Cycle Fine-Tuning"
        self._T = int(self.pipeline.config["simulation_steps"])
        self._thresholding_mode = str(self.pipeline.config.get("thresholding_mode", "<="))
        self._firing_mode = str(self.pipeline.config.get("firing_mode", "TTFS"))
        from mimarsinan.pipelining.core.platform_constraints_resolver import resolve_bias_mode

        self._bias_mode = resolve_bias_mode(self.pipeline.config)
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        contract = SpikingDeploymentContract.from_pipeline_config(self.pipeline.config)
        self._synchronized = contract.training_forward_kind() != "segment_spike"
        self._entry_perceptron_ids = None
        self.adaptation_manager.ttfs_active = True
        from mimarsinan.tuning.orchestration.ttfs_adaptation_plan import (
            TtfsAdaptationPlan,
        )
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        plan = TtfsAdaptationPlan.resolve(
            self.pipeline.config,
            synchronized=self._synchronized,
            optimization_driver=DeploymentPlan.of(self.pipeline).optimization_driver,
            calibration_resolver=_ContractCalibration(contract),
        )
        self._adaptation_plan = plan
        self._genuine_blend_ramp = plan.genuine_blend_ramp
        self._genuine_blend_fast = plan.genuine_blend_fast
        self._proxy_fast = plan.proxy_fast
        self._blend_fast_rates = plan.blend_fast_rates
        self._blend_fast_steps_per_rate = plan.blend_fast_steps_per_rate
        self._endpoint_recovery_steps = plan.endpoint_recovery_steps
        self._prefix_forward = None
        self._adopt_optimization_driver(self._resolve_prefix_ramp(plan))
        self._fast_full_transform_log = []
        self._blend_forward = None
        self._genuine_ce_alpha = float(
            self.pipeline.config.get("ttfs_genuine_blend_ce_alpha", 0.3)
        )
        self._distmatch_stats = None
        cal = plan.calibration
        self._calibration = cal
        self._boundary_surrogate_temp = cal.boundary_surrogate_temp
        self._gain_correction_stats = None
        self._maybe_apply_gain_correction()
        self._theta_cotrain = cal.theta_cotrain

    def _resolve_prefix_ramp(self, plan):
        """Settle the P4 prefix decision (needs the model's segment count) and,
        when active, retarget the fast ladder onto the frontier rates i/n."""
        # Lazy: the spiking package init pulls chip_simulation (import cycle).
        from mimarsinan.spiking.segment_partition import spike_segment_count

        self._prefix_ramp = False
        self._n_spike_segments = 0
        if plan.prefix_ramp:
            self._n_spike_segments = spike_segment_count(self.model.get_mapper_repr())
            self._prefix_ramp = self._n_spike_segments > 1
        if not self._prefix_ramp:
            return plan.driver
        n = self._n_spike_segments
        rates = [i / n for i in range(1, n + 1)]
        self._blend_fast_rates = rates
        return dataclasses.replace(plan.driver, fast_ladder_rates=rates)

    def _maybe_apply_gain_correction(self) -> None:
        """Per-cascade-depth activation_scale trim inverting the ramp decode's depth attenuation.

        A pure calibration change (decode bit-exact, NF↔SCM parity holds); COLD applies the
        full trim once, RATE-GATED RAMP co-adapts ``theta_d`` with the KD blend on each rate.
        """
        cal = self._calibration
        rule, c = cal.gain_rule, cal.gain_c
        self._gain_ramp = cal.gain_ramp

        if cal.gain_ramp:
            self._gain_ramp_rule = rule
            self._gain_ramp_c = c
            self._gain_ramp_base = None
            self._gain_ramp_factors = None
            return

        if cal.gain_cold:
            from mimarsinan.spiking.gain_correction import apply_cascaded_gain_correction

            self._gain_correction_stats = apply_cascaded_gain_correction(
                self.model, self._T, rule=rule, c=c,
            )
            self.pipeline.reporter.report(
                f"{self.name} gain_correction", self._gain_correction_stats,
            )

    def _apply_gain_at_rate(self, rate: float) -> None:
        from mimarsinan.spiking.gain_correction import apply_gain_at_rate

        base, factors = self._gain_ramp_base, self._gain_ramp_factors
        assert base is not None and factors is not None, (
            "_capture_gain_ramp_base must run before _apply_gain_at_rate"
        )
        apply_gain_at_rate(self.model, base, factors, rate)

    def _set_rate(self, rate: float) -> None:
        super()._set_rate(rate)
        if getattr(self, "_gain_ramp", False) and self._gain_ramp_base is not None:
            self._apply_gain_at_rate(rate)

    def _mbh_full_transform_forward(self, clone):
        """[MBH] The deployed cascade on ``clone``, with the rate-gated gain ramp
        (keyed by live perceptron ids) re-applied to the clone at rate 1.0.

        Prefix ramp: the gate reads the k-hybrid at the LIVE frontier — an
        honest genuine partial deployment per rung (rate 1.0 = the deployed
        cascade when no prefix forward is installed, e.g. the entry distmatch)."""
        if getattr(self, "_prefix_ramp", False):
            live = self._prefix_forward
            rate = float(live.rate) if live is not None else 1.0
            return PrefixGenuineForward(
                clone, self._T, rate=rate,
                boundary_surrogate_temp=self._boundary_surrogate_temp,
            )
        fwd = super()._mbh_full_transform_forward(clone)
        if getattr(self, "_gain_ramp", False) and self._gain_ramp_base is not None:
            from mimarsinan.spiking.gain_correction import apply_gain_at_rate

            factors = self._gain_ramp_factors
            assert factors is not None, (
                "_capture_gain_ramp_base must run before the [MBH] full-transform probe"
            )
            clone_factors = {
                id(clone_p): factors.get(id(live_p), 1.0)
                for live_p, clone_p in zip(
                    self.model.get_perceptrons(), clone.get_perceptrons()
                )
            }
            apply_gain_at_rate(clone, self._gain_ramp_base, clone_factors, 1.0)
        return fwd

    def _invalidate_lr_cache(self):
        """The genuine ramp finds the LR once and never re-finds (it is stable across the ramp)."""
        if getattr(self, "_genuine_blend_ramp", False):
            return
        super()._invalidate_lr_cache()

    def _find_lr(self):
        """Genuine ramp uses a coarse 3×10 LR sweep instead of the default 8×30 (perf)."""
        if not getattr(self, "_genuine_blend_ramp", False):
            return super()._find_lr()
        steps = min(10, self._budget.lr_steps_per_probe)
        saved = self._budget
        self._budget = dataclasses.replace(
            saved, lr_num_probes=3, lr_steps_per_probe=steps,
            max_lr_exploration_steps=3 * steps,
        )
        try:
            return super()._find_lr()
        finally:
            self._budget = saved

    def _make_target_activation(self, perceptron) -> TTFSActivation:
        return TTFSActivation(
            T=self._T,
            activation_scale=perceptron.activation_scale,
            input_scale=perceptron.input_activation_scale,
            bias=perceptron.layer.bias,
            thresholding_mode=self._thresholding_mode,
            firing_mode=self._firing_mode,
            encoding=getattr(perceptron, "is_encoding_layer", False),
            bias_mode=self._bias_mode,
        )

    def _make_ramp_strategy(self) -> RampStrategy:
        """Pick the ramp strategy from the flags settled in ``_configure`` — the
        converted-prefix frontier (multi-segment cascaded), the genuine
        teacher↔cascade blend ramp (cascaded only), or the base value-domain
        proxy ramp."""
        if self._prefix_ramp:
            return PrefixConversionRamp()
        if self._genuine_blend_ramp:
            return GenuineBlendRamp()
        return super()._make_ramp_strategy()

    def _fast_ramp(self, rate) -> None:
        """Prefix rungs run the P2 stage re-affine through the k-hybrid before training."""
        if self._prefix_ramp:
            run_prefix_stage_reaffine(self, float(rate))
        super()._fast_ramp(rate)

    def _before_ramp_forward_install(self) -> None:
        """Capture the gain-ramp base after calibration, before the ramp forward (parity-critical order)."""
        self._capture_gain_ramp_base()

    def _capture_gain_ramp_base(self) -> None:
        """Capture post-calibration base scales + target gain factors for the rate-gated ramp, and set rate 0."""
        if not getattr(self, "_gain_ramp", False):
            return
        from mimarsinan.spiking.gain_correction import cascaded_gain_factors

        self._gain_ramp_base = [
            float(p.activation_scale) for p in self.model.get_perceptrons()
        ]
        self._gain_ramp_factors = cascaded_gain_factors(
            self.model, self._T, rule=self._gain_ramp_rule, c=self._gain_ramp_c,
        )
        self._apply_gain_at_rate(0.0)
        self._gain_correction_stats = {
            "mode": "ramp", "rule": self._gain_ramp_rule, "S": int(self._T),
            "n_perceptrons": len(self._gain_ramp_base),
            "min_factor": round(min(self._gain_ramp_factors.values()), 4),
        }
        self.pipeline.reporter.report(
            f"{self.name} gain_correction", self._gain_correction_stats,
        )

    def _calibrate_to_teacher_distribution(self) -> None:
        """Distribution-match the deployed cascade to the frozen teacher ANN.

        Scale-aware [0,1] boundaries from the teacher's activation quantile + DFQ per-neuron
        bias correction, turning the full transform into a smoothly recoverable teacher→genuine ramp.
        """
        from mimarsinan.spiking.distribution_matching import (
            match_activation_distributions,
        )

        cal = self._calibration
        self._distmatch_stats = run_teacher_distmatch(
            self,
            match_activation_distributions,
            quantile=cal.distmatch_quantile,
            bias_iters=cal.distmatch_bias_iters,
            eta=cal.distmatch_bias_eta,
        )

    def _wrap_encoding_input(self, perceptron) -> None:
        if self._synchronized and id(perceptron) in self._segment_entry_ids():
            quantizer = TTFSInputGridQuantizer(
                T=self._T,
                activation_scale=perceptron.input_activation_scale,
            )
            self._append_encoding_input_module(perceptron, quantizer)

    def _segment_entry_ids(self) -> set[int]:
        if self._entry_perceptron_ids is None:
            from mimarsinan.torch_mapping.encoding_layers import (
                segment_entry_perceptrons,
            )

            self._entry_perceptron_ids = {
                id(p)
                for p in segment_entry_perceptrons(self.model.get_mapper_repr())
            }
        return self._entry_perceptron_ids

    def _make_kd_loss(self):
        """Synchronized + ``ttfs_finetune_kd_against_rung2`` distills against the rung-2 teacher flow."""
        if self._synchronized and bool(
            self.pipeline.config.get("ttfs_finetune_kd_against_rung2", False)
        ):
            return self._kd_classification_loss(self._build_rung2_teacher())
        return super()._make_kd_loss()

    def _build_rung2_teacher(self) -> _Rung2TeacherFlow:
        from mimarsinan.models.spiking.hybrid.identity_flow import (
            build_identity_spiking_flow,
        )

        cfg = self.pipeline.config
        ir_graph = self._map_teacher_to_ir()
        flow = build_identity_spiking_flow(
            cfg["input_shape"],
            ir_graph,
            self._T,
            getattr(self._teacher, "preprocessor", None),
            str(cfg.get("firing_mode", "TTFS")),
            str(cfg.get("spike_generation_mode", "TTFS")),
            self._thresholding_mode,
            spiking_mode=str(cfg.get("spiking_mode", "ttfs_cycle_based")),
            ttfs_cycle_schedule="synchronized",
        )
        mapping = flow.hybrid_mapping
        output_scales = [
            float(mapping.node_activation_scales.get(int(src.node_id), 1.0))
            for src in mapping.output_sources.flatten()
        ]
        return _Rung2TeacherFlow(flow, self._T, output_scales)

    def _map_teacher_to_ir(self):
        from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
        from mimarsinan.mapping.ir_mapping_class import IRMapping
        from mimarsinan.mapping.platform.platform_constraints import (
            resolve_platform_mapping_params,
        )
        from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
        from mimarsinan.pipelining.core.platform_constraints_resolver import (
            build_platform_constraints_resolved,
        )
        from mimarsinan.transformations.quantization_bounds import quantization_bounds

        cfg = self.pipeline.config
        params = resolve_platform_mapping_params(
            build_platform_constraints_resolved(cfg)["cores"],
        )
        bits = int(cfg["weight_bits"])
        _, q_max = quantization_bounds(bits)

        # The teacher is a frozen deepcopy of the mapper-capable model.
        mapper_repr = cast(Any, self._teacher).get_mapper_repr()
        if hasattr(mapper_repr, "assign_perceptron_indices"):
            mapper_repr.assign_perceptron_indices()
        compute_per_source_scales(mapper_repr)
        ir_graph = IRMapping(
            q_max=q_max,
            firing_mode=str(cfg.get("firing_mode", "TTFS")),
            max_axons=params.effective_max_axons,
            max_neurons=params.effective_max_neurons,
            hardware_bias=params.hardware_bias,
        ).map(mapper_repr)
        quantize_ir_graph(ir_graph, bits, weight_quantization=False)
        return ir_graph

    def _fast_loss(self, x, y):
        if not self._genuine_blend_ramp:
            return super()._fast_loss(x, y)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        genuine = self._installed_genuine_branch()
        if self._genuine_ce_alpha > 0.0 and genuine is not None:
            loss = loss + self._genuine_ce_alpha * F.cross_entropy(genuine(x), y)
        return loss

    def _fast_probe(self, rate) -> None:
        if self._genuine_blend_ramp:
            self._probe_fast_full_transform(rate)

    def _post_stabilization_hook(self):
        if not getattr(self, "_fixed_ladder_policy", False):
            return
        # P1'': the genuine segment forward (cascaded) or the class analytical
        # forward (proxy) is the deployed composition at this point.
        run_endpoint_recovery(self, base_steps=self._endpoint_recovery_steps)

    def _installed_genuine_branch(self):
        """The PURE genuine-cascade branch of the owned ``BlendedGenuineForward``
        (``None`` when no blend forward is installed — then ``model(x)`` IS genuine)."""
        blend = self._blend_forward
        return blend.genuine_logits if blend is not None else None

    def _probe_fast_full_transform(self, rate) -> None:
        """Record the r=1 full-transform (deployed) accuracy after a fast-ramp round (default off).

        The installed blend forward at rate 1.0 IS the deployed cascade, so no rebuild is needed.
        """
        if not getattr(self, "_full_transform_probe", False):
            return
        blend = self._blend_forward
        prev = float(blend.rate) if blend is not None else rate
        device = self.pipeline.config["device"]
        self._set_rate(1.0)
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in iter_val_batches(self.trainer, self._budget.eval_n_batches):
                pred = self.model(x.to(device)).argmax(dim=1)
                correct += int((pred == y.to(device)).sum())
                total += int(y.numel())
        self._set_rate(prev)
        acc = correct / total if total else 0.0
        self._fast_full_transform_log.append(
            {"rate": float(rate), "full_transform_acc": acc}
        )
        self.pipeline.reporter.report(
            f"{self.name} fast_full_transform",
            {"rate": round(float(rate), 4), "full_transform_acc": round(acc, 4)},
        )

    def _segment_spike_forward_at(self, model, T):
        """The genuine single-spike cascade forward bound to ``model`` at sim-length
        ``T`` (``None`` synchronized — the class analytical staircase IS its forward)."""
        if self._synchronized:
            return None
        return _SegmentSpikeForward(
            model, int(T), boundary_surrogate_temp=self._boundary_surrogate_temp,
        )

    def _ramp_forward_for(self, model):
        """The RAMP (training) genuine cascade forward, built at the deployment S.
        Byte-identical to ``_finalize_forward_for`` — both bind the genuine single-spike
        cascade at ``_T``."""
        return self._segment_spike_forward_at(model, self._T)

    def _finalize_forward_for(self, model):
        """Cascaded keeps the deployed single-spike cascade forward at finalize so every
        downstream step runs the exact dynamics; synchronized returns ``None`` (its class forward)."""
        return self._segment_spike_forward_at(model, self._T)
