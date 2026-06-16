"""TTFS-cycle fine-tuning: gradual value-domain ramp, genuine cascade at finalize.

By default both schedules ramp the per-perceptron ``TTFSActivation`` blend in
the value domain (the plain class forward through ``BlendActivation``), exactly
the golden LIF non-destructive ramp: rate 0 reproduces the continuous teacher,
rate 1 is the pointwise on-chip staircase composition. The blend ramps
naturally under the rate scheduler's uniform ladder (no rate pin), with KD
recovery against the frozen pre-step teacher.

By default the genuine cross-layer dynamics are installed only at **finalize**:

- **cascaded** ``ttfs_cycle_based`` is a single-spike, ramp-integrate, fire-once
  cascade — the per-perceptron staircase blend cannot stay bit-identical to the
  intra-segment sub-window spike timing, so (like LIF's chip-aligned forward)
  finalize installs :class:`TTFSSegmentForward` via ``_finalize_forward`` and
  keeps it, so the committed metric, recovery, and every downstream step run
  the exact deployed single-spike dynamics.
- **synchronized** composes per-group analytical staircases — the ramped class
  forward already *is* that deployed composition, so no instance forward is
  installed; the wire contract's stage-input grid snap q(x) is trained through
  an STE on each segment entry.

The opt-in ``ttfs_genuine_annealed_ramp`` (default off, cascaded only) instead
installs the genuine single-spike cascade for the WHOLE ramp and anneals the
spike surrogate sharpness smooth→sharp (``TTFSGenuineAxis``). Since the surrogate
``alpha`` is backward-only, rate=1 is bit-identical to the deployed cascade, so
the finalize cliff is ~0 by construction.

The opt-in ``ttfs_genuine_blend_ramp`` (default off, cascaded only) calibrates
the deployed cascade to the teacher ANN's activation distribution
(``match_activation_distributions``: scale-aware [0,1] boundaries + DFQ per-neuron
bias correction), then installs a ``BlendedGenuineForward`` as the WHOLE-ramp
forward (``out = (1-rate)*teacher + rate*genuine``). The committed rate drives
the blend live via ``GenuineBlendAxis`` (rate 0 = the frozen continuous teacher,
rate 1 = the genuine cascade exactly), with KD recovery against the same teacher;
finalize deploys the PURE genuine cascade (the teacher is dropped, so the cliff is
0 by construction). It is mutually exclusive with the annealed ramp — blend wins.

The EXPERIMENTAL ``ttfs_genuine_blend_fast`` (default off, requires the blend ramp)
SKIPS the heavy SmoothAdaptation controller: ``run`` overrides the scheduler flow to
walk a FIXED rate schedule (``ttfs_blend_fast_rates``) a FIXED number of steps each
(``ttfs_blend_fast_steps_per_rate``) with ONE Adam optimizer + warmup/cosine LR, loss
``CE((1-R)*teacher + R*genuine) + 0.3*CE(genuine)`` (the validated prototype), then
runs the same finalize (pure genuine cascade). No adaptation cycles, no RecoveryEngine,
no rollback clone/restore, no stabilization pass, no per-cycle LR find.

INVARIANT CORE vs OPTIONAL CONTROLLER (the line the fast hack must not cross).
The genuine gradual-tuning *contract* (pinned in
``tests/unit/tuning/test_genuine_gradual_invariants.py``: inert@low-rate, smooth
degradation, r=1 == deployment bit-exact, tuning lifts the r=1 endpoint, rounds
converge) is a property of the MECHANISM, not the controller — so the mechanism is
installed in ``KDBlendAdaptationTuner.__init__`` (``_install_blend`` →
``_make_kd_loss`` → ``_after_install_blend``) and is therefore ACTIVE UNDER BOTH
paths, fast and slow:

  INVARIANT CORE (always on, even under fast) — DO NOT gate behind the fast flag:
    * scale-aware [0,1] boundaries + DFQ distribution matching
      (``_calibrate_to_teacher_distribution``)
    * the teacher<->genuine ``BlendedGenuineForward`` (rate 0 == teacher, rate 1 ==
      deployed cascade bit-exact)
    * the genuine-CE objective (``_BlendGenuineKDLoss`` / the fast loop's
      ``+ 0.3*CE(genuine)``) — this is what lifts the r=1 endpoint
    * the pure-genuine finalize (``_finalize``)

  OPTIONAL CONTROLLER (the SmoothAdaptation machinery; disabled under the fast hack):
    adaptive rate scheduler + bisect, recover-to-target, rollback clone/restore,
    stabilization pass, per-cycle LR finder, catastrophic gate, target adjuster.
    These add robustness/quality (the slow path), not invariant-correctness. The
    fast hack trades them for ~30-60s; the proper controller keeps them.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.model_training.training_recipe import build_optimizer
from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.training.blended_genuine_forward import (
    BlendedGenuineForward,
)
from mimarsinan.tuning.axes.blend_axis import GenuineBlendAxis, TTFSGenuineAxis
from mimarsinan.tuning.forward_install import LazyExecutorForward
from mimarsinan.tuning.trace import DecisionRecord
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
    _KDClassificationLoss,
)


class _SegmentSpikeForward(LazyExecutorForward):
    """Picklable ``model.forward`` override driving the segment-aware spike sim."""

    def _build_executor(self):
        from mimarsinan.models.spiking.training.ttfs_segment_forward import (
            TTFSSegmentForward,
        )

        return TTFSSegmentForward(self.model.get_mapper_repr(), self.T)

    def _run(self, x):
        return self._ensure_executor(self._build_executor)(x)


class _Rung2TeacherFlow(nn.Module):
    """Frozen KD teacher evaluating the identity-mapped contract semantics.

    Wraps an identity-mapped ``SpikingHybridCoreFlow`` built from the frozen
    pre-step snapshot; outputs are normalized back from the flow's count-scaled
    logits (÷T) and restored to the value domain via per-output node scales.
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


class _BlendGenuineKDLoss(_KDClassificationLoss):
    """KD (vs frozen teacher) on the blend output + a CE on the PURE genuine logits.

    Reproduces the validated genuine-blend recipe: the base KD+CE distills the
    teacher onto the ramping ``BlendedGenuineForward`` output (``model(x)``), while
    an extra ``genuine_ce_alpha · CE(genuine_logits, y)`` term sharpens the pure
    cascade so training at intermediate rates lifts the rate-1 endpoint. The blend
    forward is resolved through ``blend_forward_provider`` (a tuner-owned reference,
    NOT ``model.__dict__``), so the term is decoupled from the install mechanism;
    when the provider returns ``None`` (e.g. finalize/stabilization on the pure
    cascade) ``model(x)`` IS the genuine cascade, so the base CE already covers it
    and the extra term is skipped.
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
        # Genuine annealed ramp (default OFF, cascaded only): train through the
        # genuine single-spike cascade for the WHOLE ramp, annealing the spike
        # surrogate sharpness instead of blending the value domain. Synchronized
        # composes per-group analytical staircases — the cascade walk is the wrong
        # dynamics for it — so the flag is ignored there.
        self._genuine_annealed_ramp = (
            bool(self.pipeline.config.get("ttfs_genuine_annealed_ramp", False))
            and not self._synchronized
        )
        # Genuine teacher->cascade blend ramp (default OFF, cascaded only): calibrate
        # the deployed cascade to the teacher distribution, then ramp a teacher<->
        # genuine OUTPUT blend with KD recovery (finalize deploys the pure cascade).
        # Mutually exclusive with the annealed ramp — blend wins (annealed forced off).
        self._genuine_blend_ramp = (
            bool(self.pipeline.config.get("ttfs_genuine_blend_ramp", False))
            and not self._synchronized
        )
        if self._genuine_blend_ramp:
            self._genuine_annealed_ramp = False
        # Fast fixed-increment genuine-blend ramp (default OFF, requires the genuine
        # blend ramp): run through the ONE orchestrator with a fixed_ladder
        # RateScheduler policy (schedule-not-search) instead of the greedy/bisect
        # controller. _run_with_scheduler reads _fixed_ladder_policy to route.
        self._genuine_blend_fast = self._genuine_blend_ramp and bool(
            self.pipeline.config.get("ttfs_genuine_blend_fast", False)
        )
        self._blend_fast_rates = [
            float(r)
            for r in self.pipeline.config.get(
                "ttfs_blend_fast_rates", [0.5, 0.75, 0.9, 0.97, 1.0]
            )
        ]
        self._blend_fast_steps_per_rate = int(
            self.pipeline.config.get("ttfs_blend_fast_steps_per_rate", 120)
        )
        self._fixed_ladder_policy = self._genuine_blend_fast
        # The fixed ladder MUST end at rate 1.0 (the deployed cascade): a ladder that
        # stops short would leave _committed_rate < 1.0, and the shared _after_run's
        # _continue_to_full_rate would then drive to 1.0 with the HEAVY controller
        # (LR-find/recovery/rollback) — off-recipe. Normalize with a trailing 1.0.
        ladder = list(self._blend_fast_rates) or [1.0]
        if abs(float(ladder[-1]) - 1.0) > 1e-9:
            ladder = [*ladder, 1.0]
        self._fixed_ladder_rates = ladder
        # Fast-path scratch (shared optimizer + spanning warmup/cosine LR across the
        # whole ladder, built once on the first attempt; reset state defaults here).
        self._fast_optimizer = None
        self._fast_lr_schedule = None
        self._fast_optimizer_steps = 0
        self._fast_blend_path = False
        self._fast_full_transform_log = []
        # The installed teacher<->genuine blend forward (owned reference so the
        # genuine-CE loss + the fast attempt resolve it without introspecting
        # model.__dict__); set in _ramp_forward, cleared in _remove_forward.
        self._blend_forward = None
        # Single read of the genuine-CE weight (canonical default in defaults.py);
        # both the KD loss and the fast loop consume this one value.
        self._genuine_ce_alpha = float(
            self.pipeline.config.get("ttfs_genuine_blend_ce_alpha", 0.3)
        )
        self._distmatch_stats = None

    def _invalidate_lr_cache(self):
        """Genuine controller perf: the LR finder is the dominant cost on the
        cascade (~55s/call — 8 probes × 30 steps through the S-step blend forward
        that computes teacher AND genuine), and the base loop re-finds it on every
        target relaxation (measured: 3-4 re-finds eating ~4-5 min before the rate
        left 0.125). The LR is stable across the blend ramp — the fast hack proves
        ONE LR suffices — so the genuine ramp finds it ONCE and never re-finds.
        Scoped to the opt-in genuine ramp → golden-safe for every other tuner."""
        if getattr(self, "_genuine_blend_ramp", False):
            return
        super()._invalidate_lr_cache()

    def _find_lr(self):
        """Genuine controller perf: a coarse 3×10 LR sweep instead of the default
        8×30. Each probe step runs the expensive blend forward (teacher+genuine),
        so the full 8×30 sweep costs ~55s — most of a <2-min budget — for a fine
        LR that barely matters (the fast hack converges on the bare recipe LR).
        Three probes around the anchor are enough to reject a destructive LR.
        Scoped to the opt-in genuine ramp → golden-safe."""
        if not getattr(self, "_genuine_blend_ramp", False):
            return super()._find_lr()
        import dataclasses
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

    # -- genuine ramps (default OFF) -------------------------------------------
    @property
    def _genuine_bare_target_ramp(self) -> bool:
        """Both genuine ramps (annealed + blend) drive the deployed single-spike
        cascade directly, so ``base_activation`` IS the bare ``TTFSActivation``
        target (no value-domain ``BlendActivation`` proxy)."""
        return self._genuine_annealed_ramp or self._genuine_blend_ramp

    def _make_axis(self):
        """Genuine annealed ramp anneals the spike surrogate alpha alongside the
        rate via ``TTFSGenuineAxis``; the genuine blend ramp drives the installed
        teacher<->genuine output blend via ``GenuineBlendAxis``; the value-domain
        proxy ramp keeps the plain ``BlendAxis``. Flag-off is byte-identical."""
        if self._genuine_blend_ramp:
            return GenuineBlendAxis()
        if self._genuine_annealed_ramp:
            return TTFSGenuineAxis()
        return super()._make_axis()

    def _make_blend(self, old, target, rate):
        """Genuine ramps bypass the value blend: ``base_activation`` IS the bare
        ``TTFSActivation`` target so the segment policy drives genuine spike nodes
        (no ReLU side of a blend corrupting the cascade). The annealed ramp still
        carries a per-perceptron ``.rate`` (axis state); the blend ramp drives the
        OUTPUT blend on the installed forward instead. Flag-off keeps the
        value-domain ``BlendActivation``."""
        if self._genuine_bare_target_ramp:
            target.rate = float(rate)
            return target
        return super()._make_blend(old, target, rate)

    def _after_install_blend(self) -> None:
        """Genuine ramps: rebuild the bare TTFS activations into the deployed
        on-chip (``ttfs_active``) state BEFORE installing the ramp forward, so the
        segment policy finds genuine ``TTFSActivation`` nodes. The blend ramp then
        calibrates the cascade to the teacher distribution (scale-aware boundaries
        live-mutate the perceptron scale Parameters the bare nodes already
        reference, so no second rebuild is needed). Flag-off defers to the base."""
        if self._genuine_bare_target_ramp:
            self._finalize_rebuild()
        if self._genuine_blend_ramp:
            self._calibrate_to_teacher_distribution()
        super()._after_install_blend()

    def _ramp_forward(self):
        """The genuine blend ramp installs a teacher<->genuine OUTPUT blend
        (``BlendedGenuineForward``, rate driven live by ``GenuineBlendAxis``) as the
        WHOLE-ramp ``model.forward``: rate 0 reads the frozen teacher exactly, rate
        1 the genuine cascade exactly. The annealed ramp installs the genuine
        single-spike cascade directly (reusing the finalize forward), with the
        spike surrogate alpha annealed — deployment dynamics exact at every rate.
        Flag-off keeps the value-domain ramp (``None``)."""
        if self._genuine_blend_ramp:
            self._blend_forward = BlendedGenuineForward(
                self.model, self._teacher, self._T, rate=0.0,
            )
            return self._blend_forward
        if self._genuine_annealed_ramp:
            return self._finalize_forward_for(self.model)
        return super()._ramp_forward()

    def _remove_forward(self) -> None:
        """Drop the owned blend reference alongside the instance forward: once the
        blend is gone the deployed forward IS the genuine cascade, so the loss must
        stop adding its genuine-CE term (the provider then returns ``None``)."""
        self._blend_forward = None
        super()._remove_forward()

    def _calibrate_to_teacher_distribution(self) -> None:
        """Distribution-match the deployed cascade to the frozen teacher ANN:
        scale-aware [0,1] boundaries from the teacher's per-perceptron activation
        quantile + DFQ per-neuron bias correction. Runs on the deployed-TTFS model
        (after ``_finalize_rebuild``) over a few concatenated validation batches,
        turning the full transform into a smoothly recoverable teacher->genuine
        ramp. Stats are stashed on ``self._distmatch_stats`` and reported."""
        from mimarsinan.spiking.distribution_matching import (
            match_activation_distributions,
        )

        config = self.pipeline.config
        cal_x = self._calibration_inputs()
        self._distmatch_stats = match_activation_distributions(
            self.model,
            self._teacher,
            cal_x,
            self._T,
            quantile=float(config.get("ttfs_distmatch_quantile", 0.99)),
            bias_iters=int(config.get("ttfs_distmatch_bias_iters", 15)),
            eta=float(config.get("ttfs_distmatch_bias_eta", 0.7)),
        )
        self.pipeline.reporter.report(
            f"{self.name} distmatch", self._distmatch_stats,
        )

    def _calibration_inputs(self, n_batches: int = 8) -> torch.Tensor:
        """A few concatenated validation batches as the distribution-match anchor."""
        device = self.pipeline.config["device"]
        batches = [
            x.to(device)
            for x, _ in self.trainer.iter_validation_batches(n_batches)
        ]
        return torch.cat(batches)

    def _wrap_encoding_input(self, perceptron) -> None:
        # Synchronized wire contract: every hybrid stage input is grid-quantized
        # q(x); train through it via an STE on each segment-entry perceptron
        # (the first on-chip core of a segment — NOT ``is_encoding_layer``,
        # which is inert under offload and the wrong seam under subsume).
        # Cascaded NF feeds genuine spike trains (the segment walk encodes).
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

    # -- KD teacher: optional rung-2 (identity-mapped contract) target ---------
    def _make_kd_loss(self):
        """Genuine blend ramp: KD vs the frozen teacher on the blend output PLUS a
        small CE on the PURE genuine logits (the validated recipe — pulls the r=1
        endpoint up). Synchronized + ``ttfs_finetune_kd_against_rung2``: distill
        against the frozen teacher evaluated under rung-2 semantics (identity-mapped
        contract flow) instead of its torch forward (design doc §6.1). Default off."""
        if self._genuine_blend_ramp:
            return _BlendGenuineKDLoss(
                self._teacher,
                genuine_ce_alpha=self._genuine_ce_alpha,
                blend_forward_provider=lambda: self._blend_forward,
            )
        if self._synchronized and bool(
            self.pipeline.config.get("ttfs_finetune_kd_against_rung2", False)
        ):
            from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
                _KDClassificationLoss,
            )

            return _KDClassificationLoss(self._build_rung2_teacher())
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

        mapper_repr = self._teacher.get_mapper_repr()
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

    # -- Fast fixed-increment genuine-blend ramp (fixed_ladder policy) ---------
    # Folded into the ONE orchestrator (review Rec 2): no bespoke ``run()`` engine.
    # ``_fixed_ladder_policy`` routes ``_run_with_scheduler`` to the ``fixed_ladder``
    # RateScheduler policy, which walks the rate ladder driving ``_driver_attempt``
    # (overridden below to train through the genuine cascade with one spanning LR),
    # then the shared ``_finalize_run`` deploys the pure cascade and measures the
    # cliff. So the fast path inherits the DecisionTrace + finalize observability.
    def run(self):
        """Reset the fast-path scratch (the tuner-owned optimizer + spanning cosine)
        so a re-run rebuilds them rather than re-stepping an exhausted schedule, then
        drive the ONE orchestrator. NOT a fork — it unconditionally delegates to the
        shared ``run()``; the fixed_ladder policy is selected in ``_run_with_scheduler``."""
        if self._genuine_blend_fast:
            self._fast_optimizer = None
            self._fast_lr_schedule = None
            self._fast_optimizer_steps = 0
        return super().run()

    def _driver_attempt(self, target):
        """The per-rate attempt the driver/scheduler calls. The fast genuine-blend
        path trains a fixed number of steps at ``target`` with the shared optimizer
        + spanning cosine and the validated ``CE(blend) + ce_alpha·CE(genuine)``
        objective; otherwise the standard predictor→corrector cycle runs."""
        if self._genuine_blend_fast:
            return self._fast_rate_attempt(target)
        return super()._driver_attempt(target)

    def _stabilization_budget(self):
        """The fast path trains through the genuine cascade for the whole ramp, so
        the finalize cliff is ~0 — skip the post-finalize stabilization pass
        (preserving the validated ~30-60s recipe). Else the base budget applies."""
        if self._genuine_blend_fast:
            return 0
        return super()._stabilization_budget()

    def _ensure_fast_optimizer(self):
        """Build the single optimizer + spanning warmup/cosine LR once, sized to the
        whole fixed ladder so the LR anneals smooth→~0 across ALL rates (exactly the
        validated prototype). Subsequent per-rate attempts reuse them."""
        if self._fast_optimizer is not None:
            return
        from mimarsinan.model_training.training_recipe import build_recipe

        device = self.pipeline.config["device"]
        self.model = self.model.to(device)
        lr = float(self.pipeline_lr)
        recipe = build_recipe(self.pipeline.config, key="tuning_recipe")
        if recipe is not None:
            self._fast_optimizer = build_optimizer(self.model, lr, recipe)
        else:
            self._fast_optimizer = self.trainer.build_step_optimizer(lr)
        steps_per_rate = max(0, int(self._blend_fast_steps_per_rate))
        total_steps = max(1, len(self._fixed_ladder_rates) * steps_per_rate)
        self._fast_lr_schedule = self._build_fast_lr_schedule(
            self._fast_optimizer, total_steps,
        )
        self._fast_blend_path = True
        self._fast_optimizer_steps = 0

    def _fast_rate_attempt(self, target):
        """Train ``steps_per_rate`` steps at ``target`` with the shared optimizer +
        spanning cosine and loss ``CE(blend) + ce_alpha·CE(genuine)`` (the INVARIANT
        CORE genuine-CE lifts the r=1 endpoint, invariants 4/5), measure a post
        accuracy, and record a commit into the trace. Always commits ``target`` (the
        well-conditioned transformation needs no rollback)."""
        self._ensure_fast_optimizer()
        t0 = time.time()
        device = self.pipeline.config["device"]
        ce_alpha = self._genuine_ce_alpha
        self._set_rate(float(target))
        for _ in range(max(0, int(self._blend_fast_steps_per_rate))):
            x, y = self.trainer.next_training_batch()
            x, y = x.to(device), y.to(device)
            self.model.train()
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            genuine = self._installed_genuine_branch()
            if ce_alpha > 0.0 and genuine is not None:
                loss = loss + ce_alpha * F.cross_entropy(genuine(x), y)
            self._fast_optimizer.zero_grad()
            loss.backward()
            self._fast_optimizer.step()
            self._fast_lr_schedule.step()
            self._fast_optimizer_steps += 1
        self._committed_rate = float(target)
        post_acc = float(self.trainer.validate_n_batches(self._budget.eval_n_batches))
        self._record_fast_cycle(float(target), post_acc, t0)
        self._last_post_acc = post_acc
        # Observability (invariant 5, probe-gated): the deployed r=1 full-transform.
        self._probe_fast_full_transform(float(target))
        self._phase_seconds["fast_blend"] = (
            self._phase_seconds.get("fast_blend", 0.0) + (time.time() - t0)
        )
        return float(target)

    def _record_fast_cycle(self, target, post_acc, t0):
        """Record one ``commit`` per scheduled rate so the fixed_ladder fast path
        inherits the DecisionTrace the bespoke loop used to drop."""
        if not hasattr(self, "_cycle_log"):
            return
        self._cycle_log.record(DecisionRecord(
            cycle_index=len(self._cycle_log),
            outcome="commit",
            rate=float(target),
            committed=float(self._committed_rate),
            elapsed_sec=time.time() - t0,
            pre_cycle_acc=getattr(self, "_last_post_acc", None),
            post_acc=float(post_acc),
            lr=float(self._fast_optimizer.param_groups[0]["lr"]),
            target=float(self._get_target()),
            validation_baseline=self._baseline_or_none(),
        ))

    def _installed_genuine_branch(self):
        """The PURE genuine-cascade branch of the owned ``BlendedGenuineForward``
        (``None`` when no blend forward is installed — then ``model(x)`` IS genuine)."""
        blend = self._blend_forward
        return blend.genuine_logits if blend is not None else None

    def _probe_fast_full_transform(self, rate) -> None:
        """Observability for invariant 5 (default off, ``tuning_full_transform_probe``):
        after a fast-ramp round, record the r=1 full-transform (deployed) accuracy so
        the convergence is visible even without the controller's per-cycle probe. The
        installed blend forward at rate 1.0 IS the deployed cascade, so no deepcopy or
        finalize-rebuild is needed — set 1.0, evaluate, restore the round's rate."""
        if not getattr(self, "_full_transform_probe", False):
            return
        blend = self._blend_forward
        prev = float(blend.rate) if blend is not None else rate
        device = self.pipeline.config["device"]
        self._set_rate(1.0)
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in self.trainer.iter_validation_batches(self._budget.eval_n_batches):
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

    def _build_fast_lr_schedule(self, optimizer, total_steps):
        """Warmup (5%, linear) -> cosine decay to ~0 over ``total_steps`` step()s."""
        total = max(1, int(total_steps))
        warmup_steps = max(1, int(round(0.05 * total)))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total - warmup_steps), eta_min=0.0,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )

    # -- finalize forward ------------------------------------------------------
    def _finalize_forward_for(self, model):
        """Cascaded builds the genuine single-spike cascade forward bound to
        ``model`` (kept at finalize, so the committed metric, recovery, and every
        downstream step — WQ/NormFusion/SCM — run the exact deployed single-spike
        dynamics; built on a clone for the genuine probe). Synchronized returns
        ``None`` — the class-level analytical staircase forward IS its deployment.
        Both ramp the value-domain blend (``_ramp_forward`` stays ``None``)."""
        if self._synchronized:
            return None
        return _SegmentSpikeForward(model, self._T)
