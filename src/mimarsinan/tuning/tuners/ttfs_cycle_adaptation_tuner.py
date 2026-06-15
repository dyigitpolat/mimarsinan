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
from mimarsinan.tuning.trace import DecisionTrace
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
    cascade so training at intermediate rates lifts the rate-1 endpoint. The
    genuine logits are read from the installed ``BlendedGenuineForward`` (its
    ``_genuine`` branch); when no blend forward is installed (e.g. finalize/
    stabilization on the pure cascade) ``model(x)`` IS the genuine cascade, so the
    base CE already covers it and the extra term is skipped.
    """

    def __init__(self, teacher, *, genuine_ce_alpha: float = 0.3,
                 temperature: float = 3.0, alpha: float = 0.3):
        super().__init__(teacher, temperature=temperature, alpha=alpha)
        self.genuine_ce_alpha = float(genuine_ce_alpha)

    def __call__(self, model, x, y):
        loss = super().__call__(model, x, y)
        blend = model.__dict__.get("forward")
        genuine = getattr(blend, "_genuine", None)
        if self.genuine_ce_alpha > 0.0 and callable(genuine):
            loss = loss + self.genuine_ce_alpha * F.cross_entropy(genuine(x), y)
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
        # EXPERIMENTAL fast fixed-increment genuine-blend ramp (default OFF, requires
        # the genuine blend ramp): SKIP the SmoothAdaptation controller and walk a
        # fixed rate schedule with a fixed step count + one Adam optimizer.
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
        self._distmatch_stats = None

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
            return BlendedGenuineForward(self.model, self._teacher, self._T, rate=0.0)
        if self._genuine_annealed_ramp:
            return self._finalize_forward_for(self.model)
        return super()._ramp_forward()

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
                genuine_ce_alpha=float(
                    self.pipeline.config.get("ttfs_genuine_blend_ce_alpha", 0.3)
                ),
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

    # -- EXPERIMENTAL fast fixed-increment genuine-blend ramp ------------------
    def run(self):
        """EXPERIMENTAL fast path (``ttfs_genuine_blend_fast``, default OFF): skip
        the SmoothAdaptation controller and walk a FIXED rate schedule with a fixed
        step count. Otherwise the existing SmoothAdaptation flow runs UNCHANGED."""
        if self._genuine_blend_fast:
            return self._run_fast_genuine_blend()
        return super().run()

    def _run_fast_genuine_blend(self):
        """Reproduce the validated prototype (``generated/_genuine_ab/full_ramp.py``):
        one Adam optimizer + warmup/cosine LR over the whole schedule, walking a
        FIXED ``[0.5, 0.75, 0.9, 0.97, 1.0]`` rate ladder a FIXED number of steps each
        with loss ``CE((1-R)*teacher + R*genuine) + 0.3*CE(genuine)``, then deploy the
        PURE genuine cascade (the existing finalize) — NO ``_adaptation`` cycles, NO
        RecoveryEngine, NO rollback clone/restore, NO ``_stabilize_at_full_rate``, NO
        per-cycle LR find. The blend output and the pure-genuine logits are read from
        the installed :class:`BlendedGenuineForward` (``model(x)`` IS the blend; its
        ``_genuine`` branch is the cascade), so this stays the prototype's recipe."""
        from mimarsinan.model_training.training_recipe import build_recipe

        self._committed_rate = 0.0
        self._cycle_log = DecisionTrace.new()
        self._phase_seconds = {}
        self._fast_blend_path = True
        self._fast_optimizer_steps = 0
        t0 = time.time()

        rates = list(self._blend_fast_rates) or [1.0]
        steps_per_rate = max(0, int(self._blend_fast_steps_per_rate))
        total_steps = max(1, len(rates) * steps_per_rate)

        device = self.pipeline.config["device"]
        self.model = self.model.to(device)
        lr = float(self.pipeline_lr)

        recipe = build_recipe(self.pipeline.config, key="tuning_recipe")
        if recipe is not None:
            optimizer = build_optimizer(self.model, lr, recipe)
        else:
            optimizer = self.trainer.build_step_optimizer(lr)
        scheduler = self._build_fast_lr_schedule(optimizer, total_steps)

        ce_alpha = float(
            self.pipeline.config.get("ttfs_genuine_blend_ce_alpha", 0.3)
        )

        for rate in rates:
            self._set_rate(float(rate))
            for _ in range(steps_per_rate):
                x, y = self.trainer.next_training_batch()
                x, y = x.to(device), y.to(device)
                self.model.train()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                genuine = self._installed_genuine_branch()
                if ce_alpha > 0.0 and genuine is not None:
                    loss = loss + ce_alpha * F.cross_entropy(genuine(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                self._fast_optimizer_steps += 1

        self._set_rate(1.0)
        self._remove_forward()
        self._finalize()
        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0

        self._phase_seconds["fast_blend"] = time.time() - t0
        self.pipeline.reporter.report(
            f"{self.name} fast_blend",
            {
                "rates": rates,
                "steps_per_rate": steps_per_rate,
                "optimizer_steps": self._fast_optimizer_steps,
                "final_metric": self._final_metric,
            },
        )
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self.pipeline.reporter.report(f"{self.name} phase_seconds", self._phase_seconds)
        return self._final_metric

    def _installed_genuine_branch(self):
        """The PURE genuine-cascade branch of the installed ``BlendedGenuineForward``
        (``None`` when no blend forward is installed — then ``model(x)`` IS genuine)."""
        installed = self.model.__dict__.get("forward")
        return getattr(installed, "_genuine", None)

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
