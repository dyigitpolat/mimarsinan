"""TTFS-cycle fine-tuning: gradual value-domain ramp, genuine cascade at finalize.

Both schedules ramp the per-perceptron ``TTFSActivation`` blend in the value
domain (the plain class forward through ``BlendActivation``), exactly the
golden LIF non-destructive ramp: rate 0 reproduces the continuous teacher,
rate 1 is the pointwise on-chip staircase composition. The blend ramps
naturally under ``SmartSmoothAdaptation`` (no rate pin), with KD recovery
against the frozen pre-step teacher.

The genuine cross-layer dynamics are installed only at **finalize**:

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
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
    _InstalledForward,
)


class _SegmentSpikeForward(_InstalledForward):
    """Picklable ``model.forward`` override driving the segment-aware spike sim."""

    def _run(self, x):
        if self._executor is None:
            from mimarsinan.models.spiking.training.ttfs_segment_forward import (
                TTFSSegmentForward,
            )

            self._executor = TTFSSegmentForward(self.model.get_mapper_repr(), self.T)
        return self._executor(x)


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
        """Synchronized + ``ttfs_finetune_kd_against_rung2``: distill against the
        frozen teacher evaluated under rung-2 semantics (identity-mapped contract
        flow) instead of its torch forward. Costs one IR map at construction and
        a contract-flow eval per KD batch; chases the mapping-level wire residual
        (design doc §6.1). Default off."""
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

    # -- finalize forward ------------------------------------------------------
    def _finalize_forward(self):
        """Cascaded installs the genuine cascade forward at finalize and **keeps**
        it, so the committed metric, recovery, and every downstream step
        (WQ/NormFusion/SCM) run the exact deployed single-spike dynamics;
        synchronized keeps the class-level analytical forward — the dynamics its
        deployment actually executes. Both ramp the value-domain blend
        (``_ramp_forward`` stays ``None``)."""
        if not self._synchronized:
            return _SegmentSpikeForward(self.model, self._T)
        return None
