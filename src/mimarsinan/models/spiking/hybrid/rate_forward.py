"""Hybrid rate-coded full-network forward."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    compute_input_state_with_shifts,
    execute_compute_op_torch,
    resolve_stage_compute_scales,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import (
    HybridStageContext,
    run_hybrid_stages,
)
from mimarsinan.chip_simulation.recording.records import SegmentSpikeRecord
from mimarsinan.chip_simulation.spiking_semantics import (
    is_cascaded_ttfs,
    requires_ttfs_firing,
)
from mimarsinan.mapping.ir import IRSource
from mimarsinan.models.spiking.hybrid.host import HybridFlowHost
from mimarsinan.models.spiking.hybrid.membrane_readout import (
    apply_membrane_corrections_to_logits,
)
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE
from mimarsinan.spiking.segment_boundary import (
    boundary_normalization_scales,
    decode_segment_output_torch,
    encode_compute_boundary,
    normalize_boundary_slices_torch,
    warn_once_lossy_negative_clamp,
)


class HybridRateForwardMixin(HybridFlowHost):
    """LIF rate-coded hybrid stage orchestration."""

    def _forward_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-coded forward; encoding LIF ComputeOps pass spike trains to the next neural segment."""
        batch_size = x.shape[0]
        device = x.device
        T = self.simulation_length

        x_compute = x.to(COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}
        state_buffer_spikes: Dict[int, torch.Tensor] = {}
        # [C2] host-read decode side channel: per-node membrane terms for the
        # final logits only; the count currency below never carries them.
        readout_corrections: Dict[int, torch.Tensor] = {}

        remaining = dict(self._build_consumer_counts())
        node_output_shifts = getattr(self.hybrid_mapping, "node_output_shifts", None)
        is_ttfs_family = requires_ttfs_firing(self.spiking_mode)
        # Rate/LIF host ComputeOps run with (1, 1) scales, leaving value-domain
        # results in the state buffer; TTFS host ops already transcode to wire
        # via resolve_stage_compute_scales.
        wire_divisors = (
            {} if is_ttfs_family
            else boundary_normalization_scales(self.hybrid_mapping)
        )

        def _ctx_factory(stage_index, stage, buf):
            return HybridStageContext(
                stage_index=stage_index,
                stage=stage,
                state_buffer=buf,
                remaining=remaining,
                state_buffer_spikes=state_buffer_spikes,
                recorder=self._recorder,
            )

        def _on_neural_rate(ctx: HybridStageContext) -> None:
            stage = ctx.stage
            spikes_buffer = ctx.state_buffer_spikes
            assert spikes_buffer is not None, "_ctx_factory always supplies state_buffer_spikes"
            seg_input_rates = self._assemble_segment_input(
                stage.input_map, ctx.state_buffer, batch_size, device
            )
            seg_input_rates = normalize_boundary_slices_torch(
                stage.input_map, seg_input_rates, wire_divisors,
            )
            seg_input_rates = self._apply_input_shifts(stage.input_map, seg_input_rates)
            warn_once_lossy_negative_clamp(stage.name, seg_input_rates)
            seg_input_rates_clamped = seg_input_rates.clamp(0.0, 1.0)
            spike_train = self._encode_segment_input(
                stage,
                seg_input_rates_clamped,
                spikes_buffer,
                T=T,
                batch_size=batch_size,
                device=device,
            )

            recorder_seg: SegmentSpikeRecord | None = None
            if ctx.recorder is not None:
                if is_cascaded_ttfs(self.spiking_mode, self.ttfs_cycle_schedule):
                    seg_input_count = spike_train.amax(dim=0)[0]
                else:
                    seg_input_count = spike_train.sum(dim=0)[0]
                recorder_seg = SegmentSpikeRecord(
                    stage_index=ctx.stage_index,
                    stage_name=stage.name,
                    schedule_segment_index=stage.schedule_segment_index,
                    schedule_pass_index=stage.schedule_pass_index,
                    seg_input_rates=seg_input_rates_clamped[0]
                        .detach().to(torch.float32).cpu().numpy().reshape(1, -1),
                    seg_input_spike_count=seg_input_count
                        .to(torch.int64).cpu().numpy(),
                    seg_output_spike_count=np.zeros(0, dtype=np.int64),
                )

            counts = self._run_neural_segment_rate(
                stage,
                input_spike_train=spike_train,
                recorder_seg=recorder_seg,
                readout_corrections=readout_corrections,
            )
            seg_output_rates = decode_segment_output_torch(counts, T)
            self._store_segment_output(
                stage.output_map, ctx.state_buffer, seg_output_rates,
            )

            if recorder_seg is not None and ctx.recorder is not None:
                recorder_seg.seg_output_spike_count = (
                    counts[0].to(torch.int64).cpu().numpy()
                )
                ctx.recorder.segments[ctx.stage_index] = recorder_seg

        def _after_neural_rate(ctx: HybridStageContext) -> None:
            remaining_counts = ctx.remaining
            assert remaining_counts is not None, "_ctx_factory always supplies remaining"
            self._decref_consumers(
                ctx.state_buffer,
                remaining_counts,
                (int(s.node_id) for s in ctx.stage.input_map),
            )

        def _on_compute_rate(ctx: HybridStageContext) -> None:
            op = ctx.stage.compute_op
            assert op is not None
            spikes_buffer = ctx.state_buffer_spikes
            assert spikes_buffer is not None, "_ctx_factory always supplies state_buffer_spikes"
            # TTFS host ops run in the value domain, mirroring the deployed
            # nevresim / SANA-FE runners (apply_ttfs=is_ttfs); LIF stays rate-domain.
            in_scale, out_scale = resolve_stage_compute_scales(
                self.hybrid_mapping,
                op.id,
                apply_ttfs=requires_ttfs_firing(self.spiking_mode),
                op=op,
            )
            gather_buffer = (
                ctx.state_buffer if is_ttfs_family
                else compute_input_state_with_shifts(
                    op, ctx.state_buffer, node_output_shifts,
                )
            )
            result = execute_compute_op_torch(
                op,
                x,
                gather_buffer,
                in_scale=in_scale,
                out_scale=out_scale,
            )
            ctx.state_buffer[op.id] = result.to(COMPUTE_DTYPE)

            spike_train = encode_compute_boundary(
                op=op,
                state_buffer=ctx.state_buffer,
                state_buffer_spikes=spikes_buffer,
                config=self._boundary_config,
                hybrid_mapping=self.hybrid_mapping,
            )
            if spike_train is not None:
                spikes_buffer[op.id] = spike_train

            if ctx.recorder is not None:
                ctx.recorder.compute_outputs[int(op.id)] = (
                    result.detach().to(torch.float32).cpu().numpy()
                )

        def _after_compute_rate(ctx: HybridStageContext) -> None:
            op = ctx.stage.compute_op
            assert op is not None
            remaining_counts = ctx.remaining
            assert remaining_counts is not None, "_ctx_factory always supplies remaining"
            self._decref_consumers(
                ctx.state_buffer,
                remaining_counts,
                (int(src.node_id) for src in op.input_sources.flatten()
                 if isinstance(src, IRSource) and src.node_id >= 0),
            )

        run_hybrid_stages(
            self.hybrid_mapping,
            state_buffer,
            on_neural=_on_neural_rate,
            on_compute=_on_compute_rate,
            after_neural=_after_neural_rate,
            after_compute=_after_compute_rate,
            context_factory=_ctx_factory,
        )

        final = self._gather_final_output(state_buffer, x_compute, batch_size, device)
        logits = final.to(torch.float32) * float(T)
        return apply_membrane_corrections_to_logits(
            logits, readout_corrections, self.hybrid_mapping.output_sources,
        )
