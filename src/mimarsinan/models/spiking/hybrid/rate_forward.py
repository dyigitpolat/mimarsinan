"""Hybrid rate-coded full-network forward."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import execute_compute_op_torch
from mimarsinan.mapping.ir import IRSource
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


class HybridRateForwardMixin:
    """LIF rate-coded hybrid stage orchestration."""

    def _forward_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-coded forward; encoding LIF ComputeOps pass spike trains to the next neural segment."""
        batch_size = x.shape[0]
        device = x.device
        T = self.simulation_length

        x_compute = x.to(COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}
        state_buffer_spikes: Dict[int, torch.Tensor] = {}
        from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import resolve_stage_compute_scales

        remaining = dict(self._build_consumer_counts())
        from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import (
            HybridStageContext,
            run_hybrid_stages,
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
            seg_input_rates = self._assemble_segment_input(
                stage.input_map, ctx.state_buffer, batch_size, device
            )
            seg_input_rates_clamped = seg_input_rates.clamp(0.0, 1.0)
            spike_train = self._build_segment_input_spike_train(
                stage,
                seg_input_rates_clamped,
                ctx.state_buffer_spikes,
                T=T,
                batch_size=batch_size,
                device=device,
            )

            recorder_seg: SegmentSpikeRecord | None = None
            if ctx.recorder is not None:
                recorder_seg = SegmentSpikeRecord(
                    stage_index=ctx.stage_index,
                    stage_name=stage.name,
                    schedule_segment_index=stage.schedule_segment_index,
                    schedule_pass_index=stage.schedule_pass_index,
                    seg_input_rates=seg_input_rates_clamped[0]
                        .detach().to(torch.float32).cpu().numpy().reshape(1, -1),
                    seg_input_spike_count=spike_train.sum(dim=0)[0]
                        .to(torch.int64).cpu().numpy(),
                    seg_output_spike_count=np.zeros(0, dtype=np.int64),
                )

            counts = self._run_neural_segment_rate(
                stage, input_spike_train=spike_train, recorder_seg=recorder_seg,
            )
            seg_output_rates = counts / float(T)
            self._store_segment_output(
                stage.output_map, ctx.state_buffer, seg_output_rates,
            )

            if recorder_seg is not None and ctx.recorder is not None:
                recorder_seg.seg_output_spike_count = (
                    counts[0].to(torch.int64).cpu().numpy()
                )
                ctx.recorder.segments[ctx.stage_index] = recorder_seg

        def _after_neural_rate(ctx: HybridStageContext) -> None:
            self._decref_consumers(
                ctx.state_buffer,
                ctx.remaining,
                (int(s.node_id) for s in ctx.stage.input_map),
            )

        def _on_compute_rate(ctx: HybridStageContext) -> None:
            op = ctx.stage.compute_op
            assert op is not None
            # Rate/LIF path: state buffer holds spike rates in [0, 1]; do not apply TTFS scales.
            in_scale, out_scale = resolve_stage_compute_scales(
                self.hybrid_mapping, op.id, apply_ttfs=False
            )
            result = execute_compute_op_torch(
                op,
                x,
                ctx.state_buffer,
                in_scale=in_scale,
                out_scale=out_scale,
            )
            ctx.state_buffer[op.id] = result.to(COMPUTE_DTYPE)

            from mimarsinan.spiking.segment_encoding import emit_compute_spike_train

            spike_train = emit_compute_spike_train(
                op=op,
                state_buffer=ctx.state_buffer,
                state_buffer_spikes=ctx.state_buffer_spikes,
                config=self._segment_encoding,
                hybrid_mapping=self.hybrid_mapping,
            )
            if spike_train is not None:
                ctx.state_buffer_spikes[op.id] = spike_train

            if ctx.recorder is not None:
                ctx.recorder.compute_outputs[int(op.id)] = (
                    result.detach().to(torch.float32).cpu().numpy()
                )

        def _after_compute_rate(ctx: HybridStageContext) -> None:
            op = ctx.stage.compute_op
            assert op is not None
            self._decref_consumers(
                ctx.state_buffer,
                ctx.remaining,
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
        return final.to(torch.float32) * float(T)
