"""Hybrid mapping TTFS segment and forward execution."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage
from mimarsinan.models.spiking.spiking_config import (
    COMPUTE_DTYPE,
    TTFS_FIRING_MODES,
    TTFS_SPIKING_MODES,
)


class HybridTtfsStepMixin:

    _TTFS_FIRING_MODES = TTFS_FIRING_MODES
    _TTFS_SPIKING_MODES = TTFS_SPIKING_MODES
    """TTFS neural segments and state-buffer TTFS forward."""

    def _run_neural_segment_ttfs(
        self,
        stage: HybridStage,
        *,
        input_activations: torch.Tensor,
        quantized: bool = False,
    ) -> torch.Tensor:
        """TTFS segment via shared ``TtfsAnalyticalExecutor``."""
        from mimarsinan.chip_simulation.ttfs.ttfs_executor import TtfsAnalyticalExecutor

        mapping = stage.hard_core_mapping
        assert mapping is not None
        device = input_activations.device
        inp = input_activations.detach().cpu().numpy().astype(np.float64)
        mode = "ttfs_quantized" if quantized else "ttfs"
        result = TtfsAnalyticalExecutor().run_segment(
            mapping, inp,
            simulation_length=self.simulation_length,
            spiking_mode=mode,
        )
        return torch.tensor(result.inter_stage, dtype=COMPUTE_DTYPE, device=device)

    def _forward_ttfs(self, x: torch.Tensor) -> torch.Tensor:
        """TTFS forward via state buffer; rescales ComputeOp bias via ``node_activation_scales``."""
        T = self.simulation_length
        batch_size = x.shape[0]
        device = x.device
        quantized = self.spiking_mode == "ttfs_quantized"

        x_compute = x.to(COMPUTE_DTYPE)
        state_buffer: Dict[int, torch.Tensor] = {-2: x_compute}
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
            )

        def _on_neural_ttfs(ctx: HybridStageContext) -> None:
            from mimarsinan.chip_simulation.ttfs.ttfs_executor import (
                run_ttfs_contract_neural_stage,
            )

            state_np = {
                k: v.detach().cpu().numpy().astype(np.float64)
                for k, v in ctx.state_buffer.items()
            }
            mode = "ttfs_quantized" if quantized else "ttfs"
            run_ttfs_contract_neural_stage(
                self.hybrid_mapping,
                ctx.stage,
                ctx.stage_index,
                state_np,
                simulation_length=self.simulation_length,
                spiking_mode=mode,
            )
            for s in ctx.stage.output_map:
                ctx.state_buffer[s.node_id] = torch.tensor(
                    state_np[s.node_id], dtype=COMPUTE_DTYPE, device=device,
                )

        def _after_neural_ttfs(ctx: HybridStageContext) -> None:
            self._decref_consumers(
                ctx.state_buffer,
                ctx.remaining,
                (int(s.node_id) for s in ctx.stage.input_map),
            )

        def _on_compute_ttfs(ctx: HybridStageContext) -> None:
            from mimarsinan.chip_simulation.ttfs.ttfs_executor import (
                run_ttfs_contract_compute_stage,
            )

            op = ctx.stage.compute_op
            assert op is not None
            state_np = {
                k: v.detach().cpu().numpy().astype(np.float64)
                for k, v in ctx.state_buffer.items()
            }
            sample = x_compute.detach().cpu().numpy()
            result = run_ttfs_contract_compute_stage(
                self.hybrid_mapping, ctx.stage, state_np, sample,
            )
            ctx.state_buffer[op.id] = torch.tensor(
                result.output, dtype=COMPUTE_DTYPE, device=device,
            )

        def _after_compute_ttfs(ctx: HybridStageContext) -> None:
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
            on_neural=_on_neural_ttfs,
            on_compute=_on_compute_ttfs,
            after_neural=_after_neural_ttfs,
            after_compute=_after_compute_ttfs,
            context_factory=_ctx_factory,
        )

        final = self._gather_final_output(state_buffer, x_compute, batch_size, device)
        # Hybrid TTFS returns count-scaled logits (× simulation_steps) for HCM legacy.
        return final.to(torch.float32) * float(T)
