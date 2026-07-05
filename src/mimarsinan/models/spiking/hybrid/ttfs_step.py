"""Hybrid mapping TTFS segment and forward execution."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    gather_final_output_numpy,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import (
    HybridStageContext,
    run_hybrid_stages,
)
from mimarsinan.chip_simulation.spiking_semantics import is_synchronized_ttfs
from mimarsinan.chip_simulation.ttfs.ttfs_executor import (
    TtfsAnalyticalExecutor,
    run_ttfs_contract_compute_stage,
    run_ttfs_contract_neural_stage,
)
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage
from mimarsinan.models.spiking.hybrid.host import HybridFlowHost
from mimarsinan.models.spiking.spiking_config import (
    COMPUTE_DTYPE,
    TTFS_FIRING_MODES,
    TTFS_SPIKING_MODES,
)


class HybridTtfsStepMixin(HybridFlowHost):
    """TTFS neural segments and state-buffer TTFS forward."""

    _TTFS_FIRING_MODES = TTFS_FIRING_MODES
    _TTFS_SPIKING_MODES = TTFS_SPIKING_MODES

    def _run_neural_segment_ttfs(
        self,
        stage: HybridStage,
        *,
        input_activations: torch.Tensor,
        quantized: bool = False,
    ) -> torch.Tensor:
        """TTFS segment via shared ``TtfsAnalyticalExecutor``."""
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
        """TTFS forward on a float64 NUMPY state buffer (the analytical executor's
        native domain): one host conversion in, one device conversion out — no
        per-stage torch round-trips (W3 wall)."""
        T = self.simulation_length
        batch_size = x.shape[0]
        device = x.device
        quantized = self.spiking_mode != "ttfs"
        mode = "ttfs_quantized" if quantized else "ttfs"
        quantize_input = is_synchronized_ttfs(
            self.spiking_mode, self.ttfs_cycle_schedule,
        )

        x_np = x.detach().to(COMPUTE_DTYPE).cpu().numpy()
        state_buffer: Dict[int, np.ndarray] = {-2: x_np}

        remaining = dict(self._build_consumer_counts())

        def _ctx_factory(stage_index, stage, buf):
            return HybridStageContext(
                stage_index=stage_index,
                stage=stage,
                state_buffer=buf,
                remaining=remaining,
            )

        def _on_neural_ttfs(ctx: HybridStageContext) -> None:
            run_ttfs_contract_neural_stage(
                self.hybrid_mapping,
                ctx.stage,
                ctx.stage_index,
                ctx.state_buffer,
                simulation_length=self.simulation_length,
                spiking_mode=mode,
                quantize_input_to_ttfs_grid=quantize_input,
            )

        def _after_neural_ttfs(ctx: HybridStageContext) -> None:
            remaining_counts = ctx.remaining
            assert remaining_counts is not None, "_ctx_factory always supplies remaining"
            self._decref_consumers(
                ctx.state_buffer,
                remaining_counts,
                (int(s.node_id) for s in ctx.stage.input_map),
            )

        def _on_compute_ttfs(ctx: HybridStageContext) -> None:
            op = ctx.stage.compute_op
            assert op is not None
            # The contract stage stores its float64 output into the buffer itself.
            run_ttfs_contract_compute_stage(
                self.hybrid_mapping, ctx.stage, ctx.state_buffer, x_np,
            )

        def _after_compute_ttfs(ctx: HybridStageContext) -> None:
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
            on_neural=_on_neural_ttfs,
            on_compute=_on_compute_ttfs,
            after_neural=_after_neural_ttfs,
            after_compute=_after_compute_ttfs,
            context_factory=_ctx_factory,
        )

        final_np = gather_final_output_numpy(
            self.hybrid_mapping.output_sources, state_buffer, x_np, batch_size,
        )
        final = torch.from_numpy(final_np).to(device=device, dtype=torch.float32)
        return final * float(T)
