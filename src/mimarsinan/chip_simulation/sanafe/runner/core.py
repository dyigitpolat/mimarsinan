"""SANA-FE backend driver; sole caller of ``sanafe.SpikingChip``."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.sanafe.runner.neural_stage import SanafeNeuralStageMixin
from mimarsinan.chip_simulation.sanafe.runner.neural_stage_record import SanafeNeuralStageRecordMixin
from mimarsinan.chip_simulation.sanafe.runner.segment_io import SanafeSegmentIOMixin

import mimarsinan.chip_simulation.sanafe.runner as _runner
from mimarsinan.chip_simulation.sanafe.runner.constants import _COMPUTE_DTYPE, _RAW_INPUT_NODE_ID
from mimarsinan.chip_simulation.sanafe.presets import PRESETS
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeArchGeometry,
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
)


class SanafeRunner(SanafeNeuralStageMixin, SanafeNeuralStageRecordMixin, SanafeSegmentIOMixin):
    """Run one hybrid-mapping sample through SANA-FE."""

    def __init__(
        self,
        mapping: Any,
        simulation_length: int,
        *,
        behavior: NeuralBehaviorConfig | None = None,
        arch_preset: str = "loihi",
        custom_arch_path: Optional[str] = None,
        thresholding_mode: str = "<=",
        spiking_mode: str = "lif",
        firing_mode: str = "Default",
        ttfs_cycle_schedule: str = "cascaded",
        contract: Any = None,
        log_potential_trace: bool = False,
        log_message_trace: bool = True,
        cores_per_tile: int = 0,
    ):
        if contract is not None:
            behavior = contract.behavior
            ttfs_cycle_schedule = contract.ttfs_cycle_schedule
        if behavior is None:
            behavior = NeuralBehaviorConfig(
                spiking_mode=str(spiking_mode),
                firing_mode=str(firing_mode),
                thresholding_mode=str(thresholding_mode),
                spike_generation_mode="Uniform",
            )
        self._behavior = behavior
        self.spiking_mode = behavior.spiking_mode
        self.thresholding_mode = behavior.thresholding_mode
        self.firing_mode = behavior.firing_mode
        self.ttfs_cycle_schedule = str(ttfs_cycle_schedule)
        behavior.require_backend("sanafe")
        if arch_preset not in PRESETS:
            raise ValueError(
                f"unknown SANA-FE arch preset {arch_preset!r}; "
                f"expected one of {sorted(PRESETS.keys())}"
            )

        self.mapping = mapping
        self._preset = PRESETS[arch_preset]
        self.T = int(simulation_length)
        self.arch_preset = arch_preset
        self.custom_arch_path = custom_arch_path
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            policy_for_spiking_mode,
        )

        policy_for_spiking_mode(
            self.spiking_mode, self.ttfs_cycle_schedule
        ).require_backend_supported(backend="sanafe", context="SanafeRunner")
        self.log_potential_trace = log_potential_trace
        self.log_message_trace = log_message_trace
        self.cores_per_tile = cores_per_tile

        self._arch: Optional[Any] = None
        self._arch_built_for_T: Optional[int] = None
        self._arch_name: str = "<unbuilt>"
        self._arch_geometry: Optional[SanafeArchGeometry] = None
        self._last_chip: Optional[Any] = None


    def run(self, sample_input: np.ndarray, sample_index: int) -> SanafeRunRecord:
        """Run one sample through every hybrid stage."""
        if sample_input.ndim != 2 or sample_input.shape[0] != 1:
            raise ValueError(
                f"sample_input must have shape (1, D); got {sample_input.shape}"
            )

        has_neural = any(s.kind == "neural" for s in self.mapping.stages)
        if has_neural:
            sanafe = _runner._sanafe()
            self._ensure_arch()
        else:
            sanafe = None

        state_buffer: Dict[int, np.ndarray] = {_RAW_INPUT_NODE_ID: sample_input}
        segments: Dict[int, SanafeSegmentRecord] = {}
        compute_outputs: Dict[int, np.ndarray] = {}

        from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import resolve_stage_compute_scales
        from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import run_hybrid_stages

        def _on_neural(stage_index, stage, state_buffer):
            segments[stage_index] = self._run_neural_stage(
                sanafe=sanafe,
                stage=stage,
                stage_index=stage_index,
                state_buffer=state_buffer,
            )

        def _on_compute(_stage_index, stage, state_buffer):
            from mimarsinan.chip_simulation.ttfs.ttfs_executor import (
                run_ttfs_contract_compute_stage,
            )

            op = stage.compute_op
            assert op is not None
            if _runner.is_ttfs_spiking_mode(self.spiking_mode):
                result = run_ttfs_contract_compute_stage(
                    self.mapping, stage, state_buffer, sample_input,
                )
                compute_outputs[result.op_id] = result.output
            else:
                in_scale, out_scale = resolve_stage_compute_scales(
                    self.mapping,
                    op.id,
                    apply_ttfs=_runner.is_ttfs_spiking_mode(self.spiking_mode),
                )
                result = _runner.execute_compute_op_numpy(
                    op, sample_input, state_buffer,
                    in_scale=in_scale, out_scale=out_scale,
                    dtype=_COMPUTE_DTYPE,
                )
                out = np.asarray(result, dtype=_COMPUTE_DTYPE)
                state_buffer[op.id] = out
                compute_outputs[op.id] = out

        run_hybrid_stages(
            self.mapping,
            state_buffer,
            on_neural=_on_neural,
            on_compute=_on_compute,
        )

        agg_e = SanafeEnergyBreakdown.zero()
        max_sim_time = 0.0
        total_spikes = 0
        total_packets = 0
        for seg in segments.values():
            agg_e = agg_e.add(seg.energy)
            if seg.sim_time_s > max_sim_time:
                max_sim_time = seg.sim_time_s
            total_spikes += seg.spikes
            total_packets += seg.packets_sent

        return SanafeRunRecord(
            arch_preset=self.arch_preset,
            arch_name=self._arch_name,
            sample_index=int(sample_index),
            T=self.T,
            segments=segments,
            compute_outputs=compute_outputs,
            aggregate_energy=agg_e,
            aggregate_sim_time_s=max_sim_time,
            total_spikes=total_spikes,
            total_packets=total_packets,
        )


    def _ensure_arch(self) -> None:
        """Lazily build the shared SANA-FE architecture."""
        from mimarsinan.chip_simulation.spiking_semantics import forces_activation_quantization

        need_T = forces_activation_quantization(self.spiking_mode)
        if self._arch is not None:
            if not need_T or self._arch_built_for_T == self.T:
                return
            self._arch = None
        spec = _runner.derive_arch_spec(
            self.mapping,
            preset_name=self.arch_preset,
            cores_per_tile=self.cores_per_tile,
        )
        self._arch_name = spec.name
        self._arch = _runner.build_architecture(
            spec,
            custom_arch_path=self.custom_arch_path,
            thresholding_mode=self.thresholding_mode,
            simulation_length=self.T,
        )
        self._arch_built_for_T = self.T if need_T else None
        self.cores_per_tile = int(spec.cores_per_tile_resolved)
        # Column-major tile coords: x = tile_id // mesh_height, y = tile_id % mesh_height.
        n_tiles = int(spec.n_tiles)
        mw = max(int(spec.mesh_width), 1)
        mh = max(int(spec.mesh_height), 1)
        tiles_xy = [[i // mh, i % mh] for i in range(n_tiles)]
        self._arch_geometry = SanafeArchGeometry(
            width=mw, height=mh, tiles_xy=tiles_xy,
        )

