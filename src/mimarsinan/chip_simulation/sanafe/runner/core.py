"""SANA-FE backend driver; sole caller of ``sanafe.SpikingChip``."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.execution_bounds import resolve_simulation_step_timeout_s
from mimarsinan.chip_simulation.sanafe.runner.neural_stage import SanafeNeuralStageMixin
from mimarsinan.chip_simulation.sanafe.runner.neural_stage_record import SanafeNeuralStageRecordMixin
from mimarsinan.chip_simulation.sanafe.runner.segment_io import SanafeSegmentIOMixin

import mimarsinan.chip_simulation.sanafe.runner as _runner
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.chip_simulation.sanafe.runner.constants import _COMPUTE_DTYPE, _RAW_INPUT_NODE_ID
from mimarsinan.chip_simulation.sanafe.runner.custom_floorplan import adopt_custom_arch_floorplan  # noqa: E501
from mimarsinan.chip_simulation.sanafe.arch_synth.spec import CUSTOM_PRESET_NAME
from mimarsinan.chip_simulation.sanafe.presets import CUSTOM_ZERO_PRESET, PRESETS
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeArchGeometry, SanafeEnergyBreakdown, SanafeRunRecord, SanafeSegmentRecord,)


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
        ttfs_cycle_schedule: str = "cascaded",
        contract: Any = None,
        log_potential_trace: bool = False,
        log_message_trace: bool = True,
        cores_per_tile: int = 0,
        tile_grid_rows: int = 0,
        tile_grid_cols: int = 0,
        declared_core_capacity: int = 0,
        simulation_step_timeout_s: float | None = None,
        read_final_potentials: bool = False,
        time_host_stages: bool = False,
        pass_transfer: str = VERBATIM,
        host_compute_device: Any = None,
    ):
        if contract is not None:
            behavior = contract.behavior
            ttfs_cycle_schedule = contract.ttfs_cycle_schedule
            # getattr: tolerate contracts pickled before this field existed.
            contract_timeout = getattr(contract, "simulation_step_timeout_s", None)
            if contract_timeout is not None:
                simulation_step_timeout_s = contract_timeout
            host_compute_device = getattr(
                contract, "host_compute_device", host_compute_device,
            )
        # Host ComputeOps are re-derived here, so they must be evaluated where
        # the census/HCM reference evaluated them: a staircase tie decided by a
        # different f32 reduction is one spike at the next segment boundary.
        self.host_compute_device = host_compute_device
        if behavior is None:
            raise TypeError(
                "SanafeRunner needs its semantics DECLARED: pass contract= (the "
                "deployment contract) or behavior= (a NeuralBehaviorConfig, whose "
                "fields are all required). The old fallback manufactured one from "
                "defaulted scalars — another silent source of semantics beside "
                "the SSOT, exactly the bypass the comparator incident rode."
            )
        self._behavior = behavior
        self.spiking_mode = behavior.spiking_mode
        self.thresholding_mode = behavior.thresholding_mode
        self.firing_mode = behavior.firing_mode
        self.ttfs_cycle_schedule = str(ttfs_cycle_schedule)
        behavior.require_backend("sanafe")
        # "" and None both mean "no custom arch": normalize at the boundary
        # so every downstream ``is not None`` gate (arch build, floorplan
        # adoption) never routes a SYNTHESIZED arch down the custom-arch path.
        custom_arch_path = custom_arch_path or None
        if arch_preset == CUSTOM_PRESET_NAME:
            if not custom_arch_path:
                raise ValueError(
                    "sanafe_arch_preset='custom' requires "
                    "sanafe_custom_arch_path (the user architecture YAML to "
                    "load in place of synthesis)"
                )
            preset = CUSTOM_ZERO_PRESET
        elif arch_preset not in PRESETS:
            raise ValueError(
                f"unknown SANA-FE arch preset {arch_preset!r}; "
                f"expected one of {sorted(PRESETS.keys()) + [CUSTOM_PRESET_NAME]}"
            )
        else:
            preset = PRESETS[arch_preset]

        self.mapping = mapping
        self._preset = preset
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
        # [C2] default-off final-membrane read: lands end-of-window soma
        # potentials per core in the segment record (additive; counts untouched).
        self.read_final_potentials = bool(read_final_potentials)
        # [W4.3] default-off host-op wall timing: each run() times its host
        # ComputeOp stages with a fresh StageTimer and surfaces the walls on
        # the per-sample record (additive; empty when off or compute-free).
        self.time_host_stages = bool(time_host_stages)
        self.cores_per_tile = cores_per_tile
        self.tile_grid_rows = int(tile_grid_rows)
        self.tile_grid_cols = int(tile_grid_cols)
        self.declared_core_capacity = int(declared_core_capacity)
        self._sim_timeout_s = resolve_simulation_step_timeout_s(
            simulation_step_timeout_s
        )

        self._arch: Optional[Any] = None
        self._arch_built_for_T: Optional[int] = None
        self._arch_name, self._last_chip = "<unbuilt>", None
        self.pass_transfer = str(pass_transfer)
        self._arch_geometry: Optional[SanafeArchGeometry] = None  # built lazily


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

        from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
            compute_input_state_with_shifts,
            resolve_stage_compute_scales,
        )
        from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import run_hybrid_stages
        from mimarsinan.chip_simulation.hybrid_run.stage_timing import StageTimer

        # [W4.3] per-sample host-op walls: a fresh timer per run so the
        # per-sample record never carries another sample's accumulation.
        stage_timer = StageTimer() if self.time_host_stages else None

        # ONE discipline per run (run_pass_transfer): carrying while another
        # enabled backend collapses would compare two computations.
        state_buffer_spikes = {} if self.pass_transfer == VERBATIM else None

        def _on_neural(stage_index, stage, state_buffer):
            segments[stage_index] = self._run_neural_stage(
                sanafe=sanafe,
                stage=stage,
                stage_index=stage_index,
                state_buffer=state_buffer,
                state_buffer_spikes=state_buffer_spikes,
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
                    op=op,
                )
                result = _runner.execute_compute_op_numpy(
                    op, sample_input,
                    compute_input_state_with_shifts(
                        op, state_buffer,
                        getattr(self.mapping, "node_output_shifts", None),
                    ),
                    in_scale=in_scale, out_scale=out_scale,
                    dtype=_COMPUTE_DTYPE,
                    device=self.host_compute_device,
                )
                out = np.asarray(result, dtype=_COMPUTE_DTYPE)
                state_buffer[op.id] = out
                compute_outputs[op.id] = out

        run_hybrid_stages(
            self.mapping,
            state_buffer,
            on_neural=_on_neural,
            on_compute=_on_compute,
            stage_timer=stage_timer,
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
            compute_stage_walls=(
                stage_timer.compute_stage_walls()
                if stage_timer is not None else []
            ),
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
            tile_grid_rows=self.tile_grid_rows,
            tile_grid_cols=self.tile_grid_cols,
            declared_core_capacity=self.declared_core_capacity,
            custom_arch_path=self.custom_arch_path,
        )
        self._arch_name = spec.name
        self._arch = _runner.build_architecture(
            spec,
            custom_arch_path=self.custom_arch_path,
            thresholding_mode=self.thresholding_mode,
            simulation_length=self.T,
        )
        self._arch_built_for_T = self.T if need_T else None
        if self.custom_arch_path is not None:
            mw, mh = self._adopt_custom_arch_floorplan()
        else:
            self.cores_per_tile = int(spec.cores_per_tile_resolved)
            mw = max(int(spec.mesh_width), 1)
            mh = max(int(spec.mesh_height), 1)
        # Column-major tile coords: x = tile_id // mesh_height, y = tile_id % mesh_height.
        n_tiles = mw * mh
        tiles_xy = [[i // mh, i % mh] for i in range(n_tiles)]
        self._arch_geometry = SanafeArchGeometry(
            width=mw, height=mh, tiles_xy=tiles_xy,
        )

    def _adopt_custom_arch_floorplan(self) -> tuple[int, int]:
        """The loaded user arch IS the floorplan SSOT (see ``custom_floorplan``)."""
        arch = self._arch
        assert arch is not None
        assert self.custom_arch_path is not None
        self.cores_per_tile, mesh = adopt_custom_arch_floorplan(
            arch,
            custom_arch_path=self.custom_arch_path,
            declared_cores_per_tile=self.cores_per_tile,
            declared_tile_grid_rows=self.tile_grid_rows,
            declared_tile_grid_cols=self.tile_grid_cols,
        )
        return mesh

