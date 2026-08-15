"""Hybrid multi-segment nevresim execution."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping, SegmentIOSlice
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    apply_input_shifts_numpy,
    assemble_segment_input_numpy,
    compute_input_state_with_shifts,
    execute_compute_op_numpy,
    gather_final_output_numpy,
    resolve_stage_compute_scales,
    store_segment_output_numpy,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import (
    run_hybrid_stages,
)
from mimarsinan.chip_simulation.spiking_semantics import is_analytical_ttfs, requires_ttfs_firing
from mimarsinan.chip_simulation.simulation_runner.emit import (
    _PreparedSegment,
    prepare_all_segments,
)
from mimarsinan.chip_simulation.simulation_runner.carry import (
    assemble_carried_input_train,
    publish_segment_trains,
)
from mimarsinan.chip_simulation.simulation_runner.segment_run import (
    run_prepared_segment,
)
from mimarsinan.chip_simulation.simulation_runner.host_contract import SimulationHostContract
from mimarsinan.chip_simulation.simulation_runner.membrane_probe import (
    stash_membrane_corrections,
)
from mimarsinan.models.spiking.hybrid.membrane_readout import (
    apply_membrane_corrections_numpy,
)
from mimarsinan.spiking.segment_boundary import (
    boundary_normalization_scales,
    normalize_boundary_slices_numpy,
)



class SimulationHybridMixin(SimulationHostContract):
    @staticmethod
    def _get_compute_op_output_size(op: ComputeOp, state_sizes: Dict[int, int]) -> int:
        """Infer flat output size for a ComputeOp. Uses output_shape when set, else dummy execute."""
        if op.output_shape is not None:
            return int(np.prod(op.output_shape))
        batch_size = 1
        dummy_input = torch.zeros(batch_size, max(state_sizes.get(-2, 1), 1), dtype=torch.float32)
        dummy_buffers = {
            k: torch.zeros(batch_size, sz, dtype=torch.float32)
            for k, sz in state_sizes.items()
            if k >= 0
        }
        result = op.execute(dummy_input, dummy_buffers)
        return int(result.shape[1])

    def _prepare_all_segments(
        self, hybrid: HybridHardCoreMapping
    ) -> Dict[int, _PreparedSegment]:
        """Emit params and compile every segment binary (see ``emit``)."""
        return prepare_all_segments(self, hybrid)

    def _run_neural_segment_precompiled(
        self,
        prepared: _PreparedSegment,
        input_data: list,
        num_proc: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None, list | None]:
        """Run a neural segment's pre-compiled binary (see ``segment_run``)."""
        return run_prepared_segment(
            prepared,
            input_data,
            simulation_length=int(self.simulation_length),
            spike_generation_mode=self.spike_generation_mode,
            timeout_s=self.simulation_step_timeout_s,
            num_proc=num_proc,
        )

    def _raw_to_rates(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw nevresim output to [0,1] rates.

        Analytical TTFS returns real-valued activations directly; LIF and cascaded
        ``ttfs_cycle_based`` return spike counts decoded as ``count / T``.
        """
        if is_analytical_ttfs(self.spiking_mode):
            return raw
        return raw / max(int(self.simulation_length), 1)

    def _run_hybrid(self, hybrid: HybridHardCoreMapping) -> float:
        """Execute a multi-stage hybrid mapping using the state buffer."""
        stages = hybrid.stages
        num_samples = len(self.test_data)
        is_ttfs = requires_ttfs_firing(self.spiking_mode)

        original_input = np.stack([d[0] for d in self.test_data])
        original_input = original_input.reshape(original_input.shape[0], -1)
        state_buffer: Dict[int, np.ndarray] = {-2: original_input}

        prepared_segments = self._prepare_all_segments(hybrid)

        # Rate/LIF host ComputeOps run with (1, 1) scales (value-domain buffers);
        # TTFS transcodes via resolve_stage_compute_scales instead.
        wire_divisors = {} if is_ttfs else boundary_normalization_scales(hybrid)
        node_output_shifts = getattr(hybrid, "node_output_shifts", None)

        seg_counter = 0
        # [C2] host-read decode side channel keyed by node id; count currencies
        # in the state buffer never carry it (mirrors the torch flow).
        membrane_corrections: Dict[int, np.ndarray] = {}
        # Verbatim pass-carry: per-node (N, T, size) trains a later pass of the
        # same segment replays. Populated only when a producing segment ran.
        state_buffer_trains: Dict[int, np.ndarray] = {}

        def on_neural(_idx, stage, buf):
            nonlocal seg_counter
            seg_mapping = stage.hard_core_mapping
            assert seg_mapping is not None
            seg_input = self._assemble_segment_input_np(
                stage.input_map, buf, num_samples
            )
            if not is_ttfs:
                seg_input = normalize_boundary_slices_numpy(
                    stage.input_map, seg_input, wire_divisors,
                )
                seg_input = apply_input_shifts_numpy(
                    stage.input_map, seg_input, node_output_shifts,
                )
            input_size = seg_input.shape[1]
            prepared = prepared_segments[seg_counter]
            input_train = None
            if prepared.input_mode == "SpikeTrain":
                input_train = assemble_carried_input_train(
                    stage, seg_input, state_buffer_trains,
                    spike_generation_mode=self.spike_generation_mode,
                    T=int(self.simulation_length),
                )
                seg_data = [
                    (input_train[i].reshape(-1), np.zeros(1))
                    for i in range(num_samples)
                ]
            else:
                seg_data = [(seg_input[i], np.zeros(1)) for i in range(num_samples)]
            print(f"  Running neural segment '{stage.name}' (input_size={input_size})")
            raw_output, membranes, spike_trains = (
                self._run_neural_segment_precompiled(prepared, seg_data))
            seg_counter += 1
            recorder = getattr(self, "stage_count_recorder", None)
            if recorder is not None:
                recorder(stage, raw_output)
            if membranes is not None:
                stash_membrane_corrections(
                    hybrid, stage, membranes, membrane_corrections,
                    half_step_charge=self.membrane_half_step_charge,
                )
            rates = self._raw_to_rates(raw_output)
            store_segment_output_numpy(stage.output_map, buf, rates)
            if spike_trains is not None and prepared.carried_output_node_ids:
                publish_segment_trains(
                    stage, prepared, spike_trains, input_train,
                    T=int(self.simulation_length),
                    state_buffer_trains=state_buffer_trains,
                )

        def on_compute(_idx, stage, buf):
            assert stage.compute_op is not None
            print(f"  Executing compute op '{stage.name}' on host")
            op_id = stage.compute_op.id
            ttfs_in_scale, ttfs_out_scale = resolve_stage_compute_scales(
                hybrid, op_id, apply_ttfs=is_ttfs, op=stage.compute_op,
            )
            gather_buf = (
                buf if is_ttfs
                else compute_input_state_with_shifts(
                    stage.compute_op, buf, node_output_shifts,
                )
            )
            buf[op_id] = execute_compute_op_numpy(
                stage.compute_op,
                original_input,
                gather_buf,
                in_scale=ttfs_in_scale,
                out_scale=ttfs_out_scale,
                # Bit-parity with the census flow: half-grid wire ties must
                # be decided by the SAME f32 reduction (device) everywhere.
                device=self.host_compute_device,
            )

        # [W4.3] the opt-in stage timer wraps host ComputeOps only; neural
        # segments are the chip simulator's to measure.
        run_hybrid_stages(
            hybrid, state_buffer, on_neural=on_neural, on_compute=on_compute,
            stage_timer=self.stage_timer,
        )

        final_output = gather_final_output_numpy(
            hybrid.output_sources, state_buffer, original_input, num_samples
        )
        if membrane_corrections:
            print(
                "  [C2] applying the deployed membrane decode to the probe's "
                f"final read ({len(membrane_corrections)} eligible node(s))"
            )
            final_output = apply_membrane_corrections_numpy(
                final_output,
                membrane_corrections,
                hybrid.output_sources,
                simulation_length=int(self.simulation_length),
            )
        predictions = np.argmax(final_output, axis=1)

        print("Evaluating simulator output...")
        return self._evaluate_chip_output(predictions)

    @staticmethod
    def _assemble_segment_input_np(
        input_map: list[SegmentIOSlice],
        state_buffer: Dict[int, np.ndarray],
        num_samples: int,
    ) -> np.ndarray:
        return assemble_segment_input_numpy(input_map, state_buffer, num_samples)

