"""Run one sample through the hybrid program with ODIN cores as the device."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    compute_input_state_with_shifts,
    execute_compute_op_numpy,
    resolve_stage_compute_scales,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_semantics import (
    NeuralSegmentResult,
    store_neural_segment_output,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import run_hybrid_stages
from mimarsinan.chip_simulation.hybrid_run.stage_timing import StageTimer
from mimarsinan.chip_simulation.odin_fpga.records import (
    BACKEND_NAME,
    OdinFpgaRunRecord,
)
from mimarsinan.chip_simulation.odin_fpga.segment import COMPUTE_DTYPE, run_odin_segment
from mimarsinan.chip_simulation.odin_fpga.transport import DeviceSession
from mimarsinan.chip_simulation.recording.records import RunRecord
from mimarsinan.chip_simulation.soma_capability import require_soma_law_supported
from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.spiking.segment_boundary import (
    boundary_normalization_scales,
    decode_segment_output,
)

RAW_INPUT_NODE_ID = -2


def execute_compute_stage(
    mapping: Any, stage: Any, sample_input: np.ndarray, buffer: Dict[int, Any],
    *, node_shifts: Any, host_device: Any,
) -> np.ndarray:
    """One HOST compute stage of a hybrid program, in the deployment's dtype.

    Shared by every runner that drives ODIN cores: the chip executes neural
    segments only, and what feeds them is this value-domain evaluation.
    """
    op = stage.compute_op
    assert op is not None
    in_scale, out_scale = resolve_stage_compute_scales(
        mapping, op.id, apply_ttfs=False, op=op)
    result = execute_compute_op_numpy(
        op, sample_input,
        compute_input_state_with_shifts(op, buffer, node_shifts),
        in_scale=in_scale, out_scale=out_scale, dtype=COMPUTE_DTYPE,
        device=host_device)
    out = np.asarray(result, dtype=COMPUTE_DTYPE)
    buffer[op.id] = out
    return out


class OdinFpgaRunner:
    """Execute a hybrid program with every neural segment on an ODIN device."""

    def __init__(
        self,
        mapping: Any,
        simulation_length: int,
        *,
        contract: Any,
        transport: Any,
        weight_bits: int,
        effective_max_axons: int,
        membrane_init: int = 0,
        weight_sign_granularity: str = "per_axon",
        pass_transfer: str,
        time_host_stages: bool = True,
    ) -> None:
        self.mapping = mapping
        self.T = int(simulation_length)
        self.contract = contract
        self.transport = transport
        self.weight_bits = int(weight_bits)
        self.effective_max_axons = int(effective_max_axons)
        self.membrane_init = int(membrane_init)
        self.weight_sign_granularity = str(weight_sign_granularity)
        self.time_host_stages = bool(time_host_stages)
        self.soma_law = contract.soma_law()
        self.spiking_mode = contract.spiking_mode
        if requires_ttfs_firing(self.spiking_mode):
            raise ValueError(
                f"{BACKEND_NAME}: spiking_mode={self.spiking_mode!r} is a TTFS "
                f"family and the ODIN crossbar implements the event/rate LIF "
                f"soma only — no timing-domain executor exists on this device")
        require_soma_law_supported(
            self.soma_law, backend=BACKEND_NAME, context="OdinFpgaRunner")
        if str(pass_transfer) == VERBATIM:
            raise ValueError(
                f"{BACKEND_NAME}: pass_transfer='verbatim' asks the device to "
                f"replay a producer's raster across a pass boundary, which this "
                f"transport does not buffer. The run-level discipline "
                f"(models/spiking/hybrid/carry.run_pass_transfer) collapses "
                f"every enabled backend's boundaries once this one is enabled; "
                f"a verbatim runner here would compare two computations.")
        self.pass_transfer = str(pass_transfer)

    def run(self, sample_input: np.ndarray, sample_index: int) -> OdinFpgaRunRecord:
        """One sample: host ops on the contract's device, segments on the chip."""
        if sample_input.ndim != 2 or sample_input.shape[0] != 1:
            raise ValueError(
                f"sample_input must have shape (1, D); got {sample_input.shape}")
        state_buffer: Dict[int, np.ndarray] = {RAW_INPUT_NODE_ID: sample_input}
        record = RunRecord(sample_index=int(sample_index), T=self.T)
        timings: List[Any] = []
        per_cycle: Dict[int, Dict] = {}
        wire_divisors = boundary_normalization_scales(self.mapping)
        node_shifts = getattr(self.mapping, "node_output_shifts", None)
        stage_timer = StageTimer() if self.time_host_stages else None

        def _on_neural(stage_index, stage, buffer):
            outcome = run_odin_segment(
                stage=stage, stage_index=stage_index, state_buffer=buffer,
                transport=self.transport, soma_law=self.soma_law,
                behavior=self.contract.behavior, timesteps=self.T,
                weight_bits=self.weight_bits,
                effective_max_axons=self.effective_max_axons,
                membrane_init=self.membrane_init,
                weight_sign_granularity=self.weight_sign_granularity,
                wire_divisors=wire_divisors, node_shifts=node_shifts,
            )
            record.segments[int(stage_index)] = outcome.record
            timings.append(outcome.timing)
            per_cycle[int(stage_index)] = outcome.counts
            store_neural_segment_output(
                self.spiking_mode, stage.output_map, buffer,
                NeuralSegmentResult(inter_stage=decode_segment_output(
                    outcome.seg_output_counts, self.T, dtype=COMPUTE_DTYPE)),
            )

        def _on_compute(_stage_index, stage, buffer):
            record.compute_outputs[stage.compute_op.id] = execute_compute_stage(
                self.mapping, stage, sample_input, buffer,
                node_shifts=node_shifts,
                host_device=getattr(self.contract, "host_compute_device", None))

        with DeviceSession(self.transport):
            run_hybrid_stages(
                self.mapping, state_buffer,
                on_neural=_on_neural, on_compute=_on_compute,
                stage_timer=stage_timer,
            )
        return OdinFpgaRunRecord(
            transport=str(getattr(self.transport, "name", "unknown")),
            record=record, timings=timings,
            compute_stage_walls=(
                stage_timer.compute_stage_walls() if stage_timer else []),
            per_cycle_counts=per_cycle,
        )
