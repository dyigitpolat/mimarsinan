"""A mapped hybrid program + real samples become one sealed deployment bundle.

THE SHAPE. The shipped bitstream is NC=1, so the network's cores run as one
host-mediated PASS each; and a hybrid program's COMPUTE stages run on the host
either way. What the bundle therefore freezes is one neural segment's cores,
plus the per-sample entry raster the host stages produced for it — which is why
a bundle is only executable for the samples it ships.

Nothing here re-derives the device path: the entry raster, the cycle-accurate
twin and the export all come from ``odin_fpga.segment.plan_odin_segment``, the
same call the physical backend makes before it touches a transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from mimarsinan.chip_simulation.hybrid_run.hybrid_semantics import (
    NeuralSegmentResult,
    store_neural_segment_output,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import run_hybrid_stages
from mimarsinan.chip_simulation.odin_fpga.records import BACKEND_NAME
from mimarsinan.chip_simulation.odin_fpga.runner import (
    RAW_INPUT_NODE_ID,
    execute_compute_stage,
)
from mimarsinan.chip_simulation.odin_fpga.segment import (
    COMPUTE_DTYPE,
    SegmentPlan,
    plan_odin_segment,
    segment_record,
)
from mimarsinan.chip_simulation.recording.records import RunRecord
from mimarsinan.chip_simulation.soma_capability import require_soma_law_supported
from mimarsinan.spiking.segment_boundary import (
    boundary_normalization_scales,
    decode_segment_output,
)


class OdinHaccExportRefusal(RuntimeError):
    """This program cannot be frozen into a board-executable bundle."""


@dataclass(frozen=True)
class FrozenSample:
    """One sample resolved to the segment seam: its raster, twin and record."""

    index: int
    label: int
    plan: SegmentPlan
    record: RunRecord


class ProgramFreezer:
    """Resolve every sample of a hybrid program up to the ODIN device seam."""

    def __init__(
        self, mapping: Any, *, contract: Any, timesteps: int, weight_bits: int,
        effective_max_axons: int, membrane_init: int,
        weight_sign_granularity: str,
    ) -> None:
        self.mapping = mapping
        self.contract = contract
        self.timesteps = int(timesteps)
        self.weight_bits = int(weight_bits)
        self.effective_max_axons = int(effective_max_axons)
        self.membrane_init = int(membrane_init)
        self.weight_sign_granularity = str(weight_sign_granularity)
        self.soma_law = contract.soma_law()
        require_soma_law_supported(
            self.soma_law, backend=BACKEND_NAME, context="ProgramFreezer")
        self.wire_divisors = boundary_normalization_scales(mapping)
        self.node_shifts = getattr(mapping, "node_output_shifts", None)
        self._require_one_neural_stage()

    def _require_one_neural_stage(self) -> None:
        neural = [s for s in self.mapping.stages if s.kind == "neural"]
        if len(neural) != 1:
            raise OdinHaccExportRefusal(
                f"the mapped program carries {len(neural)} neural segment(s); a "
                f"deployment bundle freezes ONE segment's cores as host-mediated "
                f"passes, because a second segment's entry raster depends on the "
                f"first segment's device counts and cannot be frozen ahead of the "
                f"run. Map the model to a single neural segment, or extend the "
                f"bundle schema to carry a segment chain")
        if getattr(neural[0], "retimed_level_stages", None):
            raise OdinHaccExportRefusal(
                "the neural segment is re-timed into per-level stages; the "
                "bundle schema freezes one core-per-pass order, not a level "
                "schedule")

    def freeze(self, sample: np.ndarray, *, sample_index: int, label: int
               ) -> FrozenSample:
        """One sample: host stages on the host, the segment resolved to its twin."""
        if sample.ndim != 2 or sample.shape[0] != 1:
            raise ValueError(f"sample must have shape (1, D); got {sample.shape}")
        state_buffer: Dict[int, np.ndarray] = {RAW_INPUT_NODE_ID: sample}
        record = RunRecord(sample_index=int(sample_index), T=self.timesteps)
        captured: List[SegmentPlan] = []

        def _on_neural(stage_index, stage, buffer):
            plan = plan_odin_segment(
                stage=stage, state_buffer=buffer, soma_law=self.soma_law,
                behavior=self.contract.behavior, timesteps=self.timesteps,
                weight_bits=self.weight_bits,
                effective_max_axons=self.effective_max_axons,
                membrane_init=self.membrane_init,
                weight_sign_granularity=self.weight_sign_granularity,
                wire_divisors=self.wire_divisors, node_shifts=self.node_shifts)
            segment, _per_core, seg_output = segment_record(
                plan, stage=stage, stage_index=stage_index,
                windows=plan.trace.window_counts(), counts=plan.twin_counts())
            record.segments[int(stage_index)] = segment
            captured.append(plan)
            store_neural_segment_output(
                self.contract.spiking_mode, stage.output_map, buffer,
                NeuralSegmentResult(inter_stage=decode_segment_output(
                    seg_output, self.timesteps, dtype=COMPUTE_DTYPE)),
            )

        def _on_compute(_stage_index, stage, buffer):
            record.compute_outputs[stage.compute_op.id] = execute_compute_stage(
                self.mapping, stage, sample, buffer,
                node_shifts=self.node_shifts,
                host_device=getattr(self.contract, "host_compute_device", None))

        run_hybrid_stages(
            self.mapping, state_buffer, on_neural=_on_neural,
            on_compute=_on_compute)
        return FrozenSample(
            index=int(sample_index), label=int(label), plan=captured[0],
            record=record)


def readout_core_of(segment_mapping: Any) -> int:
    """The ONE core the segment's outputs are gathered from, or a refusal.

    ``argmax`` over a readout window is only defined when the class scores are
    one core's neurons in order; a split readout would need a gather rule the
    board reader does not implement, and guessing one would report another
    network's accuracy.
    """
    sources = list(np.asarray(segment_mapping.output_sources).flatten())
    cores = {int(source.core_) for source in sources}
    if len(cores) != 1:
        raise OdinHaccExportRefusal(
            f"the segment's {len(sources)} output wire(s) are gathered from "
            f"cores {sorted(cores)}; the bundle's readout rule reads ONE core's "
            f"neurons in order")
    core = cores.pop()
    neurons = [int(source.neuron_) for source in sources]
    if neurons != list(range(len(neurons))):
        raise OdinHaccExportRefusal(
            f"the segment's output wires read core {core} neurons {neurons}, "
            f"not 0..{len(neurons) - 1} in order; the bundle's readout rule "
            f"scores a contiguous prefix of one core")
    return int(core)


def entry_rasters(frozen: Sequence[FrozenSample]) -> List[List[List[int]]]:
    return [[list(row) for row in sample.plan.raster] for sample in frozen]
