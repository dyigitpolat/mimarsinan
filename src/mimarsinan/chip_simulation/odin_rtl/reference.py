"""The cycle-accurate software twin that tells the cosimulation what to inject.

The RTL cores are independent: nothing routes between them on the stock chip
(``SPI_OPEN_LOOP=1``, plan §2.6 — v1 routing is host-mediated). To drive them
the harness must know, per cycle, what each core's axon slots receive; this
module produces exactly that, by executing the SAME gather nevresim executes
(``ComputePolicyBase::get_axon_input``) around the SAME fold the torch twins
execute (``models/spiking/serial/fold.py``).

Feeding those inputs to the RTL and then comparing the RTL's outputs against
this trace's outputs AT EVERY CYCLE is a sound closed-loop equivalence proof:
the entry inputs agree by construction, and if outputs agree at every cycle
then the gathered inputs agree at every cycle too, by induction over cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.spiking.serial.fold import lif_serial_fold

#: ``SpikeSource.core_`` sentinels, mirroring nevresim's ``constants.hpp``.
SOURCE_OFF = -1
SOURCE_INPUT = -2
SOURCE_ALWAYS_ON = -3


class ReferenceTraceError(ValueError):
    """The mapping or the raster is not something the cycle model can execute."""


@dataclass(frozen=True)
class CycleTrace:
    """Per-cycle axon inputs and neuron outputs of every core."""

    inputs: Tuple[Tuple[Tuple[int, ...], ...], ...]
    outputs: Tuple[Tuple[Tuple[int, ...], ...], ...]
    latencies: Tuple[int, ...]
    simulation_length: int
    used_neurons: Tuple[int, ...]

    @property
    def total_cycles(self) -> int:
        return len(self.outputs)

    def window_counts(self) -> Tuple[Tuple[int, ...], ...]:
        """Per-core per-neuron counts over each core's own window ``[lat, lat+S)``.

        The window convention is nevresim's ``SpikeCountRecorder::accumulate``:
        a core contributes only inside its own latency-shifted window.
        """
        totals = [
            [0] * len(self.outputs[0][core]) for core in range(len(self.latencies))
        ]
        for cycle, per_core in enumerate(self.outputs):
            for core, counts in enumerate(per_core):
                local = cycle - self.latencies[core]
                if 0 <= local < self.simulation_length:
                    for neuron, value in enumerate(counts):
                        totals[core][neuron] += int(value)
        return tuple(tuple(row) for row in totals)


def _read(values: Sequence[Any], index: int, what: str) -> Any:
    """Index a gather source loudly: an out-of-range read is a mapping defect."""
    if index < 0 or index >= len(values):
        raise ReferenceTraceError(
            f"{what} {index}, which does not exist ({len(values)} available). "
            f"The axon-source table and the geometry it is gathered from "
            f"disagree, and silently delivering zero would report a DIFFERENT "
            f"network's counts as the deployed ones.")
    return values[index]


def _source_kind(source: Any) -> int:
    if getattr(source, "is_off_", False):
        return SOURCE_OFF
    if getattr(source, "is_input_", False):
        return SOURCE_INPUT
    if getattr(source, "is_always_on_", False):
        return SOURCE_ALWAYS_ON
    return int(source.core_)


def gather_axon_counts(
    mapping: Any, previous_outputs: Sequence[Sequence[int]],
    input_counts: Sequence[int],
) -> Tuple[Tuple[int, ...], ...]:
    """One cycle's per-slot counts for every core, in canonical slot order."""
    gathered: List[Tuple[int, ...]] = []
    for index, core in enumerate(mapping.cores):
        slots: List[int] = []
        for slot, source in enumerate(core.axon_sources):
            kind = _source_kind(source)
            if kind == SOURCE_OFF:
                slots.append(0)
            elif kind == SOURCE_INPUT:
                slots.append(int(_read(
                    input_counts, int(source.neuron_),
                    f"core {index} slot {slot} reads input line")))
            elif kind == SOURCE_ALWAYS_ON:
                slots.append(1)
            else:
                producer = _read(
                    previous_outputs, kind,
                    f"core {index} slot {slot} reads core")
                slots.append(int(_read(
                    producer, int(source.neuron_),
                    f"core {index} slot {slot} reads core {kind} neuron")))
        gathered.append(tuple(slots))
    return tuple(gathered)


def simulate_cycles(
    mapping: Any,
    *,
    soma_law: SomaLaw,
    input_counts: Sequence[Sequence[int]],
    simulation_length: int,
    membrane_init: int = 0,
    chip_latency: int | None = None,
) -> CycleTrace:
    """Run ``simulation_length + chip_latency`` cycles of the per-event chip.

    ``input_counts[cycle][index]`` is the entry raster; a cycle beyond its end
    delivers nothing, exactly as an exhausted spike generator does.
    ``chip_latency`` defaults to the deepest core's own latency, which is the
    shortest run that still covers every core's window ``[lat, lat+S)``;
    ``ChipLatency.calculate()``'s figure is what nevresim's executor uses and is
    accepted here so the two run the SAME number of cycles.
    """
    latencies = tuple(int(core.latency) for core in mapping.cores)
    depth = max(latencies, default=0) if chip_latency is None else int(chip_latency)
    total = int(simulation_length) + depth
    outputs: List[List[int]] = [
        [0] * int(core.neurons_per_core) for core in mapping.cores
    ]
    membranes = [
        torch.full((1, int(core.neurons_per_core)), float(membrane_init),
                   dtype=torch.float64)
        for core in mapping.cores
    ]
    weights = [
        torch.as_tensor(np.asarray(core.get_core_matrix(), dtype=np.float64).T.copy(),
                        dtype=torch.float64)
        for core in mapping.cores
    ]
    thresholds = [
        torch.tensor(float(core.threshold), dtype=torch.float64)
        for core in mapping.cores
    ]

    trace_inputs: List[Tuple[Tuple[int, ...], ...]] = []
    trace_outputs: List[Tuple[Tuple[int, ...], ...]] = []
    for cycle in range(total):
        entry = (
            input_counts[cycle] if cycle < len(input_counts)
            else [0] * (len(input_counts[0]) if input_counts else 0)
        )
        gathered = gather_axon_counts(mapping, outputs, entry)
        trace_inputs.append(gathered)
        for index, core in enumerate(mapping.cores):
            if cycle < latencies[index]:
                continue
            events = torch.as_tensor(
                [list(gathered[index])], dtype=torch.float64)
            counts = lif_serial_fold(
                membranes[index], weights[index], events, thresholds[index],
                soma_law=soma_law,
            )
            outputs[index] = [int(value) for value in counts[0].tolist()]
        trace_outputs.append(tuple(tuple(row) for row in outputs))
    return CycleTrace(
        inputs=tuple(trace_inputs),
        outputs=tuple(trace_outputs),
        latencies=latencies,
        simulation_length=int(simulation_length),
        used_neurons=tuple(
            int(core.neurons_per_core) - int(core.available_neurons or 0)
            for core in mapping.cores
        ),
    )


def per_slot_counts_by_cycle(trace: CycleTrace) -> Tuple[Dict[int, Tuple[int, ...]], ...]:
    """The trace's inputs reshaped as the cosimulation's per-cycle injection plan."""
    return tuple(
        {core: counts for core, counts in enumerate(per_core)}
        for per_core in trace.inputs
    )
