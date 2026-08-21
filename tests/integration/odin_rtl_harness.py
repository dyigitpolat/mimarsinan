"""Shared fixtures for the ODIN RTL cosimulation gates (plan §7 rows 15-18).

Every gate here drives the BYTE-IDENTICAL vendored core through the exporter's
own sequencer program. Nothing in this file re-derives a bit layout, an event
order or a drain bound: those come from `mapping/export/odin/` and
`mapping/platform/event_order.py`, which is the point of the gate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest

from mimarsinan.chip_simulation.odin_rtl.reference import (
    CycleTrace,
    per_slot_counts_by_cycle,
    simulate_cycles,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    SimulatorUnavailable,
    available_engine,
    unavailable_reason,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.exporter import export_odin
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

#: The stock-ODIN point, fully written (plan §1.1).
ODIN_LAW = SomaLaw.resolve({
    "spiking_family": "lif", "spiking_variant": "streamed",
    "firing_mode": "Novena", "thresholding_mode": "<=",
    "firing_granularity": "per_event", "membrane_bits": 8,
})

WEIGHT_BITS = 4
EFFECTIVE_MAX_AXONS = 127

#: The 8-bit unsigned membrane's ceiling: theta may sit anywhere in [1, 255],
#: and a theta at the top is the plan's near-ceiling feasibility witness (§11).
MEMBRANE_CEILING = 255


def require_simulator() -> str:
    """The engine to run on, or a LOUD skip that names the path it looked in."""
    try:
        return available_engine()
    except SimulatorUnavailable:
        pytest.skip(unavailable_reason(("verilator", "iverilog+vvp")),
                    allow_module_level=False)
        raise  # pragma: no cover - pytest.skip raises


def hard_core(matrix, *, threshold: float, sources: Sequence[SpikeSource]) -> HardCore:
    """One biasless hard core with integral weights — what the packer consumes."""
    values = np.asarray(matrix, dtype=np.float64)
    core = HardCore(
        axons_per_core=values.shape[0], neurons_per_core=values.shape[1],
        has_bias_capability=False,
    )
    core.core_matrix = values
    core.axon_sources = list(sources)
    core.threshold = float(threshold)
    core.available_axons = 0
    core.available_neurons = 0
    return core


def mapping_of(cores: Sequence[HardCore], output_sources) -> HardCoreMapping:
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = list(cores)
    mapping.output_sources = np.asarray(list(output_sources), dtype=object)
    return mapping


def export_of(mapping: HardCoreMapping, *, membrane_init: int = 0):
    """Run the real exporter — feasibility gates included — over the mapping."""
    return export_odin(
        mapping, soma_law=ODIN_LAW, weight_bits=WEIGHT_BITS,
        weight_sign_granularity="per_axon",
        effective_max_axons=EFFECTIVE_MAX_AXONS, membrane_init=membrane_init,
    )


@dataclass(frozen=True)
class SampleTrace:
    """One sample: its reference trace and the injection plan it implies."""

    trace: CycleTrace
    per_cycle: tuple


def traces_for(mapping: HardCoreMapping, rasters: Sequence[Sequence[Sequence[int]]],
               *, simulation_length: int, membrane_init: int = 0,
               chip_latency: int | None = None) -> list[SampleTrace]:
    """The cycle-accurate reference for each sample, with fresh membranes."""
    return [
        SampleTrace(
            trace=(trace := simulate_cycles(
                mapping, soma_law=ODIN_LAW, input_counts=raster,
                simulation_length=simulation_length,
                membrane_init=membrane_init, chip_latency=chip_latency)),
            per_cycle=per_slot_counts_by_cycle(trace),
        )
        for raster in rasters
    ]


def compare_cycle_counts(result, samples: Sequence[SampleTrace]) -> list[str]:
    """Every (sample, cycle, core, neuron) difference between the RTL and the twin."""
    differences: list[str] = []
    for sample_index, sample in enumerate(samples):
        for cycle, per_core in enumerate(sample.trace.outputs):
            for core, expected in enumerate(per_core):
                got = result.cycle_counts(sample_index, cycle, core, len(expected))
                if tuple(int(v) for v in expected) != got:
                    differences.append(
                        f"sample {sample_index} cycle {cycle} core {core}: "
                        f"rtl={got} reference={tuple(int(v) for v in expected)}")
    return differences


def timed(label: str):
    """A context manager that prints an honest wall time for a gate's phase."""

    class _Timer:
        def __enter__(self):
            self.started = time.monotonic()
            return self

        def __exit__(self, *exc):
            self.seconds = time.monotonic() - self.started
            print(f"[odin-rtl] {label}: {self.seconds:.1f} s")
            return False

    return _Timer()
