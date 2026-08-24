"""Shared fixtures for the GENERATED-core cosimulation gates (plan §7 row 20).

Nothing here re-derives a geometry, a law, an event order or a packing: the spec
is projected from a declared core type and the resolved ``SomaLaw``, the RTL is
expanded from ``hw/gen``, the packer tables come from ``mapping/export/odin_gen``
and the expected numbers come from the same per-cycle policy the deployed torch
executors are built from. That is the whole point of the gate.
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
from mimarsinan.mapping.export.odin_gen import CoreSpec
from mimarsinan.mapping.export.odin_gen.packer import build_variant_core_image
from mimarsinan.mapping.export.odin_gen.variants import (  # noqa: F401
    SIGN_GRANULARITY,
    WEIGHT_BITS,
    per_event_law,
    spec_for,
    sync_fire_law,
    unbounded_law,
)
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping


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
    ChipLatency(mapping).calculate()
    return mapping


def images_for(mapping: HardCoreMapping, spec: CoreSpec, *, membrane_init: int = 0):
    """The packer tables of every core, with the REAL feasibility gate applied."""
    from mimarsinan.mapping.export.odin_gen.feasibility import check_variant_theta

    return tuple(
        build_variant_core_image(
            core, spec=spec, core_index=index,
            theta=check_variant_theta(
                core.threshold, spec=spec, core_index=index),
            membrane_init=membrane_init,
        )
        for index, core in enumerate(mapping.cores)
    )


def thetas_of(mapping: HardCoreMapping) -> dict:
    return {index: int(core.threshold) for index, core in enumerate(mapping.cores)}


@dataclass(frozen=True)
class SampleTrace:
    """One sample: its reference trace and the injection plan it implies."""

    trace: CycleTrace
    per_cycle: tuple


def traces_for(mapping: HardCoreMapping, rasters, *, soma_law: SomaLaw,
               simulation_length: int, membrane_init: int = 0,
               chip_latency: int | None = None) -> list[SampleTrace]:
    """The cycle-accurate reference for each sample, with fresh membranes."""
    return [
        SampleTrace(
            trace=(trace := simulate_cycles(
                mapping, soma_law=soma_law, input_counts=raster,
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


def suprathreshold_multiplicity(mapping: HardCoreMapping,
                                samples: Sequence[SampleTrace]) -> int:
    """The most events that alone reach theta any neuron receives in one cycle.

    Two of them in one cycle is the ONLY stimulus that separates the per-cycle
    law from the event-serial one: a serial fold fires on each, a per-cycle
    compare fires once, so a fixture whose multiplicity is 1 cannot tell the
    two laws apart.
    """
    worst = 0
    for sample in samples:
        for per_core in sample.trace.inputs:
            for core, slots in zip(mapping.cores, per_core):
                weights = np.asarray(core.get_core_matrix(), dtype=np.float64)
                reaching = (weights >= float(core.threshold)).astype(np.float64)
                arrivals = np.asarray(slots, dtype=np.float64)
                worst = max(worst, int((arrivals @ reaching).max(initial=0)))
    return worst


def rtl_cycle_multiplicity(result, neurons: Sequence[int]) -> int:
    """The most spikes any neuron emitted in one cycle ON THE WIRE."""
    return max(
        max(result.cycle_counts(sample, cycle, core, int(count)))
        for sample in range(result.plan.samples)
        for cycle in range(result.plan.cycles_per_sample)
        for core, count in enumerate(neurons)
    )


def report(label: str, result) -> None:
    """The honest per-gate line the runner's transcript is read from."""
    print(f"[odin-gen] {label} engine={result.build.engine} "
          f"build={result.build.build_seconds:.1f}s cached={result.build.cached} "
          f"sim={result.run.seconds:.1f}s cycles={result.capture.cycles} "
          f"tokens={result.token_count} events={len(result.capture.events)} "
          f"spec_failures={result.capture.spec_failures} "
          f"rail_failures={result.capture.rail_failures}")


def timed(label: str):
    """A context manager that prints an honest wall time for a gate's phase."""

    class _Timer:
        def __enter__(self):
            self.started = time.monotonic()
            return self

        def __exit__(self, *exc):
            self.seconds = time.monotonic() - self.started
            print(f"[odin-gen] {label}: {self.seconds:.1f} s")
            return False

    return _Timer()
