"""The freeze's golden gates, each refusing by name (moved out of ``freeze`` unchanged, [U-H], beside the fabric
fit gate of survey gap B6.1)."""

from __future__ import annotations

from typing import Any, Sequence

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle
from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    SHIPPED_CAPTURE_EVENTS,
)


class BundleRefusal(RuntimeError):
    """A golden gate did not hold; no bundle is written."""


def _require_witness_matches_twin(name, index, counts, traces) -> None:
    for sample, trace in enumerate(traces):
        for cycle in range(trace.total_cycles):
            want = tuple(int(v) for v in trace.outputs[cycle][index])
            got = tuple(
                int(counts.get((sample, cycle, bundle.PASS_CORE, neuron), 0))
                for neuron in range(len(want)))
            if got != want:
                raise BundleRefusal(
                    f"{name}: pass {index} sample {sample} cycle {cycle} "
                    f"witnessed {got} against the cycle-accurate twin's "
                    f"{want}. The bundle's expectations would not be a "
                    f"measurement of this network")


def _require_reader_agrees(name, build, traces, plan, cycles, aer) -> None:
    for sample, trace in enumerate(traces):
        mirrored = bundle.pass_stimulus_tokens(
            plan, [trace.inputs[cycle][build.index]
                   for cycle in range(trace.total_cycles)],
            encoding=aer, sample=0, cycles_per_sample=cycles)
        reference = list(build.reference_stimulus(trace))
        if mirrored != reference:
            raise BundleRefusal(
                f"{name}: the shipped bundle reader builds a DIFFERENT stimulus "
                f"for pass {build.index} sample {sample} than the repository's "
                f"own encoder ({len(mirrored)} vs {len(reference)} tokens). The "
                f"board would stimulate a network nobody assembled")


def _require_capture_fits(name, build, events: int) -> None:
    """The ONE fabric depth a pass can exhaust: the capture RAM.

    The op stream has no fabric depth to exhaust -- it is streamed, never
    stored -- so its length is checked for nothing here on purpose.
    """
    if events >= SHIPPED_CAPTURE_EVENTS:
        raise BundleRefusal(
            f"{name}: pass {build.index} produced {events} capture events "
            f"against the shipped fabric's {SHIPPED_CAPTURE_EVENTS}-record RAM; "
            f"a board run would refuse as truncated")


def require_cores_fit_neurons(name: str, cores: Sequence[Any], *, max_neurons: int) -> None:
    """[U-H, B6.1] Every mapped core fits the neurons per core the bundle declares its fabric to have."""
    for index, core in enumerate(cores):
        width = int(core.neurons_per_core)
        if width > int(max_neurons):
            raise BundleRefusal(
                f"{name}: core {index} is mapped {width} neurons wide and the bundle declares its fabric at "
                f"{int(max_neurons)} neurons per core; the pass would address neurons the fabric does not have")
