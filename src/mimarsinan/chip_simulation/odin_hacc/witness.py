"""WHO measured a pass: the vendored RTL, or the cycle-accurate twin.

A bundle's expectations are only worth what produced them, so the producer is a
declared object rather than an assumption: ``CosimWitness`` runs the committed
cosimulation on the vendored ODIN core (an RTL simulator must be present, which
is why the committed fixture is regenerated rarely), and ``TwinWitness`` takes
the counts of the cycle-accurate software twin the whole toolchain is gated
against. Both are recorded by name in the bundle's provenance, so a reader can
never mistake a twin-derived expectation for a silicon measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence, Tuple

from mimarsinan.chip_simulation.odin_rtl.cosim import run_cosim

Counts = Dict[Tuple[int, int, int, int], int]

#: What each witness claims, verbatim, into the bundle's provenance.
COSIM_DERIVATION = (
    "per-pass counts measured by mimarsinan.chip_simulation.odin_rtl.cosim"
    ".run_cosim on the vendored ODIN core and gated against the cycle-accurate "
    "twin at every cycle")
TWIN_DERIVATION = (
    "per-pass counts computed by the cycle-accurate twin "
    "(mimarsinan.chip_simulation.odin_rtl.reference.simulate_cycles), the same "
    "twin every committed RTL cosimulation is gated against at every cycle. NOT "
    "a silicon measurement: the board run is what measures this network")

#: The replay document's own note, per witness: what a fake pyxrt is replaying.
COSIM_CAPTURE_NOTE = (
    "What the cosimulated fabric DMA'd back, one entry per (core, "
    "sample) pass, keyed by the sha256 of the stimulus that produced "
    "it. A fake pyxrt replays these; a host that sends a stimulus not "
    "listed here gets no verdict, which is how a wrong stimulus is "
    "caught rather than answered. device_cycles is the free-running "
    "counter of the WHOLE multi-sample cosimulation of that pass, not "
    "a per-sample figure.")
TWIN_CAPTURE_NOTE = (
    "What the cycle-accurate twin says the fabric will DMA back, one entry per "
    "(core, sample) pass, keyed by the sha256 of the stimulus that produced it. "
    "A fake pyxrt replays these, which exercises the whole host path with NO "
    "device; a host that sends a stimulus not listed here gets no verdict. "
    "device_cycles is 0 because no device ran.")


@dataclass(frozen=True)
class PassMeasurement:
    """One pass's per-(sample, cycle, core, neuron) counts and its device cost."""

    counts: Counts
    device_cycles: int


class PassWitness(Protocol):
    """Something that can answer one pass over every sample."""

    name: str
    derivation: str
    capture_note: str

    def measure(self, build: Any, traces: Sequence[Any]) -> PassMeasurement:
        """The counts this witness produces for ``build`` over every trace."""
        ...


class TwinWitness:
    """The cycle-accurate twin answers the pass — no simulator, no device."""

    name = "cycle_twin"
    derivation = TWIN_DERIVATION
    capture_note = TWIN_CAPTURE_NOTE

    def measure(self, build: Any, traces: Sequence[Any]) -> PassMeasurement:
        counts: Counts = {}
        for sample, trace in enumerate(traces):
            for cycle in range(trace.total_cycles):
                for neuron, value in enumerate(trace.outputs[cycle][build.index]):
                    if value:
                        counts[(sample, cycle, 0, neuron)] = int(value)
        return PassMeasurement(counts=counts, device_cycles=0)


class CosimWitness:
    """The vendored RTL answers the pass under a local Verilog simulator."""

    name = "rtl_cosim"
    derivation = COSIM_DERIVATION
    capture_note = COSIM_CAPTURE_NOTE

    def measure(self, build: Any, traces: Sequence[Any]) -> PassMeasurement:
        measured = run_cosim(
            build.export, [build.per_cycle(trace) for trace in traces],
            latencies=[build.latency])
        return PassMeasurement(
            counts=dict(measured.counts),
            device_cycles=int(measured.capture.cycles))
