"""The device seam: what a physical ODIN target must do, and what it reports back.

One transport is one way of reaching cores that execute the exported program.
The RTL cosimulation is the FIRST device behind this seam and the board is a
transport swap, not a second backend: everything above ``DeviceTransport`` —
the segment driver, the records, the certificate — is written once against
these three calls and never learns which side of the seam it is on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple, runtime_checkable

from mimarsinan.chip_simulation.odin_rtl.cosim import window_counts_of


class DeviceTransportError(RuntimeError):
    """The device could not be reached, programmed, or read back."""


@dataclass(frozen=True)
class ProgramReceipt:
    """What one programming pass cost and what it actually wrote.

    ``wall_s`` is MEASURED host wall — the reprogram physics the deployment
    record carries — and ``basis`` says what that wall covers on this
    transport, so a cosimulated wall is never read as board SPI time.
    """

    payload: bytes
    ops: int
    cores: int
    wall_s: float
    basis: str


@dataclass(frozen=True)
class TransportRun:
    """One execution: the counts the device produced, and what it cost.

    ``counts`` is keyed ``(sample, cycle, core, neuron)`` — the per-cycle shape
    the cosimulation already returns, which is the only shape that discriminates
    a per-event law from a per-cycle one.
    """

    counts: Dict[Tuple[int, int, int, int], int]
    samples: int
    cycles_per_sample: int
    wall_s: float
    program_wall_s: float
    device_cycles: int
    detail: Mapping[str, Any] = field(default_factory=dict)

    def cycle_counts(self, sample: int, cycle: int, core: int, n_neurons: int
                     ) -> Tuple[int, ...]:
        return tuple(
            self.counts.get((sample, cycle, core, neuron), 0)
            for neuron in range(n_neurons)
        )

    def window_counts(self, *, latencies: Sequence[int], simulation_length: int,
                      neurons: Sequence[int]):
        """Per-sample per-core per-neuron counts over each core's own window."""
        return window_counts_of(
            self.counts, samples=self.samples,
            cycles_per_sample=self.cycles_per_sample,
            latencies=latencies, simulation_length=simulation_length,
            neurons=neurons,
        )


@runtime_checkable
class DeviceTransport(Protocol):
    """A session against ODIN cores that execute the exported program."""

    name: str

    def open(self) -> None:
        """Acquire the device/session; refuse loud when it is not there."""

    def close(self) -> None:
        """Release it; idempotent, so a failed run still tears down."""

    def program(self, export: Any) -> ProgramReceipt:
        """Write memories and configuration registers for one export."""
        ...

    def run_samples(
        self,
        per_cycle_inputs: Sequence[Sequence[Mapping[int, Sequence[int]]]],
        *,
        latencies: Sequence[int],
    ) -> TransportRun:
        """Execute every sample's per-cycle injection plan and read the counts."""
        ...


class DeviceSession:
    """``with`` sugar over a transport, so no run leaks an open device."""

    def __init__(self, transport: DeviceTransport) -> None:
        self.transport = transport

    def __enter__(self) -> DeviceTransport:
        self.transport.open()
        return self.transport

    def __exit__(self, *exc: Any) -> bool:
        self.transport.close()
        return False
