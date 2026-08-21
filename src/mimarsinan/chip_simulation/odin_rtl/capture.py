"""The testbench's output line protocol: events, readback/shadow statistics, drain.

The testbench prints one line per observable and nothing else, so a run's whole
verdict is a text transcript a human can read and a parser cannot silently
misread: an unrecognised line contributes nothing, a FATAL line raises, and a
run that never reached its DONE line is refused rather than reported empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


class TestbenchFailure(RuntimeError):
    """The testbench reported a fatal condition; the run produced no verdict."""


@dataclass(frozen=True)
class CaptureEvent:
    """One AER-out transaction: the neuron that fired, when, under which tag."""

    core: int
    neuron: int
    cycle: int
    tag: int


@dataclass(frozen=True)
class BarrierRecord:
    """One BARRIER stage as executed: the window, when it started, its bound."""

    tag: int
    start_cycle: int
    bound: int


@dataclass(frozen=True)
class CaptureResult:
    """Everything one testbench run reports back."""

    events: Tuple[CaptureEvent, ...]
    reads: int
    read_failures: int
    shadow_checks: int
    shadow_failures: int
    read_failure_lines: Tuple[str, ...]
    shadow_failure_lines: Tuple[str, ...]
    cycles: int
    tag_opened: Tuple[Tuple[int, int], ...] = ()
    barriers: Tuple[BarrierRecord, ...] = ()
    # [ODIN P6] the generated-core testbench's two extra verdicts: the emitted
    # RTL's own spec against the harness's parameters, and the sticky rail flag
    # of a law whose contract says the rails are unreachable.
    spec_checks: int = 0
    spec_failures: int = 0
    rail_checks: int = 0
    rail_failures: int = 0
    spec_failure_lines: Tuple[str, ...] = ()
    rail_failure_lines: Tuple[str, ...] = ()

    def counts_by_tag(self) -> Dict[Tuple[int, int, int], int]:
        """``(tag, core, neuron) -> spike count`` — the per-window READOUT."""
        counts: Dict[Tuple[int, int, int], int] = {}
        for event in self.events:
            key = (event.tag, event.core, event.neuron)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def last_event_cycle(self, tag: int) -> int | None:
        """The cycle of the last output event captured under ``tag``."""
        cycles = [event.cycle for event in self.events if event.tag == int(tag)]
        return max(cycles) if cycles else None

    def window_start_cycle(self, tag: int) -> int | None:
        """The cycle at which a READOUT window opened (just before its INJECT)."""
        for opened, cycle in self.tag_opened:
            if opened == int(tag):
                return cycle
        return None

    def drain_overruns(self) -> Tuple[Tuple[int, int, int], ...]:
        """``(tag, last_event_cycle, deadline)`` for every window the bound missed.

        The exported bound prices the WHOLE processing of a window's injected
        events -- one push plus a 512-cycle sweep each, the queue the scheduler
        may still hold, and an output handshake per emitted spike -- so it is
        measured from the instant the window opened, not from the barrier's
        start (which already sits after the pushes). An empty tuple is the gate
        passing.
        """
        overruns = []
        for record in self.barriers:
            last = self.last_event_cycle(record.tag)
            start = self.window_start_cycle(record.tag)
            deadline = (record.start_cycle if start is None else start) + record.bound
            if last is not None and last > deadline:
                overruns.append((record.tag, last, deadline))
        return tuple(overruns)


def parse_capture(stdout: str) -> CaptureResult:
    """Parse the testbench's line protocol; a FATAL line raises rather than returns."""
    events: List[CaptureEvent] = []
    read_failures: List[str] = []
    shadow_failures: List[str] = []
    tags: List[Tuple[int, int]] = []
    barriers: List[BarrierRecord] = []
    reads = read_fails = shadow_checks = shadow_fails = 0
    spec_checks = spec_fails = rail_checks = rail_fails = 0
    spec_failure_lines: List[str] = []
    rail_failure_lines: List[str] = []
    cycles = -1
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "FATAL":
            raise TestbenchFailure(f"the ODIN testbench aborted: {line.strip()}")
        if fields[0] == "EV" and len(fields) == 5:
            events.append(CaptureEvent(
                core=int(fields[1]), neuron=int(fields[2]),
                cycle=int(fields[3]), tag=int(fields[4])))
        elif fields[0] == "RBFAIL":
            read_failures.append(line.strip())
        elif fields[0] == "SHFAIL":
            shadow_failures.append(line.strip())
        elif fields[0] == "SPECFAIL":
            spec_failure_lines.append(line.strip())
        elif fields[0] == "RAILFAIL":
            rail_failure_lines.append(line.strip())
        elif fields[0] == "SPECSTAT" and len(fields) == 3:
            spec_checks, spec_fails = int(fields[1]), int(fields[2])
        elif fields[0] == "RAILSTAT" and len(fields) == 3:
            rail_checks, rail_fails = int(fields[1]), int(fields[2])
        elif fields[0] == "RBSTAT" and len(fields) == 3:
            reads, read_fails = int(fields[1]), int(fields[2])
        elif fields[0] == "SHSTAT" and len(fields) == 3:
            shadow_checks, shadow_fails = int(fields[1]), int(fields[2])
        elif fields[0] == "TAGAT" and len(fields) == 3:
            tags.append((int(fields[1]), int(fields[2])))
        elif fields[0] == "BARRIER" and len(fields) == 4:
            barriers.append(BarrierRecord(
                tag=int(fields[1]), start_cycle=int(fields[2]),
                bound=int(fields[3])))
        elif fields[0] == "DONE" and len(fields) == 3:
            cycles = int(fields[1])
    if cycles < 0:
        raise TestbenchFailure(
            "the ODIN testbench never printed its DONE line: the run was cut "
            "short (a simulator crash, a timeout, or a $finish inside a task)")
    return CaptureResult(
        events=tuple(events), reads=reads, read_failures=read_fails,
        shadow_checks=shadow_checks, shadow_failures=shadow_fails,
        read_failure_lines=tuple(read_failures),
        shadow_failure_lines=tuple(shadow_failures),
        cycles=cycles,
        tag_opened=tuple(tags),
        barriers=tuple(barriers),
        spec_checks=spec_checks, spec_failures=spec_fails,
        rail_checks=rail_checks, rail_failures=rail_fails,
        spec_failure_lines=tuple(spec_failure_lines),
        rail_failure_lines=tuple(rail_failure_lines),
    )
