"""The ODIN sequencer program: one versioned schema, one emitter, one decoder, one gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from mimarsinan.mapping.export.odin.registers import RegisterWrite
from mimarsinan.mapping.platform.event_order import drain_events

SEQUENCER_SCHEMA_VERSION = 1

STAGE_CONFIG = "CONFIG"
STAGE_GATE = "GATE"
STAGE_CLEAR = "CLEAR"
STAGE_INJECT = "INJECT"
STAGE_TREF = "TREF"
STAGE_BARRIER = "BARRIER"
STAGE_READOUT = "READOUT"

STAGE_KINDS: Tuple[str, ...] = (
    STAGE_CONFIG, STAGE_GATE, STAGE_CLEAR, STAGE_INJECT,
    STAGE_TREF, STAGE_BARRIER, STAGE_READOUT,
)

# CROSS-LANGUAGE CONTRACT — SPI_GATE_ACTIVITY (config register 0, doc/README.md
# Sec.4) gates ALL network activity and is what enables SPI access to the neuron
# and synapse memories (doc/README.md Sec.2.1), so a stage that writes a memory
# (CONFIG, CLEAR) must run with it asserted and a stage that expects the network
# to step (INJECT, TREF, BARRIER, READOUT) must run with it de-asserted. CONFIG
# asserts it through its own register write; every later change is a GATE stage,
# so a consumer executing this program literally never programs memories it then
# leaves frozen.
_REQUIRED_PAYLOAD_FIELDS: Dict[str, Tuple[Tuple[str, type], ...]] = {
    STAGE_GATE: (("on", bool),),
}

# CROSS-LANGUAGE CONTRACT — the drain bound's constants come from
# ChFrenkel/ODIN @ 1781931: a neuron spike event costs 1 push cycle plus a
# 512-cycle full-fanout sweep (doc/README.md Sec.2.2.1 input-AER table; the sweep
# is src/controller.v:208, POP_NEUR runs until `&ctrl_cnt[8:0]`), the scheduler's
# spike FIFO is 32 deep (src/scheduler.v:184-186), and every output spike stalls
# the controller for a four-phase AER handshake across a double-latching barrier
# (src/aer_out.v:96-160, AEROUT_CTRL_BUSY gating src/controller.v:174).
SCHEDULER_PUSH_CYCLES = 1
NEURON_SWEEP_CYCLES = 512
SCHEDULER_FIFO_DEPTH = 32

#: The receiver-side obligation the bound is stated against: the AER-out consumer
#: acknowledges within this many core cycles. The handshake itself costs four
#: (REQ up, ACK through two sync flops, REQ down, ACK negedge back through two);
#: eight is the conservative allowance a host-mediated router must honour. P5
#: verifies the whole bound against the RTL on adversarial cases.
AEROUT_HANDSHAKE_CYCLES = 8


class SequencerProgramError(ValueError):
    """The sequencer program does not satisfy its own schema."""


@dataclass(frozen=True)
class SequencerProgram:
    """An ordered stage list at one schema version — the ONE program format."""

    stages: Tuple[Mapping[str, Any], ...]
    schema_version: int = SEQUENCER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEQUENCER_SCHEMA_VERSION:
            raise SequencerProgramError(
                f"schema_version {self.schema_version} is not the supported "
                f"{SEQUENCER_SCHEMA_VERSION}")
        for stage in self.stages:
            kind = stage.get("kind")
            if kind not in STAGE_KINDS:
                raise SequencerProgramError(
                    f"unknown sequencer stage {kind!r}; the schema declares "
                    f"{', '.join(STAGE_KINDS)}")
            payload = stage.get("payload")
            if not isinstance(payload, Mapping):
                raise SequencerProgramError(f"stage {kind!r} carries no payload")
            for name, expected in _REQUIRED_PAYLOAD_FIELDS.get(str(kind), ()):
                if not isinstance(payload.get(name), expected):
                    raise SequencerProgramError(
                        f"stage {kind!r} needs the payload field {name!r} as a "
                        f"{expected.__name__}, got {payload.get(name)!r}")


def _stage(kind: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {"kind": kind, "payload": dict(payload)}


def config_stage(
    *,
    core_index: int,
    register_writes: Sequence[RegisterWrite],
    neuron_words: Sequence[int],
    synapse_words: Sequence[int],
) -> Dict[str, Any]:
    """Program one core: its config registers and both memory images."""
    return _stage(STAGE_CONFIG, {
        "core_index": int(core_index),
        "register_writes": [
            {"address": int(w.address), "value": int(w.value)} for w in register_writes
        ],
        "neuron_words": [int(w) for w in neuron_words],
        "synapse_words": [int(w) for w in synapse_words],
    })


def gate_stage(*, on: bool) -> Dict[str, Any]:
    """Assert (``on``) or release SPI_GATE_ACTIVITY: memory access vs. network activity."""
    return _stage(STAGE_GATE, {"on": bool(on)})


_CLEAR_WRITE_FIELDS = ("neuron", "byte_addr", "value", "mask")


def clear_stage(
    *, core_index: int, byte_writes: Sequence[Mapping[str, int]]
) -> Dict[str, Any]:
    """Rewrite the neuron-memory STATE bytes: the per-sample membrane reset."""
    rows = []
    for write in byte_writes:
        missing = [name for name in _CLEAR_WRITE_FIELDS if name not in write]
        if missing:
            raise SequencerProgramError(
                f"a CLEAR byte write needs {', '.join(_CLEAR_WRITE_FIELDS)}; "
                f"missing: {', '.join(missing)}")
        rows.append({name: int(write[name]) for name in _CLEAR_WRITE_FIELDS})
    return _stage(STAGE_CLEAR, {"core_index": int(core_index), "byte_writes": rows})


def inject_stage(
    *,
    core_index: int,
    events: Sequence[Tuple[int, int]],
    runtime_rows: Sequence[Any] = (),
) -> Dict[str, Any]:
    """Deliver ``(row, multiplicity)`` pairs — ascending, multiplicity adjacent.

    ``events`` are the deliveries known at export time (the always-on rows);
    ``runtime_rows`` names, per logical slot, which physical rows the host
    drives once :func:`plan_injection` has that sample's counts.
    """
    rows = []
    for row, multiplicity in events:
        if int(multiplicity) < 0:
            raise ValueError(
                f"multiplicity must be non-negative, got {multiplicity} at row {row}")
        rows.append([int(row), int(multiplicity)])
    return _stage(STAGE_INJECT, {
        "core_index": int(core_index),
        "events": rows,
        "runtime_rows": [
            [int(slot), [int(row) for row in slot_rows]]
            for slot, slot_rows in runtime_rows
        ],
    })


def tref_stage(*, scope: str) -> Dict[str, Any]:
    """A time-reference event: ``"all"`` neurons, or one neuron address."""
    return _stage(STAGE_TREF, {"scope": str(scope)})


def barrier_stage(*, cycles: int) -> Dict[str, Any]:
    """Wait the deterministic drain bound before the next stage observes anything."""
    if int(cycles) < 0:
        raise SequencerProgramError(f"a barrier waits a non-negative {cycles} cycles")
    return _stage(STAGE_BARRIER, {"cycles": int(cycles)})


def readout_stage(*, core_index: int, neurons: Sequence[int]) -> Dict[str, Any]:
    """Capture the tagged output events of the named neurons."""
    return _stage(STAGE_READOUT, {
        "core_index": int(core_index), "neurons": [int(n) for n in neurons],
    })


def plan_injection(
    *, counts: Sequence[int], emitting_rows: Callable[[int], Sequence[int]]
) -> Tuple[Tuple[int, int], ...]:
    """Logical per-slot counts -> the physical ``(row, multiplicity)`` delivery.

    The canonical drain (`event_order.drain_events`) is the ONE source of the
    order; rows a slot does not actually drive contribute nothing.
    """
    events = []
    for slot, multiplicity in drain_events(counts):
        for row in emitting_rows(slot):
            events.append((int(row), int(multiplicity)))
    return tuple(events)


def drain_bound_cycles(*, injected_events: int, emitted_spike_bound: int) -> int:
    """The conservative closed-form BARRIER bound (verified against RTL at P5).

    Every injected row event costs one push plus a full 512-cycle sweep; the
    scheduler FIFO may still hold ``SCHEDULER_FIFO_DEPTH`` events when the last
    push lands, so the queue depth is charged a full sweep each; and each output
    spike stalls the controller for one AER-out handshake allowance.
    """
    if int(injected_events) < 0 or int(emitted_spike_bound) < 0:
        raise SequencerProgramError(
            f"the drain bound takes non-negative counts, got "
            f"injected_events={injected_events}, "
            f"emitted_spike_bound={emitted_spike_bound}")
    sweeps = int(injected_events) + SCHEDULER_FIFO_DEPTH
    return (
        sweeps * (SCHEDULER_PUSH_CYCLES + NEURON_SWEEP_CYCLES)
        + int(emitted_spike_bound) * AEROUT_HANDSHAKE_CYCLES
    )


def emit_program(program: SequencerProgram) -> Dict[str, Any]:
    """THE emitter: a JSON document carrying the version and the stage order."""
    return {
        "schema_version": int(program.schema_version),
        "stages": [
            {"kind": str(stage["kind"]), "payload": dict(stage["payload"])}
            for stage in program.stages
        ],
    }


def decode_program(document: Mapping[str, Any]) -> SequencerProgram:
    """THE decoder: the exact inverse of :func:`emit_program`, loud on drift."""
    version = document.get("schema_version")
    if version != SEQUENCER_SCHEMA_VERSION:
        raise SequencerProgramError(
            f"schema_version {version!r} is not the supported "
            f"{SEQUENCER_SCHEMA_VERSION}")
    stages = document.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise SequencerProgramError(
            "the program document needs a 'stages' list")
    return SequencerProgram(
        stages=tuple(
            {"kind": str(stage["kind"]), "payload": dict(stage["payload"])}
            for stage in stages
        ),
        schema_version=int(version),
    )
