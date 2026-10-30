"""The device payload: one export becomes the SAME bytes on every transport.

CROSS-LANGUAGE CONTRACT — the token stream encoded here is executed by three
sequencers that must agree bit for bit: the P5 Verilog testbench
(``hw/tb/tb_odin_core.v``), the on-fabric sequencer of the Vitis RTL kernel
(``hw/fpga/kernel/odin_fpga_kernel.v``), and the host-side harness. The
encoding itself has exactly one home (``odin_rtl/stimulus.py``); this module
only splits one export's ops into the two payloads a device is handed — the
PROGRAM (memories, registers, gate) and the STIMULUS (per-sample clear, inject,
tref, barrier) — and serializes them as little-endian 32-bit words.
"""

from __future__ import annotations

import struct
from typing import Any, List, Mapping, Sequence, Tuple

from mimarsinan.chip_simulation.odin_rtl.cosim import CosimPlan, build_cosim_ops
from mimarsinan.chip_simulation.odin_rtl.stimulus import Op, encode_ops

#: One token is one 32-bit word on the wire, little-endian (the AXI convention
#: of the XDMA shell and the word order ``$readmemh`` reproduces).
TOKEN_BYTES = 4


def payload_bytes(ops: Sequence[Op]) -> bytes:
    """The END-terminated token stream of ``ops`` as device-endian words."""
    tokens = encode_ops(ops)
    return struct.pack(f"<{len(tokens)}I", *tokens)


def program_plan(export: Any) -> CosimPlan:
    """The PROGRAMMING half of an export: every CONFIG stage, then gate-off.

    Built by the same op builder the cosimulation uses with an EMPTY injection
    plan, so the programming payload is a literal prefix of the full run — a
    device that is programmed once and run many times executes exactly these
    ops, and no second encoder exists to drift from the first.
    """
    return build_cosim_ops(
        export, [], latencies=[0] * len(export.cores))


def run_plan(
    export: Any,
    per_cycle_inputs: Sequence[Sequence[Mapping[int, Sequence[int]]]],
    *,
    latencies: Sequence[int],
) -> CosimPlan:
    """The full program: the programming prefix followed by every sample."""
    return build_cosim_ops(export, per_cycle_inputs, latencies=latencies)


def stimulus_ops(program: CosimPlan, run: CosimPlan) -> Tuple[Op, ...]:
    """The sample half of ``run``: its ops after the programming prefix.

    Refuses loud when the run does not literally start with the programming
    ops — a device programmed from one payload and stimulated from another
    would run a network nobody assembled.
    """
    prefix = tuple(program.ops)
    body = tuple(run.ops)
    if body[:len(prefix)] != prefix:
        raise ValueError(
            "the run program does not start with the programming payload: the "
            "two payloads were built from different exports, and the device "
            "would be stimulated against memories it was never given")
    return body[len(prefix):]


def program_payload(export: Any) -> bytes:
    """The bytes a transport DMAs (or shifts) to program one export."""
    return payload_bytes(program_plan(export).ops)


def split_payloads(
    export: Any,
    per_cycle_inputs: Sequence[Sequence[Mapping[int, Sequence[int]]]],
    *,
    latencies: Sequence[int],
) -> Tuple[bytes, bytes, CosimPlan]:
    """``(program bytes, stimulus bytes, full plan)`` for one export + run."""
    program = program_plan(export)
    full = run_plan(export, per_cycle_inputs, latencies=latencies)
    return (
        payload_bytes(program.ops),
        payload_bytes(stimulus_ops(program, full)),
        full,
    )


def counts_from_events(plan: CosimPlan, events: Sequence[Any]) -> dict:
    """Fold captured AER-out events into ``(sample, cycle, core, neuron)`` counts."""
    counts: dict = {}
    for event in events:
        sample, cycle = plan.decode_tag(event.tag)
        key = (sample, cycle, int(event.core), int(event.neuron))
        counts[key] = counts.get(key, 0) + 1
    return counts


def op_codes(payload: bytes) -> List[int]:
    """The raw token words of a payload — what a contract test compares."""
    if len(payload) % TOKEN_BYTES:
        raise ValueError(
            f"a {len(payload)}-byte payload is not a whole number of "
            f"{TOKEN_BYTES}-byte device words")
    return list(struct.unpack(f"<{len(payload) // TOKEN_BYTES}I", payload))
