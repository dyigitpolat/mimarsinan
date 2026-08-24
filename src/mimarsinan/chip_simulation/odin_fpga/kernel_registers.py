"""The ODIN kernel's register map, its buffer layouts, and the refusals they buy.

CROSS-LANGUAGE CONTRACT — this is the host-side home of ONE table whose other
copy is the ``localparam ADDR_*`` block of
``hw/fpga/kernel/odin_fpga_kernel_top.v``. ``scripts/hacc/gen_kernel_xml.py``
emits the Vitis ``kernel.xml`` from the ``ARG_*`` ids here, so the packaged
kernel and the host that drives it cannot disagree about the offsets.

Nothing in this module talks to a device: it decodes what a device reports.
That separation is what lets the refusals below be tested without XRT.
"""

from __future__ import annotations

import struct
from typing import Any, List, Sequence

from mimarsinan.chip_simulation.odin_fpga.transport import DeviceTransportError
from mimarsinan.chip_simulation.odin_rtl.capture import CaptureEvent

#: The Vitis RTL-kernel name the packaging scripts build (scripts/hacc/): the
#: AXI WRAPPER is the packaged kernel, and the host resolves it by that name.
KERNEL_NAME = "odin_fpga_kernel_top"

#: The kernel's argument order — the s_axilite register map the Verilog
#: declares.
ARG_PROGRAM = 0
ARG_STIMULUS = 1
ARG_CAPTURE = 2
ARG_PROGRAM_WORDS = 3
ARG_STIMULUS_WORDS = 4
ARG_CAPTURE_WORDS = 5

#: AXI-lite control offsets (Vitis RTL-kernel convention: 0x00 is ap_ctrl).
#: 0x4C/0x54/0x5C are this kernel's own: the status word and the two read-only
#: capacities the fabric was COMPILED with, which no host can guess.
CTRL_OFFSET = 0x00
AP_DONE = 1 << 1
ADDR_STATUS = 0x4C
ADDR_CAPTURE_CAPACITY = 0x54
ADDR_PROGRAM_CAPACITY = 0x5C

#: Status word: bit 31 is the kernel's `err`, bits 30:0 the events the fabric
#: SAW (which may exceed what it could store — that is the truncation case).
STATUS_ERR_BIT = 1 << 31
STATUS_EVENTS_MASK = STATUS_ERR_BIT - 1

#: The capture buffer's layout, in 32-bit words: a two-word header (how many
#: events the fabric saw, how many device cycles the run took) followed by one
#: four-word record per AER-out event.
CAPTURE_HEADER_WORDS = 2
CAPTURE_RECORD_WORDS = 4
WORD_BYTES = 4

#: The host's declared capture ceiling. The REAL capacity is the minimum of
#: this and the fabric's own, which a session reads out of
#: ADDR_CAPTURE_CAPACITY.
DEFAULT_CAPTURE_EVENTS = 1 << 20

#: What the SHIPPED fabric actually holds, and why — the second copy of the
#: ``CAP_WORDS`` default in ``hw/fpga/kernel/odin_fpga_kernel_top.v``. The
#: capture RAM is a block RAM (one write port, one registered read), so its
#: depth is bought in tiles rather than in the 32 flip-flops per word the P8
#: compile-limits study measured before that fix; 16,384 words is the depth the
#: committed synthesis record (``hw/fpga/compile_limits.json``) costs, and
#: 4,095 records is what it leaves after the two header words. No host may
#: ASSUME it: a session reads ADDR_CAPTURE_CAPACITY and ``decode_capture``
#: refuses a run that reached whatever the device reported.
SHIPPED_CAPTURE_WORDS = 16384
SHIPPED_CAPTURE_EVENTS = (SHIPPED_CAPTURE_WORDS - CAPTURE_HEADER_WORDS) // (
    CAPTURE_RECORD_WORDS)


class OdinFpgaCaptureTruncated(DeviceTransportError):
    """The device saw at least as many events as the capture can hold."""


class OdinFpgaKernelError(DeviceTransportError):
    """The kernel raised `err`: it refused the program it was given."""


class OdinFpgaProgramTooLarge(DeviceTransportError):
    """The token stream does not fit the fabric's program RAM."""


def stimulus_base_word(program_words: int) -> int:
    """Where the stimulus lands in the fabric's program RAM.

    CROSS-LANGUAGE CONTRACT with `odin_fpga_kernel_top.v`: the stimulus
    OVERWRITES the program payload's END terminator, so the fabric executes one
    continuous stream rather than stopping at the end of the programming half.
    """
    return max(0, int(program_words) - 1)


def capture_buffer_bytes(capacity_events: int) -> int:
    """The host buffer one run of ``capacity_events`` records needs."""
    return WORD_BYTES * (
        CAPTURE_HEADER_WORDS + CAPTURE_RECORD_WORDS * int(capacity_events))


def decode_status(word: int) -> tuple:
    """``(err, events_seen)`` from the kernel's 0x4C status register."""
    value = int(word)
    return bool(value & STATUS_ERR_BIT), value & STATUS_EVENTS_MASK


def require_no_kernel_error(status: int, *, transport: str) -> None:
    """0x4C after ap_done: `err` means the fabric refused the program."""
    failed, events_seen = decode_status(status)
    if failed:
        raise OdinFpgaKernelError(
            f"{transport}: the kernel raised err after seeing "
            f"{events_seen} event(s) — the fabric sequencer REFUSED an opcode "
            f"it does not implement (SHADOW/PROG have no fabric path), an AER "
            f"handshake timed out, or the DMA could not fit the payload. The "
            f"counts of this run are not a network's answer and are not being "
            f"decoded")


def require_program_fits(
    program_words: int, stimulus_words: int, capacity_words: int,
    *, transport: str,
) -> None:
    """The fabric's program RAM must hold the whole token stream."""
    needed = stimulus_base_word(program_words) + int(stimulus_words)
    if needed > int(capacity_words):
        raise OdinFpgaProgramTooLarge(
            f"{transport}: the run needs {needed} program words "
            f"({program_words} programming + {stimulus_words} stimulus, the "
            f"stimulus overwriting the programming payload's END) but the "
            f"loaded xclbin's program RAM holds {capacity_words}. Rebuild the "
            f"kernel with a larger PROG_WORDS or split the run into fewer "
            f"samples per pass; a device that wrapped the address would "
            f"execute a program nobody assembled")


def decode_capture(words: Sequence[int], capacity: int) -> tuple:
    """``(events, device_cycles)`` from the capture buffer's word image."""
    if len(words) < CAPTURE_HEADER_WORDS:
        raise DeviceTransportError(
            "the capture buffer came back shorter than its own two-word header; "
            "the kernel never wrote a verdict and the run has no counts")
    written = int(words[0])
    device_cycles = int(words[1])
    if written >= int(capacity):
        raise OdinFpgaCaptureTruncated(
            f"the kernel SAW {written} events against a capacity of "
            f"{capacity} (min of this session's declaration and the fabric's "
            f"own capture RAM, read from 0x54): the capture is at or over its "
            f"limit and the events past it would read as silent neurons. "
            f"Rebuild the kernel with a larger CAP_WORDS "
            f"(hw/fpga/kernel/odin_fpga_kernel_top.v) or run fewer samples per "
            f"pass; the counts of a truncated run are not a result.")
    events: List[CaptureEvent] = []
    for index in range(written):
        base = CAPTURE_HEADER_WORDS + index * CAPTURE_RECORD_WORDS
        tag, cycle, core, neuron = words[base:base + CAPTURE_RECORD_WORDS]
        events.append(CaptureEvent(
            core=int(core), neuron=int(neuron), cycle=int(cycle), tag=int(tag)))
    return tuple(events), device_cycles


def capture_words(raw: Any) -> List[int]:
    """The capture buffer's bytes (or word list) as little-endian words."""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        buffer = bytes(raw)
        return list(struct.unpack(f"<{len(buffer) // WORD_BYTES}I", buffer))
    return [int(value) for value in raw]
