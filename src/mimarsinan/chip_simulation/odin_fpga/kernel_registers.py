"""The ODIN kernel's device protocol: arguments in, memory out, refusals by name.

CROSS-LANGUAGE CONTRACT — the ``ARG_*`` ids are the ones
``scripts/hacc/gen_kernel_xml.py`` writes into the Vitis ``kernel.xml``, and the
``CAPTURE_*`` layout is the one ``hw/fpga/kernel/odin_fpga_kernel.v`` writes at
drain time (lines 171-189: header word 0 ``events_seen``, header word 1
``cycle``, then ``{tag, cycle, core, neuron}`` per event). Both halves are
FROZEN — xclbins already built on the cluster carry them.

NO REGISTER ACCESS EXISTS. The XRT Python binding
(``github.com/Xilinx/XRT``, branch ``2024.2``,
``src/python/pybind11/src/pyxrt.cpp``) binds no ``read_register`` or
``write_register`` on ``xrt::kernel`` and exposes no standalone ``ip`` object,
so the AXI-Lite status and capacity registers this kernel implements at
0x4C/0x54 are unreachable from Python — a host that read them crashed on
the U250 at its first CSR touch on 2026-08-25. Every truth below therefore
arrives either through kernel ARGUMENTS (what the host declares) or through
MEMORY (what the fabric DMAs back), and the capacities are DECLARED from the
sources the xclbin was compiled from rather than asked of the card.

Nothing in this module talks to a device: it decodes what a device reports.
That separation is what lets the refusals below be tested without XRT.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Iterable, List, Sequence

from mimarsinan.chip_simulation.odin_fpga.transport import DeviceTransportError
from mimarsinan.chip_simulation.odin_rtl.capture import CaptureEvent

#: The Vitis RTL-kernel name the packaging scripts build (scripts/hacc/): the
#: AXI WRAPPER is the packaged kernel, and the host resolves it by that name.
KERNEL_NAME = "odin_fpga_kernel_top"

#: The kernel's argument order, and how many there are — the frozen kernel.xml.
ARG_PROGRAM = 0
ARG_STIMULUS = 1
ARG_CAPTURE = 2
ARG_PROGRAM_WORDS = 3
ARG_STIMULUS_WORDS = 4
ARG_CAPTURE_WORDS = 5
KERNEL_ARGS = 6

#: The three buffer arguments, in the order the kernel is called with them.
BUFFER_ARGS = (ARG_PROGRAM, ARG_STIMULUS, ARG_CAPTURE)

WORD_BYTES = 4

#: The capture buffer's layout, in 32-bit words: a two-word header followed by
#: one four-word record per AER-out event.
CAPTURE_HEADER_WORDS = 2
HEADER_EVENTS_SEEN = 0
HEADER_DEVICE_CYCLES = 1
CAPTURE_RECORD_WORDS = 4
RECORD_TAG = 0
RECORD_CYCLE = 1
RECORD_CORE = 2
RECORD_NEURON = 3

#: The word the host writes into BOTH header slots before it starts the kernel.
#: The fabric overwrites them at drain time, so a header that still carries this
#: sentinel means the kernel produced NO VERDICT — the only way `err` reaches a
#: host that cannot read 0x4C.
CAPTURE_NO_VERDICT = 0xFFFFFFFF

#: The host's declared capture ceiling; a session takes the min of it and the
#: capacity the package declares the bitstream was built with.
DEFAULT_CAPTURE_EVENTS = 1 << 20

#: What the SHIPPED fabric holds, and why — the second copy of the ``NC`` and
#: ``CAP_WORDS`` defaults in ``hw/fpga/kernel/odin_fpga_kernel_top.v`` at the NC
#: ``scripts/hacc/build_xclbn.sh`` is the only flow that builds. The capture
#: RAM is a block RAM (one write port, one registered read), so its depth is
#: bought in tiles; 16,384 words is the depth the committed synthesis record
#: (``hw/fpga/compile_limits.json``) costs, and 4,095 records is what it leaves
#: after the two header words. THE OP STREAM HAS NO SUCH NUMBER: it is not
#: stored on the fabric at all, so a run's length is bounded only by the 32-bit
#: word counts the host declares.
SHIPPED_KERNEL_CORES = 1
SHIPPED_CAPTURE_WORDS = 16384
SHIPPED_CAPTURE_EVENTS = (SHIPPED_CAPTURE_WORDS - CAPTURE_HEADER_WORDS) // (
    CAPTURE_RECORD_WORDS)

#: The FABRIC a bitstream was built from. The capture layout above is the same
#: for every one of them -- it is the wrapper's, and the wrapper is shared -- but
#: the CORE geometry is not, so a session says which fabric its declaration
#: belongs to. The named table lives in
#: ``chip_simulation.odin_fpga.chip_configs``; these are the defaults of the one
#: fabric that has ever been placed and routed.
SHIPPED_CHIP_CONFIG = "odin_stock_256x256"
SHIPPED_NEURONS_PER_CORE = 256
SHIPPED_AXON_SLOTS_PER_CORE = 128

#: Where the declared capacities come from, verbatim in every refusal that
#: spends one. A capacity nobody can read back is only honest if it names its
#: source.
CAPACITY_PROVENANCE = (
    "declared from hw/fpga/kernel/odin_fpga_kernel_top.v (CAP_WORDS = "
    f"{SHIPPED_CAPTURE_WORDS}) at NC = {SHIPPED_KERNEL_CORES}, the only "
    "geometry scripts/hacc/build_xclbn.sh builds — NOT read back from the card, "
    "because the XRT Python binding exposes no register read"
)


class OdinFpgaCaptureTruncated(DeviceTransportError):
    """The device saw at least as many events as the capture can hold."""


class OdinFpgaKernelError(DeviceTransportError):
    """The kernel wrote no verdict: it refused the program it was given."""


class KernelCapacity:
    """What the loaded xclbin holds, declared from what it was BUILT from.

    There is no program capacity. The fabric stores no copy of the host's op
    stream — it consumes it live through a shallow FIFO — so ``cores`` is the
    geometry the bitstream was built at rather than a depth divided by one.
    """

    def __init__(
        self,
        *,
        cores: int = SHIPPED_KERNEL_CORES,
        capture_events: int = SHIPPED_CAPTURE_EVENTS,
        provenance: str = CAPACITY_PROVENANCE,
        chip: str = SHIPPED_CHIP_CONFIG,
        neurons_per_core: int = SHIPPED_NEURONS_PER_CORE,
        axon_slots_per_core: int = SHIPPED_AXON_SLOTS_PER_CORE,
    ) -> None:
        self.cores = int(cores)
        self.capture_events = int(capture_events)
        self.provenance = str(provenance)
        self.chip = str(chip)
        self.neurons_per_core = int(neurons_per_core)
        self.axon_slots_per_core = int(axon_slots_per_core)

    def ceiling(self, host_events: int) -> int:
        """The events a session may decode: min(host declaration, the fabric's)."""
        return min(int(host_events), self.capture_events)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capture_events": self.capture_events,
            "cores": self.cores,
            "provenance": self.provenance,
            "chip": self.chip,
            "neurons_per_core": self.neurons_per_core,
            "axon_slots_per_core": self.axon_slots_per_core,
        }


def require_declared_storage(capacity: KernelCapacity, *, transport: str) -> None:
    """A package that declares no storage cannot run anything."""
    if capacity.capture_events <= 0 or capacity.cores <= 0:
        raise DeviceTransportError(
            f"{transport}: this package declares a capture capacity of "
            f"{capacity.capture_events} events across {capacity.cores} core(s) "
            f"({capacity.provenance}) — a kernel built with no storage cannot "
            f"run anything, and treating it as zero would turn every run into a "
            f"silent empty one")


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


def no_verdict_header() -> bytes:
    """The sentinel image the host DMAs into the capture header before a run."""
    return struct.pack(
        f"<{CAPTURE_HEADER_WORDS}I", *([CAPTURE_NO_VERDICT] * CAPTURE_HEADER_WORDS))


def require_kernel_verdict(words: Sequence[int], *, transport: str) -> None:
    """The capture header after ap_done: an untouched sentinel means `err`."""
    if len(words) < CAPTURE_HEADER_WORDS:
        raise DeviceTransportError(
            f"{transport}: the capture buffer came back shorter than its own "
            f"two-word header; the kernel never wrote a verdict and the run has "
            f"no counts")
    header = [int(word) for word in words[:CAPTURE_HEADER_WORDS]]
    if all(word == CAPTURE_NO_VERDICT for word in header):
        raise OdinFpgaKernelError(
            f"{transport}: the capture header came back still carrying the "
            f"host's NO-VERDICT sentinel (0x{CAPTURE_NO_VERDICT:08X} in both "
            f"words), so the fabric DMA'd nothing back. That is a capture "
            f"buffer that never reached the card, or a kernel that never "
            f"reached its drain. The likeliest cause is a loaded xclbin built "
            f"with a SMALLER CAP_WORDS than this package declares, or one built "
            f"from a different kernel entirely — the kernel says which on its "
            f"own 0x4C/0x54 registers, and NO Python host can read them (pyxrt "
            f"binds no read_register). The counts of this run are not a "
            f"network's answer and are not being decoded")


def decode_capture(
    words: Sequence[int], capacity: int, *, transport: str = "odin_fpga",
) -> tuple:
    """``(events, device_cycles)`` from the capture buffer's word image."""
    require_kernel_verdict(words, transport=transport)
    written = int(words[HEADER_EVENTS_SEEN])
    device_cycles = int(words[HEADER_DEVICE_CYCLES])
    if written >= int(capacity):
        raise OdinFpgaCaptureTruncated(
            f"the kernel SAW {written} events against a capacity of "
            f"{capacity} (min of this session's declaration and the capture RAM "
            f"this package declares the fabric was built with): the capture is "
            f"at or over its limit and the events past it would read as silent "
            f"neurons. Rebuild the kernel with a larger CAP_WORDS "
            f"(hw/fpga/kernel/odin_fpga_kernel_top.v) or run fewer samples per "
            f"pass; the counts of a truncated run are not a result.")
    if len(words) < CAPTURE_HEADER_WORDS + written * CAPTURE_RECORD_WORDS:
        raise OdinFpgaKernelError(f"capture header claims {written} events but only "
                                  f"{len(words)} words came back: a partial read, not a result.")
    events: List[CaptureEvent] = []
    for index in range(written):
        base = CAPTURE_HEADER_WORDS + index * CAPTURE_RECORD_WORDS
        record = words[base:base + CAPTURE_RECORD_WORDS]
        events.append(CaptureEvent(
            core=int(record[RECORD_CORE]), neuron=int(record[RECORD_NEURON]),
            cycle=int(record[RECORD_CYCLE]), tag=int(record[RECORD_TAG])))
    return tuple(events), device_cycles


def capture_words(raw: Any) -> List[int]:
    """The capture buffer as little-endian words.

    ``pyxrt.bo.read`` answers with a numpy ``array_t<char>``, which reaches this
    function through the buffer protocol rather than as ``bytes``.
    """
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw]
    buffer = bytes(memoryview(raw))
    return list(struct.unpack(f"<{len(buffer) // WORD_BYTES}I", buffer))


def kernel_arity(xclbin_kernels: Iterable[Any]) -> Dict[str, int]:
    """``{name: num_args}`` from an ``xclbin.get_kernels()`` listing."""
    return {
        str(entry.get_name()): int(entry.get_num_args())
        for entry in xclbin_kernels
    }


def require_kernel_in_xclbin(
    arity: Dict[str, int], *, xclbin_path: str, transport: str,
    kernel_name: str = KERNEL_NAME,
) -> int:
    """The xclbin's own metadata must carry OUR kernel, with its arguments."""
    if kernel_name not in arity:
        raise DeviceTransportError(
            f"{transport}: {xclbin_path} declares kernels "
            f"{sorted(arity) or '[]'}, none of them {kernel_name!r}. This "
            f"bitstream is not the one this package drives, and loading it would "
            f"resolve a compute unit nobody built for these arguments")
    args = arity[kernel_name]
    if args and args < KERNEL_ARGS:
        raise DeviceTransportError(
            f"{transport}: {xclbin_path} declares {kernel_name} with {args} "
            f"argument(s), but the frozen kernel.xml has {KERNEL_ARGS} "
            f"(program, stimulus, capture, and their three word counts). The "
            f"bitstream was built from a different argument layout and the "
            f"scalars would land on the wrong offsets")
    return args


def memory_banks(xclbin_mems: Iterable[Any]) -> List[Dict[str, Any]]:
    """The xclbin's memory topology, as the probe reports it."""
    return [
        {
            "index": int(mem.get_index()),
            "tag": str(mem.get_tag()),
            "size_kb": int(mem.get_size_kb()),
            "used": bool(mem.get_used()),
        }
        for mem in xclbin_mems
    ]
