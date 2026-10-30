#!/usr/bin/env python3
"""The thin ODIN board driver: one frozen fixture in, one certified verdict out.

STANDALONE BY CONSTRUCTION. This file is the ONLY host code the HACC package
ships, it imports nothing but the standard library plus ``pyxrt`` (which ships
with the Xilinx Runtime, never with pip), and it never reads the mimarsinan
repository. That is what makes it uploadable: the board node needs the zip and
an XRT, nothing else.

NO REGISTER ACCESS EXISTS, AND THAT SHAPES EVERYTHING BELOW. The XRT Python
binding (github.com/Xilinx/XRT branch 2024.2,
src/python/pybind11/src/pyxrt.cpp) binds NO ``read_register`` and NO
``write_register`` on ``xrt::kernel``, and exposes no standalone ``ip`` object.
The kernel's AXI-Lite status and capacity registers at 0x4C/0x54 are real
— the RTL implements them and the RTL testbench reads them — but they are
unreachable from Python. On 2026-08-25 a U250 loaded this package's xclbin, the
CU came up, and the previous driver died at its first CSR touch. So every truth
here arrives one of exactly two ways:

  * through kernel ARGUMENTS — what the host declares (the three buffers and
    their three word counts, the frozen kernel.xml layout);
  * through MEMORY — what the fabric DMAs back: the capture header the kernel
    writes at drain time, and the fact that it wrote one AT ALL.

The capacities the fabric was compiled with cannot be asked of the card, so they
are DECLARED from the sources the xclbin was built from, and every refusal that
spends a capacity says so in its own text.

WHAT IT DOES NOT DO. It does not export, it does not simulate, it does not
re-derive a single expected count. The counts it certifies against were frozen
by ``scripts/hacc/make_package.py`` from the committed RTL cosimulation, and
the fixture carries their self-hash so a corrupted upload refuses instead of
certifying a payload nobody produced.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import platform
import socket
import struct
import sys
import time
import zlib
from typing import Any, Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------
# THE DEVICE PROTOCOL, frozen from the host-side SSOT
# (src/mimarsinan/chip_simulation/odin_fpga/kernel_registers.py). Every fixture
# also CARRIES this table under "kernel", and the driver refuses a fixture whose
# copy disagrees with this one: two tables that drifted would drive a kernel
# nobody built.
# --------------------------------------------------------------------------

KERNEL_NAME = "odin_fpga_kernel_top"

#: ROUTE D: the link may replicate the kernel into N compute units, unit k named
#: ``odin_<k>`` (scripts/hacc/gen_link_cfg.py derives the names from the card
#: config's one unit, ``nk=odin_fpga_kernel_top:1:odin_0``). A session holds ONE
#: unit when it is given a name, opening ``odin_fpga_kernel_top:{odin_<k>}``;
#: without a name it opens the kernel by its plain name, as every archived
#: session did (which, exclusive, takes every unit the xclbin has).
CU_NAME_PREFIX = "odin"


def compute_unit_name(index: int) -> str:
    """Unit ``index``'s name, as the link names it."""
    if int(index) < 0:
        raise OdinDriverError(f"compute unit index {index} is negative")
    return f"{CU_NAME_PREFIX}_{int(index)}"


def compute_unit_names(units: int) -> List[str]:
    """The names of an N-unit link's units, in index order."""
    return [compute_unit_name(k) for k in range(int(units))]


def cu_kernel_name(cu_name: str | None) -> str:
    """The kernel name a session opens: plain, or qualified by one unit."""
    return KERNEL_NAME if cu_name is None else f"{KERNEL_NAME}:{{{cu_name}}}"

#: The frozen kernel.xml argument order (scripts/hacc/gen_kernel_xml.py).
ARG_PROGRAM, ARG_STIMULUS, ARG_CAPTURE = 0, 1, 2
ARG_PROGRAM_WORDS, ARG_STIMULUS_WORDS, ARG_CAPTURE_WORDS = 3, 4, 5
KERNEL_ARGS = 6

WORD_BYTES = 4

#: The capture buffer as hw/fpga/kernel/odin_fpga_kernel.v writes it at drain
#: time (lines 171-189): header word 0 is `events_seen`, header word 1 is the
#: free-running `cycle` at run end, then one {tag, cycle, core, neuron} record
#: per AER-out event. `events_seen` is counted even when the RAM is full, which
#: is what makes the truncation refusal below possible.
CAPTURE_HEADER_WORDS = 2
HEADER_EVENTS_SEEN = 0
HEADER_DEVICE_CYCLES = 1
CAPTURE_RECORD_WORDS = 4
RECORD_TAG, RECORD_CYCLE, RECORD_CORE, RECORD_NEURON = 0, 1, 2, 3

#: The word the host DMAs into BOTH header slots before it starts the kernel.
#: The fabric overwrites them when it drains, so a header that comes back still
#: carrying this is the kernel having REFUSED the run — the only way `err`
#: reaches a host that cannot read 0x4C.
CAPTURE_NO_VERDICT = 0xFFFFFFFF

#: The sequencer's terminator (odin_fpga_kernel.v line 62). A one-word program
#: of nothing but this is the smallest thing the fabric can be asked to run, and
#: it is what B0's null run executes.
OP_END = 0

#: The token grammar CORE SELECTION walks: each opcode's argument count, and the
#: ops that address one core through their first argument (the sequencer latches
#: it as ``active_core``). The host copy of ``odin_rtl/stimulus.py``'s
#: ``OP_ARITY`` and ``CORE_ADDRESSED_OPS``; make_package.py REFUSES to package
#: when the two disagree, or when ``retarget_payload`` below and the encoder's
#: ``retarget_ops`` retarget a probe differently.
OP_ARITY: Dict[int, int] = {0: 0, 1: 3, 2: 3, 3: 2, 4: 1, 5: 1, 6: 3, 7: 4}
CORE_ADDRESSED_OPS = frozenset({1, 2, 3, 6, 7})

#: 0 is pyxrt's "block until the run completes" (pyxrt.cpp binds wait() to
#: xrt::run::wait(0)); a positive value bounds the wait in milliseconds.
BLOCK_UNTIL_DONE_MS = 0

#: What the SHIPPED fabric holds by default: the NC and CAP_WORDS defaults of
#: hw/fpga/kernel/odin_fpga_kernel_top.v. NC is a BUILD parameter
#: (ODIN_KERNEL_CORES, scripts/hacc/chips.sh; build_xclbn.sh packages it), and
#: a session declares the NC its bitstream was built with through that same
#: variable or --declare-cores. THE OP STREAM IS NOT AMONG THEM: the fabric
#: holds no copy of it, so there is no program capacity to declare and a run's
#: length is bounded only by the word counts the host passes as arguments.
SHIPPED_KERNEL_CORES = 1
#: The build variable that carries NC; the most cores one unit can address.
KERNEL_CORES_ENV = "ODIN_KERNEL_CORES"
KERNEL_CORES_MAX = 256
SHIPPED_CAPTURE_WORDS = 16384
SHIPPED_CAPTURE_EVENTS = (SHIPPED_CAPTURE_WORDS - CAPTURE_HEADER_WORDS) // (
    CAPTURE_RECORD_WORDS)
DEFAULT_CAPTURE_EVENTS = 1 << 20

#: WHICH FABRIC. The capture layout above belongs to the WRAPPER and is the same
#: for every chip configuration; the CORE geometry is not, so a session says
#: which fabric its declaration describes. This table is the host copy of
#: `src/mimarsinan/chip_simulation/odin_fpga/chip_configs.py` (the driver imports
#: nothing but stdlib), and make_package.py REFUSES to package when the two have
#: drifted. Defaults are the stock fabric's: the only one ever placed and routed.
SHIPPED_CHIP_CONFIG = "odin_stock_256x256"
SHIPPED_NEURONS_PER_CORE = 256
SHIPPED_AXON_SLOTS_PER_CORE = 128

CHIP_CONFIGS = {
    "odin_stock_256x256": {
        "cores": 1, "capture_events": SHIPPED_CAPTURE_EVENTS,
        "neurons_per_core": 256, "axon_slots_per_core": 128,
        "weight_sign_granularity": "per_axon",
    },
    "odin_wide_1024x256_mb16": {
        "cores": 1, "capture_events": SHIPPED_CAPTURE_EVENTS,
        "neurons_per_core": 256, "axon_slots_per_core": 1024,
        "weight_sign_granularity": "per_synapse",
    },
}

SHIPPED_CAPACITY_PROVENANCE = (
    "declared from hw/fpga/kernel/odin_fpga_kernel_top.v (CAP_WORDS = "
    f"{SHIPPED_CAPTURE_WORDS}) at NC = {SHIPPED_KERNEL_CORES}, the build default "
    f"of scripts/hacc/build_xclbn.sh ({KERNEL_CORES_ENV} unset) — NOT read back "
    "from the card, because the XRT Python binding exposes no register read"
)

BACKEND = "odin_fpga"
BACKEND_CLASS = "exact"
SCHEMA = "odin_hacc_fixture/2"


class OdinDriverError(RuntimeError):
    """The driver refused; the message names what and how to fix it."""


class OdinFpgaDependencyError(OdinDriverError):
    """``pyxrt`` is not importable in this environment."""


class OdinFpgaKernelError(OdinDriverError):
    """The kernel wrote no verdict: it refused the program it was given."""


class OdinFpgaCaptureTruncated(OdinDriverError):
    """The device saw at least as many events as the capture can hold."""


class OdinFixtureNeedsMoreCores(OdinDriverError):
    """This xclbin instantiates fewer ODIN cores than the fixture programs."""


class OdinCaptureFromUnplacedCore(OdinDriverError):
    """A capture record names a fabric core the run's placement never programmed."""


class OdinBuildRecordDisagrees(OdinDriverError):
    """The build record beside the xclbin contradicts the bitstream or the declaration."""


class OdinFixtureCorrupt(OdinDriverError):
    """A fixture's self-hash or payload hash does not match its bytes."""


class OdinFpgaWordsExceedBuffer(OdinDriverError):
    """A word-count argument declares more words than its buffer object holds."""


#: The device-protocol table every fixture carries and this driver checks.
KERNEL_TABLE: Dict[str, int | str] = {
    "name": KERNEL_NAME,
    "arg_program": ARG_PROGRAM,
    "arg_stimulus": ARG_STIMULUS,
    "arg_capture": ARG_CAPTURE,
    "arg_program_words": ARG_PROGRAM_WORDS,
    "arg_stimulus_words": ARG_STIMULUS_WORDS,
    "arg_capture_words": ARG_CAPTURE_WORDS,
    "kernel_args": KERNEL_ARGS,
    "capture_header_words": CAPTURE_HEADER_WORDS,
    "header_events_seen": HEADER_EVENTS_SEEN,
    "header_device_cycles": HEADER_DEVICE_CYCLES,
    "capture_record_words": CAPTURE_RECORD_WORDS,
    "capture_no_verdict": CAPTURE_NO_VERDICT,
    "word_bytes": WORD_BYTES,
}


# --------------------------------------------------------------------------
# Fixture sealing — ONE implementation, shared with the packager
# --------------------------------------------------------------------------


def canonical_bytes(document: Dict[str, Any]) -> bytes:
    """The byte image a fixture's self-hash is taken over (self_hash excluded)."""
    body = {key: value for key, value in document.items() if key != "self_hash"}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def self_hash(document: Dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(document)).hexdigest()


def seal(document: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp a fixture with its own hash; the packager and the tests share this."""
    sealed = dict(document)
    sealed.pop("self_hash", None)
    sealed["self_hash"] = self_hash(sealed)
    return sealed


def encode_payload(payload: bytes) -> Dict[str, Any]:
    """A byte payload as the fixture carries it: compressed, hashed, base64."""
    return {
        "encoding": "zlib+base64",
        "bytes": len(payload),
        "words": len(payload) // WORD_BYTES,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "data": base64.b64encode(zlib.compress(payload, 9)).decode("ascii"),
    }


def decode_payload(entry: Dict[str, Any], *, what: str) -> bytes:
    if entry.get("encoding") != "zlib+base64":
        raise OdinFixtureCorrupt(
            f"{what}: unknown payload encoding {entry.get('encoding')!r}; this "
            f"driver only decodes 'zlib+base64'")
    payload = zlib.decompress(base64.b64decode(entry["data"]))
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry["sha256"] or len(payload) != int(entry["bytes"]):
        raise OdinFixtureCorrupt(
            f"{what}: the decoded payload is {len(payload)} bytes / {digest}, "
            f"but the fixture declares {entry['bytes']} bytes / "
            f"{entry['sha256']}. The upload is damaged — re-copy the zip; a "
            f"device programmed from corrupted bytes runs a network nobody "
            f"assembled")
    return payload


def load_fixture(path: str) -> Dict[str, Any]:
    """Read one fixture and REFUSE unless it hashes to what it says it does."""
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema") != SCHEMA:
        raise OdinFixtureCorrupt(
            f"{path}: schema {document.get('schema')!r} is not {SCHEMA!r}")
    stamped = document.get("self_hash")
    recomputed = self_hash(document)
    if stamped != recomputed:
        raise OdinFixtureCorrupt(
            f"{path}: self-hash {stamped} but the content hashes to "
            f"{recomputed}. Either the upload is damaged or the file was "
            f"edited by hand; a fixture is frozen evidence and is not a "
            f"config to tune")
    table = document["kernel"]
    if table != KERNEL_TABLE:
        raise OdinFixtureCorrupt(
            f"{path}: the fixture's device-protocol table disagrees with this "
            f"driver's. Fixture={table}, driver={KERNEL_TABLE}. The package was "
            f"assembled against a different kernel — rebuild it with "
            f"scripts/hacc/make_package.py rather than mixing halves")
    return document


def fixture_paths(directory: str, names: Sequence[str]) -> List[str]:
    everything = sorted(
        os.path.join(directory, entry)
        for entry in os.listdir(directory)
        if entry.endswith(".json") and entry != "INDEX.json"
    )
    if not names:
        return everything
    chosen = []
    for name in names:
        match = os.path.join(directory, f"{name}.json")
        if not os.path.isfile(match):
            raise OdinDriverError(
                f"no fixture {name!r} in {directory}; shipped: "
                f"{[os.path.basename(p)[:-5] for p in everything]}")
        chosen.append(match)
    return chosen


# --------------------------------------------------------------------------
# The capacity nobody can read back, and the capture arithmetic
# --------------------------------------------------------------------------


class KernelCapacity:
    """What the loaded xclbin holds, DECLARED from what it was BUILT from."""

    def __init__(self, *, cores: int = SHIPPED_KERNEL_CORES,
                 capture_events: int = SHIPPED_CAPTURE_EVENTS,
                 provenance: str = SHIPPED_CAPACITY_PROVENANCE,
                 chip: str = SHIPPED_CHIP_CONFIG,
                 neurons_per_core: int = SHIPPED_NEURONS_PER_CORE,
                 axon_slots_per_core: int = SHIPPED_AXON_SLOTS_PER_CORE) -> None:
        self.cores = int(cores)
        self.capture_events = int(capture_events)
        self.provenance = str(provenance)
        self.chip = str(chip)
        self.neurons_per_core = int(neurons_per_core)
        self.axon_slots_per_core = int(axon_slots_per_core)

    def ceiling(self, host_events: int) -> int:
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


def chip_config_named(name):
    """One chip configuration's declaration, refusing a fabric this package
    does not carry rather than declaring a geometry nobody built."""
    try:
        return CHIP_CONFIGS[str(name)]
    except KeyError:
        raise OdinDriverError(
            "--chip %r names no fabric this package carries; it knows %s. A "
            "capacity declared for a bitstream nobody built would turn every "
            "refusal into a wrong number." % (name, ", ".join(sorted(CHIP_CONFIGS))))


def require_declared_storage(capacity: KernelCapacity, *, transport: str) -> None:
    if capacity.capture_events <= 0 or capacity.cores <= 0:
        raise OdinDriverError(
            f"{transport}: this package declares a capture capacity of "
            f"{capacity.capture_events} events across {capacity.cores} core(s) "
            f"({capacity.provenance}) — a kernel built with no storage cannot "
            f"run anything, and treating it as zero would turn every run into a "
            f"silent empty one")


def capture_buffer_bytes(capacity_events: int) -> int:
    return WORD_BYTES * (
        CAPTURE_HEADER_WORDS + CAPTURE_RECORD_WORDS * int(capacity_events))


def no_verdict_header() -> bytes:
    """The sentinel image the host DMAs into the capture header before a run."""
    return struct.pack(
        f"<{CAPTURE_HEADER_WORDS}I",
        *([CAPTURE_NO_VERDICT] * CAPTURE_HEADER_WORDS))


def stimulus_base_word(program_words: int) -> int:
    """The stimulus OVERWRITES the programming payload's END terminator."""
    return max(0, int(program_words) - 1)


#: The three metric names every programming stream is reported under. The fabric
#: holds NO copy of the op stream — it consumes it live out of host memory
#: through a shallow FIFO — so what used to be a fabric storage budget is now a
#: HOST-LINK BANDWIDTH, and the bandwidth is the deployment number.
STREAM_BYTES = "program_stream_bytes"
STREAM_SECONDS = "program_stream_seconds"
STREAM_RATE = "programming_bytes_per_second"


def stream_metrics(
    nbytes: int, seconds: float, *, link: bool = True,
) -> Dict[str, Any]:
    """One programming stream as a measurement: bytes, seconds, and the rate.

    ONE implementation, because both the fixture path and the deployment's
    per-core segment boundary report the same three numbers. On this driver the
    wall always measures the real host link (pyxrt bo write + sync), so callers
    here pass the default; the mirrored repo copy gates ``link`` on the
    transport's basis. The rate is ``None`` when the wall did not advance — a
    clock too coarse to time the transfer must not be published as zero — and
    ``None`` when the wall did not measure the link.
    """
    seconds = float(seconds)
    return {
        STREAM_BYTES: int(nbytes),
        STREAM_SECONDS: seconds,
        STREAM_RATE: (
            (float(nbytes) / seconds) if (link and seconds > 0.0) else None),
    }


def stream_line(metrics: Dict[str, Any]) -> str:
    """One human line for a programming stream — the same wording everywhere."""
    rate = metrics.get(STREAM_RATE)
    speed = "unmeasurably fast" if rate is None else f"{rate / 1e6:.1f} MB/s"
    return (
        f"programming stream {int(metrics[STREAM_BYTES])} bytes in "
        f"{float(metrics[STREAM_SECONDS]) * 1e3:.3f} ms = {speed} "
        f"(the fabric stores none of it)")


def require_words_fit_buffer(handle: Any, words: int, *, what: str,
                             transport: str) -> None:
    """A word count the fabric will stream must be inside its buffer object.

    The word-count arguments are now the ONLY bound on a run's length, so a
    count past the end of the buffer object it names is the one length fault
    that remains: the read engine would stream host memory this deployment
    never wrote into the sequencer.
    """
    have = int(handle.size())
    need = int(words) * WORD_BYTES
    if need > have:
        raise OdinFpgaWordsExceedBuffer(
            f"{transport}: the {what} argument declares {int(words)} word(s) = "
            f"{need} bytes, but the buffer object it names holds {have}. The "
            f"fabric stores no copy of the op stream — it reads exactly the "
            f"declared words out of this buffer while the sequencer runs — so "
            f"the read engine would stream past the end of the buffer and "
            f"execute whatever host memory follows it. The words argument is "
            f"the only bound on a run's length and it must be the payload's own")


def require_kernel_verdict(words: Sequence[int], *, transport: str) -> None:
    """The capture header after the run: an untouched sentinel means `err`."""
    if len(words) < CAPTURE_HEADER_WORDS:
        raise OdinDriverError(
            f"{transport}: the capture buffer came back shorter than its own "
            f"two-word header; the kernel never wrote a verdict and the run has "
            f"no counts")
    if all(int(word) == CAPTURE_NO_VERDICT
           for word in words[:CAPTURE_HEADER_WORDS]):
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


def capture_words(raw: Any) -> List[int]:
    """The capture buffer as little-endian words.

    ``pyxrt.bo.read`` answers with a numpy ``array_t<char>``, which arrives here
    through the buffer protocol rather than as ``bytes``.
    """
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw]
    buffer = bytes(memoryview(raw))
    return list(struct.unpack(f"<{len(buffer) // WORD_BYTES}I", buffer))


def decode_capture(
    words: Sequence[int], capacity: int, *, transport: str = BACKEND,
) -> Tuple[List[Tuple[int, int, int, int]], int]:
    """``(events, device_cycles)`` from the capture buffer's word image."""
    require_kernel_verdict(words, transport=transport)
    written = int(words[HEADER_EVENTS_SEEN])
    device_cycles = int(words[HEADER_DEVICE_CYCLES])
    if written >= int(capacity):
        raise OdinFpgaCaptureTruncated(
            f"the kernel SAW {written} events against a capacity of {capacity} "
            f"(min of this session's declaration and the capture RAM this "
            f"package declares the fabric was built with): the capture is at or "
            f"over its limit and the events past it would read as silent "
            f"neurons. Rebuild the kernel with a larger CAP_WORDS "
            f"(hw/fpga/kernel/odin_fpga_kernel_top.v) or run fewer samples per "
            f"pass; the counts of a truncated run are not a result.")
    needed = CAPTURE_HEADER_WORDS + written * CAPTURE_RECORD_WORDS
    if len(words) < needed:
        raise OdinFpgaKernelError(
            f"the capture header claims {written} events but the buffer "
            f"holds {len(words)} words where {needed} are needed: the "
            f"readback is shorter than the header's claim — a partial read "
            f"or a corrupted header, not a result.")
    events: List[Tuple[int, int, int, int]] = []
    for index in range(written):
        base = CAPTURE_HEADER_WORDS + index * CAPTURE_RECORD_WORDS
        record = words[base:base + CAPTURE_RECORD_WORDS]
        events.append((
            int(record[RECORD_TAG]), int(record[RECORD_CYCLE]),
            int(record[RECORD_CORE]), int(record[RECORD_NEURON])))
    return events, device_cycles


# --------------------------------------------------------------------------
# Core selection: logical cores on fabric cores
# --------------------------------------------------------------------------
#
# A bitstream built at NC holds NC cores behind ONE sequencer, and nothing
# routes between them in fabric (spikes between cores cross at segment
# boundaries through the host). A program addresses LOGICAL cores; a PLACEMENT
# ``{logical: fabric}`` says which fabric core each runs on. The stream is moved
# by rewriting the core argument of every core-addressed op and nothing else,
# so no word is added or dropped and the enabled-cycle schedule is unchanged.


def identity_placement(cores: int) -> Dict[int, int]:
    """Logical core k on fabric core k: where every archived run ran."""
    return {core: core for core in range(int(cores))}


def rotation_placements(logical: int, fabric: int) -> List[Dict[int, int]]:
    """Every rotation of ``logical`` cores around a ``fabric``-core bitstream.

    Rotation r puts logical core k on fabric core (k + r) mod NC, so over the NC
    rotations every fabric core hosts every logical core once: a fixture run at
    each rotation certifies each fabric core.
    """
    logical, fabric = int(logical), int(fabric)
    if logical > fabric:
        raise OdinFixtureNeedsMoreCores(
            f"{logical} logical core(s) need {logical} fabric cores and this "
            f"session declares NC {fabric}")
    return [{core: (core + offset) % fabric for core in range(logical)}
            for offset in range(fabric)]


def payload_words(payload: bytes, *, what: str) -> List[int]:
    if len(payload) % WORD_BYTES:
        raise OdinDriverError(
            f"{what}: {len(payload)} bytes is not a whole number of "
            f"{WORD_BYTES}-byte words")
    return list(struct.unpack(f"<{len(payload) // WORD_BYTES}I", payload))


def retarget_payload(payload: bytes, placement: Dict[int, int], *,
                     what: str = "payload") -> bytes:
    """``payload`` with every core-addressed op moved by ``placement``.

    The identity placement returns the very bytes. Anything the encoder would
    never write -- an unknown opcode, an op cut off by the end of the payload,
    a payload without its END or with words after it, a core the placement does
    not place, two logical cores on one fabric core -- is refused: a fabric
    handed it would run a program nobody mapped.
    """
    mapping = {int(logical): int(fabric) for logical, fabric in placement.items()}
    if len(set(mapping.values())) != len(mapping):
        raise OdinDriverError(
            f"{what}: placement {mapping} puts two logical cores on the same "
            f"fabric core")
    words = payload_words(payload, what=what)
    moved = list(words)
    index = 0
    while True:
        if index >= len(words):
            raise OdinDriverError(
                f"{what}: the token stream is not END-terminated; the sequencer "
                f"would run past it into whatever memory follows")
        code = int(words[index])
        if code == OP_END:
            if index != len(words) - 1:
                raise OdinDriverError(
                    f"{what}: {len(words) - 1 - index} word(s) after its END "
                    f"token; the encoder writes END last and nothing after it")
            break
        arity = OP_ARITY.get(code)
        if arity is None:
            raise OdinDriverError(
                f"{what}: unknown opcode {code} at word {index}; this driver's "
                f"grammar knows {sorted(OP_ARITY)}")
        if index + arity >= len(words):
            raise OdinDriverError(
                f"{what}: opcode {code} at word {index} needs {arity} "
                f"argument(s) and the stream ends inside them")
        if code in CORE_ADDRESSED_OPS:
            core = int(words[index + 1])
            if core not in mapping:
                raise OdinDriverError(
                    f"{what}: opcode {code} at word {index} addresses core "
                    f"{core}, which placement {mapping} does not place")
            moved[index + 1] = mapping[core]
        index += 1 + arity
    if moved == words:
        return payload
    return struct.pack(f"<{len(moved)}I", *moved)


def logical_events(events: Sequence[Tuple[int, int, int, int]],
                   placement: Dict[int, int], *, transport: str
                   ) -> Tuple[List[Tuple[int, int, int, int]], Dict[int, int]]:
    """Capture records back on LOGICAL cores, and each logical core's count.

    A record from a fabric core the placement did not program is refused: an
    idle core that spiked, or a capture field that is not a core, is not part
    of the run being certified.
    """
    inverse = {int(fabric): int(logical) for logical, fabric in placement.items()}
    per_core = {int(logical): 0 for logical in placement}
    moved: List[Tuple[int, int, int, int]] = []
    for tag, cycle, fabric, neuron in events:
        logical = inverse.get(int(fabric))
        if logical is None:
            raise OdinCaptureFromUnplacedCore(
                f"{transport}: a capture record names fabric core {fabric} "
                f"(tag {tag}, neuron {neuron}), which placement {placement} did "
                f"not program; an idle core spiked or the record is not a core's, "
                f"and the run is not the one being certified")
        per_core[logical] += 1
        moved.append((int(tag), int(cycle), logical, int(neuron)))
    return moved, per_core


# --------------------------------------------------------------------------
# Counts: capture events -> per-cycle -> per-core windows
# --------------------------------------------------------------------------


def fold_events(events, run: Dict[str, Any]) -> Dict[Tuple[int, int, int, int], int]:
    """Capture events into ``(sample, cycle, core, neuron)`` counts by their tag."""
    first_tag = int(run["first_tag"])
    per_sample = int(run["cycles_per_sample"])
    counts: Dict[Tuple[int, int, int, int], int] = {}
    for tag, _device_cycle, core, neuron in events:
        offset = int(tag) - first_tag
        sample, cycle = divmod(offset, per_sample)
        key = (sample, cycle, int(core), int(neuron))
        counts[key] = counts.get(key, 0) + 1
    return counts


def window_counts(counts, run: Dict[str, Any]) -> List[List[List[int]]]:
    """Each core's own window ``[latency, latency + T)`` — nevresim's convention.

    The one folding rule a cosimulated run and a board run may be summed by;
    it is the arithmetic of ``odin_rtl/cosim.py::window_counts_of``.
    """
    samples = int(run["samples"])
    per_sample = int(run["cycles_per_sample"])
    latencies = [int(v) for v in run["latencies"]]
    neurons = [int(v) for v in run["neurons"]]
    length = int(run["simulation_length"])
    totals: List[List[List[int]]] = []
    for sample in range(samples):
        per_core: List[List[int]] = []
        for core, count in enumerate(neurons):
            row = [0] * count
            for cycle in range(per_sample):
                local = cycle - latencies[core]
                if not 0 <= local < length:
                    continue
                for neuron in range(count):
                    row[neuron] += counts.get((sample, cycle, core, neuron), 0)
            per_core.append(row)
        totals.append(per_core)
    return totals


def per_cycle_table(expected: Sequence[Sequence[int]]) -> Dict[Tuple[int, int, int, int], int]:
    return {
        (int(s), int(c), int(k), int(n)): int(v) for s, c, k, n, v in expected
    }


# --------------------------------------------------------------------------
# The certificate, in the house format
# --------------------------------------------------------------------------


class Certificate:
    """One reference<->board spike-count comparison, printed the house way."""

    def __init__(self, reference, measured, *, backend=BACKEND, samples=0):
        if len(reference) != len(measured):
            raise OdinDriverError(
                f"the board reported {len(measured)} sample(s) against the "
                f"fixture's {len(reference)}; a truncated comparison would "
                f"certify a run nobody made")
        compared = matched = 0
        max_delta = 0
        divergent: List[Tuple[int, int, int, int, int]] = []
        for sample, (ref_cores, got_cores) in enumerate(zip(reference, measured)):
            if len(ref_cores) != len(got_cores):
                raise OdinDriverError(
                    f"sample {sample}: the board reported {len(got_cores)} "
                    f"cores against the fixture's {len(ref_cores)}")
            for core, (ref_row, got_row) in enumerate(zip(ref_cores, got_cores)):
                if len(ref_row) != len(got_row):
                    raise OdinDriverError(
                        f"sample {sample} core {core}: width "
                        f"{len(got_row)} against {len(ref_row)}")
                for neuron, (ref, got) in enumerate(zip(ref_row, got_row)):
                    delta = abs(int(got) - int(ref))
                    compared += 1
                    matched += delta == 0
                    if delta > max_delta:
                        max_delta = delta
                    if delta and len(divergent) < 16:
                        divergent.append(
                            (sample, core, neuron, int(ref), int(got)))
        if compared == 0:
            raise OdinDriverError(
                "the certificate compared ZERO neuron-windows — a vacuous pass "
                "is not a certificate")
        self.backend = backend
        self.backend_class = BACKEND_CLASS
        self.samples = int(samples)
        self.neuron_windows_compared = compared
        self.exact_match_fraction = matched / compared
        self.max_abs_delta = float(max_delta)
        self.divergent = divergent
        self.passed = max_delta == 0

    def summary(self) -> str:
        return (
            f"spike-count certificate [{self.backend}/{self.backend_class}]: "
            f"{'PASS' if self.passed else 'FAIL'} "
            f"exact={self.exact_match_fraction:.6f} "
            f"max|dcount|={self.max_abs_delta:g} "
            f"over {self.neuron_windows_compared} neuron-windows, "
            f"{self.samples} sample(s)"
        )

    def line(self) -> str:
        return f"[SpikeCountCertificate] {self.summary()}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend, "backend_class": self.backend_class,
            "samples": self.samples,
            "neuron_windows_compared": self.neuron_windows_compared,
            "exact_match_fraction": self.exact_match_fraction,
            "max_abs_delta": self.max_abs_delta,
            "divergent": self.divergent, "passed": self.passed,
            "summary": self.summary(),
        }


# --------------------------------------------------------------------------
# The XRT session
# --------------------------------------------------------------------------


#: One fake module instance per path: a fake carries the state the board would
#: hold (which fixture it is answering, which fault it is simulating), so every
#: session in a process must see the SAME object or arming it would be lost.
_FAKE_MODULES: Dict[str, Any] = {}


def load_pyxrt(fake_path: str | None = None) -> Any:
    """The XRT binding, a named fake, or a refusal that names the fix."""
    if fake_path:
        cached = _FAKE_MODULES.get(fake_path)
        if cached is not None:
            return cached
        spec = importlib.util.spec_from_file_location(
            "odin_fake_pyxrt", fake_path)
        if spec is None or spec.loader is None:
            raise OdinDriverError(f"--fake-pyxrt {fake_path}: not importable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FAKE_MODULES[fake_path] = module
        return module
    try:
        import pyxrt  # type: ignore[import-not-found]  # ships with XRT, never pip
    except ImportError as exc:
        raise OdinFpgaDependencyError(
            f"pyxrt is not importable, so no Alveo device can be reached "
            f"({exc}). pyxrt ships with the Xilinx Runtime, not with pip: "
            f"source /opt/xilinx/xrt/setup.sh on a board node (README_HACC.md) "
            f"before running against a card. To exercise this whole driver "
            f"with no hardware at all, pass "
            f"--fake-pyxrt host/fake_pyxrt_for_selftest.py") from exc
    return pyxrt


#: The build record ``scripts/hacc/build_xclbn.sh`` writes beside every xclbin it
#: links: ``key=value`` lines carrying the xclbin's sha256 and UUID and the NC
#: (``kernel_cores``) its kernel was packaged at.
BUILD_RECORD_SUFFIX = ".identity"


def _uuid_text(uuid: Any) -> str | None:
    """The loaded xclbin's UUID as text, or None when the binding gives none."""
    try:
        return str(uuid.to_string())
    except Exception:  # a binding without to_string: the sha256 check still stands
        return None


def read_build_record(xclbin_path: str) -> Dict[str, str] | None:
    """The build record beside ``xclbin_path``, or None when there is none."""
    path = f"{xclbin_path}{BUILD_RECORD_SUFFIX}"
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return dict(line.rstrip("\n").split("=", 1)
                    for line in handle if "=" in line)


def require_build_record_agrees(xclbin_path: str, *, loaded_uuid: str | None,
                                declared_cores: int, transport: str = BACKEND
                                ) -> Dict[str, Any]:
    """Refuse a session whose declared NC contradicts the build that made the xclbin.

    The record must describe THIS file (its sha256) and THIS loaded bitstream
    (its UUID, where the build read one); its NC must then equal the session's
    declared one. Without a record the declaration stands, and the stamp says
    that nothing verified it.
    """
    record = read_build_record(xclbin_path)
    if record is None:
        return {"verified": False, "path": None,
                "note": (f"no build record beside {xclbin_path} "
                         f"({BUILD_RECORD_SUFFIX}): the declared NC "
                         f"{int(declared_cores)} is not verified against the build")}
    path = f"{xclbin_path}{BUILD_RECORD_SUFFIX}"
    digest = hashlib.sha256()
    with open(xclbin_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if record.get("xclbin_sha256") != digest.hexdigest():
        raise OdinBuildRecordDisagrees(
            f"{transport}: {path} records xclbin sha256 "
            f"{record.get('xclbin_sha256')}, but {xclbin_path} hashes to "
            f"{digest.hexdigest()}: the record describes another bitstream, so "
            f"it cannot say what NC this one was built with")
    recorded_uuid = record.get("xclbin_uuid", "unknown")
    if (recorded_uuid not in ("", "unknown") and loaded_uuid is not None
            and recorded_uuid.lower() != str(loaded_uuid).lower()):
        raise OdinBuildRecordDisagrees(
            f"{transport}: {path} records xclbin UUID {recorded_uuid}, but the "
            f"card loaded UUID {loaded_uuid}: the record is not this bitstream's")
    try:
        built_cores = int(record["kernel_cores"])
    except (KeyError, ValueError) as exc:
        raise OdinBuildRecordDisagrees(
            f"{transport}: {path} carries no integer kernel_cores") from exc
    if built_cores != int(declared_cores):
        raise OdinBuildRecordDisagrees(
            f"{transport}: this session declares NC {int(declared_cores)}, but "
            f"{xclbin_path} was built with NC {built_cores} ({path}). A placement "
            f"past the fabric would address a core that does not exist, and one "
            f"inside it would leave cores the build made unused; declare "
            f"--declare-cores {built_cores} (or ODIN_KERNEL_CORES={built_cores})")
    return {"verified": True, "path": path, "kernel_cores": built_cores,
            "xclbin_uuid": recorded_uuid, "xclbin_sha256": digest.hexdigest(),
            "chip_config": record.get("chip_config"),
            "compute_units": record.get("compute_units")}


class BoardSession:
    """One XRT session against an ODIN kernel on an XDMA shell."""

    name = "xrt"

    def __init__(self, *, xclbin_path: str, device_index: int = 0,
                 capture_events: int = DEFAULT_CAPTURE_EVENTS,
                 capacity: KernelCapacity | None = None,
                 run_timeout_ms: int = BLOCK_UNTIL_DONE_MS,
                 fake_pyxrt: str | None = None, cu_name: str | None = None,
                 open_kernel: bool = True) -> None:
        if not xclbin_path:
            raise OdinDriverError(
                "the board needs an xclbin: the bitstream is the device's "
                "program and there is no default one (--xclbin)")
        self.xclbin_path = str(xclbin_path)
        self.device_index = int(device_index)
        self.capture_events = int(capture_events)
        self.run_timeout_ms = int(run_timeout_ms)
        self.capacity = capacity if capacity is not None else KernelCapacity()
        self.fake_pyxrt = fake_pyxrt
        #: The one compute unit this session holds (route D), or None for the
        #: archived plain-name session; ``open_kernel=False`` loads the xclbin
        #: and holds no unit at all, so N processes can each open one.
        self.cu_name = cu_name
        self.open_kernel = bool(open_kernel)
        self._xrt: Any = None
        self._device: Any = None
        self._kernel: Any = None
        self._capture_capacity = 0
        self._xclbin_kernels: Dict[str, int] = {}
        self._memory_banks: List[Dict[str, Any]] = []
        self._identity: Dict[str, Any] = {}
        self._build_record: Dict[str, Any] = {}
        #: Every kernel start this session issued (``start_and_wait``), counted
        #: here so a window can report how many it made instead of asserting it.
        self.kernel_starts = 0

    def open(self) -> None:
        require_declared_storage(self.capacity, transport=self.name)
        self._xrt = load_pyxrt(self.fake_pyxrt)
        # The xclbin's OWN metadata, before it is pushed to a card: a bitstream
        # that does not declare this kernel is not the one this package drives.
        image = self._xrt.xclbin(self.xclbin_path)
        self._xclbin_kernels = {
            str(entry.get_name()): int(entry.get_num_args())
            for entry in image.get_kernels()
        }
        self._require_our_kernel()
        self._memory_banks = [
            {"index": int(mem.get_index()), "tag": str(mem.get_tag()),
             "size_kb": int(mem.get_size_kb()), "used": bool(mem.get_used())}
            for mem in image.get_mems()
        ]
        self._device = self._xrt.device(self.device_index)
        load_started = time.monotonic()
        uuid = self._device.load_xclbin(image)
        load_s = time.monotonic() - load_started
        # The declared NC against the NC this bitstream was BUILT with, before
        # any run: a declaration is only trusted where nothing records the build.
        self._build_record = require_build_record_agrees(
            self.xclbin_path, loaded_uuid=_uuid_text(uuid),
            declared_cores=self.capacity.cores, transport=self.name)
        if self.open_kernel:
            try:
                self._kernel = self._xrt.kernel(
                    self._device, uuid, cu_kernel_name(self.cu_name),
                    # Exclusive access: the deployment owns the board for the
                    # whole reservation, and a shared CU would let another job's
                    # run interleave with this one's capture buffer.
                    self._xrt.kernel.cu_access_mode.exclusive,
                )
            except Exception as exc:  # the binding's own error, typed here
                if self.cu_name is None:
                    raise
                raise OdinDriverError(
                    f"{self.name}: {self.xclbin_path} opened no compute unit named "
                    f"{self.cu_name!r} ({cu_kernel_name(self.cu_name)}): "
                    f"{type(exc).__name__}: {exc}. The xclbin was linked with fewer "
                    f"units than this session declares, or another process holds "
                    f"that unit exclusively") from exc
        self._capture_capacity = self.capacity.ceiling(self.capture_events)
        self._identity = self._read_identity(image, uuid, load_s)

    def _read_identity(self, image: Any, loaded_uuid: Any, load_s: float) -> Dict[str, Any]:
        """WHICH card and WHICH bitstream: the BDF and the xclbin UUID.

        Read once, after the kernel is resolved, and never allowed to fail a
        run: the binding's ``get_info`` is on the recorded XRT 2024.2 surface,
        but its behaviour inside the board VM is unverified, so a failure is
        recorded next to the stamp instead of refusing a deployment that would
        otherwise run.
        """
        identity: Dict[str, Any] = {"xclbin_load_s": load_s, "identity_errors": []}
        if self.cu_name is not None:
            identity["compute_unit"] = self.cu_name
        for field, read in (
            ("xclbin_uuid", lambda: loaded_uuid.to_string()),
            ("xclbin_uuid_metadata", lambda: image.get_uuid().to_string()),
            ("bdf", lambda: self._device.get_info(self._xrt.xrt_info_device.bdf)),
        ):
            try:
                identity[field] = str(read())
            except Exception as exc:  # the binding's own errors are recorded, not typed
                identity[field] = None
                identity["identity_errors"].append(
                    f"{field}: {type(exc).__name__}: {exc}")
        return identity

    def _require_our_kernel(self) -> None:
        if KERNEL_NAME not in self._xclbin_kernels:
            raise OdinDriverError(
                f"{self.name}: {self.xclbin_path} declares kernels "
                f"{sorted(self._xclbin_kernels) or '[]'}, none of them "
                f"{KERNEL_NAME!r}. This bitstream is not the one this package "
                f"drives, and loading it would resolve a compute unit nobody "
                f"built for these arguments")
        args = self._xclbin_kernels[KERNEL_NAME]
        if args and args < KERNEL_ARGS:
            raise OdinDriverError(
                f"{self.name}: {self.xclbin_path} declares {KERNEL_NAME} with "
                f"{args} argument(s), but the frozen kernel.xml has "
                f"{KERNEL_ARGS} (program, stimulus, capture, and their three "
                f"word counts). The bitstream was built from a different "
                f"argument layout and the scalars would land on wrong offsets")

    def close(self) -> None:
        self._kernel = None
        self._device = None
        self._capture_capacity = 0
        self._xclbin_kernels = {}
        self._memory_banks = []
        self._identity = {}

    @property
    def capture_capacity(self) -> int:
        return self._capture_capacity

    @property
    def cores_implied(self) -> int:
        """How many ODIN cores this bitstream declares it was built with."""
        return self.capacity.cores

    def arm_fake(self, **state: Any) -> bool:
        """Hand a FAKE pyxrt the frozen answers it should replay.

        The real binding has no ``arm``, so this is a no-op against a card —
        which is what keeps the arming hook out of the device path.
        """
        arm = getattr(self._xrt, "arm", None)
        if arm is None:
            return False
        arm(**state)
        return True

    def _require_session(self, what: str) -> Any:
        if self._xrt is not None and self._device is not None and not self.open_kernel:
            raise OdinDriverError(
                f"{self.name}: {what} on a session that loaded the xclbin and holds no "
                f"compute unit (open_kernel=False); open a unit to move buffers or "
                f"run the kernel")
        if self._kernel is None or self._xrt is None:
            raise OdinDriverError(
                f"{self.name}: {what} before a live session — open() loads the "
                f"xclbin and resolves the kernel, and nothing can be written "
                f"to a device that was never opened")
        return self._xrt

    def allocate(self, nbytes: int, arg: int) -> Any:
        """One buffer object in the memory bank this kernel argument lives in."""
        xrt = self._require_session("buffer allocation")
        return xrt.bo(self._device, int(nbytes), xrt.bo.flags.normal,
                      self._kernel.group_id(int(arg)))

    def _buffer(self, nbytes: int, arg: int) -> Any:
        return self.allocate(nbytes, arg)

    def dma_in(self, handle: Any, payload: bytes) -> Dict[str, float]:
        """One host->device transfer, timed as the two distinct stages it is.

        ``bo.write`` is a host memcpy into the buffer object and ``bo.sync`` is
        the DMA that moves it across PCIe; a deployment that reports one number
        for both cannot say which of the two a slow pass spent its time in.
        """
        xrt = self._require_session("host-to-device DMA")
        started = time.perf_counter()
        handle.write(payload, 0)
        written = time.perf_counter()
        handle.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(payload), 0)
        return {"bo_write_s": written - started,
                "sync_s": time.perf_counter() - written}

    def _to_device(self, handle: Any, payload: bytes) -> None:
        self.dma_in(handle, payload)

    def poison_capture(self, capture_bo: Any) -> Dict[str, float]:
        """Write the no-verdict sentinel into a capture buffer's header.

        The fabric overwrites both header words at drain time, so a header that
        comes back untouched is the kernel refusing — the only `err` signal a
        host with no register access can see. Every run needs a FRESH sentinel,
        which is why this is separate from allocating the buffer.
        """
        return self.dma_in(capture_bo, no_verdict_header())

    def _capture_buffer(self, nbytes: int) -> Any:
        """A capture buffer whose header carries the host's no-verdict sentinel."""
        capture_bo = self.allocate(nbytes, ARG_CAPTURE)
        self.poison_capture(capture_bo)
        return capture_bo

    def start_and_wait(self, program_bo: Any, stimulus_bo: Any, capture_bo: Any,
                       *, program_words: int, stimulus_words: int,
                       capture_events: int) -> float:
        """The six arguments in, ap_done out — the measured DEVICE wall."""
        xrt = self._require_session("a device run")
        require_words_fit_buffer(
            program_bo, program_words, what="program_words",
            transport=self.name)
        require_words_fit_buffer(
            stimulus_bo, stimulus_words, what="stimulus_words",
            transport=self.name)
        require_words_fit_buffer(
            capture_bo,
            CAPTURE_HEADER_WORDS + CAPTURE_RECORD_WORDS * int(capture_events),
            what="capture_events", transport=self.name)
        started = time.perf_counter()
        self.kernel_starts += 1
        handle = self._kernel(
            program_bo, stimulus_bo, capture_bo,
            int(program_words), int(stimulus_words), int(capture_events),
        )
        state = handle.wait(self.run_timeout_ms)
        wall = time.perf_counter() - started
        completed = xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
        if state != completed:
            raise OdinFpgaKernelError(
                f"{self.name}: the kernel run ended in state {state!r}, not "
                f"{completed!r}. XRT never saw ap_done, so nothing was captured "
                f"and there is no verdict in memory to decode")
        return wall

    def read_capture(self, capture_bo: Any, nbytes: int
                     ) -> Tuple[List[int], Dict[str, float]]:
        """The capture buffer back on the host, timed as sync then read."""
        xrt = self._require_session("a capture readback")
        started = time.perf_counter()
        capture_bo.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, int(nbytes), 0)
        synced = time.perf_counter()
        words = capture_words(capture_bo.read(int(nbytes), 0))
        return words, {"sync_s": synced - started,
                       "readback_s": time.perf_counter() - synced}

    def _execute(self, program: bytes, stimulus: bytes,
                 capture_events: int) -> Tuple[List[int], Dict[str, Any]]:
        """Program in, stimulus in, kernel run, capture out — the whole protocol.

        Nothing here reads a register: the run's verdict is the capture header,
        and whether the fabric wrote one at all. The programming phase is
        MEASURED rather than budgeted — its wall, its bytes and the bandwidth
        they imply are what a streamed chip costs at a segment boundary.
        """
        self._require_session("a device run")
        program_words = len(program) // WORD_BYTES
        stimulus_words = len(stimulus) // WORD_BYTES

        started = time.monotonic()
        program_bo = self.allocate(len(program), ARG_PROGRAM)
        stream = self.dma_in(program_bo, program)
        programming_s = time.monotonic() - started

        started = time.monotonic()
        stimulus_bo = self.allocate(len(stimulus), ARG_STIMULUS)
        self.dma_in(stimulus_bo, stimulus)
        capture_bytes = capture_buffer_bytes(capture_events)
        capture_bo = self._capture_buffer(capture_bytes)
        self.start_and_wait(
            program_bo, stimulus_bo, capture_bo, program_words=program_words,
            stimulus_words=stimulus_words, capture_events=capture_events)
        words, _walls = self.read_capture(capture_bo, capture_bytes)
        execution_s = time.monotonic() - started
        walls: Dict[str, Any] = {
            "programming_s": programming_s,
            "execution_s": execution_s,
            # The same wall, named for what it IS at a segment boundary: the
            # buffer setup plus the programming stream this core needs before
            # any sample of it can run.
            "segment_boundary_init_s": programming_s,
        }
        # The BANDWIDTH is taken over the stream's own wall — the buffer write
        # and the DMA — and not over the allocation the boundary also pays for.
        walls.update(stream_metrics(
            len(program), stream["bo_write_s"] + stream["sync_s"]))
        return words, walls

    def device_stamp(self) -> Dict[str, Any]:
        """Everything this session knows about what it is talking to."""
        stamp = self._device_stamp()
        if self.cu_name is not None:
            stamp["compute_unit"] = {
                "name": self.cu_name, "kernel": cu_kernel_name(self.cu_name),
                "group_ids": dict(stamp["group_ids"]),
            }
        return stamp

    def _device_stamp(self) -> Dict[str, Any]:
        return {
            "xclbin": self.xclbin_path,
            "kernel": KERNEL_NAME,
            "device_index": self.device_index,
            "cu_access_mode": "exclusive",
            "capture_capacity": self._capture_capacity,
            "cores_implied": self.cores_implied,
            "capacity": self.capacity.as_dict(),
            "xclbin_kernels": dict(self._xclbin_kernels),
            "memory_banks": list(self._memory_banks),
            "group_ids": {
                str(arg): int(self._kernel.group_id(arg))
                for arg in (ARG_PROGRAM, ARG_STIMULUS, ARG_CAPTURE)
            } if self._kernel is not None else {},
            "identity": dict(self._identity),
            "build_record": dict(self._build_record),
        }

    @property
    def identity(self) -> Dict[str, Any]:
        """The BDF, the loaded xclbin UUID and the load wall of this session."""
        return dict(self._identity)

    def null_run(self) -> Dict[str, Any]:
        """B0's smallest possible run: one END token in, one header back.

        It proves the whole path — arguments reach the CU, the AXI master reads
        host memory, the sequencer executes, the capture engine writes its
        header and the master writes it back — without programming a network.
        """
        self._require_session("null_run()")
        terminator = struct.pack("<I", OP_END)
        words, walls = self._execute(terminator, terminator, 1)
        require_kernel_verdict(words, transport=self.name)
        events_seen = int(words[HEADER_EVENTS_SEEN])
        if events_seen != 0:
            raise OdinDriverError(
                f"{self.name}: the null program (one END token, no network "
                f"programmed) came back reporting {events_seen} AER event(s). "
                f"A fabric that spikes with no weights loaded is not this "
                f"design, and no count it produces afterwards is a result")
        return {
            "header_words": words[:CAPTURE_HEADER_WORDS],
            "events_seen": events_seen,
            "device_cycles": int(words[HEADER_DEVICE_CYCLES]),
            "walls": walls,
        }

    def run_fixture(self, fixture: Dict[str, Any],
                    placement: Dict[int, int] | None = None) -> Dict[str, Any]:
        """Program, stimulate, wait, read the header, decode — in that order.

        ``placement`` maps the fixture's logical cores onto this bitstream's
        fabric cores (default: core k on core k, where every archived run ran);
        the program and the stimulus are retargeted onto it and the capture is
        decoded back per core, with one certificate per logical core beside the
        fixture's own.
        """
        self._require_session("run_fixture()")
        run = fixture["run"]
        needed_cores = int(run["cores"])
        if needed_cores > self.cores_implied:
            raise OdinFixtureNeedsMoreCores(
                f"{fixture['name']}: the fixture programs {needed_cores} ODIN "
                f"core(s) but this package declares NC {self.cores_implied} "
                f"({self.capacity.provenance}). Run it on a bitstream built "
                f"with {KERNEL_CORES_ENV} >= {needed_cores} "
                f"(scripts/hacc/build_xclbn.sh), declared the same way")
        placement = self._require_placement(fixture, placement)
        identity = placement == identity_placement(needed_cores)
        # A FAKE is handed the fixture so it can answer with the capture image
        # the frozen evidence says the fabric produced, on the fabric cores the
        # placement names; real pyxrt has no arm().
        self.arm_fake(fixture=fixture, placement=placement)

        program = decode_payload(fixture["program"], what=f"{fixture['name']}/program")
        stimulus = decode_payload(fixture["stimulus"], what=f"{fixture['name']}/stimulus")
        if not identity:
            program = retarget_payload(
                program, placement, what=f"{fixture['name']}/program")
            stimulus = retarget_payload(
                stimulus, placement, what=f"{fixture['name']}/stimulus")
        words, walls = self._execute(
            program, stimulus, self._capture_capacity)

        events, device_cycles = decode_capture(
            words, self._capture_capacity, transport=self.name)
        events, per_core_events = logical_events(
            events, placement, transport=self.name)
        counts = fold_events(events, run)
        measured = window_counts(counts, run)
        expected = fixture["expected"]["window"]
        certificate = Certificate(
            expected, measured, samples=int(run["samples"]))
        per_core = []
        for core in range(needed_cores):
            core_certificate = Certificate(
                [[sample[core]] for sample in expected],
                [[sample[core]] for sample in measured],
                samples=int(run["samples"]))
            per_core.append({
                "logical_core": core, "fabric_core": int(placement[core]),
                "events": int(per_core_events[core]),
                "window_counts": [list(sample[core]) for sample in measured],
                "certificate": core_certificate.as_dict(),
                "certificate_line": core_certificate.line(),
            })
        return {
            "fixture": fixture["name"],
            "self_hash": fixture["self_hash"],
            "transport": self.name,
            "device": self.device_stamp(),
            "placement": {str(core): int(fabric)
                          for core, fabric in sorted(placement.items())},
            "program_bytes": len(program),
            "program_words": len(program) // WORD_BYTES,
            "stimulus_words": len(stimulus) // WORD_BYTES,
            "capture_events": len(events),
            "events_seen": int(words[HEADER_EVENTS_SEEN]),
            "device_cycles": device_cycles,
            "walls": walls,
            "window_counts": measured,
            "per_core": per_core,
            "certificate": certificate.as_dict(),
            "certificate_line": certificate.line(),
            "passed": certificate.passed,
        }

    def _require_placement(self, fixture: Dict[str, Any],
                           placement: Dict[int, int] | None) -> Dict[int, int]:
        """The fixture's logical cores, each on a distinct fabric core of this NC."""
        cores = int(fixture["run"]["cores"])
        if placement is None:
            return identity_placement(cores)
        mapping = {int(logical): int(fabric) for logical, fabric in placement.items()}
        if sorted(mapping) != list(range(cores)):
            raise OdinDriverError(
                f"{fixture['name']}: placement {mapping} does not place exactly "
                f"the fixture's logical cores 0..{cores - 1}")
        for logical, fabric in sorted(mapping.items()):
            if not 0 <= fabric < self.cores_implied:
                raise OdinFixtureNeedsMoreCores(
                    f"{fixture['name']}: placement {mapping} puts logical core "
                    f"{logical} on fabric core {fabric}, and this package "
                    f"declares NC {self.cores_implied} "
                    f"({self.capacity.provenance})")
        if len(set(mapping.values())) != len(mapping):
            raise OdinDriverError(
                f"{fixture['name']}: placement {mapping} puts two logical cores "
                f"on the same fabric core")
        return mapping


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def host_stamp() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_result(results_dir: str, name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def declared_kernel_cores() -> int | None:
    """NC as the build declared it (``ODIN_KERNEL_CORES``), or None when unset."""
    raw = os.environ.get(KERNEL_CORES_ENV, "").strip()
    if not raw:
        return None
    if not raw.isdigit() or not 1 <= int(raw) <= KERNEL_CORES_MAX:
        raise OdinDriverError(
            f"{KERNEL_CORES_ENV}={raw!r} is not an integer from 1 to "
            f"{KERNEL_CORES_MAX}: it is the number of ODIN cores the bitstream "
            f"was built with (scripts/hacc/chips.sh), and a session declares "
            f"exactly that")
    return int(raw)


def capacity_from_options(options) -> KernelCapacity:
    """The capacity this session declares, and where that number came from.

    NC comes from ``--declare-cores``, else from the build variable the
    bitstream was built with (``ODIN_KERNEL_CORES``), else from the package's
    default of 1; every one of them is a declaration, and the provenance says
    which.
    """
    chip_name = str(getattr(options, "chip", "") or SHIPPED_CHIP_CONFIG)
    chip = chip_config_named(chip_name)
    built_cores = declared_kernel_cores()
    cores = int(options.declare_cores or built_cores or chip["cores"])
    capture_ram = int(options.capture_ram_events or chip["capture_events"])
    provenance = SHIPPED_CAPACITY_PROVENANCE
    if built_cores is not None:
        provenance = (
            f"declared from {KERNEL_CORES_ENV}={built_cores}, the NC the "
            f"bitstream was built with (scripts/hacc/build_xclbn.sh packages "
            f"the kernel at it), and from hw/fpga/kernel/odin_fpga_kernel_top.v "
            f"(CAP_WORDS = {SHIPPED_CAPTURE_WORDS}) — NOT read back from the "
            f"card, because the XRT Python binding exposes no register read")
    overrides = []
    if chip_name != SHIPPED_CHIP_CONFIG:
        overrides.append(f"--chip {chip_name}")
    if options.declare_cores:
        overrides.append(f"--declare-cores {cores}")
    if options.capture_ram_events:
        overrides.append(f"--capture-ram-events {capture_ram}")
    if overrides:
        provenance = (
            f"OVERRIDDEN on the command line ({', '.join(overrides)}); the "
            f"package's own declaration is: {provenance}")
    return KernelCapacity(
        cores=cores, capture_events=capture_ram, provenance=provenance,
        chip=chip_name, neurons_per_core=int(chip["neurons_per_core"]),
        axon_slots_per_core=int(chip["axon_slots_per_core"]))


def open_session(options, capacity: KernelCapacity | None = None, *,
                 open_kernel: bool = True) -> BoardSession:
    """A live session; ``options.cu`` names the one unit it holds (route D)."""
    session = BoardSession(
        xclbin_path=options.xclbin, device_index=options.device_index,
        capture_events=options.capture_events,
        capacity=capacity if capacity is not None else capacity_from_options(options),
        run_timeout_ms=options.run_timeout_ms, fake_pyxrt=options.fake_pyxrt,
        cu_name=getattr(options, "cu", None), open_kernel=open_kernel)
    session.open()
    return session


def mode_probe(options) -> int:
    """B0: the xclbin declares this kernel, the card takes it, and it RUNS.

    There is no CSR read to do — pyxrt binds none — so B0 is the round trip
    instead: introspect the bitstream's own metadata, load it, open the CU
    EXCLUSIVE, execute a one-token null program, and read the header the fabric
    DMA'd back. Everything on that path is proven except the network itself.
    """
    session = open_session(options)
    try:
        report = session.device_stamp()
        null = session.null_run()
    finally:
        session.close()
    report.update({"null_run": null, "host": host_stamp(),
                   "fake_pyxrt": options.fake_pyxrt})
    report["proves"] = [
        "the xclbin's metadata declares "
        f"{KERNEL_NAME} with {report['xclbin_kernels'].get(KERNEL_NAME)} args",
        f"the card accepted the bitstream and gave a CU on device "
        f"{report['device_index']} with EXCLUSIVE access",
        "the six kernel arguments reached the CU and its AXI master read the "
        "program and stimulus buffers out of host memory",
        "the sequencer ran to its END token and the capture engine wrote its "
        "two-word header",
        "the AXI master wrote that header back into host memory, where this "
        "driver read it",
    ]
    report["does_not_prove"] = (
        "any spike count: no network is programmed by a null run, and the "
        "fabric's `err` bit at 0x4C stays unreadable from Python — a sequencer "
        "that refuses an opcode MID-run still drains a header, so B1's "
        "certificate, not a typed refusal, is what catches it")
    print(f"[probe] xclbin           : {report['xclbin']}")
    print(f"[probe] xclbin kernels   : {report['xclbin_kernels']}")
    print(f"[probe] memory banks     : "
          f"{[bank['tag'] for bank in report['memory_banks']]}")
    print(f"[probe] kernel           : {report['kernel']} (exclusive), "
          f"group_ids {report['group_ids']}")
    print(f"[probe] capture_capacity : {report['capture_capacity']} events")
    print(f"[probe] cores declared   : NC {report['cores_implied']}")
    print(f"[probe] capacity source  : {report['capacity']['provenance']}")
    print(f"[probe] null run         : header {null['header_words']} "
          f"(events_seen={null['events_seen']}, "
          f"device_cycles={null['device_cycles']})")
    path = write_result(options.results, "probe.json", report)
    print(f"[probe] wrote {path}")
    return 0


#: Where a board run puts each fixture's logical cores: once where every
#: archived run ran (``identity``), or at every rotation of the fabric
#: (``rotations``), so each fabric core certifies each fixture.
PLACEMENT_MODES = ("identity", "rotations")


def fixture_placements(fixture: Dict[str, Any], fabric_cores: int,
                       mode: str) -> List[Dict[int, int] | None]:
    """The placements one fixture runs at; ``None`` is the identity."""
    cores = int(fixture["run"]["cores"])
    if mode not in PLACEMENT_MODES:
        raise OdinDriverError(
            f"--placements {mode!r} is not one of {PLACEMENT_MODES}")
    if mode == "identity" or cores > int(fabric_cores):
        # A fixture wider than the fabric is refused by run_fixture by name.
        return [None]
    return list(rotation_placements(cores, fabric_cores))


def mode_run(options, capacity: KernelCapacity | None = None) -> int:
    """B1: every shipped fixture, one certificate line per run and per core."""
    paths = fixture_paths(options.fixtures, options.fixture)
    mode = str(getattr(options, "placements", "identity") or "identity")
    session = open_session(options, capacity)
    rows: List[Dict[str, Any]] = []
    try:
        for path in paths:
            fixture = load_fixture(path)
            name = fixture["name"]
            placements = fixture_placements(fixture, session.cores_implied, mode)
            for offset, placement in enumerate(placements):
                label = name if mode == "identity" else f"{name}@r{offset}"
                try:
                    result = session.run_fixture(fixture, placement)
                except OdinFixtureNeedsMoreCores as exc:
                    print(f"[b1] {name}: SKIP — {exc}")
                    rows.append({"fixture": name, "skipped": True,
                                 "reason": str(exc), "passed": None})
                    break
                result["run_label"] = label
                print(f"[b1] {label}: {result['certificate_line']}")
                for core in result["per_core"]:
                    print(f"[b1] {label}: core {core['logical_core']} on fabric "
                          f"core {core['fabric_core']}: {core['events']} events, "
                          f"{core['certificate_line']}")
                print(f"[b1] {label}: events={result['capture_events']} "
                      f"device_cycles={result['device_cycles']} "
                      f"programming={result['walls']['programming_s']:.3f}s "
                      f"execution={result['walls']['execution_s']:.3f}s")
                print(f"[b1] {label}: {stream_line(result['walls'])}")
                write_result(options.results, f"fixture_{label}.json", result)
                rows.append(result)
    finally:
        session.close()
    return _summarize(options, rows, "board")


def mode_reference(options) -> int:
    """The reference component: no device, re-derive the windows, attest them."""
    paths = fixture_paths(options.fixtures, options.fixture)
    rows: List[Dict[str, Any]] = []
    for path in paths:
        fixture = load_fixture(path)
        run = fixture["run"]
        counts = per_cycle_table(fixture["expected"]["per_cycle"])
        derived = window_counts(counts, run)
        declared = fixture["expected"]["window"]
        agree = derived == declared
        certificate = Certificate(
            declared, derived, samples=int(run["samples"]))
        print(f"[ref] {fixture['name']}: {certificate.line()}")
        rows.append({
            "fixture": fixture["name"],
            "self_hash": fixture["self_hash"],
            "provenance": fixture["provenance"],
            "window_counts": derived,
            "declared_matches_derived": agree,
            "certificate": certificate.as_dict(),
            "certificate_line": certificate.line(),
            "passed": bool(agree and certificate.passed),
        })
        write_result(options.results, f"reference_{fixture['name']}.json", rows[-1])
    return _summarize(options, rows, "reference")


def mode_compare(options) -> int:
    """The join: the board's windows against the reference component's, per fixture."""
    board = _read_summary(options.compare[0])
    reference = _read_summary(options.compare[1])
    by_name = {row["fixture"]: row for row in reference["rows"]}
    rows: List[Dict[str, Any]] = []
    for row in board["rows"]:
        name = row["fixture"]
        if row.get("skipped"):
            print(f"[join] {name}: SKIP on the board component — not compared")
            rows.append({"fixture": name, "skipped": True, "passed": None})
            continue
        peer = by_name.get(name)
        if peer is None:
            raise OdinDriverError(
                f"the reference component produced no row for {name!r}: the "
                f"two components ran different fixture sets and the join would "
                f"certify nothing")
        if peer["self_hash"] != row["self_hash"]:
            raise OdinDriverError(
                f"{name}: the two components read different fixture bytes "
                f"({row['self_hash']} vs {peer['self_hash']})")
        certificate = Certificate(
            peer["window_counts"], row["window_counts"],
            samples=len(row["window_counts"]))
        print(f"[join] {name}: {certificate.line()}")
        rows.append({
            "fixture": name, "self_hash": row["self_hash"],
            "board_host": board.get("host", {}).get("hostname"),
            "reference_host": reference.get("host", {}).get("hostname"),
            "certificate": certificate.as_dict(),
            "certificate_line": certificate.line(),
            "passed": certificate.passed,
        })
    return _summarize(options, rows, "join")


def selftest_capacity(fixtures: Sequence[Dict[str, Any]]) -> KernelCapacity:
    """A DECLARED geometry wide enough to exercise every shipped fixture."""
    cores = max([int(item["run"]["cores"]) for item in fixtures] + [1])
    return KernelCapacity(
        cores=cores,
        provenance=(
            f"selftest: a DECLARED NC={cores} geometry, wide enough for every "
            f"shipped fixture, against no bitstream at all. A BOARD run at a "
            f"smaller NC SKIPS the wider fixtures — that skip is exercised by "
            f"the 'nc1' refusal below"))


def mode_selftest(options) -> int:
    """The whole driver, green, with no hardware anywhere near it.

    STRUCTURE, NOT SILICON: the fake answers with the capture image the frozen
    fixture says the fabric produced, so what passes here is the DECODE, the
    fold, the window rule, the certificate and every typed refusal — never the
    board. Its value is that a red line on the cluster is then the card.
    """
    if not options.fake_pyxrt:
        raise OdinDriverError(
            "--selftest needs --fake-pyxrt host/fake_pyxrt_for_selftest.py: "
            "the point of the selftest is that no device is involved")
    print("[selftest] STRUCTURE, NOT SILICON — the fake pyxrt answers with the "
          "frozen fixture's own capture image, over the REAL pyxrt surface "
          "(github.com/Xilinx/XRT@2024.2 src/python/pybind11/src/pyxrt.cpp): "
          "no register read exists anywhere in it.")
    loaded = [load_fixture(path)
              for path in fixture_paths(options.fixtures, options.fixture)]
    capacity = selftest_capacity(loaded)
    print(f"[selftest] capacity      : {capacity.provenance}")
    exit_code = mode_run(options, capacity)
    if exit_code:
        return exit_code
    return _selftest_refusals(options, loaded)


#: Every typed refusal, and how a device can actually produce it. A capacity
#: fault is injected as a DECLARATION (nothing can be read back off a card); a
#: capture fault is injected into the fake's answer.
_REFUSAL_CASES = (
    ("no_storage", OdinDriverError, "silent empty one",
     KernelCapacity(cores=0, capture_events=0,
                    provenance="selftest: a package declaring no storage")),
    ("kernel_err", OdinFpgaKernelError, "NO-VERDICT sentinel", None),
    ("truncated_capture", OdinFpgaCaptureTruncated, "silent neurons", None),
    # The streaming-era length fault: the word count IS the bound, so a count
    # past the end of the buffer it names is what "too long" now means.
    ("short_buffer", OdinFpgaWordsExceedBuffer, "past the end of the buffer",
     None),
)


def _selftest_refusals(options, loaded: Sequence[Dict[str, Any]]) -> int:
    """Every typed refusal, driven through the fake, on the first fixture."""
    # The refusal cases run on the SMALLEST fixture, so a capacity fault is
    # refused for the storage it names and not for holding too few cores.
    fixture = min(loaded, key=lambda item: int(item["run"]["cores"]))
    multi_core = next(
        (item for item in loaded if int(item["run"]["cores"]) > 1), None)
    cases = [(fixture, fault, error, needle, capacity)
             for fault, error, needle, capacity in _REFUSAL_CASES]
    if multi_core is not None:
        cases.append((
            multi_core, "nc1", OdinFixtureNeedsMoreCores, "NC 1",
            KernelCapacity(
                cores=1,
                provenance="selftest: an NC=1 bitstream, the build default")))
    rows = []
    module = load_pyxrt(options.fake_pyxrt)
    for case_fixture, fault, expected, needle, capacity in cases:
        session = None
        module.arm(case_fixture, fault=fault)
        try:
            session = open_session(
                options,
                capacity if capacity is not None else selftest_capacity(loaded))
            session.run_fixture(case_fixture)
        except expected as exc:
            ok = needle in str(exc)
            print(f"[selftest] refusal {fault}: {type(exc).__name__} "
                  f"{'OK' if ok else 'WRONG MESSAGE'}")
            rows.append({"fault": fault, "raised": type(exc).__name__,
                         "message_ok": ok, "passed": ok})
            continue
        except OdinDriverError as exc:
            print(f"[selftest] refusal {fault}: WRONG TYPE {type(exc).__name__}")
            rows.append({"fault": fault, "raised": type(exc).__name__,
                         "message_ok": False, "passed": False})
            continue
        finally:
            module.arm(fault=None)
            if session is not None:
                session.close()
        print(f"[selftest] refusal {fault}: NO REFUSAL — the driver ran a "
              f"broken device to completion")
        rows.append({"fault": fault, "raised": None, "message_ok": False,
                     "passed": False})
    write_result(options.results, "selftest_refusals.json",
                 {"host": host_stamp(), "rows": rows})
    failed = [row for row in rows if not row["passed"]]
    print(f"[selftest] refusals: {len(rows) - len(failed)}/{len(rows)} typed "
          f"correctly")
    return 1 if failed else 0


def _summarize(options, rows: List[Dict[str, Any]], kind: str) -> int:
    ran = [row for row in rows if not row.get("skipped")]
    failed = [row for row in ran if not row.get("passed")]
    skipped = [row for row in rows if row.get("skipped")]
    certified_cores = sorted({
        int(core["fabric_core"]) for row in ran if row.get("passed")
        for core in row.get("per_core", ()) if core["certificate"]["passed"]})
    summary = {
        "kind": kind, "host": host_stamp(), "rows": rows,
        "fixtures_total": len(rows), "fixtures_run": len(ran),
        "fixtures_skipped": len(skipped), "fixtures_failed": len(failed),
        "fabric_cores_certified": certified_cores,
        "passed": not failed and bool(ran),
    }
    path = write_result(options.results, f"summary_{kind}.json", summary)
    verdict = "PASS" if summary["passed"] else "FAIL"
    print(f"[{kind}] {verdict}: {len(ran) - len(failed)}/{len(ran)} fixtures "
          f"certified, {len(skipped)} skipped")
    print(f"[{kind}] wrote {path}")
    if not ran:
        print(f"[{kind}] REFUSING: zero fixtures ran — a vacuous pass is not a "
              f"result")
        return 1
    return 1 if failed else 0


def _read_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.abspath(__file__))
    package = os.path.dirname(here)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", default=os.path.join(package, "fixtures"),
                        help="directory of frozen fixture JSONs")
    parser.add_argument("--fixture", action="append", default=[],
                        help="run only this fixture (repeatable)")
    parser.add_argument("--results", default=os.path.join(package, "results"),
                        help="where the result JSONs are written")
    parser.add_argument("--xclbin", default=os.environ.get("ODIN_XCLBIN", ""),
                        help="the bitstream to load (ODIN_XCLBIN)")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--capture-events", type=int,
                        default=DEFAULT_CAPTURE_EVENTS,
                        help="the HOST's capture ceiling; the session takes the "
                             "min of it and the fabric's declared one")
    parser.add_argument("--chip", default=SHIPPED_CHIP_CONFIG,
                        help=f"the FABRIC this bitstream was built from "
                             f"(default {SHIPPED_CHIP_CONFIG}); known: "
                             f"{', '.join(sorted(CHIP_CONFIGS))}")
    parser.add_argument("--declare-cores", type=int, default=0,
                        help=f"declare the NC the bitstream was built with "
                             f"(default: ${KERNEL_CORES_ENV}, else "
                             f"{SHIPPED_KERNEL_CORES}; no host can read it "
                             f"back, and the declaration is recorded in every "
                             f"refusal it causes)")
    parser.add_argument("--capture-ram-events", type=int, default=0,
                        help=f"declare a capture RAM other than the packaged "
                             f"{SHIPPED_CAPTURE_EVENTS} records")
    parser.add_argument("--placements", default="identity", choices=PLACEMENT_MODES,
                        help="identity: each fixture once, logical core k on "
                             "fabric core k; rotations: each fixture at every "
                             "rotation of the declared NC, so every fabric core "
                             "certifies it")
    parser.add_argument("--run-timeout-ms", type=int,
                        default=BLOCK_UNTIL_DONE_MS,
                        help="bound run.wait(); 0 blocks until the kernel is done")
    parser.add_argument("--fake-pyxrt", default=None,
                        help="import this module instead of pyxrt (no hardware)")
    parser.add_argument("--probe", action="store_true",
                        help="B0: introspect, load, open exclusive, run a null "
                             "program, read the header back")
    parser.add_argument("--selftest", action="store_true",
                        help="run every fixture and every refusal against a fake")
    parser.add_argument("--reference", action="store_true",
                        help="no device: re-derive the windows and attest them")
    parser.add_argument("--compare", nargs=2, metavar=("BOARD", "REFERENCE"),
                        help="join two summary JSONs and certify A == B")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    if options.selftest and not options.xclbin:
        options.xclbin = "/selftest/no-such.xclbin"
    try:
        if options.compare:
            return mode_compare(options)
        if options.reference:
            return mode_reference(options)
        if options.selftest:
            return mode_selftest(options)
        if options.probe:
            return mode_probe(options)
        return mode_run(options)
    except OdinDriverError as exc:
        print(f"REFUSING: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
