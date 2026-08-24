#!/usr/bin/env python3
"""The thin ODIN board driver: one frozen fixture in, one certified verdict out.

STANDALONE BY CONSTRUCTION. This file is the ONLY host code the HACC package
ships, it imports nothing but the standard library plus ``pyxrt`` (which ships
with the Xilinx Runtime, never with pip), and it never reads the mimarsinan
repository. That is what makes it uploadable: the board node needs the zip and
an XRT, nothing else.

WHAT IT MIRRORS. The XRT call sequence here is the audited one from
``src/mimarsinan/chip_simulation/odin_fpga/xrt_transport.py`` — open the device,
load the xclbin, resolve the kernel with EXCLUSIVE access, read the two
read-only capacity registers the bitstream was compiled with, allocate one
buffer object per kernel argument in that argument's memory bank, DMA the
program, DMA the stimulus, start, wait, read the status register BEFORE
touching the capture, sync the capture back, decode. Every refusal it raises is
that module's refusal, by name.

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
import sys
import time
import zlib
from typing import Any, Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------
# The register map, frozen from the host-side SSOT
# (src/mimarsinan/chip_simulation/odin_fpga/kernel_registers.py). Every fixture
# also CARRIES this table under "kernel", and the driver refuses a fixture whose
# copy disagrees with this one: two tables that drifted would drive a kernel
# nobody built.
# --------------------------------------------------------------------------

KERNEL_NAME = "odin_fpga_kernel_top"
ARG_PROGRAM, ARG_STIMULUS, ARG_CAPTURE = 0, 1, 2
CTRL_OFFSET = 0x00
ADDR_STATUS = 0x4C
ADDR_CAPTURE_CAPACITY = 0x54
ADDR_PROGRAM_CAPACITY = 0x5C
STATUS_ERR_BIT = 1 << 31
STATUS_EVENTS_MASK = STATUS_ERR_BIT - 1
CAPTURE_HEADER_WORDS = 2
CAPTURE_RECORD_WORDS = 4
WORD_BYTES = 4
DEFAULT_CAPTURE_EVENTS = 1 << 20

#: PROG_WORDS = NC * 262144 (hw/fpga/kernel/odin_fpga_kernel_top.v line 47), so
#: the program capacity the bitstream reports names how many ODIN cores it
#: instantiates. The v1 packaging flow builds NC = 1 only (build_xclbn.sh
#: refuses more), which is why a multi-core fixture is SKIPPED rather than run.
PROG_WORDS_PER_CORE = 262144

BACKEND = "odin_fpga"
BACKEND_CLASS = "exact"
SCHEMA = "odin_hacc_fixture/1"


class OdinDriverError(RuntimeError):
    """The driver refused; the message names what and how to fix it."""


class OdinFpgaDependencyError(OdinDriverError):
    """``pyxrt`` is not importable in this environment."""


class OdinFpgaKernelError(OdinDriverError):
    """The kernel raised `err`: it refused the program it was given."""


class OdinFpgaCaptureTruncated(OdinDriverError):
    """The device saw at least as many events as the capture can hold."""


class OdinFpgaProgramTooLarge(OdinDriverError):
    """The token stream does not fit the fabric's program RAM."""


class OdinFixtureNeedsMoreCores(OdinDriverError):
    """This xclbin instantiates fewer ODIN cores than the fixture programs."""


class OdinFixtureCorrupt(OdinDriverError):
    """A fixture's self-hash or payload hash does not match its bytes."""


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
    expected = {
        "name": KERNEL_NAME, "arg_program": ARG_PROGRAM,
        "arg_stimulus": ARG_STIMULUS, "arg_capture": ARG_CAPTURE,
        "ctrl_offset": CTRL_OFFSET, "addr_status": ADDR_STATUS,
        "addr_capture_capacity": ADDR_CAPTURE_CAPACITY,
        "addr_program_capacity": ADDR_PROGRAM_CAPACITY,
        "status_err_bit": STATUS_ERR_BIT,
        "capture_header_words": CAPTURE_HEADER_WORDS,
        "capture_record_words": CAPTURE_RECORD_WORDS,
        "word_bytes": WORD_BYTES,
    }
    if table != expected:
        raise OdinFixtureCorrupt(
            f"{path}: the fixture's register table disagrees with this "
            f"driver's. Fixture={table}, driver={expected}. The package was "
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
# The register/capture arithmetic, mirrored from kernel_registers.py
# --------------------------------------------------------------------------


def capture_buffer_bytes(capacity_events: int) -> int:
    return WORD_BYTES * (
        CAPTURE_HEADER_WORDS + CAPTURE_RECORD_WORDS * int(capacity_events))


def decode_status(word: int) -> Tuple[bool, int]:
    value = int(word)
    return bool(value & STATUS_ERR_BIT), value & STATUS_EVENTS_MASK


def require_no_kernel_error(status: int, *, transport: str) -> None:
    failed, events_seen = decode_status(status)
    if failed:
        raise OdinFpgaKernelError(
            f"{transport}: the kernel raised err after seeing {events_seen} "
            f"event(s) — the fabric sequencer REFUSED an opcode it does not "
            f"implement (SHADOW/PROG have no fabric path), an AER handshake "
            f"timed out, or the DMA could not fit the payload. The counts of "
            f"this run are not a network's answer and are not being decoded")


def stimulus_base_word(program_words: int) -> int:
    """The stimulus OVERWRITES the programming payload's END terminator."""
    return max(0, int(program_words) - 1)


def require_program_fits(
    program_words: int, stimulus_words: int, capacity_words: int,
    *, transport: str,
) -> None:
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


def capture_words(raw: Any) -> List[int]:
    if isinstance(raw, (bytes, bytearray, memoryview)):
        buffer = bytes(raw)
        count = len(buffer) // WORD_BYTES
        return list(int.from_bytes(
            buffer[index * WORD_BYTES:(index + 1) * WORD_BYTES], "little")
            for index in range(count))
    return [int(value) for value in raw]


def decode_capture(words: Sequence[int], capacity: int) -> Tuple[List[Tuple[int, int, int, int]], int]:
    """``(events, device_cycles)`` from the capture buffer's word image."""
    if len(words) < CAPTURE_HEADER_WORDS:
        raise OdinDriverError(
            "the capture buffer came back shorter than its own two-word "
            "header; the kernel never wrote a verdict and the run has no counts")
    written = int(words[0])
    device_cycles = int(words[1])
    if written >= int(capacity):
        raise OdinFpgaCaptureTruncated(
            f"the kernel SAW {written} events against a capacity of {capacity} "
            f"(min of this session's declaration and the fabric's own capture "
            f"RAM, read from 0x54): the capture is at or over its limit and the "
            f"events past it would read as silent neurons. Rebuild the kernel "
            f"with a larger CAP_WORDS (hw/fpga/kernel/odin_fpga_kernel_top.v) "
            f"or run fewer samples per pass; the counts of a truncated run are "
            f"not a result.")
    events: List[Tuple[int, int, int, int]] = []
    for index in range(written):
        base = CAPTURE_HEADER_WORDS + index * CAPTURE_RECORD_WORDS
        tag, cycle, core, neuron = words[base:base + CAPTURE_RECORD_WORDS]
        events.append((int(tag), int(cycle), int(core), int(neuron)))
    return events, device_cycles


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


class BoardSession:
    """One XRT session against an ODIN kernel on a U55C XDMA shell."""

    name = "xrt"

    def __init__(self, *, xclbin_path: str, device_index: int = 0,
                 capture_events: int = DEFAULT_CAPTURE_EVENTS,
                 fake_pyxrt: str | None = None) -> None:
        if not xclbin_path:
            raise OdinDriverError(
                "the board needs an xclbin: the bitstream is the device's "
                "program and there is no default one (--xclbin)")
        self.xclbin_path = str(xclbin_path)
        self.device_index = int(device_index)
        self.capture_events = int(capture_events)
        self.fake_pyxrt = fake_pyxrt
        self._xrt: Any = None
        self._device: Any = None
        self._kernel: Any = None
        self._capture_capacity = 0
        self._program_capacity = 0

    def open(self) -> None:
        self._xrt = load_pyxrt(self.fake_pyxrt)
        self._device = self._xrt.device(self.device_index)
        uuid = self._device.load_xclbin(self._xrt.xclbin(self.xclbin_path))
        self._kernel = self._xrt.kernel(
            self._device, uuid, KERNEL_NAME,
            # Exclusive access: register reads (0x4C/0x54/0x5C) require it and
            # the deployment owns the board for the whole reservation.
            self._xrt.kernel.exclusive,
        )
        fabric_events = int(self._kernel.read_register(ADDR_CAPTURE_CAPACITY))
        self._program_capacity = int(
            self._kernel.read_register(ADDR_PROGRAM_CAPACITY))
        self._capture_capacity = min(self.capture_events, fabric_events)
        if self._capture_capacity <= 0 or self._program_capacity <= 0:
            raise OdinDriverError(
                f"{self.name}: the kernel reports a capture capacity of "
                f"{fabric_events} events and a program capacity of "
                f"{self._program_capacity} words — an xclbin built with no "
                f"storage cannot run anything, and reading it as zero would "
                f"turn every run into a silent empty one")

    def close(self) -> None:
        self._kernel = None
        self._device = None
        self._capture_capacity = 0
        self._program_capacity = 0

    @property
    def capture_capacity(self) -> int:
        return self._capture_capacity

    @property
    def program_capacity(self) -> int:
        return self._program_capacity

    @property
    def cores_implied(self) -> int:
        """How many ODIN cores this bitstream holds, per PROG_WORDS = NC*262144.

        Rounded UP, so a kernel deliberately built with a smaller PROG_WORDS is
        read as the cores it has rather than as none: under-reporting here would
        refuse a legitimate bitstream, while the payload that does not fit is
        caught by ``require_program_fits`` with the capacity the kernel named.
        """
        return -(-self._program_capacity // PROG_WORDS_PER_CORE)

    def _require_session(self, what: str) -> Any:
        if self._kernel is None or self._xrt is None:
            raise OdinDriverError(
                f"{self.name}: {what} before a live session — open() loads the "
                f"xclbin and resolves the kernel, and nothing can be written "
                f"to a device that was never opened")
        return self._xrt

    def _buffer(self, nbytes: int, arg: int) -> Any:
        xrt = self._require_session("buffer allocation")
        return xrt.bo(self._device, int(nbytes), xrt.bo.normal,
                      self._kernel.group_id(int(arg)))

    def _to_device(self, handle: Any, payload: bytes) -> None:
        xrt = self._require_session("host-to-device DMA")
        handle.write(payload, 0)
        handle.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(payload), 0)

    def status_snapshot(self) -> Dict[str, Any]:
        """The CSR read B0 is: what this bitstream says about itself."""
        self._require_session("register read")
        return {
            "xclbin": self.xclbin_path,
            "kernel": KERNEL_NAME,
            "device_index": self.device_index,
            "capture_capacity": self._capture_capacity,
            "program_capacity": self._program_capacity,
            "cores_implied": self.cores_implied,
            "status": int(self._kernel.read_register(ADDR_STATUS)),
            "ctrl": int(self._kernel.read_register(CTRL_OFFSET)),
        }

    def run_fixture(self, fixture: Dict[str, Any]) -> Dict[str, Any]:
        """Program, stimulate, wait, read the status, decode — in that order."""
        self._require_session("run_fixture()")
        # A FAKE is handed the fixture so it can answer with the capture image
        # the frozen evidence says the fabric produced; real pyxrt has no arm().
        arm = getattr(self._xrt, "arm", None)
        if arm is not None:
            arm(fixture)
        run = fixture["run"]
        needed_cores = int(run["cores"])
        if needed_cores > self.cores_implied:
            raise OdinFixtureNeedsMoreCores(
                f"{fixture['name']}: the fixture programs {needed_cores} ODIN "
                f"core(s) but this xclbin reports a program capacity of "
                f"{self._program_capacity} words = NC {self.cores_implied} "
                f"(PROG_WORDS = NC * {PROG_WORDS_PER_CORE}). The v1 packaging "
                f"flow builds NC=1 only (scripts/hacc/build_xclbn.sh refuses "
                f"more); run the single-core fixtures on this bitstream, and "
                f"widen NC before asking for this one")

        program = decode_payload(fixture["program"], what=f"{fixture['name']}/program")
        stimulus = decode_payload(fixture["stimulus"], what=f"{fixture['name']}/stimulus")
        program_words = len(program) // WORD_BYTES
        stimulus_words = len(stimulus) // WORD_BYTES
        require_program_fits(
            program_words, stimulus_words, self._program_capacity,
            transport=self.name)

        started = time.monotonic()
        program_bo = self._buffer(len(program), ARG_PROGRAM)
        self._to_device(program_bo, program)
        programming_s = time.monotonic() - started

        started = time.monotonic()
        stimulus_bo = self._buffer(len(stimulus), ARG_STIMULUS)
        self._to_device(stimulus_bo, stimulus)
        capture_bytes = capture_buffer_bytes(self._capture_capacity)
        capture_bo = self._buffer(capture_bytes, ARG_CAPTURE)
        handle = self._kernel(
            program_bo, stimulus_bo, capture_bo,
            program_words, stimulus_words, self._capture_capacity,
        )
        handle.wait()
        status = self._kernel.read_register(ADDR_STATUS)
        require_no_kernel_error(status, transport=self.name)
        capture_bo.sync(
            self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
            capture_bytes, 0)
        words = capture_words(capture_bo.read(capture_bytes, 0))
        execution_s = time.monotonic() - started

        events, device_cycles = decode_capture(words, self._capture_capacity)
        counts = fold_events(events, run)
        measured = window_counts(counts, run)
        expected = fixture["expected"]["window"]
        certificate = Certificate(
            expected, measured, samples=int(run["samples"]))
        return {
            "fixture": fixture["name"],
            "self_hash": fixture["self_hash"],
            "transport": self.name,
            "device": {
                "xclbin": self.xclbin_path,
                "kernel": KERNEL_NAME,
                "device_index": self.device_index,
                "capture_capacity": self._capture_capacity,
                "program_capacity": self._program_capacity,
                "cores_implied": self.cores_implied,
                "status": int(status),
                "ctrl": int(self._kernel.read_register(CTRL_OFFSET)),
            },
            "program_bytes": len(program),
            "program_words": program_words,
            "stimulus_words": stimulus_words,
            "capture_events": len(events),
            "device_cycles": device_cycles,
            "walls": {
                "programming_s": programming_s,
                "execution_s": execution_s,
            },
            "window_counts": measured,
            "certificate": certificate.as_dict(),
            "certificate_line": certificate.line(),
            "passed": certificate.passed,
        }


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


def mode_probe(options) -> int:
    """B0: load the xclbin, resolve the kernel, read the capacity/status CSRs."""
    session = BoardSession(
        xclbin_path=options.xclbin, device_index=options.device_index,
        capture_events=options.capture_events, fake_pyxrt=options.fake_pyxrt)
    session.open()
    try:
        snapshot = session.status_snapshot()
    finally:
        session.close()
    err, seen = decode_status(snapshot["status"])
    snapshot.update({"err": err, "events_seen": seen, "host": host_stamp(),
                     "fake_pyxrt": options.fake_pyxrt})
    print(f"[probe] xclbin           : {snapshot['xclbin']}")
    print(f"[probe] kernel           : {snapshot['kernel']} (exclusive)")
    print(f"[probe] capture_capacity : {snapshot['capture_capacity']} events")
    print(f"[probe] program_capacity : {snapshot['program_capacity']} words "
          f"(=> NC {snapshot['cores_implied']})")
    print(f"[probe] status 0x4C      : {snapshot['status']} (err={err}, "
          f"events_seen={seen})")
    path = write_result(options.results, "probe.json", snapshot)
    print(f"[probe] wrote {path}")
    if err:
        print("[probe] REFUSING: the kernel's err bit is set before any run.")
        return 1
    return 0


def mode_run(options) -> int:
    """B1: every shipped fixture, one certificate line each, one summary."""
    paths = fixture_paths(options.fixtures, options.fixture)
    session = BoardSession(
        xclbin_path=options.xclbin, device_index=options.device_index,
        capture_events=options.capture_events, fake_pyxrt=options.fake_pyxrt)
    session.open()
    rows: List[Dict[str, Any]] = []
    try:
        for path in paths:
            fixture = load_fixture(path)
            name = fixture["name"]
            try:
                result = session.run_fixture(fixture)
            except OdinFixtureNeedsMoreCores as exc:
                print(f"[b1] {name}: SKIP — {exc}")
                rows.append({"fixture": name, "skipped": True,
                             "reason": str(exc), "passed": None})
                continue
            print(f"[b1] {name}: {result['certificate_line']}")
            print(f"[b1] {name}: events={result['capture_events']} "
                  f"device_cycles={result['device_cycles']} "
                  f"programming={result['walls']['programming_s']:.3f}s "
                  f"execution={result['walls']['execution_s']:.3f}s")
            write_result(options.results, f"fixture_{name}.json", result)
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
          "frozen fixture's own capture image.")
    exit_code = mode_run(options)
    if exit_code:
        return exit_code
    return _selftest_refusals(options)


def _selftest_refusals(options) -> int:
    """Every typed refusal, driven through the fake, on the first fixture."""
    loaded = [load_fixture(path)
              for path in fixture_paths(options.fixtures, options.fixture)]
    # The refusal cases run on the SMALLEST fixture: a shrunken program RAM must
    # be refused for not fitting the payload, not for holding too few cores.
    fixture = min(loaded, key=lambda item: int(item["run"]["cores"]))
    multi_core = next(
        (item for item in loaded if int(item["run"]["cores"]) > 1), None)
    cases = [
        (fixture, "no_storage", OdinDriverError, "silent empty one"),
        (fixture, "kernel_err", OdinFpgaKernelError, "raised err"),
        (fixture, "truncated_capture", OdinFpgaCaptureTruncated,
         "silent neurons"),
        (fixture, "tiny_program_ram", OdinFpgaProgramTooLarge,
         "program RAM holds"),
    ]
    if multi_core is not None:
        cases.append(
            (multi_core, "nc1", OdinFixtureNeedsMoreCores, "NC 1"))
    rows = []
    module = load_pyxrt(options.fake_pyxrt)
    for case_fixture, fault, expected, needle in cases:
        session = BoardSession(
            xclbin_path=options.xclbin, device_index=options.device_index,
            capture_events=options.capture_events,
            fake_pyxrt=options.fake_pyxrt)
        module.arm(case_fixture, fault=fault)
        try:
            session.open()
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
    summary = {
        "kind": kind, "host": host_stamp(), "rows": rows,
        "fixtures_total": len(rows), "fixtures_run": len(ran),
        "fixtures_skipped": len(skipped), "fixtures_failed": len(failed),
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
                             "min of it and the fabric's own")
    parser.add_argument("--fake-pyxrt", default=None,
                        help="import this module instead of pyxrt (no hardware)")
    parser.add_argument("--probe", action="store_true",
                        help="B0: load, resolve, read the CSRs, run nothing")
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
