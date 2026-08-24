"""A fake ``pyxrt``: the board's CALL CONTRACT, with no board anywhere.

STRUCTURE, NOT SILICON. This module is the one shipped in
``tests/unit/chip_simulation/test_odin_fpga_transport.py`` — the contract
harness the audited ``XrtTransport`` call sequence was written against — with
two additions the package needs: it can be ARMED with a frozen fixture, so its
capture buffer answers with the very events that fixture's cosimulation
recorded, and it can be armed with a FAULT, so every typed refusal in
``odin_board_driver.py`` is exercised before the driver ever sees a card.

Nothing here proves anything about an Alveo. It proves that the driver decodes,
folds, certifies and REFUSES correctly, which is exactly the half of a board run
that a board should not be spent debugging.
"""

from __future__ import annotations

import struct
from types import SimpleNamespace
from typing import Any, Dict, List

CTRL_OFFSET = 0x00
ADDR_STATUS = 0x4C
ADDR_CAPTURE_CAPACITY = 0x54
ADDR_PROGRAM_CAPACITY = 0x5C
STATUS_ERR_BIT = 1 << 31

#: What the SHIPPED NC=1 bitstream reports: CAP_WORDS 16384 -> (16384-2)/4
#: records, PROG_WORDS = NC * 262144 (hw/fpga/kernel/odin_fpga_kernel_top.v).
SHIPPED_CAPTURE_EVENTS = (16384 - 2) // 4
PROG_WORDS_PER_CORE = 262144

_UNSET = object()

#: What this fake is currently pretending to be. A module-level state is the
#: honest model: one process talks to one board.
#:
#: ``cores`` is a property of the BITSTREAM, not of the fixture — a session
#: reads the capacity registers once at open() and every fixture afterwards
#: sees the same geometry, exactly as a card behaves. The default is wide
#: enough for every shipped fixture so the selftest exercises them all; the
#: "nc1" fault pins it to what the v1 packaging flow actually builds.
_STATE: Dict[str, Any] = {
    "fixture": None, "fault": None, "cores": 3, "log": [],
}


def arm(fixture: Any = None, fault: Any = _UNSET, cores: Any = _UNSET) -> None:
    """Point the fake at a frozen fixture, and optionally at a fault to inject.

    ``arm(fixture)`` leaves any armed fault and geometry alone — the driver
    calls it once per run, and only the selftest sets faults.
    """
    if fixture is not None:
        _STATE["fixture"] = fixture
    if fault is not _UNSET:
        _STATE["fault"] = fault
    if cores is not _UNSET:
        _STATE["cores"] = int(cores)


def call_log() -> List[tuple]:
    """Every call the fake received, in order — what a contract test asserts on."""
    return list(_STATE["log"])


def _fixture() -> Dict[str, Any]:
    fixture = _STATE["fixture"]
    if fixture is None:
        raise RuntimeError(
            "the fake pyxrt was never armed with a fixture: call arm(fixture) "
            "before opening a session, or the capture buffer would answer with "
            "zeros and a green certificate would mean nothing")
    return fixture


def _capacities() -> tuple:
    """The two read-only capacity registers this fake's bitstream reports."""
    fault = _STATE["fault"]
    if fault == "no_storage":
        return 0, 0
    program = int(_STATE["cores"]) * PROG_WORDS_PER_CORE
    if fault == "nc1":
        # What the v1 packaging flow actually builds, so the driver's
        # "this xclbin holds fewer cores than the fixture programs" skip is
        # exercised without a card.
        program = PROG_WORDS_PER_CORE
    if fault == "tiny_program_ram":
        program = 64
    return SHIPPED_CAPTURE_EVENTS, program


def _capture_image(capacity: int) -> List[int]:
    """The capture buffer the fabric would have written for the armed fixture.

    One record per spike the frozen per-cycle expectation carries, tagged the
    way the sequencer tags a cycle: ``first_tag + sample * cycles + cycle``.
    """
    fixture = _fixture()
    run = fixture["run"]
    first_tag = int(run["first_tag"])
    per_sample = int(run["cycles_per_sample"])
    words: List[int] = []
    events = 0
    for sample, cycle, core, neuron, count in fixture["expected"]["per_cycle"]:
        tag = first_tag + int(sample) * per_sample + int(cycle)
        for _ in range(int(count)):
            words.extend((tag, int(cycle), int(core), int(neuron)))
            events += 1
    if _STATE["fault"] == "extra_spike" and words:
        # A device that emitted ONE more spike than the frozen evidence says:
        # exactly the divergence a certificate exists to catch.
        words.extend(words[:4])
        events += 1
    if _STATE["fault"] == "truncated_capture":
        events = int(capacity)
    del run
    return [events, int(fixture["expected"]["device_cycles_cosim"])] + words


class _FakeBo:
    def __init__(self, log, size, group):
        self.log = log
        self.size = int(size)
        self.group = int(group)
        self.data = bytearray(int(size))

    def write(self, payload, offset):
        self.log.append(("write", self.group, len(payload), int(offset)))
        self.data[offset:offset + len(payload)] = payload

    def sync(self, direction, size, offset):
        self.log.append(("sync", self.group, direction, int(size), int(offset)))

    def read(self, size, offset):
        self.log.append(("read", self.group, int(size), int(offset)))
        return bytes(self.data[offset:offset + size])


class _FakeRun:
    def __init__(self, log):
        self.log = log

    def wait(self):
        self.log.append(("wait",))


class _FakeKernel:
    exclusive = "exclusive"

    def __init__(self, log):
        self.log = log

    def group_id(self, arg):
        return int(arg)

    def read_register(self, offset):
        self.log.append(("read_register", int(offset)))
        capture_capacity, program_capacity = _capacities()
        status = STATUS_ERR_BIT | 17 if _STATE["fault"] == "kernel_err" else 0
        return int({
            CTRL_OFFSET: 0b0110,
            ADDR_STATUS: status,
            ADDR_CAPTURE_CAPACITY: capture_capacity,
            ADDR_PROGRAM_CAPACITY: program_capacity,
        }.get(int(offset), 0))

    def __call__(self, *args):
        self.log.append(("start", tuple(
            a.group if isinstance(a, _FakeBo) else a for a in args)))
        capture = args[2]
        capacity = int(args[5])
        words = _capture_image(capacity)
        packed = struct.pack(f"<{len(words)}I", *words)
        capture.data[0:len(packed)] = packed
        return _FakeRun(self.log)


class _FakeDevice:
    def __init__(self, log, index):
        self.log = log
        self.index = int(index)

    def load_xclbin(self, xclbin):
        self.log.append(("load_xclbin", xclbin.path))
        return "uuid"


def device(index):
    # Opening a device starts a new session, and a session starts a new log.
    _STATE["log"] = [("device", int(index))]
    return _FakeDevice(_STATE["log"], index)


def xclbin(path):
    return SimpleNamespace(path=path)


def kernel(_device, _uuid, name, mode):
    _STATE["log"].append(("kernel", name, mode))
    return _FakeKernel(_STATE["log"])


kernel.exclusive = _FakeKernel.exclusive


def bo(_device, size, kind, group):
    return _FakeBo(_STATE["log"], size, group)


bo.normal = "normal"

xclBOSyncDirection = SimpleNamespace(
    XCL_BO_SYNC_BO_TO_DEVICE="to", XCL_BO_SYNC_BO_FROM_DEVICE="from")
