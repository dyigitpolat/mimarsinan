"""A fake ``pyxrt``: EXACTLY the real binding's surface, and never more than it.

PROVENANCE, and why it is stated here. Every class, method, enum and module
attribute below is mirrored from

    https://github.com/Xilinx/XRT/blob/2024.2/src/python/pybind11/src/pyxrt.cpp

(branch ``2024.2``; XRT 2.18.x is what HACC@NUS ships). The previous fake
carried a ``kernel.read_register`` that the real binding does not have, the
whole driver was written against it, and on 2026-08-25 the shipped driver
crashed on a U250 at its first CSR touch with

    AttributeError: 'pyxrt.kernel' object has no attribute 'read_register'

A fake with MORE API than reality is how that bug was born, so this module is a
strict SUBSET of the fetched surface and
``tests/unit/chip_simulation/test_odin_pyxrt_surface.py`` fails if a single name
outside it ever appears. The only names allowed to be ours are the ones listed
in ``FAKE_ONLY``, declared as data so the check can subtract exactly them.

STRUCTURE, NOT SILICON. Armed with a frozen fixture, this fake's capture buffer
answers with the very events that fixture's cosimulation recorded; armed with a
fault, it answers the way a REFUSING kernel answers — which, with no register
access anywhere, means leaving the host's no-verdict sentinel in the capture
header. Nothing here proves anything about an Alveo.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, List, Optional

#: The names in this module that the real pyxrt does not have: the harness's
#: two arming hooks and the capture-header constants this fake must mirror.
#: They are declared as data so the surface-conformance test can subtract
#: exactly them and nothing else.
FAKE_ONLY = (
    "FAKE_ONLY", "arm", "call_log",
    "CAPTURE_HEADER_WORDS", "CAPTURE_RECORD_WORDS", "CAPTURE_NO_VERDICT",
)

#: The capture header layout and the no-verdict sentinel, mirrored from
#: ``odin_board_driver.py`` (whose own source is ``odin_fpga_kernel.v``).
CAPTURE_HEADER_WORDS = 2
CAPTURE_RECORD_WORDS = 4
CAPTURE_NO_VERDICT = 0xFFFFFFFF

_UNSET = object()

#: What this fake is currently pretending to be. Module-level state is the
#: honest model: one process talks to one board.
_STATE: Dict[str, Any] = {
    "fixture": None, "fault": None, "capture_words": None, "log": [],
    "deployment": None,
}


def arm(
    fixture: Any = None, fault: Any = _UNSET, capture_words: Any = _UNSET,
    deployment: Any = _UNSET,
) -> None:
    """Point the fake at a frozen fixture, a raw capture image, or a fault.

    ``arm(fixture)`` leaves any armed fault alone — the driver calls it once per
    run, and only a test sets faults. ``deployment=`` arms the MULTI-RUN replay
    a host-mediated deployment needs: the executor programs one core and then
    sends a different stimulus per sample, so one frozen answer cannot serve
    them. The replay is keyed by the sha256 of the stimulus, which means a host
    that builds the wrong stimulus is answered with NO VERDICT rather than with
    some other pass's counts.
    """
    if fixture is not None:
        _STATE["fixture"] = fixture
    if fault is not _UNSET:
        _STATE["fault"] = fault
    if capture_words is not _UNSET:
        _STATE["capture_words"] = capture_words
    if deployment is not _UNSET:
        _STATE["deployment"] = (
            None if deployment is None else {
                str(run["stimulus_sha256"]): run
                for run in deployment["runs"]
            })


def call_log() -> List[tuple]:
    """Every call the fake received, in order — what a contract test asserts on."""
    return list(_STATE["log"])


def _log(*entry: Any) -> None:
    _STATE["log"].append(tuple(entry))


def _deployment_image(stimulus: bytes, capacity: int) -> Optional[List[int]]:
    """The answer this pass's stimulus earns, or None when nothing matches it."""
    replay = _STATE["deployment"]
    run = replay.get(hashlib.sha256(stimulus).hexdigest()) if replay else None
    if run is None:
        # No frozen answer for these bytes: the fabric never saw this program,
        # so the honest fake leaves the host's sentinel in the header.
        return None
    events = [list(record) for record in run["events"]]
    if _STATE["fault"] == "extra_spike" and events:
        events.append(list(events[0]))
    if _STATE["fault"] == "truncated_capture":
        return [int(capacity), int(run["device_cycles"])] + [
            int(word) for record in events for word in record]
    words = [len(events), int(run["device_cycles"])]
    for record in events:
        words.extend(int(word) for word in record)
    return words


def _capture_image(capacity: int, stimulus: bytes = b"") -> Optional[List[int]]:
    """The word image the fabric would have DMA'd back, or None for no verdict.

    One record per spike the frozen per-cycle expectation carries, tagged the
    way the sequencer tags a cycle: ``first_tag + sample * cycles + cycle``.
    """
    fault = _STATE["fault"]
    if fault == "kernel_err":
        # A kernel that raised err at ap_start never drains, so the host's
        # sentinel survives in the capture header. That is the ONLY way `err`
        # reaches a host with no register access.
        return None
    if _STATE["deployment"] is not None:
        return _deployment_image(stimulus, capacity)
    armed = _STATE["capture_words"]
    if armed is not None:
        return [int(word) for word in armed]
    fixture = _STATE["fixture"]
    if fixture is None:
        # The null-program answer B0 asks for: a verdict, and no events.
        return [0, 1]
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
    if fault == "extra_spike" and words:
        # A device that emitted ONE more spike than the frozen evidence says:
        # exactly the divergence a certificate exists to catch.
        words.extend(words[:CAPTURE_RECORD_WORDS])
        events += 1
    if fault == "truncated_capture":
        events = int(capacity)
    return [events, int(fixture["expected"]["device_cycles_cosim"])] + words


# ---------------------------------------------------------------------------
# The mirrored surface. Every name below exists in pyxrt.cpp; see FAKE_ONLY.
# ---------------------------------------------------------------------------


class xclBOSyncDirection:
    XCL_BO_SYNC_BO_TO_DEVICE = "XCL_BO_SYNC_BO_TO_DEVICE"
    XCL_BO_SYNC_BO_FROM_DEVICE = "XCL_BO_SYNC_BO_FROM_DEVICE"


class ert_cmd_state:
    ERT_CMD_STATE_COMPLETED = "ERT_CMD_STATE_COMPLETED"
    ERT_CMD_STATE_ERROR = "ERT_CMD_STATE_ERROR"
    ERT_CMD_STATE_TIMEOUT = "ERT_CMD_STATE_TIMEOUT"


class uuid:
    def __init__(self, text: str) -> None:
        self._text = str(text)

    def to_string(self) -> str:
        return self._text


class run:
    def __init__(self, kernel: Any) -> None:
        self._kernel = kernel
        self._state = (
            ert_cmd_state.ERT_CMD_STATE_ERROR
            if _STATE["fault"] == "not_completed"
            else ert_cmd_state.ERT_CMD_STATE_COMPLETED)

    def wait(self, timeout_ms: int = 0) -> str:
        _log("wait", int(timeout_ms))
        return self._state

    def state(self) -> str:
        return self._state


class kernel:
    class cu_access_mode:
        exclusive = "exclusive"
        shared = "shared"
        none = "none"

    # pybind11's `export_values()` puts the nested enum's members on the class
    # as well, so `pyxrt.kernel.exclusive` resolves on the real binding too.
    exclusive = cu_access_mode.exclusive
    shared = cu_access_mode.shared
    none = cu_access_mode.none

    def __init__(self, device: Any, xclbin_uuid: Any, name: str,
                 mode: str = cu_access_mode.shared) -> None:
        _log("kernel", str(name), str(mode))
        self._name = str(name)

    def group_id(self, argno: int) -> int:
        _log("group_id", int(argno))
        return int(argno)

    def __call__(self, *args: Any) -> run:
        _log("start", tuple(
            item._group if isinstance(item, bo) else item for item in args))
        capture = args[2]
        capacity = int(args[5])
        # Exactly the words the host DECLARED, not the whole buffer object: a
        # deployment reuses one oversized stimulus buffer across samples.
        stimulus = bytes(args[1]._data[:int(args[4]) * 4])
        words = _capture_image(capacity, stimulus)
        if words is not None:
            capture._store(struct.pack(f"<{len(words)}I", *words), 0)
        return run(self)


class bo:
    class flags:
        normal = "normal"
        cacheable = "cacheable"
        device_only = "device_only"
        host_only = "host_only"
        p2p = "p2p"
        svm = "svm"

    normal = flags.normal
    cacheable = flags.cacheable
    device_only = flags.device_only
    host_only = flags.host_only
    p2p = flags.p2p
    svm = flags.svm

    def __init__(self, device: Any, nbytes: int, kind: str, group: int) -> None:
        self._size = int(nbytes)
        self._group = int(group)
        self._data = bytearray(int(nbytes))
        _log("bo", self._size, self._group)

    def _store(self, payload: bytes, offset: int) -> None:
        self._data[offset:offset + len(payload)] = payload

    def write(self, payload: Any, seek: int = 0) -> None:
        buffer = bytes(memoryview(payload))
        _log("write", self._group, len(buffer), int(seek))
        self._store(buffer, int(seek))

    def read(self, size: int, skip: int = 0) -> bytes:
        _log("read", self._group, int(size), int(skip))
        return bytes(self._data[int(skip):int(skip) + int(size)])

    def sync(self, direction: str, size: Optional[int] = None,
             offset: int = 0) -> None:
        _log("sync", self._group, direction,
             self._size if size is None else int(size), int(offset))

    def size(self) -> int:
        return self._size


class xclbin:
    class xclbinkernel:
        def __init__(self, name: str, num_args: int) -> None:
            self._name = str(name)
            self._num_args = int(num_args)

        def get_name(self) -> str:
            return self._name

        def get_num_args(self) -> int:
            return self._num_args

    class xclbinmem:
        def __init__(self, index: int, tag: str, size_kb: int, used: bool) -> None:
            self._index = int(index)
            self._tag = str(tag)
            self._size_kb = int(size_kb)
            self._used = bool(used)

        def get_index(self) -> int:
            return self._index

        def get_tag(self) -> str:
            return self._tag

        def get_base_address(self) -> int:
            return self._index << 32

        def get_size_kb(self) -> int:
            return self._size_kb

        def get_used(self) -> bool:
            return self._used

    #: What the shipped NC=1 xclbin declares about itself: our kernel, its six
    #: frozen arguments, and the one DDR bank odin_u250.cfg maps gmem onto.
    _KERNEL_NAME = "odin_fpga_kernel_top"
    _KERNEL_ARGS = 6

    def __init__(self, path: str) -> None:
        # Opening the xclbin starts a new session, and a session starts a new log.
        _STATE["log"] = [("xclbin", str(path))]
        self._path = str(path)

    def get_kernels(self) -> List["xclbin.xclbinkernel"]:
        _log("get_kernels")
        name = ("some_other_kernel" if _STATE["fault"] == "wrong_kernel"
                else self._KERNEL_NAME)
        return [xclbin.xclbinkernel(name, self._KERNEL_ARGS)]

    def get_mems(self) -> List["xclbin.xclbinmem"]:
        _log("get_mems")
        return [xclbin.xclbinmem(0, "DDR[0]", 16 * 1024 * 1024, True)]

    def get_uuid(self) -> uuid:
        return uuid(f"uuid-of-{self._path}")


class device:
    def __init__(self, index: int = 0) -> None:
        _log("device", int(index))
        self._index = int(index)

    def load_xclbin(self, image: Any) -> uuid:
        _log("load_xclbin", image._path)
        return image.get_uuid()

    def get_xclbin_uuid(self) -> uuid:
        return uuid("uuid-of-loaded")


def enumerate_devices() -> int:
    return 1
