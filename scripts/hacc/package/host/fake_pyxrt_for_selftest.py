"""A fake ``pyxrt``: EXACTLY the real binding's surface, and never more than it.

PROVENANCE, and why it is stated here. Every class, method, enum and module
attribute below is mirrored from

    https://github.com/Xilinx/XRT/blob/2024.2/src/python/pybind11/src/pyxrt.cpp

(branch ``2024.2``; XRT 2.18.x is what HACC ships). The previous fake
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

POWER, ALSO STRUCTURE. ``device.get_info`` answers the electrical and thermal
keys with trees in the shape XRT 2024.2's ``core/common/sensor.cpp`` builds
(every value a string, ``power_rails`` and ``thermals`` arrays). Armed with a
``plant``, the board power is a SYNTHETIC function of the protocol state the
power protocol last wrote to its markers file (shell only, kernel idle, loop)
plus seeded Gaussian noise, held for ``refresh_s`` and quantized to
``quantum_w``: a known differential the estimator must recover, never a card's
number. The transfer control's state adds ``transfer_w``, a known share of the
loop's ``dynamic_w``; while the protocol state is ``shell_only`` (after the
session's reset) the device reads back the all-zero xclbin UUID a reset card
should.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import time
from typing import Any, Dict, List, Optional

#: The names in this module that the real pyxrt does not have: the harness's
#: two arming hooks and the capture-header constants this fake must mirror.
#: They are declared as data so the surface-conformance test can subtract
#: exactly them and nothing else.
FAKE_ONLY = (
    "FAKE_ONLY", "arm", "call_log",
    "CAPTURE_HEADER_WORDS", "CAPTURE_RECORD_WORDS", "CAPTURE_NO_VERDICT",
    "LINK_BYTES_PER_SECOND", "ARG_PROGRAM", "OP_ARITY", "CORE_ADDRESSED_OPS",
)

#: The host link this fake pretends to be, so the programming-bandwidth metric
#: a selftest reports is a PLAUSIBLE number rather than the speed of a memcpy.
#: 8 GB/s is the order a PCIe Gen3 x16 XDMA shell delivers; the host-to-device
#: sync sleeps for the payload's share of it, which is microseconds for
#: everything shipped and is the only place this fake spends wall time.
LINK_BYTES_PER_SECOND = 8.0e9

#: The kernel argument the programming stream arrives on (odin_board_driver.py):
#: the only buffer the ``short_buffer`` fault shortens.
ARG_PROGRAM = 0

#: The capture header layout and the no-verdict sentinel, mirrored from
#: ``odin_board_driver.py`` (whose own source is ``odin_fpga_kernel.v``).
CAPTURE_HEADER_WORDS = 2
CAPTURE_RECORD_WORDS = 4
CAPTURE_NO_VERDICT = 0xFFFFFFFF

#: The token grammar a multi-core fabric reads, mirrored from
#: ``odin_rtl/stimulus.py`` (each opcode's argument count; the ops whose first
#: argument names the core they address). This fake walks it on its own --
#: a second, independent reading of the stream -- so a host that retargets a
#: program wrongly is answered with NO VERDICT, not with the placed counts.
OP_ARITY = {0: 0, 1: 3, 2: 3, 3: 2, 4: 1, 5: 1, 6: 3, 7: 4}
CORE_ADDRESSED_OPS = frozenset({1, 2, 3, 6, 7})

_UNSET = object()

#: What this fake is currently pretending to be. Module-level state is the
#: honest model: one process talks to one board.
_STATE: Dict[str, Any] = {
    "fixture": None, "fault": None, "capture_words": None, "log": [],
    "deployment": None, "plant": None, "plant_rng": None, "plant_held": None,
    "compute_units": None, "placement": None,
}

#: The card a fake device answers as: the first U55C the HACC board VM listed
#: (field, 2026-09-01, ``xbutil examine``).
_FAKE_BDF = "0000:81:00.1"

#: A programmed core given an END-only stimulus spikes nothing; the fake answers
#: the resident arm's programming run with an empty capture and this many
#: device cycles per declared program word (structure, not silicon).
_PROGRAMMING_CYCLES_PER_WORD = 40


def arm(
    fixture: Any = None, fault: Any = _UNSET, capture_words: Any = _UNSET,
    deployment: Any = _UNSET, plant: Any = _UNSET, compute_units: Any = _UNSET,
    placement: Any = _UNSET,
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
        if placement is _UNSET:
            _STATE["placement"] = None
    if placement is not _UNSET:
        # {logical: fabric}: where the fixture's cores run on a multi-core
        # fabric. None (or the identity) is where every archived run ran.
        _STATE["placement"] = (
            None if placement is None
            else {int(k): int(v) for k, v in dict(placement).items()})
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
    if compute_units is not _UNSET:
        # How many units the fake xclbin was linked with (route D); None: any
        # unit name a session asks for exists.
        _STATE["compute_units"] = None if compute_units is None else int(compute_units)
    if plant is not _UNSET:
        # ``plant`` = {"markers": <path>, "shell_w", "static_w", "dynamic_w",
        # "transfer_w", "sigma_w", "quantum_w", "refresh_s", "seed"}; None
        # disarms it.
        _STATE["plant"] = None if plant is None else dict(plant)
        _STATE["plant_rng"] = (
            None if plant is None else random.Random(int(plant.get("seed", 0))))
        _STATE["plant_held"] = None


def call_log() -> List[tuple]:
    """Every call the fake received, in order — what a contract test asserts on."""
    return list(_STATE["log"])


def _log(*entry: Any) -> None:
    _STATE["log"].append(tuple(entry))


def _deployment_image(stimulus: bytes, capacity: int,
                      program_words: int = 0) -> Optional[List[int]]:
    """The answer this pass's stimulus earns, or None when nothing matches it."""
    replay = _STATE["deployment"]
    if replay and stimulus == struct.pack("<I", 0) and int(program_words) > 1:
        # The resident arm's programming run: the whole program, then END.
        return [0, _PROGRAMMING_CYCLES_PER_WORD * int(program_words)]
    walked = _walk(stimulus)
    if walked is None or len(walked[1]) > 1:
        # Not a stream the encoder writes, or one pass addressing several cores.
        return None
    canonical, cores = walked
    fabric = next(iter(cores), 0)
    run = replay.get(hashlib.sha256(canonical).hexdigest()) if replay else None
    if run is None:
        # No frozen answer for these bytes: the fabric never saw this program,
        # so the honest fake leaves the host's sentinel in the header.
        return None
    # The frozen pass ran on its own core 0; the placed one answers from the
    # fabric core its stimulus addressed.
    events = [[int(r[0]), int(r[1]), fabric, int(r[3])] for r in run["events"]]
    if _STATE["fault"] == "extra_spike" and events:
        events.append(list(events[0]))
    if _STATE["fault"] == "truncated_capture":
        return [int(capacity), int(run["device_cycles"])] + [
            int(word) for record in events for word in record]
    words = [len(events), int(run["device_cycles"])]
    for record in events:
        words.extend(int(word) for word in record)
    return words


def _words(payload: bytes) -> List[int]:
    return list(struct.unpack(f"<{len(payload) // 4}I", payload[:len(payload) // 4 * 4]))


def _walk(payload: bytes, placement: Optional[Dict[int, int]] = None):
    """``(bytes with every core argument moved, the cores it addressed)``.

    Without a placement every core argument becomes 0 (the canonical one-core
    form a deployment's frozen replay is keyed by). None when the payload is not
    an END-terminated stream of known opcodes, or names a core the placement
    does not place.
    """
    words = _words(payload)
    moved = list(words)
    cores = set()
    index = 0
    while index < len(words):
        code = words[index]
        if code == 0:
            if index != len(words) - 1:
                return None
            return struct.pack(f"<{len(moved)}I", *moved), cores
        arity = OP_ARITY.get(code)
        if arity is None or index + arity >= len(words):
            return None
        if code in CORE_ADDRESSED_OPS:
            core = words[index + 1]
            cores.add(core)
            if placement is None:
                moved[index + 1] = 0
            elif core in placement:
                moved[index + 1] = placement[core]
            else:
                return None
        index += 1 + arity
    return None


def _payload(entry: Dict[str, Any]) -> bytes:
    import base64
    import zlib
    return zlib.decompress(base64.b64decode(entry["data"]))


def _placed(fixture: Dict[str, Any], program: bytes, stimulus: bytes
            ) -> Optional[Dict[int, int]]:
    """The armed placement, when the fabric received exactly ITS streams.

    The identity placement is not checked (it moves no byte, and archived unit
    fixtures carry payloads that are not token streams); any other is: the
    program and the stimulus must equal the fixture's own, retargeted by this
    fake's own walk, or the fabric ran something nobody froze.
    """
    placement = _STATE["placement"]
    cores = int(fixture["run"]["cores"])
    if placement is None or placement == {k: k for k in range(cores)}:
        return {k: k for k in range(cores)}
    if _STATE["fault"] == "ignore_placement":
        # Play a fabric that expects the UNPLACED streams.
        expect = {k: k for k in range(cores)}
    else:
        expect = placement
    for entry, got in ((fixture["program"], program), (fixture["stimulus"], stimulus)):
        walked = _walk(_payload(entry), expect)
        if walked is None or walked[0] != got:
            return None
    return placement


def _capture_image(capacity: int, stimulus: bytes = b"",
                   program_words: int = 0,
                   program: bytes = b"") -> Optional[List[int]]:
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
        return _deployment_image(stimulus, capacity, program_words)
    armed = _STATE["capture_words"]
    if armed is not None:
        return [int(word) for word in armed]
    fixture = _STATE["fixture"]
    if fixture is None:
        # The null-program answer B0 asks for: a verdict, and no events.
        return [0, 1]
    run = fixture["run"]
    placement = _placed(fixture, program, stimulus)
    if placement is None:
        return None
    first_tag = int(run["first_tag"])
    per_sample = int(run["cycles_per_sample"])
    words: List[int] = []
    events = 0
    for sample, cycle, core, neuron, count in fixture["expected"]["per_cycle"]:
        tag = first_tag + int(sample) * per_sample + int(cycle)
        for _ in range(int(count)):
            words.extend((tag, int(cycle), placement[int(core)], int(neuron)))
            events += 1
    if fault == "stray_core" and words:
        # A record from a fabric core the run never programmed.
        stray = max(placement.values()) + 1
        words.extend((words[0], words[1], stray, words[3]))
        events += 1
    if fault == "extra_spike" and words:
        # A device that emitted ONE more spike than the frozen evidence says:
        # exactly the divergence a certificate exists to catch.
        words.extend(words[:CAPTURE_RECORD_WORDS])
        events += 1
    if fault == "truncated_capture":
        events = int(capacity)
    return [events, int(fixture["expected"]["device_cycles_cosim"])] + words


def _plant_state(plant: Dict[str, Any]) -> str:
    """The protocol state the markers file last declared ('kernel_idle' if none)."""
    path = plant.get("markers")
    state = str(plant.get("initial_state", "kernel_idle"))
    if not path:
        return state
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
    except OSError:
        return state
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("state"):
            return str(record["state"])
    return state


def _plant_watts() -> Optional[float]:
    plant = _STATE["plant"]
    if plant is None:
        return None
    now = time.monotonic()
    held = _STATE["plant_held"]
    refresh = float(plant.get("refresh_s", 0.0))
    if held is not None and refresh > 0.0 and now - held[0] < refresh:
        return held[1]
    state = _plant_state(plant)
    watts = float(plant.get("shell_w", 20.0))
    if state in ("kernel_idle", "loop", "transfer"):
        watts += float(plant.get("static_w", 0.0))
    if state == "loop":
        watts += float(plant.get("dynamic_w", 0.0))
    if state == "transfer":
        watts += float(plant.get("transfer_w", 0.0))
    watts += _STATE["plant_rng"].gauss(0.0, float(plant.get("sigma_w", 0.0)))
    quantum = float(plant.get("quantum_w", 0.0))
    if quantum > 0.0:
        watts = round(watts / quantum) * quantum
    _STATE["plant_held"] = (now, watts)
    return watts


def _electrical_tree() -> Dict[str, Any]:
    """The legacy-path tree of sensor.cpp, every value a string."""
    plant = _STATE["plant"]
    if plant is not None and plant.get("absent"):
        # sensor.cpp's answer for a card that exposes no sensors at all.
        return {"msg": "No sensors present"}
    watts = _plant_watts()
    power = "21.987654" if watts is None else f"{watts:.6f}"
    def rail(rid: str, desc: str, volts: str, amps: str) -> Dict[str, Any]:
        return {"id": rid, "description": desc,
                "voltage": {"volts": volts, "is_present": "true"},
                "current": {"amps": amps, "is_present": "true" if amps else "false"}}
    return {
        "power_consumption_max_watts": "150",
        "power_consumption_watts": power,
        "power_consumption_warning": "false",
        "power_rails": [
            rail("12v_pex", "12 Volts PCI Express", "12.143", "1.420"),
            rail("3v3_pex", "3.3 Volts PCI Express", "3.311", "0.912"),
            rail("12v_aux", "12 Volts Auxillary", "12.150", ""),
            rail("vccint", "Internal FPGA Vcc", "0.851", "11.094"),
        ],
    }


def _thermal_tree() -> Dict[str, Any]:
    plant = _STATE["plant"]
    hot = 2 if plant is not None and _plant_state(plant) == "loop" else 0
    def temp(location: str, desc: str, celsius: int) -> Dict[str, Any]:
        return {"location_id": location, "description": desc,
                "temp_C": str(celsius), "is_present": "true"}
    return {"thermals": [
        temp("pcb_top_front", "PCB Top Front", 38),
        temp("fpga0", "FPGA", 44 + hot),
        temp("fpga_hbm", "FPGA HBM", 41 + hot),
        {"location_id": "cage_temp_0", "description": "Cage0", "temp_C": "0",
         "is_present": "false"},
    ]}


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
        # "kernel:{cu}" names one compute unit (XRT's own syntax); a unit the
        # linked xclbin does not hold is refused as XRT refuses it.
        self._cu_index: Optional[int] = None
        if ":{" in self._name and self._name.endswith("}"):
            cu = self._name.split(":{", 1)[1][:-1]
            index = cu.rsplit("_", 1)[-1]
            self._cu_index = int(index) if index.isdigit() else 0
            units = _STATE["compute_units"]
            if units is not None and self._cu_index >= int(units):
                raise RuntimeError(f"No compute units matching '{self._name}'")

    def group_id(self, argno: int) -> int:
        _log("group_id", int(argno))
        # One unit's arguments all sit on its own bank (HBM[k] for unit k).
        return int(argno) if self._cu_index is None else int(self._cu_index)

    def __call__(self, *args: Any) -> run:
        _log("start", tuple(
            item._group if isinstance(item, bo) else item for item in args))
        capture = args[2]
        capacity = int(args[5])
        # Exactly the words the host DECLARED, not the whole buffer object: a
        # deployment reuses one oversized stimulus buffer across samples.
        stimulus = bytes(args[1]._data[:int(args[4]) * 4])
        program = bytes(args[0]._data[:int(args[3]) * 4])
        words = _capture_image(capacity, stimulus, int(args[3]), program)
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
        if _STATE["fault"] == "short_buffer" and self._group == ARG_PROGRAM:
            # An allocator that gave back LESS than was asked for: the host
            # would then declare more words than the buffer holds, and the
            # fabric — which stores no copy of the stream — would read past it.
            self._size = max(0, self._size - 4)
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
        moved = self._size if size is None else int(size)
        _log("sync", self._group, direction, moved, int(offset))
        if direction == xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE:
            # A DMA takes link time. Without it the driver's programming-
            # bandwidth metric would report the speed of a memcpy, which is not
            # a plausible stand-in for a number the owner reads off a board.
            time.sleep(moved / LINK_BYTES_PER_SECOND)

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


class xrt_info_device:
    """The ``xrt::info::device`` keys the power path reads (a subset)."""

    bdf = "bdf"
    name = "name"
    electrical = "electrical"
    thermal = "thermal"
    mechanical = "mechanical"
    platform = "platform"
    dynamic_regions = "dynamic_regions"


class device:
    def __init__(self, index: int = 0) -> None:
        _log("device", int(index))
        self._index = int(index)

    def load_xclbin(self, image: Any) -> uuid:
        _log("load_xclbin", image._path)
        return image.get_uuid()

    def get_xclbin_uuid(self) -> uuid:
        plant = _STATE["plant"]
        if plant is not None and _plant_state(plant) == "shell_only":
            # What XRT reads back from a card whose user region holds nothing.
            return uuid("00000000-0000-0000-0000-000000000000")
        return uuid("uuid-of-loaded")

    def get_info(self, key: Any) -> str:
        """A string, as pyxrt returns it: the BDF, or a report as JSON text."""
        _log("get_info", str(key))
        if key == xrt_info_device.bdf:
            return _FAKE_BDF
        if key == xrt_info_device.name:
            return "xilinx_u55c_gen3x16_xdma_base_3"
        if key == xrt_info_device.electrical:
            return json.dumps(_electrical_tree())
        if key == xrt_info_device.thermal:
            return json.dumps(_thermal_tree())
        return json.dumps({"msg": "not modelled by the fake"})


def enumerate_devices() -> int:
    return 1
